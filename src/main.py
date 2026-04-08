import warnings
warnings.filterwarnings("ignore",message=r".*pynvml package is deprecated.*",
    category=FutureWarning,)

import sys
import os

# add project root (/mnt/d/FedMKD) into sys.path so `import easyfl` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# also keep src itself in path for `import reid`
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


import time
from types import SimpleNamespace

import numpy as np
import random
import json
import pandas as pd
import traceback
from collections import OrderedDict

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import cpu_count
import copy
import time
import scipy.sparse as sp
from torchvision import models

from reid.datasets.make_dataloader import make_dataloader, build_client_train_loader, build_train_transforms
from reid.reid_train import local_train_reid
import torch.nn.functional as F

from reid.make_loss import make_loss
from reid.make_optimizer import make_optimizer
from reid.make_scheduler import create_scheduler
from reid.public_ssl_dataset import PublicReIDSSL
from reid.reid_split import split_train_by_pid
from reid.reid_test import test_reid
from reid.reid_wrapper import ReIDWrapper



import torch
from torch.autograd import Variable
from torch.backends import cudnn
import parse
args = parse.args
# os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(args.gpu)

# 自动混合精度
from torch.cuda import amp

torch.backends.cudnn.enabled = False
torch.multiprocessing.set_sharing_strategy('file_system')

import easyfl
import utils

from model import get_model, BYOL
from dataset import get_semi_supervised_dataset
from easyfl.datasets.data import CIFAR100
from easyfl.coordinator import Coordinator
from client import FedSSLClient
from server import MyDistillServer
# from Fedmd_server import FedmdServer
# from fedema_server import FedSSLServer
# from distillServer import distillServer
import logging
import torch.distributed as dist
from pathlib import Path

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


model_dir = args.save_model_path
CIFAR100 = "cifar100"
logger = logging.getLogger(__name__)


def setup_run_logger(log_dir: Path, level=logging.INFO):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Prevent duplicate handlers when script is relaunched in the same process.
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return log_file


def ignore_resize_warning(message, category, filename, lineno, file=None, line=None):
    if "An output with one or more elements was resized" in str(message):
        return True
    return False


def make_reid_cfg_for_client(args, sampler, max_epochs):
    return SimpleNamespace(
        DATALOADER=SimpleNamespace(SAMPLER=sampler),
        MODEL=SimpleNamespace(
            METRIC_LOSS_TYPE="triplet",
            NO_MARGIN=False,
            IF_LABELSMOOTH="on",
            ID_LOSS_WEIGHT=1.0,
            TRIPLET_LOSS_WEIGHT=1.0,
        ),
        SOLVER=SimpleNamespace(
            # optimizer
            BASE_LR=getattr(args, "reid_lr", 3e-4),
            WEIGHT_DECAY=getattr(args, "reid_weight_decay", 5e-4),
            BIAS_LR_FACTOR=getattr(args, "reid_bias_lr_factor", 2.0),
            WEIGHT_DECAY_BIAS=getattr(args, "reid_weight_decay_bias", 0.0),
            LARGE_FC_LR=getattr(args, "reid_large_fc_lr", True),
            OPTIMIZER_NAME=getattr(args, "reid_optimizer", "AdamW"),
            MOMENTUM=getattr(args, "reid_momentum", 0.9),
            CENTER_LR=getattr(args, "reid_center_lr", 0.5),

            # loss margin
            MARGIN=getattr(args, "reid_margin", 0.3),

            # scheduler
            MAX_EPOCHS=max_epochs,
            WARMUP_EPOCHS=getattr(args, "reid_warmup_epochs", 5),
        ),
    )

def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(x) for x in d]
    return d

# 将警告过滤器应用到特定的警告消息
warnings.showwarning = ignore_resize_warning

def client_main(client_list, data_list, device, epoch, clients, train_data, test_data, config):
    # return client model parameters
    new_model_weights = []
    for idx in range(len(client_list)):
        client = client_list[idx]
        client_name = data_list[idx]
        client_i = FedSSLClient(client_name, config, train_data, test_data, device, epoch)
        if config.framework == 'fedema':
            client_i.compressed_model = clients[idx]
            tmp_model_path = os.path.join(model_dir, 'saved_model/', task_id, '/local_model', client_name, '.pth')
            if os.path.exists(tmp_model_path):
                client_i.local_model = clients[idx]
                client_i.local_model.load_state_dict(torch.load(tmp_model_path))
            client_i.update_model()
        else:
            client_i.model = clients[idx]  # 直接重置了模型参数

        # Train
        client_i.train(config.client, device=device)
        # save model
        if (epoch + 1) % args.test_every == 0:
            client_i.save_model()
        tmp_model = client_i.model.cpu()
        new_model_weights.append(tmp_model)
    return new_model_weights


def server_train(client_models, server_model, public_dataset, test_data, device, epoch, public_loader=None, distill_devices=None):
    # multi-teacher distillation
    # input: clients model, server model, device, public dataset
    # output: server model
    if args.framework in ['ours','oursnoalign']:
        server = MyDistillServer(None, config, public_dataset, test_data, device, epoch)
    else:
        raise ValueError(f"framework type is wrong")
    server.model = server_model
    server.client_models = client_models
    server.teacher_devices = distill_devices
    start_time = time.time()

    if public_loader is not None:
        server.train_loader = public_loader
    # 调用训练过程
    # Train
    server.train(config.server, device=device)
    server.save_model()
    # # Test
    # if (epoch + 1) % args.test_every == 0:
    #     server.test()
    return server.model.cpu()


def fedAvg_agg(client_models, avg_weights, public_dataset, device, epoch):
    # server = MyDistillServer(None, config, public_dataset, test_data, device, epoch)
    # server = FedSSLServer(config, public_dataset, test_data)
    # start_time = time.time()
    # # fedAvg_weight = FedAvg(client_models, avg_weights)
    # # server.model = fedAvg_weight
    # server.client_model = client_models
    # server.weight = avg_weights
    # server.model = client_models[0]
    # server.aggregation()
    # server.save_model()
    # # Test
    # if (epoch + 1) % args.test_every == 0:
    #     server.test(device)
    # logger.info(f"---------Time cost of fedavg is {time.time() - start_time}s---------")
    # return server.model
    pass


def client_align(ori_client, server_model, public_dataset, device, config,public_loader=None,client_loaders=None):
    # update clients model
    # input: clients model, server model, device, public dataset
    # output: clients model
    new_model_weights = []
    start_time = time.time()
    for idx in range(len(ori_client)):
        client_config = config
        client_config.client.momentum_update = False
        client_config.client.local_epoch = 1
        s_client = FedSSLClient(None, client_config, public_dataset, None, device, None)
        s_client.model = ori_client[idx]
        s_client.model.target_encoder = server_model.online_encoder
        if client_loaders is not None:
            s_client.train_loader = client_loaders[idx]
        elif public_loader is not None:
            s_client.train_loader = public_loader
        # Train
        s_client.align_train(config.client, device=device)
        tmp = {
            "bnneck": copy.deepcopy(s_client.model.bnneck.state_dict()),
            "classifier": copy.deepcopy(s_client.model.classifier.state_dict()),
        }
        new_model_weights.append(tmp)
    logger.info(f"--------Time of align is {time.time() - start_time}s--------")
    return new_model_weights


def _reid_train_chunk(client_ids, clients_chunk, client_loaders_chunk, client_num_classes_chunk, device, args):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.set_device(device.index)

    out = []
    for j, cid in enumerate(client_ids):
        print(f"------------client[{cid}] training-----------")
        model_i = clients_chunk[j].to(device)
        model_i.train()

        cfg_i = make_reid_cfg_for_client(args, sampler=args.reid_sampler, max_epochs=args.local_epoch)
        loss_fn_i, center_criterion = make_loss(cfg_i, num_classes=client_num_classes_chunk[j])
        optimizer, optimizer_center = make_optimizer(cfg_i, model_i, center_criterion)
        scheduler = create_scheduler(cfg_i, optimizer)

        model_i = local_train_reid(
            cfg_i, model_i, client_loaders_chunk[j],
            optimizer, optimizer_center, scheduler, loss_fn_i, center_criterion,
            device=device, local_epochs=args.local_epoch, use_amp=True
        )
        out.append(model_i.cpu())
    return out


def _reid_align_chunk(usr_model_weights_chunk, server_model, device, config, client_loaders_chunk):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.set_device(device.index)
    return client_align(
        usr_model_weights_chunk,
        server_model,
        public_dataset=None,
        device=device,
        config=config,
        client_loaders=client_loaders_chunk,
    )


def _state_dict_cpu(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _reid_train_worker_entry(slot_id, client_ids, clients_chunk, client_loaders_chunk,
                             client_num_classes_chunk, device, args, result_dict):
    try:
        trained_models = _reid_train_chunk(
            client_ids,
            clients_chunk,
            client_loaders_chunk,
            client_num_classes_chunk,
            device,
            args,
        )
        result_dict[slot_id] = {
            "ok": True,
            "client_ids": list(client_ids),
            "states": [_state_dict_cpu(m.state_dict()) for m in trained_models],
        }
    except Exception:
        result_dict[slot_id] = {
            "ok": False,
            "error": traceback.format_exc(),
        }


def _reid_align_worker_entry(slot_id, usr_model_weights_chunk, server_model, device,
                             config, client_loaders_chunk, result_dict):
    try:
        aligned_weights = _reid_align_chunk(
            usr_model_weights_chunk,
            server_model,
            device,
            config,
            client_loaders_chunk,
        )
        cpu_weights = []
        for item in aligned_weights:
            cpu_weights.append({
                "bnneck": _state_dict_cpu(item["bnneck"]),
                "classifier": _state_dict_cpu(item["classifier"]),
            })
        result_dict[slot_id] = {
            "ok": True,
            "weights": cpu_weights,
        }
    except Exception:
        result_dict[slot_id] = {
            "ok": False,
            "error": traceback.format_exc(),
        }


# 断点续训
def _checkpoint_path(save_root: str, task_id: str) -> str:
    # save_root 通常是 args.save_model_path；为空时用 cwd/saved_models/task_id
    if save_root == "":
        save_root = os.path.join(os.getcwd(), "saved_models", task_id)
    else:
        save_root = os.path.join(save_root, "saved_models", task_id) if not save_root.endswith("saved_models") else os.path.join(save_root, task_id)
    os.makedirs(save_root, exist_ok=True)
    return os.path.join(save_root, "checkpoint.pth")


def save_checkpoint(path: str, round_id: int, server_model, clients):
    ckpt = {
        "round_id": int(round_id),
        "server_model": server_model.cpu().state_dict(),
        "clients": [c.cpu().state_dict() for c in clients],
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, server_model, clients, device):
    ckpt = torch.load(path, map_location="cpu",weights_only=False)

    server_model.load_state_dict(ckpt["server_model"], strict=False)

    # 恢复每个 client 的权重（假设 clients 列表长度不变）
    client_states = ckpt.get("clients", [])
    if client_states and len(client_states) == len(clients):
        for c, sd in zip(clients, client_states):
            c.load_state_dict(sd, strict=False)

    # 恢复随机数状态（让续训更接近真正断点）
    rng = ckpt.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])

    # round_id 表示“已经完成到哪一轮”
    round_id = int(ckpt.get("round_id", 0))

    server_model.to(device)
    for c in clients:
        c.to(device)

    return round_id

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
     # args.gpu 兼容: "0,1,2" / [0,1,2] / ["0","1"]
    if isinstance(args.gpu, str):
        gpu_index = [int(x.strip()) for x in args.gpu.split(",") if x.strip() != ""]
    elif isinstance(args.gpu, (list, tuple)):
        gpu_index = [int(x) for x in args.gpu]
    else:
        gpu_index = [int(args.gpu)]

    devices = [torch.device("cuda", i) for i in gpu_index]
    # 一般建议每张卡一个进程
    num_processes = min(len(devices), args.num_of_clients)

    client_list = list(range(args.num_of_clients))
    step = int(np.ceil(len(client_list) / num_processes))

    class_per_client = args.class_per_client

    if args.dataset == CIFAR100:
        class_per_client *= 10

    task_id = args.task_id
    if task_id == "":
        task_id = f"{args.dataset}_{args.framework}_{args.client_type}_{args.server_model}_{args.num_of_clients}_" \
                  f"{args.encoder_network}_{args.data_partition}_{args.dir_alpha}_" \
                  f"{args.local_epoch}_{args.server_epoch}_{args.rounds}_{args.batch_size}_{args.public}_{args.data_size}_{args.public_size}"

    # print(task_id)
    momentum_update = True
    image_size = 32

    SRC_DIR = Path(__file__).resolve().parent
    TRACK_DIR = SRC_DIR / "logs" / task_id
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_DB = TRACK_DIR / "tracking.sqlite"
    run_log_file = setup_run_logger(TRACK_DIR)
    logger.info(f"Run log file: {run_log_file}")

    config = {
        "task_id": task_id,
        "seed": args.seed,
        "framework": args.framework,
        "client_type": args.client_type,
        "data_number": args.data_size,
        "track": False,
        # "tracker_addr": "",  # 本地存储不需要 remote address，就给空字符串即可
        # "tracking": {
        #     "database": str(TRACK_DB),  # 这里 MyDistillServer 会当成 path 传给 init_tracking
        # },
        "data": {
            "dataset": args.dataset,
            "num_of_clients": args.num_of_clients,
            "split_type": args.data_partition,
            "class_per_client": class_per_client,
            "data_amount": 1,
            "iid_fraction": 1,
            "min_size": 10,
            "alpha": args.dir_alpha,
            "public": args.public,
        },
        "client_model": args.client_model,
        "server_model": args.server_model,
        "test_mode": "test_in_server",
        "server": {
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "test_every": args.test_every,
            "save_model_every": args.save_model_every,
            "save_model_path": args.save_model_path,
            "clients_per_round": args.clients_per_round,
            "random_selection": args.random_selection,
            "save_predictor": args.save_predictor,
            "test_all": True,
            "model": args.server_model,
            "track":False,
            "optimizer": {
                "type": args.optimizer_type,
                "lr_type": args.lr_type,
                "lr": args.lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "drop_last": True,
            "server_epoch": args.server_epoch,
            "gaussian": False,
            "image_size": image_size,
            "aggregate_encoder": args.aggregate_encoder,
            "update_encoder": args.update_encoder,
            "update_predictor": args.update_predictor,
            "random_selection": args.random_selection,

            "encoder_weight": args.encoder_weight,
            "predictor_weight": args.predictor_weight,

            "momentum_update": momentum_update,
            "data_number": args.data_size,
        },
        "client": {
            "drop_last": True,
            "batch_size": args.batch_size,
            "local_epoch": args.local_epoch,
            "data_number": args.data_size,
            "optimizer": {
                "type": args.optimizer_type,
                "lr_type": args.lr_type,
                "lr": args.lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            # application specific
            "model": args.client_model,
            "rounds": args.rounds,
            "round_id": 0,
            "gaussian": False,
            "image_size": image_size,
            "save_model_path": args.save_model_path,
            "save_predictor": args.save_predictor,  #后加

            "aggregate_encoder": args.aggregate_encoder,
            "update_encoder": args.update_encoder,
            "update_predictor": args.update_predictor,
            "random_selection": args.random_selection,
            "dapu_threshold": args.dapu_threshold,
            "weight_scaler": args.weight_scaler,
            "auto_scaler": args.auto_scaler,
            "auto_scaler_target": args.auto_scaler_target,

            "encoder_weight": args.encoder_weight,
            "predictor_weight": args.predictor_weight,
            "momentum_update": momentum_update,
        }
    }
    config = dict_to_ns(config)

    # ---- ReID patch ----
    REID_DATASETS = {"market1501", "dukemtmc", "msmt17"} 
    if args.dataset.lower() in REID_DATASETS:
        # 用 ReID 的输入尺寸
        config.server.image_size = [args.reid_height, args.reid_width]
        config.client.image_size = [args.reid_height, args.reid_width]

        # 如果你的 get_transformation 需要 square size，你就自己决定用 height 或 width
        # 这里保持 (H, W) 一致最安全

        # public loader batch size：建议和 args.batch_size 一致即可
        config.server.batch_size = args.batch_size
        config.client.batch_size = args.batch_size

        # public augmentation gaussian：ReID 先关掉或按需要开
        config.server.gaussian = False
        config.client.gaussian = False

    is_reid = args.dataset in ["market1501", "dukemtmcreid", "msmt17"]

    if is_reid and num_processes > 1:
        print(
            f"[ReID] Using spawn Process backend (non-daemon), keep reid_num_workers={args.reid_num_workers}."
        )

    if is_reid:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, cam_num, view_num,dataset = make_dataloader(args)

        public_transform = utils.get_transformation(args.client_model)(
            (args.reid_height, args.reid_width),
            getattr(config.server, "gaussian", False),
        )
        public_ds = PublicReIDSSL(dataset.train, public_transform)
        public_loader = DataLoader(
            public_ds,
            batch_size=config.server.batch_size,
            shuffle=True,
            num_workers=args.reid_num_workers,
            drop_last=True,
            pin_memory=True,
        )
        train_transforms = build_train_transforms(args)

        client_train_lists = split_train_by_pid(dataset.train, args.num_of_clients)

        client_loaders = []
        client_num_classes = []
        for i in range(args.num_of_clients):
            loader_i, ncls_i, _pid_map = build_client_train_loader(args, client_train_lists[i], train_transforms)
            client_loaders.append(loader_i)
            client_num_classes.append(ncls_i)

        if args.framework in ['ours', 'single', 'oursnoalign']:
            server_model = get_model(args.server_model, args.encoder_network, args.predictor_network)
        else:
            raise ValueError(f"Unsupported framework in ReID branch: {args.framework}")

        clients = []
        for i in range(args.num_of_clients):
            base = get_model(args.client_model, args.encoder_network, args.predictor_network)
            cli = ReIDWrapper(base, num_classes=client_num_classes[i], feat_dim=2048)
            clients.append(cli)


        print('---------Start ReID training---------')

        ckpt_path = _checkpoint_path(args.save_model_path, task_id)
        start_round = 0

        if os.path.exists(ckpt_path):
            print(f"[CKPT] Found checkpoint: {ckpt_path}, resuming...")
            # 这里 device 用你实际训练的 devices[0]
            last_done_round = load_checkpoint(ckpt_path, server_model, clients, devices[0])
            start_round = last_done_round  # last_done_round 表示已完成轮数
        else:
            print(f"[CKPT] No checkpoint found, training from scratch.")

        # ReID 这边的 “data_user / weight / avg_weights” 如果你不做 FedAvg 可以不需要
        client_list = list(range(args.num_of_clients))
        data_user = client_list

        for epoch in range(start_round,args.rounds):
            config.client.round_id += 1
            time_start = time.time()

            # ---- 8.1 client 本地监督训练 ----
            ctx = torch.multiprocessing.get_context("spawn")
            manager = ctx.Manager()
            train_result_dict = manager.dict()
            train_procs = []
            active_slots = []

            for pidx in range(num_processes):
                l = pidx * step
                r = min((pidx + 1) * step, len(client_list))
                if l >= r:
                    continue
                slot_id = pidx
                proc = ctx.Process(
                    target=_reid_train_worker_entry,
                    args=(
                        slot_id,
                        client_list[l:r],
                        clients[l:r],
                        client_loaders[l:r],
                        client_num_classes[l:r],
                        devices[pidx % len(devices)],
                        args,
                        train_result_dict,
                    ),
                )
                proc.start()
                train_procs.append((slot_id, proc))
                active_slots.append(slot_id)

            for _, proc in train_procs:
                proc.join()

            errors = []
            for slot_id, proc in train_procs:
                if slot_id not in train_result_dict:
                    errors.append(f"train worker slot={slot_id} exited with code={proc.exitcode} and no result")
                    continue
                result = train_result_dict[slot_id]
                if not result.get("ok", False):
                    errors.append(f"train worker slot={slot_id} failed:\n{result.get('error', '')}")

            if errors:
                manager.shutdown()
                raise RuntimeError("ReID parallel local training failed:\n" + "\n".join(errors))

            usr_model_weights = [None] * len(client_list)
            for slot_id in sorted(active_slots):
                result = train_result_dict[slot_id]
                cids = result["client_ids"]
                states = result["states"]
                for cid, sd in zip(cids, states):
                    clients[cid].load_state_dict(sd, strict=False)
                    usr_model_weights[cid] = clients[cid].cpu()

            manager.shutdown()

            if any(x is None for x in usr_model_weights):
                raise RuntimeError("ReID parallel local training returned incomplete client models.")

            clients = usr_model_weights
            print('---------Finishing training clients---------')
            print(f"---------Time cost of ReID client train epoch {epoch} is {time.time() - time_start}s---------")
            logger.info(
                f"---------Time cost of ReID client train epoch {epoch} is {time.time() - time_start}s---------")

            # ---- 8.2 server 蒸馏：传 public_loader 注入 ----
            if config.framework == 'ours':
                start_time = time.time()
                server_model = server_train(
                    usr_model_weights,
                    server_model,
                    public_dataset=None,  # ReID 分支不需要 EasyFL public_data
                    test_data = val_loader,
                    device=devices[0],
                    epoch=epoch,
                    public_loader=public_loader,
                    distill_devices=devices,
                )
                print("---------Finish distill---------")
                print(f"---------Time cost of distill is {time.time() - start_time}s---------")
                logger.info(f"---------Time cost of distill is {time.time() - start_time}s---------")

                # ---- 8.3 align：建议只对齐 encoder/target_encoder（避免 classifier shape mismatch）----
                align_time = time.time()
                ctx = torch.multiprocessing.get_context("spawn")
                manager = ctx.Manager()
                align_result_dict = manager.dict()
                align_procs = []
                active_slots = []

                for pidx in range(num_processes):
                    l = pidx * step
                    r = min((pidx + 1) * step, len(usr_model_weights))
                    if l >= r:
                        continue
                    slot_id = pidx
                    proc = ctx.Process(
                        target=_reid_align_worker_entry,
                        args=(
                            slot_id,
                            usr_model_weights[l:r],
                            server_model,
                            devices[pidx % len(devices)],
                            config,
                            client_loaders[l:r],
                            align_result_dict,
                        ),
                    )
                    proc.start()
                    align_procs.append((slot_id, proc))
                    active_slots.append(slot_id)

                for _, proc in align_procs:
                    proc.join()

                errors = []
                for slot_id, proc in align_procs:
                    if slot_id not in align_result_dict:
                        errors.append(f"align worker slot={slot_id} exited with code={proc.exitcode} and no result")
                        continue
                    result = align_result_dict[slot_id]
                    if not result.get("ok", False):
                        errors.append(f"align worker slot={slot_id} failed:\n{result.get('error', '')}")

                if errors:
                    manager.shutdown()
                    raise RuntimeError("ReID parallel align failed:\n" + "\n".join(errors))

                model_weights = []
                for slot_id in sorted(active_slots):
                    model_weights.extend(align_result_dict[slot_id]["weights"])

                manager.shutdown()

                for i in range(len(clients)):
                    clients[i].bnneck.load_state_dict(model_weights[i]["bnneck"])
                    clients[i].classifier.load_state_dict(model_weights[i]["classifier"])

                print(f"---------Time cost of align is {time.time() - align_time}s---------")
                logger.info(f"---------Time cost of align is {time.time() - align_time}s---------")

                if (epoch + 1) % args.test_every == 0:
                    results = test_reid(
                        server_model.online_encoder,
                        val_loader,
                        num_query=num_query,
                        device=devices[0],
                        reranking=getattr(args, "reid_rerank", False),
                    )
                    print(f"[Global Test][Epoch {epoch + 1}] {results}")
                    logger.info(f"[Global Test][Epoch {epoch + 1}] {results}")

            elif config.framework == 'fedavg':
                raise ValueError(
                    "ReID pid-split mode does not support FedAvg on full model (classifier shapes differ).")
            else:
                raise ValueError(f"framework type is wrong: {config['framework']}")

            print("---------Time cost of epoch {:d} is {:.1f}s---------".format(epoch, time.time() - time_start))
            # 这一轮完成后保存 checkpoint，round_id 用 epoch+1 表示“完成到第几轮”
            save_checkpoint(ckpt_path, round_id=epoch + 1, server_model=server_model, clients=clients)
            print(f"[CKPT] Saved checkpoint at round {epoch + 1} -> {ckpt_path}")

        sys.exit(0) 
    else:
        # split public data first
        if args.semi_supervised:
            print('true')
            train_data, test_data, _ = get_semi_supervised_dataset(args.dataset,
                                                                   args.num_of_clients,
                                                                   args.data_partition,
                                                                   class_per_client,
                                                                   args.label_ratio)
            print(train_data.dtype())

        small_client_network = get_model(args.client_model, "resnet18", args.predictor_network)
        middel_client_network = get_model(args.client_model, "resnet34", args.predictor_network)
        vgg_client_network = get_model(args.client_model, "vgg", args.predictor_network)

        model = get_model(args.client_model, "resnet18", args.predictor_network)
        model.eval()

        if args.framework in ['ours', 'single', 'oursnoalign']:
            # define server model
            server_model = get_model(args.server_model, args.encoder_network, args.predictor_network)
            model_path = os.path.join(model_dir, 'saved_models', task_id, 'global_model.pth')
            print(model_path)
            if os.path.exists(model_path):
                print("load successfully")
                load_model = torch.load(model_path)
                new_model = OrderedDict()
                for k, v in load_model.items():
                    if k[:15] == 'online_encoder.':
                        name = k
                        new_model[name] = v
                    elif k[:15] == 'target_encoder.':
                        pass
                    else:
                        name = k
                        new_model[k] = v
                server_model.load_state_dict(new_model)

        coord = Coordinator()
        coord, config = easyfl.init(config, init_all=False)
        train_data = coord.train_data
        test_data = coord.test_data
        public_data = coord.public_data
        data_user = coord.train_data.users
        weight = []
        for user in data_user:
            weight.append(len(train_data.data[user]['y']))

        # define local model list
        if config.client_type == 'resnet18':
            # clients = [small_client_network] * args.num_of_clients
            clients = [copy.deepcopy(small_client_network) for _ in range(args.num_of_clients)]
        elif config.client_type == 'resnet34':
            clients = [middel_client_network] * args.num_of_clients
        elif config.client_type == 'vgg':
            clients = [vgg_client_network] * args.num_of_clients
        elif config.client_type == 'mix':
            # choices = [small_client_network, middel_client_network]
            # clients = np.random.choice(choices, args.num_of_clients, replace=True)
            clients = [small_client_network, small_client_network, vgg_client_network, vgg_client_network,
                       vgg_client_network]
        idx = 0
        for user in data_user:
            model_path = os.path.join(model_dir, 'saved_models', task_id, 'local_model', user, '.pth')
            if os.path.exists(model_path):
                clients[idx].load_state_dict(torch.load(model_path))
            idx += 1

        # logger.info(f"{clients}")

        avg_weights = weight / np.sum(weight)
        print('---------Start training---------')

        ckpt_path = _checkpoint_path(args.save_model_path, task_id)
        start_round = 0

        if os.path.exists(ckpt_path):
            print(f"[CKPT] Found checkpoint: {ckpt_path}, resuming...")
            # 这里 device 用你实际训练的 devices[0]
            last_done_round = load_checkpoint(ckpt_path, server_model, clients, devices[0])
            start_round = last_done_round  # last_done_round 表示已完成轮数
        else:
            print(f"[CKPT] No checkpoint found, training from scratch.")

        for epoch in range(start_round, args.rounds):
            config.client.round_id += 1

            # multi-process
            time_start = time.time()
            # ctx = torch.multiprocessing.get_context("spawn")
            # pool = ctx.Pool(processes=num_processes)
            # process_arr = []
            # for i in range(num_processes):
            #     device = devices[i]
            #     process_arr.append(
            #         pool.apply_async(client_main, args=(
            #             client_list[i * step:(i + 1) * step], data_user[i * step:(i + 1) * step], devices[i], epoch,
            #             clients[i * step:(i + 1) * step], train_data, test_data, config)))
            # pool.close()
            # pool.join()
            # print('---------Finishing training clients---------')
            #
            # # each process should get model parameters.
            # usr_model_weights_t = []
            # if num_processes > 1:
            #     usr_model_weights_t = process_arr[0].get()
            #     for process in process_arr[1:]:
            #         tmp_usr_model_weights_t = process.get()
            #         usr_model_weights_t += tmp_usr_model_weights_t
            #     usr_model_weights = usr_model_weights_t
            # else:
            #     usr_model_weights = process_arr[0].get()

            # 单进程（num_processes=1）直接跑 client_main，避免 multiprocessing + CUDA 资源句柄问题
            device = devices[0]
            usr_model_weights = client_main(
                client_list,
                data_user,
                device,
                epoch,
                clients,
                train_data,
                test_data,
                config
            )
            print('---------Finishing training clients---------')

            # print("---------Time cost of train of epoch {:d} is {:.1f}s---------".format(epoch, time.time() - time_start))
            logger.info(f"---------Time cost of train of epoch {epoch} is {time.time() - time_start}s---------")
            # print('user model weights is :', len(usr_model_weights))

            if config.framework == 'ours':
                # train server model use distill, server only need the client online encoder
                start_time = time.time()
                server_model = server_train(
                    usr_model_weights,
                    server_model,
                    public_data,
                    devices[0],
                    epoch,
                    distill_devices=devices,
                )
                print("---------Finish distill---------")
                logger.info(f"---------Time cost of distill is {time.time() - start_time}s---------")
                # align local model in server, change target network

                # # multi-process version
                align_time = time.time()
                # # ctx = torch.multiprocessing.get_context("spawn")
                # # torch.multiprocessing.set_start_method(method='forkserver', force=True)
                # pool = ctx.Pool(processes=num_processes)
                # process_arr = []
                # for i in range(num_processes):
                #     device = devices[i]
                #     torch.cuda.set_device(device)
                #     process_arr.append(
                #         pool.apply_async(client_align, args=(
                #             usr_model_weights[i * step:(i + 1) * step], server_model, public_data, device, config)))
                # pool.close()
                # pool.join()
                #
                # # each process should get model parameters.
                # model_weights_t = []
                # if num_processes > 1:
                #     model_weights_t = process_arr[0].get()
                #     for process in process_arr[1:]:
                #         tmp_model_weights_t = process.get()
                #         model_weights_t += tmp_model_weights_t
                #     model_weights = model_weights_t
                # else:
                #     model_weights = process_arr[0].get()
                #
                # for i in range(len(clients)):
                #     clients[i].target_encoder = model_weights[i]

                # 单进程 align（避免把 server_model/public_data 这些复杂对象传入子进程）
                model_weights = client_align(
                    usr_model_weights,
                    server_model,
                    public_data,
                    devices[0],
                    config
                )

                for i in range(len(clients)):
                    clients[i].target_encoder = model_weights[i]

                logger.info(f"---------Time cost of align is {time.time() - align_time}s---------")

            elif config.framework == 'fedavg':
                fedAvg_weight = fedAvg_agg(usr_model_weights, avg_weights, public_data, devices[0], epoch)
                for idx in range(len(clients)):
                    clients[idx] = fedAvg_weight

            else:
                raise ValueError(f"framework type is wrong")

            print(
                "---------Time cost of train of epoch {:d} is {:.1f}s---------".format(epoch, time.time() - time_start))
            # 这一轮完成后保存 checkpoint，round_id 用 epoch+1 表示“完成到第几轮”
            save_checkpoint(ckpt_path, round_id=epoch + 1, server_model=server_model, clients=clients)
            print(f"[CKPT] Saved checkpoint at round {epoch + 1} -> {ckpt_path}")



