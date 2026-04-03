import copy
import gc
import logging
import time
from collections import Counter

import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast, GradScaler
import os

import model
import utils
from communication import ONLINE, TARGET, BOTH, LOCAL, GLOBAL, DAPU, NONE, EMA, DYNAMIC_DAPU, DYNAMIC_EMA_ONLINE, \
    SELECTIVE_EMA
from easyfl.client.base import BaseClient
from easyfl.tracking import metric
from easyfl.tracking.client import init_tracking
from knn_monitor import knn_monitor
from easyfl.utils.float import rounding
from easyfl.datasets.data import CIFAR100
from torchvision import datasets

torch.multiprocessing.set_sharing_strategy('file_system')

import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm

logger = logging.getLogger(__name__)
import parse

args = parse.args

L2 = "l2"
ACCURACY = "accuracy"
LOSS = "loss"
CLIENT_METRICS = "client_metrics"
TARGET = "target"


def ignore_resize_warning(message, category, filename, lineno, file=None, line=None):
    if "An output with one or more elements was resized" in str(message):
        return True
    return False


warnings.showwarning = ignore_resize_warning


class MyDistillServer(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, round):
        super(MyDistillServer, self).__init__(cid, conf, train_data, test_data, device, round)
        self._local_model = None
        self.encoder_distance = 1
        self.encoder_distances = []
        self.weight_scaler = None
        self.client_models = None
        self.projection_size = 2048
        self.round = round
        self._accuracies = []
        self._tracker = None
        self.init_tracker()

    def update_model(self):
        if self.conf.model in [model.MoCo, model.MoCoV2]:
            self.model.encoder_q = self.compressed_model.encoder_q
            # self.model.encoder_k = copy.deepcopy(self._local_model.encoder_k)
        elif self.conf.model == model.SimCLR:
            self.model.online_encoder = self.compressed_model.online_encoder
        elif self.conf.model in [model.SimSiam, model.SimSiamNoSG]:
            if self._local_model is None:
                self.model.online_encoder = self.compressed_model.online_encoder
                self.model.online_predictor = self.compressed_model.online_predictor
                return

            if self.conf.update_encoder == ONLINE:
                online_encoder = self.compressed_model.online_encoder
            else:
                raise ValueError(f"Encoder: aggregate {self.conf.aggregate_encoder}, "
                                 f"update {self.conf.update_encoder} is not supported")

            if self.conf.update_predictor == GLOBAL:
                predictor = self.compressed_model.online_predictor
            else:
                raise ValueError(f"Predictor: {self.conf.update_predictor} is not supported")

            self.model.online_encoder = copy.deepcopy(online_encoder)
            self.model.online_predictor = copy.deepcopy(predictor)

        elif self.conf.model in [model.Symmetric, model.SymmetricNoSG]:
            self.model.online_encoder = self.compressed_model.online_encoder

        elif self.conf.model in [model.BYOL, model.BYOLNoSG, model.BYOLNoPredictor]:

            if self._local_model is None:
                logger.info("Use aggregated encoder and predictor")
                self.model.online_encoder = self.compressed_model.online_encoder
                self.model.target_encoder = self.compressed_model.online_encoder
                self.model.online_predictor = self.compressed_model.online_predictor
                return

            def ema_online():
                self._calculate_weight_scaler()
                logger.info(f"Encoder: update online with EMA of global encoder @ round {self.conf.round_id}")
                weight = self.encoder_distance
                weight = min(1, self.weight_scaler * weight)
                weight = 1 - weight
                self.compressed_model = self.compressed_model.cpu()
                online_encoder = self.compressed_model.online_encoder
                target_encoder = self._local_model.target_encoder
                ema_updater = model.EMA(weight)
                model.update_moving_average(ema_updater, online_encoder, self._local_model.online_encoder)
                return online_encoder, target_encoder

            def ema_predictor():
                logger.info(f"Predictor: use dynamic DAPU")
                distance = self.encoder_distance
                distance = min(1, distance * self.weight_scaler)
                if distance > 0.5:
                    weight = distance
                    ema_updater = model.EMA(weight)
                    predictor = self._local_model.online_predictor
                    model.update_moving_average(ema_updater, predictor, self.compressed_model.online_predictor)
                else:
                    weight = 1 - distance
                    ema_updater = model.EMA(weight)
                    predictor = self.compressed_model.online_predictor
                    model.update_moving_average(ema_updater, predictor, self._local_model.online_predictor)
                return predictor

            if self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == ONLINE:
                logger.info("Encoder: aggregate online, update online")
                online_encoder = self.compressed_model.online_encoder
                target_encoder = self._local_model.target_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == ONLINE:
                logger.info("Encoder: aggregate target, update online")
                online_encoder = self.compressed_model.target_encoder
                target_encoder = self._local_model.target_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == TARGET:
                logger.info("Encoder: aggregate target, update target")
                online_encoder = self._local_model.online_encoder
                target_encoder = self.compressed_model.target_encoder
            elif self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == TARGET:
                logger.info("Encoder: aggregate online, update target")
                online_encoder = self._local_model.online_encoder
                target_encoder = self.compressed_model.online_encoder
            elif self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == BOTH:
                logger.info("Encoder: aggregate online, update both")
                online_encoder = self.compressed_model.online_encoder
                target_encoder = self.compressed_model.online_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == BOTH:
                logger.info("Encoder: aggregate target, update both")
                online_encoder = self.compressed_model.target_encoder
                target_encoder = self.compressed_model.target_encoder
            elif self.conf.update_encoder == NONE:
                logger.info("Encoder: use local online and target encoders")
                online_encoder = self._local_model.online_encoder
                target_encoder = self._local_model.target_encoder
            elif self.conf.update_encoder == EMA:
                logger.info(f"Encoder: use EMA, weight {self.conf.encoder_weight}")
                online_encoder = self._local_model.online_encoder
                ema_updater = model.EMA(self.conf.encoder_weight)
                model.update_moving_average(ema_updater, online_encoder, self.compressed_model.online_encoder)
                target_encoder = self._local_model.target_encoder
            elif self.conf.update_encoder == DYNAMIC_EMA_ONLINE:
                # Use FedEMA to update online encoder
                online_encoder, target_encoder = ema_online()
            elif self.conf.update_encoder == SELECTIVE_EMA:
                # Use FedEMA to update online encoder
                # For random selection, only update with EMA when the client is selected in previous round.
                if self.previous_trained_round + 1 == self.conf.round_id:
                    online_encoder, target_encoder = ema_online()
                else:
                    logger.info(f"Encoder: update online and target @ round {self.conf.round_id}")
                    online_encoder = self.compressed_model.online_encoder
                    target_encoder = self.compressed_model.online_encoder
            else:
                raise ValueError(f"Encoder: aggregate {self.conf.aggregate_encoder}, "
                                 f"update {self.conf.update_encoder} is not supported")

            if self.conf.update_predictor == GLOBAL:
                logger.info("Predictor: use global predictor")
                predictor = self.compressed_model.online_predictor
            elif self.conf.update_predictor == LOCAL:
                logger.info("Predictor: use local predictor")
                predictor = self._local_model.online_predictor

            elif self.conf.update_predictor == SELECTIVE_EMA:
                # For random selection, only update with EMA when the client is selected in previous round.
                if self.previous_trained_round + 1 == self.conf.round_id:
                    predictor = ema_predictor()
                else:
                    logger.info("Predictor: use global predictor")
                    predictor = self.compressed_model.online_predictor
            elif self.conf.update_predictor == EMA:
                logger.info(f"Predictor: use EMA, weight {self.conf.predictor_weight}")
                predictor = self._local_model.online_predictor
                ema_updater = model.EMA(self.conf.predictor_weight)
                model.update_moving_average(ema_updater, predictor, self.compressed_model.online_predictor)
            else:
                raise ValueError(f"Predictor: {self.conf.update_predictor} is not supported")

            self.model.online_encoder = copy.deepcopy(online_encoder)
            self.model.target_encoder = copy.deepcopy(target_encoder)
            self.model.online_predictor = copy.deepcopy(predictor)

    def train(self, conf, device):
        import time
        import copy
        import torch
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()
        start_time = time.time()

        # 原始风格：一般不需要 init_distributed_mode（除非你真用DDP）
        # utils.init_distributed_mode(args)

        self.model.train()
        self.model.to(device)

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=conf.optimizer.lr,
            momentum=conf.optimizer.momentum,
            weight_decay=conf.optimizer.weight_decay
        )

        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)

        self.train_loss = []

        # -----------------------------
        # 回退关键点：teacher 常驻 GPU
        # -----------------------------
        if self.client_models is not None:
            new_client_models = []
            for c_models in self.client_models:
                enc = c_models.online_encoder
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad_(False)
                enc.to(device)  # 直接放GPU常驻
                new_client_models.append(enc)
            self.client_models = new_client_models

        for epoch in range(conf.server_epoch):
            batch_losses = []
            t0 = time.time()

            for it, ((batched_x1, batched_x2), _) in enumerate(self.train_loader):
                x1 = batched_x1.to(device, non_blocking=True)
                x2 = batched_x2.to(device, non_blocking=True)

                # -------- teacher 输出（不搬运）--------
                if self.client_models is None:
                    client_result = None
                else:
                    # N = num_of_clients
                    outs1 = []
                    outs2 = []
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        for teacher in self.client_models:
                            outs1.append(teacher(x1))  # (B, K)
                            outs2.append(teacher(x2))  # (B, K)

                    # (B, N, K)
                    R_clis_x1 = torch.stack(outs1, dim=1).to(torch.float16)
                    R_clis_x2 = torch.stack(outs2, dim=1).to(torch.float16)

                    # (2B, N, K)
                    client_result = torch.cat([R_clis_x1, R_clis_x2], dim=0)

                # -------- student 训练 --------
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = self.model(x1, x2, client_result, device)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 这里用 detach，避免每步 item() 同步
                batch_losses.append(loss.detach().float())

                if (it + 1) % 50 == 0:
                    # 端到端吞吐
                    dt = time.time() - t0
                    avg_sec = dt / 50.0
                    loss_avg = torch.stack(batch_losses[-10:]).mean().item()
                    print(f"[server epoch {epoch + 1}] iter {it + 1}/{len(self.train_loader)} "
                          f"avg_sec/iter={avg_sec:.3f} loss_avg={loss_avg:.4f}")
                    t0 = time.time()

                if conf.model in [model.BYOL, model.BYOLServer] and conf.momentum_update:
                    self.model.update_moving_average()

            epoch_loss = torch.stack(batch_losses).mean().item()
            self.train_loss.append(float(epoch_loss))

        self.train_time = time.time() - start_time
        self._local_model = copy.deepcopy(self.model).cpu()
        return self._local_model

    def test(self):
        """Testing process of federated learning."""
        self.print_("--- start testing ---")
        transformation = self._load_transform_test()
        if self.conf.data.dataset == CIFAR100:
            data_path = "./data/cifar100"
            train_dataset = datasets.CIFAR100(data_path, download=True, transform=transformation)
            test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
        else:
            data_path = "./data/cifar10"
            train_dataset = datasets.CIFAR10(data_path, download=True, transform=transformation)
            test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=8)

        test_begin_time = time.time()
        test_results = {metric.TEST_ACCURACY: 0, metric.TEST_LOSS: 0, metric.TEST_TIME: 0}
        test_results = self.test_in_server(self.conf, self.device)
        test_results[metric.TEST_TIME] = time.time() - test_begin_time

        self.track_test_results(test_results)
        self.save_tracker()
        logger.info(f"Accuracies: {rounding(self._accuracies, 4)}")

    def init_tracker(self):
        """Initialize tracking"""
        if self.conf.server.track:
            self._tracker = init_tracking(self.conf.tracking.database, self.conf.tracker_addr)

    def track_test_results(self, results):
        """Track test results collected from clients.

        Args:
            results (dict): Test metrics, format in {"test_loss": value, "test_accuracy": value, "test_time": value}
        """
        self._accuracies.append(results[metric.TEST_ACCURACY])

        for metric_name in results:
            self.track(metric_name, results[metric_name])

        self.print_('Test time {:.2f}s, Test loss: {:.2f}, Test accuracy: {:.2f}%'.format(
            results[metric.TEST_TIME], results[metric.TEST_LOSS], results[metric.TEST_ACCURACY]))

    def save_tracker(self):
        """Save metrics in the tracker to database."""
        if self._tracker:
            self._tracker.save_round()

    def load_optimizer(self, conf):
        lr = conf.optimizer.lr
        # if conf.optimizer.lr_type == "cosine":
        #     lr = compute_lr(conf.round_id, conf.rounds, 0, conf.optimizer.lr)

        # movo_v1 should use the default learning rate
        if conf.model == model.MoCo:
            lr = conf.optimizer.lr

        params = self.model.parameters()
        if conf.model in [model.BYOL]:
            params = [
                {'params': self.model.online_encoder.parameters()},
                {'params': self.model.online_predictor.parameters()}
            ]

        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr)
        else:
            optimizer = torch.optim.SGD(params,
                                        lr=lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        return optimizer

    def _load_transform(self, conf):
        transformation = utils.get_transformation(conf.model)
        return transformation(conf.image_size, conf.gaussian)

    def _load_transform_test(self):
        transformation = utils.get_transformation(self.conf.model)
        return transformation().test_transform

    def load_loader(self, conf):
        drop_last = conf.drop_last
        train_loader = self.train_data.loader(conf.batch_size,
                                              self.cid,
                                              shuffle=True,
                                              drop_last=drop_last,
                                              # seed=conf.seed,
                                              seed=getattr(conf, "seed", 0),
                                              transform=self._load_transform(conf))
        return train_loader

    def test_in_server(self, conf, device):
        testing_model = self.model.online_encoder
        testing_model.eval()
        testing_model.to(device)

        with torch.no_grad():
            accuracy = knn_monitor(testing_model, self.train_loader, self.test_loader, device=device)

        test_results = {
            metric.TEST_ACCURACY: float(accuracy),
            metric.TEST_LOSS: 0,
        }
        return test_results

    def _get_testing_model(self, net=False):
        if self.conf.server.model in [model.MoCo, model.MoCoV2]:
            testing_model = self.model.encoder_q
        else:
            # # BYOL
            # if self.conf.client.aggregate_encoder == TARGET:
            #     self.print_("Use aggregated target encoder for testing")
            #     testing_model = self.model.target_encoder
            # else:
            #     self.print_("Use aggregated online encoder for testing")
            #     testing_model = self.model.online_encoder
            testing_model = self.model
        return testing_model

    def save_model(self):
        # if self._do_every(self.conf.server.save_model_every, self._current_round, self.conf.server.rounds):
        save_path = self.conf.server.save_model_path
        if save_path == "":
            save_path = os.path.join(os.getcwd(), "saved_models", self.conf.task_id)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path,
                                 "global_model.pth")

        torch.save(self._get_testing_model().cpu().state_dict(), save_path)
        self.print_("Encoder model saved at {}".format(save_path))

        if self.conf.server.save_predictor:
            if self.conf.model in [model.SimSiam, model.BYOL]:
                save_path = save_path.replace("global_model", "predictor")
                torch.save(self.model.online_predictor.cpu().state_dict(), save_path)
                self.print_("Predictor model saved at {}".format(save_path))

    def print_(self, content):
        logger.info(content)
