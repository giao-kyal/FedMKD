#coding=utf-8
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
from communication import ONLINE, TARGET, BOTH, LOCAL, GLOBAL, DAPU, NONE, EMA, DYNAMIC_DAPU, DYNAMIC_EMA_ONLINE, SELECTIVE_EMA
from easyfl.client.base import BaseClient
from tqdm import tqdm

from src.reid.reid_wrapper import ReIDWrapper
from src.reid.triplet_loss import TripletLoss

logger = logging.getLogger(__name__)

L2 = "l2"


class FedSSLClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, round):
        super(FedSSLClient, self).__init__(cid, conf, train_data, test_data, device, round)
        self.local_model = None
        self.DAPU_predictor = LOCAL
        self.encoder_distance = 1
        self.encoder_distances = []
        self.previous_trained_round = -1
        self.weight_scaler = None
        self.round = round

    def decompression(self):
        if self.model is None:
            # Initialization at beginning of the task
            self.model = self.compressed_model

        self.update_model()

    def update_model(self):
        if self.conf.model in [model.MoCo, model.MoCoV2]:
            self.model.encoder_q = self.compressed_model.encoder_q
            # self.model.encoder_k = copy.deepcopy(self.local_model.encoder_k)
        elif self.conf.model == model.SimCLR:
            self.model.online_encoder = self.compressed_model.online_encoder
        elif self.conf.model in [model.SimSiam, model.SimSiamNoSG]:
            if self.local_model is None:
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

            if self.local_model is None:
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
                target_encoder = self.local_model.target_encoder
                ema_updater = model.EMA(weight)
                model.update_moving_average(ema_updater, online_encoder, self.local_model.online_encoder)
                return online_encoder, target_encoder

            def ema_predictor():
                logger.info(f"Predictor: use dynamic DAPU")
                distance = self.encoder_distance
                distance = min(1, distance * self.weight_scaler)
                if distance > 0.5:
                    weight = distance
                    ema_updater = model.EMA(weight)
                    predictor = self.local_model.online_predictor
                    model.update_moving_average(ema_updater, predictor, self.compressed_model.online_predictor)
                else:
                    weight = 1 - distance
                    ema_updater = model.EMA(weight)
                    predictor = self.compressed_model.online_predictor
                    model.update_moving_average(ema_updater, predictor, self.local_model.online_predictor)
                return predictor

            if self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == ONLINE:
                logger.info("Encoder: aggregate online, update online")
                online_encoder = self.compressed_model.online_encoder
                target_encoder = self.local_model.target_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == ONLINE:
                logger.info("Encoder: aggregate target, update online")
                online_encoder = self.compressed_model.target_encoder
                target_encoder = self.local_model.target_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == TARGET:
                logger.info("Encoder: aggregate target, update target")
                online_encoder = self.local_model.online_encoder
                target_encoder = self.compressed_model.target_encoder
            elif self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == TARGET:
                logger.info("Encoder: aggregate online, update target")
                online_encoder = self.local_model.online_encoder
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
                online_encoder = self.local_model.online_encoder
                target_encoder = self.local_model.target_encoder
            elif self.conf.update_encoder == EMA:
                logger.info(f"Encoder: use EMA, weight {self.conf.encoder_weight}")
                online_encoder = self.local_model.online_encoder
                ema_updater = model.EMA(self.conf.encoder_weight)
                model.update_moving_average(ema_updater, online_encoder, self.compressed_model.online_encoder)
                target_encoder = self.local_model.target_encoder
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
                predictor = self.local_model.online_predictor
            elif self.conf.update_predictor == DAPU:
                # Divergence-aware predictor update (DAPU)
                logger.info(f"Predictor: use DAPU, mu {self.conf.dapu_threshold}")
                if self.DAPU_predictor == GLOBAL:
                    predictor = self.compressed_model.online_predictor
                elif self.DAPU_predictor == LOCAL:
                    predictor = self.local_model.online_predictor
                else:
                    raise ValueError(f"Predictor: DAPU predictor can either use local or global predictor")
            elif self.conf.update_predictor == DYNAMIC_DAPU:
                # Use FedEMA to update predictor
                predictor = ema_predictor()
            elif self.conf.update_predictor == SELECTIVE_EMA:
                # For random selection, only update with EMA when the client is selected in previous round.
                if self.previous_trained_round + 1 == self.conf.round_id:
                    predictor = ema_predictor()
                else:
                    logger.info("Predictor: use global predictor")
                    predictor = self.compressed_model.online_predictor
            elif self.conf.update_predictor == EMA:
                logger.info(f"Predictor: use EMA, weight {self.conf.predictor_weight}")
                predictor = self.local_model.online_predictor
                ema_updater = model.EMA(self.conf.predictor_weight)
                model.update_moving_average(ema_updater, predictor, self.compressed_model.online_predictor)
            else:
                raise ValueError(f"Predictor: {self.conf.update_predictor} is not supported")

            self.model.online_encoder = copy.deepcopy(online_encoder)
            self.model.target_encoder = copy.deepcopy(target_encoder)
            self.model.online_predictor = copy.deepcopy(predictor)

    def train(self, conf, device):
        scaler = GradScaler()
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        if conf.model in [model.MoCo, model.MoCoV2]:
            self.model.reset_key_encoder()
        self.train_loss = []
        self.model.to(device)
        old_model = copy.deepcopy(nn.Sequential(*list(self.model.children())[:-1])).cpu()
        for i in range(conf.local_epoch):
            if conf.data_number == 'small':
                idx = 30
            else:
                idx = len(self.train_loader)

            batch_loss = []
            for (batched_x1, batched_x2), _ in self.train_loader:
                x1, x2 = batched_x1.to(device), batched_x2.to(device)
                optimizer.zero_grad()

                if conf.model in [model.MoCo, model.MoCoV2]:
                    loss = self.model(x1, x2, device)
                elif conf.model == model.SimCLR:
                    images = torch.cat((x1, x2), dim=0)
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = loss_fn(logits, labels)
                else:
                    with autocast():
                        loss = self.model(x1, x2)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                batch_loss.append(loss.item())
                scaler.update()

                if conf.model in [model.BYOL] and conf.momentum_update:
                    self.model.update_moving_average()
                idx = idx - 1
                if idx == 0:
                    break
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
        self.train_time = time.time() - start_time

        # store trained model locally
        self.local_model = copy.deepcopy(self.model).cpu()
        self.previous_trained_round = conf.round_id
        if conf.update_predictor in [DAPU, DYNAMIC_DAPU, SELECTIVE_EMA] or conf.update_encoder in [DYNAMIC_EMA_ONLINE, SELECTIVE_EMA]:
            new_model = copy.deepcopy(nn.Sequential(*list(self.model.children())[:-1])).cpu()
            self.encoder_distance = self._calculate_divergence(old_model, new_model)
            self.encoder_distances.append(self.encoder_distance.item())
            self.DAPU_predictor = self._DAPU_predictor_usage(self.encoder_distance)
            if self.conf.auto_scaler == 'y' and self.conf.random_selection:
                self._calculate_weight_scaler()
            if (conf.round_id + 1) % 100 == 0:
                logger.info(f"Client {self.cid}, encoder distances: {self.encoder_distances}")

    def align_train(self, conf, device):
        scaler = GradScaler()
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        optimizer = torch.optim.SGD(
            [
                {"params": self.model.bnneck.parameters()},
                {"params": self.model.classifier.parameters()},
            ],
            lr=conf.optimizer.lr,
            momentum=conf.optimizer.momentum,
            weight_decay=conf.optimizer.weight_decay,
        )
        # 1) optimizer 里参数数量
        num_opt_params = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        num_opt_trainable = sum(p.numel() for g in optimizer.param_groups for p in g["params"] if p.requires_grad)

        # 2) 检查这些参数在什么 device
        for gi, g in enumerate(optimizer.param_groups):
            ps = [p for p in g["params"] if p is not None]

        triplet = TripletLoss(margin=getattr(conf, "triplet_margin", 0.3),
                              hard_factor=getattr(conf, "hard_factor", 0.0))
        tri_w = getattr(conf, "triplet_weight", 1.0)  # 你可以在配置里加这个
        ce_w = getattr(conf, "ce_weight", 1.0)
        if conf.model in [model.MoCo, model.MoCoV2]:
            self.model.reset_key_encoder()
        self.train_loss = []
        self.model.to(device)
        old_model = copy.deepcopy(nn.Sequential(*list(self.model.children())[:-1])).cpu()
        idx = len(self.train_loader)
        for i in range(conf.local_epoch):
            print(i)
            batch_loss = []
            for batch in self.train_loader:
                # 兼容：有的reid loader返回 (img, pid, camid, ...)
                # 你当前代码是 ((x1,x2), _)，所以先按两种情况处理
                if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
                    (batched_x1, batched_x2), target = batch
                    x1, x2 = batched_x1.to(device, non_blocking=True), batched_x2.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                else:
                    # (img, target, ...)
                    batched_x1 = batch[0]
                    target = batch[1]
                    x1 = batched_x1.to(device, non_blocking=True)
                    x2 = None
                    target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if conf.model in [model.MoCo, model.MoCoV2]:
                    loss = self.model(x1, x2, device)

                elif conf.model == model.SimCLR:
                    images = torch.cat((x1, x2), dim=0)
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = loss_fn(logits, labels)

                else:
                    # ---- ReID supervised align (CE) ----
                    with autocast():
                        score1, feat1 = self.model(x1, target=target)
                        loss_ce1 = loss_fn(score1, target)

                    # Triplet 通常建议用 fp32 更稳（尤其是 distance matrix）
                    feat1_fp32 = feat1.float()
                    loss_tri1, dist_ap1, dist_an1 = triplet(feat1_fp32, target, normalize_feature=False)

                    loss = ce_w * loss_ce1 + tri_w * loss_tri1

                    if x2 is not None:
                        with autocast():
                            score2, feat2 = self.model(x2, target=target)
                            loss_ce2 = loss_fn(score2, target)

                        feat2_fp32 = feat2.float()
                        loss_tri2, dist_ap2, dist_an2 = triplet(feat2_fp32, target, normalize_feature=False)

                        loss = 0.5 * (loss + (ce_w * loss_ce2 + tri_w * loss_tri2))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_loss.append(float(loss.detach()))

            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
        self.train_time = time.time() - start_time

        # store trained model locally
        self.local_model = copy.deepcopy(self.model).cpu()
        self.previous_trained_round = conf.round_id
        if conf.update_predictor in [DAPU, DYNAMIC_DAPU, SELECTIVE_EMA] or conf.update_encoder in [DYNAMIC_EMA_ONLINE, SELECTIVE_EMA]:
            new_model = copy.deepcopy(nn.Sequential(*list(self.model.children())[:-1])).cpu()
            self.encoder_distance = self._calculate_divergence(old_model, new_model)
            self.encoder_distances.append(self.encoder_distance.item())
            self.DAPU_predictor = self._DAPU_predictor_usage(self.encoder_distance)
            if self.conf.auto_scaler == 'y' and self.conf.random_selection:
                self._calculate_weight_scaler()
            if (conf.round_id + 1) % 100 == 0:
                logger.info(f"Client {self.cid}, encoder distances: {self.encoder_distances}")

    def _DAPU_predictor_usage(self, distance):
        if distance < self.conf.dapu_threshold:
            return GLOBAL
        else:
            return LOCAL

    def _calculate_divergence(self, old_model, new_model, typ=L2):
        size = 0
        total_distance = 0
        old_dict = old_model.state_dict()
        new_dict = new_model.state_dict()
        for name, param in old_model.named_parameters():
            if 'conv' in name and 'weight' in name:
                total_distance += self._calculate_distance(old_dict[name].detach().clone().view(1, -1),
                                                           new_dict[name].detach().clone().view(1, -1),
                                                           typ)
                size += 1
        distance = total_distance / size
        logger.info(f"Model distance: {distance} = {total_distance}/{size}")
        return distance

    def _calculate_distance(self, m1, m2, typ=L2):
        if typ == L2:
            return torch.dist(m1, m2, 2)

    def _calculate_weight_scaler(self):
        if not self.weight_scaler:
            if self.conf.auto_scaler == 'y':
                self.weight_scaler = self.conf.auto_scaler_target / self.encoder_distance
            else:
                self.weight_scaler = self.conf.weight_scaler
            logger.info(f"Client {self.cid}: weight scaler {self.weight_scaler}")

    def load_loader(self, conf):
        drop_last = conf.drop_last
        train_loader = self.train_data.loader(conf.batch_size,
                                              self.cid,
                                              shuffle=True,
                                              drop_last=drop_last,
                                              seed=conf.seed,
                                              transform=self._load_transform(conf))
        if self.cid is not None:
            _print_label_count(self.cid, self.train_data.data[self.cid]['y'])
        return train_loader

    def load_optimizer(self, conf):
        lr = conf.optimizer.lr
        if conf.optimizer.lr_type == "cosine":
            lr = compute_lr(conf.round_id, conf.rounds, 0, conf.optimizer.lr)

        if conf.model == model.MoCo:
            lr = conf.optimizer.lr

        # 默认：优化整个模型（适用于 ReIDWrapper）
        params = self.model.parameters()

        # BYOL：只有当模型真有 online_encoder/online_predictor 才用该分组
        if conf.model in [model.BYOL] and hasattr(self.model, "online_encoder"):
            param_groups = [{'params': self.model.online_encoder.parameters()}]
            if hasattr(self.model, "online_predictor") and self.model.online_predictor is not None:
                param_groups.append({'params': self.model.online_predictor.parameters()})
            params = param_groups

        # 可选：如果你希望 ReIDWrapper 对齐阶段只训 bnneck+classifier（更常见）
        if isinstance(self.model, ReIDWrapper):
            params = [
                {'params': self.model.bnneck.parameters()},
                {'params': self.model.classifier.parameters()},
            ]

        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr)
        else:
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=conf.optimizer.momentum,
                weight_decay=conf.optimizer.weight_decay
            )
        return optimizer

    def _load_transform(self, conf):
        transformation = utils.get_transformation(conf.model)
        return transformation(conf.image_size, conf.gaussian)

    def info_nce_loss(self, features, n_views=2, temperature=0.07):
        labels = torch.cat([torch.arange(self.conf.batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     n_views * self.conf.batch_size, n_views * self.conf.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return logits, labels

    def _get_testing_model(self, net=False):
        if self.conf.client.model in [model.MoCo, model.MoCoV2]:
            testing_model = self.model.encoder_q
        else:
            # BYOL
            # if self.conf.client.aggregate_encoder == TARGET:
            #     self.print_("Use aggregated target encoder for testing")
            #     testing_model = self.model.target_encoder
            # else:
            #     self.print_("Use aggregated online encoder for testing")
            #     testing_model = self.model.online_encoder
            testing_model = self.model
        return testing_model

    def save_model(self):
        save_path = self.conf.client.save_model_path
        if save_path == "":
            save_path = os.path.join(os.getcwd(), "saved_models", self.conf.task_id)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path,
                                 "local_model_{}_{}.pth".format(self.cid, self.round))

        torch.save(self._get_testing_model().cpu().state_dict(), save_path)
        self.print_("Encoder model saved at {}".format(save_path))

        if self.conf.client.save_predictor:
            if self.conf.client.model in [model.SimSiam, model.BYOL]:
                save_path = save_path.replace("model", "predictor")
                torch.save(self.model.online_predictor.cpu().state_dict(), save_path)
                self.print_("Predictor model saved at {}".format(save_path))

    def print_(self, content):
        logger.info(content)

def compute_lr(current_round, rounds=800, eta_min=0, eta_max=0.3):
    """Compute learning rate as cosine decay"""
    pi = np.pi
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * current_round / rounds) + 1)
    return eta_t


def _print_label_count(cid, labels):
    logger.info(f"client {cid}: {Counter(labels)}")
