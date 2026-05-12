import copy
import easyfl

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

from easyfl.models.model import BaseModel
from easyfl.models.resnet import ResNet18, ResNet34, ResNet50
from easyfl.models.simple_cnn import Model
from easyfl.models.vgg9 import VGG9
import logging
# from cka import cka_score

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore")

SimSiam = "simsiam"
SimCLR = "simclr"
MoCo = "moco"
MoCoV2 = "moco_v2"
BYOL = "byol"
BYOLServer = "byolserver"
FedMD = "fedmd"
Symmetric = "symmetric"
SymmetricNoSG = "symmetric_no_sg"

OneLayer = "1_layer"
TwoLayer = "2_layer"

CNN = "cnn"
VGG = "vgg"
RESNET18 = "resnet18"
RESNET34 = "resnet34"
RESNET50 = "resnet50"
SRESNET18 = "server_resnet"

def get_encoder(arch=RESNET18):
    return models.__dict__[arch]


def get_model(model, encoder_network, predictor_network=TwoLayer):
    mlp = False
    T = 0.07
    stop_gradient = True
    has_predictor = True
    if model == SymmetricNoSG:
        stop_gradient = False
        model = Symmetric
    elif model == MoCoV2:
        model = MoCo
        mlp = True
        T = 0.2

    if encoder_network == RESNET18:
        net = ResNet18()
    elif encoder_network == RESNET34:
        net = ResNet34()
    elif encoder_network == RESNET50:
        net = ResNet50()
    elif encoder_network == CNN:
        net = Model()
    elif encoder_network == VGG:
        net = VGG9()
    elif encoder_network == SRESNET18:
        net = models.resnet18(pretrained = True)

    if model == Symmetric:
        if encoder_network == RESNET50:
            return SymmetricModel(net=ResNet50(), stop_gradient=stop_gradient)
        else:
            return SymmetricModel(stop_gradient=stop_gradient)
    elif model == SimSiam:
        # net = ResNet18()
        # if encoder_network == RESNET50:
        #     net = ResNet50()
        return SimSiamModel(net=net, stop_gradient=stop_gradient)
    elif model == MoCo:
        # net = ResNet18
        # if encoder_network == RESNET50:
        #     net = ResNet50
        return MoCoModel(net=net, mlp=mlp, T=T)
    elif model == BYOL:
        return BYOLModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                         predictor_network=predictor_network)
    elif model == BYOLServer:
        return BYOLServerModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                               predictor_network=predictor_network)
    elif model == 'KLserver':
        return KLServerModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                               predictor_network=predictor_network)
    elif model == 'WeightServer':
        return WeightServerModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                                predictor_network=predictor_network)
    elif model == FedMD:
        return FedmdServerModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                                predictor_network=predictor_network)
    elif model == 'single':
        return SingleServerModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                                 predictor_network=predictor_network)
    elif model == 'ssfl':
        return SSFLServerModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                               predictor_network=predictor_network)
    elif model == 'fedet':
        return WeightServerModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                                predictor_network=predictor_network)
                                
    elif model == SimCLR:
        # net = ResNet18()
        # if encoder_network == RESNET50:
        #     net = ResNet50()
        return SimCLRModel(net=net)
    else:
        raise NotImplementedError


def get_encoder_network(model, encoder_network, num_classes=10, projection_size=2048, projection_hidden_size=4096):
    if model in [MoCo, MoCoV2]:
        num_classes = 128

    if encoder_network == RESNET18:
        resnet = ResNet18(num_classes=num_classes)
    elif encoder_network == RESNET34:
        resnet = ResNet34(num_classes=num_classes)
    elif encoder_network == RESNET50:
        resnet = ResNet50(num_classes=num_classes)
    elif encoder_network == VGG:
        resnet = VGG9()
    elif encoder_network == CNN:
        resnet = Model()
    else:
        raise NotImplementedError

    if not hasattr(resnet, 'feature_dim'):
        feature_dim = list(resnet.children())[-1].in_features
    else:
        feature_dim = resnet.feature_dim

    if model in [Symmetric, SimSiam, BYOL, SymmetricNoSG, SimCLR]:
        resnet.fc = MLP(feature_dim, projection_size, projection_hidden_size)
    if model == MoCoV2:
        resnet.fc = MLP(feature_dim, num_classes, resnet.feature_dim)

    return resnet



class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=TwoLayer):
        super().__init__()
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)


class SimilarityModel(nn.Module):
    def __init__(self, vector_size, num_vectors):
        super(SimilarityModel, self).__init__()
        # self.p_weights = torch.nn.Parameter(torch.randn(vector_size, vector_size))
        # self.q_weights = torch.nn.Parameter(torch.randn(vector_size, vector_size))
        self.weights = torch.nn.Parameter(torch.randn(vector_size, vector_size))
        self.weights.requires_grad_(True)

    def forward(self, query_vector, vectors):
        # 计算KL散度
        p = F.softmax(query_vector @ self.weights, dim=1)
        q = F.softmax(vectors @ self.weights, dim=1)
        kl_divergences = F.kl_div(p, q, reduction='none').sum(dim=2)

        # 使用softmax函数来计算权重，使得权重之和为1
        weights = F.softmax(-kl_divergences, dim=1)

        return weights


class SelfAttention(nn.Module):
    # def __init__(self, k_dim):
    #     super(SelfAttention, self).__init__()
    #     self.query_weight = nn.Parameter(torch.randn(k_dim, 1))
    #     self.key_weight = nn.Parameter(torch.randn(k_dim, k_dim))
    #     self.scale = torch.sqrt(torch.tensor(k_dim, dtype=torch.float32))
    #
    # def forward(self, query, keys):
    #     """
    #     Args:
    #     query (torch.Tensor): K维向量，形状为[K]
    #     keys (torch.Tensor): N个K维向量，形状为[N, K]
    #
    #     Returns:
    #     torch.Tensor: N个权重，形状为[N]
    #     """
    #     query = torch.matmul(query, self.query_weight).transpose(0, -1) / self.scale
    #     keys = torch.matmul(keys, self.key_weight) / self.scale
    #     scores = torch.matmul(query, keys.transpose(-1, -2))  # 计算query和keys之间的点积，得分形状为[1, N]
    #     weights = F.softmax(scores, dim=-1)  # 在最后一个维度上应用softmax以得到权重，权重的形状为[1, N]
    #
    #     return weights.squeeze(0)  # 去掉第一个维度，得到形状为[N]的权重张量
    def __init__(self, k_dim):
        super(SelfAttention, self).__init__()
        self.scale = (k_dim ** 0.5)

    def forward(self, query, keys):
        # query: [B, 1, K], keys: [B, N, K]
        scores = torch.matmul(query, keys.transpose(-1, -2)) / self.scale  # [B, 1, N]
        weights = F.softmax(scores, dim=-1).squeeze(1)  # [B, N]
        return weights


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


def D_NO_SG(p, z, version='simplified'):  # negative cosine similarity without stop gradient
    if version == 'original':
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception


# ------------- BYOL Model -----------------


class BYOLModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer,
    ):
        super().__init__()

        self.online_encoder = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim
        self.online_encoder.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor

        # debug purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two):
        online_pred_one = self.online_encoder(image_one)  # batch*2048
        online_pred_two = self.online_encoder(image_two)

        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        if self.stop_gradient:
            with torch.no_grad():
                if self.target_encoder is None:
                    self.target_encoder = self._get_target_encoder()
                target_proj_one = self.target_encoder(image_one)
                target_proj_two = self.target_encoder(image_two)

                target_proj_one = target_proj_one.detach()
                target_proj_two = target_proj_two.detach()

        else:
            if self.target_encoder is None:
                self.target_encoder = self._get_target_encoder()
            target_proj_one = self.target_encoder(image_one)
            target_proj_two = self.target_encoder(image_two)

        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two

        return loss.mean()


# ===== BYOL server =======
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


# adapter model
class Adapter():
    def __init__(self, in_models, pool_size):
        # representations of teachers
        pool_ch = pool_size[1]  # 64
        pool_w = pool_size[2]  # 8
        LR_list = []
        torch.manual_seed(1)
        self.theta = torch.randn(len(in_models), pool_ch)  # [3, 64]
        self.theta.requires_grad_(True)

        self.max_feat = nn.MaxPool2d(kernel_size=(pool_w, pool_w), stride=pool_w)
        self.W = torch.randn(pool_ch, 1)
        self.W.requires_grad_(True)
        self.val = False

    def loss(self, y, labels, weighted_logits, T=10.0, alpha=0.7):
        ls = nn.KLDivLoss()(F.log_softmax(y / T), weighted_logits) * (T * T * 2.0 * alpha) + F.cross_entropy(y,
                                                                                                             labels) * (
                     1. - alpha)
        if not self.val:
            ls += 0.1 * (torch.sum(self.W * self.W) + torch.sum(torch.sum(self.theta * self.theta, dim=1), dim=0))
        return ls

    def gradient(self, lr=0.01):
        self.W.data = self.W.data - lr * self.W.grad.data
        # Manually zero the gradients after updating weights
        self.W.grad.data.zero_()

    def eval(self):
        self.val = True
        self.theta.detach()
        self.W.detach()

    # input size: [64, 8, 8], [128, 3, 10]
    def forward(self, conv_map, te_logits_list):
        beta = self.max_feat(conv_map)
        beta = torch.squeeze(beta)  # [128, 64]

        latent_factor = []
        for t in self.theta:
            latent_factor.append(beta * t)
        #         latent_factor = torch.stack(latent_factor, dim=0)  # [3, 128, 64]
        alpha = []
        for lf in latent_factor:  # lf.size:[128, 64]
            alpha.append(lf.mm(self.W))
        alpha = torch.stack(alpha, dim=0)  # [3, 128, 1]
        alpha = torch.squeeze(alpha).transpose(0, 1)  # [128, 3]
        miu = F.softmax(alpha)  # [128, 3]
        miu = torch.unsqueeze(miu, dim=2)
        weighted_logits = miu * te_logits_list  # [128, 3, 10]
        weighted_logits = torch.sum(weighted_logits, dim=1)
        #         print(weighted_logits)

        return weighted_logits


class BYOLServerModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer,
            K=10,
            T=0.1,
            N=5,
            queue_len=1024,  # negative number
    ):
        super().__init__()

        self.online_encoder = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim
        self.online_encoder.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)

        self.K_predictor = MLP(projection_size, K, projection_hidden_size, predictor_network)

        self.client_proj = nn.Linear(projection_size, projection_size)

        # self.sim_module = SimilarityModel(K, N)
        self.sim_module = SelfAttention(K)

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor

        # create the queue
        self.register_buffer("queue", torch.randn(projection_size, projection_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K
        self.T = T
        self.queue_len = queue_len
        # debug  purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x, device):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, q, k, device):
        # q is
        # compute query features
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            k, idx_unshuffle = self._batch_shuffle_single_gpu(k, device)
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # 这一行计算了"正样本"的得分，其中 q 和 k 是两个张量（tensor）。
        # 'nc,nc->n' 是 Einstein Summation Notation（爱因斯坦求和符号）的一部分，表示对两个张量 q 和 k 进行逐元素相乘，并在最后一维上求和，得到一个大小为 N 的张量，其中 N 是样本数量。
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # 这一行计算了"负样本"的得分。
        # 同样，q 是一个张量，但是 self.queue.clone().detach() 则是一个包含了负样本信息的张量。
        # 'nc,ck->nk' 也是爱因斯坦求和符号的一部分，表示对 q 和 self.queue 进行逐元素相乘，但这次是在 c 这个维度上求和，最后得到一个大小为 NxK 的张量，其中 N 是样本数量，K 是负样本的数量。
        queue = self.queue.to(q.dtype)  # q 是 fp16 时 queue 也 fp16
        l_neg = torch.einsum('nc,ck->nk', [q, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        loss = nn.CrossEntropyLoss().to(device)(logits, labels)

        return loss, q, k

    def forward(self, image_one, image_two, client_result, device):
        online_pred_one = self.online_encoder(image_one)  # batch*2048
        online_pred_two = self.online_encoder(image_two)

        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        online_pred = torch.cat([online_pred_one, online_pred_two], dim=0)  # 2B,Kdim
        # online_pred: (2B, K0)   client_result: (2B, N, D)
        B2, N, D = client_result.shape  # B2 = 2B
        # (2B, N, D) -> (2B*N, D)
        client_flat = client_result.reshape(B2 * N, D)
        # 一次性投影与预测： (2B*N, D) -> (2B*N, K)
        k_flat = self.K_predictor(self.client_proj(client_flat))
        # reshape 回 (2B, N, K)
        K_result = k_flat.view(B2, N, self.K)

        student_logits = self.K_predictor(online_pred)
        weight = self.sim_module(student_logits.unsqueeze(1), K_result)  # (2B, N)
        teacher_logits = (weight.unsqueeze(2) * K_result).sum(dim=1)
        teacher_labels = teacher_logits.detach().argmax(dim=1)
        return F.cross_entropy(student_logits, teacher_labels)

class WeightServerModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer,
            K=10,
            T=0.1,
            N=5,
            queue_len=1024,  # negative number
    ):
        super().__init__()

        self.online_encoder = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim
        self.online_encoder.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)

        self.K_predictor = MLP(projection_size, K, projection_hidden_size, predictor_network)

        self.sim_module = SimilarityModel(K, N)
        # self.sim_module = SelfAttention(K)

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor

        # create the queue
        self.register_buffer("queue", torch.randn(projection_size, projection_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K
        self.T = T
        self.queue_len = queue_len
        # debug  purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x, device):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, q, k, device):
        # q is
        # compute query features
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            k, idx_unshuffle = self._batch_shuffle_single_gpu(k, device)
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # 这一行计算了"正样本"的得分，其中 q 和 k 是两个张量（tensor）。
        # 'nc,nc->n' 是 Einstein Summation Notation（爱因斯坦求和符号）的一部分，表示对两个张量 q 和 k 进行逐元素相乘，并在最后一维上求和，得到一个大小为 N 的张量，其中 N 是样本数量。
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # 这一行计算了"负样本"的得分。
        # 同样，q 是一个张量，但是 self.queue.clone().detach() 则是一个包含了负样本信息的张量。
        # 'nc,ck->nk' 也是爱因斯坦求和符号的一部分，表示对 q 和 self.queue 进行逐元素相乘，但这次是在 c 这个维度上求和，最后得到一个大小为 NxK 的张量，其中 N 是样本数量，K 是负样本的数量。
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        loss = nn.CrossEntropyLoss().to(device)(logits, labels)

        return loss, q, k

    def forward(self, image_one, image_two, client_result, device):
        online_pred_one = self.online_encoder(image_one)  # batch*2048
        online_pred_two = self.online_encoder(image_two)

        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        online_pred = torch.cat([online_pred_one, online_pred_two], dim=0)  # 2B * 1 * K
        K_result = torch.zeros(online_pred.size(0), client_result.size(1), self.K).to(online_pred.device)  # 2B * N * K
        for i in range(client_result.size(1)):
            tmp_client_result = client_result[:, i, :].squeeze(dim=1)
            K_result[:, i, :] = self.K_predictor(tmp_client_result)
        student_logits = self.K_predictor(online_pred)
        weight = torch.ones(
            student_logits.size(0),
            client_result.size(1),
            dtype=student_logits.dtype,
            device=student_logits.device,
        ) / client_result.size(1)
        teacher_logits = (weight.unsqueeze(2) * K_result).sum(dim=1)
        teacher_labels = teacher_logits.detach().argmax(dim=1)
        return F.cross_entropy(student_logits, teacher_labels)


class KLServerModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer,
            K=10,
            T=0.1,
            N=5,
            queue_len=1024,  # negative number
    ):
        super().__init__()

        self.online_encoder = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim
        self.online_encoder.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)

        self.K_predictor = MLP(projection_size, K, projection_hidden_size, predictor_network)

        self.sim_module = SimilarityModel(K, N)
        # self.sim_module = SelfAttention(K)

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor

        # create the queue
        self.register_buffer("queue", torch.randn(projection_size, projection_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K
        self.T = T
        self.queue_len = queue_len
        # debug  purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x, device):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, q, k, device):
        # q is
        # compute query features
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            k, idx_unshuffle = self._batch_shuffle_single_gpu(k, device)
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # 这一行计算了"正样本"的得分，其中 q 和 k 是两个张量（tensor）。
        # 'nc,nc->n' 是 Einstein Summation Notation（爱因斯坦求和符号）的一部分，表示对两个张量 q 和 k 进行逐元素相乘，并在最后一维上求和，得到一个大小为 N 的张量，其中 N 是样本数量。
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # 这一行计算了"负样本"的得分。
        # 同样，q 是一个张量，但是 self.queue.clone().detach() 则是一个包含了负样本信息的张量。
        # 'nc,ck->nk' 也是爱因斯坦求和符号的一部分，表示对 q 和 self.queue 进行逐元素相乘，但这次是在 c 这个维度上求和，最后得到一个大小为 NxK 的张量，其中 N 是样本数量，K 是负样本的数量。
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        loss = nn.CrossEntropyLoss().to(device)(logits, labels)

        return loss, q, k

    def forward(self, image_one, image_two, client_result, device):
        online_pred_one = self.online_encoder(image_one)  # batch*2048
        online_pred_two = self.online_encoder(image_two)

        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        online_pred = torch.cat([online_pred_one, online_pred_two], dim=0)  # 2B * 1 * K
        K_result = torch.zeros(online_pred.size(0), client_result.size(1), self.K).to(online_pred.device)  # 2B * N * K
        for i in range(client_result.size(1)):
            tmp_client_result = client_result[:, i, :].squeeze(dim=1)
            K_result[:, i, :] = self.K_predictor(tmp_client_result)
        student_logits = self.K_predictor(online_pred)
        weight = self.sim_module(student_logits.unsqueeze(1), K_result)  # 2B * N
        teacher_logits = (weight.unsqueeze(2) * K_result).sum(dim=1)
        teacher_labels = teacher_logits.detach().argmax(dim=1)
        return F.cross_entropy(student_logits, teacher_labels)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

