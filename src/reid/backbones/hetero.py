import os
import sys
import types

import torch
import torch.nn as nn
from torchvision import models

from src.reid.reid_wrapper import ReIDWrapper


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
THIRD_PARTY = os.path.join(PROJECT_ROOT, "third_party")
TRANSREID_BACKBONES = os.path.join(THIRD_PARTY, "TransReID", "model", "backbones")
AGW_ROOT = os.path.join(THIRD_PARTY, "ReID-AGWbaseline")
PRETRAIN_DIR = os.path.join(THIRD_PARTY, "pretrained")

for path in (TRANSREID_BACKBONES, AGW_ROOT):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

if "torch._six" not in sys.modules:
    import collections.abc

    torch_six = types.ModuleType("torch._six")
    torch_six.container_abcs = collections.abc
    sys.modules["torch._six"] = torch_six

from vit_pytorch import deit_small_patch16_224_TransReID, vit_base_patch16_224_TransReID
from modeling.baseline import Baseline as AGWBaseline


class OnlineEncoderModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.online_encoder = encoder

    def forward(self, x):
        return self.online_encoder(x)


class TorchvisionResNet18ReID(nn.Module):
    feature_dim = 2048

    def __init__(self):
        super().__init__()
        resnet = _torchvision_resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, self.feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class TransReIDEncoder(nn.Module):
    feature_dim = 2048

    def __init__(self, variant):
        super().__init__()
        if variant == "transreid_vit_base":
            self.backbone = vit_base_patch16_224_TransReID(
                img_size=(256, 128),
                stride_size=16,
                camera=0,
                view=0,
                local_feature=False,
            )
            in_dim = 768
        elif variant == "transreid_deit_small":
            self.backbone = deit_small_patch16_224_TransReID(
                img_size=(256, 128),
                stride_size=16,
                camera=0,
                view=0,
                local_feature=False,
            )
            in_dim = 384
        else:
            raise ValueError(f"Unsupported TransReID variant: {variant}")
        _load_transreid_pretrain(self.backbone, variant)
        self.proj = nn.Linear(in_dim, self.feature_dim)

    def forward(self, x):
        x = self.backbone(x, cam_label=None, view_label=None)
        return self.proj(x)


class AGWEncoder(nn.Module):
    feature_dim = 2048

    def __init__(self):
        super().__init__()
        self.model = AGWBaseline(
            num_classes=1,
            last_stride=1,
            model_path="",
            model_name="resnet50",
            gem_pool="on",
            pretrain_choice="none",
        )
        _load_torchvision_resnet50_into(self.model.base, "AGW")

    def forward(self, x):
        x = self.model.base(x)
        x = self.model.global_pool(x)
        return torch.flatten(x, 1)


class PCBEncoder(nn.Module):
    feature_dim = 2048

    def __init__(self, part_num=6):
        super().__init__()
        self.part_num = part_num
        resnet = _torchvision_resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((part_num, 1))
        self.proj = nn.Linear(part_num * 2048, self.feature_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.proj(x)


def build_reid_client_model(encoder_name, num_classes):
    encoder_name = encoder_name.lower()
    if encoder_name == "resnet18":
        encoder = TorchvisionResNet18ReID()
    elif encoder_name in {"transreid_vit_base", "vit_base", "vit"}:
        encoder = TransReIDEncoder("transreid_vit_base")
    elif encoder_name in {"transreid_deit_small", "deit_small", "deit"}:
        encoder = TransReIDEncoder("transreid_deit_small")
    elif encoder_name == "agw":
        encoder = AGWEncoder()
    elif encoder_name == "pcb":
        encoder = PCBEncoder()
    else:
        raise ValueError(f"Unsupported ReID client encoder: {encoder_name}")

    return ReIDWrapper(
        OnlineEncoderModel(encoder),
        num_classes=num_classes,
        feat_dim=encoder.feature_dim,
    )


def _torchvision_resnet18(pretrained=False):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    except Exception as exc:
        if pretrained:
            print(f"[ReIDBackbone] Failed to load ImageNet pretrained ResNet18, fallback to random init: {exc}")
        try:
            return models.resnet18(weights=None)
        except TypeError:
            return models.resnet18(pretrained=False)


def _torchvision_resnet50(pretrained=False):
    try:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    except Exception as exc:
        if pretrained:
            print(f"[ReIDBackbone] Failed to load ImageNet pretrained ResNet50, fallback to random init: {exc}")
        try:
            return models.resnet50(weights=None)
        except TypeError:
            return models.resnet50(pretrained=False)


def _load_torchvision_resnet50_into(module, module_name):
    resnet = _torchvision_resnet50(pretrained=True)
    missing, unexpected = module.load_state_dict(resnet.state_dict(), strict=False)
    print(
        f"[ReIDBackbone] Loaded ImageNet ResNet50 weights into {module_name}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def _load_transreid_pretrain(backbone, variant):
    path = _resolve_transreid_pretrain_path(variant)
    if not path:
        print(
            f"[ReIDBackbone] No local pretrained weights found for {variant}; "
            f"set FEDMKD_{variant.upper()}_PRETRAIN or put weights under {PRETRAIN_DIR}."
        )
        return
    try:
        backbone.load_param(path)
        print(f"[ReIDBackbone] Loaded pretrained TransReID weights for {variant}: {path}")
    except Exception as exc:
        print(f"[ReIDBackbone] Failed to load TransReID pretrained weights for {variant} from {path}: {exc}")


def _resolve_transreid_pretrain_path(variant):
    env_name = f"FEDMKD_{variant.upper()}_PRETRAIN"
    env_path = os.environ.get(env_name, "")
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = {
        "transreid_vit_base": [
            "vit_base.pth",
            "vit_base_patch16_224.pth",
            "jx_vit_base_p16_224-80ecf9dd.pth",
        ],
        "transreid_deit_small": [
            "deit_small.pth",
            "deit_small_patch16_224.pth",
            "deit_small_distilled_patch16_224-649709d9.pth",
        ],
    }.get(variant, [])

    for name in candidates:
        path = os.path.join(PRETRAIN_DIR, name)
        if os.path.isfile(path):
            return path
    return None
