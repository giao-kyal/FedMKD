import torch
import torch.nn as nn
import torch.nn.functional as F

class ReIDWrapper(nn.Module):
    """
    One wrapper to rule them all:
    - forward(img) -> (score, feat) for supervised ReID local training
    - .online_encoder for FedMKD server distillation/align code
    """
    def __init__(self, base_model: nn.Module, num_classes: int, feat_dim: int = 2048, feat_norm: bool = False):
        super().__init__()
        self.base_model = base_model
        self.feat_dim = feat_dim
        self.feat_norm = feat_norm

        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

    @property
    def online_encoder(self):
        # support both base_model.online_encoder and base_model.byol.online_encoder
        if hasattr(self.base_model, "online_encoder"):
            return self.base_model.online_encoder
        if hasattr(self.base_model, "byol") and hasattr(self.base_model.byol, "online_encoder"):
            return self.base_model.byol.online_encoder
        raise AttributeError("base_model has no online_encoder (nor base_model.byol.online_encoder).")

    @property
    def target_encoder(self):
        if hasattr(self.base_model, "target_encoder"):
            return self.base_model.target_encoder
        if hasattr(self.base_model, "byol") and hasattr(self.base_model.byol, "target_encoder"):
            return self.base_model.byol.target_encoder
        return None

    @target_encoder.setter
    def target_encoder(self, value):
        if hasattr(self.base_model, "byol") and hasattr(self.base_model.byol, "target_encoder"):
            self.base_model.byol.target_encoder = value
        else:
            self.base_model.target_encoder = value

    def extract_feat(self, img):
        feat = self.online_encoder(img)
        if feat.dim() == 4:
            feat = torch.flatten(feat, 1)
        feat = self.bnneck(feat)
        if self.feat_norm:
            feat = F.normalize(feat, dim=1)
        return feat

    def forward(self, img, target=None, cam_label=None, view_label=None, **kwargs):
        feat = self.extract_feat(img)
        score = self.classifier(feat)
        return score, feat
