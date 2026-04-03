import numpy as np
import torch
from .metrics import R1_mAP_eval

@torch.no_grad()
def test_reid(model, val_loader, num_query, device, reranking=False, feat_norm=True):
    """
    Supports:
      - model(imgs) -> feat
      - model(imgs) -> (logits, feat)
    val_collate_fn returns:
      imgs, pids, camids, camids_batch, viewids, img_paths
    """
    model.eval()
    model.to(device)

    evaluator = R1_mAP_eval(num_query=num_query, feat_norm=feat_norm, reranking=reranking)
    evaluator.reset()

    for imgs, pids, camids, camids_batch, viewids, img_paths in val_loader:
        imgs = imgs.to(device, non_blocking=True)

        out = model(imgs)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            _, feat = out
        else:
            feat = out

        # flatten if encoder returns feature maps
        if torch.is_tensor(feat) and feat.dim() == 4:
            feat = torch.flatten(feat, 1)

        pid_batch = np.asarray(pids)
        camid_batch = camids_batch.detach().cpu().numpy()

        evaluator.update((feat.detach().cpu(), pid_batch, camid_batch))

    cmc, mAP, *_ = evaluator.compute()
    return {
        "mAP": float(mAP),
        "Rank-1": float(cmc[0]),
        "Rank-5": float(cmc[4]) if len(cmc) > 4 else None,
        "Rank-10": float(cmc[9]) if len(cmc) > 9 else None,
    }