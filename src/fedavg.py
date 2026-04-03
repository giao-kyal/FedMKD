import copy
import torch

@torch.no_grad()
def fedavg_state_dict(state_dicts, weights=None):
    """
    state_dicts: list[OrderedDict[str, Tensor]]
    weights: list[float] sum to 1; if None use uniform
    """
    n = len(state_dicts)
    if weights is None:
        weights = [1.0 / n] * n

    out = copy.deepcopy(state_dicts[0])
    for k in out.keys():
        out[k] = out[k].float() * weights[0]
        for i in range(1, n):
            out[k] += state_dicts[i][k].float() * weights[i]
    return out