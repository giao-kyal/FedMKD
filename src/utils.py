import torch
import os
import transform
from model import SimSiam, MoCo


def get_transformation(model):
    """
    model: str or class
      - if str: 'simsiam' / 'moco' / 'simclr' / 'byol' / ...
      - if class: SimSiam / MoCo / ...
    """
    # normalize string
    if isinstance(model, str):
        key = model.lower()
        if key == "simsiam":
            return transform.SimSiamTransform
        if key in ("moco", "moco_v2"):
            return transform.MoCoTransform
        # byol/simclr/fedet/... -> SimCLRTransform (two-view)
        return transform.SimCLRTransform

    # keep backward-compat for class inputs
    if model == SimSiam:
        return transform.SimSiamTransform
    if model == MoCo:
        return transform.MoCoTransform
    return transform.SimCLRTransform


def calculate_model_distance(m1, m2):
    distance, count = 0, 0
    d1, d2 = m1.state_dict(), m2.state_dict()
    for name, param in m1.named_parameters():
        if 'conv' in name and 'weight' in name:
            distance += torch.dist(d1[name].detach().clone().view(1, -1), d2[name].detach().clone().view(1, -1), 2)
            count += 1
    return distance / count


def normalize(arr):
    maxx = max(arr)
    minn = min(arr)
    diff = maxx - minn
    if diff == 0:
        return arr
    return [(x - minn) / diff for x in arr]

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)