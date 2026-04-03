import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from ..pid_remap import remap_pids

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(args):
    # ====== 替换为args ======
    size_train = (args.reid_height, args.reid_width)
    size_test = (args.reid_height, args.reid_width)

    train_transforms = T.Compose([
        T.Resize(size_train, interpolation=3),
        T.RandomHorizontalFlip(p=args.reid_flip_prob),
        T.Pad(args.reid_padding),
        T.RandomCrop(size_train),
        T.ToTensor(),
        T.Normalize(mean=args.reid_pixel_mean, std=args.reid_pixel_std),
        RandomErasing(
            probability=args.reid_re_prob,
            mode='pixel',
            max_count=1,
            device='cpu'
        ),
        # RandomErasing(probability=args.reid_re_prob, mean=args.reid_pixel_mean)
    ])

    val_transforms = T.Compose([
        T.Resize(size_test),
        T.ToTensor(),
        T.Normalize(mean=args.reid_pixel_mean, std=args.reid_pixel_std),
    ])

    # ====== dataset ======
    num_workers = args.reid_num_workers

    # TransReID uses __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    # Here we use args.dataset and args.reid_root
    dataset = __factory[args.dataset](root=args.reid_root)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    # ====== train loader ======
    if 'triplet' in args.reid_sampler:
        if getattr(args, "reid_dist_train", False):
            print('DIST_TRAIN START')
            mini_batch_size = args.batch_size // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                dataset.train,
                args.batch_size,
                args.reid_num_instance
            )
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                data_sampler, mini_batch_size, True
            )
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                sampler=RandomIdentitySampler(
                    dataset.train,
                    args.batch_size,
                    args.reid_num_instance
                ),
                num_workers=num_workers,
                collate_fn=train_collate_fn
            )
    elif args.reid_sampler == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print(
            'unsupported sampler! expected softmax or triplet but got {}'.format(
                args.reid_sampler
            )
        )
        raise ValueError(f"Unsupported sampler: {args.reid_sampler}")

    # ====== val loader (query+gallery) ======
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=args.reid_test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    # ====== train loader for feature extraction (no augmentation) ======
    train_loader_normal = DataLoader(
        train_set_normal,
        batch_size=args.reid_test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    # return signature consistent with your original cfg version
    num_query = len(dataset.query)
    return train_loader, train_loader_normal, val_loader, num_query, num_classes, cam_num, view_num,dataset

def build_train_transforms(args):
    size_train = (args.reid_height, args.reid_width)
    return T.Compose([
        T.Resize(size_train, interpolation=3),
        T.RandomHorizontalFlip(p=args.reid_flip_prob),
        T.Pad(args.reid_padding),
        T.RandomCrop(size_train),
        T.ToTensor(),
        T.Normalize(mean=args.reid_pixel_mean, std=args.reid_pixel_std),
        RandomErasing(probability=args.reid_re_prob, mode='pixel', max_count=1, device='cpu'),
    ])

def build_client_train_loader(args, client_train_list, train_transforms):
    """
    client_train_list: list of (img_path, pid, camid, viewid) AFTER split_train_by_pid
    returns:
      train_loader, client_num_classes
    """
    # remap pid to 0..K-1 for this client
    client_train_list, pid_map, client_num_classes = remap_pids(client_train_list)

    train_set = ImageDataset(client_train_list, train_transforms)

    if 'triplet' in args.reid_sampler:
        loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=RandomIdentitySampler(
                client_train_list,  # 注意 sampler 应该用 remap 后的 list
                args.batch_size,
                args.reid_num_instance
            ),
            num_workers=args.reid_num_workers,
            collate_fn=train_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    elif args.reid_sampler == 'softmax':
        loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.reid_num_workers,
            collate_fn=train_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unsupported sampler: {args.reid_sampler}")

    return loader, client_num_classes,pid_map

def build_train_transforms(args):
    size_train = (args.reid_height, args.reid_width)

    return T.Compose([
        T.Resize(size_train, interpolation=3),
        T.RandomHorizontalFlip(p=args.reid_flip_prob),
        T.Pad(args.reid_padding),
        T.RandomCrop(size_train),
        T.ToTensor(),
        T.Normalize(mean=args.reid_pixel_mean, std=args.reid_pixel_std),
        RandomErasing(probability=args.reid_re_prob, mode='pixel', max_count=1, device='cpu'),
    ])