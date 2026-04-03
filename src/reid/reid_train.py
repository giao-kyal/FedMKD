import torch
import torch.nn as nn
from torch.cuda import amp
from torch.cuda.amp import GradScaler, autocast


def train_reid_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss, total, correct = 0.0, 0, 0
    for imgs, pids, camids, viewids in loader:
        imgs = imgs.to(device, non_blocking=True)
        pids = pids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            logits, feat = model(imgs)
            loss = ce(logits, pids)
            loss.backward()
            optimizer.step()
        else:
            with amp.autocast(True):
                logits, feat = model(imgs)
                loss = ce(logits, pids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total += bs
        correct += int((logits.argmax(1) == pids).sum().item())

    return total_loss / max(total, 1), correct / max(total, 1)

def local_train_reid(cfg, model, train_loader,
                     optimizer, optimizer_center, scheduler,
                     loss_fn, center_criterion,
                     device, local_epochs=1, use_amp=True):
    model.train()
    model.to(device)

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, local_epochs + 1):
        if scheduler is not None:
            # 你用的是 timm CosineLRScheduler，按 epoch step 是 OK 的
            try:
                scheduler.step(epoch)
            except TypeError:
                scheduler.step()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device, non_blocking=True)
            target = vid.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True)
            target_view = target_view.to(device, non_blocking=True)

            with amp.autocast(enabled=use_amp):
                # 对齐 TransReID：传 target/cam_label/view_label（如果你的模型不支持，会报 TypeError）
                score, feat = model(img, target=target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)

                scaler.step(optimizer_center)
                scaler.update()

            # （可选）acc 统计
            # if isinstance(score, list):
            #     acc = (score[0].max(1)[1] == target).float().mean()
            # else:
            #     acc = (score.max(1)[1] == target).float().mean()

    return model