import os
import sys
import ast
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from reid.datasets.make_dataloader import make_dataloader
from reid.reid_test import test_reid
from reid.backbones.hetero import build_reid_client_model


def read_client_plan_from_log(run_log_path):
    if not os.path.exists(run_log_path):
        return None
    with open(run_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '[ReID] Client encoder plan:' in line:
                try:
                    part = line.split(':', 1)[1].strip()
                    plan = ast.literal_eval(part)
                    return plan
                except Exception:
                    return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help='Path to run directory containing run.log and saved_models')
    parser.add_argument('--task_id', required=True, help='task_id (folder name under saved_models)')
    parser.add_argument('--dataset', default=None, help='reid dataset name, e.g. market1501')
    parser.add_argument('--reid_root', default=None, help='root path for reid datasets')
    parser.add_argument('--num_clients', type=int, default=None, help='number of clients to evaluate (default: all)')
    args = parser.parse_args()

    # Build a lightweight namespace instead of importing src/parse.py,
    # which would parse argv too early and reject eval-only flags.
    global_args = argparse.Namespace(
        dataset=args.dataset or 'market1501',
        reid_root=args.reid_root or os.path.join(PROJECT_ROOT, 'data'),
        batch_size=64,
        reid_height=256,
        reid_width=128,
        reid_pixel_mean=[0.485, 0.456, 0.406],
        reid_pixel_std=[0.229, 0.224, 0.225],
        reid_num_workers=0,
        reid_test_batch_size=256,
        reid_sampler='softmax_triplet',
        reid_dist_train=False,
        reid_rerank=False,
    )

    # build dataloader
    _, _, val_loader, num_query, _, _, _, _ = make_dataloader(global_args)

    run_log = os.path.join(args.run_dir, 'run.log')
    plan = read_client_plan_from_log(run_log)
    if plan is None:
        print('Warning: cannot read client encoder plan from run.log. Will assume homogeneous clients.')

    # load checkpoint
    saved_models_dir = os.path.join(args.run_dir, 'saved_models', args.task_id)
    ckpt_path = os.path.join(saved_models_dir, 'checkpoint.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(saved_models_dir, 'best_checkpoint.pth')
    if not os.path.exists(ckpt_path):
        print(f'No checkpoint found at {saved_models_dir}. Exiting.')
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    client_states = ckpt.get('clients', [])
    if not client_states:
        print('Checkpoint contains no client states. Exiting.')
        return

    total_clients = len(client_states)
    eval_n = args.num_clients if args.num_clients is not None else total_clients
    eval_n = min(eval_n, total_clients)

    print(f'Evaluating {eval_n}/{total_clients} clients from checkpoint: {ckpt_path}')

    results = []
    for i in range(eval_n):
        enc_name = plan[i] if plan and i < len(plan) else None

        state_dict = client_states[i]
        classifier_weight = None
        for key in ("classifier.weight", "module.classifier.weight"):
            if key in state_dict:
                classifier_weight = state_dict[key]
                break
        if classifier_weight is None:
            raise RuntimeError(f'Cannot infer num_classes for client {i}; classifier weight missing.')

        num_classes = int(classifier_weight.shape[0])
        client_model = build_reid_client_model(enc_name or 'resnet18', num_classes=num_classes)

        # load state (use strict=False to be tolerant of minor mismatches)
        try:
            client_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f'Warning: failed to load state for client {i} strictly: {e}')

        client_model.eval()
        metrics = test_reid(
            client_model,
            val_loader,
            num_query=num_query,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            reranking=getattr(global_args, 'reid_rerank', False),
        )
        print(f'[Client Eval][{i}] {metrics}')
        results.append(metrics)

    # print summary
    for i, m in enumerate(results):
        print(f'Client {i}: Rank-1={m.get("Rank-1")}, mAP={m.get("mAP")}')


if __name__ == '__main__':
    main()
