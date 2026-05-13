"""
main.py — Entry point for running all experiments.

Usage:
    python src/main.py --data_dir /path/to/spacenet7/train
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

from config import CFG
from dataset import prepare_data
from trainer import run_experiment
from visualize import plot_results, save_results_csv


def parse_args():
    parser = argparse.ArgumentParser(description='Point-Supervised Building Segmentation')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to SpaceNet-7 train directory')
    parser.add_argument('--epochs', type=int, default=CFG['epochs'],
                        help='Number of training epochs per run')
    parser.add_argument('--skip_exp1', action='store_true',
                        help='Skip Experiment 1 (sampling ratio ablation)')
    parser.add_argument('--skip_exp2', action='store_true',
                        help='Skip Experiment 2 (strategy comparison)')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])

    print('Python    :', sys.version.split()[0])
    print('PyTorch   :', torch.__version__)
    print('CUDA avail:', torch.cuda.is_available())
    print('GPU count :', torch.cuda.device_count())

    train_dir = Path(args.data_dir)
    train_imgs, val_imgs, train_masks, val_masks = prepare_data(train_dir)

    exp1_results = {}
    exp2_results = {}
    EXP1_RATIOS  = [0.005, 0.01, 0.02, 0.05]

    # ── Experiment 1: Label Sampling Ratio ──────────────────────────────
    if not args.skip_exp1:
        print('\n' + '='*50)
        print('EXPERIMENT 1: Effect of Label Sampling Ratio')
        print('='*50)
        for r in EXP1_RATIOS:
            name = f'ratio_{int(r * 1000):04d}'
            hist, best_iou = run_experiment(
                run_name=name,
                train_imgs=train_imgs, val_imgs=val_imgs,
                train_masks=train_masks, val_masks=val_masks,
                strategy='boundary', ratio=r,
                epochs=args.epochs,
            )
            exp1_results[r] = {'history': hist, 'best_iou': best_iou}

        plot_results(exp1_results, 'Exp1: Sampling Ratio',
                     {r: f'{r*100:.1f}%' for r in EXP1_RATIOS})

    # ── Experiment 2: Sampling Strategy ─────────────────────────────────
    if not args.skip_exp2:
        print('\n' + '='*50)
        print('EXPERIMENT 2: Random vs Boundary-Aware Sampling')
        print('='*50)
        for strat in ['random', 'boundary']:
            name = f'strategy_{strat}'
            hist, best_iou = run_experiment(
                run_name=name,
                train_imgs=train_imgs, val_imgs=val_imgs,
                train_masks=train_masks, val_masks=val_masks,
                strategy=strat, ratio=0.01,
                epochs=args.epochs,
            )
            exp2_results[strat] = {'history': hist, 'best_iou': best_iou}

        plot_results(exp2_results, 'Exp2: Sampling Strategy',
                     {'random': 'Random', 'boundary': 'Boundary-Aware'})

    # ── Save combined results ────────────────────────────────────────────
    if exp1_results or exp2_results:
        save_results_csv(exp1_results, exp2_results, EXP1_RATIOS)


if __name__ == '__main__':
    main()
