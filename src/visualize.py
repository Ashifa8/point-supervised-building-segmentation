import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from config import CFG


def plot_results(results: dict, title: str, label_map: dict):
    """Plot learning curves and best mIoU bar chart for an experiment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for k, v in results.items():
        axes[0].plot(v['history']['val_iou'], label=label_map[k])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val mIoU')
    axes[0].set_title(title + ' - Learning Curves')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    labels    = [label_map[k] for k in results]
    best_ious = [v['best_iou'] for v in results.values()]
    bars = axes[1].bar(labels, best_ious, color='steelblue', edgecolor='black')
    for bar, val in zip(bars, best_ious):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    axes[1].set_ylabel('Best Val mIoU')
    axes[1].set_title(title + ' - Best mIoU')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = CFG['save_dir'] / f'{title.replace(" ", "_").lower()}.png'
    plt.savefig(save_path, dpi=120)
    print(f'Saved plot: {save_path}')
    plt.show()


def save_results_csv(exp1_results: dict, exp2_results: dict,
                     exp1_ratios: list, out_path: Path = None):
    """Save a combined results summary CSV."""
    rows = []
    for r, v in exp1_results.items():
        rows.append({'Experiment': 'Ratio',
                     'Config'    : f'{r*100:.1f}%',
                     'Best mIoU' : round(v['best_iou'], 4)})
    for s, v in exp2_results.items():
        rows.append({'Experiment': 'Strategy',
                     'Config'    : s.capitalize(),
                     'Best mIoU' : round(v['best_iou'], 4)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_path = out_path or CFG['save_dir'] / 'results_summary.csv'
    df.to_csv(out_path, index=False)
    print(f'Saved results: {out_path}')
    return df
