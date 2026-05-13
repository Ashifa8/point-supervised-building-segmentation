import torch
import torch.nn as nn
import numpy as np

from config import CFG


class PartialCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss that only computes gradients on labeled pixels.
    Pixels marked with ignore_index (255) contribute zero gradient.
    """

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred   : (B, C, H, W)
        # target : (B, H, W)  — 255 for unlabelled pixels
        loss_map     = self.ce(pred, target)
        labeled_mask = (target != self.ignore_index).float()

        n_labeled = labeled_mask.sum()
        if n_labeled == 0:
            return loss_map.sum() * 0.0

        return (loss_map * labeled_mask).sum() / n_labeled


def compute_iou(pred_logits: torch.Tensor, gt_mask: torch.Tensor,
                num_classes: int = 2, ignore_index: int = 255) -> float:
    """Compute mean IoU over all classes, ignoring unlabelled pixels."""
    preds = pred_logits.argmax(dim=1)
    valid = (gt_mask != ignore_index)
    preds = preds[valid].cpu().numpy()
    gt    = gt_mask[valid].cpu().numpy()

    iou_per_class = []
    for c in range(num_classes):
        tp    = ((preds == c) & (gt == c)).sum()
        fp    = ((preds == c) & (gt != c)).sum()
        fn    = ((preds != c) & (gt == c)).sum()
        denom = tp + fp + fn
        iou_per_class.append(tp / denom if denom > 0 else 0.0)

    return float(np.mean(iou_per_class))
