"""
Ordinal classification metrics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ordered_labels: List[int] | List[float],
) -> Dict[str, float]:
    """
    Compute macro-F1, balanced accuracy, class MAE, and adjacent-class accuracy.
    """
    label_to_idx = {lab: i for i, lab in enumerate(ordered_labels)}

    true_idx = np.array([label_to_idx[v] for v in y_true], dtype=int)
    pred_idx = np.array([label_to_idx[v] for v in y_pred], dtype=int)

    class_mae = float(np.mean(np.abs(true_idx - pred_idx)))
    adjacent_acc = float(np.mean(np.abs(true_idx - pred_idx) <= 1))

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "class_mae": class_mae,
        "adjacent_acc": adjacent_acc,
    }
