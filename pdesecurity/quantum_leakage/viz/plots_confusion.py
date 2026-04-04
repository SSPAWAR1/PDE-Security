"""
Confusion-matrix plotting helpers.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int] | List[float] | List[str],
    title: str,
    outpath: str,
    normalize: bool = True,
) -> None:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    normalize
        If True, row-normalize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=[str(l) for l in labels],
        yticklabels=[str(l) for l in labels],
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
