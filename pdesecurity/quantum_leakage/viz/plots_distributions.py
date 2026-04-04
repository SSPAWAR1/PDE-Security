"""
Feature-distribution plotting helpers.
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_distributions(
    df: pd.DataFrame,
    feature: str,
    label_col: str,
    label_names: Dict[int, str],
    title: str,
    outpath: str,
    bins: int = 20,
) -> None:
    """
    Plot one feature's class-conditional distributions.
    """
    plt.figure(figsize=(8, 5))

    for label, name in label_names.items():
        vals = df[df[label_col] == label][feature].values
        plt.hist(vals, bins=bins, alpha=0.6, density=True, label=name)

    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
