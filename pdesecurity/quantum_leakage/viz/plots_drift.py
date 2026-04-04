"""
Drift-curve and correlation plotting helpers.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_drift_curves(
    results_df: pd.DataFrame,
    task_name: str,
    topology_family: str,
    metric: str,
    feature_groups: Dict[str, List[str]],
    outpath: str,
    model_name: str | None = None,
) -> None:
    """
    Plot performance-vs-drift curves by feature group.

    Parameters
    ----------
    model_name
        If provided, restrict plot to one model.
    """
    sub = results_df[
        (results_df["task"] == task_name)
        & (results_df["topology_family"] == topology_family)
    ].copy()

    if model_name is not None and "model" in sub.columns:
        sub = sub[sub["model"] == model_name].copy()

    plt.figure(figsize=(8, 5))

    for group_name in feature_groups.keys():
        g = sub[sub["feature_group"] == group_name].sort_values("drift")
        if g.empty:
            continue

        plt.plot(g["drift"], g[f"{metric}_mean"], marker="o", label=group_name)
        plt.fill_between(
            g["drift"],
            g[f"{metric}_mean"] - g[f"{metric}_std"],
            g[f"{metric}_mean"] + g[f"{metric}_std"],
            alpha=0.15,
        )

    plt.xlabel("Hardware-drift severity")
    plt.ylabel(metric.replace("_", " ").title())
    title = f"{topology_family}: {task_name} | {metric}"
    if model_name is not None:
        title += f" | {model_name}"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    outpath: str,
    method: str = "spearman",
) -> None:
    """
    Plot a correlation heatmap over selected columns.
    """
    corr = df[columns].corr(method=method)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
