"""
Drift evaluation and result summarisation helpers.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .classifiers import make_models
from .ordinal import ordinal_metrics


def evaluate_task_under_drift(
    df_clean: pd.DataFrame,
    task_name: str,
    topology_family: str,
    label_col: str,
    feature_groups: Dict[str, List[str]],
    drift_levels: List[float],
    drift_fn: Callable[..., pd.DataFrame],
    n_repeats: int = 10,
    seed: int = 42,
    ordered_scale_labels: List[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluate model performance under increasing drift.

    Notes:
    - Models are fit once per repeat and feature group on the clean train split.
    - Drift is applied only to the test split.
    """
    rows = []

    for rep in range(n_repeats):
        rep_seed = seed + 1000 * rep
        y = df_clean[label_col].values

        train_idx, test_idx = train_test_split(
            np.arange(len(df_clean)),
            test_size=0.30,
            stratify=y,
            random_state=rep_seed,
        )

        train_clean = df_clean.iloc[train_idx].copy()
        test_clean = df_clean.iloc[test_idx].copy()

        for feature_group, cols in feature_groups.items():
            models = make_models(rep_seed)

            for model_name, model in models.items():
                clf = clone(model)
                clf.fit(train_clean[cols], train_clean[label_col].values)

                for drift in drift_levels:
                    test_drift = drift_fn(
                        test_clean,
                        severity=drift,
                        topology_family=topology_family,
                        seed=rep_seed + int(100 * drift) + 17,
                    )

                    y_pred = clf.predict(test_drift[cols])

                    if task_name == "scale":
                        if ordered_scale_labels is None:
                            raise ValueError("ordered_scale_labels is required for scale task.")
                        mets = ordinal_metrics(
                            test_drift[label_col].values,
                            y_pred,
                            ordered_scale_labels,
                        )
                        rows.append({
                            "task": task_name,
                            "topology_family": topology_family,
                            "repeat": rep,
                            "drift": drift,
                            "feature_group": feature_group,
                            "model": model_name,
                            **mets,
                        })
                    else:
                        rows.append({
                            "task": task_name,
                            "topology_family": topology_family,
                            "repeat": rep,
                            "drift": drift,
                            "feature_group": feature_group,
                            "model": model_name,
                            "macro_f1": float(f1_score(test_drift[label_col].values, y_pred, average="macro")),
                            "bal_acc": float(balanced_accuracy_score(test_drift[label_col].values, y_pred)),
                        })

    return pd.DataFrame(rows)


def summarise_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise mean and std over repeats.
    """
    group_cols = ["task", "topology_family", "feature_group", "model", "drift"]
    metric_cols = [c for c in raw_df.columns if c not in group_cols + ["repeat"]]

    grouped = raw_df.groupby(group_cols, as_index=False)

    summary_rows = []
    for _, g in grouped:
        row = {col: g[col].iloc[0] for col in group_cols}
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
