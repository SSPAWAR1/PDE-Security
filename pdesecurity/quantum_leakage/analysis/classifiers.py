"""
Classifier model builders and evaluation helpers.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ordinal import ordinal_metrics


def make_models(seed: int = 42) -> Dict[str, object]:
    """
    Canonical model registry for experiments.
    """
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=seed)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
        ),
    }


def grouped_cv_classifier_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Group-aware CV for non-ordinal classification.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = make_models(seed)

    rows = []
    for model_name, model in models.items():
        f1s = []
        bals = []

        for train_idx, test_idx in cv.split(X, y, groups):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            clf = clone(model)
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)

            f1s.append(f1_score(yte, yp, average="macro"))
            bals.append(balanced_accuracy_score(yte, yp))

        rows.append({
            "model": model_name,
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "bal_acc_mean": float(np.mean(bals)),
            "bal_acc_std": float(np.std(bals)),
        })

    return pd.DataFrame(rows)


def family_holdout_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    family_col: str,
    label_col: str,
    task_type: str,
    seed: int = 42,
    ordered_labels: list[int] | list[float] | None = None,
) -> pd.DataFrame:
    """
    Train on one family and test on another.
    """
    families = sorted(df[family_col].unique())
    models = make_models(seed)
    rows = []

    for train_fam in families:
        for test_fam in families:
            if train_fam == test_fam:
                continue

            train_df = df[df[family_col] == train_fam].copy()
            test_df = df[df[family_col] == test_fam].copy()

            Xtr = train_df[feature_cols].to_numpy(dtype=float)
            ytr = train_df[label_col].to_numpy()
            Xte = test_df[feature_cols].to_numpy(dtype=float)
            yte = test_df[label_col].to_numpy()

            for model_name, model in models.items():
                clf = clone(model)
                clf.fit(Xtr, ytr)
                yp = clf.predict(Xte)

                if task_type == "ordinal":
                    if ordered_labels is None:
                        raise ValueError("ordered_labels must be provided for ordinal evaluation.")
                    metrics = ordinal_metrics(yte, yp, ordered_labels)
                    rows.append({
                        "train_family": train_fam,
                        "test_family": test_fam,
                        "model": model_name,
                        **metrics,
                    })
                else:
                    rows.append({
                        "train_family": train_fam,
                        "test_family": test_fam,
                        "model": model_name,
                        "macro_f1": float(f1_score(yte, yp, average="macro")),
                        "bal_acc": float(balanced_accuracy_score(yte, yp)),
                    })

    return pd.DataFrame(rows)
