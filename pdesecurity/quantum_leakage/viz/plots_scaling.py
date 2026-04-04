"""
Scaling-law plotting helpers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_loglog_scaling(
    df: pd.DataFrame,
    topology_family: str,
    feature: str,
    family_stats: Dict[str, Tuple[float, float, float]],
    family_col: str,
    topology_col: str,
    outpath: str,
) -> None:
    """
    Plot log-log scaling for a feature across template families.

    Parameters
    ----------
    family_stats
        Mapping family -> (mean exponent, lo, hi)
    """
    plt.figure(figsize=(8, 6))

    colors = {
        "A": "tab:blue",
        "B": "tab:red",
    }

    families = sorted(df[family_col].unique())
    for fam in families:
        sub = df[
            (df[topology_col] == topology_family)
            & (df[family_col] == fam)
        ].copy()

        if sub.empty:
            continue

        x = np.log(sub["N"].values)
        y = np.log(sub[feature].values + 1e-8)

        k, lo, hi = family_stats[fam]
        color = colors.get(fam, None)

        sns.regplot(
            x=x,
            y=y,
            scatter_kws={"alpha": 0.30, "color": color},
            line_kws={
                "color": color,
                "label": f"Family {fam}: k={k:.2f} [{lo:.2f}, {hi:.2f}]",
            },
        )

    plt.xlabel("log(Resolution N)")
    plt.ylabel(f"log({feature})")
    plt.title(f"{topology_family}: scaling of {feature}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
