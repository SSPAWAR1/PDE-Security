"""
Helpers for operational-feature augmentation and hardware-drift simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def augment_operational_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Derive operational/provider-facing surrogate features from compiled artefacts.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    sched = (
        6.0 * out["routed_depth"].values
        + 8.0 * out["extra_twoq"].values
        + rng.normal(0.0, 10.0, len(out))
    )
    sched = np.maximum(1.0, sched)

    idle = (
        0.015 * out["routed_depth"].values
        + 0.10 * out["swap_equiv"].values
        + 0.06 * out["extra_depth"].values
        + rng.normal(0.0, 0.6, len(out))
    )
    idle = np.maximum(0.0, idle)

    out["sched_duration_ms"] = sched
    out["idle_variance"] = idle
    return out


def apply_hardware_drift(
    df: pd.DataFrame,
    severity: float,
    topology_family: str,
    seed: int,
) -> pd.DataFrame:
    """
    Apply drift to primitive / near-primitive observables first,
    then recompute derived ratios.

    Notes
    -----
    This assumes the following columns are present:
    - swap_equiv
    - extra_twoq
    - routed_depth
    - extra_depth
    - transpile_ms
    - logical_depth
    - logical_twoq
    """
    if severity == 0.0:
        return df.copy()

    rng = np.random.default_rng(seed)
    out = df.copy()

    topo_gain = 1.10 if topology_family == "line" else 1.00

    route_load = out["extra_twoq"].values / max(float(out["extra_twoq"].max()), 1.0)
    depth_load = out["routed_depth"].values / max(float(out["routed_depth"].max()), 1.0)
    transpile_load = out["transpile_ms"].values / max(float(out["transpile_ms"].max()), 1.0)

    # Drift primitive / near-primitive observables
    out["swap_equiv"] = np.maximum(
        0.0,
        out["swap_equiv"].values
        * (
            1.0
            + topo_gain * severity * (0.18 * route_load)
            + rng.normal(0.0, 0.05 * severity, len(out))
        ),
    )

    out["extra_twoq"] = np.maximum(
        0.0,
        out["extra_twoq"].values
        * (
            1.0
            + topo_gain * severity * (0.20 * route_load)
            + rng.normal(0.0, 0.06 * severity, len(out))
        ),
    )

    out["routed_depth"] = np.maximum(
        1.0,
        out["routed_depth"].values
        * (
            1.0
            + topo_gain * severity * (0.10 * route_load + 0.08 * depth_load)
            + rng.normal(0.0, 0.04 * severity, len(out))
        ),
    )

    out["extra_depth"] = np.maximum(
        0.0,
        out["extra_depth"].values
        * (
            1.0
            + topo_gain * severity * (0.15 * depth_load)
            + rng.normal(0.0, 0.06 * severity, len(out))
        ),
    )

    out["transpile_ms"] = np.maximum(
        1.0,
        out["transpile_ms"].values
        * (
            1.0
            + topo_gain * severity * (0.20 + 0.10 * transpile_load + 0.10 * route_load)
            + rng.normal(0.0, 0.10 * severity, len(out))
        ),
    )

    if "sched_duration_ms" in out.columns:
        out["sched_duration_ms"] = np.maximum(
            1.0,
            out["sched_duration_ms"].values
            * (
                1.0
                + topo_gain * severity * (0.18 + 0.10 * depth_load)
                + rng.normal(0.0, 0.08 * severity, len(out))
            ),
        )

    if "idle_variance" in out.columns:
        out["idle_variance"] = np.maximum(
            0.0,
            out["idle_variance"].values
            + topo_gain * severity * (0.10 * route_load + 0.08 * depth_load)
            + rng.normal(0.0, 0.10 * severity, len(out)),
        )

    # Recompute derived features
    logical_depth = np.maximum(out["logical_depth"].values, 1.0)
    logical_twoq = np.maximum(out["logical_twoq"].values, 1.0)

    out["depth_overhead"] = out["routed_depth"].values / logical_depth
    out["twoq_overhead"] = (out["extra_twoq"].values + out["logical_twoq"].values) / logical_twoq

    # Approximate recomputation for ratios
    routed_twoq = out["extra_twoq"].values + out["logical_twoq"].values
    out["swap_fraction"] = np.clip(out["swap_equiv"].values / np.maximum(routed_twoq, 1.0), 0.0, 1.0)

    if "logical_total_ops" in out.columns:
        approx_total_ops = np.maximum(
            out["logical_total_ops"].values + out["extra_twoq"].values + out["extra_depth"].values,
            1.0,
        )
        out["cx_fraction"] = np.clip(routed_twoq / approx_total_ops, 0.0, 1.0)

    return out
