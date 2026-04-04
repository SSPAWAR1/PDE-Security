"""
Helpers for operational-feature augmentation and hardware-drift simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ==============================
# Operational feature augmentation
# ==============================

def augment_operational_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Derive execution-side surrogate features from compiled artefacts.

    These are not semantic features — they represent scheduler / runtime behaviour.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    sched = (
        6.0 * out["routed_depth"].values
        + 8.0 * out["extra_twoq"].values
        + rng.normal(0.0, 10.0, len(out))
    )
    out["sched_duration_ms"] = np.maximum(1.0, sched)

    idle = (
        0.015 * out["routed_depth"].values
        + 0.10 * out["swap_equiv"].values
        + 0.06 * out["extra_depth"].values
        + rng.normal(0.0, 0.6, len(out))
    )
    out["idle_variance"] = np.maximum(0.0, idle)

    return out


# ==============================
# Hardware drift model
# ==============================

def apply_hardware_drift(
    df: pd.DataFrame,
    severity: float,
    topology_family: str,
    seed: int,
) -> pd.DataFrame:
    """
    Apply physically consistent drift to compiled observables.

    Design principles
    -----------------
    - Drift acts on *compiled observables*, not logical labels.
    - Drift magnitude depends on routing / depth burden.
    - Derived features are ALWAYS recomputed from primitives.
    """

    if severity == 0.0:
        return df.copy()

    rng = np.random.default_rng(seed)
    out = df.copy()

    topo_gain = 1.10 if topology_family == "line" else 1.00

    # ------------------------------
    # Normalised workload loads
    # ------------------------------
    route_load = out["extra_twoq"].values / max(out["extra_twoq"].max(), 1.0)
    depth_load = out["routed_depth"].values / max(out["routed_depth"].max(), 1.0)
    transpile_load = out["transpile_ms"].values / max(out["transpile_ms"].max(), 1.0)

    # ------------------------------
    # Drift compiled primitives
    # ------------------------------

    # Reconstruct routed_twoq if not stored
    routed_twoq = out.get(
        "routed_twoq",
        out["logical_twoq"].values + out["extra_twoq"].values
    )

    routed_twoq = np.maximum(
        1.0,
        routed_twoq * (
            1.0
            + topo_gain * severity * (0.20 * route_load)
            + rng.normal(0.0, 0.06 * severity, len(out))
        ),
    )

    routed_depth = np.maximum(
        1.0,
        out["routed_depth"].values * (
            1.0
            + topo_gain * severity * (0.10 * route_load + 0.08 * depth_load)
            + rng.normal(0.0, 0.04 * severity, len(out))
        ),
    )

    transpile_ms = np.maximum(
        1.0,
        out["transpile_ms"].values * (
            1.0
            + topo_gain * severity * (0.20 + 0.10 * transpile_load + 0.10 * route_load)
            + rng.normal(0.0, 0.10 * severity, len(out))
        ),
    )

    # ------------------------------
    # Recompute derived quantities
    # ------------------------------

    logical_twoq = np.maximum(out["logical_twoq"].values, 1.0)
    logical_depth = np.maximum(out["logical_depth"].values, 1.0)

    extra_twoq = np.maximum(0.0, routed_twoq - logical_twoq)
    extra_depth = np.maximum(0.0, routed_depth - logical_depth)

    swap_equiv = extra_twoq / 3.0

    # If routed_total_ops not present → approximate
    if "routed_total_ops" in out.columns:
        routed_total_ops = np.maximum(out["routed_total_ops"].values, 1.0)
    else:
        routed_total_ops = np.maximum(
            out["logical_total_ops"].values + extra_twoq + extra_depth,
            1.0,
        )

    # ------------------------------
    # Write back primitives
    # ------------------------------

    out["routed_twoq"] = routed_twoq
    out["routed_depth"] = routed_depth
    out["transpile_ms"] = transpile_ms
    out["extra_twoq"] = extra_twoq
    out["extra_depth"] = extra_depth
    out["swap_equiv"] = swap_equiv

    # ------------------------------
    # Recompute ALL ratios (clean)
    # ------------------------------

    out["depth_overhead"] = routed_depth / logical_depth
    out["twoq_overhead"] = routed_twoq / logical_twoq

    out["swap_fraction"] = np.clip(
        swap_equiv / np.maximum(routed_twoq, 1.0),
        0.0,
        1.0,
    )

    out["cx_fraction"] = np.clip(
        routed_twoq / routed_total_ops,
        0.0,
        1.0,
    )

    # ------------------------------
    # Operational drift (optional)
    # ------------------------------

    if "sched_duration_ms" in out.columns:
        out["sched_duration_ms"] = np.maximum(
            1.0,
            out["sched_duration_ms"].values * (
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

    return out
