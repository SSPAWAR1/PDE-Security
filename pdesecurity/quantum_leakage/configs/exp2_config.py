"""
Configuration for Experiment 2: Scale leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Exp2Config:
    random_seed: int = 42

    # Dataset
    resolutions: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 16])
    n_samples_per_res: int = 120
    n_steps: int = 3

    # Topologies / families
    topology_families: List[str] = field(default_factory=lambda: [
        "line",
        "gridish",
    ])
    template_families: List[str] = field(default_factory=lambda: [
        "A",
        "B",
    ])

    # MI / stats
    n_mi_boot: int = 300
    n_mi_perm: int = 500

    # Features
    feature_columns: List[str] = field(default_factory=lambda: [
        "swap_equiv",
        "swap_fraction",
        "cx_fraction",
        "routed_depth",
        "depth_overhead",
        "twoq_overhead",
        "extra_twoq",
        "extra_depth",
    ])

    logical_footprint_columns: List[str] = field(default_factory=lambda: [
        "logical_depth",
        "logical_twoq",
        "logical_total_ops",
    ])

    scaling_features: List[str] = field(default_factory=lambda: [
        "routed_depth",
        "extra_twoq",
    ])

    # Output naming
    output_prefix: str = "exp2_scale_leakage"
