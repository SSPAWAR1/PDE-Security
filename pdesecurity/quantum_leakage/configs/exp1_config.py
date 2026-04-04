"""
Configuration for Experiment 1: Boundary-topology leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Exp1Config:
    random_seed: int = 42

    # Dataset
    num_qubits: int = 8
    n_samples_per_class: int = 180
    n_steps: int = 3

    # MI / stats
    n_mi_boot: int = 300
    n_mi_perm: int = 500
    n_group_cv_splits: int = 5
    n_pair_perm: int = 5000

    # Topologies
    topology_families: List[str] = field(default_factory=lambda: [
        "line",
        "ladder",
    ])

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

    # Output naming
    output_prefix: str = "exp1_boundary_topology"
