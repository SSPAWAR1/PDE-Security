"""
Configuration for Experiment 4: Veracity / accuracy leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Exp4Config:
    random_seed: int = 42

    # Shared
    num_qubits: int = 8

    # Binary task
    binary_accuracies: List[str] = field(default_factory=lambda: ["low", "high"])
    n_binary_samples_per_class: int = 180
    low_accuracy_value: float = 1e-2
    high_accuracy_value: float = 1e-4

    # Ordinal task
    ordinal_accuracies: List[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4, 1e-5])
    n_ordinal_samples_per_level: int = 100

    # Families / topologies
    workload_families: List[str] = field(default_factory=lambda: [
        "time_evolution",
        "optimization",
    ])
    topology_families: List[str] = field(default_factory=lambda: [
        "line",
        "gridish",
        "ladder",
    ])

    # MI / stats
    n_mi_boot: int = 300
    n_mi_perm: int = 500
    n_group_cv_splits: int = 5

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

    # Output naming
    output_prefix: str = "exp4_veracity_leakage"
