"""
Configuration for Experiment 3: Drift ablation and stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Exp3Config:
    random_seed: int = 42

    # Boundary dataset
    boundary_num_qubits: int = 8
    boundary_samples_per_class: int = 120

    # Scale dataset
    scale_resolutions: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 16])
    scale_samples_per_res: int = 80

    # Shared generation
    n_steps: int = 3
    n_repeats: int = 10

    # Drift
    drift_levels: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Topologies / families
    topology_families: List[str] = field(default_factory=lambda: [
        "line",
        "gridish",
    ])
    template_families: List[str] = field(default_factory=lambda: [
        "A",
        "B",
    ])

    # Ordinal labels
    ordered_scale_labels: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 16])

    # Feature groups
    feature_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "Topological": [
            "swap_equiv",
            "swap_fraction",
            "extra_twoq",
        ],
        "Complexity": [
            "routed_depth",
            "depth_overhead",
            "twoq_overhead",
            "extra_depth",
            "cx_fraction",
        ],
        "Operational": [
            "transpile_ms",
            "sched_duration_ms",
            "idle_variance",
        ],
        "All": [
            "swap_equiv",
            "swap_fraction",
            "extra_twoq",
            "routed_depth",
            "depth_overhead",
            "twoq_overhead",
            "extra_depth",
            "cx_fraction",
            "transpile_ms",
            "sched_duration_ms",
            "idle_variance",
        ],
    })

    # Output naming
    output_prefix: str = "exp3_drift_ablation"
