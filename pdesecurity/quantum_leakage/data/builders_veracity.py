"""
Dataset builders for binary and ordinal veracity / accuracy leakage experiments.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd

from .schemas import VeracityDataset


def build_binary_veracity_dataset(
    topology_name: str,
    coupling_map,
    workload_families: List[str],
    n_samples_per_class: int,
    seed: int,
    num_qubits: int,
    generate_time_evolution_circuit: Callable,
    generate_optimization_circuit: Callable,
    compile_and_extract_features: Callable,
    low_accuracy_value: float = 1e-2,
    high_accuracy_value: float = 1e-4,
) -> VeracityDataset:
    """
    Build dataset for binary accuracy/veracity classification:
    low-accuracy vs high-accuracy.
    """
    rng = np.random.default_rng(seed)
    rows = []

    accuracy_specs = [
        ("low", 0, low_accuracy_value),
        ("high", 1, high_accuracy_value),
    ]

    for accuracy_level, label, accuracy_value in accuracy_specs:
        for family in workload_families:
            for local_sample_idx in range(n_samples_per_class):
                logical_seed = int(rng.integers(0, 2**30))
                transpile_seed = int(rng.integers(0, 2**30))
                sample_rng = np.random.default_rng(logical_seed)

                instance_id = (
                    f"veracity_binary_{topology_name}_{family}_{accuracy_level}_"
                    f"{local_sample_idx:05d}"
                )

                if family == "time_evolution":
                    qc = generate_time_evolution_circuit(
                        num_qubits, accuracy_value, sample_rng
                    )
                elif family == "optimization":
                    qc = generate_optimization_circuit(
                        num_qubits, accuracy_value, sample_rng
                    )
                else:
                    raise ValueError(f"Unknown workload family: {family}")

                features = compile_and_extract_features(
                    qc=qc,
                    coupling_map=coupling_map,
                    seed=transpile_seed,
                )

                features.update({
                    "task": "veracity_binary",
                    "task_type": "binary",
                    "label": label,
                    "label_name": accuracy_level,
                    "accuracy_level": accuracy_level,
                    "accuracy_value": accuracy_value,
                    "workload_family": family,
                    "topology": topology_name,
                    "instance_id": instance_id,
                    "local_sample_idx": local_sample_idx,
                    "logical_seed": logical_seed,
                    "transpile_seed": transpile_seed,
                    "num_qubits": num_qubits,
                })

                rows.append(features)

    df = pd.DataFrame(rows)
    return VeracityDataset(df=df, task_type="binary")


def build_ordinal_veracity_dataset(
    topology_name: str,
    coupling_map,
    workload_families: List[str],
    accuracy_levels: List[float],
    n_samples_per_level: int,
    seed: int,
    num_qubits: int,
    generate_time_evolution_circuit: Callable,
    generate_optimization_circuit: Callable,
    compile_and_extract_features: Callable,
) -> VeracityDataset:
    """
    Build dataset for ordinal accuracy/veracity classification:
    multiple ordered accuracy levels.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for label, accuracy in enumerate(accuracy_levels):
        for family in workload_families:
            for local_sample_idx in range(n_samples_per_level):
                logical_seed = int(rng.integers(0, 2**30))
                transpile_seed = int(rng.integers(0, 2**30))
                sample_rng = np.random.default_rng(logical_seed)

                instance_id = (
                    f"veracity_ordinal_{topology_name}_{family}_{accuracy:.0e}_"
                    f"{local_sample_idx:05d}"
                )

                if family == "time_evolution":
                    qc = generate_time_evolution_circuit(
                        num_qubits, accuracy, sample_rng
                    )
                elif family == "optimization":
                    qc = generate_optimization_circuit(
                        num_qubits, accuracy, sample_rng
                    )
                else:
                    raise ValueError(f"Unknown workload family: {family}")

                features = compile_and_extract_features(
                    qc=qc,
                    coupling_map=coupling_map,
                    seed=transpile_seed,
                )

                features.update({
                    "task": "veracity_ordinal",
                    "task_type": "ordinal",
                    "label": label,
                    "label_name": f"{accuracy:.0e}",
                    "accuracy": accuracy,
                    "workload_family": family,
                    "topology": topology_name,
                    "instance_id": instance_id,
                    "local_sample_idx": local_sample_idx,
                    "logical_seed": logical_seed,
                    "transpile_seed": transpile_seed,
                    "num_qubits": num_qubits,
                })

                rows.append(features)

    df = pd.DataFrame(rows)
    return VeracityDataset(df=df, task_type="ordinal")
