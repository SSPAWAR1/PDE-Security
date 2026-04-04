"""
Dataset builders for binary and ordinal veracity / accuracy leakage experiments.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .schemas import VeracityDataset

# Assuming you place the default_verify_transpilation_batch in a shared utils file
from .utils import default_verify_transpilation_batch


def build_binary_veracity_dataset(
    topology_name: str,
    coupling_map: CouplingMap,
    workload_families: list[str],
    n_samples_per_class: int,
    seed: int,
    num_qubits: int,
    generate_time_evolution_circuit: Callable[[int, float, np.random.Generator], QuantumCircuit],
    generate_optimization_circuit: Callable[[int, float, np.random.Generator], QuantumCircuit],
    compile_and_extract_features: Callable[..., dict[str, float | bool]],
    low_accuracy_value: float = 1e-2,
    high_accuracy_value: float = 1e-4,
    verify_batch_fn: Optional[Callable[[pd.DataFrame, str], None]] = None,
    verify_features: bool = True,
    optimization_level: int = 1,
    basis_gates: list[str] | None = None,
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
                    basis_gates=basis_gates,
                    optimization_level=optimization_level,
                    verify=verify_features,
                )

                # Unpack dictionaries to prevent mutating the upstream return objects
                row = {
                    **features,
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
                }

                rows.append(row)

    df = pd.DataFrame(rows)

    if verify_batch_fn is None:
        verify_batch_fn = default_verify_transpilation_batch
    verify_batch_fn(df, f"veracity_binary/{topology_name}")

    return VeracityDataset(df=df, task_type="binary")


def build_ordinal_veracity_dataset(
    topology_name: str,
    coupling_map: CouplingMap,
    workload_families: list[str],
    accuracy_levels: list[float],
    n_samples_per_level: int,
    seed: int,
    num_qubits: int,
    generate_time_evolution_circuit: Callable[[int, float, np.random.Generator], QuantumCircuit],
    generate_optimization_circuit: Callable[[int, float, np.random.Generator], QuantumCircuit],
    compile_and_extract_features: Callable[..., dict[str, float | bool]],
    verify_batch_fn: Optional[Callable[[pd.DataFrame, str], None]] = None,
    verify_features: bool = True,
    optimization_level: int = 1,
    basis_gates: list[str] | None = None,
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
                    basis_gates=basis_gates,
                    optimization_level=optimization_level,
                    verify=verify_features,
                )

                # Unpack dictionaries to prevent mutating the upstream return objects
                row = {
                    **features,
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
                }

                rows.append(row)

    df = pd.DataFrame(rows)

    if verify_batch_fn is None:
        verify_batch_fn = default_verify_transpilation_batch
    verify_batch_fn(df, f"veracity_ordinal/{topology_name}")

    return VeracityDataset(df=df, task_type="ordinal")
