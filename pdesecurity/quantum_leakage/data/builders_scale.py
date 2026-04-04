"""
Dataset builders for the scale-leakage experiment.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .schemas import ScaleDataset

# Assuming you place the default_verify_transpilation_batch in a shared utils file, 
# or you can paste it at the top of this file as you did with the boundary builder.
from .utils import default_verify_transpilation_batch 


def build_scale_dataset(
    resolutions: list[int],
    n_samples_per_res: int,
    topology_families: list[str],
    template_families: list[str],
    n_steps: int,
    seed: int,
    make_coupling_map: Callable[[int, str], CouplingMap],
    generate_scale_surrogate: Callable[..., QuantumCircuit],
    compile_and_extract_features: Callable[..., dict[str, float | bool]],
    verify_batch_fn: Optional[Callable[[pd.DataFrame, str], None]] = None,
    verify_features: bool = True,
    optimization_level: int = 1,
    basis_gates: list[str] | None = None,
) -> ScaleDataset:
    """
    Build a scale-leakage dataset over multiple resolutions, template families,
    and topology families.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for topology_family in topology_families:
        for template_family in template_families:
            for N in resolutions:
                cmap = make_coupling_map(N, topology_family)

                for local_sample_idx in range(n_samples_per_res):
                    logical_seed = int(rng.integers(0, 10_000_000))
                    transpile_seed = int(rng.integers(0, 10_000_000))

                    instance_id = (
                        f"scale_{topology_family}_{template_family}_N{N}_"
                        f"{local_sample_idx:05d}"
                    )

                    qc = generate_scale_surrogate(
                        num_qubits=N,
                        template_family=template_family,
                        n_steps=n_steps,
                        seed=logical_seed,
                    )

                    feats = compile_and_extract_features(
                        qc=qc,
                        coupling_map=cmap,
                        seed=transpile_seed,
                        basis_gates=basis_gates,
                        optimization_level=optimization_level,
                        verify=verify_features,
                    )

                    # Unpack dictionaries to prevent mutating the upstream return objects
                    row = {
                        **feats,
                        "task": "scale",
                        "label": N,
                        "label_name": str(N),
                        "N": N,
                        "instance_id": instance_id,
                        "local_sample_idx": local_sample_idx,
                        "template_family": template_family,
                        "topology_family": topology_family,
                        "logical_seed": logical_seed,
                        "transpile_seed": transpile_seed,
                        "n_steps": n_steps,
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)

    # Add batch verification to catch scale-specific routing failures
    if verify_batch_fn is None:
        verify_batch_fn = default_verify_transpilation_batch
    
    # We can pass a generalized label, or run the verification inside the loops 
    # to label it by topology_family, but grouping it at the end is usually fine.
    verify_batch_fn(df, "scale_dataset_all_topologies")

    return ScaleDataset(df=df)
