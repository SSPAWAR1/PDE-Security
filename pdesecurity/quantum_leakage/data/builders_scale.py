"""
Dataset builders for the scale-leakage experiment.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np"""
Dataset builders for the boundary-topology leakage experiment.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .schemas import BoundaryDataset


def default_verify_transpilation_batch(df: pd.DataFrame, label: str) -> None:
    """
    Lightweight verification summary printer.
    Assumes verif_passed, verif_tvd, verif_fidelity were added upstream.
    """
    if "verif_passed" not in df.columns:
        print(f"[{label}] No verification columns found.")
        return

    n_total = len(df)
    # Robust handling for mixed bool/NaN types
    passed_mask = df["verif_passed"] == True
    n_passed = int(passed_mask.sum())
    n_failed = n_total - n_passed

    mean_tvd = float(df["verif_tvd"].mean()) if "verif_tvd" in df.columns else float("nan")
    max_tvd = float(df["verif_tvd"].max()) if "verif_tvd" in df.columns else float("nan")
    mean_fid = float(df["verif_fidelity"].mean()) if "verif_fidelity" in df.columns else float("nan")

    print(
        f"[{label}] verification: {n_passed}/{n_total} passed | "
        f"mean TVD={mean_tvd:.4f}, max TVD={max_tvd:.4f}, mean fidelity={mean_fid:.4f}"
    )

    if n_failed > 0:
        warnings.warn(
            f"[{label}] {n_failed}/{n_total} circuits failed verification.",
            RuntimeWarning,
            stacklevel=2,
        )


def build_boundary_dataset(
    topology_family: str,
    num_qubits: int,
    n_samples_per_class: int,
    n_steps: int,
    seed: int,
    make_coupling_map: Callable[[int, str], CouplingMap],
    generate_pde_surrogate: Callable[..., QuantumCircuit],
    compile_and_extract_features: Callable[..., dict[str, float | bool]],
    verify_batch_fn: Optional[Callable[[pd.DataFrame, str], None]] = None,
    verify_features: bool = True,
    optimization_level: int = 1,
    basis_gates: list[str] | None = None,
) -> BoundaryDataset:
    """
    Build a matched Dirichlet/Periodic dataset for boundary-topology inference.
    """
    rng = np.random.default_rng(seed)
    rows = []

    cmap = make_coupling_map(num_qubits, topology_family)

    for local_sample_idx in range(n_samples_per_class):
        logical_seed = int(rng.integers(0, 10_000_000))
        transpile_seed_dir = int(rng.integers(0, 10_000_000))
        transpile_seed_per = int(rng.integers(0, 10_000_000))

        # ID Generation for precise tracking and grouped CV
        pair_id = f"{topology_family}_boundary_pair_{local_sample_idx:05d}"
        instance_id_dir = f"{pair_id}_dir"
        instance_id_per = f"{pair_id}_per"

        qc_dir = generate_pde_surrogate(
            num_qubits=num_qubits,
            boundary_condition="dirichlet",
            n_steps=n_steps,
            seed=logical_seed,
        )
        qc_per = generate_pde_surrogate(
            num_qubits=num_qubits,
            boundary_condition="periodic",
            n_steps=n_steps,
            seed=logical_seed,
        )

        feat_dir = compile_and_extract_features(
            qc=qc_dir,
            coupling_map=cmap,
            seed=transpile_seed_dir,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            verify=verify_features,
        )
        feat_per = compile_and_extract_features(
            qc=qc_per,
            coupling_map=cmap,
            seed=transpile_seed_per,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            verify=verify_features,
        )

        # Unpack dictionaries to prevent mutating the upstream return objects
        row_dir = {
            **feat_dir,
            "task": "boundary",
            "label": 0,
            "label_name": "dirichlet",
            "boundary": "dirichlet",
            "topology_family": topology_family,
            "pair_id": pair_id,
            "instance_id": instance_id_dir,
            "local_sample_idx": local_sample_idx,
            "logical_seed": logical_seed,
            "transpile_seed": transpile_seed_dir,
            "num_qubits": num_qubits,
            "n_steps": n_steps,
        }
        
        row_per = {
            **feat_per,
            "task": "boundary",
            "label": 1,
            "label_name": "periodic",
            "boundary": "periodic",
            "topology_family": topology_family,
            "pair_id": pair_id,
            "instance_id": instance_id_per,
            "local_sample_idx": local_sample_idx,
            "logical_seed": logical_seed,
            "transpile_seed": transpile_seed_per,
            "num_qubits": num_qubits,
            "n_steps": n_steps,
        }

        rows.extend([row_dir, row_per])

    df = pd.DataFrame(rows)

    if verify_batch_fn is None:
        verify_batch_fn = default_verify_transpilation_batch
    verify_batch_fn(df, f"boundary/{topology_family}")

    return BoundaryDataset(df=df)
import pandas as pd

from .schemas import ScaleDataset


def build_scale_dataset(
    resolutions: List[int],
    n_samples_per_res: int,
    topology_families: List[str],
    template_families: List[str],
    n_steps: int,
    seed: int,
    make_coupling_map: Callable,
    generate_scale_surrogate: Callable,
    compile_and_extract_features: Callable,
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
                    )

                    feats.update({
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
                    })
                    rows.append(feats)

    return ScaleDataset(df=pd.DataFrame(rows))
