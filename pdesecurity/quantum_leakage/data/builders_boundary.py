"""
Dataset builders for the boundary-topology leakage experiment.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

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
    n_passed = int(df["verif_passed"].sum())
    n_failed = n_total - n_passed

    mean_tvd = float(df["verif_tvd"].mean()) if "verif_tvd" in df.columns else float("nan")
    max_tvd = float(df["verif_tvd"].max()) if "verif_tvd" in df.columns else float("nan")
    mean_fid = float(df["verif_fidelity"].mean()) if "verif_fidelity" in df.columns else float("nan")

    print(
        f"[{label}] verification: {n_passed}/{n_total} passed | "
        f"mean TVD={mean_tvd:.4f}, max TVD={max_tvd:.4f}, mean fidelity={mean_fid:.4f}"
    )

    if n_failed > 0:
        import warnings
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
    make_coupling_map: Callable,
    generate_pde_surrogate: Callable,
    compile_and_extract_features: Callable,
    verify_batch_fn: Optional[Callable[[pd.DataFrame, str], None]] = None,
) -> BoundaryDataset:
    """
    Build a matched Dirichlet/Periodic dataset for boundary-topology inference.

    Parameters
    ----------
    topology_family
        Name of backend topology family, e.g. "line", "ladder".
    num_qubits
        Number of logical qubits.
    n_samples_per_class
        Number of matched logical seeds to generate.
    n_steps
        Number of logical evolution / stencil steps.
    seed
        Base RNG seed.
    make_coupling_map
        Function that constructs a coupling map from (num_qubits, topology_family)
        or a topology helper compatible with your project.
    generate_pde_surrogate
        Function that builds logical circuits for boundary experiments.
    compile_and_extract_features
        Canonical feature extractor.
    verify_batch_fn
        Optional batch verification summary function.
    """
    rng = np.random.default_rng(seed)
    rows = []

    cmap = make_coupling_map(num_qubits, topology_family)

    for local_sample_idx in range(n_samples_per_class):
        logical_seed = int(rng.integers(0, 10_000_000))
        transpile_seed_dir = int(rng.integers(0, 10_000_000))
        transpile_seed_per = int(rng.integers(0, 10_000_000))

        pair_id = f"{topology_family}_boundary_pair_{local_sample_idx:05d}"

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

        feat_dir = compile_and_extract_features(qc_dir, cmap, transpile_seed_dir)
        feat_per = compile_and_extract_features(qc_per, cmap, transpile_seed_per)

        feat_dir.update({
            "task": "boundary",
            "label": 0,
            "label_name": "dirichlet",
            "boundary": "dirichlet",
            "topology_family": topology_family,
            "pair_id": pair_id,
            "local_sample_idx": local_sample_idx,
            "logical_seed": logical_seed,
            "transpile_seed": transpile_seed_dir,
            "num_qubits": num_qubits,
            "n_steps": n_steps,
        })
        feat_per.update({
            "task": "boundary",
            "label": 1,
            "label_name": "periodic",
            "boundary": "periodic",
            "topology_family": topology_family,
            "pair_id": pair_id,
            "local_sample_idx": local_sample_idx,
            "logical_seed": logical_seed,
            "transpile_seed": transpile_seed_per,
            "num_qubits": num_qubits,
            "n_steps": n_steps,
        })

        rows.extend([feat_dir, feat_per])

    df = pd.DataFrame(rows)

    if verify_batch_fn is None:
        verify_batch_fn = default_verify_transpilation_batch
    verify_batch_fn(df, f"boundary/{topology_family}")

    return BoundaryDataset(df=df)
