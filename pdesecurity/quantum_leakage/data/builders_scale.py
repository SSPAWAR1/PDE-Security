"""
Dataset builders for the scale-leakage experiment.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
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
