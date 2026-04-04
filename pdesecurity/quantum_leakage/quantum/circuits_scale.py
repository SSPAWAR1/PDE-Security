"""
Scale-leakage surrogate circuits with template-family variation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit

from .topologies import grid_index, infer_grid_shape


def unique_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out = []
    for a, b in edges:
        e = tuple(sorted((a, b)))
        if a != b and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def apply_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """
    Apply random local rotations.
    """
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * np.pi), q)
        qc.rx(rng.uniform(0, 2 * np.pi), q)


def build_partitions(rows: int, cols: int) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build grid partitions used by template families.
    """
    parts = {"h_even": [], "h_odd": [], "v_even": [], "v_odd": []}

    for r in range(rows):
        for c in range(cols - 1):
            a = grid_index(r, c, cols)
            b = grid_index(r, c + 1, cols)
            if c % 2 == 0:
                parts["h_even"].append((a, b))
            else:
                parts["h_odd"].append((a, b))

    for r in range(rows - 1):
        for c in range(cols):
            a = grid_index(r, c, cols)
            b = grid_index(r + 1, c, cols)
            if r % 2 == 0:
                parts["v_even"].append((a, b))
            else:
                parts["v_odd"].append((a, b))

    return parts


def choose_scale_edges(
    rows: int,
    cols: int,
    template_family: str,
    step_idx: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Choose family-specific edge schedules for scale experiments.
    """
    parts = build_partitions(rows, cols)
    names = ["h_even", "h_odd", "v_even", "v_odd"]

    if template_family == "A":
        primary = names[step_idx % 4]
        secondary = names[(step_idx + 1) % 4]
        sec_frac = 0.35
    elif template_family == "B":
        primary = names[(2 * step_idx) % 4]
        secondary = names[(2 * step_idx + 3) % 4]
        sec_frac = 0.50
    else:
        raise ValueError(f"Unknown template_family={template_family}")

    primary_edges = list(parts[primary])
    secondary_pool = list(parts[secondary])
    rng.shuffle(primary_edges)
    rng.shuffle(secondary_pool)

    n_secondary = int(round(sec_frac * len(secondary_pool)))
    chosen = unique_edges(primary_edges + secondary_pool[:n_secondary])
    return chosen


def apply_coupling_block(
    qc: QuantumCircuit,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    """
    Weighted two-qubit coupling surrogate.
    """
    for a, b in edges:
        theta = rng.uniform(0.15, 1.25) * np.pi
        qc.cx(a, b)
        qc.rz(theta, b)
        qc.cx(a, b)


def generate_scale_surrogate(
    num_qubits: int,
    template_family: str,
    n_steps: int,
    seed: int,
) -> QuantumCircuit:
    """
    Generate a scale-surrogate circuit where hidden scientific scale is
    operationalised through circuit size / resolution.
    """
    rng = np.random.default_rng(seed)
    rows, cols = infer_grid_shape(num_qubits)
    qc = QuantumCircuit(num_qubits)

    apply_local_layer(qc, rng)

    for step_idx in range(n_steps):
        apply_local_layer(qc, rng)
        edges = choose_scale_edges(rows, cols, template_family, step_idx, rng)
        apply_coupling_block(qc, edges, rng)

        for q in range(num_qubits):
            qc.rz(rng.uniform(-0.25, 0.25) * np.pi, q)

    apply_local_layer(qc, rng)
    return qc
