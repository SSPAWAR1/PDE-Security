"""
PDE-inspired surrogate circuits for boundary-topology experiments.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit

from .topologies import grid_index, infer_grid_shape


def unique_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Deduplicate undirected edges while preserving order.
    """
    seen = set()
    out = []
    for a, b in edges:
        e = tuple(sorted((a, b)))
        if a != b and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def build_stencil_partitions(rows: int, cols: int) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build checkerboard-style local stencil partitions.
    """
    parts = {
        "h_even": [],
        "h_odd": [],
        "v_even": [],
        "v_odd": [],
    }

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


def build_periodic_wrap_edges(rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    Periodic wrap edges for the grid.
    """
    wraps = []

    if cols > 2:
        for r in range(rows):
            a = grid_index(r, 0, cols)
            b = grid_index(r, cols - 1, cols)
            wraps.append((a, b))

    if rows > 2:
        for c in range(cols):
            a = grid_index(0, c, cols)
            b = grid_index(rows - 1, c, cols)
            wraps.append((a, b))

    return unique_edges(wraps)


def apply_random_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """
    Apply random local one-qubit rotations.
    """
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * np.pi), q)
        qc.rx(rng.uniform(0, 2 * np.pi), q)


def apply_coupling_block(
    qc: QuantumCircuit,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    """
    Apply a PDE-like pairwise coupling block.
    """
    for a, b in edges:
        theta = rng.uniform(0.15, 1.25) * np.pi
        qc.cx(a, b)
        qc.rz(theta, b)
        qc.cx(a, b)


def choose_step_edges(
    rows: int,
    cols: int,
    boundary_condition: str,
    step_idx: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Choose structured stencil edges for one logical step.

    - Dirichlet uses only local edges.
    - Periodic replaces a subset with wrap-around edges.
    """
    partitions = build_stencil_partitions(rows, cols)
    wrap_edges = build_periodic_wrap_edges(rows, cols)

    part_names = ["h_even", "h_odd", "v_even", "v_odd"]

    primary_name = part_names[step_idx % len(part_names)]
    primary_edges = list(partitions[primary_name])

    secondary_name = part_names[(step_idx + 1) % len(part_names)]
    secondary_pool = list(partitions[secondary_name])
    rng.shuffle(secondary_pool)

    n_secondary = 0
    if len(secondary_pool) > 0:
        n_secondary = rng.integers(0, max(1, len(secondary_pool) // 2) + 1)

    base_edges = unique_edges(primary_edges + secondary_pool[:n_secondary])
    target_count = len(base_edges)

    if boundary_condition == "dirichlet" or len(wrap_edges) == 0:
        return base_edges

    if boundary_condition != "periodic":
        raise ValueError(f"Unknown boundary_condition={boundary_condition}")

    edges = list(base_edges)
    rng.shuffle(edges)

    replace_count = min(len(wrap_edges), max(1, int(round(0.25 * max(target_count, 1)))))
    replace_count = min(replace_count, len(edges))

    kept_edges = edges[replace_count:]

    wraps = list(wrap_edges)
    rng.shuffle(wraps)
    selected_wraps = wraps[:replace_count]

    periodic_edges = unique_edges(kept_edges + selected_wraps)

    if len(periodic_edges) > target_count:
        periodic_edges = periodic_edges[:target_count]

    if len(periodic_edges) < target_count:
        pool = [e for e in edges if e not in periodic_edges]
        rng.shuffle(pool)
        need = target_count - len(periodic_edges)
        periodic_edges.extend(pool[:need])
        periodic_edges = unique_edges(periodic_edges)

    return periodic_edges[:target_count]


def generate_pde_surrogate(
    num_qubits: int,
    boundary_condition: str,
    n_steps: int,
    seed: int,
) -> QuantumCircuit:
    """
    Generate a PDE-inspired surrogate circuit.

    Intended for matched-pair boundary experiments where the systematic
    difference is the hidden boundary condition.
    """
    if num_qubits < 4:
        raise ValueError("num_qubits should be at least 4")

    rows, cols = infer_grid_shape(num_qubits)
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    apply_random_local_layer(qc, rng)

    for step_idx in range(n_steps):
        apply_random_local_layer(qc, rng)
        edges = choose_step_edges(rows, cols, boundary_condition, step_idx, rng)
        apply_coupling_block(qc, edges, rng)

        for q in range(num_qubits):
            qc.rz(rng.uniform(-0.25, 0.25) * np.pi, q)

    apply_random_local_layer(qc, rng)
    return qc
