"""
PDE-inspired surrogate circuits for boundary-topology experiments.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from .topologies import grid_index, infer_grid_shape


def unique_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
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


def build_stencil_partitions(rows: int, cols: int) -> dict[str, list[tuple[int, int]]]:
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


def build_periodic_wrap_edges(rows: int, cols: int) -> list[tuple[int, int]]:
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


def apply_random_local_layer(qc: QuantumCircuit, param_rng: np.random.Generator) -> None:
    """
    Apply random local one-qubit rotations.
    """
    for q in range(qc.num_qubits):
        qc.rz(param_rng.uniform(0, 2 * np.pi), q)
        qc.rx(param_rng.uniform(0, 2 * np.pi), q)


def apply_coupling_block(
    qc: QuantumCircuit,
    edges: list[tuple[int, int]],
    param_rng: np.random.Generator,
) -> None:
    """
    Apply a PDE-like pairwise coupling block.
    """
    for a, b in edges:
        theta = param_rng.uniform(0.15, 1.25) * np.pi
        qc.cx(a, b)
        qc.rz(theta, b)
        qc.cx(a, b)


def choose_step_edges(
    rows: int,
    cols: int,
    boundary_condition: Literal["dirichlet", "periodic"],
    step_idx: int,
    structure_seed: int,
) -> list[tuple[int, int]]:
    """
    Choose structured stencil edges for one logical step.
    
    Uses a strictly isolated structural RNG to ensure edge count and sequence
    does not desynchronize the main parameter RNG.
    """
    structure_rng = np.random.default_rng(structure_seed)
    
    partitions = build_stencil_partitions(rows, cols)
    wrap_edges = build_periodic_wrap_edges(rows, cols)

    part_names = ["h_even", "h_odd", "v_even", "v_odd"]

    primary_name = part_names[step_idx % len(part_names)]
    primary_edges = list(partitions[primary_name])

    secondary_name = part_names[(step_idx + 1) % len(part_names)]
    secondary_pool = list(partitions[secondary_name])
    structure_rng.shuffle(secondary_pool)

    n_secondary = 0
    if len(secondary_pool) > 0:
        n_secondary = structure_rng.integers(0, max(1, len(secondary_pool) // 2) + 1)

    base_edges = unique_edges(primary_edges + secondary_pool[:n_secondary])
    target_count = len(base_edges)

    if boundary_condition == "dirichlet" or len(wrap_edges) == 0:
        return base_edges

    if boundary_condition != "periodic":
        raise ValueError(f"Unknown boundary_condition={boundary_condition}")

    # --- Periodic Wrap Logic ---
    edges = list(base_edges)
    structure_rng.shuffle(edges)

    replace_count = min(len(wrap_edges), max(1, int(round(0.25 * max(target_count, 1)))))
    replace_count = min(replace_count, len(edges))

    kept_edges = edges[replace_count:]

    wraps = list(wrap_edges)
    structure_rng.shuffle(wraps)
    selected_wraps = wraps[:replace_count]

    periodic_edges = unique_edges(kept_edges + selected_wraps)

    # Strictly enforce target_count so the ML model cannot cheat by counting gates
    if len(periodic_edges) > target_count:
        periodic_edges = periodic_edges[:target_count]

    if len(periodic_edges) < target_count:
        # Optimized list-membership filtering using sets
        periodic_edge_set = {tuple(sorted(e)) for e in periodic_edges}
        pool = [e for e in base_edges if tuple(sorted(e)) not in periodic_edge_set]
        
        structure_rng.shuffle(pool)
        need = target_count - len(periodic_edges)
        periodic_edges.extend(pool[:need])

    final_edges = unique_edges(periodic_edges)
    if len(final_edges) != target_count:
        raise RuntimeError(
            f"Edge count mismatch in periodic pair matching: expected {target_count}, got {len(final_edges)}."
        )
    
    return final_edges


def generate_pde_surrogate(
    num_qubits: int,
    boundary_condition: Literal["dirichlet", "periodic"],
    n_steps: int,
    seed: int,
) -> QuantumCircuit:
    """
    Generate a PDE-inspired surrogate circuit.

    Intended for matched-pair boundary experiments. Structural random choices
    and parameter random choices are rigidly isolated to guarantee that 
    the bulk volume of matched pairs remains perfectly identical.
    """
    if num_qubits < 4:
        raise ValueError("num_qubits should be at least 4")

    rows, cols = infer_grid_shape(num_qubits)
    
    # param_rng strictly dictates gate parameters (thetas).
    param_rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    apply_random_local_layer(qc, param_rng)

    for step_idx in range(n_steps):
        apply_random_local_layer(qc, param_rng)
        
        # structure_seed strictly dictates topology shapes.
        # Advancing it cleanly prevents boundary logic from desyncing param_rng.
        structure_seed = (seed * 10007 + step_idx * 1009 + 777) % (2**32)
        
        edges = choose_step_edges(
            rows=rows, 
            cols=cols, 
            boundary_condition=boundary_condition, 
            step_idx=step_idx, 
            structure_seed=structure_seed
        )
        
        # Because len(edges) is perfectly matched between Dirichlet/Periodic,
        # param_rng advances by the exact same amount for both circuit variants.
        apply_coupling_block(qc, edges, param_rng)

        for q in range(num_qubits):
            qc.rz(param_rng.uniform(-0.25, 0.25) * np.pi, q)

    apply_random_local_layer(qc, param_rng)
    return qc
