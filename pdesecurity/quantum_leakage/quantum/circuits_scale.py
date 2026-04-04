"""
Scale-leakage surrogate circuits with template-family variation.

These are PDE-inspired, stencil-structured prototypes designed to isolate
the compilation footprint of scaling operations, rather than fully 
equivalent Hamiltonian simulations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from .topologies import grid_index, infer_grid_shape

TemplateFamily = Literal["A", "B"]


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


def apply_local_layer(
    qc: QuantumCircuit, 
    rng: np.random.Generator, 
    strength: float = 1.0
) -> None:
    """
    Apply random local rotations with tunable strength to prevent 
    washing out the higher-order coupling structure.
    """
    for q in range(qc.num_qubits):
        qc.rz(strength * rng.uniform(0, 2 * np.pi), q)
        if rng.random() < 0.7:
            qc.rx(strength * rng.uniform(0, 2 * np.pi), q)


def build_partitions(rows: int, cols: int) -> dict[str, list[tuple[int, int]]]:
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
    template_family: TemplateFamily,
    step_idx: int,
    rng: np.random.Generator,
    density: float = 1.0,
) -> list[tuple[int, int]]:
    """
    Choose family-specific edge schedules using controlled stochasticity.
    This provides a statistical signature without a perfectly rigid codebook.
    """
    parts = build_partitions(rows, cols)
    names = ["h_even", "h_odd", "v_even", "v_odd"]

    if template_family == "A":
        primary_weights = np.array([0.40, 0.25, 0.20, 0.15])
        sec_frac = 0.35
    elif template_family == "B":
        primary_weights = np.array([0.15, 0.20, 0.25, 0.40])
        sec_frac = 0.50
    else:
        raise ValueError(f"Unknown template_family={template_family}")

    # Rotate weights over time so there is temporal structure but not a rigid mapping
    primary_weights = np.roll(primary_weights, step_idx % 4)
    primary_idx = rng.choice(len(names), p=primary_weights / primary_weights.sum())
    primary = names[primary_idx]

    secondary_candidates = [n for n in names if n != primary]
    secondary = secondary_candidates[rng.integers(len(secondary_candidates))]

    primary_edges = list(parts[primary])
    secondary_pool = list(parts[secondary])
    rng.shuffle(primary_edges)
    rng.shuffle(secondary_pool)

    n_primary = max(1, int(round(density * len(primary_edges))))
    n_secondary = int(round(density * sec_frac * len(secondary_pool)))

    chosen = unique_edges(primary_edges[:n_primary] + secondary_pool[:n_secondary])
    return chosen


def apply_coupling_block(
    qc: QuantumCircuit,
    edges: list[tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    """
    Weighted two-qubit coupling surrogate utilizing a small library 
    of equivalent-ish decomposable motifs to ensure structural heterogeneity.
    """
    for a, b in edges:
        theta = rng.uniform(0.15, 1.25) * np.pi
        mode = rng.integers(3)

        if mode == 0:
            qc.cx(a, b)
            qc.rz(theta, b)
            qc.cx(a, b)
        elif mode == 1:
            qc.cx(b, a)
            qc.rx(theta, a)
            qc.cx(b, a)
        else:
            qc.rzz(theta, a, b)


def generate_scale_surrogate(
    num_qubits: int,
    template_family: TemplateFamily,
    scale_level: int,
    seed: int,
) -> QuantumCircuit:
    """
    Generate a scale-surrogate circuit where hidden scientific scale is
    operationalised through density, interaction depth, and grid resolution.
    """
    # 1. Input Validation
    if num_qubits <= 1:
        raise ValueError("num_qubits must be at least 2")
    if scale_level < 0:
        raise ValueError("scale_level must be non-negative")
    if template_family not in {"A", "B"}:
        raise ValueError("template_family must be one of {'A', 'B'}")

    rows, cols = infer_grid_shape(num_qubits)
    if rows * cols != num_qubits:
        raise ValueError(
            f"infer_grid_shape({num_qubits}) returned ({rows}, {cols}), "
            "which does not match num_qubits. Ensure num_qubits is factorable."
        )

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    # 2. Derive scale-dependent parameters
    n_steps = 2 + scale_level
    base_density = min(1.0, 0.3 + 0.15 * scale_level)

    # 3. Apply strong initial state preparation
    apply_local_layer(qc, rng, strength=1.0)

    # 4. Evolution steps with progressive density scaling
    for step_idx in range(n_steps):
        # Weaker local mixing inside the loop preserves the structural signature
        apply_local_layer(qc, rng, strength=0.35)
        
        density = min(1.0, base_density + 0.05 * step_idx)
        edges = choose_scale_edges(
            rows=rows, 
            cols=cols, 
            template_family=template_family, 
            step_idx=step_idx, 
            rng=rng, 
            density=density
        )
        
        apply_coupling_block(qc, edges, rng)

        for q in range(num_qubits):
            qc.rz(rng.uniform(-0.25, 0.25) * np.pi, q)

    # 5. Final local mixing
    apply_local_layer(qc, rng, strength=1.0)
    
    return qc
