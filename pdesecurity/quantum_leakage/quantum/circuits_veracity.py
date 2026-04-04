"""
Circuits for accuracy / veracity leakage experiments.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .topologies import grid_index, infer_grid_shape


def apply_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """
    Apply random local rotations.
    """
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * np.pi), q)
        qc.rx(rng.uniform(0, 2 * np.pi), q)


def accuracy_to_trotter_steps(accuracy: float) -> int:
    """
    Map target accuracy to a surrogate Trotter step count.

    Rough story:
        error ~ 1 / n^2
        so n ~ sqrt(1 / error)
    """
    base_steps = int(np.sqrt(1.0 / accuracy))
    return max(2, base_steps)


def generate_time_evolution_circuit(
    n_qubits: int,
    accuracy: float,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """
    Generate a time-evolution-style circuit with accuracy-dependent depth.
    """
    qc = QuantumCircuit(n_qubits)
    n_steps = accuracy_to_trotter_steps(accuracy)

    rows, cols = infer_grid_shape(n_qubits)

    for _ in range(n_steps):
        apply_local_layer(qc, rng)

        for r in range(rows):
            for c in range(cols - 1):
                a = grid_index(r, c, cols)
                b = grid_index(r, c + 1, cols)
                angle = rng.uniform(0, np.pi / 4)
                qc.cx(a, b)
                qc.rz(angle, b)
                qc.cx(a, b)

        for r in range(rows - 1):
            for c in range(cols):
                a = grid_index(r, c, cols)
                b = grid_index(r + 1, c, cols)
                angle = rng.uniform(0, np.pi / 4)
                qc.cx(a, b)
                qc.rz(angle, b)
                qc.cx(a, b)

    return qc


def generate_optimization_circuit(
    n_qubits: int,
    accuracy: float,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """
    Generate a QAOA-like optimization circuit with accuracy-dependent layers.
    """
    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.h(q)

    n_layers = accuracy_to_trotter_steps(accuracy)
    rows, cols = infer_grid_shape(n_qubits)

    for _ in range(n_layers):
        for r in range(rows):
            for c in range(cols - 1):
                a = grid_index(r, c, cols)
                b = grid_index(r, c + 1, cols)
                gamma = rng.uniform(0, np.pi / 2)
                qc.cx(a, b)
                qc.rz(gamma, b)
                qc.cx(a, b)

        for q in range(n_qubits):
            beta = rng.uniform(0, np.pi / 2)
            qc.rx(beta, q)

    return qc
