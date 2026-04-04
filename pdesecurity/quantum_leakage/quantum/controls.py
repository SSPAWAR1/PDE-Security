"""
Control-circuit generators.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def apply_random_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """
    Random local single-qubit layer.
    """
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * np.pi), q)
        qc.rx(rng.uniform(0, 2 * np.pi), q)


def generate_random_control(
    num_qubits: int,
    n_steps: int,
    logical_twoq_target: int,
    seed: int,
) -> QuantumCircuit:
    """
    Generate a random control circuit with approximately matched two-qubit volume,
    but without meaningful topology-linked structure.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    total_twoq_added = 0
    while total_twoq_added < logical_twoq_target:
        apply_random_local_layer(qc, rng)

        block_size = min(num_qubits, logical_twoq_target - total_twoq_added)
        used_pairs = set()

        for _ in range(block_size):
            q1, q2 = sorted(rng.choice(num_qubits, size=2, replace=False))
            if (q1, q2) in used_pairs:
                continue
            used_pairs.add((q1, q2))

            qc.cx(q1, q2)
            total_twoq_added += 1

            if total_twoq_added >= logical_twoq_target:
                break

        apply_random_local_layer(qc, rng)

    return qc
