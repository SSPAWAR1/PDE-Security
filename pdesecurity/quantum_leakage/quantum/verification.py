"""
Transpilation verification helpers.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def strip_non_unitary_ops(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Remove measurements, resets, and barriers so the circuit can be
    compared using statevector simulation.
    """
    stripped = QuantumCircuit(qc.num_qubits)
    for inst, qargs, cargs in qc.data:
        if inst.name not in {"measure", "reset", "barrier"}:
            stripped.append(inst, qargs, cargs)
    return stripped


def verify_transpilation(
    original: QuantumCircuit,
    transpiled: QuantumCircuit,
    tvd_threshold: float = 0.05,
) -> Dict[str, float | bool]:
    """
    Verify that transpilation preserved circuit semantics.

    Returns
    -------
    dict with:
      - tvd
      - fidelity
      - passed
    """
    sv_orig = Statevector.from_instruction(strip_non_unitary_ops(original))
    sv_trans = Statevector.from_instruction(strip_non_unitary_ops(transpiled))

    probs_orig = sv_orig.probabilities()
    probs_trans = sv_trans.probabilities()

    tvd = float(0.5 * np.sum(np.abs(probs_orig - probs_trans)))
    fidelity = float(state_fidelity(sv_orig, sv_trans))
    passed = bool(tvd < tvd_threshold)

    if not passed:
        warnings.warn(
            f"[verify_transpilation] TVD={tvd:.4f} exceeds threshold "
            f"{tvd_threshold:.4f}. Fidelity={fidelity:.4f}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "tvd": tvd,
        "fidelity": fidelity,
        "passed": passed,
    }
