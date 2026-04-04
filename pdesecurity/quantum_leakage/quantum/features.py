"""
Provider-visible feature extraction from transpiled circuits.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .transpilation import transpile_circuit
from .verification import verify_transpilation


def count_two_qubit_ops(qc: QuantumCircuit) -> int:
    """
    Count two-qubit operations in a circuit.
    """
    return sum(1 for inst, qargs, _ in qc.data if len(qargs) == 2)


def count_total_ops(qc: QuantumCircuit) -> int:
    """
    Count total operations excluding barriers.
    """
    return sum(1 for inst, _, _ in qc.data if inst.name not in {"barrier"})


def compile_and_extract_features(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    seed: int,
    basis_gates: Optional[list[str]] = None,
    optimization_level: int = 1,
    layout_method: str = "sabre",
    routing_method: str = "sabre",
    verify: bool = True,
    tvd_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compile a circuit and extract provider-visible metadata features.

    Notes
    -----
    - `swap_equiv` is estimated from extra two-qubit gates as `extra_twoq / 3.0`
      because routing overhead is often decomposed into CXs rather than explicit SWAPs.
    - Ratio features are derived from primitive logical/compiled counts.
    """
    logical_depth = float(qc.depth())
    logical_twoq = float(count_two_qubit_ops(qc))
    logical_total_ops = float(count_total_ops(qc))

    result = transpile_circuit(
        qc=qc,
        coupling_map=coupling_map,
        seed_transpiler=seed,
        basis_gates=basis_gates or ["rz", "sx", "x", "cx"],
        optimization_level=optimization_level,
        layout_method=layout_method,
        routing_method=routing_method,
    )

    tqc: QuantumCircuit = result["transpiled_circuit"]
    transpile_ms = float(result["transpile_ms"])

    verif_tvd = float("nan")
    verif_fidelity = float("nan")
    verif_passed = float("nan")

    if verify:
        verif = verify_transpilation(qc, tqc, tvd_threshold=tvd_threshold)
        verif_tvd = float(verif["tvd"])
        verif_fidelity = float(verif["fidelity"])
        verif_passed = float(verif["passed"])

    routed_depth = float(tqc.depth())
    routed_twoq = float(count_two_qubit_ops(tqc))
    routed_total_ops = float(count_total_ops(tqc))

    extra_twoq = max(0.0, routed_twoq - logical_twoq)
    extra_depth = max(0.0, routed_depth - logical_depth)

    # Canonical routing proxy
    swap_equiv = extra_twoq / 3.0

    # Ratios
    swap_fraction = float(swap_equiv / max(routed_twoq, 1.0))
    cx_fraction = float(routed_twoq / max(routed_total_ops, 1.0))
    depth_overhead = float(routed_depth / max(logical_depth, 1.0))
    twoq_overhead = float(routed_twoq / max(logical_twoq, 1.0))

    return {
        "swap_equiv": float(swap_equiv),
        "swap_fraction": float(np.clip(swap_fraction, 0.0, 1.0)),
        "cx_fraction": float(np.clip(cx_fraction, 0.0, 1.0)),
        "routed_depth": float(routed_depth),
        "depth_overhead": float(depth_overhead),
        "twoq_overhead": float(twoq_overhead),
        "extra_twoq": float(extra_twoq),
        "extra_depth": float(extra_depth),
        "transpile_ms": float(transpile_ms),
        "logical_depth": float(logical_depth),
        "logical_twoq": float(logical_twoq),
        "logical_total_ops": float(logical_total_ops),
        "verif_tvd": verif_tvd,
        "verif_fidelity": verif_fidelity,
        "verif_passed": verif_passed,
    }
