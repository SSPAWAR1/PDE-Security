"""
Shared transpilation helpers.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap


def transpile_circuit(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    seed_transpiler: int,
    basis_gates: Optional[list[str]] = None,
    optimization_level: int = 1,
    layout_method: str = "sabre",
    routing_method: str = "sabre",
) -> Dict[str, Any]:
    """
    Transpile a circuit and measure transpilation time.
    """
    t0 = time.perf_counter()

    tqc = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        layout_method=layout_method,
        routing_method=routing_method,
        seed_transpiler=seed_transpiler,
    )

    transpile_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "transpiled_circuit": tqc,
        "transpile_ms": float(transpile_ms),
    }
