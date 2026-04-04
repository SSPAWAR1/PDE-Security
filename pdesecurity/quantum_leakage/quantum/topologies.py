"""
Topology helpers and coupling-map constructors.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from qiskit.transpiler import CouplingMap


def infer_grid_shape(num_qubits: int) -> Tuple[int, int]:
    """
    Choose a factorisation as close to square as possible.

    Examples
    --------
    8  -> (2, 4)
    12 -> (3, 4)
    16 -> (4, 4)
    """
    best_rows, best_cols = 1, num_qubits
    best_gap = num_qubits - 1

    for rows in range(1, int(math.sqrt(num_qubits)) + 1):
        if num_qubits % rows == 0:
            cols = num_qubits // rows
            gap = abs(cols - rows)
            if gap < best_gap:
                best_rows, best_cols = rows, cols
                best_gap = gap

    return best_rows, best_cols


def grid_index(r: int, c: int, cols: int) -> int:
    """
    Convert (row, col) into a flattened qubit index.
    """
    return r * cols + c


def line_edges(n: int, bidirectional: bool = False) -> List[Tuple[int, int]]:
    """
    Line topology edges.
    """
    if bidirectional:
        edges: List[Tuple[int, int]] = []
        for i in range(n - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        return edges

    return [(i, i + 1) for i in range(n - 1)]


def ladder_edges(rows: int, cols: int, bidirectional: bool = False) -> List[Tuple[int, int]]:
    """
    Build a ladder topology.

    For rows=2, cols=4:
      0 - 1 - 2 - 3
      |   |   |   |
      4 - 5 - 6 - 7
    """
    edges: List[Tuple[int, int]] = []

    def _add(a: int, b: int) -> None:
        edges.append((a, b))
        if bidirectional:
            edges.append((b, a))

    # Horizontal edges
    for r in range(rows):
        offset = r * cols
        for c in range(cols - 1):
            _add(offset + c, offset + c + 1)

    # Vertical edges
    for c in range(cols):
        _add(c, cols + c)

    return edges


def gridish_edges(n: int, bidirectional: bool = False) -> List[Tuple[int, int]]:
    """
    Near-square grid topology for an arbitrary number of qubits.
    """
    rows, cols = infer_grid_shape(n)
    edges: List[Tuple[int, int]] = []

    def _add(a: int, b: int) -> None:
        edges.append((a, b))
        if bidirectional:
            edges.append((b, a))

    for r in range(rows):
        for c in range(cols - 1):
            _add(grid_index(r, c, cols), grid_index(r, c + 1, cols))

    for r in range(rows - 1):
        for c in range(cols):
            _add(grid_index(r, c, cols), grid_index(r + 1, c, cols))

    return edges


def make_coupling_map(num_qubits: int, topology_family: str) -> CouplingMap:
    """
    Construct a coupling map from a topology family name.
    """
    if topology_family == "line":
        return CouplingMap(line_edges(num_qubits))
    if topology_family == "gridish":
        return CouplingMap(gridish_edges(num_qubits))
    if topology_family == "ladder":
        rows, cols = infer_grid_shape(num_qubits)
        if rows != 2:
            rows, cols = 2, num_qubits // 2
        return CouplingMap(ladder_edges(rows, cols))
    raise ValueError(f"Unknown topology_family={topology_family}")


def make_topologies(num_qubits: int) -> Dict[str, CouplingMap]:
    """
    Convenience helper for common topology maps.
    """
    return {
        "line": make_coupling_map(num_qubits, "line"),
        "gridish": make_coupling_map(num_qubits, "gridish"),
        "ladder": make_coupling_map(num_qubits, "ladder"),
    }
