"""
Topology helpers and coupling-map constructors.
"""

from __future__ import annotations

import math
from typing import Literal

from qiskit.transpiler import CouplingMap

TopologyFamily = Literal["line", "rectangular_grid", "ladder"]


def infer_grid_shape(num_qubits: int) -> tuple[int, int]:
    """
    Choose an exact integer factorisation whose aspect ratio is as close to square
    as possible. For prime numbers, this returns (1, num_qubits).

    Examples
    --------
    8  -> (2, 4)
    12 -> (3, 4)
    16 -> (4, 4)
    7  -> (1, 7)
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be strictly positive.")

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


def line_edges(n: int, bidirectional: bool = True) -> list[tuple[int, int]]:
    """
    Line topology edges.
    """
    if n <= 0:
        raise ValueError("Number of qubits (n) must be strictly positive.")

    edges: list[tuple[int, int]] = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        if bidirectional:
            edges.append((i + 1, i))
            
    return edges


def ladder_edges(cols: int, bidirectional: bool = True) -> list[tuple[int, int]]:
    """
    Build a 2-row ladder topology. Total qubits = 2 * cols.

    For cols=4:
      0 - 1 - 2 - 3
      |   |   |   |
      4 - 5 - 6 - 7
    """
    if cols <= 0:
        raise ValueError("Number of columns must be strictly positive.")

    edges: list[tuple[int, int]] = []

    def _add(a: int, b: int) -> None:
        edges.append((a, b))
        if bidirectional:
            edges.append((b, a))

    # Horizontal edges
    for r in range(2):
        offset = r * cols
        for c in range(cols - 1):
            _add(offset + c, offset + c + 1)

    # Vertical edges
    for c in range(cols):
        _add(c, cols + c)

    return edges


def rectangular_grid_edges(n: int, bidirectional: bool = True) -> list[tuple[int, int]]:
    """
    Exact rectangular grid topology for an arbitrary number of qubits.
    If n is prime, this mathematically devolves into a line.
    """
    if n <= 0:
        raise ValueError("Number of qubits (n) must be strictly positive.")

    rows, cols = infer_grid_shape(n)
    edges: list[tuple[int, int]] = []

    def _add(a: int, b: int) -> None:
        edges.append((a, b))
        if bidirectional:
            edges.append((b, a))

    # Horizontal edges
    for r in range(rows):
        for c in range(cols - 1):
            _add(grid_index(r, c, cols), grid_index(r, c + 1, cols))

    # Vertical edges
    for r in range(rows - 1):
        for c in range(cols):
            _add(grid_index(r, c, cols), grid_index(r + 1, c, cols))

    return edges


def make_coupling_map(
    num_qubits: int, 
    topology_family: TopologyFamily, 
    bidirectional: bool = True
) -> CouplingMap:
    """
    Construct a coupling map from a topology family name.
    """
    if topology_family == "line":
        return CouplingMap(line_edges(num_qubits, bidirectional=bidirectional))
        
    if topology_family == "rectangular_grid":
        return CouplingMap(rectangular_grid_edges(num_qubits, bidirectional=bidirectional))
        
    if topology_family == "ladder":
        if num_qubits % 2 != 0:
            raise ValueError(
                f"Ladder topology requires an even number of qubits, got {num_qubits}."
            )
        cols = num_qubits // 2
        return CouplingMap(ladder_edges(cols, bidirectional=bidirectional))
        
    raise ValueError(f"Unknown topology_family={topology_family}")


def make_topologies(num_qubits: int, bidirectional: bool = True) -> dict[str, CouplingMap]:
    """
    Convenience helper for common topology maps. 
    Intelligently omits topologies that are invalid for the given qubit count.
    """
    tops = {
        "line": make_coupling_map(num_qubits, "line", bidirectional),
        "rectangular_grid": make_coupling_map(num_qubits, "rectangular_grid", bidirectional),
    }
    
    # Only generate a ladder if the math supports it cleanly
    if num_qubits % 2 == 0:
        tops["ladder"] = make_coupling_map(num_qubits, "ladder", bidirectional)
        
    return tops
