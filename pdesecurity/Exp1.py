# ==============================================================================
# EXPERIMENT 1: TOPOLOGICAL FINGERPRINTING OF BOUNDARY TOPOLOGY
# ------------------------------------------------------------------------------
# Purpose
#   Test whether changing only the hidden boundary regime
#   (Dirichlet vs Periodic) produces measurable information leakage in
#   provider-visible hardware-compilation artefacts.
#
# What this fixes relative to earlier prototypes
#   1) Uses a permutation-based null MI estimate (not a single label shuffle)
#   2) Uses multiple artefacts, not just SWAP count
#   3) Uses held-out classifier validation in addition to MI
#   4) Compares a constrained topology vs a more permissive topology
#   5) Reports effect sizes and permutation p-values
#
# Notes
#   - This is still a PDE-inspired surrogate, not yet your final digital twin.
#   - It is designed to be a strong, clean Phase-1 experiment prototype.
#   - Later, you should swap the surrogate generator with your actual PDE circuits.
# ==============================================================================

import time
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import wilcoxon
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Transpilation verification
# ------------------------------------------------------------------------------

def verify_transpilation(
    original: QuantumCircuit,
    transpiled: QuantumCircuit,
    shots: int = 4096,
    tvd_threshold: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Verify that a transpiled circuit produces the same output distribution as
    the original logical circuit.

    Strategy
    --------
    We use Qiskit's statevector simulator (no noise) to obtain exact probability
    vectors for both circuits, then compute the Total Variation Distance (TVD).
    A TVD near zero means the transpiler preserved semantics; a high TVD flags a
    correctness bug.

    Returns a dict with keys:
      - 'tvd'        : float  – Total Variation Distance (0 = identical, 1 = orthogonal)
      - 'fidelity'   : float  – State fidelity |<ψ_orig|ψ_trans>|² (1 = identical)
      - 'passed'     : bool   – True when TVD < tvd_threshold
      - 'max_bitstr' : str    – Bitstring with the largest probability (sanity check)

    Notes
    -----
    * Statevector simulation is exact and fast for ≤ ~20 qubits.
    * The transpiled circuit must not contain measurement gates for statevector
      simulation; measurements are stripped automatically.
    * Circuits with parameterised gates must be bound before calling this function.
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, state_fidelity

    def _strip_measurements(qc: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of qc with all measurement operations removed."""
        stripped = QuantumCircuit(qc.num_qubits)
        for inst, qargs, cargs in qc.data:
            if inst.name not in {"measure", "reset", "barrier"}:
                stripped.append(inst, qargs, cargs)
        return stripped

    orig_clean = _strip_measurements(original)
    trans_clean = _strip_measurements(transpiled)

    sv_orig = Statevector.from_instruction(orig_clean)
    sv_trans = Statevector.from_instruction(trans_clean)

    # Probabilities over all 2^n basis states
    probs_orig = sv_orig.probabilities()
    probs_trans = sv_trans.probabilities()

    tvd = float(0.5 * np.sum(np.abs(probs_orig - probs_trans)))
    fidelity = float(state_fidelity(sv_orig, sv_trans))
    passed = tvd < tvd_threshold

    max_idx = int(np.argmax(probs_orig))
    n = original.num_qubits
    max_bitstr = format(max_idx, f"0{n}b")

    if not passed:
        warnings.warn(
            f"[verify_transpilation] TVD={tvd:.4f} exceeds threshold {tvd_threshold}. "
            f"Fidelity={fidelity:.4f}. The transpiled circuit may not be semantically "
            f"equivalent to the logical circuit. Check basis gates and routing.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "tvd": tvd,
        "fidelity": fidelity,
        "passed": passed,
        "max_bitstr": max_bitstr,
    }

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

RANDOM_SEED = 42
rng_global = np.random.default_rng(RANDOM_SEED)

N_QUBITS = 8
N_SAMPLES_PER_CLASS = 180
N_STEPS_RANGE = (2, 5)            # stochastic logical workload complexity
N_MI_BOOT = 300                   # bootstrap replications for observed MI
N_MI_PERM = 500                   # permutation replications for null MI
N_CV_SPLITS = 5
N_CV_REPEATS = 8

OUTPUT_PREFIX = "exp1_topological_fingerprinting"

# Provider-visible feature set used for the experiment
FEATURE_COLUMNS = [
    "swap_equiv",
    "swap_fraction",
    "cx_fraction",
    "routed_depth",
    "depth_overhead",
    "twoq_overhead",
    "extra_twoq",
    "extra_depth",
]

# ------------------------------------------------------------------------------
# Topology helpers
# ------------------------------------------------------------------------------

def topology_holdout_scores(all_pde_df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
    }

    rows = []
    topologies = sorted(all_pde_df["topology"].unique())

    for train_topo in topologies:
        for test_topo in topologies:
            if train_topo == test_topo:
                continue

            train_df = all_pde_df[all_pde_df["topology"] == train_topo].copy()
            test_df = all_pde_df[all_pde_df["topology"] == test_topo].copy()

            Xtr = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
            ytr = train_df["label"].to_numpy(dtype=int)
            Xte = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
            yte = test_df["label"].to_numpy(dtype=int)

            for model_name, model in models.items():
                model.fit(Xtr, ytr)
                yp = model.predict(Xte)

                rows.append({
                    "train_topology": train_topo,
                    "test_topology": test_topo,
                    "model": model_name,
                    "macro_f1": float(f1_score(yte, yp, average="macro")),
                    "bal_acc": float(balanced_accuracy_score(yte, yp)),
                })

    return pd.DataFrame(rows)


def line_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]

def ladder_edges(rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    Build a 2 x 4 ladder for 8 qubits.
    Layout:
      0 - 1 - 2 - 3
      |   |   |   |
      4 - 5 - 6 - 7
    """
    assert rows * cols == N_QUBITS
    edges = []
    # horizontal
    for r in range(rows):
        offset = r * cols
        for c in range(cols - 1):
            edges.append((offset + c, offset + c + 1))
    # vertical
    for c in range(cols):
        edges.append((c, cols + c))
    return edges

def make_topologies() -> Dict[str, CouplingMap]:
    return {
        "line_8": CouplingMap(line_edges(N_QUBITS)),
        "ladder_2x4": CouplingMap(ladder_edges(2, 4)),
    }

# ------------------------------------------------------------------------------
# Circuit-generation helpers
# ------------------------------------------------------------------------------

def count_two_qubit_ops(qc: QuantumCircuit) -> int:
    total = 0
    for inst, qargs, _ in qc.data:
        if len(qargs) == 2:
            total += 1
    return total

def count_total_ops(qc: QuantumCircuit) -> int:
    total = 0
    for inst, _, _ in qc.data:
        if inst.name not in {"barrier"}:
            total += 1
    return total

def apply_random_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    for q in range(qc.num_qubits):
        theta_z = rng.uniform(0, 2 * np.pi)
        theta_x = rng.uniform(0, 2 * np.pi)
        qc.rz(theta_z, q)
        qc.rx(theta_x, q)

def entangling_pattern_dirichlet(n: int) -> List[Tuple[int, int]]:
    """
    PDE-inspired local stencil pattern with matched interaction count.
    """
    even_edges = [(i, i + 1) for i in range(0, n - 1, 2)]     # 0-1, 2-3, ...
    odd_edges = [(i, i + 1) for i in range(1, n - 1, 2)]      # 1-2, 3-4, ...
    extra_local = [(n // 2, n // 2 + 1)]                      # duplicate local interaction
    return even_edges + odd_edges + extra_local

def entangling_pattern_periodic(n: int) -> List[Tuple[int, int]]:
    """
    Same interaction count as Dirichlet, but replace the extra local edge
    with a wrap-around edge to encode periodic topology.
    """
    even_edges = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_edges = [(i, i + 1) for i in range(1, n - 1, 2)]
    wrap_edge = [(n - 1, 0)]
    return even_edges + odd_edges + wrap_edge

import math
import numpy as np
from qiskit import QuantumCircuit


def infer_grid_shape(num_qubits: int) -> tuple[int, int]:
    """
    Choose a factorisation as close to square as possible.
    Examples:
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
    return r * cols + c


def build_stencil_partitions(rows: int, cols: int) -> dict[str, list[tuple[int, int]]]:
    """
    Build checkerboard-style local edge partitions.
    These mimic staggered stencil update groups and give non-trivial but structured variation.
    """
    parts = {
        "h_even": [],
        "h_odd": [],
        "v_even": [],
        "v_odd": [],
    }

    # Horizontal nearest-neighbour edges
    for r in range(rows):
        for c in range(cols - 1):
            a = grid_index(r, c, cols)
            b = grid_index(r, c + 1, cols)
            if c % 2 == 0:
                parts["h_even"].append((a, b))
            else:
                parts["h_odd"].append((a, b))

    # Vertical nearest-neighbour edges
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
    Periodic wrap edges.
    For 2x4 this mainly gives horizontal wrap-around per row.
    For taller grids, vertical wrap edges are also added.
    """
    wraps = []

    # Horizontal wrap: first <-> last column in each row
    if cols > 2:
        for r in range(rows):
            a = grid_index(r, 0, cols)
            b = grid_index(r, cols - 1, cols)
            wraps.append((a, b))

    # Vertical wrap: first <-> last row in each column
    # Only meaningful if there are more than 2 rows
    if rows > 2:
        for c in range(cols):
            a = grid_index(0, c, cols)
            b = grid_index(rows - 1, c, cols)
            wraps.append((a, b))

    # Remove accidental duplicates
    dedup = []
    seen = set()
    for e in wraps:
        edge = tuple(sorted(e))
        if edge not in seen:
            dedup.append(edge)
            seen.add(edge)
    return dedup


def unique_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = set()
    out = []
    for a, b in edges:
        e = tuple(sorted((a, b)))
        if e not in seen and a != b:
            out.append(e)
            seen.add(e)
    return out


def choose_step_edges(
    rows: int,
    cols: int,
    boundary_condition: str,
    step_idx: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Build a structured but stochastic edge set for one logical PDE step.

    Key design choice:
    - Dirichlet and Periodic use the SAME target interaction count.
    - Periodic replaces a small subset of local stencil edges with wrap edges.
    - This avoids making the label trivially recoverable by simple gate count.
    """
    partitions = build_stencil_partitions(rows, cols)
    wrap_edges = build_periodic_wrap_edges(rows, cols)

    part_names = ["h_even", "h_odd", "v_even", "v_odd"]

    # Primary local stencil block rotates by step index
    primary_name = part_names[step_idx % len(part_names)]
    primary_edges = list(partitions[primary_name])

    # Add a small amount of nuisance variation from a secondary partition
    secondary_name = part_names[(step_idx + 1) % len(part_names)]
    secondary_pool = list(partitions[secondary_name])
    rng.shuffle(secondary_pool)

    # Add at most half of the secondary partition to avoid making every step identical
    n_secondary = 0
    if len(secondary_pool) > 0:
        n_secondary = rng.integers(0, max(1, len(secondary_pool) // 2) + 1)

    base_edges = unique_edges(primary_edges + secondary_pool[:n_secondary])
    target_count = len(base_edges)

    if boundary_condition == "dirichlet" or len(wrap_edges) == 0:
        return base_edges

    if boundary_condition != "periodic":
        raise ValueError(f"Unknown boundary_condition={boundary_condition}")

    # Periodic case:
    # Replace a small subset of local edges with wrap edges, not add them.
    edges = list(base_edges)
    rng.shuffle(edges)

    replace_count = min(len(wrap_edges), max(1, int(round(0.25 * max(target_count, 1)))))
    replace_count = min(replace_count, len(edges))

    # Remove some local edges
    kept_edges = edges[replace_count:]

    # Select wrap edges
    wraps = list(wrap_edges)
    rng.shuffle(wraps)
    selected_wraps = wraps[:replace_count]

    periodic_edges = unique_edges(kept_edges + selected_wraps)

    # Keep logical volume matched exactly if possible
    if len(periodic_edges) > target_count:
        periodic_edges = periodic_edges[:target_count]

    # If de-duplication reduced the count, refill using unused local edges
    if len(periodic_edges) < target_count:
        pool = [e for e in edges if e not in periodic_edges]
        rng.shuffle(pool)
        need = target_count - len(periodic_edges)
        periodic_edges.extend(pool[:need])
        periodic_edges = unique_edges(periodic_edges)

    return periodic_edges[:target_count]


def apply_random_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """
    Local one-qubit layer.
    Mildly structured but random enough to stop exact repeated templates.
    """
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * np.pi), q)
        qc.rx(rng.uniform(0, 2 * np.pi), q)


def apply_coupling_block(
    qc: QuantumCircuit,
    edges: list[tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    """
    Apply a PDE-like pairwise coupling block on each selected edge.

    CX-RZ-CX is used as a simple, basis-friendly surrogate for a weighted
    interaction term. It is richer than a bare CX, but still transpiles cleanly.
    """
    for a, b in edges:
        theta = rng.uniform(0.15, 1.25) * np.pi
        qc.cx(a, b)
        qc.rz(theta, b)
        qc.cx(a, b)


def generate_pde_surrogate(
    num_qubits: int,
    boundary_condition: str,
    n_steps: int,
    seed: int,
) -> QuantumCircuit:
    """
    More developed PDE-inspired surrogate.

    Features:
      - infers a near-square 2D lattice from num_qubits
      - uses checkerboard-style local stencil partitions
      - periodic boundaries substitute wrap edges rather than simply adding them
      - preserves interaction count between Dirichlet and Periodic
      - adds stochastic but paired nuisance variation through the shared seed
      - uses structured CX-RZ-CX coupling blocks to mimic weighted interactions

    Important:
      Use the SAME `seed` and `n_steps` when generating Dirichlet and Periodic
      versions of the same matched pair. Then the only systematic difference is
      boundary topology.
    """
    if num_qubits < 4:
        raise ValueError("num_qubits should be at least 4 for this surrogate")

    rows, cols = infer_grid_shape(num_qubits)
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    # Optional initial preparation
    apply_random_local_layer(qc, rng)

    for step_idx in range(n_steps):
        # Local on-site term / coefficient variation
        apply_random_local_layer(qc, rng)

        # Structured boundary-dependent interaction pattern
        edges = choose_step_edges(rows, cols, boundary_condition, step_idx, rng)

        # Pairwise coupling block
        apply_coupling_block(qc, edges, rng)

        # Light mixing layer to avoid overly rigid templates
        for q in range(num_qubits):
            qc.rz(rng.uniform(-0.25, 0.25) * np.pi, q)

    # Final local layer
    apply_random_local_layer(qc, rng)

    return qc

def generate_random_control(
    num_qubits: int,
    n_steps: int,
    logical_twoq_target: int,
    seed: int,
) -> QuantumCircuit:
    """
    Random control with matched approximate two-qubit volume and similar local layers,
    but no meaningful topological label.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    total_twoq_added = 0
    while total_twoq_added < logical_twoq_target:
        apply_random_local_layer(qc, rng)
        # one block of random two-qubit interactions
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

# ------------------------------------------------------------------------------
# Compilation and feature extraction
# ------------------------------------------------------------------------------

def compile_and_extract_features(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    seed_transpiler: int,
) -> Dict[str, float]:
    logical_depth = float(qc.depth())
    logical_twoq = float(count_two_qubit_ops(qc))
    logical_total_ops = float(count_total_ops(qc))

    t0 = time.perf_counter()
    tqc = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed_transpiler,
    )
    transpile_ms = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    # Distribution verification: confirm the transpiled circuit is
    # semantically equivalent to the original logical circuit.
    # ------------------------------------------------------------------
    verif = verify_transpilation(qc, tqc, seed=seed_transpiler)

    ops = tqc.count_ops()
    explicit_swap_count = float(ops.get("swap", 0))   # usually 0 in CX basis
    cx_count = float(ops.get("cx", 0))
    routed_depth = float(tqc.depth())

    total_ops = float(count_total_ops(tqc))
    total_twoq = float(count_two_qubit_ops(tqc))

    # --------------------------------------------------------------------------
    # Key fix:
    # In a CX-only basis, routing overhead appears mostly as EXTRA 2Q gates,
    # not explicit SWAP instructions.
    # A SWAP is roughly 3 CXs, so we estimate "swap-equivalent" overhead.
    # --------------------------------------------------------------------------
    extra_twoq = max(0.0, total_twoq - logical_twoq)
    swap_equiv = extra_twoq / 3.0
    extra_depth = max(0.0, routed_depth - logical_depth)

    features = {
        # Backward-compatible names:
        # These now represent routing overhead in a way that survives SWAP decomposition.
        "swap_count": swap_equiv,
        "swap_fraction": extra_twoq / max(total_twoq, 1.0),

        # Existing useful features
        "cx_fraction": cx_count / max(total_ops, 1.0),
        "routed_depth": routed_depth,
        "depth_overhead": routed_depth / max(logical_depth, 1.0),
        "twoq_overhead": total_twoq / max(logical_twoq, 1.0),

        # New explicit routing-overhead features
        "extra_twoq": extra_twoq,
        "swap_equiv": swap_equiv,
        "extra_depth": extra_depth,

        # Operational feature
        "transpile_ms": transpile_ms,

        # Keep logical baselines for debugging / analysis
        "logical_depth": logical_depth,
        "logical_twoq": logical_twoq,
        "logical_total_ops": logical_total_ops,

        # Optional diagnostic
        "explicit_swap_count": explicit_swap_count,

        # Transpilation verification
        "verif_tvd": verif["tvd"],
        "verif_fidelity": verif["fidelity"],
        "verif_passed": float(verif["passed"]),
    }
    return features

# ------------------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------------------

def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """
    Family-wise error rate control (Holm step-down).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    False discovery rate control (BH).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ordered = p[order]

    adjusted_ordered = np.empty(m, dtype=float)
    running_min = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ordered[i] * m / rank
        running_min = min(running_min, val)
        adjusted_ordered[i] = min(running_min, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def paired_cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d_z for paired samples, using within-pair differences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    delta = y - x
    if len(delta) < 2:
        return np.nan
    sd = delta.std(ddof=1)
    if np.isclose(sd, 0.0):
        return 0.0
    return delta.mean() / sd


def paired_signflip_permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """
    Exact-style paired permutation via random sign-flips on within-pair deltas.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    delta = y - x

    obs = abs(delta.mean())
    hits = 0

    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(delta))
        stat = abs(np.mean(signs * delta))
        if stat >= obs:
            hits += 1

    return (hits + 1.0) / (n_perm + 1.0)


def paired_feature_tests(
    pde_df: pd.DataFrame,
    feature_columns: List[str],
    n_perm: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Paired univariate tests using matched Dirichlet/Periodic pairs by sample_id.
    """
    rows = []

    for feat in feature_columns:
        wide = (
            pde_df[["sample_id", "boundary", feat]]
            .pivot(index="sample_id", columns="boundary", values=feat)
            .dropna()
        )

        dir_vals = wide["dirichlet"].to_numpy(dtype=float)
        per_vals = wide["periodic"].to_numpy(dtype=float)
        delta = per_vals - dir_vals

        dz = paired_cohens_dz(dir_vals, per_vals)
        perm_p = paired_signflip_permutation_pvalue(
            dir_vals, per_vals, n_perm=n_perm, seed=seed
        )

        try:
            _, wilcox_p = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            wilcox_p = 1.0

        rows.append({
            "feature": feat,
            "mean_delta_periodic_minus_dirichlet": float(delta.mean()),
            "median_delta_periodic_minus_dirichlet": float(np.median(delta)),
            "cohen_dz": float(dz),
            "wilcoxon_p": float(wilcox_p),
            "paired_perm_p": float(perm_p),
        })

    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm_correction(out["wilcoxon_p"].to_numpy())
    out["paired_perm_p_holm"] = holm_correction(out["paired_perm_p"].to_numpy())
    out["wilcoxon_p_bh"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy())
    out["paired_perm_p_bh"] = benjamini_hochberg(out["paired_perm_p"].to_numpy())
    return out


def paired_label_swap_null_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Pair-preserving permutation null for MI:
    randomly swap labels *within* each matched pair, rather than globally.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)

    pair_indices = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) != 2:
            raise ValueError("Each group must contain exactly 2 rows for paired MI null.")
        pair_indices.append(idx)

    null_dist = []
    for _ in range(n_perm):
        yp = np.asarray(y).copy()
        for idx in pair_indices:
            if rng.random() < 0.5:
                yp[idx] = yp[idx[::-1]]
        mi_nats = mutual_info_classif(X, yp, discrete_features=False, random_state=seed)
        null_dist.append(mi_nats / np.log(2.0))

    return np.asarray(null_dist)


def grouped_cv_classifier_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Group-aware repeated CV so matched pairs never split across train/test.
    """
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
        ),
    }

    rows = []
    for model_name, model in models.items():
        f1s, bals = [], []

        for rep in range(N_CV_REPEATS):
            cv = StratifiedGroupKFold(
                n_splits=N_CV_SPLITS,
                shuffle=True,
                random_state=seed + rep,
            )

            for train_idx, test_idx in cv.split(X, y, groups):
                Xtr, Xte = X[train_idx], X[test_idx]
                ytr, yte = y[train_idx], y[test_idx]

                model.fit(Xtr, ytr)
                yp = model.predict(Xte)

                f1s.append(f1_score(yte, yp, average="macro"))
                bals.append(balanced_accuracy_score(yte, yp))

        rows.append({
            "model": model_name,
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "bal_acc_mean": float(np.mean(bals)),
            "bal_acc_std": float(np.std(bals)),
        })

    return pd.DataFrame(rows)

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if pooled == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled

def bootstrap_mi_bits_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap MI CIs by resampling matched pairs/groups, not individual rows.
    """
    rng = np.random.default_rng(seed)

    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    mis = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])

        Xb = X[idx]
        yb = y[idx]

        mi_nats = mutual_info_classif(
            Xb, yb, discrete_features=False, random_state=seed
        )
        mis.append(mi_nats / np.log(2.0))

    mis = np.asarray(mis)
    mean = mis.mean(axis=0)
    lo = np.percentile(mis, 2.5, axis=0)
    hi = np.percentile(mis, 97.5, axis=0)
    return mean, lo, hi, mis



def permutation_null_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Proper permutation-based null distribution for MI.
    Returns:
      null_dist: shape [n_perm, n_features]
      pvals: permutation p-values for observed MI must be computed separately
    """
    rng = np.random.default_rng(seed)
    null_dist = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        mi_nats = mutual_info_classif(X, yp, discrete_features=False, random_state=seed)
        null_dist.append(mi_nats / np.log(2.0))
    return np.asarray(null_dist)

def permutation_pvals(observed: np.ndarray, null_dist: np.ndarray) -> np.ndarray:
    pvals = []
    for j in range(observed.shape[0]):
        p = (1.0 + np.sum(null_dist[:, j] >= observed[j])) / (1.0 + len(null_dist))
        pvals.append(p)
    return np.asarray(pvals)

def repeated_cv_classifier_scores(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(
        n_splits=N_CV_SPLITS,
        n_repeats=N_CV_REPEATS,
        random_state=seed,
    )

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
        ),
    }

    rows = []
    for model_name, model in models.items():
        f1s, bals = [], []
        for train_idx, test_idx in cv.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            model.fit(Xtr, ytr)
            yp = model.predict(Xte)

            f1s.append(f1_score(yte, yp, average="macro"))
            bals.append(balanced_accuracy_score(yte, yp))

        rows.append({
            "model": model_name,
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "bal_acc_mean": float(np.mean(bals)),
            "bal_acc_std": float(np.std(bals)),
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------------------------

@dataclass
class ExperimentDataset:
    pde_df: pd.DataFrame
    random_df: pd.DataFrame

def build_dataset_for_topology(
    topology_name: str,
    coupling_map: CouplingMap,
    n_samples_per_class: int = N_SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> ExperimentDataset:
    rng = np.random.default_rng(seed)

    pde_rows = []
    rand_rows = []

    for sample_id in range(n_samples_per_class):
        n_steps = int(rng.integers(N_STEPS_RANGE[0], N_STEPS_RANGE[1] + 1))
        logical_seed = int(rng.integers(0, 10_000_000))

        # PDE pair with matched nuisance variation, differing only by BC
        qc_dir = generate_pde_surrogate(N_QUBITS, "dirichlet", n_steps, logical_seed)
        qc_per = generate_pde_surrogate(N_QUBITS, "periodic", n_steps, logical_seed)

        dir_feat = compile_and_extract_features(
            qc_dir, coupling_map, seed_transpiler=int(rng.integers(0, 10_000_000))
        )
        per_feat = compile_and_extract_features(
            qc_per, coupling_map, seed_transpiler=int(rng.integers(0, 10_000_000))
        )

        dir_feat.update({
            "label": 0,
            "boundary": "dirichlet",
            "topology": topology_name,
            "sample_id": sample_id,
            "logical_seed": logical_seed,
            "n_steps": n_steps,
        })
        per_feat.update({
            "label": 1,
            "boundary": "periodic",
            "topology": topology_name,
            "sample_id": sample_id,
            "logical_seed": logical_seed,
            "n_steps": n_steps,
        })

        pde_rows.extend([dir_feat, per_feat])

        # Random controls matched to periodic logical volume
        logical_twoq_target = count_two_qubit_ops(qc_per)

        qc_rand_1 = generate_random_control(
            N_QUBITS, n_steps, logical_twoq_target, int(rng.integers(0, 10_000_000))
        )
        qc_rand_2 = generate_random_control(
            N_QUBITS, n_steps, logical_twoq_target, int(rng.integers(0, 10_000_000))
        )

        rand_feat_1 = compile_and_extract_features(
            qc_rand_1, coupling_map, seed_transpiler=int(rng.integers(0, 10_000_000))
        )
        rand_feat_2 = compile_and_extract_features(
            qc_rand_2, coupling_map, seed_transpiler=int(rng.integers(0, 10_000_000))
        )

        rand_feat_1.update({
            "topology": topology_name,
            "sample_id": sample_id,
            "n_steps": n_steps,
        })
        rand_feat_2.update({
            "topology": topology_name,
            "sample_id": sample_id,
            "n_steps": n_steps,
        })

        rand_rows.extend([rand_feat_1, rand_feat_2])

    pde_df = pd.DataFrame(pde_rows)
    random_df = pd.DataFrame(rand_rows)

    return ExperimentDataset(pde_df=pde_df, random_df=random_df)

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

def plot_mi_bars(
    feature_names: List[str],
    observed_mean: np.ndarray,
    observed_lo: np.ndarray,
    observed_hi: np.ndarray,
    null_dist: np.ndarray,
    title: str,
    outpath: str,
) -> None:
    x = np.arange(len(feature_names))
    yerr = np.vstack([observed_mean - observed_lo, observed_hi - observed_mean])

    null_mean = null_dist.mean(axis=0)
    null_hi = np.percentile(null_dist, 97.5, axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(x, observed_mean, yerr=yerr, capsize=4)
    plt.plot(x, null_mean, marker="o", linestyle="--", label="Permutation-null mean")
    plt.fill_between(x, 0, null_hi, alpha=0.15, label="Permutation-null 95% upper band")
    plt.xticks(x, feature_names, rotation=20)
    plt.ylabel("Mutual information (bits)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_feature_distributions(
    df: pd.DataFrame,
    feature: str,
    topology_name: str,
    outpath: str,
) -> None:
    dir_vals = df[df["boundary"] == "dirichlet"][feature].values
    per_vals = df[df["boundary"] == "periodic"][feature].values

    plt.figure(figsize=(8, 5))
    bins = 20
    plt.hist(dir_vals, bins=bins, alpha=0.6, density=True, label="Dirichlet")
    plt.hist(per_vals, bins=bins, alpha=0.6, density=True, label="Periodic")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"{topology_name}: {feature} distribution by boundary regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Main analysis routine
# ------------------------------------------------------------------------------
def run_topological_experiment():
    topologies = make_topologies()
    all_summary_rows = []
    all_pde_dfs = []

    for topo_name, cmap in topologies.items():
        print("\n" + "=" * 90)
        print(f"RUNNING TOPOLOGICAL FINGERPRINTING ON TOPOLOGY: {topo_name}")
        print("=" * 90)

        ds = build_dataset_for_topology(topo_name, cmap)

        pde_df = ds.pde_df.copy()
        rand_df = ds.random_df.copy()

        # collect for unseen-topology holdout later
        all_pde_dfs.append(pde_df.copy())

        # Labels and groups
        y_pde = pde_df["label"].to_numpy().astype(int)
        groups = pde_df["sample_id"].to_numpy()

        # Proper null-hypothesis controls:
        #   (A) random controls with random label assignments
        #   (B) pair-preserving permutation null on the observed PDE features themselves
        y_null = np.random.default_rng(RANDOM_SEED).permutation(y_pde)

        X_pde = pde_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        X_rand = rand_df[FEATURE_COLUMNS].to_numpy(dtype=float)

        # Observed MI point estimate on the full dataset
        obs_point_nats = mutual_info_classif(
            X_pde,
            y_pde,
            discrete_features=False,
            random_state=RANDOM_SEED,
        )
        obs_point = obs_point_nats / np.log(2.0)

        # Bootstrap CI for MI
        # If you added bootstrap_mi_bits_grouped(...), use that here instead
        obs_mean, obs_lo, obs_hi, _ = bootstrap_mi_bits_grouped(
            X_pde, y_pde, groups=groups, n_boot=N_MI_BOOT, seed=RANDOM_SEED
        )

        # Null MI from random control + permutation
        null_rand = permutation_null_mi_bits(
            X_rand, y_null, n_perm=N_MI_PERM, seed=RANDOM_SEED
        )

        # Pair-preserving MI null on the real PDE features
        null_perm = paired_label_swap_null_mi_bits(
            X_pde, y_pde, groups=groups, n_perm=N_MI_PERM, seed=RANDOM_SEED + 1
        )

        # IMPORTANT: use obs_point, not obs_mean, for permutation p-values
        pvals_raw = permutation_pvals(obs_point, null_perm)
        pvals_holm = holm_correction(pvals_raw)
        pvals_bh = benjamini_hochberg(pvals_raw)

        # Paired effect testing
        effect_df = paired_feature_tests(
            pde_df,
            feature_columns=FEATURE_COLUMNS,
            n_perm=5000,
            seed=RANDOM_SEED,
        )
        effect_df.insert(0, "topology", topo_name)

        # Group-aware held-out classifier validation
        cv_df = grouped_cv_classifier_scores(
            X_pde, y_pde, groups=groups, seed=RANDOM_SEED
        )

        print("\nObserved MI vs pair-preserving null baselines")
        print("-" * 120)
        print(
            f"{'Feature':<18} {'Observed MI bits [95% CI]':<30} "
            f"{'Rand-null mean':<16} {'Pair-null 95%':<16} "
            f"{'p-raw':<10} {'p-Holm':<10} {'p-BH':<10}"
        )
        for j, feat in enumerate(FEATURE_COLUMNS):
            obs_str = f"{obs_point[j]:.3f} [{obs_lo[j]:.3f}, {obs_hi[j]:.3f}]"
            rand_null_mean = float(null_rand[:, j].mean())
            pair_null_95 = float(np.percentile(null_perm[:, j], 95))
            print(
                f"{feat:<18} {obs_str:<30} {rand_null_mean:<16.3f} "
                f"{pair_null_95:<16.3f} {pvals_raw[j]:<10.4f} "
                f"{pvals_holm[j]:<10.4f} {pvals_bh[j]:<10.4f}"
            )

        print("\nPaired effect tests (Periodic - Dirichlet)")
        print("-" * 120)
        print(effect_df.to_string(index=False))

        print("\nHeld-out classifier validation")
        print("-" * 90)
        print(cv_df.to_string(index=False))

        # Save plots
        plot_mi_bars(
            feature_names=FEATURE_COLUMNS,
            observed_mean=obs_mean,
            observed_lo=obs_lo,
            observed_hi=obs_hi,
            null_dist=null_perm,
            title=f"{topo_name}: observed MI vs pair-preserving null",
            outpath=f"{OUTPUT_PREFIX}_{topo_name}_mi.png",
        )

        for feat in ["swap_count", "swap_fraction", "cx_fraction", "routed_depth"]:
            plot_feature_distributions(
                pde_df, feat, topo_name, f"{OUTPUT_PREFIX}_{topo_name}_{feat}.png"
            )

        # Summaries for cross-topology table
        for _, row in cv_df.iterrows():
            all_summary_rows.append({
                "topology": topo_name,
                "model": row["model"],
                "macro_f1_mean": row["macro_f1_mean"],
                "bal_acc_mean": row["bal_acc_mean"],
                "swap_equiv_mi_bits": obs_point[FEATURE_COLUMNS.index("swap_equiv")],
                "cxfrac_mi_bits": obs_point[FEATURE_COLUMNS.index("cx_fraction")],
                "depth_mi_bits": obs_point[FEATURE_COLUMNS.index("routed_depth")],
            })

    # --------------------------------------------------------------------------
    # Unseen-topology holdout
    # --------------------------------------------------------------------------
    all_pde_df = pd.concat(all_pde_dfs, axis=0, ignore_index=True)
    topo_holdout_df = topology_holdout_scores(all_pde_df)

    print("\n" + "=" * 90)
    print("UNSEEN-TOPOLOGY HOLDOUT")
    print("=" * 90)
    print(topo_holdout_df.to_string(index=False))

    topo_holdout_df.to_csv(f"{OUTPUT_PREFIX}_topology_holdout.csv", index=False)
    print(f"\nSaved topology holdout CSV to: {OUTPUT_PREFIX}_topology_holdout.csv")

    # --------------------------------------------------------------------------
    # Cross-topology summary
    # --------------------------------------------------------------------------
    summary_df = pd.DataFrame(all_summary_rows)
    print("\n" + "=" * 90)
    print("CROSS-TOPOLOGY SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(f"{OUTPUT_PREFIX}_summary.csv", index=False)
    print(f"\nSaved summary CSV to: {OUTPUT_PREFIX}_summary.csv")
    print("Saved MI and feature-distribution plots as PNG files.")
# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run_topological_experiment()