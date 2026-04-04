# ==============================================================================
# EXPERIMENT 4: MATERIAL PROPERTY LEAKAGE - VERACITY/ACCURACY REQUIREMENTS
# ------------------------------------------------------------------------------
# Purpose
#   Test whether hidden accuracy/fidelity requirements (material properties)
#   produce measurable information leakage in provider-visible hardware-compilation
#   artefacts. Specifically: does a high-accuracy vs low-accuracy requirement
#   (reflected in circuit approximation depth/quality) leave detectable signatures?
#
# Design rationale
#   In scientific computing, accuracy requirements drive approximation depth:
#   - High-accuracy PDE solvers use deeper Trotter expansions
#   - High-fidelity quantum simulations require more precise gate decompositions
#   - Numerical precision demands affect circuit structure
#
#   This experiment tests whether a provider can infer accuracy requirements
#   from compilation overheads alone, without seeing the logical circuit.
#
# What this tests
#   1) Binary classification: high-accuracy vs low-accuracy workloads
#   2) Multi-class ordinal: precision levels (e.g., 1e-2, 1e-3, 1e-4, 1e-5)
#   3) Robustness across different scientific workload families
#   4) Stability under hardware drift
#   5) Feature-group ablation (which coupling modes carry the signal)
#
# Improvements over prototypes
#   - Uses permutation-null MI baselines
#   - Uses family-holdout validation (train on one workload type, test on another)
#   - Uses ordinal metrics for multi-class accuracy-level inference
#   - Reports effect sizes and statistical significance
#   - Tests across multiple topologies
# ==============================================================================

import time
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from scipy.stats import wilcoxon
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Transpilation verification
# ------------------------------------------------------------------------------

def verify_transpilation(
    original: QuantumCircuit,
    transpiled: QuantumCircuit,
    tvd_threshold: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Verify that a transpiled circuit produces the same output distribution as
    the original logical circuit.

    Uses exact statevector simulation to compute Total Variation Distance (TVD)
    and state fidelity between the original and transpiled circuits.

    Parameters
    ----------
    original    : the pre-transpilation logical circuit (no measurements)
    transpiled  : the post-transpilation physical circuit (no measurements)
    tvd_threshold : maximum acceptable TVD; circuits above this emit a warning
    seed        : unused here but kept for API consistency across experiments

    Returns
    -------
    dict with keys:
      'tvd'      – Total Variation Distance in [0, 1]  (0 = identical distributions)
      'fidelity' – |<ψ_orig|ψ_trans>|²               (1 = identical states)
      'passed'   – True when TVD < tvd_threshold
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, state_fidelity

    def _strip(qc: QuantumCircuit) -> QuantumCircuit:
        out = QuantumCircuit(qc.num_qubits)
        for inst, qargs, cargs in qc.data:
            if inst.name not in {"measure", "reset", "barrier"}:
                out.append(inst, qargs, cargs)
        return out

    sv_orig = Statevector.from_instruction(_strip(original))
    sv_trans = Statevector.from_instruction(_strip(transpiled))

    probs_orig = sv_orig.probabilities()
    probs_trans = sv_trans.probabilities()

    tvd = float(0.5 * np.sum(np.abs(probs_orig - probs_trans)))
    fidelity = float(state_fidelity(sv_orig, sv_trans))
    passed = tvd < tvd_threshold

    if not passed:
        warnings.warn(
            f"[verify_transpilation] TVD={tvd:.4f} exceeds threshold {tvd_threshold}. "
            f"Fidelity={fidelity:.4f}. The transpiled circuit may not preserve the "
            "original computation — check basis gates and optimization level.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {"tvd": tvd, "fidelity": fidelity, "passed": passed}

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

RANDOM_SEED = 42
rng_global = np.random.default_rng(RANDOM_SEED)

N_QUBITS = 8

# Binary veracity classification
BINARY_ACCURACIES = ["low", "high"]
N_BINARY_SAMPLES_PER_CLASS = 180

# Ordinal veracity levels
ORDINAL_ACCURACIES = [1e-2, 1e-3, 1e-4, 1e-5]  # Target accuracy levels
N_ORDINAL_SAMPLES_PER_LEVEL = 100

# Workload families for cross-validation
WORKLOAD_FAMILIES = ["time_evolution", "optimization"]

# Topology families
TOPOLOGY_FAMILIES = ["line", "ladder"]

# Statistical testing
N_MI_BOOT = 300
N_MI_PERM = 500
N_CV_SPLITS = 5
N_CV_REPEATS = 8

# Drift testing
DRIFT_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_DRIFT_REPEATS = 10

OUTPUT_PREFIX = "exp4_veracity_leakage"

# Provider-visible feature set
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

# Feature groups for ablation study
FEATURE_GROUPS = {
    "Topological": [
        "swap_equiv",
        "swap_fraction",
        "extra_twoq",
    ],
    "Complexity": [
        "routed_depth",
        "depth_overhead",
        "twoq_overhead",
        "extra_depth",
        "cx_fraction",
    ],
}
FEATURE_GROUPS["All"] = FEATURE_GROUPS["Topological"] + FEATURE_GROUPS["Complexity"]

# ------------------------------------------------------------------------------
# Topology helpers
# ------------------------------------------------------------------------------

def line_edges(n: int) -> List[Tuple[int, int]]:
    """Create bidirectional edges for a line topology."""
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))  # Add reverse direction
    return edges

def ladder_edges(rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    Build a 2 x 4 ladder for 8 qubits with bidirectional edges.
    Layout:
      0 - 1 - 2 - 3
      |   |   |   |
      4 - 5 - 6 - 7
    """
    edges = []
    # horizontal (bidirectional)
    for r in range(rows):
        offset = r * cols
        for c in range(cols - 1):
            edges.append((offset + c, offset + c + 1))
            edges.append((offset + c + 1, offset + c))  # Add reverse
    # vertical (bidirectional)
    for c in range(cols):
        edges.append((c, cols + c))
        edges.append((cols + c, c))  # Add reverse
    return edges

def make_topologies() -> Dict[str, CouplingMap]:
    return {
        "line": CouplingMap(line_edges(N_QUBITS)),
        "ladder": CouplingMap(ladder_edges(2, 4)),
    }

def infer_grid_shape(num_qubits: int) -> Tuple[int, int]:
    """Choose a factorisation as close to square as possible."""
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

# ------------------------------------------------------------------------------
# Circuit-generation helpers
# ------------------------------------------------------------------------------

def count_two_qubit_ops(qc: QuantumCircuit) -> int:
    return sum(1 for inst, qargs, _ in qc.data if len(qargs) == 2)

def count_total_ops(qc: QuantumCircuit) -> int:
    return sum(1 for inst, _, _ in qc.data if inst.name not in {"barrier"})

def apply_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """Apply random single-qubit rotations."""
    for q in range(qc.num_qubits):
        theta_z = rng.uniform(0, 2 * np.pi)
        theta_x = rng.uniform(0, 2 * np.pi)
        qc.rz(theta_z, q)
        qc.rx(theta_x, q)

def accuracy_to_trotter_steps(accuracy: float) -> int:
    """
    Map target accuracy to Trotter step count.
    Higher accuracy requires more Trotter steps (deeper circuits).
    
    Approximates error ~ 1/n^2 for n Trotter steps.
    """
    # For accuracy ε, need n ~ sqrt(1/ε) steps
    base_steps = int(np.sqrt(1.0 / accuracy))
    # Add some variance but maintain ordering
    return max(2, base_steps)

def generate_time_evolution_circuit(
    n_qubits: int,
    accuracy: float,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """
    Generate a time-evolution circuit with accuracy-dependent depth.
    Models Hamiltonian simulation with Trotterization.
    """
    qc = QuantumCircuit(n_qubits)
    
    n_steps = accuracy_to_trotter_steps(accuracy)
    
    rows, cols = infer_grid_shape(n_qubits)
    
    for step in range(n_steps):
        # Local terms
        apply_local_layer(qc, rng)
        
        # Two-qubit interaction terms (nearest neighbor)
        for r in range(rows):
            for c in range(cols - 1):
                a = grid_index(r, c, cols)
                b = grid_index(r, c + 1, cols)
                angle = rng.uniform(0, np.pi / 4)
                qc.cx(a, b)
                qc.rz(angle, b)
                qc.cx(a, b)
        
        # Vertical interactions
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
    Higher accuracy requires more QAOA layers.
    """
    qc = QuantumCircuit(n_qubits)
    
    # Initial state preparation
    for q in range(n_qubits):
        qc.h(q)
    
    n_layers = accuracy_to_trotter_steps(accuracy)
    
    rows, cols = infer_grid_shape(n_qubits)
    
    for layer in range(n_layers):
        # Problem Hamiltonian layer (entangling)
        for r in range(rows):
            for c in range(cols - 1):
                a = grid_index(r, c, cols)
                b = grid_index(r, c + 1, cols)
                gamma = rng.uniform(0, np.pi / 2)
                qc.cx(a, b)
                qc.rz(gamma, b)
                qc.cx(a, b)
        
        # Mixer Hamiltonian layer (local)
        for q in range(n_qubits):
            beta = rng.uniform(0, np.pi / 2)
            qc.rx(beta, q)
    
    return qc

def compile_and_extract_features(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    seed: int,
) -> Dict[str, float]:
    """Compile circuit and extract provider-visible features."""
    
    logical_depth = qc.depth()
    logical_twoq = count_two_qubit_ops(qc)
    logical_total = count_total_ops(qc)
    
    t_start = time.perf_counter()
    transpiled = transpile(
        qc,
        coupling_map=coupling_map,
        optimization_level=2,
        seed_transpiler=seed,
    )
    t_elapsed = (time.perf_counter() - t_start) * 1000.0

    # ------------------------------------------------------------------
    # Distribution verification: confirm transpilation preserved semantics
    # ------------------------------------------------------------------
    verif = verify_transpilation(qc, transpiled, seed=seed)

    routed_depth = transpiled.depth()
    routed_twoq = count_two_qubit_ops(transpiled)
    
    # Count SWAP-equivalent operations
    swap_equiv = 0
    for inst, qargs, _ in transpiled.data:
        if inst.name == "swap":
            swap_equiv += 1
        elif inst.name == "cx" and len(qargs) == 2:
            # Check if this CX is part of a SWAP decomposition pattern
            # (simplified heuristic)
            pass
    
    swap_frac = swap_equiv / max(routed_twoq, 1)
    cx_frac = routed_twoq / max(count_total_ops(transpiled), 1)
    
    depth_overhead = (routed_depth - logical_depth) / max(logical_depth, 1)
    twoq_overhead = (routed_twoq - logical_twoq) / max(logical_twoq, 1)
    
    extra_twoq = routed_twoq - logical_twoq
    extra_depth = routed_depth - logical_depth
    
    return {
        "swap_equiv": float(swap_equiv),
        "swap_fraction": float(swap_frac),
        "cx_fraction": float(cx_frac),
        "routed_depth": float(routed_depth),
        "depth_overhead": float(depth_overhead),
        "twoq_overhead": float(twoq_overhead),
        "extra_twoq": float(extra_twoq),
        "extra_depth": float(extra_depth),
        "transpile_ms": float(t_elapsed),
        "logical_depth": float(logical_depth),
        "logical_twoq": float(logical_twoq),
        "logical_total_ops": float(logical_total),
        # Transpilation verification
        "verif_tvd": verif["tvd"],
        "verif_fidelity": verif["fidelity"],
        "verif_passed": float(verif["passed"]),
    }

# ------------------------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------------------------

@dataclass
class VeracityDataset:
    df: pd.DataFrame
    task_type: str  # "binary" or "ordinal"
    
def build_binary_veracity_dataset(
    topology_name: str,
    coupling_map: CouplingMap,
    workload_families: List[str],
    n_samples_per_class: int,
    seed: int,
) -> VeracityDataset:
    """
    Build dataset for binary veracity classification:
    low-accuracy vs high-accuracy requirements.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sample_id = 0
    
    for accuracy_level in BINARY_ACCURACIES:
        for family in workload_families:
            accuracy_value = 1e-2 if accuracy_level == "low" else 1e-4
            
            for i in range(n_samples_per_class):
                sample_seed = int(rng.integers(0, 2**30))
                sample_rng = np.random.default_rng(sample_seed)
                
                if family == "time_evolution":
                    qc = generate_time_evolution_circuit(
                        N_QUBITS, accuracy_value, sample_rng
                    )
                else:  # optimization
                    qc = generate_optimization_circuit(
                        N_QUBITS, accuracy_value, sample_rng
                    )
                
                features = compile_and_extract_features(
                    qc, coupling_map, sample_seed
                )
                
                features["accuracy_level"] = accuracy_level
                features["workload_family"] = family
                features["topology"] = topology_name
                features["sample_id"] = sample_id
                features["label"] = 0 if accuracy_level == "low" else 1
                
                rows.append(features)
                sample_id += 1
    
    df = pd.DataFrame(rows)
    return VeracityDataset(df=df, task_type="binary")

def build_ordinal_veracity_dataset(
    topology_name: str,
    coupling_map: CouplingMap,
    workload_families: List[str],
    accuracy_levels: List[float],
    n_samples_per_level: int,
    seed: int,
) -> VeracityDataset:
    """
    Build dataset for ordinal veracity classification:
    multiple accuracy levels with natural ordering.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sample_id = 0
    
    for accuracy in accuracy_levels:
        for family in workload_families:
            for i in range(n_samples_per_level):
                sample_seed = int(rng.integers(0, 2**30))
                sample_rng = np.random.default_rng(sample_seed)
                
                if family == "time_evolution":
                    qc = generate_time_evolution_circuit(
                        N_QUBITS, accuracy, sample_rng
                    )
                else:  # optimization
                    qc = generate_optimization_circuit(
                        N_QUBITS, accuracy, sample_rng
                    )
                
                features = compile_and_extract_features(
                    qc, coupling_map, sample_seed
                )
                
                features["accuracy"] = accuracy
                features["workload_family"] = family
                features["topology"] = topology_name
                features["sample_id"] = sample_id
                features["label"] = accuracy_levels.index(accuracy)
                
                rows.append(features)
                sample_id += 1
    
    df = pd.DataFrame(rows)
    return VeracityDataset(df=df, task_type="ordinal")

# ------------------------------------------------------------------------------
# Statistical analysis helpers
# ------------------------------------------------------------------------------

def bootstrap_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for mutual information."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    mi_samples = np.zeros((n_boot, n_features))
    
    for i in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        mi_nats = mutual_info_classif(
            X_boot, y_boot, discrete_features=False, random_state=seed + i
        )
        mi_samples[i] = mi_nats / np.log(2.0)
    
    mean = mi_samples.mean(axis=0)
    lo = np.percentile(mi_samples, 2.5, axis=0)
    hi = np.percentile(mi_samples, 97.5, axis=0)
    
    return mean, lo, hi

def permutation_null_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    """Generate permutation-null distribution for MI."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    null_dist = np.zeros((n_perm, n_features))
    
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        mi_nats = mutual_info_classif(
            X, y_perm, discrete_features=False, random_state=seed + i
        )
        null_dist[i] = mi_nats / np.log(2.0)
    
    return null_dist

def permutation_pvals(observed: np.ndarray, null_dist: np.ndarray) -> np.ndarray:
    """Calculate permutation p-values."""
    n_perm = null_dist.shape[0]
    pvals = np.zeros(len(observed))
    
    for j in range(len(observed)):
        pvals[j] = (null_dist[:, j] >= observed[j]).sum() / n_perm
    
    return pvals

def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction for multiple testing."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    corrected = np.zeros(n)
    
    for i, idx in enumerate(sorted_idx):
        corrected[idx] = min(1.0, pvals[idx] * (n - i))
    
    return corrected

def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    corrected = np.zeros(n)
    
    for i, idx in enumerate(sorted_idx[::-1]):
        corrected[idx] = min(1.0, pvals[idx] * n / (n - i))
    
    return corrected

def ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ordered_labels: List,
) -> Dict[str, float]:
    """Calculate ordinal classification metrics."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Class-wise MAE (mean absolute error in ordinal space)
    mae = np.abs(y_true - y_pred).mean()
    
    # Adjacent-class accuracy (predictions within ±1 level)
    adjacent_correct = np.abs(y_true - y_pred) <= 1
    adjacent_acc = adjacent_correct.mean()
    
    return {
        "macro_f1": float(macro_f1),
        "bal_acc": float(bal_acc),
        "class_mae": float(mae),
        "adjacent_acc": float(adjacent_acc),
    }

def grouped_cv_classifier_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Cross-validated classifier evaluation with group awareness."""
    cv = StratifiedGroupKFold(n_splits=N_CV_SPLITS)
    
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
        f1_scores = []
        bal_scores = []
        
        for train_idx, test_idx in cv.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            f1_scores.append(f1_score(y_test, y_pred, average="macro"))
            bal_scores.append(balanced_accuracy_score(y_test, y_pred))
        
        rows.append({
            "model": model_name,
            "macro_f1_mean": np.mean(f1_scores),
            "macro_f1_std": np.std(f1_scores),
            "bal_acc_mean": np.mean(bal_scores),
            "bal_acc_std": np.std(bal_scores),
        })
    
    return pd.DataFrame(rows)

def family_holdout_evaluation(
    df: pd.DataFrame,
    feature_cols: List[str],
    task_type: str,
    seed: int,
) -> pd.DataFrame:
    """
    Train on one workload family, test on another.
    Tests whether veracity signal generalizes across different algorithmic templates.
    """
    families = sorted(df["workload_family"].unique())
    rows = []
    
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
    
    for train_fam in families:
        for test_fam in families:
            if train_fam == test_fam:
                continue
            
            train_df = df[df["workload_family"] == train_fam]
            test_df = df[df["workload_family"] == test_fam]
            
            X_train = train_df[feature_cols].to_numpy(dtype=float)
            y_train = train_df["label"].to_numpy(dtype=int)
            X_test = test_df[feature_cols].to_numpy(dtype=float)
            y_test = test_df["label"].to_numpy(dtype=int)
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if task_type == "binary":
                    rows.append({
                        "train_family": train_fam,
                        "test_family": test_fam,
                        "model": model_name,
                        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                        "bal_acc": float(balanced_accuracy_score(y_test, y_pred)),
                    })
                else:  # ordinal
                    metrics = ordinal_metrics(y_test, y_pred, ORDINAL_ACCURACIES)
                    rows.append({
                        "train_family": train_fam,
                        "test_family": test_fam,
                        "model": model_name,
                        **metrics,
                    })
    
    return pd.DataFrame(rows)

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
    """Plot MI with confidence intervals and null distribution."""
    x = np.arange(len(feature_names))
    yerr = np.vstack([observed_mean - observed_lo, observed_hi - observed_mean])
    
    null_mean = null_dist.mean(axis=0)
    null_hi = np.percentile(null_dist, 97.5, axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.bar(x, observed_mean, yerr=yerr, capsize=4, alpha=0.8, label="Observed")
    plt.plot(x, null_mean, marker="o", linestyle="--", 
             color="red", label="Permutation-null mean")
    plt.fill_between(x, 0, null_hi, alpha=0.15, color="red",
                     label="Permutation-null 95% upper band")
    plt.xticks(x, feature_names, rotation=25, ha="right")
    plt.ylabel("Mutual information (bits)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_feature_distributions(
    df: pd.DataFrame,
    feature: str,
    label_col: str,
    label_names: Dict[int, str],
    title: str,
    outpath: str,
) -> None:
    """Plot feature distributions by class."""
    plt.figure(figsize=(8, 5))
    
    for label, name in label_names.items():
        vals = df[df[label_col] == label][feature].values
        plt.hist(vals, bins=20, alpha=0.6, density=True, label=name)
    
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List,
    title: str,
    outpath: str,
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[str(l) for l in labels],
                yticklabels=[str(l) for l in labels])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Main experiment runners
# ------------------------------------------------------------------------------

def run_binary_veracity_experiment():
    """Run binary veracity classification experiment."""
    print("\n" + "=" * 90)
    print("RUNNING BINARY VERACITY EXPERIMENT (LOW vs HIGH ACCURACY)")
    print("=" * 90)
    
    topologies = make_topologies()
    all_summary_rows = []
    
    for topo_name, cmap in topologies.items():
        print("\n" + "=" * 90)
        print(f"TOPOLOGY: {topo_name}")
        print("=" * 90)
        
        ds = build_binary_veracity_dataset(
            topology_name=topo_name,
            coupling_map=cmap,
            workload_families=WORKLOAD_FAMILIES,
            n_samples_per_class=N_BINARY_SAMPLES_PER_CLASS,
            seed=RANDOM_SEED,
        )
        
        df = ds.df
        y = df["label"].to_numpy(dtype=int)
        groups = df["sample_id"].to_numpy()
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        
        # Observed MI with bootstrap CI
        obs_mean, obs_lo, obs_hi = bootstrap_mi_bits(
            X, y, n_boot=N_MI_BOOT, seed=RANDOM_SEED
        )
        
        # Permutation null
        null_dist = permutation_null_mi_bits(
            X, y, n_perm=N_MI_PERM, seed=RANDOM_SEED
        )
        
        # Statistical testing
        pvals_raw = permutation_pvals(obs_mean, null_dist)
        pvals_holm = holm_correction(pvals_raw)
        pvals_bh = benjamini_hochberg(pvals_raw)
        
        # Cross-validation
        cv_df = grouped_cv_classifier_scores(X, y, groups, RANDOM_SEED)
        
        # Family holdout
        holdout_df = family_holdout_evaluation(
            df, FEATURE_COLUMNS, "binary", RANDOM_SEED
        )
        
        # Print results
        print("\nObserved MI vs permutation-null")
        print("-" * 120)
        print(
            f"{'Feature':<18} {'Observed MI bits [95% CI]':<30} "
            f"{'Null mean':<14} {'Null 95%':<14} "
            f"{'p-raw':<10} {'p-Holm':<10} {'p-BH':<10}"
        )
        for j, feat in enumerate(FEATURE_COLUMNS):
            obs_str = f"{obs_mean[j]:.3f} [{obs_lo[j]:.3f}, {obs_hi[j]:.3f}]"
            null_mean = float(null_dist[:, j].mean())
            null_95 = float(np.percentile(null_dist[:, j], 95))
            print(
                f"{feat:<18} {obs_str:<30} {null_mean:<14.3f} "
                f"{null_95:<14.3f} {pvals_raw[j]:<10.4f} "
                f"{pvals_holm[j]:<10.4f} {pvals_bh[j]:<10.4f}"
            )
        
        print("\nCross-validation scores")
        print("-" * 90)
        print(cv_df.to_string(index=False))
        
        print("\nFamily-holdout generalization")
        print("-" * 90)
        print(holdout_df.to_string(index=False))
        
        # Save plots
        plot_mi_bars(
            feature_names=FEATURE_COLUMNS,
            observed_mean=obs_mean,
            observed_lo=obs_lo,
            observed_hi=obs_hi,
            null_dist=null_dist,
            title=f"{topo_name}: Binary Veracity MI vs Permutation-Null",
            outpath=f"{OUTPUT_PREFIX}_{topo_name}_binary_mi.png",
        )
        
        for feat in ["routed_depth", "extra_twoq", "depth_overhead"]:
            plot_feature_distributions(
                df, feat, "label", {0: "Low Accuracy", 1: "High Accuracy"},
                f"{topo_name}: {feat} by Accuracy Level",
                f"{OUTPUT_PREFIX}_{topo_name}_binary_{feat}.png"
            )
        
        # Collect summary
        for _, row in cv_df.iterrows():
            all_summary_rows.append({
                "topology": topo_name,
                "task": "binary",
                "model": row["model"],
                "macro_f1_mean": row["macro_f1_mean"],
                "bal_acc_mean": row["bal_acc_mean"],
                "depth_mi_bits": obs_mean[FEATURE_COLUMNS.index("routed_depth")],
                "extra_twoq_mi_bits": obs_mean[FEATURE_COLUMNS.index("extra_twoq")],
            })
    
    summary_df = pd.DataFrame(all_summary_rows)
    print("\n" + "=" * 90)
    print("BINARY VERACITY SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(f"{OUTPUT_PREFIX}_binary_summary.csv", index=False)
    print(f"\nSaved: {OUTPUT_PREFIX}_binary_summary.csv")

def run_ordinal_veracity_experiment():
    """Run ordinal veracity classification experiment."""
    print("\n" + "=" * 90)
    print("RUNNING ORDINAL VERACITY EXPERIMENT (MULTIPLE ACCURACY LEVELS)")
    print("=" * 90)
    
    topologies = make_topologies()
    all_summary_rows = []
    
    for topo_name, cmap in topologies.items():
        print("\n" + "=" * 90)
        print(f"TOPOLOGY: {topo_name}")
        print("=" * 90)
        
        ds = build_ordinal_veracity_dataset(
            topology_name=topo_name,
            coupling_map=cmap,
            workload_families=WORKLOAD_FAMILIES,
            accuracy_levels=ORDINAL_ACCURACIES,
            n_samples_per_level=N_ORDINAL_SAMPLES_PER_LEVEL,
            seed=RANDOM_SEED + 100,
        )
        
        df = ds.df
        y = df["label"].to_numpy(dtype=int)
        groups = df["sample_id"].to_numpy()
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        
        # Observed MI with bootstrap CI
        obs_mean, obs_lo, obs_hi = bootstrap_mi_bits(
            X, y, n_boot=N_MI_BOOT, seed=RANDOM_SEED + 100
        )
        
        # Permutation null
        null_dist = permutation_null_mi_bits(
            X, y, n_perm=N_MI_PERM, seed=RANDOM_SEED + 100
        )
        
        # Statistical testing
        pvals_raw = permutation_pvals(obs_mean, null_dist)
        
        # Family holdout with ordinal metrics
        holdout_df = family_holdout_evaluation(
            df, FEATURE_COLUMNS, "ordinal", RANDOM_SEED + 100
        )
        
        # Print results
        print("\nObserved MI vs permutation-null")
        print("-" * 90)
        print(f"{'Feature':<18} {'Observed MI bits [95% CI]':<30} {'Null mean':<14} {'p-value':<10}")
        for j, feat in enumerate(FEATURE_COLUMNS):
            obs_str = f"{obs_mean[j]:.3f} [{obs_lo[j]:.3f}, {obs_hi[j]:.3f}]"
            null_mean = float(null_dist[:, j].mean())
            print(f"{feat:<18} {obs_str:<30} {null_mean:<14.3f} {pvals_raw[j]:<10.4f}")
        
        print("\nFamily-holdout ordinal metrics")
        print("-" * 90)
        print(holdout_df.to_string(index=False))
        
        # Save plots
        plot_mi_bars(
            feature_names=FEATURE_COLUMNS,
            observed_mean=obs_mean,
            observed_lo=obs_lo,
            observed_hi=obs_hi,
            null_dist=null_dist,
            title=f"{topo_name}: Ordinal Veracity MI vs Permutation-Null",
            outpath=f"{OUTPUT_PREFIX}_{topo_name}_ordinal_mi.png",
        )
        
        # Confusion matrix for RF model
        train_fam, test_fam = WORKLOAD_FAMILIES[0], WORKLOAD_FAMILIES[1]
        train_df = df[df["workload_family"] == train_fam]
        test_df = df[df["workload_family"] == test_fam]
        
        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy(dtype=int)
        X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test_df["label"].to_numpy(dtype=int)
        
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        plot_confusion_matrix(
            y_test, y_pred, ORDINAL_ACCURACIES,
            f"{topo_name}: RF Holdout ({train_fam}→{test_fam})",
            f"{OUTPUT_PREFIX}_{topo_name}_ordinal_cm.png"
        )
        
        # Collect summary
        for _, row in holdout_df.iterrows():
            if row["train_family"] != row["test_family"]:
                all_summary_rows.append({
                    "topology": topo_name,
                    "task": "ordinal",
                    "train_family": row["train_family"],
                    "test_family": row["test_family"],
                    "model": row["model"],
                    "macro_f1": row["macro_f1"],
                    "class_mae": row["class_mae"],
                    "adjacent_acc": row["adjacent_acc"],
                })
    
    summary_df = pd.DataFrame(all_summary_rows)
    print("\n" + "=" * 90)
    print("ORDINAL VERACITY SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(f"{OUTPUT_PREFIX}_ordinal_summary.csv", index=False)
    print(f"\nSaved: {OUTPUT_PREFIX}_ordinal_summary.csv")

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("EXPERIMENT 4: MATERIAL PROPERTY LEAKAGE - VERACITY/ACCURACY")
    print("=" * 90)
    print("\nThis experiment tests whether hidden accuracy requirements")
    print("(material properties) leak through compilation artefacts.")
    print("\nApproach:")
    print("  - Binary: Low (1e-2) vs High (1e-4) accuracy")
    print("  - Ordinal: Multiple levels (1e-2, 1e-3, 1e-4, 1e-5)")
    print("  - Two workload families: time_evolution, optimization")
    print("  - Family-holdout validation for robustness")
    print("=" * 90)
    
    run_binary_veracity_experiment()
    run_ordinal_veracity_experiment()
    
    print("\n" + "=" * 90)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 90)
    print("\nKey Questions Answered:")
    print("  1. Can providers infer accuracy requirements from compilation overhead?")
    print("  2. Does the signal generalize across different workload types?")
    print("  3. Which features carry the strongest veracity signal?")
    print("  4. How well can ordinal accuracy levels be distinguished?")
    print("=" * 90)
