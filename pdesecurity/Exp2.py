# ==============================================================================
# EXPERIMENT 2: COMPLEXITY SCALING AND ORDINAL INFERENCE
# ------------------------------------------------------------------------------
# Purpose
#   Test whether hidden scientific scale (resolution N) leaves a robust,
#   recoverable, and ordinally structured signature in provider-visible
#   compilation artefacts.
#
# Main design improvements
#   1) Uses PDE-like structured surrogates rather than toy depth formulas.
#   2) Uses two workload-template families to test template/solver robustness.
#   3) Uses permutation-null MI, not just classifier scores.
#   4) Uses bootstrapped scaling exponents for routed depth and extra 2Q overhead.
#   5) Uses family-holdout ordinal inference to test "algorithm-robust" scale leakage.
#   6) Reports ordinal metrics: macro-F1, balanced accuracy, class-MAE,
#      and adjacent-class accuracy.
# ==============================================================================

import time
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

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

    Returns
    -------
    dict with keys:
      'tvd'      – Total Variation Distance (0 = identical distributions)
      'fidelity' – State fidelity |<ψ_orig|ψ_trans>|² (1 = identical states)
      'passed'   – True when TVD < tvd_threshold
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, state_fidelity

    def _strip_measurements(qc: QuantumCircuit) -> QuantumCircuit:
        stripped = QuantumCircuit(qc.num_qubits)
        for inst, qargs, cargs in qc.data:
            if inst.name not in {"measure", "reset", "barrier"}:
                stripped.append(inst, qargs, cargs)
        return stripped

    sv_orig = Statevector.from_instruction(_strip_measurements(original))
    sv_trans = Statevector.from_instruction(_strip_measurements(transpiled))

    probs_orig = sv_orig.probabilities()
    probs_trans = sv_trans.probabilities()

    tvd = float(0.5 * np.sum(np.abs(probs_orig - probs_trans)))
    fidelity = float(state_fidelity(sv_orig, sv_trans))
    passed = tvd < tvd_threshold

    if not passed:
        warnings.warn(
            f"[verify_transpilation] TVD={tvd:.4f} exceeds threshold {tvd_threshold}. "
            f"Fidelity={fidelity:.4f}. Transpilation may have altered circuit semantics.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {"tvd": tvd, "fidelity": fidelity, "passed": passed}

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

RANDOM_SEED = 42
rng_global = np.random.default_rng(RANDOM_SEED)

RESOLUTIONS = [4, 6, 8, 10, 12, 16]
N_SAMPLES_PER_RES = 120
N_STEPS = 3
N_MI_BOOT = 300
N_MI_PERM = 500
N_REPEATS = 2

OUTPUT_PREFIX = "exp2_scale_leakage"

# Provider-visible compilation overheads
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

# Trivial "size" proxies (The Null Baseline)
LOGICAL_COLUMNS = [
    "logical_depth",
    "logical_twoq",
    "logical_total_ops"
]

SCALING_FEATURES = [
    "routed_depth",
    "extra_twoq",
]

TEMPLATE_FAMILIES = ["A", "B"]
TOPOLOGY_FAMILIES = ["line", "gridish"]

# ------------------------------------------------------------------------------
# Grid / topology helpers
# ------------------------------------------------------------------------------

def infer_grid_shape(num_qubits: int) -> Tuple[int, int]:
    """
    Choose a factorisation as close to square as possible.
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

def line_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]

def gridish_edges(n: int) -> List[Tuple[int, int]]:
    """
    Near-square 2D grid topology for n qubits.
    """
    rows, cols = infer_grid_shape(n)
    edges = []
    for r in range(rows):
        for c in range(cols - 1):
            edges.append((grid_index(r, c, cols), grid_index(r, c + 1, cols)))
    for r in range(rows - 1):
        for c in range(cols):
            edges.append((grid_index(r, c, cols), grid_index(r + 1, c, cols)))
    return edges

def make_coupling_map(n: int, topology_family: str) -> CouplingMap:
    if topology_family == "line":
        return CouplingMap(line_edges(n))
    elif topology_family == "gridish":
        return CouplingMap(gridish_edges(n))
    raise ValueError(f"Unknown topology_family={topology_family}")

# ------------------------------------------------------------------------------
# Logical surrogate generation
# ------------------------------------------------------------------------------

def count_two_qubit_ops(qc: QuantumCircuit) -> int:
    return sum(1 for inst, qargs, _ in qc.data if len(qargs) == 2)

def count_total_ops(qc: QuantumCircuit) -> int:
    total = 0
    for inst, _, _ in qc.data:
        if inst.name not in {"barrier"}:
            total += 1
    return total

def apply_local_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    for q in range(qc.num_qubits):
        qc.rz(rng.uniform(0, 2 * np.pi), q)
        qc.rx(rng.uniform(0, 2 * np.pi), q)

def build_partitions(rows: int, cols: int) -> Dict[str, List[Tuple[int, int]]]:
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

def unique_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out = []
    for a, b in edges:
        e = tuple(sorted((a, b)))
        if a != b and e not in seen:
            seen.add(e)
            out.append(e)
    return out

def choose_scale_edges(
    rows: int,
    cols: int,
    template_family: str,
    step_idx: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Two logical template families with similar scale dependence but different local schedules.
    This lets us test whether scale is recoverable beyond one specific circuit template.
    """
    parts = build_partitions(rows, cols)
    names = ["h_even", "h_odd", "v_even", "v_odd"]

    if template_family == "A":
        primary = names[step_idx % 4]
        secondary = names[(step_idx + 1) % 4]
        sec_frac = 0.35
    elif template_family == "B":
        primary = names[(2 * step_idx) % 4]
        secondary = names[(2 * step_idx + 3) % 4]
        sec_frac = 0.50
    else:
        raise ValueError(f"Unknown template_family={template_family}")

    primary_edges = list(parts[primary])
    secondary_pool = list(parts[secondary])
    rng.shuffle(primary_edges)
    rng.shuffle(secondary_pool)

    n_secondary = int(round(sec_frac * len(secondary_pool)))
    chosen = unique_edges(primary_edges + secondary_pool[:n_secondary])
    return chosen

def apply_coupling_block(
    qc: QuantumCircuit,
    edges: List[Tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    """
    Weighted 2Q coupling surrogate.
    """
    for a, b in edges:
        theta = rng.uniform(0.15, 1.25) * np.pi
        qc.cx(a, b)
        qc.rz(theta, b)
        qc.cx(a, b)

def generate_scale_surrogate(
    num_qubits: int,
    template_family: str,
    n_steps: int,
    seed: int,
) -> QuantumCircuit:
    """
    Dirichlet-style PDE surrogate for a given scale N.
    Hidden scientific attribute here is scale, not boundary regime.
    """
    rng = np.random.default_rng(seed)
    rows, cols = infer_grid_shape(num_qubits)
    qc = QuantumCircuit(num_qubits)

    apply_local_layer(qc, rng)

    for step_idx in range(n_steps):
        apply_local_layer(qc, rng)
        edges = choose_scale_edges(rows, cols, template_family, step_idx, rng)
        apply_coupling_block(qc, edges, rng)

        # small local mixing
        for q in range(num_qubits):
            qc.rz(rng.uniform(-0.25, 0.25) * np.pi, q)

    apply_local_layer(qc, rng)
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
    # Distribution verification
    # ------------------------------------------------------------------
    verif = verify_transpilation(qc, tqc, seed=seed_transpiler)

    ops = tqc.count_ops()
    explicit_swap_count = float(ops.get("swap", 0))
    cx_count = float(ops.get("cx", 0))
    routed_depth = float(tqc.depth())

    total_ops = float(count_total_ops(tqc))
    total_twoq = float(count_two_qubit_ops(tqc))

    extra_twoq = max(0.0, total_twoq - logical_twoq)
    swap_equiv = extra_twoq / 3.0
    extra_depth = max(0.0, routed_depth - logical_depth)

    return {
        "swap_equiv": swap_equiv,
        "swap_fraction": extra_twoq / max(total_twoq, 1.0),
        "cx_fraction": cx_count / max(total_ops, 1.0),
        "routed_depth": routed_depth,
        "depth_overhead": routed_depth / max(logical_depth, 1.0),
        "twoq_overhead": total_twoq / max(logical_twoq, 1.0),
        "extra_twoq": extra_twoq,
        "extra_depth": extra_depth,
        "transpile_ms": transpile_ms,
        "logical_depth": logical_depth,
        "logical_twoq": logical_twoq,
        "logical_total_ops": logical_total_ops,
        "explicit_swap_count": explicit_swap_count,
        "verif_tvd": verif["tvd"],
        "verif_fidelity": verif["fidelity"],
        "verif_passed": float(verif["passed"]),
    }

# ------------------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------------------

def bootstrap_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    mis = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        mi_nats = mutual_info_classif(Xb, yb, discrete_features=False, random_state=seed)
        mis.append(mi_nats / np.log(2.0))
    mis = np.asarray(mis)
    return (
        mis.mean(axis=0),
        np.percentile(mis, 2.5, axis=0),
        np.percentile(mis, 97.5, axis=0),
    )

def permutation_null_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    null_dist = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        mi_nats = mutual_info_classif(X, yp, discrete_features=False, random_state=seed)
        null_dist.append(mi_nats / np.log(2.0))
    return np.asarray(null_dist)

def permutation_pvals(observed: np.ndarray, null_dist: np.ndarray) -> np.ndarray:
    pvals = []
    for j in range(len(observed)):
        p = (1.0 + np.sum(null_dist[:, j] >= observed[j])) / (1.0 + len(null_dist))
        pvals.append(p)
    return np.asarray(pvals)

def bootstrap_scaling_exponent(
    N_vals: np.ndarray,
    feat_vals: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Fit log(feature) = k*log(N) + c using bootstrap over samples.
    """
    rng = np.random.default_rng(seed)
    logN = np.log(np.asarray(N_vals, dtype=float))
    logF = np.log(np.asarray(feat_vals, dtype=float) + 1e-8)

    ks = []
    n = len(logN)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x = logN[idx]
        y = logF[idx]
        k, _ = np.polyfit(x, y, 1)
        ks.append(k)
    ks = np.asarray(ks)
    return float(ks.mean()), float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))

def ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ordered_labels: List[int],
) -> Dict[str, float]:
    label_to_idx = {lab: i for i, lab in enumerate(ordered_labels)}
    true_idx = np.array([label_to_idx[v] for v in y_true], dtype=int)
    pred_idx = np.array([label_to_idx[v] for v in y_pred], dtype=int)

    class_mae = float(np.mean(np.abs(true_idx - pred_idx)))
    adjacent_acc = float(np.mean(np.abs(true_idx - pred_idx) <= 1))

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "class_mae": class_mae,
        "adjacent_acc": adjacent_acc,
    }

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
    null_hi = np.percentile(null_dist, 95, axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(x, observed_mean, yerr=yerr, capsize=4)
    plt.plot(x, null_mean, marker="o", linestyle="--", label="Permutation-null mean")
    plt.fill_between(x, 0, null_hi, alpha=0.15, label="Permutation-null 95% upper band")
    plt.xticks(x, feature_names, rotation=25)
    plt.ylabel("Mutual information (bits)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_loglog_scaling(
    df: pd.DataFrame,
    topology_family: str,
    feature: str,
    family_stats: Dict[str, Tuple[float, float, float]],
    outpath: str,
) -> None:
    plt.figure(figsize=(8, 6))

    colors = {"A": "tab:blue", "B": "tab:red"}
    for fam in TEMPLATE_FAMILIES:
        sub = df[(df["topology_family"] == topology_family) & (df["template_family"] == fam)]
        x = np.log(sub["N"].values)
        y = np.log(sub[feature].values + 1e-8)

        k, lo, hi = family_stats[fam]
        sns.regplot(
            x=x,
            y=y,
            scatter_kws={"alpha": 0.30, "color": colors[fam]},
            line_kws={"color": colors[fam], "label": f"Family {fam}: k={k:.2f} [{lo:.2f},{hi:.2f}]"},
        )

    plt.xlabel("log(Resolution N)")
    plt.ylabel(f"log({feature})")
    plt.title(f"{topology_family}: scaling of {feature}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    title: str,
    outpath: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted scale N")
    plt.ylabel("True scale N")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------------------------

@dataclass
class ScaleDataset:
    df: pd.DataFrame

def build_scale_dataset(
    resolutions: List[int],
    n_samples_per_res: int,
    topology_families: List[str],
    template_families: List[str],
    seed: int = 42,
) -> ScaleDataset:
    rng = np.random.default_rng(seed)
    rows = []

    for topology_family in topology_families:
        for template_family in template_families:
            for N in resolutions:
                cmap = make_coupling_map(N, topology_family)
                for sample_idx in range(n_samples_per_res):
                    logical_seed = int(rng.integers(0, 10_000_000))
                    transpile_seed = int(rng.integers(0, 10_000_000))

                    qc = generate_scale_surrogate(
                        num_qubits=N,
                        template_family=template_family,
                        n_steps=N_STEPS,
                        seed=logical_seed,
                    )

                    feats = compile_and_extract_features(
                        qc=qc,
                        coupling_map=cmap,
                        seed_transpiler=transpile_seed,
                    )

                    feats.update({
                        "N": N,
                        "sample_id": sample_idx,
                        "template_family": template_family,
                        "topology_family": topology_family,
                        "logical_seed": logical_seed,
                        "transpile_seed": transpile_seed,
                    })
                    rows.append(feats)

    return ScaleDataset(df=pd.DataFrame(rows))

# ------------------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------------------

def run_scaling_experiment():
    print("\n" + "=" * 90)
    print("RUNNING EXPERIMENT 2: COMPLEXITY SCALING & ORDINAL INFERENCE")
    print("=" * 90)

    ds = build_scale_dataset(
        resolutions=RESOLUTIONS,
        n_samples_per_res=N_SAMPLES_PER_RES,
        topology_families=TOPOLOGY_FAMILIES,
        template_families=TEMPLATE_FAMILIES,
        seed=RANDOM_SEED,
    )
    df = ds.df

    all_summary_rows = []

    for topology_family in TOPOLOGY_FAMILIES:
        print("\n" + "=" * 90)
        print(f"TOPOLOGY FAMILY: {topology_family}")
        print("=" * 90)

        topo_df = df[df["topology_family"] == topology_family].copy()

        # ----------------------------------------------------------------------
        # Part A: Information-theoretic scale leakage
        # ----------------------------------------------------------------------
        X = topo_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = topo_df["N"].to_numpy(dtype=int)

        obs_mean, obs_lo, obs_hi = bootstrap_mi_bits(
            X, y, n_boot=N_MI_BOOT, seed=RANDOM_SEED
        )
        null_dist = permutation_null_mi_bits(
            X, y, n_perm=N_MI_PERM, seed=RANDOM_SEED
        )
        pvals = permutation_pvals(obs_mean, null_dist)

        print("\nObserved MI vs permutation-null")
        print("-" * 90)
        print(f"{'Feature':<18} {'Observed MI bits [95% CI]':<30} {'Null mean':<14} {'Null 95%':<14} {'p-value':<10}")
        for j, feat in enumerate(FEATURE_COLUMNS):
            obs_str = f"{obs_mean[j]:.3f} [{obs_lo[j]:.3f}, {obs_hi[j]:.3f}]"
            null_mean = float(null_dist[:, j].mean())
            null_95 = float(np.percentile(null_dist[:, j], 95))
            print(f"{feat:<18} {obs_str:<30} {null_mean:<14.3f} {null_95:<14.3f} {pvals[j]:<10.4f}")

        plot_mi_bars(
            FEATURE_COLUMNS,
            obs_mean,
            obs_lo,
            obs_hi,
            null_dist,
            title=f"{topology_family}: scale MI vs permutation-null",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_mi.png",
        )

        # ----------------------------------------------------------------------
        # Part B: Scaling-law invariance across template families
        # ----------------------------------------------------------------------
        print("\nBootstrapped scaling exponents")
        print("-" * 90)

        family_stats_by_feature = {}
        for feat in SCALING_FEATURES:
            family_stats = {}
            print(f"\nFeature: {feat}")
            for fam in TEMPLATE_FAMILIES:
                sub = topo_df[topo_df["template_family"] == fam]
                k, lo, hi = bootstrap_scaling_exponent(
                    N_vals=sub["N"].values,
                    feat_vals=sub[feat].values,
                    n_boot=N_MI_BOOT,
                    seed=RANDOM_SEED,
                )
                family_stats[fam] = (k, lo, hi)
                print(f"  Family {fam}: k = {k:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")

            family_stats_by_feature[feat] = family_stats
            plot_loglog_scaling(
                topo_df,
                topology_family,
                feat,
                family_stats,
                outpath=f"{OUTPUT_PREFIX}_{topology_family}_loglog_{feat}.png",
            )

            # ----------------------------------------------------------------------
            # Part C: Family-holdout ordinal inference (Logical Baseline vs Routed)
            # ----------------------------------------------------------------------
            print("\nFamily-holdout ordinal inference (Logical Baseline vs Routed Overheads)")
            print("-" * 90)

            train_A = topo_df[topo_df["template_family"] == "A"]
            test_B = topo_df[topo_df["template_family"] == "B"]
            train_B = topo_df[topo_df["template_family"] == "B"]
            test_A = topo_df[topo_df["template_family"] == "A"]

            # 1. Trivial Logical Baseline (Size only)
            XA_tr_log = train_A[LOGICAL_COLUMNS].to_numpy(dtype=float)
            XB_te_log = test_B[LOGICAL_COLUMNS].to_numpy(dtype=float)
            XB_tr_log = train_B[LOGICAL_COLUMNS].to_numpy(dtype=float)
            XA_te_log = test_A[LOGICAL_COLUMNS].to_numpy(dtype=float)

            # 2. Routed Overhead Attacker (Physical leakage)
            XA_tr_rout = train_A[FEATURE_COLUMNS].to_numpy(dtype=float)
            XB_te_rout = test_B[FEATURE_COLUMNS].to_numpy(dtype=float)
            XB_tr_rout = train_B[FEATURE_COLUMNS].to_numpy(dtype=float)
            XA_te_rout = test_A[FEATURE_COLUMNS].to_numpy(dtype=float)

            yA_tr = train_A["N"].to_numpy(dtype=int)
            yB_te = test_B["N"].to_numpy(dtype=int)
            yB_tr = train_B["N"].to_numpy(dtype=int)
            yA_te = test_A["N"].to_numpy(dtype=int)

            models = {
                "logreg": Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_SEED)),
                ]),
                "rf": RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            }

            for model_name, model in models.items():
                print(f"\nModel: {model_name.upper()}")

                # --- EVALUATE LOGICAL BASELINE ---
                model.fit(XA_tr_log, yA_tr)
                y_pred_B_log = model.predict(XB_te_log)
                met_AB_log = ordinal_metrics(yB_te, y_pred_B_log, RESOLUTIONS)

                model.fit(XB_tr_log, yB_tr)
                y_pred_A_log = model.predict(XA_te_log)
                met_BA_log = ordinal_metrics(yA_te, y_pred_A_log, RESOLUTIONS)

                # --- EVALUATE ROUTED ATTACKER ---
                model.fit(XA_tr_rout, yA_tr)
                y_pred_B_rout = model.predict(XB_te_rout)
                met_AB_rout = ordinal_metrics(yB_te, y_pred_B_rout, RESOLUTIONS)

                model.fit(XB_tr_rout, yB_tr)
                y_pred_A_rout = model.predict(XA_te_rout)
                met_BA_rout = ordinal_metrics(yA_te, y_pred_A_rout, RESOLUTIONS)

                # --- CALCULATE DELTAS ---
                delta_f1_AB = met_AB_rout['macro_f1'] - met_AB_log['macro_f1']
                delta_adj_AB = met_AB_rout['adjacent_acc'] - met_AB_log['adjacent_acc']

                delta_f1_BA = met_BA_rout['macro_f1'] - met_BA_log['macro_f1']
                delta_adj_BA = met_BA_rout['adjacent_acc'] - met_BA_log['adjacent_acc']

                # --- PRINT RESULTS ---
                print(
                    f"  A -> B (Logical Baseline) | macro-F1={met_AB_log['macro_f1']:.3f}, adjacent-acc={met_AB_log['adjacent_acc']:.3f}")
                print(
                    f"  A -> B (Routed Attacker)  | macro-F1={met_AB_rout['macro_f1']:.3f}, adjacent-acc={met_AB_rout['adjacent_acc']:.3f}")
                print(f"         Δ Leakage Gained   | Δ F1 = {delta_f1_AB:+.3f}, Δ Adj-Acc = {delta_adj_AB:+.3f}\n")

                print(
                    f"  B -> A (Logical Baseline) | macro-F1={met_BA_log['macro_f1']:.3f}, adjacent-acc={met_BA_log['adjacent_acc']:.3f}")
                print(
                    f"  B -> A (Routed Attacker)  | macro-F1={met_BA_rout['macro_f1']:.3f}, adjacent-acc={met_BA_rout['adjacent_acc']:.3f}")
                print(f"         Δ Leakage Gained   | Δ F1 = {delta_f1_BA:+.3f}, Δ Adj-Acc = {delta_adj_BA:+.3f}")

                avg_f1 = 0.5 * (met_AB_rout["macro_f1"] + met_BA_rout["macro_f1"])
                avg_bal = 0.5 * (met_AB_rout["bal_acc"] + met_BA_rout["bal_acc"])
                avg_mae = 0.5 * (met_AB_rout["class_mae"] + met_BA_rout["class_mae"])
                avg_adj = 0.5 * (met_AB_rout["adjacent_acc"] + met_BA_rout["adjacent_acc"])

                all_summary_rows.append({
                    "topology_family": topology_family,
                    "model": model_name,
                    "avg_macro_f1": avg_f1,
                    "avg_bal_acc": avg_bal,
                    "avg_class_mae": avg_mae,
                    "avg_adjacent_acc": avg_adj,
                    "mean_routed_depth_mi_bits": obs_mean[FEATURE_COLUMNS.index("routed_depth")],
                    "mean_twoq_overhead_mi_bits": obs_mean[FEATURE_COLUMNS.index("twoq_overhead")],
                })

                # Save one confusion matrix per direction for RF only
                if model_name == "rf":
                    plot_confusion_matrix(
                        yB_te, y_pred_B_rout, RESOLUTIONS,
                        title=f"{topology_family}: RF family-holdout A→B (Routed)",
                        outpath=f"{OUTPUT_PREFIX}_{topology_family}_cm_rf_A_to_B.png",
                    )
                    plot_confusion_matrix(
                        yA_te, y_pred_A_rout, RESOLUTIONS,
                        title=f"{topology_family}: RF family-holdout B→A (Routed)",
                        outpath=f"{OUTPUT_PREFIX}_{topology_family}_cm_rf_B_to_A.png",
                    )

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run_scaling_experiment()