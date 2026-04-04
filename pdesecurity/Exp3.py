# ==============================================================================
# EXPERIMENT 3: HARDWARE STABILITY & COUPLING-MODE ABLATION
# ------------------------------------------------------------------------------
# Purpose
#   Test which coupling modes carry the leakage signal, and how stable that
#   signal remains under increasing hardware drift.
#
# Assumes already defined elsewhere in your file:
#   - generate_pde_surrogate(...)
#   - generate_scale_surrogate(...)
#   - make_coupling_map(...)
#   - compile_and_extract_features(...)
# ==============================================================================

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from Exp1 import (
    generate_pde_surrogate,
    compile_and_extract_features
)
from Exp2 import (
    make_coupling_map,
generate_scale_surrogate
)


warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Transpilation verification helpers
# ------------------------------------------------------------------------------

def _verify_transpilation_batch(df: pd.DataFrame, label: str) -> None:
    """
    After a dataset is built, check that all samples passed distribution
    verification. Prints a summary and warns if any failed.

    The `compile_and_extract_features` functions in Exp1/Exp2 already store
    'verif_passed', 'verif_tvd', and 'verif_fidelity' in every row dict.
    This function aggregates those columns and surfaces any anomalies.
    """
    if "verif_passed" not in df.columns:
        print(f"  [{label}] No verification data found — "
              "compile_and_extract_features did not include verif_ columns.")
        return

    n_total = len(df)
    n_passed = int(df["verif_passed"].sum())
    n_failed = n_total - n_passed
    mean_tvd = df["verif_tvd"].mean()
    max_tvd = df["verif_tvd"].max()
    mean_fid = df["verif_fidelity"].mean()

    print(
        f"  [{label}] Transpilation verification: "
        f"{n_passed}/{n_total} passed | "
        f"mean TVD={mean_tvd:.4f}, max TVD={max_tvd:.4f}, "
        f"mean fidelity={mean_fid:.4f}"
    )

    if n_failed > 0:
        import warnings as _w
        _w.warn(
            f"[Exp3 – {label}] {n_failed}/{n_total} circuits failed "
            f"distribution verification (TVD threshold exceeded). "
            "Check basis gates, routing method, and optimization level.",
            RuntimeWarning,
            stacklevel=2,
        )

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

RANDOM_SEED = 42
rng_global = np.random.default_rng(RANDOM_SEED)

BOUNDARY_NUM_QUBITS = 8
BOUNDARY_SAMPLES_PER_CLASS = 120

SCALE_RESOLUTIONS = [4, 6, 8, 10, 12, 16]
SCALE_SAMPLES_PER_RES = 80

N_STEPS = 3
N_REPEATS = 10

# Dimensionless hardware-drift severity
DRIFT_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

TOPOLOGY_FAMILIES = ["line", "gridish"]
TEMPLATE_FAMILIES = ["A", "B"]

OUTPUT_PREFIX = "exp3_stability_ablation"

# Coupling-mode feature groups
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
    "Operational": [
        "transpile_ms",
        "sched_duration_ms",
        "idle_variance",
    ],
}

FEATURE_GROUPS["All"] = (
    FEATURE_GROUPS["Topological"]
    + FEATURE_GROUPS["Complexity"]
    + FEATURE_GROUPS["Operational"]
)

ORDERED_SCALE_LABELS = SCALE_RESOLUTIONS

# ------------------------------------------------------------------------------
# Feature augmentation and hardware drift
# ------------------------------------------------------------------------------

def augment_operational_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Derive operational/provider-facing features from compiled artefacts.
    These are not payload-semantic; they are execution-footprint surrogates.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    # Scheduling proxy: execution duration grows with depth and extra 2Q burden
    sched = (
        6.0 * out["routed_depth"].values
        + 8.0 * out["extra_twoq"].values
        + rng.normal(0.0, 10.0, len(out))
    )
    sched = np.maximum(1.0, sched)

    # Idle-variance proxy: routing turbulence and depth inflation create uneven schedules
    idle = (
        0.015 * out["routed_depth"].values
        + 0.10 * out["swap_equiv"].values
        + 0.06 * out["extra_depth"].values
        + rng.normal(0.0, 0.6, len(out))
    )
    idle = np.maximum(0.0, idle)

    out["sched_duration_ms"] = sched
    out["idle_variance"] = idle
    return out


def apply_hardware_drift(
    df: pd.DataFrame,
    severity: float,
    topology_family: str,
    seed: int,
) -> pd.DataFrame:
    """
    Apply drift to provider-visible artefacts only.

    Important:
    - Labels are never touched.
    - Drift magnitude depends on existing routing / depth burden, not on the hidden label.
    - This models the fact that already-hard-to-route jobs are more sensitive to calibration drift.
    """
    if severity == 0.0:
        return df.copy()

    rng = np.random.default_rng(seed)
    out = df.copy()

    topo_gain = 1.10 if topology_family == "line" else 1.00

    # Normalized workload burden signals
    route_load = out["extra_twoq"].values / max(out["extra_twoq"].max(), 1.0)
    depth_load = out["routed_depth"].values / max(out["routed_depth"].max(), 1.0)
    transpile_load = out["transpile_ms"].values / max(out["transpile_ms"].max(), 1.0)

    # Topological features
    out["swap_equiv"] = np.maximum(
        0.0,
        out["swap_equiv"].values
        * (
            1.0
            + topo_gain * severity * (0.18 * route_load)
            + rng.normal(0.0, 0.05 * severity, len(out))
        ),
    )

    out["extra_twoq"] = np.maximum(
        0.0,
        out["extra_twoq"].values
        * (
            1.0
            + topo_gain * severity * (0.20 * route_load)
            + rng.normal(0.0, 0.06 * severity, len(out))
        ),
    )

    out["swap_fraction"] = np.clip(
        out["swap_fraction"].values
        + topo_gain * severity * (0.04 * route_load)
        + rng.normal(0.0, 0.015 * severity, len(out)),
        0.0,
        1.0,
    )

    # Complexity features
    out["routed_depth"] = np.maximum(
        1.0,
        out["routed_depth"].values
        * (
            1.0
            + topo_gain * severity * (0.10 * route_load + 0.08 * depth_load)
            + rng.normal(0.0, 0.04 * severity, len(out))
        ),
    )

    out["extra_depth"] = np.maximum(
        0.0,
        out["extra_depth"].values
        * (
            1.0
            + topo_gain * severity * (0.15 * depth_load)
            + rng.normal(0.0, 0.06 * severity, len(out))
        ),
    )

    out["cx_fraction"] = np.clip(
        out["cx_fraction"].values
        + topo_gain * severity * (0.01 * route_load + 0.005 * depth_load)
        + rng.normal(0.0, 0.004 * severity, len(out)),
        0.0,
        1.0,
    )

    # Operational features
    out["transpile_ms"] = np.maximum(
        1.0,
        out["transpile_ms"].values
        * (
            1.0
            + topo_gain * severity * (0.20 + 0.10 * transpile_load + 0.10 * route_load)
            + rng.normal(0.0, 0.10 * severity, len(out))
        ),
    )

    out["sched_duration_ms"] = np.maximum(
        1.0,
        out["sched_duration_ms"].values
        * (
            1.0
            + topo_gain * severity * (0.18 + 0.10 * depth_load)
            + rng.normal(0.0, 0.08 * severity, len(out))
        ),
    )

    out["idle_variance"] = np.maximum(
        0.0,
        out["idle_variance"].values
        + topo_gain * severity * (0.10 * route_load + 0.08 * depth_load)
        + rng.normal(0.0, 0.10 * severity, len(out)),
    )

    # Recompute derived ratios from perturbed observables
    out["depth_overhead"] = out["routed_depth"].values / np.maximum(out["logical_depth"].values, 1.0)
    out["twoq_overhead"] = (
        (out["extra_twoq"].values + out["logical_twoq"].values)
        / np.maximum(out["logical_twoq"].values, 1.0)
    )

    return out

# ------------------------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------------------------

def build_boundary_dataset(
    topology_family: str,
    n_samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """
    Boundary-inference dataset: hidden attribute is boundary topology.
    """
    rng = np.random.default_rng(seed)
    rows = []
    cmap = make_coupling_map(BOUNDARY_NUM_QUBITS, topology_family)

    for sample_idx in range(n_samples_per_class):
        logical_seed = int(rng.integers(0, 10_000_000))
        transpile_seed_dir = int(rng.integers(0, 10_000_000))
        transpile_seed_per = int(rng.integers(0, 10_000_000))

        qc_dir = generate_pde_surrogate(
            num_qubits=BOUNDARY_NUM_QUBITS,
            boundary_condition="dirichlet",
            n_steps=N_STEPS,
            seed=logical_seed,
        )
        qc_per = generate_pde_surrogate(
            num_qubits=BOUNDARY_NUM_QUBITS,
            boundary_condition="periodic",
            n_steps=N_STEPS,
            seed=logical_seed,
        )

        feat_dir = compile_and_extract_features(qc_dir, cmap, transpile_seed_dir)
        feat_per = compile_and_extract_features(qc_per, cmap, transpile_seed_per)

        feat_dir.update({
            "task": "boundary",
            "label": 0,
            "label_name": "dirichlet",
            "topology_family": topology_family,
            "sample_id": sample_idx,
            "logical_seed": logical_seed,
        })
        feat_per.update({
            "task": "boundary",
            "label": 1,
            "label_name": "periodic",
            "topology_family": topology_family,
            "sample_id": sample_idx,
            "logical_seed": logical_seed,
        })

        rows.extend([feat_dir, feat_per])

    df = pd.DataFrame(rows)
    _verify_transpilation_batch(df, f"boundary/{topology_family}")
    df = augment_operational_features(df, seed + 111)
    return df


def build_scale_dataset(
    topology_family: str,
    resolutions: List[int],
    n_samples_per_res: int,
    seed: int,
) -> pd.DataFrame:
    """
    Scale-inference dataset: hidden attribute is resolution N.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for template_family in TEMPLATE_FAMILIES:
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

                feat = compile_and_extract_features(qc, cmap, transpile_seed)
                feat.update({
                    "task": "scale",
                    "label": N,
                    "label_name": str(N),
                    "N": N,
                    "template_family": template_family,
                    "topology_family": topology_family,
                    "sample_id": sample_idx,
                    "logical_seed": logical_seed,
                })
                rows.append(feat)

    df = pd.DataFrame(rows)
    _verify_transpilation_batch(df, f"scale/{topology_family}")
    df = augment_operational_features(df, seed + 222)
    return df

# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray, ordered_labels: List[int]) -> Dict[str, float]:
    label_to_idx = {lab: i for i, lab in enumerate(ordered_labels)}
    true_idx = np.array([label_to_idx[v] for v in y_true], dtype=int)
    pred_idx = np.array([label_to_idx[v] for v in y_pred], dtype=int)

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "class_mae": float(np.mean(np.abs(true_idx - pred_idx))),
        "adjacent_acc": float(np.mean(np.abs(true_idx - pred_idx) <= 1)),
    }

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

def plot_drift_curves(
    results_df: pd.DataFrame,
    task_name: str,
    topology_family: str,
    metric: str,
    outpath: str,
) -> None:
    sub = results_df[
        (results_df["task"] == task_name)
        & (results_df["topology_family"] == topology_family)
    ]

    plt.figure(figsize=(8, 5))
    for group_name in FEATURE_GROUPS.keys():
        g = sub[sub["feature_group"] == group_name].sort_values("drift")
        plt.plot(g["drift"], g[f"{metric}_mean"], marker="o", label=group_name)
        plt.fill_between(
            g["drift"],
            g[f"{metric}_mean"] - g[f"{metric}_std"],
            g[f"{metric}_mean"] + g[f"{metric}_std"],
            alpha=0.15,
        )

    plt.xlabel("Hardware-drift severity")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{topology_family}: {task_name} | {metric}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    outpath: str,
) -> None:
    corr_cols = list(dict.fromkeys(FEATURE_GROUPS["All"]))
    corr = df[corr_cols].corr(method="spearman")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate_task_under_drift(
    df_clean: pd.DataFrame,
    task_name: str,
    topology_family: str,
    label_col: str,
    drift_levels: List[float],
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for rep in range(n_repeats):
        rep_seed = seed + 1000 * rep
        df_rep = df_clean.copy()

        X_full = df_rep[FEATURE_GROUPS["All"]]
        y = df_rep[label_col].values

        train_idx, test_idx = train_test_split(
            np.arange(len(df_rep)),
            test_size=0.30,
            stratify=y,
            random_state=rep_seed,
        )

        train_clean = df_rep.iloc[train_idx].copy()
        test_clean = df_rep.iloc[test_idx].copy()

        for drift in drift_levels:
            test_drift = apply_hardware_drift(
                test_clean,
                severity=drift,
                topology_family=topology_family,
                seed=rep_seed + int(100 * drift) + 17,
            )

            for feature_group, cols in FEATURE_GROUPS.items():
                clf = RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    random_state=rep_seed,
                    class_weight="balanced",
                )

                clf.fit(train_clean[cols], train_clean[label_col].values)
                y_pred = clf.predict(test_drift[cols])

                if task_name == "boundary":
                    rows.append({
                        "task": task_name,
                        "topology_family": topology_family,
                        "repeat": rep,
                        "drift": drift,
                        "feature_group": feature_group,
                        "macro_f1": float(f1_score(test_drift[label_col].values, y_pred, average="macro")),
                        "bal_acc": float(balanced_accuracy_score(test_drift[label_col].values, y_pred)),
                    })
                else:
                    mets = ordinal_metrics(
                        test_drift[label_col].values,
                        y_pred,
                        ORDERED_SCALE_LABELS,
                    )
                    rows.append({
                        "task": task_name,
                        "topology_family": topology_family,
                        "repeat": rep,
                        "drift": drift,
                        "feature_group": feature_group,
                        **mets,
                    })

    return pd.DataFrame(rows)


def summarise_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in raw_df.columns if c not in {
        "task", "topology_family", "repeat", "drift", "feature_group"
    }]

    grouped = raw_df.groupby(["task", "topology_family", "feature_group", "drift"], as_index=False)

    summary_rows = []
    for _, g in grouped:
        row = {
            "task": g["task"].iloc[0],
            "topology_family": g["topology_family"].iloc[0],
            "feature_group": g["feature_group"].iloc[0],
            "drift": g["drift"].iloc[0],
        }
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

# ------------------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------------------

def run_stability_ablation_experiment():
    print("\n" + "=" * 90)
    print("RUNNING EXPERIMENT 3: HARDWARE STABILITY & COUPLING-MODE ABLATION")
    print("=" * 90)

    all_raw = []

    for topology_family in TOPOLOGY_FAMILIES:
        print("\n" + "=" * 90)
        print(f"TOPOLOGY FAMILY: {topology_family}")
        print("=" * 90)

        # Build clean datasets
        df_boundary = build_boundary_dataset(
            topology_family=topology_family,
            n_samples_per_class=BOUNDARY_SAMPLES_PER_CLASS,
            seed=RANDOM_SEED + 11,
        )

        df_scale = build_scale_dataset(
            topology_family=topology_family,
            resolutions=SCALE_RESOLUTIONS,
            n_samples_per_res=SCALE_SAMPLES_PER_RES,
            seed=RANDOM_SEED + 22,
        )

        # Save correlation heatmaps on clean day
        plot_correlation_heatmap(
            df_boundary,
            title=f"{topology_family}: boundary-task feature correlations (clean day)",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_corr_boundary.png",
        )
        plot_correlation_heatmap(
            df_scale,
            title=f"{topology_family}: scale-task feature correlations (clean day)",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_corr_scale.png",
        )

        # Evaluate drift + ablation
        raw_boundary = evaluate_task_under_drift(
            df_clean=df_boundary,
            task_name="boundary",
            topology_family=topology_family,
            label_col="label",
            drift_levels=DRIFT_LEVELS,
            n_repeats=N_REPEATS,
            seed=RANDOM_SEED + 100,
        )

        raw_scale = evaluate_task_under_drift(
            df_clean=df_scale,
            task_name="scale",
            topology_family=topology_family,
            label_col="label",
            drift_levels=DRIFT_LEVELS,
            n_repeats=N_REPEATS,
            seed=RANDOM_SEED + 200,
        )

        all_raw.extend([raw_boundary, raw_scale])

    raw_df = pd.concat(all_raw, axis=0, ignore_index=True)
    summary_df = summarise_results(raw_df)

    # Print tables
    print("\n" + "=" * 90)
    print("ABLATION & STABILITY RESULTS (Mean ± Std)")
    print("=" * 90)

    for topology_family in TOPOLOGY_FAMILIES:
        for task_name in ["boundary", "scale"]:
            print(f"\nTOPOLOGY={topology_family.upper()} | TASK={task_name.upper()}")
            print("-" * 90)

            sub = summary_df[
                (summary_df["topology_family"] == topology_family)
                & (summary_df["task"] == task_name)
            ].copy()

            if task_name == "boundary":
                for fg in FEATURE_GROUPS.keys():
                    g = sub[sub["feature_group"] == fg].sort_values("drift")
                    vals = [
                        f"{row['macro_f1_mean']:.2f}±{row['macro_f1_std']:.02f}"
                        for _, row in g.iterrows()
                    ]
                    print(f"{fg:<12} | " + " | ".join([f"d={d:<3}: {v}" for d, v in zip(g['drift'], vals)]))
            else:
                for fg in FEATURE_GROUPS.keys():
                    g = sub[sub["feature_group"] == fg].sort_values("drift")
                    vals = [
                        f"F1={row['macro_f1_mean']:.2f}±{row['macro_f1_std']:.02f}, "
                        f"MAE={row['class_mae_mean']:.2f}, Adj={row['adjacent_acc_mean']:.2f}"
                        for _, row in g.iterrows()
                    ]
                    print(f"{fg:<12} | " + " | ".join([f"d={d:<3}: {v}" for d, v in zip(g['drift'], vals)]))

    # Save plots
    for topology_family in TOPOLOGY_FAMILIES:
        plot_drift_curves(
            summary_df,
            task_name="boundary",
            topology_family=topology_family,
            metric="macro_f1",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_boundary_f1.png",
        )
        plot_drift_curves(
            summary_df,
            task_name="boundary",
            topology_family=topology_family,
            metric="bal_acc",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_boundary_balacc.png",
        )
        plot_drift_curves(
            summary_df,
            task_name="scale",
            topology_family=topology_family,
            metric="macro_f1",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_scale_f1.png",
        )
        plot_drift_curves(
            summary_df,
            task_name="scale",
            topology_family=topology_family,
            metric="class_mae",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_scale_mae.png",
        )
        plot_drift_curves(
            summary_df,
            task_name="scale",
            topology_family=topology_family,
            metric="adjacent_acc",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_scale_adjacc.png",
        )

    raw_df.to_csv(f"{OUTPUT_PREFIX}_raw.csv", index=False)
    summary_df.to_csv(f"{OUTPUT_PREFIX}_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("CONCLUSION TEMPLATE:")
    print("1. Boundary leakage should be strongest under Topological and All feature groups.")
    print("2. Scale leakage should be strongest under Complexity and All feature groups.")
    print("3. Operational features should be weaker alone, but should degrade more slowly under drift.")
    print("4. Realistic drift should reduce performance gradually, not collapse it immediately.")
    print("5. If the ordering holds across both topologies, Section V's coupling-mode story is supported.")
    print("=" * 90)
    print(f"Saved raw results to: {OUTPUT_PREFIX}_raw.csv")
    print(f"Saved summary results to: {OUTPUT_PREFIX}_summary.csv")
    print("Saved drift-curve plots and correlation heatmaps as PNG files.")

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run_stability_ablation_experiment()