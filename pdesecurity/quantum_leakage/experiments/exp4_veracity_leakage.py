"""
Experiment 4: Veracity / accuracy leakage.

Tests whether hidden accuracy requirements leak through provider-visible
compilation artefacts.
"""

from __future__ import annotations

import pandas as pd

from analysis.classifiers import family_holdout_evaluation, grouped_cv_classifier_scores
from analysis.mi import (
    bootstrap_mi_bits,
    compute_observed_mi_bits,
    permutation_null_mi_bits,
    permutation_pvals,
)
from data.builders_veracity import (
    build_binary_veracity_dataset,
    build_ordinal_veracity_dataset,
)
from quantum.circuits_veracity import (
    generate_optimization_circuit,
    generate_time_evolution_circuit,
)
from quantum.features import compile_and_extract_features
from quantum.topologies import make_topologies
from viz.plots_confusion import plot_confusion_matrix
from viz.plots_distributions import plot_feature_distributions
from viz.plots_mi import plot_mi_bars


RANDOM_SEED = 42
N_QUBITS = 8

BINARY_ACCURACIES = ["low", "high"]
N_BINARY_SAMPLES_PER_CLASS = 180

ORDINAL_ACCURACIES = [1e-2, 1e-3, 1e-4, 1e-5]
N_ORDINAL_SAMPLES_PER_LEVEL = 100

WORKLOAD_FAMILIES = ["time_evolution", "optimization"]

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

LOGICAL_FOOTPRINT_COLUMNS = [
    "logical_depth",
    "logical_twoq",
    "logical_total_ops",
]

N_MI_BOOT = 300
N_MI_PERM = 500

OUTPUT_PREFIX = "exp4_veracity_leakage"


def run_binary_experiment() -> pd.DataFrame:
    topologies = make_topologies(N_QUBITS)
    all_summary_rows = []

    print("\n" + "=" * 90)
    print("RUNNING EXPERIMENT 4A: BINARY VERACITY")
    print("=" * 90)

    for topo_name, cmap in topologies.items():
        print(f"\n[Exp4-Binary] Topology: {topo_name}")

        ds = build_binary_veracity_dataset(
            topology_name=topo_name,
            coupling_map=cmap,
            workload_families=WORKLOAD_FAMILIES,
            n_samples_per_class=N_BINARY_SAMPLES_PER_CLASS,
            seed=RANDOM_SEED,
            num_qubits=N_QUBITS,
            generate_time_evolution_circuit=generate_time_evolution_circuit,
            generate_optimization_circuit=generate_optimization_circuit,
            compile_and_extract_features=compile_and_extract_features,
        )

        df = ds.df
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=int)
        groups = df["workload_family"].astype(str) + "_" + df["accuracy_level"].astype(str)

        obs_point = compute_observed_mi_bits(X, y, seed=RANDOM_SEED)
        obs_mean, obs_lo, obs_hi = bootstrap_mi_bits(X, y, n_boot=N_MI_BOOT, seed=RANDOM_SEED)
        null_dist = permutation_null_mi_bits(X, y, n_perm=N_MI_PERM, seed=RANDOM_SEED)
        pvals = permutation_pvals(obs_point, null_dist)

        # Here grouped CV is only meaningful if groups are true grouped units.
        # For now, use instance_id-derived groups if needed later; this is a placeholder.
        cv_df = grouped_cv_classifier_scores(
            X=X,
            y=y,
            groups=df["instance_id"].to_numpy(),
            seed=RANDOM_SEED,
            n_splits=5,
        )

        holdout_df = family_holdout_evaluation(
            df=df,
            feature_cols=FEATURE_COLUMNS,
            family_col="workload_family",
            label_col="label",
            task_type="binary",
            seed=RANDOM_SEED,
        )

        print("\nCross-validation")
        print(cv_df.to_string(index=False))

        print("\nFamily holdout")
        print(holdout_df.to_string(index=False))

        plot_mi_bars(
            feature_names=FEATURE_COLUMNS,
            observed_mean=obs_mean,
            observed_lo=obs_lo,
            observed_hi=obs_hi,
            null_dist=null_dist,
            title=f"{topo_name}: binary veracity MI",
            outpath=f"{OUTPUT_PREFIX}_{topo_name}_binary_mi.png",
        )

        for feat in ["routed_depth", "extra_twoq", "depth_overhead"]:
            plot_feature_distributions(
                df=df,
                feature=feat,
                label_col="label",
                label_names={0: "Low accuracy", 1: "High accuracy"},
                title=f"{topo_name}: {feat} by binary accuracy class",
                outpath=f"{OUTPUT_PREFIX}_{topo_name}_binary_{feat}.png",
            )

        for _, row in holdout_df.iterrows():
            all_summary_rows.append({
                "topology": topo_name,
                "task": "binary",
                "train_family": row["train_family"],
                "test_family": row["test_family"],
                "model": row["model"],
                "macro_f1": row["macro_f1"],
                "bal_acc": row["bal_acc"],
                "routed_depth_mi_bits": obs_point[FEATURE_COLUMNS.index("routed_depth")],
                "extra_twoq_mi_bits": obs_point[FEATURE_COLUMNS.index("extra_twoq")],
                "mean_perm_p": float(pvals.mean()),
            })

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(f"{OUTPUT_PREFIX}_binary_summary.csv", index=False)
    print("\nSaved:", f"{OUTPUT_PREFIX}_binary_summary.csv")
    return summary_df


def run_ordinal_experiment() -> pd.DataFrame:
    topologies = make_topologies(N_QUBITS)
    all_summary_rows = []

    print("\n" + "=" * 90)
    print("RUNNING EXPERIMENT 4B: ORDINAL VERACITY")
    print("=" * 90)

    for topo_name, cmap in topologies.items():
        print(f"\n[Exp4-Ordinal] Topology: {topo_name}")

        ds = build_ordinal_veracity_dataset(
            topology_name=topo_name,
            coupling_map=cmap,
            workload_families=WORKLOAD_FAMILIES,
            accuracy_levels=ORDINAL_ACCURACIES,
            n_samples_per_level=N_ORDINAL_SAMPLES_PER_LEVEL,
            seed=RANDOM_SEED + 100,
            num_qubits=N_QUBITS,
            generate_time_evolution_circuit=generate_time_evolution_circuit,
            generate_optimization_circuit=generate_optimization_circuit,
            compile_and_extract_features=compile_and_extract_features,
        )

        df = ds.df
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=int)

        obs_point = compute_observed_mi_bits(X, y, seed=RANDOM_SEED + 100)
        obs_mean, obs_lo, obs_hi = bootstrap_mi_bits(
            X, y, n_boot=N_MI_BOOT, seed=RANDOM_SEED + 100
        )
        null_dist = permutation_null_mi_bits(
            X, y, n_perm=N_MI_PERM, seed=RANDOM_SEED + 100
        )
        pvals = permutation_pvals(obs_point, null_dist)

        holdout_df = family_holdout_evaluation(
            df=df,
            feature_cols=FEATURE_COLUMNS,
            family_col="workload_family",
            label_col="label",
            task_type="ordinal",
            seed=RANDOM_SEED + 100,
            ordered_labels=list(range(len(ORDINAL_ACCURACIES))),
        )

        print("\nFamily holdout")
        print(holdout_df.to_string(index=False))

        plot_mi_bars(
            feature_names=FEATURE_COLUMNS,
            observed_mean=obs_mean,
            observed_lo=obs_lo,
            observed_hi=obs_hi,
            null_dist=null_dist,
            title=f"{topo_name}: ordinal veracity MI",
            outpath=f"{OUTPUT_PREFIX}_{topo_name}_ordinal_mi.png",
        )

        # Example RF confusion matrix: time_evolution -> optimization
        train_df = df[df["workload_family"] == WORKLOAD_FAMILIES[0]].copy()
        test_df = df[df["workload_family"] == WORKLOAD_FAMILIES[1]].copy()

        from analysis.classifiers import make_models
        rf = make_models(RANDOM_SEED)["rf"]
        rf.fit(train_df[FEATURE_COLUMNS], train_df["label"])
        y_pred = rf.predict(test_df[FEATURE_COLUMNS])

        plot_confusion_matrix(
            y_true=test_df["label"].to_numpy(),
            y_pred=y_pred,
            labels=list(range(len(ORDINAL_ACCURACIES))),
            title=f"{topo_name}: RF holdout {WORKLOAD_FAMILIES[0]}→{WORKLOAD_FAMILIES[1]}",
            outpath=f"{OUTPUT_PREFIX}_{topo_name}_ordinal_cm.png",
        )

        for _, row in holdout_df.iterrows():
            all_summary_rows.append({
                "topology": topo_name,
                "task": "ordinal",
                "train_family": row["train_family"],
                "test_family": row["test_family"],
                "model": row["model"],
                "macro_f1": row["macro_f1"],
                "bal_acc": row["bal_acc"],
                "class_mae": row["class_mae"],
                "adjacent_acc": row["adjacent_acc"],
                "routed_depth_mi_bits": obs_point[FEATURE_COLUMNS.index("routed_depth")],
                "extra_twoq_mi_bits": obs_point[FEATURE_COLUMNS.index("extra_twoq")],
                "mean_perm_p": float(pvals.mean()),
            })

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(f"{OUTPUT_PREFIX}_ordinal_summary.csv", index=False)
    print("\nSaved:", f"{OUTPUT_PREFIX}_ordinal_summary.csv")
    return summary_df


def run_experiment() -> pd.DataFrame:
    binary_df = run_binary_experiment()
    ordinal_df = run_ordinal_experiment()
    return pd.concat([binary_df, ordinal_df], axis=0, ignore_index=True)


if __name__ == "__main__":
    run_experiment()
