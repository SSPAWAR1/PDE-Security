"""
Experiment 2: Scale leakage.

Tests whether hidden scientific scale / resolution leaves a robust signal
in provider-visible compilation artefacts.
"""

from __future__ import annotations

import pandas as pd

from analysis.classifiers import family_holdout_evaluation
from analysis.mi import (
    bootstrap_mi_bits,
    compute_observed_mi_bits,
    permutation_null_mi_bits,
    permutation_pvals,
)
from analysis.scaling import bootstrap_scaling_exponent
from data.builders_scale import build_scale_dataset
from quantum.circuits_scale import generate_scale_surrogate
from quantum.features import compile_and_extract_features
from quantum.topologies import make_coupling_map
from viz.plots_confusion import plot_confusion_matrix
from viz.plots_mi import plot_mi_bars
from viz.plots_scaling import plot_loglog_scaling


RANDOM_SEED = 42
RESOLUTIONS = [4, 6, 8, 10, 12, 16]
N_SAMPLES_PER_RES = 120
N_STEPS = 3
N_MI_BOOT = 300
N_MI_PERM = 500

TOPOLOGY_FAMILIES = ["line", "gridish"]
TEMPLATE_FAMILIES = ["A", "B"]

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

SCALING_FEATURES = ["routed_depth", "extra_twoq"]

OUTPUT_PREFIX = "exp2_scale_leakage"


def run_experiment() -> pd.DataFrame:
    all_summary_rows = []

    print("\n" + "=" * 90)
    print("RUNNING EXPERIMENT 2: SCALE LEAKAGE")
    print("=" * 90)

    ds = build_scale_dataset(
        resolutions=RESOLUTIONS,
        n_samples_per_res=N_SAMPLES_PER_RES,
        topology_families=TOPOLOGY_FAMILIES,
        template_families=TEMPLATE_FAMILIES,
        n_steps=N_STEPS,
        seed=RANDOM_SEED,
        make_coupling_map=make_coupling_map,
        generate_scale_surrogate=generate_scale_surrogate,
        compile_and_extract_features=compile_and_extract_features,
    )

    df = ds.df

    for topology_family in TOPOLOGY_FAMILIES:
        print(f"\n[Exp2] Topology: {topology_family}")
        topo_df = df[df["topology_family"] == topology_family].copy()

        X = topo_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = topo_df["label"].to_numpy(dtype=int)

        obs_point = compute_observed_mi_bits(X, y, seed=RANDOM_SEED)
        obs_mean, obs_lo, obs_hi = bootstrap_mi_bits(
            X, y, n_boot=N_MI_BOOT, seed=RANDOM_SEED
        )
        null_dist = permutation_null_mi_bits(
            X, y, n_perm=N_MI_PERM, seed=RANDOM_SEED
        )
        pvals = permutation_pvals(obs_point, null_dist)

        print("\nFeature-wise MI")
        for feat, mi_val, p in zip(FEATURE_COLUMNS, obs_point, pvals):
            print(f"  {feat:<18} MI={mi_val:.3f} bits | p={p:.4f}")

        plot_mi_bars(
            feature_names=FEATURE_COLUMNS,
            observed_mean=obs_mean,
            observed_lo=obs_lo,
            observed_hi=obs_hi,
            null_dist=null_dist,
            title=f"{topology_family}: scale MI vs permutation null",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_mi.png",
        )

        family_stats_by_feature = {}
        for feat in SCALING_FEATURES:
            family_stats = {}
            for fam in TEMPLATE_FAMILIES:
                sub = topo_df[topo_df["template_family"] == fam]
                k, lo, hi = bootstrap_scaling_exponent(
                    N_vals=sub["N"].values,
                    feat_vals=sub[feat].values,
                    n_boot=N_MI_BOOT,
                    seed=RANDOM_SEED,
                )
                family_stats[fam] = (k, lo, hi)

            family_stats_by_feature[feat] = family_stats

            plot_loglog_scaling(
                df=topo_df,
                topology_family=topology_family,
                feature=feat,
                family_stats=family_stats,
                family_col="template_family",
                topology_col="topology_family",
                outpath=f"{OUTPUT_PREFIX}_{topology_family}_{feat}_loglog.png",
            )

        holdout_df = family_holdout_evaluation(
            df=topo_df,
            feature_cols=FEATURE_COLUMNS,
            family_col="template_family",
            label_col="label",
            task_type="ordinal",
            seed=RANDOM_SEED,
            ordered_labels=RESOLUTIONS,
        )

        print("\nFamily-holdout results")
        print(holdout_df.to_string(index=False))

        # Example confusion matrix: RF, A -> B
        train_df = topo_df[topo_df["template_family"] == "A"].copy()
        test_df = topo_df[topo_df["template_family"] == "B"].copy()

        from analysis.classifiers import make_models
        rf = make_models(RANDOM_SEED)["rf"]
        rf.fit(train_df[FEATURE_COLUMNS], train_df["label"])
        y_pred = rf.predict(test_df[FEATURE_COLUMNS])

        plot_confusion_matrix(
            y_true=test_df["label"].to_numpy(),
            y_pred=y_pred,
            labels=RESOLUTIONS,
            title=f"{topology_family}: RF family-holdout A→B",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_rf_cm_A_to_B.png",
        )

        for _, row in holdout_df.iterrows():
            all_summary_rows.append({
                "topology_family": topology_family,
                "train_family": row["train_family"],
                "test_family": row["test_family"],
                "model": row["model"],
                "macro_f1": row["macro_f1"],
                "bal_acc": row["bal_acc"],
                "class_mae": row["class_mae"],
                "adjacent_acc": row["adjacent_acc"],
                "routed_depth_mi_bits": obs_point[FEATURE_COLUMNS.index("routed_depth")],
                "extra_twoq_mi_bits": obs_point[FEATURE_COLUMNS.index("extra_twoq")],
            })

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(f"{OUTPUT_PREFIX}_summary.csv", index=False)

    print("\nSaved:", f"{OUTPUT_PREFIX}_summary.csv")
    return summary_df


if __name__ == "__main__":
    run_experiment()
