"""
Experiment 1: Boundary-topology leakage.

Tests whether hidden boundary regime (Dirichlet vs Periodic) is recoverable
from provider-visible compilation artefacts.
"""

from __future__ import annotations

import pandas as pd

from analysis.classifiers import grouped_cv_classifier_scores
from analysis.mi import (
    bootstrap_mi_bits_grouped,
    compute_observed_mi_bits,
    paired_label_swap_null_mi_bits,
    permutation_pvals,
)
from analysis.paired_tests import paired_feature_tests
from data.builders_boundary import build_boundary_dataset
from quantum.circuits_pde import generate_pde_surrogate
from quantum.features import compile_and_extract_features
from quantum.topologies import make_coupling_map
from viz.plots_distributions import plot_feature_distributions
from viz.plots_mi import plot_mi_bars


RANDOM_SEED = 42
NUM_QUBITS = 8
N_SAMPLES_PER_CLASS = 180
N_STEPS = 3
N_MI_BOOT = 300
N_MI_PERM = 500

TOPOLOGY_FAMILIES = ["line", "ladder"]

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

OUTPUT_PREFIX = "exp1_boundary_topology"


def run_experiment() -> pd.DataFrame:
    all_summary_rows = []

    print("\n" + "=" * 90)
    print("RUNNING EXPERIMENT 1: BOUNDARY TOPOLOGY LEAKAGE")
    print("=" * 90)

    for topology_family in TOPOLOGY_FAMILIES:
        print(f"\n[Exp1] Topology: {topology_family}")

        ds = build_boundary_dataset(
            topology_family=topology_family,
            num_qubits=NUM_QUBITS,
            n_samples_per_class=N_SAMPLES_PER_CLASS,
            n_steps=N_STEPS,
            seed=RANDOM_SEED,
            make_coupling_map=make_coupling_map,
            generate_pde_surrogate=generate_pde_surrogate,
            compile_and_extract_features=compile_and_extract_features,
        )

        df = ds.df
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=int)
        groups = df["pair_id"].to_numpy()

        obs_point = compute_observed_mi_bits(X, y, seed=RANDOM_SEED)
        obs_mean, obs_lo, obs_hi, _ = bootstrap_mi_bits_grouped(
            X, y, groups=groups, n_boot=N_MI_BOOT, seed=RANDOM_SEED
        )
        null_dist = paired_label_swap_null_mi_bits(
            X, y, groups=groups, n_perm=N_MI_PERM, seed=RANDOM_SEED
        )
        pvals = permutation_pvals(obs_point, null_dist)

        cv_df = grouped_cv_classifier_scores(
            X=X,
            y=y,
            groups=groups,
            seed=RANDOM_SEED,
            n_splits=5,
        )

        effect_df = paired_feature_tests(
            df=df,
            feature_columns=FEATURE_COLUMNS,
            pair_id_col="pair_id",
            boundary_col="boundary",
            a_label="dirichlet",
            b_label="periodic",
            n_perm=5000,
            seed=RANDOM_SEED,
        )

        print("\nCross-validation")
        print(cv_df.to_string(index=False))

        print("\nPaired feature tests")
        print(effect_df.to_string(index=False))

        plot_mi_bars(
            feature_names=FEATURE_COLUMNS,
            observed_mean=obs_mean,
            observed_lo=obs_lo,
            observed_hi=obs_hi,
            null_dist=null_dist,
            title=f"{topology_family}: boundary MI vs paired null",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_mi.png",
        )

        for feat in ["swap_equiv", "extra_twoq", "routed_depth"]:
            plot_feature_distributions(
                df=df,
                feature=feat,
                label_col="label",
                label_names={0: "Dirichlet", 1: "Periodic"},
                title=f"{topology_family}: {feat} by boundary regime",
                outpath=f"{OUTPUT_PREFIX}_{topology_family}_{feat}.png",
            )

        for _, row in cv_df.iterrows():
            all_summary_rows.append({
                "topology_family": topology_family,
                "model": row["model"],
                "macro_f1_mean": row["macro_f1_mean"],
                "bal_acc_mean": row["bal_acc_mean"],
                "swap_equiv_mi_bits": obs_point[FEATURE_COLUMNS.index("swap_equiv")],
                "routed_depth_mi_bits": obs_point[FEATURE_COLUMNS.index("routed_depth")],
                "extra_twoq_mi_bits": obs_point[FEATURE_COLUMNS.index("extra_twoq")],
                "mean_perm_p": float(pvals.mean()),
            })

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(f"{OUTPUT_PREFIX}_summary.csv", index=False)

    print("\nSaved:", f"{OUTPUT_PREFIX}_summary.csv")
    return summary_df


if __name__ == "__main__":
    run_experiment()
