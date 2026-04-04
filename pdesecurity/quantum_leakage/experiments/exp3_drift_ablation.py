"""
Experiment 3: Drift stability and feature-group ablation.
"""

from __future__ import annotations

import pandas as pd

from analysis.drift import evaluate_task_under_drift, summarise_results
from data.builders_boundary import build_boundary_dataset
from data.builders_drift import apply_hardware_drift, augment_operational_features
from data.builders_scale import build_scale_dataset
from quantum.circuits_pde import generate_pde_surrogate
from quantum.circuits_scale import generate_scale_surrogate
from quantum.features import compile_and_extract_features
from quantum.topologies import make_coupling_map
from viz.plots_drift import plot_correlation_heatmap, plot_drift_curves


RANDOM_SEED = 42
BOUNDARY_NUM_QUBITS = 8
BOUNDARY_SAMPLES_PER_CLASS = 120

SCALE_RESOLUTIONS = [4, 6, 8, 10, 12, 16]
SCALE_SAMPLES_PER_RES = 80

N_STEPS = 3
N_REPEATS = 10
DRIFT_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

TOPOLOGY_FAMILIES = ["line", "gridish"]
TEMPLATE_FAMILIES = ["A", "B"]

FEATURE_GROUPS = {
    "Topological": ["swap_equiv", "swap_fraction", "extra_twoq"],
    "Complexity": ["routed_depth", "depth_overhead", "twoq_overhead", "extra_depth", "cx_fraction"],
    "Operational": ["transpile_ms", "sched_duration_ms", "idle_variance"],
}
FEATURE_GROUPS["All"] = (
    FEATURE_GROUPS["Topological"]
    + FEATURE_GROUPS["Complexity"]
    + FEATURE_GROUPS["Operational"]
)

ORDERED_SCALE_LABELS = SCALE_RESOLUTIONS
OUTPUT_PREFIX = "exp3_drift_ablation"


def run_experiment() -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("RUNNING EXPERIMENT 3: DRIFT ABLATION")
    print("=" * 90)

    all_raw = []

    for topology_family in TOPOLOGY_FAMILIES:
        print(f"\n[Exp3] Topology: {topology_family}")

        boundary_ds = build_boundary_dataset(
            topology_family=topology_family,
            num_qubits=BOUNDARY_NUM_QUBITS,
            n_samples_per_class=BOUNDARY_SAMPLES_PER_CLASS,
            n_steps=N_STEPS,
            seed=RANDOM_SEED + 11,
            make_coupling_map=make_coupling_map,
            generate_pde_surrogate=generate_pde_surrogate,
            compile_and_extract_features=compile_and_extract_features,
        )
        df_boundary = augment_operational_features(boundary_ds.df, RANDOM_SEED + 111)

        scale_ds = build_scale_dataset(
            resolutions=SCALE_RESOLUTIONS,
            n_samples_per_res=SCALE_SAMPLES_PER_RES,
            topology_families=[topology_family],
            template_families=TEMPLATE_FAMILIES,
            n_steps=N_STEPS,
            seed=RANDOM_SEED + 22,
            make_coupling_map=make_coupling_map,
            generate_scale_surrogate=generate_scale_surrogate,
            compile_and_extract_features=compile_and_extract_features,
        )
        df_scale = augment_operational_features(scale_ds.df, RANDOM_SEED + 222)

        corr_cols = list(dict.fromkeys(FEATURE_GROUPS["All"]))

        plot_correlation_heatmap(
            df=df_boundary,
            columns=corr_cols,
            title=f"{topology_family}: boundary correlations",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_boundary_corr.png",
        )
        plot_correlation_heatmap(
            df=df_scale,
            columns=corr_cols,
            title=f"{topology_family}: scale correlations",
            outpath=f"{OUTPUT_PREFIX}_{topology_family}_scale_corr.png",
        )

        raw_boundary = evaluate_task_under_drift(
            df_clean=df_boundary,
            task_name="boundary",
            topology_family=topology_family,
            label_col="label",
            feature_groups=FEATURE_GROUPS,
            drift_levels=DRIFT_LEVELS,
            drift_fn=apply_hardware_drift,
            n_repeats=N_REPEATS,
            seed=RANDOM_SEED + 100,
        )

        raw_scale = evaluate_task_under_drift(
            df_clean=df_scale,
            task_name="scale",
            topology_family=topology_family,
            label_col="label",
            feature_groups=FEATURE_GROUPS,
            drift_levels=DRIFT_LEVELS,
            drift_fn=apply_hardware_drift,
            n_repeats=N_REPEATS,
            seed=RANDOM_SEED + 200,
            ordered_scale_labels=ORDERED_SCALE_LABELS,
        )

        all_raw.extend([raw_boundary, raw_scale])

    raw_df = pd.concat(all_raw, axis=0, ignore_index=True)
    summary_df = summarise_results(raw_df)

    for topology_family in TOPOLOGY_FAMILIES:
        for metric in ["macro_f1", "bal_acc"]:
            plot_drift_curves(
                results_df=summary_df,
                task_name="boundary",
                topology_family=topology_family,
                metric=metric,
                feature_groups=FEATURE_GROUPS,
                outpath=f"{OUTPUT_PREFIX}_{topology_family}_boundary_{metric}.png",
                model_name="rf",
            )

        for metric in ["macro_f1", "class_mae", "adjacent_acc"]:
            plot_drift_curves(
                results_df=summary_df,
                task_name="scale",
                topology_family=topology_family,
                metric=metric,
                feature_groups=FEATURE_GROUPS,
                outpath=f"{OUTPUT_PREFIX}_{topology_family}_scale_{metric}.png",
                model_name="rf",
            )

    raw_df.to_csv(f"{OUTPUT_PREFIX}_raw.csv", index=False)
    summary_df.to_csv(f"{OUTPUT_PREFIX}_summary.csv", index=False)

    print("\nSaved:", f"{OUTPUT_PREFIX}_raw.csv")
    print("Saved:", f"{OUTPUT_PREFIX}_summary.csv")
    return summary_df


if __name__ == "__main__":
    run_experiment()
