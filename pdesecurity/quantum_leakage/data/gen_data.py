"""
Main data generation orchestrator for quantum circuit leakage experiments.

This module provides high-level functions to build complete experimental datasets
by coordinating the individual builders for boundary, scale, and drift experiments.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from schemas import (
    BoundaryDataset,
    DriftDatasetBundle,
    ExperimentDataset,
    ScaleDataset,
    VeracityDataset,
)
from builders_boundary import build_boundary_dataset
from builders_scale import build_scale_dataset
from builders_drift import (
    apply_hardware_drift,
    augment_operational_features,
)


def build_complete_boundary_experiment(
    topology_families: List[str],
    num_qubits: int,
    n_samples_per_class: int,
    n_steps: int,
    seed: int,
    make_coupling_map: Callable,
    generate_pde_surrogate: Callable,
    compile_and_extract_features: Callable,
    verify_batch_fn: Optional[Callable] = None,
) -> Dict[str, BoundaryDataset]:
    """
    Build boundary-topology datasets for multiple topology families.
    
    Parameters
    ----------
    topology_families : List[str]
        List of topology families to generate (e.g., ["line", "ladder", "grid"])
    num_qubits : int
        Number of logical qubits for all experiments
    n_samples_per_class : int
        Number of matched pairs per topology family
    n_steps : int
        Number of evolution steps in PDE surrogate
    seed : int
        Base random seed
    make_coupling_map : Callable
        Function to create coupling maps: (num_qubits, topology_family) -> coupling_map
    generate_pde_surrogate : Callable
        PDE circuit generator: (num_qubits, boundary_condition, n_steps, seed) -> circuit
    compile_and_extract_features : Callable
        Feature extractor: (circuit, coupling_map, seed) -> dict
    verify_batch_fn : Optional[Callable]
        Optional verification function for transpilation quality
        
    Returns
    -------
    Dict[str, BoundaryDataset]
        Dictionary mapping topology_family -> BoundaryDataset
    """
    import numpy as np
    
    rng = np.random.default_rng(seed)
    datasets = {}
    
    for topo_family in topology_families:
        topo_seed = int(rng.integers(0, 10_000_000))
        
        print(f"\n{'='*60}")
        print(f"Building boundary dataset: {topo_family}")
        print(f"{'='*60}")
        
        dataset = build_boundary_dataset(
            topology_family=topo_family,
            num_qubits=num_qubits,
            n_samples_per_class=n_samples_per_class,
            n_steps=n_steps,
            seed=topo_seed,
            make_coupling_map=make_coupling_map,
            generate_pde_surrogate=generate_pde_surrogate,
            compile_and_extract_features=compile_and_extract_features,
            verify_batch_fn=verify_batch_fn,
        )
        
        datasets[topo_family] = dataset
        print(f"✓ Generated {len(dataset.df)} samples for {topo_family}")
    
    return datasets


def build_complete_scale_experiment(
    resolutions: List[int],
    n_samples_per_res: int,
    topology_families: List[str],
    template_families: List[str],
    n_steps: int,
    seed: int,
    make_coupling_map: Callable,
    generate_scale_surrogate: Callable,
    compile_and_extract_features: Callable,
) -> ScaleDataset:
    """
    Build a unified scale-leakage dataset across all experimental factors.
    
    Parameters
    ----------
    resolutions : List[int]
        List of qubit counts to sweep (e.g., [10, 20, 30, 40, 50])
    n_samples_per_res : int
        Samples per resolution/topology/template combination
    topology_families : List[str]
        Backend topology families
    template_families : List[str]
        Logical circuit template families
    n_steps : int
        Evolution steps per circuit
    seed : int
        Base random seed
    make_coupling_map : Callable
        Coupling map constructor
    generate_scale_surrogate : Callable
        Scale surrogate generator: (num_qubits, template_family, n_steps, seed) -> circuit
    compile_and_extract_features : Callable
        Feature extractor
        
    Returns
    -------
    ScaleDataset
        Unified dataset with all scale experiments
    """
    print(f"\n{'='*60}")
    print(f"Building scale-leakage dataset")
    print(f"{'='*60}")
    print(f"Resolutions: {resolutions}")
    print(f"Topologies: {topology_families}")
    print(f"Templates: {template_families}")
    print(f"Samples per config: {n_samples_per_res}")
    
    dataset = build_scale_dataset(
        resolutions=resolutions,
        n_samples_per_res=n_samples_per_res,
        topology_families=topology_families,
        template_families=template_families,
        n_steps=n_steps,
        seed=seed,
        make_coupling_map=make_coupling_map,
        generate_scale_surrogate=generate_scale_surrogate,
        compile_and_extract_features=compile_and_extract_features,
    )
    
    total_configs = len(resolutions) * len(topology_families) * len(template_families)
    expected_samples = total_configs * n_samples_per_res
    
    print(f"✓ Generated {len(dataset.df)} samples ({total_configs} configurations)")
    assert len(dataset.df) == expected_samples, \
        f"Expected {expected_samples} samples, got {len(dataset.df)}"
    
    return dataset


def prepare_drift_datasets(
    boundary_datasets: Dict[str, BoundaryDataset],
    scale_dataset: ScaleDataset,
    operational_seed: int,
) -> DriftDatasetBundle:
    """
    Prepare clean reference datasets for drift experiments.
    
    Combines boundary datasets and augments with operational features.
    
    Parameters
    ----------
    boundary_datasets : Dict[str, BoundaryDataset]
        Dictionary of boundary datasets by topology family
    scale_dataset : ScaleDataset
        Unified scale dataset
    operational_seed : int
        Seed for operational feature augmentation
        
    Returns
    -------
    DriftDatasetBundle
        Clean datasets ready for drift simulation
    """
    print(f"\n{'='*60}")
    print(f"Preparing drift reference datasets")
    print(f"{'='*60}")
    
    # Combine all boundary datasets
    boundary_dfs = [ds.df for ds in boundary_datasets.values()]
    combined_boundary = pd.concat(boundary_dfs, ignore_index=True)
    
    print(f"Combined boundary dataset: {len(combined_boundary)} samples")
    print(f"  Topologies: {combined_boundary['topology_family'].unique().tolist()}")
    print(f"  Classes: {combined_boundary['label_name'].unique().tolist()}")
    
    # Augment with operational features
    boundary_augmented = augment_operational_features(combined_boundary, seed=operational_seed)
    scale_augmented = augment_operational_features(scale_dataset.df, seed=operational_seed + 1)
    
    operational_cols = ["sched_duration_ms", "idle_variance"]
    print(f"✓ Added operational features: {operational_cols}")
    
    bundle = DriftDatasetBundle(
        boundary_df=boundary_augmented,
        scale_df=scale_augmented,
    )
    
    return bundle


def simulate_hardware_drift_scenarios(
    clean_bundle: DriftDatasetBundle,
    severity_levels: List[float],
    seed: int,
) -> Dict[Tuple[str, float], pd.DataFrame]:
    """
    Generate multiple drift scenarios across different severity levels.
    
    Parameters
    ----------
    clean_bundle : DriftDatasetBundle
        Clean reference datasets
    severity_levels : List[float]
        Drift severity multipliers (e.g., [0.0, 0.5, 1.0, 2.0])
    seed : int
        Base seed for drift simulation
        
    Returns
    -------
    Dict[Tuple[str, float], pd.DataFrame]
        Dictionary mapping (dataset_type, severity) -> drifted DataFrame
        dataset_type is either "boundary" or "scale"
    """
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"Simulating hardware drift scenarios")
    print(f"{'='*60}")
    print(f"Severity levels: {severity_levels}")
    
    rng = np.random.default_rng(seed)
    drift_scenarios = {}
    
    for dataset_type, df in [("boundary", clean_bundle.boundary_df), 
                             ("scale", clean_bundle.scale_df)]:
        
        topology_families = df["topology_family"].unique()
        
        for severity in severity_levels:
            scenario_key = (dataset_type, severity)
            drift_seed = int(rng.integers(0, 10_000_000))
            
            # Apply drift per topology family, then combine
            drifted_dfs = []
            
            for topo in topology_families:
                topo_df = df[df["topology_family"] == topo].copy()
                topo_seed = int(rng.integers(0, 10_000_000))
                
                drifted = apply_hardware_drift(
                    df=topo_df,
                    severity=severity,
                    topology_family=topo,
                    seed=topo_seed,
                )
                drifted_dfs.append(drifted)
            
            combined_drifted = pd.concat(drifted_dfs, ignore_index=True)
            drift_scenarios[scenario_key] = combined_drifted
            
            print(f"  ✓ {dataset_type:8s} @ severity={severity:.1f}: {len(combined_drifted):5d} samples")
    
    return drift_scenarios


def build_veracity_dataset(
    base_df: pd.DataFrame,
    task_type: str,
    noise_fraction: float,
    seed: int,
) -> VeracityDataset:
    """
    Build a veracity/accuracy leakage dataset with noisy labels.
    
    Parameters
    ----------
    base_df : pd.DataFrame
        Clean dataset to corrupt
    task_type : str
        Either "binary" or "ordinal"
    noise_fraction : float
        Fraction of labels to corrupt (0.0 to 1.0)
    seed : int
        Random seed for label corruption
        
    Returns
    -------
    VeracityDataset
        Dataset with corrupted labels and ground truth
    """
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"Building veracity dataset ({task_type})")
    print(f"{'='*60}")
    print(f"Noise fraction: {noise_fraction:.2%}")
    
    rng = np.random.default_rng(seed)
    df = base_df.copy()
    
    # Store ground truth
    df["label_true"] = df["label"].copy()
    df["label_name_true"] = df["label_name"].copy()
    
    # Identify labels to corrupt
    n_total = len(df)
    n_corrupt = int(n_total * noise_fraction)
    corrupt_indices = rng.choice(n_total, size=n_corrupt, replace=False)
    
    unique_labels = sorted(df["label"].unique())
    
    # Corrupt labels
    for idx in corrupt_indices:
        true_label = df.loc[idx, "label"]
        
        if task_type == "binary":
            # Flip binary labels
            new_label = 1 - true_label
        elif task_type == "ordinal":
            # Shift ordinal labels randomly
            possible = [lbl for lbl in unique_labels if lbl != true_label]
            new_label = rng.choice(possible)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        df.loc[idx, "label"] = new_label
        # Update label_name if present
        if "label_name" in df.columns:
            df.loc[idx, "label_name"] = str(new_label)
    
    df["is_corrupted"] = False
    df.loc[corrupt_indices, "is_corrupted"] = True
    
    print(f"✓ Corrupted {n_corrupt}/{n_total} labels ({noise_fraction:.1%})")
    print(f"  Unique labels: {unique_labels}")
    
    return VeracityDataset(df=df, task_type=task_type)


def export_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_dir: str,
    format: str = "parquet",
) -> None:
    """
    Export multiple datasets to disk.
    
    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary mapping dataset_name -> DataFrame
    output_dir : str
        Output directory path
    format : str
        Export format: "parquet", "csv", or "pickle"
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Exporting {len(datasets)} datasets to {output_dir}")
    print(f"{'='*60}")
    
    for name, df in datasets.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        
        if format == "parquet":
            filepath = os.path.join(output_dir, f"{safe_name}.parquet")
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            filepath = os.path.join(output_dir, f"{safe_name}.csv")
            df.to_csv(filepath, index=False)
        elif format == "pickle":
            filepath = os.path.join(output_dir, f"{safe_name}.pkl")
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"  ✓ {name:30s} → {os.path.basename(filepath):40s} ({len(df):5d} rows)")


def generate_summary_statistics(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate summary statistics across all datasets.
    
    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of datasets to summarize
        
    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    summaries = []
    
    for name, df in datasets.items():
        summary = {
            "dataset": name,
            "n_samples": len(df),
            "n_features": len([c for c in df.columns if c not in ["label", "label_name", "instance_id"]]),
        }
        
        if "topology_family" in df.columns:
            summary["topologies"] = ", ".join(sorted(df["topology_family"].unique()))
        
        if "label" in df.columns:
            summary["n_classes"] = df["label"].nunique()
            summary["class_balance"] = f"{df['label'].value_counts().min()}/{df['label'].value_counts().max()}"
        
        if "task" in df.columns:
            summary["task"] = df["task"].iloc[0]
        
        # Feature stats
        numeric_cols = df.select_dtypes(include=["number"]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["label", "local_sample_idx"]]
        
        if len(numeric_cols) > 0:
            summary["mean_feature_std"] = df[numeric_cols].std().mean()
        
        summaries.append(summary)
    
    summary_df = pd.DataFrame(summaries)
    return summary_df


# Convenience function for end-to-end generation
def generate_all_experiments(
    config: Dict,
    make_coupling_map: Callable,
    generate_pde_surrogate: Callable,
    generate_scale_surrogate: Callable,
    compile_and_extract_features: Callable,
    output_dir: str,
    verify_batch_fn: Optional[Callable] = None,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end experiment generation pipeline.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary with keys:
        - topology_families
        - num_qubits
        - n_samples_boundary
        - resolutions
        - n_samples_scale
        - template_families
        - n_steps
        - drift_severities
        - base_seed
    make_coupling_map : Callable
        Coupling map constructor
    generate_pde_surrogate : Callable
        PDE surrogate generator
    generate_scale_surrogate : Callable
        Scale surrogate generator
    compile_and_extract_features : Callable
        Feature extractor
    output_dir : str
        Output directory for datasets
    verify_batch_fn : Optional[Callable]
        Verification function
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        All generated datasets
    """
    import numpy as np
    
    print("\n" + "="*60)
    print("QUANTUM CIRCUIT LEAKAGE EXPERIMENT SUITE")
    print("="*60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    
    all_datasets = {}
    rng = np.random.default_rng(config["base_seed"])
    
    # 1. Boundary experiments
    boundary_datasets = build_complete_boundary_experiment(
        topology_families=config["topology_families"],
        num_qubits=config["num_qubits"],
        n_samples_per_class=config["n_samples_boundary"],
        n_steps=config["n_steps"],
        seed=int(rng.integers(0, 10_000_000)),
        make_coupling_map=make_coupling_map,
        generate_pde_surrogate=generate_pde_surrogate,
        compile_and_extract_features=compile_and_extract_features,
        verify_batch_fn=verify_batch_fn,
    )
    
    for topo, ds in boundary_datasets.items():
        all_datasets[f"boundary_{topo}"] = ds.df
    
    # 2. Scale experiments
    scale_dataset = build_complete_scale_experiment(
        resolutions=config["resolutions"],
        n_samples_per_res=config["n_samples_scale"],
        topology_families=config["topology_families"],
        template_families=config["template_families"],
        n_steps=config["n_steps"],
        seed=int(rng.integers(0, 10_000_000)),
        make_coupling_map=make_coupling_map,
        generate_scale_surrogate=generate_scale_surrogate,
        compile_and_extract_features=compile_and_extract_features,
    )
    
    all_datasets["scale"] = scale_dataset.df
    
    # 3. Drift experiments
    drift_bundle = prepare_drift_datasets(
        boundary_datasets=boundary_datasets,
        scale_dataset=scale_dataset,
        operational_seed=int(rng.integers(0, 10_000_000)),
    )
    
    drift_scenarios = simulate_hardware_drift_scenarios(
        clean_bundle=drift_bundle,
        severity_levels=config["drift_severities"],
        seed=int(rng.integers(0, 10_000_000)),
    )
    
    for (ds_type, severity), df in drift_scenarios.items():
        all_datasets[f"drift_{ds_type}_sev{severity:.1f}"] = df
    
    # 4. Export
    export_datasets(all_datasets, output_dir, format="parquet")
    
    # 5. Summary
    summary = generate_summary_statistics(all_datasets)
    print("\n" + "="*60)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    
    return all_datasets
