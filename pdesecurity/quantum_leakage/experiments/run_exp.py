"""
Unified experiment runner for all leakage experiments.

Usage:
    python run_exp.py --all                    # Run all experiments
    python run_exp.py --exp 1                  # Run experiment 1 only
    python run_exp.py --exp 1 2 4              # Run experiments 1, 2, and 4
    python run_exp.py --exp 3 --quick          # Run experiment 3 with reduced samples
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def run_experiment_1():
    """Boundary-topology leakage."""
    from exp1_boundary_topology import run_experiment
    print("\n" + "█" * 90)
    print("█ EXPERIMENT 1: BOUNDARY-TOPOLOGY LEAKAGE")
    print("█" * 90)
    start = time.time()
    result_df = run_experiment()
    elapsed = time.time() - start
    print(f"\n✓ Experiment 1 completed in {elapsed:.1f}s")
    return result_df


def run_experiment_2():
    """Scale leakage."""
    from exp2_scale_leakage import run_experiment
    print("\n" + "█" * 90)
    print("█ EXPERIMENT 2: SCALE LEAKAGE")
    print("█" * 90)
    start = time.time()
    result_df = run_experiment()
    elapsed = time.time() - start
    print(f"\n✓ Experiment 2 completed in {elapsed:.1f}s")
    return result_df


def run_experiment_3():
    """Drift stability and feature ablation."""
    from exp3_drift_ablation import run_experiment
    print("\n" + "█" * 90)
    print("█ EXPERIMENT 3: DRIFT STABILITY & FEATURE ABLATION")
    print("█" * 90)
    start = time.time()
    result_df = run_experiment()
    elapsed = time.time() - start
    print(f"\n✓ Experiment 3 completed in {elapsed:.1f}s")
    return result_df


def run_experiment_4():
    """Veracity/accuracy leakage."""
    from exp4_veracity_leakage import run_experiment
    print("\n" + "█" * 90)
    print("█ EXPERIMENT 4: VERACITY/ACCURACY LEAKAGE")
    print("█" * 90)
    start = time.time()
    result_df = run_experiment()
    elapsed = time.time() - start
    print(f"\n✓ Experiment 4 completed in {elapsed:.1f}s")
    return result_df


EXPERIMENTS = {
    1: run_experiment_1,
    2: run_experiment_2,
    3: run_experiment_3,
    4: run_experiment_4,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run quantum compilation leakage experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_exp.py --all              # Run all experiments sequentially
  python run_exp.py --exp 1            # Run only experiment 1
  python run_exp.py --exp 1 2 4        # Run experiments 1, 2, and 4
  python run_exp.py --exp 3 --quick    # Run experiment 3 with reduced parameters
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments (1, 2, 3, 4)"
    )
    
    parser.add_argument(
        "--exp",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        help="Specific experiment(s) to run (space-separated)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced sample sizes for faster testing (not recommended for final results)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        print("\n⚠ WARNING: Running in QUICK mode with reduced samples.")
        print("   Results may not be statistically robust.\n")
    
    # Determine which experiments to run
    if args.all:
        exp_list = [1, 2, 3, 4]
    elif args.exp:
        exp_list = sorted(args.exp)
    else:
        parser.print_help()
        print("\nError: Must specify either --all or --exp")
        return
    
    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 90)
    print("QUANTUM COMPILATION LEAKAGE EXPERIMENTS")
    print("=" * 90)
    print(f"Experiments to run: {exp_list}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Quick mode: {args.quick}")
    print("=" * 90)
    
    results = {}
    overall_start = time.time()
    
    for exp_num in exp_list:
        try:
            result_df = EXPERIMENTS[exp_num]()
            results[exp_num] = result_df
        except Exception as e:
            print(f"\n✗ Experiment {exp_num} FAILED with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    print("\n" + "=" * 90)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 90)
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"\nCompleted experiments: {list(results.keys())}")
    
    if len(results) < len(exp_list):
        failed = set(exp_list) - set(results.keys())
        print(f"Failed experiments: {list(failed)}")
    
    print("\nGenerated outputs:")
    for exp_num in results.keys():
        if exp_num == 1:
            print(f"  Exp {exp_num}: exp1_boundary_topology_*.png, exp1_boundary_topology_summary.csv")
        elif exp_num == 2:
            print(f"  Exp {exp_num}: exp2_scale_leakage_*.png, exp2_scale_leakage_summary.csv")
        elif exp_num == 3:
            print(f"  Exp {exp_num}: exp3_drift_ablation_*.png, exp3_drift_ablation_*.csv")
        elif exp_num == 4:
            print(f"  Exp {exp_num}: exp4_veracity_leakage_*.png, exp4_veracity_leakage_*.csv")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
