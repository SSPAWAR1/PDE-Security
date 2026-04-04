"""
Project entry point.

Provides a small CLI for listing and running experiments.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from experiments.exp1_boundary_topology import run_experiment as run_exp1
from experiments.exp2_scale_leakage import run_experiment as run_exp2
from experiments.exp3_drift_ablation import run_experiment as run_exp3
from experiments.exp4_veracity_leakage import run_experiment as run_exp4


EXPERIMENTS: Dict[str, Callable] = {
    "exp1": run_exp1,
    "exp2": run_exp2,
    "exp3": run_exp3,
    "exp4": run_exp4,
}


def list_experiments() -> None:
    """
    Print available experiment names.
    """
    print("Available experiments:")
    for name in EXPERIMENTS:
        print(f"  - {name}")
    print("  - all")


def run_selected_experiment(name: str) -> None:
    """
    Run one experiment by name.
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")

    print(f"\nRunning {name} ...")
    EXPERIMENTS[name]()


def run_all_experiments() -> None:
    """
    Run all experiments in sequence.
    """
    for name, fn in EXPERIMENTS.items():
        print("\n" + "=" * 90)
        print(f"RUNNING {name.upper()}")
        print("=" * 90)
        fn()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run quantum leakage experiments."
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment to run: exp1, exp2, exp3, exp4, or all",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_experiments()
        return

    if args.exp is None:
        print("No experiment specified.")
        print("Use --list to see options or --exp exp1|exp2|exp3|exp4|all")
        return

    if args.exp == "all":
        run_all_experiments()
        return

    run_selected_experiment(args.exp)


if __name__ == "__main__":
    main()
