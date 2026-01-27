#!/usr/bin/env python
"""
Demo script for RLT1 CPLEX solver.

Runs the RLT1 solver on QAPLIB instances with optional warm-start solutions.

Usage:
    # Activate conda environment
    conda activate LP

    # Run on a single instance
    python demo_rlt1.py -i /path/to/instance.dat

    # Run on a single instance with warm-start
    python demo_rlt1.py -i /path/to/instance.dat -w /path/to/warmstart.json

    # Run on all instances in a directory
    python demo_rlt1.py --data-dir /path/to/QAPLIB -o results/
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from python.qap.modules.rlt1_cplex.solver import RLT1CPLEXSolver


def run_single_instance(
    instance_path: str,
    warm_start_path: str = None,
    output_dir: str = None,
    is_relax: bool = False,
    time_limit: float = 120,
) -> None:
    """Run solver on a single instance."""

    config = {
        "solver": "rlt1_cplex",
        "formulation": "rlt1",
        "is_relax": is_relax,
        "time_limit": time_limit,
        "threads": 8,
        "log_output": True,
        "preprocessing_symmetry": 5,
    }

    solver = RLT1CPLEXSolver(config)

    instance_name = Path(instance_path).stem
    output_path = None
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_dir) / f"{instance_name}_rlt1.json")

    print(f"\n{'='*70}")
    print(f"RLT1 CPLEX Solver - {instance_name}")
    print(f"{'='*70}")
    print(f"Instance: {instance_path}")
    print(f"Relax:    {is_relax}")
    print(f"Time limit: {time_limit}s")
    if warm_start_path:
        print(f"Warm-start: {warm_start_path}")
    print(f"{'='*70}")

    result = solver.solve_instance(
        instance_path, output_path=output_path, warmstart_path=warm_start_path
    )

    print(f"\nResults:")
    print(f"  Objective:   {result.objective}")
    print(f"  Lower Bound: {result.lower_bound}")
    print(f"  Gap:         {result.gap:.2f}%" if result.gap else "  Gap:         N/A")
    print(f"  Time:        {result.time:.2f}s")

    if output_path:
        print(f"\nResult saved to: {output_path}")


def run_directory(
    data_dir: str,
    output_dir: str = None,
    is_relax: bool = False,
    time_limit: float = 120,
    pattern: str = "*.dat",
) -> None:
    """Run solver on all instances in a directory."""

    data_path = Path(data_dir)
    instances = sorted(data_path.glob(pattern))

    if not instances:
        print(f"No instances found in {data_dir}")
        return

    print(f"\nFound {len(instances)} instances")

    results = []

    for instance_file in instances:
        config = {
            "solver": "rlt1_cplex",
            "formulation": "rlt1",
            "is_relax": is_relax,
            "time_limit": time_limit,
            "threads": 8,
            "log_output": False,
            "preprocessing_symmetry": 5,
        }

        solver = RLT1CPLEXSolver(config)

        instance_name = instance_file.stem
        output_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(output_dir) / f"{instance_name}_rlt1.json")

        print(f"\nSolving: {instance_name}...", end=" ", flush=True)

        result = solver.solve_instance(instance_file, output_path=output_path)

        status = "✓" if result.objective else "✗"
        print(f"{status} obj={result.objective} gap={result.gap:.2f}%" if result.gap else "obj={result.objective}")

        results.append(
            {
                "instance": instance_name,
                "objective": result.objective,
                "lower_bound": result.lower_bound,
                "gap": result.gap,
                "time": result.time,
            }
        )

    # Save summary
    if output_dir:
        summary_file = Path(output_dir) / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    total_solved = sum(1 for r in results if r["objective"])
    print(f"Solved: {total_solved}/{len(results)}")
    if total_solved > 0:
        avg_gap = sum(r["gap"] for r in results if r["gap"]) / total_solved
        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"Avg Gap: {avg_gap:.2f}%")
        print(f"Avg Time: {avg_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLT1 CPLEX Solver Demo")
    parser.add_argument("-i", "--instance", help="Path to single instance file")
    parser.add_argument("-w", "--warmstart", help="Path to warm-start file (JSON)")
    parser.add_argument("-d", "--data-dir", help="Directory with QAPLIB instances")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument(
        "-r", "--relax", action="store_true", help="Use relaxed 0-1 variables"
    )
    parser.add_argument(
        "-t", "--time-limit", type=float, default=120, help="Time limit in seconds"
    )

    args = parser.parse_args()

    if args.instance:
        run_single_instance(
            args.instance,
            warm_start_path=args.warmstart,
            output_dir=args.output,
            is_relax=args.relax,
            time_limit=args.time_limit,
        )
    elif args.data_dir:
        run_directory(
            args.data_dir,
            output_dir=args.output,
            is_relax=args.relax,
            time_limit=args.time_limit,
        )
    else:
        # Default: run on example instance
        data_dir = "/home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB"
        instance_file = Path(data_dir) / "chr12a.dat"

        if instance_file.exists():
            run_single_instance(
                str(instance_file),
                output_dir="results/",
                is_relax=False,
                time_limit=120,
            )
        else:
            print(f"Default instance not found: {instance_file}")
            print("\nUsage examples:")
            print("  python demo_rlt1.py -i /path/to/instance.dat")
            print("  python demo_rlt1.py -d /path/to/QAPLIB -o results/")
