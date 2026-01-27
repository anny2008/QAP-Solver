#!/usr/bin/env python3
"""
Demo script for RLT1 SCIP Solver

Shows how to:
- Solve single instances
- Batch solve multiple instances
- Use warm-start solutions
- Configure the solver
"""

import sys
import json
import argparse
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.qap.core import Problem, Solution
from python.qap.modules.rlt1_scip import RLT1SCIPSolver


def solve_single_instance(instance_path: str, config_dict: dict = None, warm_start: bool = False):
    """
    Solve a single QAP instance.
    
    Args:
        instance_path: Path to QAPLIB instance file
        config_dict: Configuration dictionary (optional)
        warm_start: Whether to use warm-start (loads .sln file if available)
    """
    print(f"\n{'='*70}")
    print(f"Solving: {Path(instance_path).name}")
    print(f"{'='*70}")
    
    # Default config
    if config_dict is None:
        config_dict = {
            "solver": "rlt1_scip",
            "formulation": "rlt1",
            "is_relax": False,
            "time_limit": 120,
            "threads": 8,
            "log_output": True,
            "preprocessing_symmetry": 5,
        }
    
    # Create solver
    solver = RLT1SCIPSolver(config_dict)
    
    # Try to load warm-start solution
    warm_start_path = None
    if warm_start:
        sln_path = Path(instance_path).with_suffix(".sln")
        if sln_path.exists():
            print(f"Using warm-start from {sln_path.name}")
            warm_start_path = str(sln_path)
    
    # Solve
    try:
        solution = solver.solve_instance(
            instance_path,
            warm_start_path=warm_start_path,
            output_path=f"rlt1_scip_{Path(instance_path).stem}.json"
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  Instance: {solution.instance}")
        print(f"  Feasible: {solution.feasible}")
        print(f"  Objective: {solution.objective}")
        print(f"  Lower Bound: {solution.lower_bound}")
        print(f"  Gap: {solution.gap:.2f}%" if solution.gap else "  Gap: N/A")
        print(f"  Time: {solution.time:.2f}s")
        
        return solution
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_solve_directory(directory: str, pattern: str = "*.dat", time_limit: int = 60):
    """
    Solve all instances matching pattern in directory.
    
    Args:
        directory: Path to directory containing instances
        pattern: File pattern to match (default: *.dat)
        time_limit: Time limit per instance in seconds
    """
    print(f"\n{'='*70}")
    print(f"Batch solving directory: {directory}")
    print(f"{'='*70}")
    
    path = Path(directory)
    if not path.exists():
        print(f"Error: Directory not found: {directory}")
        return []
    
    # Find instances
    instances = sorted(path.glob(pattern))
    print(f"Found {len(instances)} instances\n")
    
    if not instances:
        print("No instances found!")
        return []
    
    # Create config
    config = {
        "solver": "rlt1_scip",
        "formulation": "rlt1",
        "is_relax": False,
        "time_limit": time_limit,
        "threads": 8,
        "log_output": False,
        "preprocessing_symmetry": 5,
    }
    
    results = []
    start_batch = time.time()
    
    for idx, instance_path in enumerate(instances, 1):
        instance_name = instance_path.stem
        print(f"[{idx}/{len(instances)}] {instance_name}...", end=" ", flush=True)
        
        try:
            solver = RLT1SCIPSolver(config)
            start = time.time()
            
            solution = solver.solve_instance(str(instance_path))
            elapsed = time.time() - start
            
            obj_str = f"{solution.objective:.0f}" if solution.objective else "N/A"
            lb_str = f"{solution.lower_bound:.0f}" if solution.lower_bound else "N/A"
            gap_str = f"{solution.gap:.2f}%" if solution.gap else "N/A"
            
            print(f"✓ Obj={obj_str}, LB={lb_str}, Gap={gap_str}, Time={elapsed:.2f}s")
            
            results.append({
                "instance": instance_name,
                "feasible": solution.feasible,
                "objective": solution.objective,
                "lower_bound": solution.lower_bound,
                "gap": solution.gap,
                "time": elapsed,
            })
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "instance": instance_name,
                "error": str(e),
            })
    
    total_time = time.time() - start_batch
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total instances: {len(instances)}")
    print(f"Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in results if 'error' in r)}")
    print(f"Total time: {total_time:.2f}s")
    
    # Print table
    print(f"\n{'Instance':<15} {'Feasible':<10} {'Objective':<15} {'Gap':<10} {'Time':<10}")
    print("-" * 60)
    for r in results:
        if "error" not in r:
            inst = r['instance'][:14]
            feas = "Yes" if r['feasible'] else "No"
            obj = f"{r['objective']:.0f}" if r['objective'] else "N/A"
            gap = f"{r['gap']:.2f}%" if r['gap'] else "N/A"
            print(f"{inst:<15} {feas:<10} {obj:<15} {gap:<10} {r['time']:<10.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="RLT1 SCIP Solver Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve single instance
  python demo_rlt1_scip.py -i /path/to/chr12a.dat
  
  # Solve with warm-start
  python demo_rlt1_scip.py -i /path/to/chr12a.dat -w
  
  # Batch solve with custom time limit
  python demo_rlt1_scip.py -d /path/to/instances -t 60
  
  # Relaxed mode (continuous variables)
  python demo_rlt1_scip.py -i /path/to/chr12a.dat -r
        """
    )
    
    parser.add_argument(
        "-i", "--instance",
        help="Path to instance file (QAPLIB format)"
    )
    parser.add_argument(
        "-d", "--directory",
        help="Directory containing instances"
    )
    parser.add_argument(
        "-w", "--warm-start",
        action="store_true",
        help="Use warm-start solutions (.sln files)"
    )
    parser.add_argument(
        "-t", "--time-limit",
        type=int,
        default=120,
        help="Time limit in seconds (default: 120)"
    )
    parser.add_argument(
        "-r", "--relax",
        action="store_true",
        help="Use relaxed 0-1 variables instead of binary"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose SCIP output"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = {
        "solver": "rlt1_scip",
        "formulation": "rlt1",
        "is_relax": args.relax,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "log_output": args.verbose,
        "preprocessing_symmetry": 5,
    }
    
    if args.instance:
        solve_single_instance(args.instance, config, args.warm_start)
    elif args.directory:
        batch_solve_directory(args.directory, time_limit=args.time_limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
