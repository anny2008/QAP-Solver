#!/usr/bin/env python3
"""
Quick Start Guide: Using Unified Solution I/O

This file demonstrates practical patterns for using the unified solution I/O interface
with the RLT1 SCIP solver.
"""

# ============================================================================
# Pattern 1: Basic File Reading and Writing
# ============================================================================

def pattern_1_file_io():
    """Read and write QAP solutions in QAPLIB format."""
    from qap.core.solution_io import read_solution, write_solution
    
    # Write a solution to file
    write_solution(
        filepath="my_solution.sln",
        n=12,
        assignment=[11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1],  # 0-indexed
        objective=9552.0
    )
    print("✓ Saved solution")
    
    # Read it back
    n, assignment, objective = read_solution("my_solution.sln")
    print(f"✓ Loaded solution: n={n}, obj={objective}")
    print(f"  Assignment: {assignment}")


# ============================================================================
# Pattern 2: Warm-Start from Saved Solution
# ============================================================================

def pattern_2_warmstart_from_file():
    """Load a warm-start from a saved solution file."""
    from qap.modules.rlt1_scip import RLT1SCIPSolver
    from qap.core import Problem
    
    # Configuration
    config = {
        "solver": "rlt1_scip",
        "formulation": "rlt1",
        "time_limit": 60,
        "threads": 4,
        "log_output": False
    }
    
    # Initialize solver
    solver = RLT1SCIPSolver(config)
    
    # Method 1: Using solver's built-in method
    warm_start = solver.load_warmstart_from_file("previous_solution.sln")
    
    # Method 2: Using solution_io directly
    from qap.core.solution_io import read_warmstart
    warm_start = read_warmstart("previous_solution.sln")
    
    # Load problem
    problem = Problem.from_qaplib("chr12a.dat")
    
    # Solve with warm-start
    solution = solver.solve(problem, warm_start=warm_start)
    print(f"✓ Solved with warm-start: obj={solution.objective}")


# ============================================================================
# Pattern 3: Two-Stage Solving
# ============================================================================

def pattern_3_two_stage_solving():
    """
    Use heuristic to get initial solution, then warm-start exact solver.
    Demonstrates typical optimization workflow.
    """
    from qap.modules.rlt1_scip import RLT1SCIPSolver
    from qap.core import Problem
    from qap.core.solution_io import read_warmstart, write_solution
    
    problem = Problem.from_qaplib("chr12a.dat")
    
    # Stage 1: Quick heuristic (not implemented here, use your own)
    print("Stage 1: Running heuristic...")
    heuristic_assignment = [11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1]  # example
    heuristic_objective = 9600.0  # estimated, not optimal
    
    # Save heuristic solution
    write_solution("heuristic.sln", len(heuristic_assignment), 
                   heuristic_assignment, heuristic_objective)
    print(f"  → Heuristic found obj={heuristic_objective}")
    
    # Stage 2: Exact solver with warm-start
    print("Stage 2: Running exact solver with warm-start...")
    config = {
        "solver": "rlt1_scip",
        "formulation": "rlt1",
        "time_limit": 120,
        "threads": 8,
        "log_output": True
    }
    solver = RLT1SCIPSolver(config)
    
    warm_start = read_warmstart("heuristic.sln")
    solution = solver.solve(problem, warm_start=warm_start)
    print(f"  → Exact solver found obj={solution.objective}")
    
    # Save final solution
    if solution.assignment is not None:
        write_solution("exact_solution.sln", solution.n, 
                      solution.assignment, solution.objective)
        print(f"✓ Final solution saved")


# ============================================================================
# Pattern 4: Batch Processing Multiple Instances
# ============================================================================

def pattern_4_batch_processing():
    """
    Solve multiple QAP instances and manage solutions systematically.
    """
    from pathlib import Path
    from qap.modules.rlt1_scip import RLT1SCIPSolver
    from qap.core import Problem
    from qap.core.solution_io import write_solution, read_solution
    
    # Configuration
    config = {
        "solver": "rlt1_scip",
        "formulation": "rlt1",
        "time_limit": 30,
        "threads": 4,
        "log_output": False
    }
    solver = RLT1SCIPSolver(config)
    
    # Instance directory
    instance_dir = Path("./data/QAPLIB")
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    
    # Process instances
    results = []
    for instance_file in sorted(instance_dir.glob("*.dat"))[:3]:  # first 3 instances
        instance_name = instance_file.stem
        print(f"Solving {instance_name}...")
        
        # Load and solve
        problem = Problem.from_qaplib(str(instance_file))
        solution = solver.solve_instance(str(instance_file))
        
        # Save solution
        output_file = output_dir / f"{instance_name}.sln"
        write_solution(str(output_file), problem.n, 
                      solution.assignment, solution.objective)
        
        # Record result
        results.append({
            "instance": instance_name,
            "n": problem.n,
            "objective": solution.objective,
            "file": output_file
        })
        
        print(f"  ✓ obj={solution.objective}, saved to {output_file}")
    
    # Summary
    print("\nSummary:")
    for r in results:
        print(f"  {r['instance']}: n={r['n']}, obj={r['objective']}")


# ============================================================================
# Pattern 5: Comparing Solutions
# ============================================================================

def pattern_5_compare_solutions():
    """Compare multiple solution files."""
    from pathlib import Path
    from qap.core.solution_io import read_solution
    
    solution_files = [
        "solution_v1.sln",
        "solution_v2.sln",
        "solution_v3.sln"
    ]
    
    print("Comparing solutions:")
    print("-" * 50)
    
    best_obj = float('inf')
    best_file = None
    
    for sln_file in solution_files:
        try:
            n, assignment, objective = read_solution(sln_file)
            is_best = "← BEST" if objective < best_obj else ""
            print(f"{sln_file:20s} n={n:3d}  obj={objective:10.1f} {is_best}")
            
            if objective < best_obj:
                best_obj = objective
                best_file = sln_file
                
        except FileNotFoundError:
            print(f"{sln_file:20s} NOT FOUND")
    
    print("-" * 50)
    print(f"Best: {best_file} with obj={best_obj}")


# ============================================================================
# Pattern 6: Working with Warm-Start Dictionary
# ============================================================================

def pattern_6_warmstart_dict():
    """Advanced: manipulate warm-start dictionary directly."""
    from qap.core.solution_io import read_warmstart
    
    warm_start = read_warmstart("solution.sln")
    
    # Inspect warm-start
    print(f"Warm-start has {len(warm_start)} fixed variables")
    print("First 5 assignments:")
    for i, ((facility, location), value) in enumerate(list(warm_start.items())[:5]):
        print(f"  x[{facility},{location}] = {value}")
    
    # Modify warm-start (e.g., remove some variables)
    # This is useful if you want to partially warm-start
    n = len(warm_start)
    partial_warm_start = {k: v for i, (k, v) in enumerate(warm_start.items()) if i < n//2}
    print(f"\nPartial warm-start: {len(partial_warm_start)} variables")


# ============================================================================
# Main: Run All Patterns
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    print("=" * 70)
    print("Unified Solution I/O - Quick Start Patterns")
    print("=" * 70)
    
    print("\n[Pattern 1] Basic File I/O")
    print("-" * 70)
    try:
        pattern_1_file_io()
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n[Pattern 5] Compare Solutions")
    print("-" * 70)
    try:
        pattern_5_compare_solutions()
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n[Pattern 6] Warm-Start Dictionary")
    print("-" * 70)
    try:
        pattern_6_warmstart_dict()
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n" + "=" * 70)
    print("To run other patterns with actual problem data:")
    print("  - Pattern 2: Requires previous_solution.sln and chr12a.dat")
    print("  - Pattern 3: Two-stage solving demonstration")
    print("  - Pattern 4: Batch process QAPLIB instances")
    print("=" * 70)
