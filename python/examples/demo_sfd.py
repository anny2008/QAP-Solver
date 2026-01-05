"""
SFD Solver Example

Demonstrates Subgraph Flow Decomposition (SFD) solver with warm-start.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from python.qap.modules.sfd import SFDSolver
from python.qap.core import Problem
from python.qap.core.solution_io import write_solution, read_warmstart


def example_sfd_solver():
    """Demonstrate SFD solver usage."""
    
    # Configuration
    config = {
        "solver": "sfd",
        "formulation": "sfd",
        "decomposition": "value_layer",  # or 'value_only'
        "use_cuts": True,
        "time_limit": 600,  # 10 minutes
        "threads": 8,
        "log_output": True,
        "is_relax": False,
    }
    
    # Create solver
    solver = SFDSolver(config)
    
    print("SFD Solver Example")
    print("=" * 60)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Note: This is a demonstration structure
    # The actual SFD solver requires QAPProblem class and decomposition functions
    # from QAP_New_formulation module
    
    print("\nUsage Example:")
    print("""
    # Load problem
    problem = Problem.from_qaplib("chr12a.dat")
    
    # Create decomposition (from qap_new_formulation module)
    from qap_new_formulation import decompose_flow_by_value_layer
    subgraphs = decompose_flow_by_value_layer(problem)
    
    # Solve without warm-start
    solution = solver.solve(problem, subgraphs)
    print(f"Objective: {solution.objective}")
    
    # Or with warm-start from file
    warm_start = solver.load_warmstart_from_file("previous.sln")
    solution_with_ws = solver.solve(problem, subgraphs, warm_start=warm_start)
    """)


if __name__ == "__main__":
    example_sfd_solver()
