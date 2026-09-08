"""
RLT1 CPLEX Solver Module

Relaxation-based Tightened Linear (RLT1) formulation solved with CPLEX.
Supports binary variables, relaxed 0-1 variables, and warm-start solutions.

Reference: Based on solve_QAP_RLT1_cplex from qap_new_formulation.py
"""

import time
import numpy as np

# Fix numpy 2.0 compatibility with docplex
np.float_ = np.float64
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from docplex.mp.model import Model

from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart
import scipy.sparse as sp
from numpy.linalg import norm

from .find_variables import find_variables
from .find_neighbours import find_neighbours

from .build_Ae import build_Ae
from .build_BB import build_BB
from .build_DD import build_DD

from .addmm_csdp import *



class CSDPSolver:
    """CSDP solver for the Quadratic Assignment Problem."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize solver with configuration.

        Args:
            config (dict): Configuration dictionary
                Required keys: solver, formulation
                Optional keys:
                    - is_relax (bool): Use relaxed 0-1 variables (default: False)
                    - time_limit (float): Time limit in seconds (default: 120)
                    - threads (int): Number of threads (default: 8)
                    - log_output (bool): Print solver output (default: False)
                    - preprocessing_symmetry (int): Symmetry level (default: 5)
        """
        self.config = config
        self.instance_path = config.get("instance", "")
        self.instance = Path(self.instance_path).stem if self.instance_path else "unknown"
        self.solver_name = config.get("solver", "admm_solver")
        self.formulation = config.get("formulation", "sdp")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)


    @classmethod
    def from_config_file(cls, config_path: str) -> "ADMMSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            ADMMSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def solve(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Solution:
        """
        Solve the QAP problem using RLT1 formulation.

        Args:
            problem (Problem): QAP problem instance
            fixed_variables (list, optional): List of (i, u) to fix x[i,u] = 1
            warmstart (dict, optional): Dictionary {(i,u): value} for warm-start

        Returns:
            Solution: Solution object with result
        """

        A = problem.D
        B = -problem.F

        print(A)
        print(B)


        I = np.eye(A.shape[0])

        print(
            np.trace(A @ I @ B @ I.T)
        )

        for _ in range(10):
            p = np.random.permutation(A.shape[0])
            P = np.eye(A.shape[0])[p]

            val = np.trace(A @ P @ B @ P.T)

            print(val)


        n = problem.n
        solver = CSDPADMM(
            clique_size=2,
            max_iter=50,
            tol=1e-4
        )

        start_time = time.time()
        results = solver.solve(
            A,
            B
        )
        

        elapsed_time = time.time() - start_time
        
        result = Solution(
            instance=self.instance,  # Will be set by caller
            solver=self.solver_name,
            assignment=None,
            objective=0,
            lower_bound=0,
            time=elapsed_time,
        )

        return result
    
    def solve_instance(
        self,
        instance_path: str,
        output_path: Optional[str] = None,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart_path: Optional[str] = None,
    ) -> Solution:
        """
        Solve a problem instance from file.

        Args:
            instance_path (str): Path to QAPLIB instance file
            output_path (str, optional): Path to save result JSON
            fixed_variables (list, optional): List of (i, u) to fix
            warmstart_path (str, optional): Path to warmstart file (JSON dict)

        Returns:
            Solution: Solution object
        """
        # Load problem
        if "QAPLIB" in instance_path:
            problem = Problem.from_qaplib(instance_path)
        else:
            problem = Problem.from_full_instance(
                matrix_file=instance_path,
                workstations_file=instance_path.replace(".txt", "_workstations.txt"),
                machines_file=instance_path.replace(".txt", "_machines.txt"),
                fixed_file=instance_path.replace(".txt", "_fixed.txt")
            )
            
        # if n > m, we need to pad the problem to make it square
        if problem.n > problem.m:
            print(f"Padding problem from n={problem.n}, m={problem.m} to n={problem.n}, m={problem.n}")
            F = np.zeros((problem.n, problem.n))
            F[:problem.m, :problem.m] = problem.F
            problem.F = F

        # Load warm-start if provided
        if warmstart_path is not None:
            warmstart = read_warmstart(warmstart_path)
        else:
            warmstart = None

        # Solve
        solution = self.solve(
            problem,
            fixed_variables=fixed_variables,
            warmstart=warmstart,
        )

        # Update instance name
        solution.instance = Path(instance_path).stem

        # Save result if output path provided
        if output_path:
            write_result(solution, output_path)

        return solution

# Example usage
if __name__ == "__main__":
    import sys

    # Create solver
    config = {
        "solver": "admm",
        "formulation": "sdp",
        "is_relax": True,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = CSDPSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="admm_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )
