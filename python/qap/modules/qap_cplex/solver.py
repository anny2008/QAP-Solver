"""
Subgraph Flow Decomposition (SFD) Solver

A decomposition-based formulation for QAP that decomposes the problem into
value layers and solves using constraint programming with lazy constraints.

References:
    Form3 from QAP_New_formulation
    Uses value_layer or value_only decomposition strategies
"""

# Temporary compatibility shim for NumPy 2.x + docplex
import numpy as np
np.float_ = np.float64

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from docplex.mp.model import Model
from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from cplex import SparsePair

# Handle imports for different calling contexts
try:
    from python.qap.core import Problem, Solution, write_result
    from python.qap.core.solution_io import read_warmstart
    from python.qap.decomposition import decompose_value_layer, decompose_value_only
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from qap.core import Problem, Solution, write_result
    from qap.core.solution_io import read_warmstart
    from qap.decomposition import decompose_value_layer, decompose_value_only


class QAPCplexSolver:
    """
    CPLEX solver for QAP using original formulation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize solver with configuration.

        Args:
            config (dict): Configuration dictionary
                Required keys: solver, formulation
                Optional keys:
                    - time_limit (float): Time limit in seconds (default: 3600)
                    - log_output (bool): Print solver output (default: False)
                    - is_relax (bool): Use relaxed constraints (default: False)
        """
        self.config = config
        self.solver_name = config.get("solver", "qap_cplex")
        self.use_cuts = config.get("use_cuts", True)
        self.time_limit = config.get("time_limit", 3600)
        self.log_output = config.get("log_output", False)
        self.is_relax = config.get("is_relax", False)

    def _create_model(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[Model, Dict]:
        """
        Create CPLEX model for SFD formulation.

        Args:
            problem: QAP problem instance
            subgraphs: Decomposed subgraphs {k: (f_k, G_k, G_n_k), ...}
            fixed_variables: List of (i, u) to fix x[i,u] = 1
            warmstart: Dictionary {(i,u): value} for warm-starting

        Returns:
            model: CPLEX model
            x_vars: Assignment variables
        """
        n = problem.n
        m = problem.m
        V = list(range(n))
        M = list(range(m))
        distances = problem.D
        flows = problem.F

        model = Model(name="QAP_SFD")
        
        if self.is_relax:
            x = model.continuous_var_matrix(n, m, name="x", lb=0, ub=1)
        else:
            x = model.binary_var_matrix(n, m, name="x")
            
        # check if problem has fixed assignments and add constraints
        if hasattr(problem, "fixed_assignments") and problem.fixed_assignments:
            print(f"Adding {problem.fixed_assignments} fixed assignment constraints.")
            for i, u in problem.fixed_assignments.items():
                model.add_constraint(x[i, u] == 1, ctname=f"fixed_{i}_{u}")

        # Objective function
        model.minimize(model.sum(
            distances[i, j] * flows[u, v] * x[i, u] * x[j, v] for i in V for j in V for u in M for v in M
        ))
        
        # Assignment constraints
        for i in V:
            model.add_constraint(
                model.sum(x[i, u] for u in M) <= 1,
                ctname=f"assign_facility_{i}"
            )
        for u in M:
            model.add_constraint(
                model.sum(x[i, u] for i in V) == 1,
                ctname=f"assign_location_{u}"
            )


        # Fix variables if provided
        if fixed_variables is not None:
            for i, u in fixed_variables:
                model.add_constraint(x[i, u] == 1, ctname=f"fix_x_{i}_{u}")

        # Warm-start if provided
        if warmstart is not None:
            ws = model.new_solution()
            for (i, u), val in warmstart.items():
                ws.add_var_value(x[i, u], val)
            model.add_mip_start(ws)

        # Configure CPLEX parameters
        model.parameters.timelimit = self.time_limit
        model.parameters.mip.tolerances.mipgap = 0.0
        model.parameters.optimalitytarget = 3
        
        return model, x

    def solve(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Solution:
        """
        Solve QAP using SFD formulation.

        Args:
            problem: QAP problem instance
            subgraphs: Decomposed subgraphs
            fixed_variables: Variables to fix
            warmstart: Warm-start solution

        Returns:
            Solution object
        """
        start_time = time.time()

        # Create model
        model, x = self._create_model(problem, fixed_variables, warmstart)

        # Solve
        solution = model.solve(log_output=self.log_output)

        elapsed_time = time.time() - start_time

        if solution:
            # Extract assignment
            assignment = [None] * problem.n
            for i in range(problem.n):
                for u in range(problem.n):
                    if (i, u) in x:
                        if solution.get_value(x[i, u]) > 0.5:
                            assignment[i] = u
                            break
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=assignment,
                objective=solution.objective_value,
                lower_bound=model.solve_details.best_bound,
                time=elapsed_time,
            )
        else:
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=None,
                objective=None,
                lower_bound=model.solve_details.best_bound,
                time=elapsed_time,
            )


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


    def load_warmstart_from_file(self, filepath: str) -> Dict[Tuple[int, int], float]:
        """
        Load warm-start solution from QAPLIB format file.
        
        Args:
            filepath: Path to .sln file
            
        Returns:
            dict: {(i, u): 1.0} format for warm-starting
        """
        return read_warmstart(filepath)


if __name__ == "__main__":
    print("SFD Solver Module")
    print("Subgraph Flow Decomposition formulation for QAP")
