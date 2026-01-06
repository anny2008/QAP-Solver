"""
RTL1 CPLEX Solver Module

Relaxation-based Tightened Linear (RTL1) formulation solved with CPLEX.
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


class RTL1CPLEXSolver:
    """RTL1 formulation solver using CPLEX."""

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
        self.solver_name = config.get("solver", "rtl1_cplex")
        self.formulation = config.get("formulation", "rtl1")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)

        # Validate config
        assert self.formulation == "rtl1", "RTL1 solver requires formulation='rtl1'"

    @classmethod
    def from_config_file(cls, config_path: str) -> "RTL1CPLEXSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            RTL1CPLEXSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def _create_model(
        self, problem: Problem, fixed_variables: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[Model, Dict, Dict]:
        """
        Create CPLEX model for RTL1 formulation.

        Args:
            problem (Problem): QAP problem instance
            fixed_variables (list, optional): List of (i, u) to fix x[i,u] = 1

        Returns:
            tuple: (model, x_vars, y_vars)
        """
        n = problem.n
        distances = problem.D
        flows = problem.F

        model = Model(name="QAP_RTL1")

        # Create variables
        if self.is_relax:
            x = model.continuous_var_matrix(n, n, name="x", lb=0, ub=1)
            
        else:
            x = model.binary_var_matrix(n, n, name="x")
            
        y = model.continuous_var_dict(
            (
                (i, u, j, v)
                for i in range(n)
                for u in range(n)
                for j in range(n)
                for v in range(n)
            ),
            name="y",
            lb=0,
            ub=1,
        )

        # Objective function
        model.minimize(
            model.sum(
                distances[i, j] * flows[u, v] * y[i, u, j, v]
                for i in range(n)
                for j in range(n)
                for u in range(n)
                for v in range(n)
            )
        )

        # Assignment constraints
        for i in range(n):
            model.add_constraint(
                model.sum(x[i, u] for u in range(n)) == 1, ctname=f"assign_fac_{i}"
            )
        for u in range(n):
            model.add_constraint(
                model.sum(x[i, u] for i in range(n)) == 1,
                ctname=f"assign_loc_{u}",
            )

        # Linking constraints for y variables
        for i in range(n):
            for u in range(n):
                for j in range(n):
                    # y[i,u,j,*] constraints
                    model.add_constraint(
                        model.sum(y[i, u, j, v] for v in range(n)) == x[i, u],
                        ctname=f"link1_{i}_{u}_{j}",
                    )
                    for v in range(n):
                        # Symmetry: y[i,u,j,v] == y[j,v,i,u]
                        model.add_constraint(
                            y[i, u, j, v] == y[j, v, i, u],
                            ctname=f"sym_{i}_{u}_{j}_{v}",
                        )
                for v in range(n):
                    # y[i,u,*,v] constraints
                    model.add_constraint(
                        model.sum(y[i, u, j, v] for j in range(n)) == x[i, u],
                        ctname=f"link2_{i}_{u}_{v}",
                    )

        # Fix variables if provided
        if fixed_variables is not None:
            for i, u in fixed_variables:
                model.add_constraint(x[i, u] == 1, ctname=f"fix_x_{i}_{u}")

        # Configure solver parameters
        model.parameters.timelimit = self.time_limit
        model.parameters.threads = self.threads
        model.parameters.preprocessing.symmetry = self.preprocessing_symmetry

        return model, x, y

    def solve(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Solution:
        """
        Solve the QAP problem using RTL1 formulation.

        Args:
            problem (Problem): QAP problem instance
            fixed_variables (list, optional): List of (i, u) to fix x[i,u] = 1
            warmstart (dict, optional): Dictionary {(i,u): value} for warm-start

        Returns:
            Solution: Solution object with result
        """
        start_time = time.time()

        # Create model
        model, x, y = self._create_model(problem, fixed_variables)

        # Add warm-start if provided
        # docplex's add_mip_start requires a SolveSolution object
        if warmstart is not None:
            # Create a SolveSolution object for warm-start
            ws_solution = model.new_solution()
            for (i, u), value in warmstart.items():
                ws_solution.add_var_value(x[i, u], value)
                # Set y variables: y[i,u,j,v] = 1 if both x[i,u]=1 and x[j,v]=1
                for (j, v), value2 in warmstart.items():
                    ws_solution.add_var_value(y[i, u, j, v], value*value2)
            # Add MIP start using the solution object
            model.add_mip_start(ws_solution)

        # Solve
        solution = model.solve(log_output=self.log_output)

        elapsed_time = time.time() - start_time

        # Extract results
        if solution:
            # Extract assignment from x variables
            assignment = []
            is_feasible = True
            for i in range(problem.n):
                best_u = -1
                best_val = -1
                for u in range(problem.n):
                    val = solution.get_value(x[i, u])
                    if val > best_val:
                        best_val = val
                        best_u = u
                
                if best_val > 0.5:
                    assignment.append(best_u)
                else:
                    # Fractional solution - round best value
                    assignment.append(best_u)
                    is_feasible = False

            # Compute actual QAP objective from assignment if feasible
            if is_feasible and len(assignment) == problem.n:
                actual_objective = problem.evaluate_assignment(assignment)
            else:
                # Can't evaluate non-integer solution
                actual_objective = solution.objective_value
            
            # The RTL1 objective value is a lower bound on the QAP
            # (since RTL1 linearizes and relaxes the QAP)
            rtl1_lower_bound = solution.objective_value

            # Create solution object
            result = Solution(
                instance="unknown",  # Will be set by caller
                solver=self.solver_name,
                assignment=assignment if is_feasible else None,
                objective=actual_objective,
                lower_bound=rtl1_lower_bound,
                time=elapsed_time,
            )

            return result
        else:
            # No solution found
            result = Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=None,
                objective=None,
                lower_bound=model.solve_details.best_bound
                if model.solve_details
                else None,
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
        problem = Problem.from_qaplib(instance_path)

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

    def benchmark_instance(self, instance_path: str) -> Dict[str, Any]:
        """
        Benchmark solver on a single instance.

        Args:
            instance_path (str): Path to QAPLIB instance file

        Returns:
            dict: Results with timing and statistics
        """
        instance_name = Path(instance_path).stem

        # Solve without warm-start
        print(f"\n{'='*60}")
        print(f"Instance: {instance_name}")
        print(f"{'='*60}")

        result = self.solve_instance(instance_path)

        results = {
            "instance": instance_name,
            "solver": self.solver_name,
            "is_relax": self.is_relax,
            "objective": result.objective,
            "lower_bound": result.lower_bound,
            "gap": result.gap,
            "time": result.time,
        }

        # Print results
        print(f"Objective:   {result.objective}")
        print(f"Lower Bound: {result.lower_bound}")
        print(f"Gap:         {result.gap:.2f}%" if result.gap else "Gap: N/A")
        print(f"Time:        {result.time:.2f}s")

        return results


# Example usage
if __name__ == "__main__":
    import sys

    # Create solver
    config = {
        "solver": "rtl1_cplex",
        "formulation": "rtl1",
        "is_relax": False,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = RTL1CPLEXSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="rtl1_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )
