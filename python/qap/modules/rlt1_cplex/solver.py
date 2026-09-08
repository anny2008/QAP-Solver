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


class RLT1CPLEXSolver:
    """RLT1 formulation solver using CPLEX."""

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
        self.solver_name = config.get("solver", "rlt1_cplex")
        self.formulation = config.get("formulation", "rlt1")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)
        self.lpmethod = config.get("lpmethod", "auto")

        # Validate config
        assert self.formulation == "rlt1", "RLT1 solver requires formulation='rlt1'"

    @classmethod
    def from_config_file(cls, config_path: str) -> "RLT1CPLEXSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            RLT1CPLEXSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def _create_model(
        self, problem: Problem, fixed_variables: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[Model, Dict, Dict]:
        """
        Create CPLEX model for RLT1 formulation.

        Args:
            problem (Problem): QAP problem instance
            fixed_variables (list, optional): List of (i, u) to fix x[i,u] = 1

        Returns:
            tuple: (model, x_vars, y_vars)
        """
        n = problem.n
        m = problem.m
        distances = problem.D
        flows = problem.F
        M = list(range(m))
        V = list(range(n))

        model = Model(name="QAP_RLT1")

        # Create variables
        if self.is_relax:
            x = model.continuous_var_matrix(n, m, name="x", lb=0, ub=1)
            
        else:
            x = model.binary_var_matrix(n, m, name="x")
            
        y = model.continuous_var_dict(
            (
                (i, u, j, v)
                for i in V
                for u in M
                for j in V
                for v in M
                if i < j or (i == j and u == v)
            ),
            name="y",
            lb=0,
            ub=1,
        )

        # Objective function
        model.minimize(
            model.sum(
                distances[i, j] * flows[u, v] * (  y[i, u, j, v] if (i, u, j, v) in y else y[j, v, i, u] if (j, v, i, u) in y else 0 )
                for i in V
                for j in V
                for u in M
                for v in M
            )
        )

        # Assignment constraints
        for i in V:
            model.add_constraint(
                model.sum(x[i, u] for u in M) <= 1, ctname=f"assign_fac_{i}"
            )
        for u in M:
            model.add_constraint(
                model.sum(x[i, u] for i in V) == 1,
                ctname=f"assign_loc_{u}",
            )

        # Linking constraints for y variables
        for i in V:
            for u in M:
                for j in V:
                    # y[i,u,j,*] constraints
                    model.add_constraint(
                        model.sum(y[i, u, j, v] if (i, u, j, v) in y else y[j, v, i, u] if (j, v, i, u) in y else 0 for v in M) <= x[i, u],
                        ctname=f"link1_{i}_{u}_{j}",
                    )
                for v in M:
                    # y[i,u,*,v] constraints
                    model.add_constraint(
                        model.sum(y[i, u, j, v] if (i, u, j, v) in y else y[j, v, i, u] if (j, v, i, u) in y else 0 for j in V) == x[i, u],
                        ctname=f"link2_{i}_{u}_{v}",
                    )

        # Fix variables if provided
        if fixed_variables is not None:
            for i, u in fixed_variables:
                model.add_constraint(x[i, u] == 1, ctname=f"fix_x_{i}_{u}")

        # Configure solver parameters
        # model.parameters.timelimit = self.time_limit
        # model.parameters.threads = self.threads
        # model.parameters.preprocessing.symmetry = self.preprocessing_symmetry

        return model, x, y

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
                    if (i, u, j, v) in y or (j, v, i, u) in y:
                        ws_solution.add_var_value(y[i, u, j, v] if (i, u, j, v) in y else y[j, v, i, u], value*value2)
            # Add MIP start using the solution object
            model.add_mip_start(ws_solution)

        # Solve
        if self.lpmethod == "barrier":
            model.parameters.lpmethod = 4  # Barrier method
            print(f"Using barrier method for CPLEX.")
        elif self.lpmethod == "dual_simplex":
            model.parameters.lpmethod = 2  # Dual simplex
            print(f"Using dual simplex method for CPLEX.")
        elif self.lpmethod == "primal_simplex":
            model.parameters.lpmethod = 1  # Primal simplex
            print(f"Using primal simplex method for CPLEX.")
        else:
            print(f"Using default LP method (auto) for CPLEX.")
        model.parameters.timelimit = self.time_limit
            
        solution = model.solve(log_output=self.log_output)
        cpx = model.get_cplex()
        
        elapsed_time = time.time() - start_time
        # solution = cpx.solution
        # Extract results
        if solution:
            # Extract assignment from x variables
            num_y = sum(1 for var in y.values() if solution.get_value(var) > 1e-5)
            print(f"Number of positive y variables: {num_y}: {num_y / len(y) * 100:.2f}%")
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
            # print the number of nodes visited in the branch and bound tree
            nb_nodes = getattr(model.solve_details, "nb_nodes_processed", None)
            if nb_nodes is None:
                try:
                    nb_nodes = model.cplex.solution.progress.get_num_nodes_processed()
                except Exception:
                    nb_nodes = "unavailable"
            print(f"Number of nodes explored: {nb_nodes}")
            # Compute actual QAP objective from assignment if feasible
            # if is_feasible and len(assignment) == problem.n:
            #     actual_objective = problem.evaluate_assignment(assignment)
            # else:
            #     # Can't evaluate non-integer solution
            #     actual_objective = solution.get_objective_value()
            
            # check solution feasibility
            is_feasible = True
            # check fixed assignments
            
            
            # # sum_i x_i_u == 1 for all u
            # for u in range(problem.n):
            #     lhs = np.sum(solution.get_value(x[i, u]) for i in range(problem.n) )
            #     if abs(lhs - 1) > 1e-5:
            #         print(f"Warning: sum_i x_i_{u} = {lhs} != 1.")
            #         is_feasible = False
            #         break
            # # sum_u x_i_u == 1 for all i
            # for i in range(problem.n):
            #     lhs = np.sum(solution.get_value(x[i, u]) for u in range(problem.n) )
            #     if abs(lhs - 1) > 1e-5:
            #         print(f"Warning: sum_u x_{i}_u = {lhs} != 1.")
            #         is_feasible = False
            #         break
            # # sum_v y_i_u_j_v == x_i_u for all i,u,j
            # for i in range(problem.n):
            #     for u in range(problem.n):
            #         for j in range(problem.n):
            #             lhs = np.sum(solution.get_value(y[i, u, j, v]) if (i, u, j, v) in y else solution.get_value(y[j, v, i, u]) if (j, v, i, u) in y else 0 for v in range(problem.n))
            #             if abs(lhs - solution.get_value(x[i, u])) > 1e-5:
            #                 print(f"Warning: sum_v y_{i}_{u}_{j}_v = {lhs} != x_{i}_{u} = {solution.get_value(x[i, u])}.")
            #                 is_feasible = False
            #                 break
            #         if not is_feasible:
            #             break
            #     if not is_feasible:
            #         break
            # # sum_j y_i_u_j_v == x_i_u for all i,u,v
            # for i in range(problem.n):
            #     for u in range(problem.n):
            #         for v in range(problem.n):
            #             lhs = np.sum(solution.get_value(y[i, u, j, v]) if (i, u, j, v) in y else solution.get_value(y[j, v, i, u]) if (j, v, i, u) in y else 0 for j in range(problem.n))
            #             if abs(lhs - solution.get_value(x[i, u])) > 1e-5:
            #                 print(f"Warning: sum_j y_{i}_{u}_j_{v} = {lhs} != x_{i}_{u} = {solution.get_value(x[i, u])}.")
            #                 is_feasible = False
            #                 break
            #         if not is_feasible:
            #             break
            #     if not is_feasible:
            #         break
            
            # Create solution object
            result = Solution(
                instance=self.instance,  # Will be set by caller
                solver=self.solver_name,
                assignment=assignment if is_feasible else None,
                objective=cpx.solution.get_objective_value(),
                lower_bound=model.solve_details.best_bound,
                time=elapsed_time,
            )

            return result
        else:
            # No solution found
            result = Solution(
                instance=self.instance,
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
        if "QAPLIB" in instance_path:
            print(f"Loading QAPLIB instance from {instance_path}")
            problem = Problem.from_qaplib(instance_path)
        else:
            print(f"Loading custom instance from {instance_path}")
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
        
        
        if np.all(problem.F == problem.F.T) and not np.all(problem.D == problem.D.T):
            # convert distance matrix to symmetric by averaging with its transpose
            new_D = (problem.D + problem.D.T) / 2
            problem.D = new_D
            print("Converted distance matrix to symmetric for RLT1 formulation.")
        elif np.all(problem.D == problem.D.T) and not np.all(problem.F == problem.F.T):
            # convert flow matrix to symmetric by averaging with its transpose
            new_F = (problem.F + problem.F.T) / 2
            problem.F = new_F
            print("Converted flow matrix to symmetric for RLT1 formulation.")
        

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
        "solver": "rlt1_cplex",
        "formulation": "rlt1",
        "is_relax": False,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = RLT1CPLEXSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="rlt1_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )
