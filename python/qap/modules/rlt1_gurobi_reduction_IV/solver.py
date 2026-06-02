"""
RLT1 Gurobi Solver Module

Relaxation-based Tightened Linear (RLT1) formulation solved with Gurobi.
Supports binary variables, relaxed 0-1 variables, and warm-start solutions.

Reference: Based on solve_QAP_RLT1_gurobi from qap_new_formulation.py
"""

import time
import numpy as np

# Fix numpy 2.0 compatibility with docplex
np.float_ = np.float64
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import gurobipy as gp
from gurobipy import GRB

from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart


class RLT1GurobiSolver:
    """RLT1 formulation solver using Gurobi."""

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
        self.solver_name = config.get("solver", "rlt1_gurobi_reduction_IV")
        self.formulation = config.get("formulation", "rlt1_reduced")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.lpmethod = config.get("lpmethod", "auto")
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)

        # Validate config
        assert self.formulation == "rlt1_reduced", "RLT1 solver requires formulation='rlt1_reduced'"

    @classmethod
    def from_config_file(cls, config_path: str) -> "RLT1GurobiSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            RLT1GurobiSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def _create_model(
            self, problem: Problem, fixed_variables: Optional[List[Tuple[int, int]]] = None
        ) -> Tuple[gp.Model, Dict[Tuple[int, int], gp.Var], Dict[Tuple[int, int, int, int], gp.Var]]:
        """
        Create Gurobi model for RLT1 formulation.

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
        # Fn0 = [(u,v) for u in M for v in M if flows[u, v] > 0]
        F0 = [(u,v) for u in M for v in M if v > u and flows[u, v]+flows[v, u] == 0]
        I0 = [u for u in M if any(flows[u, v]+flows[v, u] == 0 for v in M if v != u)]
        print(I0)
        
        model = gp.Model("QAP_RLT1")

        # Create variables
        if self.is_relax:
            x = model.addVars(n, m, vtype=GRB.CONTINUOUS if self.is_relax else GRB.BINARY, lb=0, name="x")
        else:
            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
        
        
        all_indices = [(i, u, j, v)
                for i in V
                for u in M
                for j in V
                for v in M
                if u < v and i != j]
            
        y_keys = [
                (i, u, j, v)
                for (i, u, j, v) in all_indices
                if (u, v) not in F0]
        
        y_var = model.addVars(y_keys, vtype=GRB.CONTINUOUS, lb=0, name="y")
        y = {}
        for (i,u,j,v) in y_keys:
            y[i,u,j,v] = y_var[i,u,j,v]
            y[j,v,i,u] = y_var[i,u,j,v]  # Symmetry: y[i,u,j,v] = y[j,v,i,u]
        print(f"Created {len(x)} x variables and {len(y)} y variables")
        
        # Objective
        obj = gp.quicksum(distances[i, j] * flows[u, v] * y[i, u, j, v]
                          for i in V for j in V for u in M for v in M if (i, u, j, v) in y)
        model.setObjective(obj, GRB.MINIMIZE)

        # Assignment constraints
        for i in V:
            if problem.n > problem.m:
                model.addConstr(gp.quicksum(x[i, u] for u in M) <= 1, name=f"assign_fac_{i}")
            else:
                model.addConstr(gp.quicksum(x[i, u] for u in M) == 1, name=f"assign_fac_{i}")
        for u in M:
            model.addConstr(gp.quicksum(x[i, u] for i in V) == 1, name=f"assign_loc_{u}")

        # Linking constraints for y variables
        for i in V:
            for u in M:
                if u not in I0:
                    for j in V:
                        lhs = [y[i, u, j, v] for v in M if (i, u, j, v) in y]
                        if len(lhs) > 0:
                            if problem.n > problem.m:
                                model.addConstr(gp.quicksum(lhs) <= x[i, u], name=f"link1_{i}_{u}_{j}")
                            else:
                                model.addConstr(gp.quicksum(lhs) == x[i, u], name=f"link1_{i}_{u}_{j}")
                
                for v in M:
                    # y[i,u,*,v] constraints
                    if (u, v) not in F0:
                        lhs = [y[i, u, j, v] for j in V if (i, u, j, v) in y]
                        if len(lhs) > 0:
                            model.addConstr(gp.quicksum(lhs) == x[i, u], name=f"link2_{i}_{u}_{v}")
                
        # for i in V:
        #     for j in V:
        #         if i != j:
        #             model.add_constraint(
        #                 model.sum(y[i, u, j, v] for u in M for v in M if (i, u, j, v) in y) == 1,
        #                 ctname=f"link2_{i}_{j}",
        #             )

        # Fix variables if provided
        # if fixed_variables is not None:
        #     for i, u in fixed_variables:
        #         model.add_constraint(x[i, u] == 1, ctname=f"fix_x_{i}_{u}")
        
        if hasattr(problem, "fixed_assignments") and problem.fixed_assignments is not None:
            print(f"Adding {len(problem.fixed_assignments)} fixed assignment constraints from problem instance")
            # print(problem.fixed_assignments)
            for i,u in problem.fixed_assignments.items():
                model.addConstr(x[i, u] == 1, name=f"fixed_{i}_{u}")
                print(f"Fixed assignment: x[{i},{u}] = 1")

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
        if warmstart is not None:
            for (i, u), value in warmstart.items():
                x[i, u].Start = value
                # Set y variables: y[i,u,j,v] = 1 if both x[i,u]=1 and x[j,v]=1
                for (j, v), value2 in warmstart.items():
                    if (i, u, j, v) in y or (j, v, i, u) in y:
                        y_key = y[i, u, j, v] if (i, u, j, v) in y else y[j, v, i, u]
                        y_key.Start = value * value2
        if self.lpmethod != "auto":
            if self.lpmethod == "barrier":
                model.setParam("Method", 2)  # Barrier method
            elif self.lpmethod == "dualsimplex":
                model.setParam("Method", 1)  # Dual simplex
            elif self.lpmethod == "primalsimplex":
                model.setParam("Method", 0)  # Primal simplex
        # disable crossover
        model.setParam("Crossover", 0)

        # Solve
        # model.setParam("TimeLimit", self.time_limit)
        # model.setParam("Threads", self.threads)
        model.setParam("OutputFlag", 1 if self.log_output else 0)
        # model.setParam("Symmetry", self.preprocessing_symmetry)
        model.optimize()
        
        elapsed_time = time.time() - start_time
        print("Solve with lpmethod =", model.Params.Method)
        print(f"Model created with {model.NumVars} variables and {model.NumConstrs} constraints.")
        # solution = cpx.solution
        # Extract results
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if not self.is_relax:
                assignment = {(i, u): x[i, u].X for i in range(problem.n) for u in range(problem.m) if x[i, u].X > 0.5}
                is_feasible = len(assignment) == problem.n
            else:
                is_feasible = True
                assignment = None
            objective = model.objVal if model.objVal is not None else None
            best_bound = model.objBound if model.objBound is not None else None
            # get the solution y values
            y_values = {(i, u, j, v): y[i, u, j, v].X for (i, u, j, v) in y if y[i, u, j, v].X > 1e-6}
            print(len(y_values), "y variables with values:")
            dual_values = {}
            # get the dual values
            # recompute objetive from dual values and y values for debugging
            if self.is_relax:
                objective_1 = 0
                for i in range(problem.n):
                    constr = model.getConstrByName(f"assign_fac_{i}")
                    dual_values[f"assign_fac_{i}"] = constr.Pi
                    objective_1 += constr.Pi
                for u in range(problem.m):
                    constr = model.getConstrByName(f"assign_loc_{u}")
                    dual_values[f"assign_loc_{u}"] = constr.Pi
                    objective_1 += constr.Pi
                for i in range(problem.n):
                    for u in range(problem.m):
                        for j in range(problem.n):
                            constr = model.getConstrByName(f"link1_{i}_{u}_{j}")
                            if constr is not None and constr.Pi is not None:
                                dual_values[f"link1_{i}_{u}_{j}"] = constr.Pi
                for i in range(problem.n):
                    for u in range(problem.m):
                        for v in range(problem.m):
                            constr = model.getConstrByName(f"link2_{i}_{u}_{v}")
                            if constr is not None and constr.Pi is not None:
                                dual_values[f"link2_{i}_{u}_{v}"] = constr.Pi
                print(len(dual_values), "linking constraints with dual values:")
                print(f"Objective computed from dual values: {objective_1}")
            objective_2 = 0
            for (i, u, j, v), value in y_values.items():
                objective_2 += problem.D[i, j] * problem.F[u, v] * value
            print(f"Objective computed from y values: {objective_2}")
            # save the solution y to a file for debugging
            with open(f"{self.instance}_y_solution.json", "w") as f:
                json.dump({f"y[{i},{u},{j},{v}]": value for (i, u, j, v), value in y_values.items()}, f, indent=2)
            # Create solution object
            result = Solution(
                instance=self.instance,  # Will be set by caller
                solver=self.solver_name,
                assignment=assignment if is_feasible else None,
                objective=objective,
                lower_bound=best_bound,
                time=elapsed_time,
            )

            return result
        else:
            print(f"Solver ended with status {model.status}, no solution found.")
            # No solution found
            result = Solution(
                instance=self.instance,
                solver=self.solver_name,
                assignment=None,
                objective=None,
                lower_bound=None,
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
