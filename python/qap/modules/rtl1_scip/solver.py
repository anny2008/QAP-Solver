"""
RTL1 SCIP Solver Module

Relaxation-based Tightened Linear (RTL1) formulation solved with SCIP.
Uses PySCIPOpt Python interface for maximum compatibility.
Supports binary variables, relaxed 0-1 variables, and warm-start solutions.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from pyscipopt import Model, quicksum

# Handle imports for different calling contexts
try:
    from python.qap.core import Problem, Solution, write_result
    from python.qap.core.solution_io import read_warmstart
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from qap.core import Problem, Solution, write_result
    from qap.core.solution_io import read_warmstart


class RTL1SCIPSolver:
    """RTL1 formulation solver using SCIP via PySCIPOpt."""

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
        self.solver_name = config.get("solver", "rtl1_scip")
        self.formulation = config.get("formulation", "rtl1")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)

        # Validate config
        assert self.formulation == "rtl1", "RTL1 solver requires formulation='rtl1'"

    @classmethod
    def from_config_file(cls, config_path: str) -> "RTL1SCIPSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            RTL1SCIPSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def _create_model(
        self, problem: Problem, fixed_variables: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[Model, Dict, Dict]:
        """
        Create SCIP model for RTL1 formulation.

        Args:
            problem (Problem): QAP problem instance
            fixed_variables (list, optional): List of (i, u) to fix x[i,u] = 1

        Returns:
            tuple: (model, x_vars, y_vars)
        """
        n = problem.n
        distances = problem.D
        flows = problem.F

        model = Model("QAP_RTL1")

        # Configure solver
        model.hideOutput(not self.log_output)
        model.setRealParam("limits/time", self.time_limit)

        # Create variables
        x_vars = {}
        y_vars = {}

        if self.is_relax:
            # Continuous 0-1 variables for x (relaxation of binary)
            for i in range(n):
                for u in range(n):
                    x_vars[(i, u)] = model.addVar(
                        lb=0, ub=1, vtype="C", name=f"x_{i}_{u}"
                    )
        else:
            # Binary variables for x (assignment variables)
            for i in range(n):
                for u in range(n):
                    x_vars[(i, u)] = model.addVar(
                        vtype="B", name=f"x_{i}_{u}"
                    )
        # CONTINUOUS variables for y (RTL1 linearization variables are always continuous!)
        for i in range(n):
            for u in range(n):
                for j in range(n):
                    for v in range(n):
                        y_vars[(i, u, j, v)] = model.addVar(
                            lb=0, ub=1, vtype="C", name=f"y_{i}_{u}_{j}_{v}"
                        )

        # Objective function: min Σ F[u,v] * D[i,j] * y[i,u,j,v]
        # F[u,v] = flow between facilities u,v
        # D[i,j] = distance between locations i,j
        # y[i,u,j,v] linearizes x[i,u] * x[j,v]
        obj = quicksum(
            flows[u, v] * distances[i, j] * y_vars[(i, u, j, v)]
            for i in range(n)
            for j in range(n)
            for u in range(n)
            for v in range(n)
        )
        model.setObjective(obj, "minimize")

        # Assignment constraints
        # Each facility to exactly one location
        for i in range(n):
            model.addCons(quicksum(x_vars[(i, u)] for u in range(n)) == 1,
                         name=f"assign_fac_{i}")

        # Each location to exactly one facility
        for u in range(n):
            model.addCons(quicksum(x_vars[(i, u)] for i in range(n)) == 1,
                         name=f"assign_loc_{u}")

        # Linking constraints: Σ_j y[i,u,j,v] = x[i,u] (for all i,u,v)
        for i in range(n):
            for u in range(n):
                for v in range(n):
                    model.addCons(
                        quicksum(y_vars[(i, u, j, v)] for j in range(n)) == x_vars[(i, u)],
                        name=f"link1_{i}_{u}_{v}",
                    )

        # Linking constraints: Σ_v y[i,u,j,v] = x[i,u] (for all i,u,j)
        for i in range(n):
            for u in range(n):
                for j in range(n):
                    model.addCons(
                        quicksum(y_vars[(i, u, j, v)] for v in range(n)) == x_vars[(i, u)],
                        name=f"link2_{i}_{u}_{j}",
                    )

        # Symmetry constraints: y[i,u,j,v] = y[j,v,i,u] (for all i,u,j,v)
        for i in range(n):
            for u in range(n):
                for j in range(n):
                    for v in range(n):
                        model.addCons(
                            y_vars[(i, u, j, v)] == y_vars[(j, v, i, u)],
                            name=f"link3_{i}_{u}_{j}_{v}",
                        )

        # Fix variables if provided
        if fixed_variables is not None:
            for i, u in fixed_variables:
                model.addCons(x_vars[(i, u)] == 1, name=f"fix_x_{i}_{u}")

        return model, x_vars, y_vars

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
        model, x_vars, y_vars = self._create_model(problem, fixed_variables)

        # Add warm-start if provided
        if warmstart is not None:
            # Create a partial solution for SCIP
            partial_sol = model.createPartialSol()
            
            x_count = 0
            y_count = 0
            
            for key in x_vars:
                value = warmstart.get(key, 0.0)
                model.setSolVal(partial_sol, x_vars[key], value)
                x_count += 1
                
            
            for key in y_vars:
                value = warmstart.get(key, 0.0)
                model.setSolVal(partial_sol, y_vars[key], value)
                y_count += 1
            
            
            # Try to add the solution
            accepted = model.addSol(partial_sol)
            
            if self.log_output:
                if accepted:
                    print(f"✓ Warm-start accepted: {x_count} x-vars, {y_count} y-vars")
                else:
                    print(f"⚠ Warm-start not accepted: {x_count} x-vars, {y_count} y-vars (may still be used)")
                        

        # Solve
        model.optimize()

        elapsed_time = time.time() - start_time

        # Extract results
        solution_found = model.getStatus() == "optimal" or model.getStatus() == "feasible"
        assignment = None
        is_feasible = False

        if solution_found:
            best_sol = model.getBestSol()
            
            # Extract assignment from x variables
            assignment = []
            for i in range(problem.n):
                best_u = -1
                best_val = -1
                for u in range(problem.n):
                    val = model.getSolVal(best_sol, x_vars[(i, u)])
                    if val > best_val:
                        best_val = val
                        best_u = u

                if best_u >= 0:
                    assignment.append(best_u)
            
            if self.log_output:
                print(f"DEBUG: Extracted assignment: {assignment}")

            # Check if integer feasible (all values close to 0 or 1)
            is_feasible = all(
                abs(model.getSolVal(best_sol, x_vars[(i, u)]) - round(model.getSolVal(best_sol, x_vars[(i, u)]))) < 1e-6
                for i in range(problem.n)
                for u in range(problem.n)
            )

            # Compute actual QAP objective if feasible
            if is_feasible and len(assignment) == problem.n:
                actual_objective = problem.evaluate_assignment(assignment)
            else:
                # Use model objective (RTL1 linearization)
                actual_objective = model.getSolObjVal(best_sol)
        else:
            # No solution found
            actual_objective = None

        # Get lower bound
        lower_bound = model.getDualbound()

        # Create solution object
        result = Solution(
            instance="unknown",  # Will be set by caller
            solver=self.solver_name,
            assignment=assignment if is_feasible else None,
            objective=actual_objective,
            lower_bound=lower_bound,
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
            warmstart_path (str, optional): Path to warm-start file (JSON dict)

        Returns:
            Solution: Solution object
        """
        # Load problem
        problem = Problem.from_qaplib(instance_path)

        # Load warm-start if provided
        warmstart = None
        if warmstart_path:
            warmstart = self.load_warmstart_from_file(warmstart_path)
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
            filepath (str): Path to .sln file
            
        Returns:
            dict: {(i, u): 1.0} format for warm-starting
        """
        warmstart = {}
        warmstart_dict = read_warmstart(filepath)
        count_y_for_debug = 0
        for (i,u), value in warmstart_dict.items():
            warmstart[(i,u)] = value
            
            for (j,v), value2 in warmstart_dict.items():
                warmstart[(i,u,j,v)] = value*value2
                count_y_for_debug += 1
        return warmstart

    def benchmark_instance(self, instance_path: str) -> Dict[str, Any]:
        """
        Benchmark solver on a single instance.

        Args:
            instance_path (str): Path to QAPLIB instance file

        Returns:
            dict: Results with timing and statistics
        """
        instance_name = Path(instance_path).stem

        print(f"\n{'='*60}")
        print(f"Instance: {instance_name}")
        print(f"Solver: RTL1 SCIP (PySCIPOpt)")
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
        "solver": "rtl1_scip",
        "formulation": "rtl1",
        "is_relax": False,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = RTL1SCIPSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="rtl1_scip_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )

