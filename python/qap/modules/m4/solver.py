"""
M4 CPLEX Solver Module
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
from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from cplex import SparsePair


from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart

def identify_one_violated_constraint(solution, y_vars, n, eps=1e-6):
    """
    Identify violated constraints in the current solution.
    
    Args:
        solution: Current solution object
        y_vars: Binary assignment variables y[i,u,j,v]
        n: Problem size
        eps: Tolerance for violation detection
    Returns:
        Tuple (i,u,j) or (i,u,v) for which the constraint is violated, or None if none found
    """
    violated = None
    inds = [(i,u,j) for i in range(n) for u in range(n) for j in range(n)]
    for (i,u,j) in inds:
        if j == 0:
            continue
        left_sum = np.sum([solution.get_value(y_vars[i, u, 0, vv]) for vv in range(n) if (i,u,j,vv) in y_vars])
        right_sum = np.sum([solution.get_value(y_vars[i, u, j, vv]) for vv in range(n) if (i,u,j,vv) in y_vars])
        if abs(left_sum - right_sum) > eps:
            return "j", (i,u,j)
    for (i,u,v) in inds:
        left_sum = np.sum([solution.get_value(y_vars[i, u, 0, vv]) for vv in range(n) if (i,u,0,vv) in y_vars])
        right_sum = np.sum([solution.get_value(y_vars[i, u, jj, v]) for jj in range(n) if (i,u,jj,v) in y_vars])
        if abs(left_sum - right_sum) > eps:
            return "v", (i,u,v)
    return None

class M4LazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    """
    Lazy constraint callback for M4 formulation.
    
    Implements valid inequality generation for improving the bound.
    """
    
    def __init__(self, env):
        """Initialize callback."""
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_lazy = 0
        self.eps = 1e-6

    def initialize(self, y_vars: Dict, qap_problem: Problem):
        """
        Initialize callback with model variables and problem data.
        
        Args:
            y_vars: Binary assignment variables y[i,u,j,v]
            qap_problem: QAP problem instance
        """
        self.y = y_vars
        self.qap_problem = qap_problem

    def __call__(self):
        """Called by CPLEX at each lazy constraint check."""
        # Build solution from current node
        sol = self.make_complete_solution()
        y = self.y
        n = self.qap_problem.n
        # Find all variables set to 1.0
        count_one = 0
        violated_index = identify_one_violated_constraint(sol, y, n, self.eps)
        if violated_index is not None:
            constraint_type, indices = violated_index
            if constraint_type == "j":
                i, u, j = indices
                self._add_cut(i, u, j, None)
                print(f"Having {self.nb_lazy} lazy constraints. Added cut for (i={i},u={u},j={j})")
            elif constraint_type == "v":
                i, u, v = indices
                self._add_cut(i, u, None, v)
                print(f"Having {self.nb_lazy} lazy constraints. Added cut for (i={i},u={u},v={v})")
            self.nb_lazy += 1
            return
                        
    def _add_cut(self, i: int, u: int, j: int, v: int):
        """
        Add a single cut to the model.
        
        Args:
            i, u, j, v: Assignment quadruple indices
        """
        y = self.y
        n = self.qap_problem.n
        
        # for i in range(n):
        #     for u in range(n):
        #         for j in range(n):
        #             for v in range(n):
        #                 model.add_constraint(
        #                     model.sum(y[i,u,jj,v] for jj in range(n) if (i,u,jj,v) in y)
        #                     == model.sum(y[i,u,j,vv] for vv in range(n) if (i,u,j,vv) in y),
        #                 )
        
        # Build sparse constraint
        if j is None:
            # sum_{jj} y[i,u,jj,v] - sum_{vv} y[i,u,0,vv] = 0
            indx = [y[i, u, jj, v].index for jj in range(n) if (i, u, jj, v) in y]
            coeffs = [1.0] * len(indx)
            indx += [y[i, u, 0, vv].index for vv in range(n) if (i, u, 0, vv) in y]
            coeffs += [-1.0] * (len(indx) - len(coeffs))
        else:
            # sum_{vv} y[i,u,j,vv] - sum_{jj} y[i,u,jj,0] = 0
            indx = [y[i, u, j, vv].index for vv in range(n) if (i, u, j, vv) in y]
            coeffs = [1.0] * len(indx)
            indx += [y[i, u, 0, vv].index for vv in range(n) if (i, u, 0, vv) in y]
            coeffs += [-1.0] * (len(indx) - len(coeffs))
        
        # Add constraint
        lhs = SparsePair(ind=indx, val=coeffs)
        rhs = float(0)
        
        self.add(lhs, 'E', rhs)


class M4CPLEXSolver:
    """M4 formulation solver using CPLEX."""

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

    @classmethod
    def from_config_file(cls, config_path: str) -> "M4CPLEXSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            M4CPLEXSolver: Initialized solver
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

        model = Model(name="QAP_M4")

        # Create variables
        if self.is_relax:
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
            
        else:
            y = model.binary_var_dict(
                (
                    (i, u, j, v)
                    for i in range(n)
                    for u in range(n)
                    for j in range(n)
                    for v in range(n)
                ),
                name="y",
            )

        x = model.continuous_var_dict(
            ((i,u) for i in range(n) for u in range(n)),
            name="x",
            lb=0,
            ub=1,
        )
        a = model.continuous_var_dict(
            ((u,v) for u in range(n) for v in range(n)),
            name="a",
            lb=0,
            ub=np.inf,
        )
        b = model.continuous_var_dict(
            ((i,u,j) for i in range(n) for u in range(n) for j in range(n)),
            name="b",
            lb=0,
            ub=np.inf,
        )
        c = model.continuous_var_dict(
            ((i,u,v) for i in range(n) for u in range(n) for v in range(n)),
            name="c",
            lb=0,
            ub=np.inf,
        )
        big_M = np.max(distances)* np.max(flows) * n * n # Big-M value
        print(f"Max distance: {np.max(distances)}, Max flow: {np.max(flows)}")
        print(f"Using big-M value: {big_M}")
        # Objective function
        model.minimize(
            model.sum(
                distances[i, j] * flows[u, v] * y[i, u, j, v]
                for i in range(n)
                for j in range(n)
                for u in range(n)
                for v in range(n)
                if (i,u,j,v) in y
            ) + big_M * model.sum(a[u, v] for u in range(n) for v in range(n))
            + big_M * model.sum(b[i, u, j] for i in range(n) for u in range(n) for j in range(n))
            + big_M * model.sum(c[i, u, v] for i in range(n) for u in range(n) for v in range(n))
        )
        
        # model.add_constraint(X == 0) 

        # Linking constraints for y variables
        
        for i in range(n):
            for j in range(n):
                model.add_constraint(
                    model.sum(y[i, u, j, v] for u in range(n) for v in range(n) if (i,u,j,v) in y) <= 1,
                )
        for u in range(n):
            for v in range(n):
                model.add_constraint(
                    model.sum(y[i, u, j, v] for i in range(n) for j in range(n) if (i,u,j,v) in y) == 1,
                )
        for i in range(n):
            for v in range(n):
                model.add_constraint(
                    model.sum(y[i, u, j, v] for u in range(n) for j in range(n) if (i,u,j,v) in y) <= 1,
                )
        for j in range(n):
            for u in range(n):
                model.add_constraint(
                    model.sum(y[i, u, j, v] for i in range(n) for v in range(n) if (i,u,j,v) in y) <= 1,
                )
        for i in range(n):
            for u in range(n):
                for j in range(i+1,n):
                    for v in range(n):
                        if (i,u,j,v) in y and (j,v,i,u) in y:
                            model.add_constraint(
                                y[i, u, j, v] == y[j, v, i, u],
                            )
                j = i
                for v in range(n):
                    if (i,u,j,v) in y and (j,v,i,u) in y:
                        model.add_constraint(
                            y[i, u, j, v] == y[j, v, i, u],
                        )
                            
        for i in range(n):
            for u in range(n):
                for j in range(n):
                    model.add_constraint(
                        model.sum(y[i, u, j, v] for v in range(n) if (i,u,j,v) in y) + b[i, u, j] == x[i, u],
                    )
                for v in range(n):
                    model.add_constraint(
                        model.sum(y[i, u, j, v] for j in range(n) if (i,u,j,v) in y) + c[i, u, v] == x[i, u],
                    )

        V = range(n)
        M = range(n)
        if self.is_relax:
            for i in V:
                for u in M:
                    for v in M:
                        if u != v:
                            model.add_constraint(
                                y[i, u, i, v] == 0, ctname=f"fix_y_{i}_{u}_{i}_{v}"
                            )
            for u in M:
                for i in V:
                    for j in V:
                        if i != j:
                            model.add_constraint(
                                y[i, u, j, u] == 0, ctname=f"fix_y_{i}_{u}_{j}_{u}"
                            )
        # Configure solver parameters
        model.parameters.timelimit = self.time_limit
        model.parameters.threads = self.threads
        model.parameters.preprocessing.symmetry = self.preprocessing_symmetry

        return model, (x,y,a,b,c), big_M, 

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
        model, (x,y,a,b,c), M = self._create_model(problem, fixed_variables)
        model.parameters.lpmethod = 2                 # 2 = Dual Simplex
        # if self.is_relax:
        #     solution = model.solve(log_output=False)
        #     violated = identify_one_violated_constraint(solution, y, problem.n)
            
        #     print(f"Current objective: {solution.objective_value}")
        #     nb_cuts = 0
        #     while violated is not None:
        #         nb_cuts += 1
        #         i,u,j,v = violated
        #         print(f"Having {nb_cuts} cuts, Adding cut for (i={i},u={u},j={j},v={v}) in relaxation")
        #         model.add_constraint(
        #             model.sum(y[i,u,jj,v] for jj in range(problem.n) if (i,u,jj,v) in y)
        #             == model.sum(y[i,u,j,vv] for vv in range(problem.n) if (i,u,j,vv) in y),
        #         )
        #         solution = model.solve(log_output=False)
        #         assert solution is not None, "No solution found in relaxation after adding cuts"
                
        #         print(f"Current objective: {solution.objective_value}")
        #         violated = identify_one_violated_constraint(solution, y, problem.n)
                    
        # else:
        #     cb = model.register_callback(M4LazyCallback)
        #     cb.initialize(y, problem)
        #     # Add warm-start if provided
        #     # docplex's add_mip_start requires a SolveSolution object
        #     if warmstart is not None:
        #         # Create a SolveSolution object for warm-start
        #         ws_solution = model.new_solution()
        #         for (i,u,j,v) in y:
        #             value = warmstart.get((i, u), 0)
        #             value2 = warmstart.get((j, v), 0)
        #             # if value*value2 == 1:
        #             #     print(f"Warmstart y[{i},{u},{j},{v}] = 1")
        #             ws_solution.add_var_value(y[i, u, j, v], value*value2)
        #         # Add MIP start using the solution object
        #         model.add_mip_start(ws_solution)
        #     # Solve
        #     solution = model.solve(log_output=self.log_output)
        solution = model.solve(log_output=self.log_output)
        elapsed_time = time.time() - start_time

        # Extract results
        if solution:
            print(f"Solution found with objective {solution.objective_value} in {elapsed_time:.2f} seconds.")
            # print X value
            print(f"Big-M value used: {M}")
            # Extract assignment from y variables
            assignment = [-1] * problem.n
            is_feasible = True
            for (i, u, j, v), var in y.items():
                val = solution.get_value(var)
                if val is not None and val > 0.5:
                    assignment[i] = u
                    assignment[j] = v
            

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
                objective=solution.objective_value,
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
        "solver": "m4_cplex",
        "formulation": "m4",
        "is_relax": False,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = M4CPLEXSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="m4_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )
