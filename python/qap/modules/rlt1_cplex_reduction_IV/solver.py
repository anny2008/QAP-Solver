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

import cplex
from cplex import SparsePair

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
        self.solver_name = config.get("solver", "rlt1_cplex_reduction_IV")
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
    ) -> Tuple[Any, Dict, Dict]:
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
            
        model = cplex.Cplex()
        model.set_problem_name("QAP_RLT1")
        model.objective.set_sense(model.objective.sense.minimize)
        # Fn0 = [(u,v) for u in M for v in M if flows[u, v] > 0]
        F0 = [(u,v) for u in M for v in M if flows[u, v]+flows[v, u] == 0]
        I0 = [u for u in M if any(flows[u, v] == 0 for v in M if v != u)]
        print(I0)

        # Create variables
        x = {}
        if self.is_relax:
            x_names = [f"x_{i}_{u}" for i in V for u in M]
            model.variables.add(
                names=x_names,
                lb=[0.0] * len(x_names),
                ub=[1.0] * len(x_names),
                types=["C"] * len(x_names),
            )
        else:
            x_names = [f"x_{i}_{u}" for i in V for u in M]
            model.variables.add(
                names=x_names,
                lb=[0.0] * len(x_names),
                ub=[1.0] * len(x_names),
                types=["B"] * len(x_names),
            )
        for i in V:
            for u in M:
                x[i, u] = f"x_{i}_{u}"
            
        all_indices = [(i, u, j, v)
                for i in V
                for u in M
                for j in V
                for v in M
                if u < v and i != j]

        y_var = {}
        y_names = []
        for (i, u, j, v) in all_indices:
            if (u, v) not in F0:
                name = f"y_{i}_{u}_{j}_{v}"
                y_var[i, u, j, v] = name
                y_names.append(name)
        if y_names:
            model.variables.add(
                names=y_names,
                lb=[0.0] * len(y_names),
                ub=[1.0] * len(y_names),
                types=["C"] * len(y_names),
            )

        y = {}
        for (i,u,j,v) in y_var:
            y[i, u, j, v] = y_var[i, u, j, v]
            y[j, v, i, u] = y_var[i, u, j, v]  # Symmetry: y[i,u,j,v] = y[j,v,i,u]
        
        print(f"Number of all indices: {len(all_indices)}")
        print(f"Number of y variables with positive flow: {len(y_var)}")
            
        # Objective function
        obj_by_var = {}
        for (i, u, j, v), y_name in y.items():
            coef = float(distances[i, j] * flows[u, v])
            if coef != 0.0:
                obj_by_var[y_name] = obj_by_var.get(y_name, 0.0) + coef
        if obj_by_var:
            model.objective.set_linear(list(obj_by_var.items()))

        # Assignment constraints
        for i in V:
            model.linear_constraints.add(
                lin_expr=[SparsePair(ind=[x[i, u] for u in M], val=[1.0] * len(M))],
                senses=["E"],
                rhs=[1.0],
                names=[f"assign_fac_{i}"],
            )
        for u in M:
            model.linear_constraints.add(
                lin_expr=[SparsePair(ind=[x[i, u] for i in V], val=[1.0] * len(V))],
                senses=["E"],
                rhs=[1.0],
                names=[f"assign_loc_{u}"],
            )

        # Linking constraints for y variables
        for i in V:
            for u in M:
                if u not in I0:
                    for j in V:
                        # y[i,u,j,*] constraints
                        inds = [y[i, u, j, v] for v in M if (i, u, j, v) in y] + [x[i, u]]
                        vals = [1.0] * (len(inds) - 1) + [-1.0]
                        model.linear_constraints.add(
                            lin_expr=[SparsePair(ind=inds, val=vals)],
                            senses=["E"],
                            rhs=[0.0],
                            names=[f"link1_{i}_{u}_{j}"],
                        )
                # else:
                #     for j in V:
                #         # y[i,u,j,*] constraints
                #         model.add_constraint(
                #             model.sum(y[i, u, j, v] for v in M if (i, u, j, v) in y) <= x[i, u],
                #             ctname=f"link1_{i}_{u}_{j}",
                #         )
                    
                for v in M:
                    # y[i,u,*,v] constraints
                    if (u, v) not in F0:
                        inds = [y[i, u, j, v] for j in V if (i, u, j, v) in y] + [x[i, u]]
                        vals = [1.0] * (len(inds) - 1) + [-1.0]
                        model.linear_constraints.add(
                            lin_expr=[SparsePair(ind=inds, val=vals)],
                            senses=["E"],
                            rhs=[0.0],
                            names=[f"link2_{i}_{u}_{v}"],
                        )
        
        # for i in V:
        #     for u in M:
        #         for j in V:
        #             for v in M:
        #                 if (i,u,j,v) in y and (j,v,i,u) in y:
        #                     # Symmetry constraint: y[i,u,j,v] = y[j,v,i,u]
        #                     model.add_constraint(
        #                         y[i, u, j, v] == y[j, v, i, u],
        #                         ctname=f"symmetry_{i}_{u}_{j}_{v}",
        #                     )

        # Fix variables if provided
        if fixed_variables is not None:
            for i, u in fixed_variables:
                model.linear_constraints.add(
                    lin_expr=[SparsePair(ind=[x[i, u]], val=[1.0])],
                    senses=["E"],
                    rhs=[1.0],
                    names=[f"fix_x_{i}_{u}"],
                )

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
        self.is_relax = True
        # Create model
        model, x, y = self._create_model(problem, fixed_variables)

        # Add warm-start if provided
        # docplex's add_mip_start requires a SolveSolution object
        if warmstart is not None:
            print(f"Adding warm-start with {len(warmstart)} variables")
            start_ind = []
            start_val = []
            for (i, u), value in warmstart.items():
                start_ind.append(x[i, u])
                start_val.append(float(value))
                # Set y variables: y[i,u,j,v] = 1 if both x[i,u]=1 and x[j,v]=1
                for (j, v), value2 in warmstart.items():
                    if (i, u, j, v) in y or (j, v, i, u) in y:
                        start_ind.append(y[i, u, j, v] if (i, u, j, v) in y else y[j, v, i, u])
                        start_val.append(float(value * value2))
            if start_ind:
                model.MIP_starts.add(SparsePair(ind=start_ind, val=start_val), model.MIP_starts.effort_level.auto)
            # turn off heuristic 
            model.parameters.mip.strategy.heuristicfreq.set(-1)

        # if self.lpmethod != "auto":
        #     if self.lpmethod == "barrier":
        #         model.parameters.lpmethod.set(4)  # Barrier method
        #     elif self.lpmethod == "dualsimplex":
        #         model.parameters.lpmethod.set(2)  # Dual simplex
        #     elif self.lpmethod == "primalsimplex":
        #         model.parameters.lpmethod.set(1)  # Primal simplex

        model.parameters.timelimit.set(float(self.time_limit))
        
        if not self.log_output:
            model.set_log_stream(None)
            model.set_warning_stream(None)
            model.set_results_stream(None)
            model.set_error_stream(None)

        model.set_problem_type(model.problem_type.LP)
        model.parameters.lpmethod.set(2)  # Dual simplex
            
        # Solve
        model.solve()
        solution = model.solution
        
        elapsed_time = time.time() - start_time
        # print objective value and status
        model_status = model.solution.get_status()
        print(f"Initial solve status: {model_status} - {model.solution.get_status_string()}")
        print(f"Initial objective value: {model.solution.get_objective_value()}")
        # get all y variable values that are zeros
        y_zeros = [y[i, u, j, v] for (i, u, j, v) in y if solution.get_values(y[i, u, j, v]) < 1e-5 and u < v]
        # print y_zeros 
        # fix them to zero and re-solve sum_ y = 0 constraints
        if y_zeros:
            # print(f"Fixing {len(y_zeros)} y variables to zero and re-solving")
            # model.linear_constraints.add(
            #     lin_expr=[SparsePair(ind=y_zeros, val=[1.0] * len(y_zeros))],
            #     senses=["E"],
            #     rhs=[0.0],
            #     names=["fix_y_zero"],
            # )
            # add all y_zeros to the objective 
            # for i,u,j,v in y:
            #     if u < v:
            #         if solution.get_values(y[i, u, j, v]) == 0.0:
            #             model.objective.set_linear(y[i, u, j, v], 1)
            #         else:
            #             model.objective.set_linear(y[i, u, j, v], 0)
            # change x variables to binary if we were in relaxation
            if self.is_relax:
                for i in range(problem.n):
                    for u in range(problem.m):
                        model.variables.set_types(x[i, u], model.variables.type.binary)
                self.is_relax = False  # Update state to reflect change to binary variables
            
            model.set_problem_type(model.problem_type.MILP)
            model.solve()
            solution = model.solution
            # print status
            model_status = model.solution.get_status()
            print(f"Model status after fixing y variables: {model_status} - {model.solution.get_status_string()}")
            
        
        
        # Extract results
        if solution.is_primal_feasible():
            # Extract assignment from x variables
            num_y = sum(1 for var in y.values() if solution.get_values(var) > 1e-5)
            print(f"Number of positive y variables: {num_y}: {num_y / len(y) * 100:.2f}%")
            assignment = []
            is_feasible = True
            for i in range(problem.n):
                best_u = -1
                best_val = -1
                for u in range(problem.n):
                    val = solution.get_values(x[i, u])
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
            try:
                nb_nodes = model.solution.progress.get_num_nodes_processed()
            except Exception:
                nb_nodes = "unavailable"
            print(f"Number of nodes explored: {nb_nodes}")
            # Create solution object
            if not self.is_relax:
                try:
                    lower_bound = model.solution.MIP.get_best_objective()
                except Exception:
                    lower_bound = model.solution.get_objective_value()
            result = Solution(
                instance=self.instance,  # Will be set by caller
                solver=self.solver_name,
                assignment=assignment,
                objective=model.solution.get_objective_value(),
                lower_bound=lower_bound,
                time=elapsed_time,
            )

            return result
        else:
            # print status message if available
            try:
                status = model.solution.get_status_string()
                print(f"Solver status: {status}")
            except Exception:
                pass
            # No solution found
            try:
                lower_bound = model.solution.MIP.get_best_objective()
            except Exception:
                lower_bound = None
            result = Solution(
                instance=self.instance,
                solver=self.solver_name,
                assignment=None,
                objective=None,
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
            
        # Swap flow and distance for decomposition if specified in config
        # problem.D, problem.F = problem.F, problem.D
        # if warmstart:
        #     warmstart = {(u, i): 1.0 for i, u in warmstart}  # Swap indices for warmstart as well
        
        
        # if np.all(problem.F == problem.F.T) and not np.all(problem.D == problem.D.T):
        #     # convert distance matrix to symmetric by averaging with its transpose
        #     new_D = (problem.D + problem.D.T) / 2
        #     problem.D = new_D
        #     print("Converted distance matrix to symmetric for RLT1 formulation.")
        # elif np.all(problem.D == problem.D.T) and not np.all(problem.F == problem.F.T):
        #     # convert flow matrix to symmetric by averaging with its transpose
        #     new_F = (problem.F + problem.F.T) / 2
        #     problem.F = new_F
        #     print("Converted flow matrix to symmetric for RLT1 formulation.")
        

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
