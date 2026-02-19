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


class SFDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    """
    Lazy constraint callback for SFD formulation.
    
    Implements valid inequality generation for improving the bound.
    """
    
    def __init__(self, env):
        """Initialize callback."""
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_lazy = 0
        self.eps = 1e-6

    def initialize(self, x_vars: Dict, e_vars: Dict, qap_problem: Problem, subgraphs: Dict):
        """
        Initialize callback with model variables and problem data.
        
        Args:
            x_vars: Binary assignment variables x[i,u]
            qap_problem: QAP problem instance
            subgraphs: Subgraph decomposition data
        """
        self.x = x_vars
        self.e = e_vars
        self.qap_problem = qap_problem
        self.subgraphs = subgraphs

    def __call__(self):
        """Called by CPLEX at each lazy constraint check."""
        # Build solution from current node
        sol = self.make_complete_solution()
        # self.add_2_opt_cuts(sol)
        self.add_subgraph_constraints(sol)
    
    def add_subgraph_constraints(self, sol):
        """Add subgraph flow constraints based on current solution."""
        # This method can be implemented to add additional constraints based on the current solution
        subgraphs = self.subgraphs
        V = list(range(self.qap_problem.n))
        M = list(range(self.qap_problem.m))
        one_variables = []
        # Find all variables set to 1.0        assignment = {}
        for i in V:
            for u in M:
                var = self.x[i, u]
                if sol.get_value(var) >= 0.99:  # Handle floating point
                    one_variables.append((i, u))
                    
        found_violated = False
        # Subgraph constraints
        for k, (f_k, G_k, G_n_k) in subgraphs.items():
            for i,u in one_variables:
                for j,v in one_variables:
                    if (u,v) in G_k:
                        # Check if e[i,j,k] >= 1 is violated and add cut if necessary
                        if sol.get_value(self.e[i, j, k]) < 0.99:
                            # Add cut to enforce e[i,j,k] - x[i,u] - x[j,v] >= -1
                            lhs = SparsePair(ind=[self.e[i, j, k].index, self.x[i, u].index, self.x[j, v].index], val=[1.0, -1.0, -1.0])
                            rhs = -1.0
                            self.add(lhs, 'G', rhs)
                    
                            self.nb_lazy += 1
                            # print(f"Added subgraph cut for edge ({i},{j}) in layer {k} with flow {f_k}")
                            found_violated = True
            # if found_violated:
            #     print(f"Added {self.nb_lazy} lazy constraints so far.")
                # break  # Add cuts for one layer at a time to avoid too many cuts in one callback         
                            
        
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():
        #     for i in V:
        #         for j in V:
        #             for u in G_n_k:
        #                 # Check if constraint is violated: e[i,j,k] >= x[i,u] + sum(x[j,v] for v in G_n_k if (u,v) in G_k) - 1
        #                 lhs_value = sol.get_value(self.e[i, j, k])
        #                 rhs_value = sol.get_value(self.x[i, u]) + sum(sol.get_value(self.x[j, v]) for v in G_n_k if (u, v) in G_k) - 1
        #                 if lhs_value < rhs_value - self.eps:
        #                     # Add constraint e[i,j,k] >= x[i,u] + sum(x[j,v] for v in G_n_k if (u,v) in G_k) - 1
        #                     lhs = SparsePair(ind=[self.x[i, u].index] + [self.x[j, v].index for v in G_n_k if (u, v) in G_k],
        #                                     val=[1.0] + [1.0 for v in G_n_k if (u, v) in G_k])
        #                     rhs = 1.0
        #                     self.add(lhs, 'G', rhs)
        #                 # print(f"Added subgraph constraint for i={i}, j={j}, k={k}, u={u}")
                        # return

    def add_2_opt_cuts(self, sol):
        """Add 2-opt cuts based on the current solution."""
        # Find all variables set to 1.0
        count_one = 0
        one_variable_indices = []
        for i, u in self.x:
            var = self.x[i, u]
            if sol.get_value(var) >= 0.99:  # Handle floating point
                count_one += 1
                one_variable_indices.append((i, u))

        # Add cuts for paired assignments
        if count_one > 0:
            M = 1e6
            x = self.x

            for r, (i, u) in enumerate(one_variable_indices):
                for (j, v) in one_variable_indices[r + 1:]:
                    if i == j or u == v:
                        continue

                    # Compute delta values for valid inequality
                    delta_u_ij = np.sum([
                        sol.get_value(x[k, g]) *
                        (self.qap_problem.F[g, u] * (self.qap_problem.D[k, j] - self.qap_problem.D[k, i]) +
                         self.qap_problem.F[u, g] * (self.qap_problem.D[j, k] - self.qap_problem.D[i, k]))
                        for k in range(self.qap_problem.n)
                        for g in range(self.qap_problem.n)
                        if k != i and k != j and g != u and g != v
                    ])

                    delta_v_ji = np.sum([
                        sol.get_value(x[k, g]) *
                        (self.qap_problem.F[g, v] * (self.qap_problem.D[k, i] - self.qap_problem.D[k, j]) +
                         self.qap_problem.F[v, g] * (self.qap_problem.D[i, k] - self.qap_problem.D[j, k]))
                        for k in range(self.qap_problem.n)
                        for g in range(self.qap_problem.n)
                        if k != i and k != j and g != u and g != v
                    ])

                    # Add cut if violated
                    if delta_u_ij + delta_v_ji < -self.eps:
                        self._add_cut(i, u, j, v, delta_u_ij, delta_v_ji, M)
                        self.nb_lazy += 1

    def _add_cut(self, i: int, u: int, j: int, v: int, delta_u_ij: float, delta_v_ji: float, M: float):
        """
        Add a single cut to the model.
        
        Args:
            i, u, j, v: Assignment pair indices
            delta_u_ij: Delta value for first pair
            delta_v_ji: Delta value for second pair
            M: Big-M constant
        """
        x = self.x
        n = self.qap_problem.n
        
        # Build sparse constraint
        indx = []
        coeffs = []
        
        # Add x[k,g] coefficients for delta_u_ij
        for k in range(n):
            for g in range(n):
                if k != i and k != j and g != u and g != v:
                    indx.append(x[k, g].index)
                    coeff = (self.qap_problem.F[g, u] * (self.qap_problem.D[k, j] - self.qap_problem.D[k, i]) +
                             self.qap_problem.F[u, g] * (self.qap_problem.D[j, k] - self.qap_problem.D[i, k]))
                    coeffs.append(float(coeff))
        
        # Add x[k,g] coefficients for delta_v_ji
        for k in range(n):
            for g in range(n):
                if k != i and k != j and g != u and g != v:
                    indx.append(x[k, g].index)
                    coeff = (self.qap_problem.F[g, v] * (self.qap_problem.D[k, i] - self.qap_problem.D[k, j]) +
                             self.qap_problem.F[v, g] * (self.qap_problem.D[i, k] - self.qap_problem.D[j, k]))
                    coeffs.append(float(coeff))
        
        # Add big-M terms for x[i,u] and x[j,v]
        indx.extend([x[i, u].index, x[j, v].index])
        coeffs.extend([-M, -M])
        
        # Add constraint
        lhs = SparsePair(ind=indx, val=coeffs)
        rhs = float(-M * 2)
        
        self.add(lhs, 'G', rhs)
        self.add_local(lhs, 'G', rhs)


class SFDSolver:
    """
    Subgraph Flow Decomposition Solver for QAP.
    
    Decomposes the problem into value layers and solves using CPLEX
    with lazy constraint callbacks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize solver with configuration.

        Args:
            config (dict): Configuration dictionary
                Required keys: solver, formulation
                Optional keys:
                    - decomposition (str): 'value_layer' or 'value_only' (default: 'value_layer')
                    - use_cuts (bool): Use lazy constraint cuts (default: True)
                    - time_limit (float): Time limit in seconds (default: 3600)
                    - threads (int): Number of threads (default: 8)
                    - log_output (bool): Print solver output (default: False)
                    - is_relax (bool): Use relaxed constraints (default: False)
        """
        self.config = config
        self.solver_name = config.get("solver", "sfd")
        self.formulation = config.get("formulation", "sfd")
        self.decomposition = config.get("decomposition", "value_layer")
        self.use_cuts = config.get("use_cuts", True)
        self.time_limit = config.get("time_limit", 3600)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.is_relax = config.get("is_relax", False)

    def _create_model(
        self,
        problem: Problem,
        subgraphs: Dict,
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
        
        # # Variables
        x = model.binary_var_matrix(n, m, name="x")
        # e = model.continuous_var_dict(
        #     ((i, j, k) for i in V for j in V for k in subgraphs),
        #     name="e",
        #     lb=0,
        #     ub=1
        # )
        # Variables
        # x = model.continuous_var_matrix(n, m, name="x", ub=1, lb=0)
        e = model.binary_var_dict(
            ((i, j, k) for i in V for j in V for k in subgraphs),
            name="e",
        )
        # if m < n:
        #     # add dummy locations if m < n
        #     x_dummy = model.binary_var_matrix(n, 1, name="x_dummy")
        
        # check if problem has fixed assignments and add constraints
        if problem.fixed_assignments:
            print(f"Adding {problem.fixed_assignments} fixed assignment constraints.")
            for i, u in problem.fixed_assignments.items():
                model.add_constraint(x[i, u] == 1, ctname=f"fixed_{i}_{u}")

        # Objective function
        model.minimize(model.sum(
            distances[i, j] * subgraphs[k][0] * e[i, j, k]
            for i in V for j in V for k in subgraphs
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

        # # Assignment constraints
        # if m < n:
        #     # Add constraints to ensure dummy locations are only assigned to one facility
        #     for i in V:
        #         model.add_constraint(
        #             model.sum(x[i, u] for u in M) + x_dummy[i, 0] == 1,
        #             ctname=f"assign_facility_{i}"
        #         )
        #     model.add_constraint(
        #         model.sum(x_dummy[i, 0] for i in V) == n - m,
        #         ctname=f"assign_location_dummy"
        #     )


        # else:
        #     for i in V:
        #         model.add_constraint(
        #             model.sum(x[i, u] for u in M) == 1,
        #             ctname=f"assign_facility_{i}"
        #         )
        # for u in M:
        #     model.add_constraint(
        #         model.sum(x[i, u] for i in V) == 1,
        #         ctname=f"assign_location_{u}"
        #     )

        if not self.use_cuts:
            # Add subgraph constraints directly if not using lazy constraints
            # Subgraph constraints
            for k, (f_k, G_k, G_n_k) in subgraphs.items():
                if not self.is_relax:
                    for i in V:
                        for j in V:
                            for u in G_n_k:
                                model.add_constraint(
                                    e[i, j, k] >= x[i, u] + model.sum(
                                        x[j, v] for v in G_n_k if (u, v) in G_k
                                    ) - 1,
                                    ctname=f"edge_link_{i}_{j}_{u}_{k}"
                                )

            # Compute degree sequences
            degree_out = {}
            for u, v in G_k:
                degree_out[u] = degree_out.get(u, 0) + 1
            
            degree_in = {}
            for u, v in G_k:
                degree_in[v] = degree_in.get(v, 0) + 1

            # Flow conservation constraints
            for i in V:
                model.add_constraint(
                    model.sum(e[i, j, k] for j in V) ==
                    model.sum(x[i, u] * degree_out.get(u, 0) for u in set(u for u, v in G_k)),
                    ctname=f"flow_out_{i}_{k}"
                )
                model.add_constraint(
                    model.sum(e[j, i, k] for j in V) ==
                    model.sum(x[i, v] * degree_in.get(v, 0) for v in set(v for u, v in G_k)),
                    ctname=f"flow_in_{i}_{k}"
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
                for (j, v), val in warmstart.items():
                    for k in subgraphs:
                        if (u, v) in subgraphs[k][1]:
                            ws.add_var_value(e[j, i, k], min(val, warmstart.get((j, v), 0.0)))
            model.add_mip_start(ws)

        # Configure CPLEX parameters
        model.parameters.timelimit = self.time_limit
        model.parameters.threads = self.threads
        model.parameters.mip.tolerances.mipgap = 0.0
        
        return model, x, e

    def solve(
        self,
        problem: Problem,
        subgraphs: Dict,
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
        model, x, e = self._create_model(problem, subgraphs, fixed_variables, warmstart)

        # Register lazy callback if using cuts
        if self.use_cuts:
            cb = model.register_callback(SFDLazyCallback)
            cb.initialize(x, e, problem, subgraphs)

        # Solve
        solution = model.solve(log_output=self.log_output)

        elapsed_time = time.time() - start_time

        if solution:
            # Extract assignment
            assignment = [None] * problem.n
            for i in range(problem.n):
                for u in range(problem.n):
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
        
        # Decompose problem into subgraphs
        if self.decomposition == "value_layer":
            subgraphs = decompose_value_layer(problem)
        elif self.decomposition == "value_only":
            subgraphs = decompose_value_only(problem)
        print(f"Decomposed into {len(subgraphs)} subgraphs using {self.decomposition} strategy.")
        # Solve
        solution = self.solve(
            problem,
            fixed_variables=fixed_variables,
            warmstart=warmstart,
            subgraphs=subgraphs
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
