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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from qap.core import Problem, Solution, write_result
from qap.core.solution_io import read_warmstart
from qap.decomposition import decompose_value_layer, decompose_value_only, decompose_value_only_no_cycle3


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
        self.n = qap_problem.n
        self.m = qap_problem.m

        # Cache x variable indices row-wise to query values in bulk.
        self.x_idx_by_i = [
            [x_vars[i, u].index for u in range(self.m)]
            for i in range(self.n)
        ]

    def __call__(self):
        """Called by CPLEX at each lazy constraint check."""
        # Avoid building a full docplex solution object in callback hot path.
        self.add_subgraph_constraints()
    
    def add_subgraph_constraints(self):
        """Add subgraph flow constraints based on current solution."""
        subgraphs = self.subgraphs
        x_idx_by_i = self.x_idx_by_i
        get_values = self.get_values
        tol = 0.99

        # Extract integral assignments using one bulk query per facility row.
        loc_to_fac = {}
        for i in range(self.n):
            row_vals = get_values(x_idx_by_i[i])
            for u, val in enumerate(row_vals):
                if val >= tol:
                    loc_to_fac[u] = i
                    break

        if not loc_to_fac:
            return

        # Build all candidate (k,i,j,u,v,e_var) first, then query e in one bulk call.
        candidates = []
        e_indices = []
        x = self.x
        found_violated = False
        for k, (f_k, G_k, G_n_k) in subgraphs.items():
            for u, v in G_k:
                i = loc_to_fac.get(u)
                j = loc_to_fac.get(v)
                if i is None or j is None:
                    continue

                e_var = self.e.get((k, i, j))
                if e_var is None:
                    continue

                candidates.append((e_var, i, u, j, v))
                e_indices.append(e_var.index)

        if not candidates:
            return

        e_vals = get_values(e_indices)

        for (e_var, i, u, j, v), e_val in zip(candidates, e_vals):
            # If x[i,u]=1 and x[j,v]=1, enforce e[k,i,j] >= 1
            if e_val < tol:
                lhs = SparsePair(
                    ind=[e_var.index, x[i, u].index, x[j, v].index],
                    val=[1.0, -1.0, -1.0]
                )
                self.add(lhs, 'G', -1.0)
                self.nb_lazy += 1
                found_violated = True

        if found_violated:
            print(f"Added {self.nb_lazy} lazy constraints so far.")
            # break  # Add cuts for one layer at a time to avoid too many cuts in one callback
        
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():
        #     for i in V:
        #         for j in V:
        #             for u in G_n_k:
        #                 # Check if constraint is violated: e[k, i, j] >= x[i,u] + sum(x[j,v] for v in G_n_k if (u,v) in G_k) - 1
        #                 lhs_value = sol.get_value(self.e[k, i, j])
        #                 rhs_value = sol.get_value(self.x[i, u]) + sum(sol.get_value(self.x[j, v]) for v in G_n_k if (u, v) in G_k) - 1
        #                 if lhs_value < rhs_value - self.eps:
        #                     # Add constraint e[k, i, j] >= x[i,u] + sum(x[j,v] for v in G_n_k if (u,v) in G_k) - 1
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
        self.matrix_for_decomposition = config.get("matrix_for_decomposition", "flow")  # 'flow' or 'distance'
        self.binary_variables = config.get("binary_variables", "x")  

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
            ub_warmstart: Upper bound warm-start value
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
        # check if the diagonal of flow matrix is zero, if not print warning
        if np.any(np.diag(flows) != 0):
            print("Warning: Flow matrix has nonzero diagonal entries, which may affect the validity of the SFD formulation.")
        is_symmetric = np.all(flows == flows.T) and np.all(distances == distances.T)
        if is_symmetric:
            e_set = [(k, i, j) for i in V for j in V for k in subgraphs if i <= j]
        else:
            e_set = [(k, i, j) for i in V for j in V for k in subgraphs]
        model = Model(name="QAP_SFD")
        
        if self.is_relax:
            x = model.continuous_var_matrix(n, m, name="x", lb=0, ub=1)
            e = model.continuous_var_dict(
                ((k,i,j) for (k,i,j) in e_set),
                name="e",
                lb=0,
                ub=1
            )
            
        else:
        # # Variables
            if self.binary_variables == "x":
                print("Using binary variables for x and continuous variables for e.")
                x = model.binary_var_matrix(n, m, name="x")
                e = model.continuous_var_dict(
                    e_set,
                    name="e",
                    lb=0,
                    ub=1
                )
            elif self.binary_variables == "e":
                print("Using continuous variables for x and binary variables for e.")
                x = model.continuous_var_matrix(n, m, name="x", lb=0, ub=1)
                e = model.binary_var_dict(
                    e_set,
                    name="e"
                )
        e_vars = {}
        if is_symmetric:
            for (k, i, j) in e_set:
                e_vars[k, i, j] = e[k, i, j]
                e_vars[k, j, i] = e[k, i, j]  # Symmetry: e[k,j,i] = e[k,i,j]
        else:
            e_vars = { (k, i, j): e[k, i, j] for (k, i, j) in e_set }
        e = e_vars
        
        h = model.continuous_var_dict(
            ((k, i, j) for (k, i, j) in e),
            name="h",
            lb=0,
            ub=2
        )
                
        print(len(e), "e variables created out of", n*n*len(subgraphs), "possible")
        # check if problem has fixed assignments and add constraints
        if hasattr(problem, "fixed_assignments") and problem.fixed_assignments:
            print(f"Adding {problem.fixed_assignments} fixed assignment constraints.")
            for i, u in problem.fixed_assignments.items():
                model.add_constraint(x[i, u] == 1, ctname=f"fixed_{i}_{u}")

        # Objective function
        model.minimize(model.sum(
            distances[i, j] * subgraphs[k][0] * e[k, i, j]
            for i in V for j in V for k in subgraphs if (k, i, j) in e
        ))
        
        # Assignment constraints
        for i in V:
            model.add_constraint(
                model.sum(x[i, u] for u in M) == 1,
                ctname=f"assign_facility_{i}"
            )
        for u in M:
            model.add_constraint(
                model.sum(x[i, u] for i in V) == 1,
                ctname=f"assign_location_{u}"
            )

        if not self.use_cuts and not self.is_relax:
            # Add subgraph constraints directly if not using lazy constraints
            # Subgraph constraints
            for k, (f_k, G_k, G_n_k) in subgraphs.items():
                if not self.is_relax:
                    for i in V:
                        for j in V:
                            if (k, i, j) in e:
                                for u in G_n_k:
                                    model.add_constraint(
                                        e[k, i, j] >= -1 + x[i, u] + model.sum(
                                            x[j, v] for v in G_n_k if (u, v) in G_k
                                        ),
                                        ctname=f"edge_link_{i}_{j}_{u}_{k}"
                                    )
                            # else:
                            #     # e_ij_k = 0
                            #     for u in G_n_k:
                            #         model.add_constraint(
                            #             x[i, u] + model.sum(
                            #                 x[j, v] for v in G_n_k if (u, v) in G_k
                            #             ) - 1 <= 0,
                            #             ctname=f"no_edge_link_{i}_{j}_{u}_{k}"
                            #         )

        for k, (f_k, G_k, G_n_k) in subgraphs.items():
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
                    model.sum(e[k, i, j] for j in V if (k, i, j) in e) ==
                    model.sum(x[i, u] * degree_out.get(u, 0) for u in set(u for u, v in G_k)),
                    ctname=f"flow_out_{i}_{k}"
                )
                if not is_symmetric:
                    model.add_constraint(
                        model.sum(e[k, j, i] for j in V if (k, j, i) in e) ==
                        model.sum(x[i, u] * degree_in.get(u, 0) for u in set(v for u, v in G_k)),
                        ctname=f"flow_in_{i}_{k}"
                    )
        # # sum_i,j e^k_ij = number of edge of G_k
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():
        #     model.add_constraint(
        #         model.sum(e[k, i, j] for i in V for j in V if (k, i, j) in e) == len(G_k),
        #         ctname=f"flow_conservation_{k}"
        #     )
        
        # e_ij^k + sum_{u \notin G_k} x[i,u] <= 1 for all (i,j) and k
        # e_ij^k + sum_{v \notin G_k} x[j,v] <= 1 for all (i,j) and k
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():
            # let set_u_not_in_G_k be the set of u in M that there is no arc u->v in G_k for any v,
            # and set_v_not_in_G_k be the set of v in M that there is no arc u->v in G_k for any u
            # set_u_not_in_G_k = set(u for u in M if all((u, v) not in G_k for v in M))
            # set_v_not_in_G_k = set(v for v in M if all((u, v) not in G_k for u in M))
            # print(k, set_u_not_in_G_k, set_v_not_in_G_k)
            # # for i in V:
                
            # #     # model.add_constraint(
            # #     #     model.sum(e[k, i, j] for j in V if (k, i, j) in e)
            # #     #     + model.sum(x[i, u] for u in set_u_not_in_G_k) 
            # #     #      >= 1,
            # #     #     ctname=f"edge_assignment_both_{i}_{k}"
            # #     # )
            # #     for j in V:
            # #         if i != j and (k, i, j) in e:
            # #             model.add_constraint(
            # #                 e[k, i, j] + model.sum(x[i, u] for u in M if u not in G_n_k) <= 1,
            # #                 ctname=f"edge_assignment_i_{i}_{j}_{k}"
            # #             )
            # #             model.add_constraint(
            # #                 e[k, i, j] + model.sum(x[j, v] for v in M if v not in G_n_k) <= 1,
            # #                 ctname=f"edge_assignment_j_{i}_{j}_{k}"
            # #             )
            
        # 2e_ij^k + h_ij^k = sum_{u \in G_n_k first} x_iu + sum_{v \in G_n_k second} x_jv for all (i,j) and k
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():     
        #     u_set = set([u for (u,v) in G_k])
        #     v_set = set([v for (u,v) in G_k])
        #     print(f"Subgraph {k}: u_set={u_set}, v_set={v_set}, G_k={G_k}")
        #     for i in V:
        #         for j in V:
        #             if i != j and (k, i, j) in e:
        #                 model.add_constraint(
        #                     e[k, i, j] + h[k, i, j] == model.sum(x[i, u] for u in u_set) + model.sum(x[j, v] for v in v_set),
        #                     ctname=f"edge_assignment_both_{i}_{j}_{k}"
        #                 )
        #                 h[k, i, j] + e[k, i, j] <= 2
        #                 model.add_constraint(
        #                     h[k, i, j] + e[k, i, j] <= 2,
        #                     ctname=f"h_edge_link_{i}_{j}_{k}"
        #                 )
        #                 model.add_constraint(
        #                     e[k, i, j] <= h[k, i, j],
        #                     ctname=f"e_h_link_{i}_{j}_{k}"
        #                 )
        # # sum_k h_ij^k + sum_k e_ij^k = 1 for all (i,j)
        # for i in V:
        #     for j in V:
        #         if i != j:
        #             model.add_constraint(
        #                 model.sum(h[k, i, j] + e[k, i, j] for k in subgraphs if (k, i, j) in e) == 1,
        #                 ctname=f"flow_conservation_{i}_{j}"
        #             )

        # sum_k e^k_ij <= 1 if using value_only
        if self.decomposition == "value_only":
            for i in V:
                for j in V:
                    if i != j:
                        model.add_constraint(
                            model.sum(e[k, i, j] for k in subgraphs if (k, i, j) in e) == 1,
                            ctname=f"flow_value_{i}_{j}"
                        )
                        

        # Fix variables if provided
        if fixed_variables is not None:
            print(f"Adding {len(fixed_variables)} fixed variable constraints.")
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
                            if (k, i, j) in e:
                                ws.add_var_value(e[k, i, j], min(val, warmstart.get((j, v), 0.0)))
            model.add_mip_start(ws)
            # disable heuristic improvement in CPLEX to rely more on the warm-start solution
            model.parameters.mip.strategy.heuristiceffort = 0

        # Configure CPLEX parameters
        model.parameters.timelimit = self.time_limit
        
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
                    if (i, u) in x:
                        if solution.get_value(x[i, u]) > 0.5:
                            assignment[i] = u
                            break
            e_vals = {}
            for k in subgraphs:
                e_vals[k] = {(i, j): solution.get_value(e[k, i, j]) for i in range(problem.n) for j in range(problem.n) if (k, i, j) in e and  solution.get_value(e[k, i, j]) > 0.0}
            print(f"Nonzero e values: {sum(len(e_vals[k]) for k in e_vals)}")
            # print the number of nodes visited in the branch and bound tree
            nb_nodes = getattr(model.solve_details, "nb_nodes_processed", None)
            if nb_nodes is None:
                try:
                    nb_nodes = model.cplex.solution.progress.get_num_nodes_processed()
                except Exception:
                    nb_nodes = "unavailable"
            print(f"Number of nodes explored: {nb_nodes}")
            # check if there is any 3 cycle uv vw wu in any subgraph k with e[k,i,j] > 0 for i assigned to u and j assigned to v
            # for k, (f_k, G_k, G_n_k) in subgraphs.items():
            #     for (i,j) in e_vals[k]:
            #         for l in range(problem.n):
            #             if (j,l) in e_vals[k] and (l,i) in e_vals[k]:
            #                 if e_vals[k][i,j]+ e_vals[k][j,l] + e_vals[k][l,i] > 2:
            #                     print(f"Found 3-cycle in subgraph {k} with edges ({i},{j}), ({j},{l}), ({l},{i}) with e values {e_vals[k][i,j]}, {e_vals[k][j,l]}, {e_vals[k][l,i]}")
                            
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=assignment,
                objective=solution.objective_value,
                lower_bound=model.solve_details.best_bound,
                time=elapsed_time,
            )
        else:
            solution = model.cplex.solution
            if solution:
                return Solution(
                    instance="unknown",
                    solver=self.solver_name,
                    assignment=None,
                    objective=solution.get_objective_value(),
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
        print(f"Solving instance {instance_path} with SFD solver...")
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
            # use LocalSearchSolver to find a warm-start solution
            from qap.modules.local_search import LocalSearchSolver
            
            print(f"Running local search to find initial solution...")
            # local_solver = LocalSearchSolver({})
            # local_solution = local_solver.solve(problem, fixed_variables=None)
            # print(f"Local search initial solution: obj={local_solution.objective:.6f}")
            # if hasattr(problem, "fixed_assignments"):
            #     for i, u in problem.fixed_assignments.items():
            #         if local_solution.assignment[i] != u:
            #             print(f"Warning: Local search solution violates fixed assignment at location {i}: assigned {local_solution.assignment[i]} vs fixed {u}")
                    
            # warmstart = {(i, u): 1.0 for i, u in enumerate(local_solution.assignment)}
            warmstart = None  # Do not use warmstart if local search is not used
            
        if self.matrix_for_decomposition == "distance":
            # Swap flow and distance for decomposition if specified in config
            problem.D, problem.F = problem.F, problem.D
            if warmstart_path is not None:
                warmstart = {(u, i): 1.0 for i, u in warmstart}  # Swap indices for warmstart as well
        
        # print(problem.F)
        # if problem is in QAPLIB and is symmetric convert the flow matrix to assymmetric
        # if "QAPLIB" in instance_path and np.all(problem.F == problem.F.T) and np.all(problem.D == problem.D.T):
            # print("Converting symmetric flow matrix to asymmetric for SFD formulation.")
            # new_F = np.zeros_like(problem.F)
            # for u in range(problem.n):
            #     for v in range(u + 1, problem.n):
            #             new_F[u, v] = 2*problem.F[u, v]
            #             new_F[v, u] = 0
            # problem.F = new_F
            # new_D = np.zeros_like(problem.D)
            # for i in range(problem.n):
            #     for j in range(i + 1, problem.n):
            #             new_D[i, j] = 2*problem.D[i, j]
            #             new_D[j, i] = 0
            # problem.D = new_D
        
        # print(problem.F)
        if np.all(problem.F == problem.F.T) and not np.all(problem.D == problem.D.T):
            # convert distance matrix to symmetric by averaging with its transpose
            new_D = (problem.D + problem.D.T) / 2
            problem.D = new_D
            print("Converted distance matrix to symmetric for SFD formulation.")
        elif np.all(problem.D == problem.D.T) and not np.all(problem.F == problem.F.T):
            # convert flow matrix to symmetric by averaging with its transpose
            new_F = (problem.F + problem.F.T) / 2
            problem.F = new_F
            print("Converted flow matrix to symmetric for SFD formulation.")
        
        # Decompose problem into subgraphs
        if self.decomposition == "value_layer":
            subgraphs = decompose_value_layer(problem)
        elif self.decomposition == "value_only":
            subgraphs = decompose_value_only(problem)
        elif self.decomposition == "value_only_no_cycle3":
            subgraphs = decompose_value_only_no_cycle3(problem)
        print(f"Decomposed into {len(subgraphs)} subgraphs using {self.decomposition} strategy.")
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():
        #     print(f"Subgraph {k}: flow={f_k}, |G_k|={len(G_k)}, |G_n_k|={len(G_n_k)}")
        # check if any subgraph with 3 edge cycles uv vw wu in G_k
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():
        #     cycle = set()
        #     for u, v in G_k:
        #         for w in G_n_k:
        #             if w != u and w != v:
        #                 if (v, w) in G_k and (w, u) in G_k:
        #                     cycle.add(tuple(sorted((u, v, w))))
        #     if cycle:
        #         print(f"Found {len(cycle)} 3-edge cycles in subgraph {k}")
        # # check if there any subgraph with 4 edge cycles uv vw wo ou in G_k
        # for k, (f_k, G_k, G_n_k) in subgraphs.items():
        #     cycle = set()
        #     for u, v in G_k:
        #         for w in G_n_k:
        #             if w != u and w != v:
        #                 if (v, w) in G_k:
        #                     for o in G_n_k:
        #                         if o != u and o != v and o != w:
        #                             if (w, o) in G_k and (o, u) in G_k:
        #                                 cycle.add(tuple(sorted((u, v, w, o))))
        #     if cycle:
        #         print(f"Found {len(cycle)} 4-edge cycles in subgraph {k}")
                
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
