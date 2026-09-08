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

import cplex.callbacks as cpx_cb
import cplex._internal._constants as cpxcst
import cplex
from cplex.callbacks import Context
from cplex.callbacks import UserCutCallback


import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from qap.core import Problem, Solution, write_result
from qap.core.solution_io import read_warmstart
from qap.decomposition import decompose_value_layer, decompose_value_only, decompose_value_only_no_cycle3


def solve_fixed_e_and_get_cuts(qap_problem, V, M, subgraphs, x_e_sol, e_sol):
    cpx_fix_e = cplex.Cplex()
    cpx_fix_e.set_problem_name(f"SFD_Cut_Generation")
    key_iu = [(i, u) for i in V for u in M]

    y_keys = [(i, u, j, v) for (i, u) in key_iu for (j,v) in key_iu if i < j and u != v]
    y_index = {key: idx for idx, key in enumerate(y_keys)}
    y_names = [f"y_{i}_{u}_{j}_{v}" for (i, u, j, v) in y_keys]
    # print(len(y_names), "y variables created out of", len(V)*len(M)*len(V)*len(M), "possible")
    # obj = [qap_problem.D[i][j] * qap_problem.F[u][v] + qap_problem.D[j][i] * qap_problem.F[v][u] for (i, u, j, v) in y_keys]
    
    cpx_fix_e.variables.add(
        obj=[0.0] * len(y_keys),
        lb=[0.0] * len(y_keys),
        names=y_names,
    )
    
    y = {(i, u, j, v): y_index[(i, u, j, v)] for (i, u, j, v) in y_keys}
    y.update({(j, v, i, u): y_index[(i, u, j, v)] for (i, u, j, v) in y_keys})
    cpx_fix_e.objective.set_sense(cpx_fix_e.objective.sense.minimize)

    lin_expr = []
    senses = []
    rhs = []
    names = []
    
    for (i,u) in key_iu:
        for j in V:
            if i != j:
                # sum_v y_iujv >= x_iu for all i,u,j
                ind = [y[(i, u, j, v)] for v in M if (i, u, j, v) in y]
                lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * len(ind)))
                senses.append("E")
                rhs.append(x_e_sol[i,u])
                names.append(f"y_row_{i}_{u}_{j}")
                
                
        for v in M:
            if v != u:
                ind = [y[(i, u, j, v)] for j in V if (i, u, j, v) in y]
                lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * len(ind)))
                senses.append("E")
                rhs.append(x_e_sol[i,u])
                names.append(f"y_col_{i}_{u}_{v}")
                    
                    
    for k, (flow_value, arcs, nodes) in subgraphs.items():
        for i in V:
            for j in V:
                if i != j:
                    ind = [y[(i, u, j, v)] for u in M for v in M if (u, v) in arcs and (i, u, j, v) in y]
                    lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * len(ind)))
                    senses.append("E")
                    rhs.append(float(e_sol[k, i, j]))
                    names.append(f"y_e_def_{k}_{i}_{j}")
    
    
    cpx_fix_e.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=names)
    cpx_fix_e.parameters.preprocessing.presolve.set(0)
    # disable CPLEX output
    cpx_fix_e.set_log_stream(None)
    cpx_fix_e.set_error_stream(None)
    cpx_fix_e.set_warning_stream(None)
    cpx_fix_e.set_results_stream(None)

    cpx_fix_e.solve()
    status_string = cpx_fix_e.solution.get_status_string()
    # print(f"Status: {status_string}")
    if "optimal" in status_string.lower():
        # return the dual optimal solution
        # print("Optimal, returning dual optimal solution...")
        duals = cpx_fix_e.solution.get_dual_values()
        row_names = cpx_fix_e.linear_constraints.get_names()
        non_zero_duals = [(row_names[idx], val) for idx, val in enumerate(duals) if abs(val) > 1e-6]
        # print(len(non_zero_duals), "non-zero dual values")
        z_value = cpx_fix_e.solution.get_objective_value()
        # print(f"Objective value: {z_value}")
        return True,z_value,non_zero_duals
    elif "infeasible" in status_string.lower():
        # print("Infeasible, getting dual Farkas ray...")
        ray, scale = cpx_fix_e.solution.advanced.dual_farkas()
        # print(b)
        row_names = cpx_fix_e.linear_constraints.get_names()
        non_zero_farkasray = [(row_names[idx], val) for idx, val in enumerate(ray) if abs(val) > 1e-6]
        # print(len(non_zero_farkasray), "non-zero entries in farkas ray")
        return False,scale,non_zero_farkasray
    else:
        # print("Status is neither optimal nor infeasible, something went wrong.")
        pass
    return None,None,None

class SFDLazyCallback(cpx_cb.LazyConstraintCallback):
    def __call__(self):
        if not hasattr(self, "cut_generation_count"):
            self.cut_generation_count = 0
        if not hasattr(self, "cut_bender_count"):
            self.cut_bender_count = 0
        if not hasattr(self, "cut_x_e_count"):
            self.cut_x_e_count = 0
        # get current relaxation solution values
        x_keys = self.x_keys
        e_keys = self.e_keys
        e_pos = self.e_pos
        x_vals = self.get_values(list(range(0, len(x_keys))))
        e_vals = self.get_values(list(range(len(x_keys), len(x_keys) + len(e_keys))))
        x_e_sol = {x_keys[idx]: float(x_vals[idx]) for idx in range(len(x_keys))}
        e_base_sol = {e_keys[idx]: float(e_vals[idx]) for idx in range(len(e_keys))}
        e_sol = {(k, i, j): e_base_sol[(k, i, j) if i < j else (k, j, i)] for k in self.subgraphs for i in self.V for j in self.V if i != j}
        self.add_Bender_cut(x_e_sol, e_sol)
        # self.add_x_e_cut(x_e_sol, e_sol)
        
    def add_x_e_cut(self, x_e_sol, e_sol):
        # get current relaxation solution values
        x_keys = self.x_keys
        e_keys = self.e_keys
        e_pos = self.e_pos
        # e_kij >= x_iu + sum_{v in G_n_k: u,v in A_k} x_jv - 1 for all i,j,u,k
        # This constraint active only with integral solutions, so we can add it as a lazy constraint
        x_one_sol = {(i, u): x_e_sol[(i, u)] for (i, u) in x_keys if abs(x_e_sol[(i, u)] - 1.0) < 1e-6}
        for i in self.V:
            for j in self.V:
                if i != j:
                    for u in self.M:
                        if (i, u) in x_one_sol:
                            # add cut if violated by current solution
                            # check if e_sol[k, i, j] < x_e_sol[i, u] + sum(x_e_sol[j, v] for v in self.M if (u, v) in self.subgraphs[k][1]) - 1 - 1e-6 for any k
                            
                            for k, (flow_value, arcs, nodes) in self.subgraphs.items():
                                e_sol_kij = e_sol.get((k, i, j), 0.0)
                                if e_sol_kij < 0.1:
                                    sum_x_jv = sum(x_e_sol.get((j, v), 0.0) for v in self.M if (u, v) in arcs)
                                    if sum_x_jv > 0.99:
                                        ind = [e_pos(k, i, j)] + [self.x_index[(i, u)]] + [self.x_index[(j, v)] for v in nodes if (u, v) in arcs]
                                        val = [1.0] + [-1.0] + [-1.0] * sum(1 for v in nodes if (u, v) in arcs)
                                        self.add(cplex.SparsePair(ind=ind, val=val), sense="G", rhs=-1.0)
                                        self.cut_x_e_count += 1
                                        if self.cut_x_e_count % 100 == 0:
                                            print(f"Added x-e cut #{self.cut_x_e_count} total cuts: {self.cut_bender_count + self.cut_x_e_count}")
    def add_Bender_cut(self, x_e_sol, e_sol):
        # get current relaxation solution values
        e_pos = self.e_pos
        # print("SFD Callback: Solving fixed-e subproblem to check for violated cuts...")
        optimal, z_value, non_zero_pi = solve_fixed_e_and_get_cuts(self.qap_problem, self.V, self.M, self.subgraphs, x_e_sol, e_sol)
        if optimal is None:
            return

        if not optimal:
            vars_to_add_G = {}
            verify_cut = 0.0
            for constraint_name, val in non_zero_pi:
                if constraint_name.startswith("y_e_def_"):
                    _, _, _, k, i, j = constraint_name.split("_")
                    k = int(k); i = int(i); j = int(j)
                    verify_cut += val * e_sol.get((k, i, j), 0.0)
                    vars_to_add_G[(k, i, j)] = (e_pos(k, i, j), vars_to_add_G.get((k, i, j), (None, 0.0))[1] + val)
                elif constraint_name.startswith("y_row_") or constraint_name.startswith("y_col_"):
                    _, _, i, u, j = constraint_name.split("_")
                    i = int(i); u = int(u); j = int(j)
                    verify_cut += val * x_e_sol.get((i, u), 0.0)
                    vars_to_add_G[(i, u)] = (self.x_index[(i, u)], vars_to_add_G.get((i, u), (None, 0.0))[1] + val)
            if len(vars_to_add_G) > 0:
                ind = [idx for _, (idx, v) in vars_to_add_G.items()]
                val = [v * -1.0 for _, (idx, v) in vars_to_add_G.items()]
                # verify that the cut is violated by current solution
                if verify_cut >= -1e-6:
                    print(f"Generated Bender cut is not violated by current solution (lhs={verify_cut}), skipping.")
                    return
                self.add(cplex.SparsePair(ind=ind, val=val), sense="G", rhs=0.0)
                self.cut_bender_count += 1
                if self.cut_bender_count % 100 == 0:
                    print(f"Added Bender cut #{self.cut_bender_count} total cuts: {self.cut_bender_count + self.cut_x_e_count}")
                # print(f"Added cut #{self.cut_bender_count} with {len(ind)} variables, scale={z_value}, non_zero_pi={len(non_zero_pi)}")

class SFDUserCutCallback(cpx_cb.UserCutCallback):
    def __call__(self):
        if not hasattr(self, "cut_generation_count"):
            self.cut_generation_count = 0
        if not hasattr(self, "cut_bender_count"):
            self.cut_bender_count = 0
        if not hasattr(self, "cut_x_e_count"):
            self.cut_x_e_count = 0
        depth = self.get_current_node_depth()
        # Limit cut generation to avoid too much overhead
        if self.cut_generation_count < self.cut_frequency:
            self.cut_generation_count += 1
            return 
        else:
            self.cut_generation_count = 0
        # get current relaxation solution values
        x_keys = self.x_keys
        e_keys = self.e_keys
        e_pos = self.e_pos
        x_vals = self.get_values(list(range(0, len(x_keys))))
        e_vals = self.get_values(list(range(len(x_keys), len(x_keys) + len(e_keys))))
        x_e_sol = {x_keys[idx]: float(x_vals[idx]) for idx in range(len(x_keys))}
        e_base_sol = {e_keys[idx]: float(e_vals[idx]) for idx in range(len(e_keys))}
        e_sol = {(k, i, j): e_base_sol[(k, i, j) if i < j else (k, j, i)] for k in self.subgraphs for i in self.V for j in self.V if i != j}
        self.add_Bender_cut(x_e_sol, e_sol)
        
    def add_Bender_cut(self, x_e_sol, e_sol):
        # get current relaxation solution values
        e_pos = self.e_pos
        # print("SFD Callback: Solving fixed-e subproblem to check for violated cuts...")
        start_time = time.time()
        optimal, z_value, non_zero_pi = solve_fixed_e_and_get_cuts(self.qap_problem, self.V, self.M, self.subgraphs, x_e_sol, e_sol)
        
        end_time = time.time()
        print(f"Time taken to solve fixed-e subproblem: {end_time - start_time} seconds")

        if optimal is None:
            return

        if not optimal:
            verify_cut = 0.0
            vars_to_add_G = {}
            for constraint_name, val in non_zero_pi:
                if constraint_name.startswith("y_e_def_"):
                    _, _, _, k, i, j = constraint_name.split("_")
                    k = int(k); i = int(i); j = int(j)
                    verify_cut += val * e_sol.get((k, i, j), 0.0)
                    vars_to_add_G[(k, i, j)] = (e_pos(k, i, j), vars_to_add_G.get((k, i, j), (None, 0.0))[1] + val)
                elif constraint_name.startswith("y_row_") or constraint_name.startswith("y_col_"):
                    _, _, i, u, j = constraint_name.split("_")
                    i = int(i); u = int(u); j = int(j)
                    verify_cut += val * x_e_sol.get((i, u), 0.0)
                    vars_to_add_G[(i, u)] = (self.x_index[(i, u)], vars_to_add_G.get((i, u), (None, 0.0))[1] + val)
            if len(vars_to_add_G) > 0:
                if verify_cut >= -1e-6:
                    print(f"Generated Bender cut is not violated by current solution (lhs={verify_cut}), skipping.")
                    return
                ind = [idx for _, (idx, v) in vars_to_add_G.items()]
                val = [v * -1.0 for _, (idx, v) in vars_to_add_G.items()]
                self.add(cplex.SparsePair(ind=ind, val=val), sense="G", rhs=0.0)
                self.cut_bender_count += 1
                if self.cut_bender_count % 100 == 0:
                    print(f"Added Bender cut #{self.cut_bender_count} total cuts: {self.cut_bender_count + self.cut_x_e_count}")
                # print(f"Added Bender cut #{self.cut_bender_count} total cuts: {self.cut_bender_count + self.cut_x_e_count}")
                # print(f"Added cut #{self.cut_bender_count} with {len(ind)} variables, scale={z_value}, non_zero_pi={len(non_zero_pi)}")


class SFDGenericCallback:

    def __init__(
        self,
        x_keys,
        e_keys,
        e_pos,
        x_index,
        V,
        M,
        subgraphs,
        qap_problem,
        cut_frequency
    ):

        self.x_keys = x_keys
        self.e_keys = e_keys
        self.e_pos = e_pos
        self.x_index = x_index
        self.V = V
        self.M = M
        self.subgraphs = subgraphs
        self.qap_problem = qap_problem
        self.cut_frequency = cut_frequency

        self.cut_generation_count = 0
        self.cut_bender_count = 0
        self.cut_x_e_count = 0
        

    #
    # Generic callback entry point
    #
    def invoke(self, context):

        #
        # Only relaxation context
        #
        if not context.in_relaxation():
            return

        depth = context.get_long_info(
            Context.info.node_depth
        )

        #
        # Limit separation frequency
        #
        if self.cut_generation_count < self.cut_frequency:
            self.cut_generation_count += 1
            return
        else:
            self.cut_generation_count = 0

        #
        # Optional depth limit
        #
        if depth > 10:
            return

        #
        # Current LP solution
        #
        x_vals = context.get_relaxation_point(
            list(range(0, len(self.x_keys)))
        )

        e_vals = context.get_relaxation_point(
            list(range(
                len(self.x_keys),
                len(self.x_keys) + len(self.e_keys)
            ))
        )

        x_e_sol = {
            self.x_keys[idx]: float(x_vals[idx])
            for idx in range(len(self.x_keys))
        }

        e_base_sol = {
            self.e_keys[idx]: float(e_vals[idx])
            for idx in range(len(self.e_keys))
        }

        e_sol = {
            (k, i, j):
                e_base_sol[
                    (k, i, j)
                    if i < j
                    else (k, j, i)
                ]
            for k in self.subgraphs
            for i in self.V
            for j in self.V
            if i != j
        }

        #
        # Solve separation problem
        #
        start_time = time.time()

        optimal, z_value, non_zero_pi = (
            solve_fixed_e_and_get_cuts(
                self.qap_problem,
                self.V,
                self.M,
                self.subgraphs,
                x_e_sol,
                e_sol
            )
        )

        end_time = time.time()

        print(
            "Time taken to solve fixed-e subproblem:",
            end_time - start_time
        )

        #
        # No cut
        #
        if optimal is None:
            return

        #
        # Violated cut found
        #
        if not optimal:

            vars_to_add_G = {}

            for constraint_name, val in non_zero_pi:

                if constraint_name.startswith("y_e_def_"):

                    _, _, _, k, i, j = (
                        constraint_name.split("_")
                    )

                    k = int(k)
                    i = int(i)
                    j = int(j)

                    vars_to_add_G[(k, i, j)] = (
                        self.e_pos(k, i, j),
                        vars_to_add_G.get(
                            (k, i, j),
                            (None, 0.0)
                        )[1] + val
                    )

                elif (
                    constraint_name.startswith("y_row_")
                    or
                    constraint_name.startswith("y_col_")
                ):

                    _, _, i, u, j = (
                        constraint_name.split("_")
                    )

                    i = int(i)
                    u = int(u)
                    j = int(j)

                    vars_to_add_G[(i, u)] = (
                        self.x_index[(i, u)],
                        vars_to_add_G.get(
                            (i, u),
                            (None, 0.0)
                        )[1] + val
                    )

            if len(vars_to_add_G) > 0:

                ind = [
                    idx
                    for _, (idx, v)
                    in vars_to_add_G.items()
                ]

                val = [
                    -1.0 * v
                    for _, (idx, v)
                    in vars_to_add_G.items()
                ]

                cut = cplex.SparsePair(
                    ind=ind,
                    val=val
                )

                #
                # Add user cut
                #
                context.add_user_cut(
                    cut=cut,
                    sense="G",
                    rhs=0.0,
                    cutmanagement=UserCutCallback.use_cut.purge,
                    local=False
                )

                self.cut_bender_count += 1

                print(
                    f"Added Bender cut "
                    f"#{self.cut_bender_count}"
                )

class SFDLazyCallback:

    def __init__(
        self,
        x_keys,
        e_keys,
        e_pos,
        x_index,
        V,
        M,
        subgraphs,
        qap_problem
    ):
        self.x_keys = x_keys
        self.e_keys = e_keys
        self.e_pos = e_pos
        self.x_index = x_index
        self.V = V
        self.M = M
        self.subgraphs = subgraphs
        self.qap_problem = qap_problem

        self.cut_generation_count = 0
        self.cut_bender_count = 0
        self.cut_x_e_count = 0

    #
    # IMPORTANT: modern entry point
    #
    def invoke(self, context):

        #
        # Lazy constraints ONLY at integer candidates
        #
        if not context.in_candidate():
            return

        #
        # Get candidate solution (integer solution)
        #
        x_vals = context.get_candidate_point(
            list(range(0, len(self.x_keys)))
        )

        e_vals = context.get_candidate_point(
            list(range(
                len(self.x_keys),
                len(self.x_keys) + len(self.e_keys)
            ))
        )

        #
        # Build solution dictionaries
        #
        x_e_sol = {
            self.x_keys[idx]: float(x_vals[idx])
            for idx in range(len(self.x_keys))
        }

        e_base_sol = {
            self.e_keys[idx]: float(e_vals[idx])
            for idx in range(len(self.e_keys))
        }

        e_sol = {
            (k, i, j):
                e_base_sol[
                    (k, i, j) if i < j else (k, j, i)
                ]
            for k in self.subgraphs
            for i in self.V
            for j in self.V
            if i != j
        }

        #
        # Run both cut generators
        #
        # self.add_x_e_cut(context, x_e_sol, e_sol)
        self.add_benders_cut(context, x_e_sol, e_sol)

    # -------------------------
    # x-e lazy feasibility cuts
    # -------------------------
    def add_x_e_cut(self, context, x_e_sol, e_sol):

        x_one_sol = {
            (i, u): x_e_sol[(i, u)]
            for (i, u) in self.x_keys
            if abs(x_e_sol[(i, u)] - 1.0) < 1e-6
        }

        for i in self.V:
            for j in self.V:
                if i == j:
                    continue

                for u in self.M:

                    if (i, u) not in x_one_sol:
                        continue

                    for k, (flow_value, arcs, nodes) in self.subgraphs.items():

                        e_val = e_sol.get((k, i, j), 0.0)

                        if e_val >= 0.1:
                            continue

                        sum_x_jv = sum(
                            x_e_sol.get((j, v), 0.0)
                            for v in self.M
                            if (u, v) in arcs
                        )

                        if sum_x_jv <= 0.99:
                            continue

                        ind = (
                            [self.e_pos(k, i, j)]
                            + [self.x_index[(i, u)]]
                            + [
                                self.x_index[(j, v)]
                                for v in nodes
                                if (u, v) in arcs
                            ]
                        )

                        val = (
                            [1.0]
                            + [-1.0]
                            + [-1.0] * len([
                                v for v in nodes
                                if (u, v) in arcs
                            ])
                        )

                        cut = cplex.SparsePair(
                            ind=ind,
                            val=val
                        )

                        #
                        # Reject candidate solution (lazy cut)
                        #
                        context.reject_candidate(
                            constraints=[cut],
                            senses="G",
                            rhs=[-1.0]
                        )

                        self.cut_x_e_count += 1

                        if self.cut_x_e_count % 100 == 0:
                            print(
                                f"x-e cuts: {self.cut_x_e_count}"
                            )

                        return  # important: stop after first violation

    # -------------------------
    # Benders feasibility cut
    # -------------------------
    def add_benders_cut(self, context, x_e_sol, e_sol):

        optimal, z_value, non_zero_pi = (
            solve_fixed_e_and_get_cuts(
                self.qap_problem,
                self.V,
                self.M,
                self.subgraphs,
                x_e_sol,
                e_sol
            )
        )

        if optimal is None:
            return

        if optimal:
            return

        vars_to_add = {}

        for name, val in non_zero_pi:

            if name.startswith("y_e_def_"):

                _, _, _, k, i, j = name.split("_")
                k, i, j = int(k), int(i), int(j)

                vars_to_add[(k, i, j)] = (
                    self.e_pos(k, i, j),
                    vars_to_add.get((k, i, j), (None, 0.0))[1] + val
                )

            elif name.startswith("y_row_") or name.startswith("y_col_"):

                _, _, i, u, j = name.split("_")
                i, u, j = int(i), int(u), int(j)

                vars_to_add[(i, u)] = (
                    self.x_index[(i, u)],
                    vars_to_add.get((i, u), (None, 0.0))[1] + val
                )

        if not vars_to_add:
            return

        ind = [idx for _, (idx, v) in vars_to_add.items()]
        val = [-1.0 * v for _, (idx, v) in vars_to_add.items()]

        cut = cplex.SparsePair(ind=ind, val=val)

        context.reject_candidate(
            constraints=[cut],
            senses="G",
            rhs=[0.0]
        )

        self.cut_bender_count += 1

        if self.cut_bender_count % 100 == 0:
            print(
                f"Benders cuts: {self.cut_bender_count}"
            )
            
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
        self.use_cuts = config.get("use_cuts", False)
        self.cut_frequency = config.get("cut_frequency", 10)  # Generate cuts every 10 nodes
        self.use_lazy_constraints = config.get("use_lazy_constraints", False)
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
        ) -> Tuple[cplex.Cplex, Dict]:
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
        
        cpx = cplex.Cplex()
        cpx.set_problem_name(f"SFD_QAP_{n}_{m}")
        
        
        x_keys = [(i, u) for i in V for u in M]
        x_index = {key: idx for idx, key in enumerate(x_keys)}
        x_names = [f"x_{i}_{u}" for (i, u) in x_keys]
        
        
        if is_symmetric:
            e_keys = [(k, i, j) for k in subgraphs for i in V for j in V if i <= j]
        else:
            e_keys = [(k, i, j) for k in subgraphs for i in V for j in V]
        e_local_index = {key: idx for idx, key in enumerate(e_keys)}
        e_index = {key: idx + len(x_keys) for idx, key in enumerate(e_keys)}
        e_obj = [0.0] * len(e_keys)
        e_names = [f"e_{k}_{i}_{j}" for (k, i, j) in e_keys]

        for k, (flow_value, arcs, nodes) in subgraphs.items():
            for i in V:
                for j in V:
                    if i != j:
                        if is_symmetric and i > j:
                            key = (k, j, i)
                        else:
                            key = (k, i, j)
                        e_obj[e_local_index[key]] += float(distances[i, j] * flow_value)

        if self.is_relax:
            cpx.variables.add(
                obj=[0.0] * len(x_keys),
                lb=[0.0] * len(x_keys),
                ub=[1.0] * len(x_keys),
                names=x_names
            )
            cpx.variables.add(obj=e_obj, lb=[0.0] * len(e_keys), names=e_names)

            
        else:
            cpx.variables.add(
                obj=[0.0] * len(x_keys),
                lb=[0.0] * len(x_keys),
                ub=[1.0] * len(x_keys),
                types=[cpx.variables.type.binary] * len(x_keys),
                names=x_names
            )
            cpx.variables.add(obj=e_obj, lb=[0.0] * len(e_keys), names=e_names)

        print(len(e_names), "e variables created out of", n*n*len(subgraphs), "possible")
        def e_pos(k, i, j):
            if is_symmetric and i > j:
                return e_index[(k, j, i)]
            return e_index[(k, i, j)]

        lin_expr = []
        senses = []
        rhs = []
        names = []

        for i in V:
            ind = [x_index[(i, u)] for u in M]
            lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * len(ind)))
            senses.append("E")
            rhs.append(1.0)
            names.append(f"assign_{i}")

        for u in M:
            ind = [x_index[(i, u)] for i in V]
            lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * len(ind)))
            senses.append("E")
            rhs.append(1.0)
            names.append(f"assigned_{u}")

        for k, (flow_value, arcs, nodes) in subgraphs.items():
            degree_out = {u: 0 for u in M}
            
            for u, v in arcs:
                degree_out[u] += 1
            for i in V:
                ind = [e_pos(k, i, j) for j in V if j != i] + [x_index[(i, u)] for u in M]
                val = [1.0] * sum(1 for j in V if j != i) + [-float(degree_out[u]) for u in M]
                lin_expr.append(cplex.SparsePair(ind=ind, val=val))
                senses.append("E")
                rhs.append(0.0)
                names.append(f"e_def_{k}_{i}")

        for i in V:
            for j in V:
                if i != j:
                    ind = [e_pos(k, i, j) for k in subgraphs]
                    lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * len(ind)))
                    senses.append("E")
                    rhs.append(1.0)
                    names.append(f"e_unique_{i}_{j}")

        if not self.use_lazy_constraints and not self.is_relax:
            # Add subgraph constraints directly if not using lazy constraints
            # Subgraph constraints
            for k, (f_k, G_k, G_n_k) in subgraphs.items():
                if not self.is_relax:
                    for i in V:
                        for j in V:
                            if i != j:
                                for u in G_n_k:
                                    ind = [e_pos(k, i, j)] + [x_index[(i, u)]] + [x_index[(j, v)] for v in G_n_k if (u, v) in G_k]
                                    val = [1.0] + [-1.0] + [-1.0] * sum(1 for v in G_n_k if (u, v) in G_k)
                                    lin_expr.append(cplex.SparsePair(ind=ind, val=val))
                                    senses.append("G")
                                    rhs.append(-1.0)
                                    names.append(f"edge_link_{i}_{j}_{u}_{k}")
                                    # model.add_constraint(
                                    #     e[k, i, j] >= -1 + x[i, u] + model.sum(
                                    #         x[j, v] for v in G_n_k if (u, v) in G_k
                                    #     ),
                                    #     ctname=f"edge_link_{i}_{j}_{u}_{k}"
                                    # )

        
        cpx.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=names)

        # Warm-start if provided
        if warmstart is not None:
            initial_vars = [0.0] * (len(x_keys) + len(e_keys))
            for (i, u), val in warmstart.items():
                initial_vars[x_index[(i, u)]] = val
                for (j, v), val in warmstart.items():
                    for k in subgraphs:
                        if (u, v) in subgraphs[k][1]:
                            if i != j:
                                if is_symmetric and i > j:
                                    e_key = (k, j, i)
                                else:
                                    e_key = (k, i, j)
                                if e_key in e_index:
                                    initial_vars[e_index[e_key]] = 1.0

            # add warm-start solution as MIP start so that CPLEX has to use it as a starting point for the search
            cpx.MIP_starts.add(cplex.SparsePair(ind=list(range(len(initial_vars))), val=initial_vars), cpx.MIP_starts.effort_level.auto)


        # Configure CPLEX parameters
        # model.parameters.timelimit = self.time_limit
        self.cpx = cpx
        self.x_keys = x_keys
        self.e_keys = e_keys
        self.x_index = x_index
        self.e_index = e_index
        self.V = V
        self.M = M
        self.subgraphs = subgraphs
        self.qap_problem = problem
        self.e_pos = e_pos
        
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
        self._create_model(problem, subgraphs, fixed_variables, warmstart)
        if self.use_lazy_constraints:
            # callback = self.cpx.register_callback(SFDLazyCallback)
            # callback.x_keys = self.x_keys
            # callback.e_keys = self.e_keys
            # callback.V = self.V
            # callback.M = self.M
            # callback.subgraphs = self.subgraphs
            # callback.x_index = self.x_index
            # callback.qap_problem = self.qap_problem
            # callback.e_pos = self.e_pos
            
            callback = SFDLazyCallback(
                x_keys=self.x_keys,
                e_keys=self.e_keys,
                e_pos=self.e_pos,
                x_index=self.x_index,
                V=self.V,
                M=self.M,
                subgraphs=self.subgraphs,
                qap_problem=self.qap_problem
            )

            self.cpx.set_callback(
                callback,
                contextmask=Context.id.candidate
            )
            
            
        # callback = self.cpx.register_callback(SFDUserCutCallback)
        # callback.x_keys = self.x_keys
        # callback.e_keys = self.e_keys
        # callback.V = self.V
        # callback.M = self.M
        # callback.subgraphs = self.subgraphs
        # callback.x_index = self.x_index
        # callback.qap_problem = self.qap_problem
        # callback.e_pos = self.e_pos
        if self.use_cuts:
            callback = SFDGenericCallback(
                x_keys=self.x_keys,
                e_keys=self.e_keys,
                e_pos=self.e_pos,
                x_index=self.x_index,
                V=self.V,
                M=self.M,
                subgraphs=self.subgraphs,
                qap_problem=self.qap_problem,
                cut_frequency=self.cut_frequency
            )

            self.cpx.set_callback(
                callback,
                contextmask=Context.id.relaxation
            )
        
        self.cpx.parameters.mip.cuts.mircut.set(-1)
        self.cpx.parameters.mip.cuts.gomory.set(-1)
        self.cpx.parameters.mip.cuts.flowcovers.set(-1)
        # Solve
        self.cpx.solve()
        num_nodes = self.cpx.solution.progress.get_num_nodes_processed()
        if self.use_cuts:
            print(f"# cut frequency {self.cut_frequency}")
        else:
            print("No cuts")
        print(f"CPLEX explored {num_nodes} nodes in the search tree.")

        elapsed_time = time.time() - start_time

        if self.cpx.solution.is_primal_feasible():
            solution = self.cpx.solution
            # Extract assignment
            assignment = [0] * problem.n
            for (i, u), idx in self.x_index.items():
                if solution.get_values(idx) > 0.5:
                    assignment[i] = u
                    
            # number of nodes visited in the search tree
            # print(f"Number of nodes explored: {self.cpx.solution.get_num_nodes()}")
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=assignment,
                objective=solution.get_objective_value(),
                lower_bound=solution.get_best_bound() if hasattr(solution, "get_best_bound") else None,
                time=elapsed_time,
            )
        else:
            print(f"Solver did not find a feasible solution. Status: {self.cpx.solution.get_status_string()}")
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=None,
                objective=float('inf'),
                lower_bound=self.cpx.solution.get_best_bound() if hasattr(self.cpx.solution, "get_best_bound") else None,
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
        warmstart = None
        # Load warm-start if provided
        if warmstart_path is not None:
            warmstart = read_warmstart(warmstart_path)
        # else:
        #     # use LocalSearchSolver to find a warm-start solution
        #     from qap.modules.local_search import LocalSearchSolver
            
        #     print(f"Running local search to find initial solution...")
        #     local_solver = LocalSearchSolver({})
        #     local_solution = local_solver.solve(problem, fixed_variables=None)
        #     print(f"Local search initial solution: obj={local_solution.objective:.6f}")
        #     if hasattr(problem, "fixed_assignments"):
        #         for i, u in problem.fixed_assignments.items():
        #             if local_solution.assignment[i] != u:
        #                 print(f"Warning: Local search solution violates fixed assignment at location {i}: assigned {local_solution.assignment[i]} vs fixed {u}")
                    
        #     warmstart = {(i, u): 1.0 for i, u in enumerate(local_solution.assignment)}
        
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
