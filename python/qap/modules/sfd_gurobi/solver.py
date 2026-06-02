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

import gurobipy as gp
from gurobipy import GRB

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from qap.core import Problem, Solution, write_result
from qap.core.solution_io import read_warmstart
from qap.decomposition import decompose_value_layer, decompose_value_only, decompose_value_only_no_cycle3

def find_violated_subgraph_constraints_binary_x(model, x_sol, e_sol, subgraphs):
    EPS = 1e-5
    # Build assignment map for variables that are (almost) 1
    x_sol_1 = {}
    assign_of_u = {}
    x_items = x_sol.items() if hasattr(x_sol, 'items') else [(k, x_sol[k]) for k in x_sol]
    for key, val in x_items:
        if val > 1 - EPS:
            x_sol_1[key] = val
            assign_of_u[key[1]] = key[0]

    if not x_sol_1:
        return []

    # map node u -> list of subgraph indices k where u appears
    k_u = {}
    for k, (_f_k, G_k, G_n_k) in subgraphs.items():
        for u in G_n_k:
            k_u.setdefault(u, []).append(k)

    result = []
    e_get = e_sol.get if hasattr(e_sol, 'get') else (lambda k: e_sol[k])
    lazy = model._lazy_constraints
    subs = subgraphs

    # For each assigned (i,u), check relevant k and neighbor v->j quickly
    for (i, u) in x_sol_1:
        ks = k_u.get(u)
        if not ks:
            continue
        for k in ks:
            _f_k, G_k, G_n_k = subs[k]
            # iterate neighbors directly from edge set G_k (faster than scanning G_n_k)
            # G_k contains (a,b) edges
            list_v = [v for (a, v) in G_k if a == u]
            for v in list_v:
                j = assign_of_u.get(v)
                if j is None:
                    continue
                key = (k, i, j, u)
                if key in lazy:
                    continue
                if e_get((k, i, j), 0.0) < 1 - 1e-12:
                    result.append((k, i, j, u))
    return result
    

def find_violated_subgraph_constraints_relax_x(model, x_sol, e_sol, subgraphs):
    EPS = 1e-5
    # Build assignment map for variables that are (almost) 1
    x_sol_1 = {}
    x_items = x_sol.items() if hasattr(x_sol, 'items') else [(k, x_sol[k]) for k in x_sol]
    for key, val in x_items:
        if val > 1 - EPS:
            x_sol_1[key] = val

    if not x_sol_1:
        return [], []

    assign_of_u = {u: i for (i, u) in x_sol_1}

    # Precompute for each subgraph k the out- and in-neighbors per node u
    out_neighbors = {}
    in_neighbors = {}
    for k, (_f_k, G_k, G_n_k) in subgraphs.items():
        out_map = {}
        in_map = {}
        for (a, b) in G_k:
            out_map.setdefault(a, []).append(b)
            in_map.setdefault(b, []).append(a)
        out_neighbors[k] = out_map
        in_neighbors[k] = in_map

    # Compute current node signature: set of fixed (i,u) with x~1 at this node
    current_node_sig = frozenset((i, u) for (i, u), val in x_sol.items() if val > 1 - EPS)

    result1 = []
    result2 = []
    x_get = x_sol.get if hasattr(x_sol, 'get') else (lambda k: x_sol[k])
    e_get = e_sol.get if hasattr(e_sol, 'get') else (lambda k: e_sol[k])
    lazy_uv = model._lazy_constraints_uv
    lazy_vu = model._lazy_constraints_vu
    n = model._n

    # For each assigned (i,u), check relevant k and j values
    for (i, u) in x_sol_1:
        ks = [k for k in subgraphs if u in subgraphs[k][2]]
        if not ks:
            continue
        x_i_u = x_get((i, u), 0.0)
        for k in ks:
            list_v1 = out_neighbors[k].get(u, [])
            list_v2 = in_neighbors[k].get(u, [])
            if not list_v1 and not list_v2:
                continue
            # iterate over possible facilities j
            for j in range(n):
                if i == j:
                    continue
                key_uv = (k, i, j, u)
                skip_uv = False
                existing_uv = lazy_uv.get(key_uv)
                if existing_uv is not None:
                    if hasattr(existing_uv, 'issubset'):
                        if existing_uv.issubset(current_node_sig):
                            skip_uv = True
                    else:
                        skip_uv = True
                if list_v1 and not skip_uv:
                    sum_x_jv = 0.0
                    for v in list_v1:
                        sum_x_jv += x_get((j, v), 0.0)
                    if e_get((k, i, j), 0.0) < -1 + x_i_u + sum_x_jv - EPS:
                        result1.append((k, i, j, u))
                skip_vu = False
                existing_vu = lazy_vu.get(key_uv)
                if existing_vu is not None:
                    if hasattr(existing_vu, 'issubset'):
                        if existing_vu.issubset(current_node_sig):
                            skip_vu = True
                    else:
                        skip_vu = True
                if list_v2 and not skip_vu:
                    sum_x_jv = 0.0
                    for v in list_v2:
                        sum_x_jv += x_get((j, v), 0.0)
                    if e_get((k, i, j), 0.0) < -1 + x_i_u + sum_x_jv - EPS:
                        result2.append((k, i, j, u))
    return (result1, result2)
                
def find_bender_cuts(master_model, x_sol, e_sol, subgraphs):
    # solve a subproblem
    # create a model for subproblem
    
    sub_model = gp.Model(name="Benders_subproblem")
    M = list(range(master_model._n))
    V = list(range(master_model._n))
    #  y_iujv
            
    y_keys = [
            (i, u, j, v)
                for i in V
                for u in M
                for j in V
                for v in M
                if i < j and u != v]
    
    y_var = sub_model.addVars(y_keys, vtype=GRB.CONTINUOUS, lb=0, name="y")
    y = {}
    for (i,u,j,v) in y_keys:
        y[i,u,j,v] = y_var[i,u,j,v]
        y[j,v,i,u] = y_var[i,u,j,v]  # Symmetry: y[i,u,j,v] = y[j,v,i,u]
        
    # sum_{(u,v) in G_k} y[i,u,j,v] == e_sol[k,i,j] for all k,i,j
    for k, (f_k, G_k, G_n_k) in subgraphs.items():
        for i in V:
            for j in V:
                if i != j:
                    sub_model.addConstr(
                        gp.quicksum(y[i,u,j,v] for (u,v) in G_k) == e_sol[k, i, j],
                        name=f"benders_sub_e_{k}_{i}_{j}"
                    )
    # sum_{j} y[i,u,j,v] = x_sol[i,u] for all i,u,v
    for i in V:
        for u in M:
            for v in M:
                if u != v:
                    sub_model.addConstr(
                        gp.quicksum(y[i,u,j,v] for j in V if j != i) == x_sol[i, u],
                        name=f"benders_sub_iuv_{i}_{u}_{v}"
                    )
    # sum_{v} y[i,u,j,v] = x_sol[i,u] for all i,u,j
    for i in V:
        for u in M:
            for j in V:
                if j != i:
                    sub_model.addConstr(
                        gp.quicksum(y[i,u,j,v] for v in M if v != u) == x_sol[i, u],
                        name=f"benders_sub_iuj_{i}_{u}_{j}"
                    )
    # objective: minimize sum_{i,j,u,v} D[i,j] * F[u,v] * y[i,u,j,v]
    sub_model.setObjective(0.0)
    # sub_model.setObjective(
    #     gp.quicksum(master_model._problem.D[i, j] * master_model._problem.F[u, v] * y[i, u, j, v] for i in V for j in V for u in M for v in M if i != j and u != v),
    #     GRB.MINIMIZE
    # )
    # sub_model.setParam("Method", 0)
    sub_model.Params.LogToConsole = 0
    sub_model.Params.Presolve = 0
    sub_model.Params.InfUnbdInfo = 1
    sub_model.Params.DualReductions = 0
    sub_model.optimize()
    
    result ={}
    alpha = {}
    beta = {}
    gamma = {}
    # if status is infeasible, get the Dual farkas ray and generate a cut
    if sub_model.Status == GRB.INFEASIBLE:
        for c in sub_model.getConstrs():
            if abs(c.FarkasDual) > 1e-8:
                # print(c.ConstrName, c.FarkasDual)
                if "benders_sub_e_" in c.ConstrName:
                    _,_,_, k, i, j = c.ConstrName.split("_")
                    k = int(k)
                    i = int(i)
                    j = int(j)
                    e_k_ij = master_model._e.get((k, i, j))
                    # print((e_k_ij, c.FarkasDual))
                    result[e_k_ij] = result.get(e_k_ij, 0) + c.FarkasDual
                    gamma[k, i, j] = c.FarkasDual
                if "benders_sub_iuv_" in c.ConstrName:
                    _,_,_, i, u, v = c.ConstrName.split("_")
                    i = int(i)
                    u = int(u)
                    v = int(v)
                    x_i_u = master_model._x[i, u]
                    # print((x_i_u, c.FarkasDual))
                    result[x_i_u] = result.get(x_i_u, 0) + c.FarkasDual
                    alpha[i, u, v] = c.FarkasDual
                if "benders_sub_iuj_" in c.ConstrName:
                    _,_,_, i, u, j = c.ConstrName.split("_")
                    i = int(i)
                    u = int(u)
                    j = int(j)
                    x_i_u = master_model._x[i, u]
                    # print((x_i_u, c.FarkasDual))
                    result[x_i_u] = result.get(x_i_u, 0) + c.FarkasDual
                    beta[i, u, j] = c.FarkasDual
        # add the cut to the master model
        # master_model.addConstr(
        #     gp.quicksum(key * val for key, val in result.items()) <= 0
        # )
        # master_model.update()
        
    # test Farkas ray yTA= 0
    # bTy = 0
    # for (i,u,j,v), var in y.items():
    #     aTy = 0
    #     bTy += alpha.get((i, u, v), 0)*x_sol.get((i, u), 0) + alpha.get((j, v, u), 0)*x_sol.get((j, v), 0)
    #     bTy += beta.get((i, u, j), 0)*x_sol.get((i, u), 0) + beta.get((j, v, i), 0)*x_sol.get((j, v), 0)
    #     for k, (f_k, G_k, G_n_k) in subgraphs.items():
    #         if (u,v) in G_k:
    #             aTy += gamma.get((k, i, j), 0)
    #             bTy += e_sol.get((k, i, j), 0)*gamma.get((k, i, j), 0)
    #         if (v,u) in G_k:
    #             aTy += gamma.get((k, j, i), 0)
    #             bTy += e_sol.get((k, j, i), 0)*gamma.get((k, j, i), 0)
                
    #     aTy += alpha.get((i, u, v), 0) + alpha.get((j, v, u), 0)
    #     aTy += beta.get((i, u, j), 0) + beta.get((j, v, i), 0)
        
    #     if aTy < -1e-6:
    #         print(f"Farkas ray violation for y[{i},{u},{j},{v}]: {aTy}")
    # if bTy > 1e-6:
    #     print(f"Farkas ray check: bTy={bTy}")
    return result

def sfdcallback(model, where):
    if where == GRB.Callback.MIPSOL:
        if model._use_lazy_constraints:
            x_sol = model.cbGetSolution(model._x)
            e_sol = model.cbGetSolution(model._e)
            
            # find violated subgraph constraints and add them as lazy constraints
            # violated_constraints = find_violated_subgraph_constraints_binary_x(model, x_sol, e_sol, model._subgraphs)
            # for (k, i, j, u) in violated_constraints:
            #     model.cbLazy(model._e[k, i, j] >= -1 + model._x[i, u] + gp.quicksum(model._x[j, v] for v in model._subgraphs[k][2] if (u, v) in model._subgraphs[k][1]))
            #     model._total_added_lazy += 1
            
            # find benders cuts and add them as lazy constraints
            benders_cuts = find_bender_cuts(model, x_sol, e_sol, model._subgraphs)
            for key, val in benders_cuts.items():
                model.cbLazy(key * val <= 0)
                model._total_added_benders += 1
            
    elif where == GRB.Callback.MIPNODE:
        if model._use_cuts:
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)

            if status == GRB.OPTIMAL:

                x_sol = model.cbGetNodeRel(model._x)
                e_sol = model.cbGetNodeRel(model._e)
                
                # find benders cuts and add them as cuts
                benders_cuts = find_bender_cuts(model, x_sol, e_sol, model._subgraphs)
                for key, val in benders_cuts.items():
                    model.cbCut(key * val <= 0)
                    model._total_added_benders += 1

class SFDGurobiSolver:
    """
    Subgraph Flow Decomposition Solver for QAP.
    
    Decomposes the problem into value layers and solves using Gurobi
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
        self.use_lazy_constraints = config.get("use_lazy_constraints", False)
        self.time_limit = config.get("time_limit", 3600)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.is_relax = config.get("is_relax", False)
        self.matrix_for_decomposition = config.get("matrix_for_decomposition", "flow")  # 'flow' or 'distance'
        self.binary_variables = config.get("binary_variables", "x")  
        self.lpmethod = config.get("lpmethod", "auto")  # 'auto', 'primal_simplex', 'dual_simplex', 'barrier'

    def _create_model(
            self,
            problem: Problem,
            subgraphs: Dict,
            fixed_variables: Optional[List[Tuple[int, int]]] = None,
            warmstart: Optional[Dict[Tuple[int, int], float]] = None,
        ) -> Tuple[gp.Model, Dict, Dict]:
        """
        Create Gurobi model for SFD formulation.

        Args:
            problem: QAP problem instance
            subgraphs: Decomposed subgraphs {k: (f_k, G_k, G_n_k), ...}
            fixed_variables: List of (i, u) to fix x[i,u] = 1
            warmstart: Dictionary {(i,u): value} for warm-starting
            ub_warmstart: Upper bound warm-start value
        Returns:
            model: Gurobi model
            x_vars: Assignment variables
            e_vars: Subgraph variables  
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
            print("Problem is symmetric. Using symmetric variable reduction for e[k,i,j].")
            e_set = [(k, i, j) for i in V for j in V for k in subgraphs if i < j]
        else:
            e_set = [(k, i, j) for i in V for j in V for k in subgraphs if i != j]
        model = gp.Model(name="QAP_SFD")
        
        if self.is_relax:
            x = model.addVars(n, m, vtype=GRB.CONTINUOUS, name="x", lb=0, ub=1)
            e = model.addVars(e_set, vtype=GRB.CONTINUOUS, name="e", lb=0, ub=1)
            
        else:
        # # Variables
            if self.binary_variables == "x":
                print("Using binary variables for x and continuous variables for e.")
                x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
                e = model.addVars(
                    e_set,
                    vtype=GRB.CONTINUOUS,
                    name="e",
                    lb=0,
                    ub=1
                )
            elif self.binary_variables == "e":
                print("Using continuous variables for x and binary variables for e.")
                x = model.addVars(n, m, vtype=GRB.CONTINUOUS, name="x", lb=0, ub=1)
                e = model.addVars(
                    e_set,
                    vtype=GRB.BINARY,
                    name="e"
                )

        print(len(e), "e variables created out of", n*n*len(subgraphs), "possible")
        e_vars = {}
        if is_symmetric:
            for (k, i, j) in e_set:
                e_vars[k, i, j] = e[k, i, j]
                e_vars[k, j, i] = e[k, i, j]  # Symmetry: e[k,j,i] = e[k,i,j]
        else:
            e_vars = { (k, i, j): e[k, i, j] for (k, i, j) in e_set }
        e = e_vars
                
        # check if problem has fixed assignments and add constraints
        if hasattr(problem, "fixed_assignments") and problem.fixed_assignments:
            print(f"Adding {problem.fixed_assignments} fixed assignment constraints.")
            for i, u in problem.fixed_assignments.items():
                # model.add_constraint(x[i, u] == 1, ctname=f"fixed_{i}_{u}")
                model.addConstr(x[i, u] == 1, name=f"fixed_{i}_{u}")

        # Objective function
        model.setObjective(gp.quicksum(
            distances[i, j] * subgraphs[k][0] * e[k, i, j]
            for i in V for j in V for k in subgraphs if (k, i, j) in e
        ), GRB.MINIMIZE)
        
        # Assignment constraints
        for i in V:
            if problem.n > problem.m:
                model.addConstr(
                    gp.quicksum(x[i, u] for u in M) == 1,
                    name=f"assign_facility_{i}"
                )
            else:
                model.addConstr(
                    gp.quicksum(x[i, u] for u in M) == 1,
                    name=f"assign_facility_{i}"
                )
        for u in M:
            model.addConstr(
                gp.quicksum(x[i, u] for i in V) == 1,
                name=f"assign_location_{u}"
            )

        if not self.use_lazy_constraints and not self.is_relax:
            # Add subgraph constraints directly if not using lazy constraints
            # Subgraph constraints
            total_subgraph_constraints = 0
            for k, (f_k, G_k, G_n_k) in subgraphs.items():
                if not self.is_relax:
                    for i in V:
                        for j in V:
                            if (k, i, j) in e:
                                for u in G_n_k:
                                    model.addConstr(
                                        e[k, i, j] >= -1 + x[i, u] + gp.quicksum(
                                            x[j, v] for v in G_n_k if (u, v) in G_k
                                        ),
                                        name=f"edge_link_{i}_{j}_{u}_{k}"
                                    )
                                    total_subgraph_constraints += 1
            print(f"Added {total_subgraph_constraints} subgraph constraints directly to the model.")

        for k, (f_k, G_k, G_n_k) in subgraphs.items():
            # Compute degree sequences
            degree_out = {}
            degree_in = {}
            for u, v in G_k:
                degree_out[u] = degree_out.get(u, 0) + 1
                degree_in[v] = degree_in.get(v, 0) + 1

            # Extract unique nodes
            nodes_out = set(u for u, v in G_k)
            nodes_in = set(v for u, v in G_k)

            # Flow conservation constraints
            for i in V:
                # Outflow constraint
                model.addConstr(
                    gp.quicksum(e[k, i, j] for j in V if (k, i, j) in e) ==
                    gp.quicksum(x[i, u] * degree_out.get(u, 0) for u in nodes_out),
                    name=f"flow_out_{i}_{k}"
                )
                if not is_symmetric:
                    # Inflow constraint
                    model.addConstr(
                        gp.quicksum(e[k, j, i] for j in V if (k, j, i) in e) ==
                        gp.quicksum(x[i, u] * degree_in.get(u, 0) for u in nodes_in),
                        name=f"flow_in_{i}_{k}"
                    )

        # sum_k e^k_ij <= 1 if using value_only
        if self.decomposition == "value_only":
            for i in V:
                for j in V:
                    if i != j:
                        model.addConstr(
                            gp.quicksum(e[k, i, j] for k in subgraphs if (k, i, j) in e) == 1,
                            name=f"flow_value_{i}_{j}"
                        )
                        

        # Fix variables if provided
        if fixed_variables is not None:
            print(f"Adding {len(fixed_variables)} fixed variable constraints.")
            for i, u in fixed_variables:
                model.addConstr(x[i, u] == 1, name=f"fix_x_{i}_{u}")

        # Warm-start if provided
        if warmstart is not None:
            for (i, u), val in warmstart.items():
                x[i, u].Start = val
                for (j, v), val in warmstart.items():
                    for k in subgraphs:
                        if (u, v) in subgraphs[k][1]:
                            if (k, i, j) in e:
                                e[k, i, j].Start = 1
            # disable heuristic improvement in Gurobi to rely more on the warm-start solution
            model.Params.Heuristics = 0

        # Configure Gurobi parameters
        model.Params.TimeLimit = self.time_limit
        
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

        # self.is_relax = True
        # Create model
        start_model_time = time.time()
        model, x, e = self._create_model(problem, subgraphs, fixed_variables, warmstart)
        print(f"Model creation time: {time.time() - start_model_time}")

        if self.lpmethod == "dual_simplex":
            model.Params.Method = 1
            # model.Params.NodeMethod = 1  # Use dual simplex at nodes as well
            print(f"Using dual simplex method for Gurobi.")
        elif self.lpmethod == "primal_simplex":
            model.Params.Method = 0
            # model.Params.NodeMethod = 0  # Use primal simplex at nodes as well
            print(f"Using primal simplex method for Gurobi.")
        elif self.lpmethod == "barrier":
            model.Params.Method = 2
            # model.Params.NodeMethod = 2  # Use barrier at nodes as well
            print(f"Using barrier method for Gurobi.")
        else:
            print(f"Using default LP method for Gurobi.")
            
        
        # model.setParam("Crossover", 2)
        # model.Params.NodeMethod = 1
        # # solve relaxation first
        # model.optimize()
        # print("=================================================")
        # # optain the linear relaxation solution variables that are non basic
        # e_non_basic = { (k, i, j): e[k, i, j] for (k, i, j) in e if e[k, i, j].VBasis != 0 }
        # x_non_basic = { (i, u): x[i, u] for (i, u) in x if x[i, u].VBasis != 0 }
        # print(f"Relaxation solution has {len(e_non_basic)} e variables non-basic and {len(x_non_basic)} x variables non-basic.")
        
        # # fix e[k,i,j] = 0 for those that are zero in the relaxation
        # # for (k, i, j), var in e_non_basic.items():
        # #     var.lb = 0
        # #     var.ub = 0
        # # fix x[i,u] = 0 for those that are zero in the relaxation
        # # change x to binary
        # for (i, u), var in x.items():
        #     var.vtype = GRB.BINARY
            
        # for (i, u), var in x_non_basic.items():
        #     model.addConstr(var == 0, name=f"fix_relax_x_{i}_{u}")
        # model.update()
        # print("=================================================")
        
        if self.use_lazy_constraints or self.use_cuts:
            model._x = x
            model._e = e
            model._use_lazy_constraints = self.use_lazy_constraints
            model._use_cuts = self.use_cuts
            model._problem = problem
            model._subgraphs = subgraphs
            model._lazy_constraints = {}
            model._lazy_constraints_uv = {}
            model._lazy_constraints_vu = {}
            model._total_added_lazy = 0
            model._total_added_user = 0
            model._total_added_benders = 0
            model._n = problem.n
            model.Params.LazyConstraints = 1
            model.optimize(sfdcallback)
        else:
            # Solve
            model.optimize()

        elapsed_time = time.time() - start_time
        if self.use_lazy_constraints or self.use_cuts:
            print(f"Total lazy constraints added: {model._total_added_lazy}")
            print(f"Total user cuts added: {model._total_added_user}")
            print(f"Total Benders cuts added: {model._total_added_benders}")
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Extract assignment
            assignment = [None] * problem.n
            for i in range(problem.n):
                for u in range(problem.n):
                    if (i, u) in x:
                        if x[i, u].X > 0.5:
                            assignment[i] = u
                            break
            e_vals = {}
            for k in subgraphs:
                e_vals[k] = {(i, j): e[k, i, j].X for i in range(problem.n) for j in range(problem.n) if (k, i, j) in e and e[k, i, j].X > 0.0}
            print(f"Nonzero e values: {sum(len(e_vals[k]) for k in e_vals)}")
            # print the number of nodes visited in the branch and bound tree
            nb_nodes = model.NodeCount
            print(f"Number of nodes explored: {nb_nodes}")
            
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=assignment,
                objective=model.ObjVal,
                lower_bound=model.ObjBound,
                time=elapsed_time,
            )
        else:
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=None,
                objective=None,
                lower_bound=None,
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
        # elif not self.is_relax:
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
        else:
            warmstart = None
            
        if problem.n > problem.m:
            # fill in dummy locations for unassigned facilities
            new_F = np.zeros((problem.n, problem.n))
            new_F[:problem.m, :problem.m] = problem.F
            problem.F = new_F
            
            problem.m = problem.n
        
        if self.matrix_for_decomposition == "distance":
            # Swap flow and distance for decomposition if specified in config
            problem.D, problem.F = problem.F, problem.D
            if warmstart is not None:
                warmstart = {(u, i): 1.0 for i, u in warmstart}  # Swap indices for warmstart as well
        
        diag_F = np.diag(problem.F)
        # if the diagonal is nonzero, make them zero
        if np.any(np.diag(problem.F) != 0):
            print("Warning: Flow matrix has nonzero diagonal entries, which may affect the validity of the SFD formulation. Setting diagonal entries to zero.")
            np.fill_diagonal(problem.F, 0)
        
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
        
        # if np.all(problem.D == problem.D.T) and not np.all(problem.F == problem.F.T):
        #     # convert flow matrix to symmetric by averaging with its transpose
        #     new_F = (problem.F + problem.F.T) / 2
        #     problem.F = new_F
        #     print("Converted flow matrix to symmetric for SFD formulation.")
        # if np.all(problem.F == problem.F.T) and np.all(problem.D == problem.D.T):
            # # convert flow matrix to asymmetric by taking upper triangular part
            # print("Converting symmetric flow matrix to asymmetric for SFD formulation.")
            # new_F = np.zeros_like(problem.F)
            # for u in range(problem.n):
            #     for v in range(u + 1, problem.n):
            #             new_F[u, v] = problem.F[u, v]*2
            #             new_F[v, u] = 0
            
            # convert distance matrix to asymmetric by taking upper triangular part
            # print("Converting symmetric distance matrix to asymmetric for SFD formulation.")
            # new_D = np.zeros_like(problem.D)
            # for i in range(problem.n):
            #     for j in range(i + 1, problem.n):
            #             new_D[i, j] = problem.D[i, j]*2
            #             new_D[j, i] = 0
            # problem.D = new_D
        
        
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
        
        # after solving, add diagonal entries of flow matrix back to solution objective if they were originally nonzero
        if np.any(diag_F != 0):
            problem.F = problem.F + diag_F
            assignment = solution.assignment
            # recompute objective value with the original flow matrix
            objective = 0
            for i in range(problem.n):
                for j in range(problem.n):
                    u = assignment[i]
                    v = assignment[j]
                    objective += problem.D[i, j] * problem.F[u, v]
            solution.objective = objective
            print(f"Updated objective value with original flow matrix diagonal entries: {solution.objective:.6f}")

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
