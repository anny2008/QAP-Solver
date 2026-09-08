"""
Subgraph Flow Both (SFB) Solver

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
from typing import Dict, Any, Optional, Set, Tuple, List

import gurobipy as gp
from gurobipy import GRB

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from qap.core import Problem, Solution, write_result
from qap.core.solution_io import read_warmstart
from qap.decomposition import decompose_value_layer, decompose_value_only, decompose_value_only_no_cycle3

class SFBGurobiSolver:
    """
    Subgraph Flow Both Solver for QAP.
    
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
        self.time_limit = config.get("time_limit", 3600)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.is_relax = config.get("is_relax", False)
        self.matrix_for_decomposition = config.get("matrix_for_decomposition", "flow")  # 'flow' or 'distance'
        self.binary_variables = config.get("binary_variables", "x")  

    def _create_model(
            self,
            problem: Problem,
            subgraphs_F: Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]],
            subgraphs_D: Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]],
            fixed_variables: Optional[List[Tuple[int, int]]] = None,
            warmstart: Optional[Dict[Tuple[int, int], float]] = None,
        ) -> Tuple[gp.Model, Dict, Dict]:
        """
        Create Gurobi model for SFB formulation.

        Args:
            problem: QAP problem instance
            subgraphs_F: Decomposed subgraphs for flow matrix {k: (f_k, G_k, G_n_k), ...}
            subgraphs_D: Decomposed subgraphs for distance matrix {k: (f_k, G_k, G_n_k), ...}
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
            print("Warning: Flow matrix has nonzero diagonal entries, which may affect the validity of the SFB formulation.")
        e_set = [(k,s,i) for k in subgraphs_F for s in subgraphs_D for i in subgraphs_D[s][2]]
        ub_e = [len(subgraphs_D[s][1]) for (k, s, i) in e_set]
        
        model = gp.Model(name="QAP_SFB")
        
        if self.is_relax:
            x = model.addVars(n, m, vtype=GRB.CONTINUOUS, name="x", lb=0, ub=1)
            e = model.addVars(e_set, vtype=GRB.CONTINUOUS, name="e", lb=0, ub=ub_e)
            
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
                    ub=ub_e
                )
            elif self.binary_variables == "e":
                print("Using continuous variables for x and integer variables for e.")
                x = model.addVars(n, m, vtype=GRB.CONTINUOUS, name="x", lb=0, ub=1)
                e = model.addVars(
                    e_set,
                    vtype=GRB.INTEGER,
                    name="e",
                    lb=0,
                    ub=ub_e
                )
        print(f"Created model with {n*m} x variables and {len(e_set)} e variables.")
                
        # check if problem has fixed assignments and add constraints
        if hasattr(problem, "fixed_assignments") and problem.fixed_assignments:
            print(f"Adding {problem.fixed_assignments} fixed assignment constraints.")
            for i, u in problem.fixed_assignments.items():
                model.add_constraint(x[i, u] == 1, ctname=f"fixed_{i}_{u}")

        # Objective function
        model.setObjective(gp.quicksum(
            subgraphs_D[s][0] * subgraphs_F[k][0] * e[k, s, i] for (k, s, i) in e
        ), GRB.MINIMIZE)
        
        # Assignment constraints
        for i in V:
            model.addConstr(
                gp.quicksum(x[i, u] for u in M) == 1,
                name=f"assign_facility_{i}"
            )
        for u in M:
            model.addConstr(
                gp.quicksum(x[i, u] for i in V) == 1,
                name=f"assign_location_{u}"
            )
        
        if not self.is_relax:
            # e_ksi >= x_iu -1 + sum_{(j,v): (u,v) in G_k and (i,j) in G_s} x[j,v] for all (k, s, i) in e, u in M
            for (k, s, i) in e:
                f_k, G_k, G_n_k = subgraphs_F[k]
                f_s, G_s, G_n_s = subgraphs_D[s]
                
                for u in M:
                    set_iv = [(j, v) for j in V for v in M if (u, v) in G_k and (i, j) in G_s]
                    set_j = set(j for j, v in set_iv)
                    set_v = set(v for j, v in set_iv)
                    
                    model.addConstr(
                        e[k, s, i] >= (x[i, u] - 1)*min(len(set_j), len(set_v))
                            + gp.quicksum(x[j, v] for (j, v) in set_iv),
                        name=f"e_{k}_{s}_{i}_{u}"
                    )
        for i in V:
            degree_i = 0
            for s, (f_s, G_s, G_n_s) in subgraphs_D.items():
                for i2, j in G_s:
                    if i2 == i:
                        degree_i += 1
            model.addConstr(
                gp.quicksum(e[k, s, i] for k in subgraphs_F for s in subgraphs_D if (k, s, i) in e and i in subgraphs_D[s][2]) == degree_i,
                name=f"e_flow_node_{i}"
            )
        
        for s, (f_s, G_s, G_n_s) in subgraphs_D.items():
            model.addConstr(
                gp.quicksum(e[k, s, i] for (k, s2, i) in e if s2 == s) == len(G_s),
                name=f"e_flow_subgraph_{s}"
            )
        
        
        # for k, (f_k, G_k, G_n_k) in subgraphs_F.items():
        #     model.addConstr(
        #         gp.quicksum(e[k, s, i] for (k2, s, i) in e if k2 == k) >= len(G_k),
        #         name=f"e_flow_subgraph_{k}"
        #     )
                    
        # for k, (f_k, G_k, G_n_k) in subgraphs_F.items():
        #     # Compute degree sequences
        #     degree_out = {}
        #     degree_in = {}
        #     for u, v in G_k:
        #         degree_out[u] = degree_out.get(u, 0) + 1
        #         degree_in[v] = degree_in.get(v, 0) + 1

        #     # Extract unique nodes
        #     nodes_out = set(u for u, v in G_k)
        #     nodes_in = set(v for u, v in G_k)

        #     # sum_s e_ksi = sum_u x_iu * degree_out(u) for all k, i
        #     for i in nodes_out:
        #         model.addConstr(
        #             gp.quicksum(e[k, s, i] for s in subgraphs_D if (k, s, i) in e) ==
        #             gp.quicksum(x[i, u] * degree_out.get(u, 0) for u in M),
        #             name=f"flow_out_{k}_{i}"
        #         )
                
        
        # for s, (f_s, G_s, G_n_s) in subgraphs_D.items():
        #     # Compute degree sequences
        #     degree_out = {}
        #     degree_in = {}
        #     for i,j  in G_s:
        #         degree_out[i] = degree_out.get(i, 0) + 1
        #         degree_in[j] = degree_in.get(j, 0) + 1

            # Extract unique nodes
            nodes_out = set(i for i, j in G_s)
            nodes_in = set(j for i, j in G_s)
            # sum_k e_ksi = sum_u x_iu * (degree_out(i) + degree_in(i)) for all s, i in G_n_s
            # for i in G_n_s:
            #     model.addConstr(
            #         gp.quicksum(e[k, s, i] for k in subgraphs_F if (k, s, i) in e) ==
            #         (degree_out.get(i, 0)+degree_in.get(i, 0)) * gp.quicksum(x[i, u] for u in M),
            #         name=f"flow_node_{s}_{i}"
            #     )
        # sum_ksi e_ksi = n*n
        # model.addConstr(
        #     gp.quicksum(e[k, s, i] for (k, s, i) in e) == 2*n*n,
        #     name=f"total_flow"
        # )
                        

        # Fix variables if provided
        if fixed_variables is not None:
            print(f"Adding {len(fixed_variables)} fixed variable constraints.")
            for i, u in fixed_variables:
                model.addConstr(x[i, u] == 1, name=f"fix_x_{i}_{u}")

        # Warm-start if provided
        if warmstart is not None:
            for (i, u), val in warmstart.items():
                x[i, u].Start = val
            for (i,u) in x:
                if (i, u) not in warmstart:
                    x[i, u].Start = 0.0
            warmstart_e = {}
            for k, (f_k, G_k, G_n_k) in subgraphs_F.items():
                for s, (f_s, G_s, G_n_s) in subgraphs_D.items():
                    for i in G_n_s:
                        if (k, s, i) in e:
                            # set e[k,s,i] = sum_{(j,u,v): (i,j) in G_s, (u,v) in G_k} x_iu*x_jv
                            e[k, s, i].Start = sum(warmstart.get((i, u), 0) * warmstart.get((j, v), 0) for (i2, j) in G_s for (u, v) in G_k if i2 == i)
                            warmstart_e[(k, s, i)] = sum(warmstart.get((i, u), 0) * warmstart.get((j, v), 0) for (i2, j) in G_s for (u, v) in G_k if i2 == i)
                        else:
                            print(f"Warning: (k={k}, s={s}, i={i}) not found in e variables for warm-starting.")
                            
            for k, (f_k, G_k, G_n_k) in subgraphs_F.items():
                sum_e = sum(warmstart_e.get((k, s, i), 0) for (k2, s, i) in e if k2 == k)
                print(f"Warm-start sum of e for flow subgraph {k}: {sum_e} (should be {len(G_k)})")
            for s, (f_s, G_s, G_n_s) in subgraphs_D.items():
                sum_e = sum(warmstart_e.get((k, s, i), 0) for k in subgraphs_F for i in subgraphs_D[s][2])
                print(f"Warm-start sum of e for distance subgraph {s}: {sum_e} (should be {len(G_s)})")
            for i in V:
                sum_e = sum(warmstart_e.get((k, s, i), 0) for k in subgraphs_F for s in subgraphs_D if i in subgraphs_D[s][2])
                degree_i = 0
                for s, (f_s, G_s, G_n_s) in subgraphs_D.items():
                    for i2, j2 in G_s:
                        if i2 == i:
                            degree_i += 1
                        # if j2 == i:
                        #     degree_i += 1
                print(f"Degree of node {i} in distance graph: {degree_i}, sum of warm-start e for node {i}: {sum_e}")
            # disable heuristic improvement in Gurobi to rely more on the warm-start solution
            model.Params.Heuristics = 0
        
        

        
        # for k, (f_k, G_k, G_n_k) in subgraphs_F.items():
        #     for s, (f_s, G_s, G_n_s) in subgraphs_D.items():
        #         model.addConstr(
        #             gp.quicksum(e[k, s, i] for (k2, s2, i) in e if k2 == k and s2 == s)
        #             ==  sum(warmstart_e.get((k2, s2, i), 0) for (k2, s2, i) in e if k2 == k and s2 == s),
        #             name=f"e_flow_subgraph_{k}"
        #         )

        # Configure CPLEX parameters
        model.Params.TimeLimit = self.time_limit
        
        return model, x, e

    def solve(
        self,
        problem: Problem,
        subgraphs_F: Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]],
        subgraphs_D: Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]],
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
        model, x, e = self._create_model(problem, subgraphs_F, subgraphs_D, fixed_variables, warmstart)

        # Solve
        model.optimize()

        elapsed_time = time.time() - start_time

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Extract assignment
            assignment = [None] * problem.n
            for i in range(problem.n):
                for u in range(problem.n):
                    if (i, u) in x:
                        if x[i, u].X > 0.5:
                            assignment[i] = u
                            break
            e_vals = {(k,s,i): e[k,s,i].X for (k, s, i) in e}
            print(f"Nonzero e values: {[(k, s, i, val) for (k, s, i), val in e_vals.items() if val > 1e-6]}")
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
        else:
            # use LocalSearchSolver to find a warm-start solution
            from qap.modules.local_search import LocalSearchSolver
            
            print(f"Running local search to find initial solution...")
            local_solver = LocalSearchSolver({})
            local_solution = local_solver.solve(problem, fixed_variables=None)
            print(f"Local search initial solution: obj={local_solution.objective:.6f}")
            if hasattr(problem, "fixed_assignments"):
                for i, u in problem.fixed_assignments.items():
                    if local_solution.assignment[i] != u:
                        print(f"Warning: Local search solution violates fixed assignment at location {i}: assigned {local_solution.assignment[i]} vs fixed {u}")
                    
            warmstart = {(i, u): 1.0 for i, u in enumerate(local_solution.assignment)}
        
        subgraphs_F = decompose_value_only(problem, matrix='flow')
        subgraphs_D = decompose_value_only(problem, matrix='distance')
        print(f"Decomposed flow matrix into {len(subgraphs_F)} subgraphs")
        print(f"Decomposed distance matrix into {len(subgraphs_D)} subgraphs.")
                
        # Solve
        solution = self.solve(
            problem,
            fixed_variables=fixed_variables,
            warmstart=warmstart,
            subgraphs_F=subgraphs_F,
            subgraphs_D=subgraphs_D
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
