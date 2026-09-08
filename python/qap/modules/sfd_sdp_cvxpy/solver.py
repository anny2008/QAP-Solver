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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from qap.core import Problem, Solution, write_result
from qap.core.solution_io import read_warmstart
from qap.decomposition import decompose_value_layer, decompose_value_only, decompose_value_only_no_cycle3

import cvxpy as cp

class SFDSDPSolver:
    """
    Subgraph Flow Decomposition Solver for QAP.
    
    Decomposes the problem into value layers and solves using SDP
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
        self.use_constr = config.get("use_constr", 1)
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
        ) -> Tuple[Dict[int, cp.Variable], cp.Variable, List[cp.Constraint], cp.Expression]:
        """
        Create SDP model for SFD formulation.

        Args:
            problem: QAP problem instance
            subgraphs: Decomposed subgraphs {k: (f_k, G_k, G_n_k), ...}
            fixed_variables: List of (i, u) to fix X[i,u] = 1
            warmstart: Dictionary {(i,u): value} for warm-starting
            ub_warmstart: Upper bound warm-start value
        Returns:
            model: SDP model
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
        if not is_symmetric:
            print("Warning: Flow and distance matrices are not symmetric. SFD formulation assumes symmetric matrices.")
        
        E = {}
        constraints = []
        # Variables
        # for each subgraph k, create E[k] as a DNN variable
        for k, (f_k, G_k, G_n_k) in subgraphs.items():
            E[k] = cp.Variable((n, n), symmetric=True)
            # constraints.append(E[k] >> 0)  # E[k] is positive semidefinite
            constraints.append(E[k] >= 0)  # E[k] is elementwise non-negative
            # e_kii = 0 for all i in V
            for i in V:
                constraints.append(E[k][i, i] == 0)
        
        X = cp.Variable((n, n))  # Assignment variables x[i,u] in {0,1}


        # check if problem has fixed assignments and add constraints
        if hasattr(problem, "fixed_assignments") and problem.fixed_assignments:
            print(f"Adding {problem.fixed_assignments} fixed assignment constraints.")
            for i, u in problem.fixed_assignments.items():
                constraints.append(X[i, u] == 1)

        # Objective function
        objective = cp.Minimize(
            cp.sum(
                [f_k * cp.trace(E[k] @ distances)
                for k, (f_k, G_k, G_n_k) in subgraphs.items()]
            )
        )

        # Assignment constraints
        for i in V:
            constr = cp.sum([X[i, u] for u in M]) == 1
            constraints.append(constr)
        for u in M:
            constr = cp.sum([X[i, u] for i in V]) == 1
            constraints.append(constr)

        for k, (f_k, G_k, G_n_k) in subgraphs.items():
            # Compute degree sequences
            degree_out = {}
            degree_in = {}
            for u, v in G_k:
                degree_out[u] = degree_out.get(u, 0) + 1
                degree_in[v] = degree_in.get(v, 0) + 1

            # Extract unique nodes
            nodes_out = set(u for u, v in G_k)
            # nodes_in = set(v for u, v in G_k)

            # Flow conservation constraints
            for i in V:
                # Outflow constraint
                constraints.append(
                    cp.sum([E[k][i, j] for j in V]) ==
                    cp.sum([X[i, u] * degree_out.get(u, 0) for u in nodes_out])
                )

        # sum_k e^k_ij == 1 if using value_only
        if self.decomposition == "value_only":
            for i in V:
                for j in V:
                    if i != j:
                        constraints.append(
                            cp.sum([E[k][i, j] for k in subgraphs]) == 1
                        )
                        

        # Fix variables if provided
        if fixed_variables is not None:
            print(f"Adding {len(fixed_variables)} fixed variable constraints.")
            for i, u in fixed_variables:
                constraints.append(
                    X[i, u] == 1
                )

        return E, X, constraints, objective

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
        E, X, constraints, objective = self._create_model(problem, subgraphs, fixed_variables, warmstart)
        
        model = cp.Problem(objective, constraints)
        print(f"Model creation time: {time.time() - start_model_time}")

        
        start_solve_time = time.time()
        result = model.solve(solver=cp.MOSEK, verbose=self.log_output)
        print(f"Model solving time: {time.time() - start_solve_time}")
        solving_time = time.time() - start_time
        print(f"Total elapsed time: {solving_time:.2f} seconds, status: {model.status}, objective: {model.value}")
        elapsed_time = time.time() - start_time
        if self.use_lazy_constraints or self.use_cuts:
            print(f"Total lazy constraints added: {model._total_added_lazy}")
            print(f"Total user cuts added: {model._total_added_user}")
            print(f"Total Benders cuts added: {model._total_added_benders}")
        if model.status in ["optimal", "optimal_inaccurate"]:
            # Extract assignment
            assignment = [None] * problem.n
            # for i in range(problem.n):
            #     for u in range(problem.n):
            #         if (i, u) in x:
            #             if x[i, u].X > 0.5:
            #                 assignment[i] = u
            #                 break
            # e_vals = {}
            # for k in subgraphs:
            #     e_vals[k] = {(i, j): e[k, i, j].X for i in range(problem.n) for j in range(problem.n) if (k, i, j) in e and e[k, i, j].X > 0.0}
            # print(f"Nonzero e values: {sum(len(e_vals[k]) for k in e_vals)}")
            # print the number of nodes visited in the branch and bound tree
            
            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=assignment,
                objective=model.value,
                lower_bound=model.value,
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
            print("Using distance matrix for decomposition instead of flow matrix.")
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
        # if np.all(problem.F == problem.F.T) and not np.all(problem.D == problem.D.T):
        #     # convert distance matrix to symmetric by averaging with its transpose
        #     new_D = (problem.D + problem.D.T) / 2
        #     problem.D = new_D
        #     print("Converted distance matrix to symmetric for SFD formulation.")
        
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
        
        if np.all(problem.F == problem.F.T) and np.all(problem.D == problem.D.T):
            print("Both flow and distance matrices are symmetric.")
        
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
