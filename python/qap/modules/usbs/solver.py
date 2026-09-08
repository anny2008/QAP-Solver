"""
USBS SDP Solver Module

Relaxation-based Tightened Linear (RLT1) formulation solved with SDP.
Supports binary variables, relaxed 0-1 variables, and warm-start solutions.

Reference: Based on solve_QAP_RLT1_sdp from qap_new_formulation.py
"""

import time
import os
import numpy as np

# Fix numpy 2.0 compatibility with dosdp
# np.float_ = np.float64
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# import cvxpy as cp

from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart

from itertools import combinations
from scipy import sparse
import jax
import jax.numpy as jnp
import json
import pickle
import sys

from .usbs import *
from .utils import *
from .lanczos import *



def build_cliques_from_graph(B, C):
    """
    Replicates the MATLAB implementation from find_variables.m.

    Parameters
    ----------
    B : ndarray (n,n)
        Sparse adjacency matrix.
    C : int
        Desired clique/variable size.

    Returns
    -------
    cliques : list[list[int]]
    """

    if C < 2:
        raise ValueError("C must be >= 2")

    # -----------------------------------
    # find_pairs(B)
    # -----------------------------------
    n = B.shape[0]

    cliques = []

    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j] != 0 or B[j, i] != 0:
                cliques.append(tuple(sorted((i, j))))

    # -----------------------------------
    # merge_variables(...)
    # -----------------------------------
    for _ in range(C - 2):
        new_cliques = set()
        for idx, cur in enumerate(cliques):
            cur_set = set(cur)

            # find clique with largest overlap
            best_overlap = -1
            best_clique = None

            for jdx, other in enumerate(cliques):
                if idx == jdx:
                    continue

                overlap = len(cur_set.intersection(other))

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_clique = other

            if best_clique is None:
                continue

            union_nodes = sorted(cur_set.union(best_clique))

            # MATLAB:
            # nchoosek(all_nodes, k+1)
            target_size = len(cur) + 1

            for subset in combinations(union_nodes, target_size):
                new_cliques.add(tuple(subset))

        cliques = sorted(new_cliques)

    return [list(c) for c in cliques]


class USBSSolver:
    """USBS SDP formulation solver using SDP."""

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
        jax.config.update("jax_enable_x64", True)
        # jax.config.update("jax_platform_name", "cpu")
        self.config = config
        self.instance_path = config.get("instance", "")
        self.instance = Path(self.instance_path).stem if self.instance_path else "unknown"
        self.solver_name = config.get("solver", "csdp_cvxpy")
        self.formulation = config.get("formulation", "zhao")
        self.max_iters = config.get("max_iters", 100)
        self.max_time = config.get("max_time", 120)
        self.obj_gap_eps = config.get("obj_gap_eps", -np.inf)
        self.infeas_gap_eps = config.get("infeas_gap_eps", -np.inf)
        self.max_infeas_eps = config.get("max_infeas_eps", -np.inf)
        self.lanczos_max_restarts = config.get("lanczos_max_restarts", 100)
        self.subprob_eps = config.get("subprob_eps", 1e-7)
        self.subprob_max_iters = config.get("subprob_max_iters", 15)
        self.cond_exp_base = config.get("cond_exp_base", 2.0)
        self.k_curr = config.get("k_curr", 1)
        self.k_past = config.get("k_past", 0)
        self.trace_factor = config.get("trace_factor", 1.0)
        self.rho = config.get("rho", 0.1)
        self.beta = config.get("beta", 0.25)
        self.num_drop = config.get("num_drop", 0)
        self.warm_start_strategy = config.get("warm_start_strategy", "none")


    @classmethod
    def from_config_file(cls, config_path: str) -> "USBSSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            USBSSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def _create_model_csdp(self, problem: Problem, cliques: list[list]) -> tuple:
        """
        Create SDP model for C-SDP formulation.

        Args:
            problem (Problem): QAP problem instance
            cliques (list[list]): List of cliques to consider

        Returns:
            tuple: (model, x_vars, y_vars)
        """
        n = problem.n
        distances = problem.D
        flows = problem.F

        required_pairs = set()
        for clique in cliques:
            for i in clique:
                for j in clique:
                    required_pairs.add((i, j))


        constraints = []
        # Create variables Q = vec(P) * vec(P)^T, where P is the assignment matrix
        # y_iujv = P(i, u) * P(j, v) = Q_ij(u, v)
        Q_mat = cp.Variable((n*n, n*n), symmetric=True)
        # constraints.append(Q_mat >> 0)
        constraints.append(Q_mat >= 0)
        # Q = vec(P) vec(P)^T
        # Q_ij is the (i,j)-th n x n block of Q
        # Q_ij = p_i p_j^T
        # where p_i is column i of P
        Q = {}
        for i in range(n):
            for j in range(n):
                Q[i, j] = Q_mat[i*n:(i+1)*n, j*n:(j+1)*n]
        # P = vec(P) is the vectorized assignment matrix
        # P[i*n + u] = P(i, u) = Q_ii(u, u) = y_iujv = Q_mat[i*n + u, i*n + u]
        P = cp.diag(Q_mat).reshape([-1, 1], order='F')  # vec(P) is the diagonal of Q_mat

        # 
        # Objective function
        K = sparse.kron(distances, flows)
        objective = cp.Minimize(cp.sum(cp.multiply(K, Q_mat)))

        # Constraints
        # Assignment constraints
        for i in range(n):
            constraints.append(cp.sum(P[i*n:(i+1)*n]) == 1)
        for u in range(n):
            constraints.append(cp.sum([P[i*n + u] for i in range(n)]) == 1)
        
        #
        # PSD constraint for each clique
        #
        for clique in cliques:
            k = len(clique)
            rows = []

            for a in clique:
                row = []
                # Q blocks
                for b in clique:
                    row.append(Q[a, b])

                # p_a column
                row.append(cp.reshape(P[a*n:(a+1)*n], (n, 1), order='F'))
                rows.append(row)

            # Last row
            last_row = []
            for a in clique:
                last_row.append(cp.reshape(P[a*n:(a+1)*n], (1, n), order='F'))

            last_row.append(np.ones((1, 1)))
            rows.append(last_row)
            constraints.append(cp.bmat(rows) >> 0)

        # sum_i Q_ii = In for all i
        for i in range(n):
            constraints.append(cp.sum([Q_mat[i*n + u, i*n + u] for u in range(n)]) == 1)
        
        # trace Q_ij = 0 for all i != j
        for i in range(n):
            for j in range(n):
                if i != j:
                    constraints.append(cp.sum(cp.diag(Q[i, j])) == 0)
        # # trace Q_ij Jn = 1 for all i,j
        # for i in range(n):
        #     for j in range(n):
        #         constraints.append(cp.trace(Q[i, j] @ np.ones((n, n))) == 1)
        # trace Q_ij (Jn - In) = 1 for all i != j
        M = np.ones((n,n)) - np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    constraints.append(cp.trace(Q[i, j] @ M) == 1)
        
        # trace Q_ii In = 1 for all i
        for i in range(n):
            constraints.append(cp.sum(cp.diag(Q[i, i] @ np.ones((n, n)))) == 1)

        # # diag Q_ii = p_i for all i
        # for i in range(n):
        #     constraints.append(cp.diag(Q[i, i]) == P[i*n:(i+1)*n])
        
        
        # # Q_ii(j,j = P(j,i) for all i,j
        # for i in range(n):
        #     for u in range(n):
        #         constraints.append(Q[i, i][u, u] == P[i*n + u])

        if(hasattr(problem, 'fixed_assignments') and problem.fixed_assignments is not None):
            for (i, u) in problem.fixed_assignments.items():
                print(f"Fixing assignment: P({i},{u}) = 1")
                constraints.append(P[i*n + u] == 1)
                # Also fix Q_ii(u,u) = 1
                # constraints.append(Q[i, i][u, u] == 1)

        return Q, constraints, objective


    def solve(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Solution:
        """
        Solve the QAP problem using C-SDP formulation.

        Args:
            problem (Problem): QAP problem instance
            fixed_variables (list, optional): List of (i, u) to fix x[i,u] = 1
            warmstart (dict, optional): Dictionary {(i,u): value} for warm-start

        Returns:
            Solution: Solution object with result
        """
        start_time = time.time()
        
        l, D, W, C = load_and_process_qap(problem)
        # set sketch_dim = -1 if we do not want sketch
        sdp_state = initialize_state(problem, C=C, sketch_dim=l+self.num_drop)
    
        trace_ub = self.trace_factor * float(l + 1) * sdp_state.SCALE_X
    
        k_curr = self.k_curr
        k_past = self.k_past
        
        if hasattr(problem, 'fixed_assignments') and problem.fixed_assignments is not None:
            fixed_assignment = problem.fixed_assignments
        else:
            fixed_assignment = {}
        n_fixed = len(fixed_assignment)
        callback_static_args = pickle.dumps({"l": l, "n_fixed": n_fixed})
        callback_nonstatic_args = {"D": D, "W": W, "fixed_assignment": fixed_assignment}
    
        if self.num_drop == 0:
            print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")
        else:
            print("\n+++++++++++++++++++++++++++++ WARM-START ++++++++++++++++++++++++++++++++++\n")
    
        
    
        sdp_state = usbs(
            sdp_state=sdp_state,
            n=sdp_state.C.shape[0],
            m=sdp_state.b.shape[0],
            trace_ub=trace_ub,
            trace_factor=self.trace_factor,
            rho=self.rho,
            beta=self.beta,
            k_curr=k_curr,
            k_past=k_past,
            max_iters=self.max_iters,
            max_time=self.max_time,
            obj_gap_eps=self.obj_gap_eps,
            infeas_gap_eps=self.infeas_gap_eps,
            max_infeas_eps=self.max_infeas_eps,
            lanczos_inner_iterations=32,
            lanczos_max_restarts=self.lanczos_max_restarts,
            subprob_eps=self.subprob_eps,
            subprob_max_iters=self.subprob_max_iters,
            cond_exp_base=self.cond_exp_base,
            callback_fn=qap_round,
            callback_static_args=callback_static_args,
            callback_nonstatic_args=callback_nonstatic_args,
        )
    
        ellapsed_time = time.time() - start_time
        print(f"\n+++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++\n")
        objective_value = -sdp_state.best_ub
        lower_bound = -sdp_state.best_obj_gap
        # Extract results
        print(f"Objective: {objective_value}, Time: {ellapsed_time:.2f}s")
        assignment = []
        result = Solution(
            instance=self.instance,  # Will be set by caller
            solver=self.solver_name,
            assignment=assignment,
            objective=objective_value,
            lower_bound=lower_bound,
            time=ellapsed_time,
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
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platform_name", "cpu")
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
            
            # padding machines to match number of facilities
            if problem.n > problem.m:
                print(f"Padding machines to match number of facilities: {problem.n} > {problem.m}")
                new_F = np.zeros((problem.n, problem.n))
                new_F[:problem.m, :problem.m] = problem.F
                problem.F = new_F
                problem.m = problem.n
            # swap distances and flows matrix
            problem.F, problem.D = problem.D, problem.F
            
            # swap i,u to u,i in fixed assignments
            if hasattr(problem, 'fixed_assignments') and problem.fixed_assignments is not None:
                print(f"Swapping fixed assignments from (i,u) to (u,i)")
                problem.fixed_assignments = {u: i for i, u in problem.fixed_assignments.items()}
                # shift fixed assignments to the end of the list
                
                problem.shiftFixedAssignmentsToEnd()

        # Load warm-start if provided
        if warmstart_path is not None:
            warmstart = read_warmstart(warmstart_path)
        else:
            warmstart = None

        if problem.n > problem.m:
            # padding machines to match number of facilities
            print(f"Padding machines to match number of facilities: {problem.n} > {problem.m}")
            new_F = np.zeros((problem.n, problem.n))
            new_F[:problem.m, :problem.m] = problem.F
            problem.F = new_F
            problem.m = problem.n
        
        
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
        "solver": "rlt1_sdp",
        "formulation": "rlt1",
        "is_relax": False,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = CSDPSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="rlt1_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )
