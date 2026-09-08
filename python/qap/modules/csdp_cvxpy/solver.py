"""
RLT1 SDP Solver Module

Relaxation-based Tightened Linear (RLT1) formulation solved with SDP.
Supports binary variables, relaxed 0-1 variables, and warm-start solutions.

Reference: Based on solve_QAP_RLT1_sdp from qap_new_formulation.py
"""

import time
import numpy as np

# Fix numpy 2.0 compatibility with dosdp
# np.float_ = np.float64
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cvxpy as cp

from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart

from itertools import combinations
from scipy import sparse



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


class CSDPSolver:
    """C-SDP formulation solver using SDP."""

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
        self.solver_name = config.get("solver", "csdp_cvxpy")
        self.formulation = config.get("formulation", "csdp")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)
        self.lpmethod = config.get("lpmethod", "auto")
        self.n_cliques = config.get("n_cliques", 2)  # Default clique size for C-SDP

        # Validate config
        assert self.formulation == "csdp", "C-SDP solver requires formulation='csdp'"

    @classmethod
    def from_config_file(cls, config_path: str) -> "CSDPSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            CSDPSolver: Initialized solver
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

        # Create model
        cliques = build_cliques_from_graph(problem.D, self.n_cliques)
        print(f"Created {len(cliques)} cliques of size {self.n_cliques} for CSDP formulation.")
        print(f"Cliques: {cliques}")
        Q, constraints, objective = self._create_model_csdp(problem, cliques)

        model = cp.Problem(objective, constraints)
        start_solving_time = time.time()
        # Solve model
        result = model.solve(solver=cp.MOSEK, verbose=self.log_output)

        solving_time = time.time() - start_solving_time
        elapsed_time = time.time() - start_time
        print(f"Solver finished in {solving_time:.2f} seconds with status {model.status}")
        print(f"Objective value: {model.value}")
        # solution = cpx.solution
        # Extract results
        if model.status in ["optimal", "optimal_inaccurate"]:
            # Extract assignment from x variables
            assignment = []
            result = Solution(
                instance=self.instance,  # Will be set by caller
                solver=self.solver_name,
                assignment=assignment,
                objective=model.value,
                lower_bound=model.value,
                time=elapsed_time,
            )

            return result
        else:
            # No solution found
            result = Solution(
                instance=self.instance,
                solver=self.solver_name,
                assignment=None,
                objective=None,
                lower_bound=None,
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
                problem.fixed_assignments = {u: i for i, u in problem.fixed_assignments.items()}

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
