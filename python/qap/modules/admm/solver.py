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

from docplex.mp.model import Model

from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart
import scipy.sparse as sp
from numpy.linalg import norm

from .admm_qap import ADMM_QAP
from .admm_qap_sparse import ADMM_QAPs


def build_Vchoice3(n):
    """
    Fully correct translation of MATLAB Vchoice = 3 (sparsest V).
    Returns:
        V : n x (n-1)
    """

    temp = np.zeros((n, n-1))
    dscale = np.ones(n-1)

    iblk = 1
    ii = 0

    sizeblocks = []
    normalize = []

    sizeblocks.append(n // (2**iblk))
    normalize.append(np.linalg.norm(np.ones(2**iblk)))

    while sizeblocks[iblk-1] >= 1:
        s = sizeblocks[iblk-1]          # number of columns
        ncols = s

        # base vector for current block
        basevec = np.concatenate([
            np.ones(2**(iblk-1)),
            -np.ones(2**(iblk-1))
        ])  # length = 2^iblk

        # total height = s * 2^iblk
        block_height = s * (2**iblk)

        block = np.zeros((n, ncols))

        for j in range(ncols):
            col = np.kron(np.eye(s)[j], basevec)
            block[:len(col), j] = col

        # place into temp
        temp[:, ii:ii+ncols] = block[:, :ncols]
        dscale[ii:ii+ncols] = normalize[iblk-1]

        ii += ncols
        iblk += 1

        sizeblocks.append(n // (2**iblk))
        normalize.append(np.linalg.norm(np.ones(2**iblk)))

    # Fix zero columns by nullspace completion
    col_norms = np.sum(temp*temp, axis=0)
    zero_ids = np.where(col_norms == 0)[0]
    if len(zero_ids) > 0:
        M = np.hstack([temp, np.ones((n, 1))])
        U, Svals, VT = np.linalg.svd(M.T)
        nullvecs = VT[-len(zero_ids):, :].T
        temp[:, zero_ids] = nullvecs

    # Normalize columns
    dd = np.sqrt(np.sum(temp*temp, axis=0))
    V = temp / dd

    V[np.abs(V) < 1e-12] = 0

    return V

def build_Vhat(n):
    V = build_Vchoice3(n)
    KVV = np.kron(V, V)        # shape (n^2, (n-1)^2)

    first_col = np.zeros((n*n+1, 1))
    first_col[0,0] = np.sqrt(1/2)
    first_col[1:,0] = (np.sqrt(1/2)/n)

    bottom = np.vstack([np.zeros((1, KVV.shape[1])), KVV])

    Vhat = np.hstack([first_col, bottom])
    Vhat[np.abs(Vhat) < 1e-12] = 0

    return Vhat

def build_gangster_mask(n):
    In = np.eye(n)
    En = np.ones((n, n))
    Es21 = np.triu(np.ones((n, n)), 1)

    YJ = np.kron(In, np.triu(En, 1)) + np.kron(Es21, In)
    YJ = np.block([
        [np.array([[1]]), np.zeros((1, n*n))],
        [np.zeros((n*n, 1)), YJ]
    ])
    YJ = YJ + YJ.T

    return (YJ != 0)

class ADMMSolver:
    """ADMM solver for the Quadratic Assignment Problem."""

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
        self.solver_name = config.get("solver", "admm_solver")
        self.formulation = config.get("formulation", "sdp")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)


    @classmethod
    def from_config_file(cls, config_path: str) -> "ADMMSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            ADMMSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

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
        
        n = problem.n
        A = problem.F
        B = problem.D
        L = np.block([
            [np.array([[0]]), np.zeros((1, n*n))],
            [np.zeros((n*n, 1)), np.kron(B, A)]
        ])

        Vhat = build_Vhat(n)
        J = build_gangster_mask(n)

        R0 = np.eye(Vhat.shape[1])
        Y0 = np.eye(n*n+1)
        Z0 = np.zeros_like(Y0)

        opts = {
            "R0": R0, "Y0": Y0, "Z0": Z0,
            "A": A, "B": B,
            "maxit": 1000000,
            "beta": n/3,
            "tol": 1e-5
        }
        print("==================================================")
        print("\nRunning dense ADMM...")
        start_time_1 = time.time()
        R1, Y1, Out1 = ADMM_QAP(L, Vhat, J, opts)
        elapsed_time_1 = time.time() - start_time_1
        print("Iterations:", Out1["iter"])
        print("Objective:", Out1["obj"])
        print("Time (dense): {:.2f}s".format(elapsed_time_1))

        # print("==================================================")
        # print("\nRunning sparse ADMM...")
        # start_time_2 = time.time()
        # R2, Y2, Out2 = ADMM_QAPs(L, Vhat, J, opts)
        # elapsed_time_2 = time.time() - start_time_2
        # print("Iterations:", Out2["iter"])
        # print("Objective:", Out2["obj"][-1])
        # print("Time (sparse): {:.2f}s".format(elapsed_time_2))
        elapsed_time = time.time() - start_time
        
        result = Solution(
            instance=self.instance,  # Will be set by caller
            solver=self.solver_name,
            assignment=None,
            objective=0,
            lower_bound=0,
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
            problem = Problem.from_qaplib(instance_path)
        else:
            problem = Problem.from_full_instance(
                matrix_file=instance_path,
                workstations_file=instance_path.replace(".txt", "_workstations.txt"),
                machines_file=instance_path.replace(".txt", "_machines.txt"),
                fixed_file=instance_path.replace(".txt", "_fixed.txt")
            )
            
        # if n > m, we need to pad the problem to make it square
        if problem.n > problem.m:
            print(f"Padding problem from n={problem.n}, m={problem.m} to n={problem.n}, m={problem.n}")
            F = np.zeros((problem.n, problem.n))
            F[:problem.m, :problem.m] = problem.F
            problem.F = F

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

# Example usage
if __name__ == "__main__":
    import sys

    # Create solver
    config = {
        "solver": "admm",
        "formulation": "sdp",
        "is_relax": True,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = ADMMSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="admm_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )
