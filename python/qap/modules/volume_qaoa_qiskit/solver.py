"""
QAOA Qiskit module: Solve QAP using QAOA algorithm (Qiskit 1.x).
"""
from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
from itertools import product

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import torch
# Some docplex stacks still touch np.float_; make it an alias to be safe.
np.float_ = np.float64  # no-op if already defined

# ---------------- Project-specific imports (kept as in your code) ----------------
from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart

from qap.modules.volume_qaoa_qiskit.volume_pytorch import (
    VolumeParams,
    VolumeHooks,
    VolumeAlgorithm,
)


# =========================================================================
# QAP -> QUBO helpers (index-based, no Qiskit dependency)
# =========================================================================

def _build_qap_quadratic_terms(A: np.ndarray, B: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Build the quadratic coefficients for the QAP objective:
        sum_{i,j,p,q} A[i,j] * B[p,q] * x_{i,p} * x_{j,q}
    Returns a dict keyed by (idx1, idx2) with idx in [0, n*n).
    """
    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1]

    n = A.shape[0]
    quadratic: Dict[Tuple[int, int], float] = defaultdict(float)

    for i, j, p, q in product(range(n), repeat=4):
        coeff = float(A[i, j] * B[p, q])
        if coeff == 0.0:
            continue
        idx1 = i * n + p
        idx2 = j * n + q
        a, b = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
        quadratic[(a, b)] += coeff

    return quadratic


def _decode_assignment(x: np.ndarray, n: int) -> np.ndarray:
    return x.reshape((n, n))


def _compute_violations(x: np.ndarray, n: int) -> np.ndarray:
    xmat = _decode_assignment(x, n)
    row_viol = xmat.sum(axis=1) - 1.0
    col_viol = xmat.sum(axis=0) - 1.0
    return np.concatenate([row_viol, col_viol], axis=0)


def _qubo_energy(
    x: np.ndarray,
    linear: np.ndarray,
    neighbors: List[List[Tuple[int, float]]],
) -> float:
    energy = float(np.dot(linear, x))
    for i, adj in enumerate(neighbors):
        if x[i] == 0:
            continue
        for j, coeff in adj:
            if j <= i:
                continue
            if x[j] == 1:
                energy += coeff
    return energy


def _solve_qubo_local_search(
    linear: np.ndarray,
    neighbors: List[List[Tuple[int, float]]],
    sweeps: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    num_vars = linear.shape[0]

    best_x = None
    best_cost = float("inf")

    for _ in range(batch_size):
        x = rng.integers(0, 2, size=num_vars, dtype=np.int8)
        cost = _qubo_energy(x, linear, neighbors)

        for _ in range(sweeps):
            for j in rng.permutation(num_vars):
                if x[j] == 1:
                    delta = -linear[j]
                else:
                    delta = linear[j]

                for i, coeff in neighbors[j]:
                    if x[i] == 1:
                        delta += coeff if x[j] == 0 else -coeff

                if delta < 0:
                    x[j] = 1 - x[j]
                    cost += delta

        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()

    return best_x, best_cost


def _solve_qubo_bruteforce(
    linear: np.ndarray,
    neighbors: List[List[Tuple[int, float]]],
) -> Tuple[np.ndarray, float]:
    num_vars = linear.shape[0]
    best_x = None
    best_cost = float("inf")

    for mask in range(1 << num_vars):
        x = np.fromiter(((mask >> k) & 1 for k in range(num_vars)), dtype=np.int8)
        cost = _qubo_energy(x, linear, neighbors)
        if cost < best_cost:
            best_cost = cost
            best_x = x

    return best_x, best_cost


def solve_qap_with_volume(
    qap_problem: Problem,
    batch_size: int = 512,
    sweeps: int = 80,
    maxiters: int = 800,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Solve QAP using a simple Volume Algorithm loop with a QUBO subproblem.
    Returns:
        dict with keys:
          'x' : np.ndarray (binary vector length n*n)
          'lcost' : float (best primal objective value)
    """
    A = qap_problem.D
    B = qap_problem.F
    n = A.shape[0]
    num_vars = n * n

    quadratic = _build_qap_quadratic_terms(A, B)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hooks = QuboHooks(
        quadratic=quadratic,
        n=n,
        batch_size=batch_size,
        sweeps=sweeps,
        seed=seed,
        device=device,
    )

    params = VolumeParams(maxsgriters=maxiters)
    algo = VolumeAlgorithm(psize=num_vars, dsize=2 * n, hooks=hooks, params=params, device=device)
    result = algo.solve()

    u = result["dual_u"]
    rc = hooks.compute_rc(u)
    _, x_best, _, pcost = hooks.solve_subproblem(u, rc)

    x = x_best.detach().cpu().numpy()
    return {"x": x, "lcost": float(pcost)}
# ============================================================================
# Solver class
# ============================================================================
class QuboHooks(VolumeHooks):
    def __init__(
        self,
        quadratic: Dict[Tuple[int, int], float],
        n: int,
        batch_size: int,
        sweeps: int,
        seed: int,
        device: str,
    ) -> None:
        self.n = n
        self.num_vars = n * n
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.rng = np.random.default_rng(seed)
        self.device = device

        self.neighbors: List[List[Tuple[int, float]]] = [list() for _ in range(self.num_vars)]
        self.base_linear = np.zeros(self.num_vars, dtype=np.float64)

        for (i, j), coeff in quadratic.items():
            if i == j:
                self.base_linear[i] += coeff
            else:
                self.neighbors[i].append((j, coeff))
                self.neighbors[j].append((i, coeff))

    def compute_rc(self, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.num_vars, device=self.device)

    @torch.no_grad()
    def solve_subproblem(self, u: torch.Tensor, rc: torch.Tensor):
        """Solve the QUBO subproblem given dual variables u.
        Returns:
            lcost (float): Lagrangian cost
            x_best (np.ndarray): Best solution found (binary vector)
            v (torch.Tensor): Subgradient vector
            pcost (float): Primal cost
        """
        u_np = u.detach().cpu().numpy()

        dual_linear = np.zeros(self.num_vars, dtype=np.float64)
        for i in range(self.n):
            for p in range(self.n):
                dual_linear[i * self.n + p] = u_np[i] + u_np[self.n + p]

        linear_total = self.base_linear + dual_linear

        if self.num_vars <= 20:
            x_best, cost = _solve_qubo_bruteforce(linear_total, self.neighbors)
        else:
            x_best, cost = _solve_qubo_local_search(
                linear_total,
                self.neighbors,
                sweeps=self.sweeps,
                batch_size=self.batch_size,
                rng=self.rng,
            )

        v_np = _compute_violations(x_best, self.n)

        lcost = float(cost - u_np.sum())
        pcost = float(_qubo_energy(x_best, self.base_linear, self.neighbors))

        x_best_t = torch.tensor(x_best, dtype=torch.float32, device=self.device)
        v_t = torch.tensor(v_np, dtype=torch.float32, device=self.device)

        return lcost, x_best_t, v_t, pcost

    def heuristics(self, x: torch.Tensor) -> float:
        return float("inf")
    
class VolumeQAOAQiskitSolver:
    """Solving QAP using QAOA algorithm and The Volume Algorithm (Qiskit 2.x)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Expected keys in config:
          - solver (str)
          - time_limit (float, optional)
          - seed (int, optional)
        """
        self.config = config
        self.solver_name = config.get("solver", "volume_qaoa_qiskit")
        self.time_limit = config.get("time_limit", 120)
        self.seed = config.get("seed", 42)

    def solve(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Solution:

        start = time.time()
        # For testing, cut problem to size 4
        problem = problem.cut_problem(4)

        # NOTE: fixed_variables and warmstart are not used in this simple QAOA flow.
        result = solve_qap_with_volume(problem, batch_size=512, sweeps=80, maxiters=800)
        
        return Solution(
            instance="unknown",
            solver=self.solver_name,
            assignment=result['x'],
            objective=result['lcost'],
            lower_bound=result['lcost'],   # QUBO objective value
            time=time.time() - start,
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "VolumeQAOAQiskitSolver":
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def solve_instance(
        self,
        instance_path: str,
        output_path: Optional[str] = None,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart_path: Optional[str] = None,
    ) -> Solution:

        problem = Problem.from_qaplib(instance_path)

        warmstart = read_warmstart(warmstart_path) if warmstart_path else None

        solution = self.solve(
            problem,
            fixed_variables=fixed_variables,
            warmstart=warmstart,
        )
        solution.instance = Path(instance_path).stem

        if output_path:
            write_result(solution, output_path)

        return solution


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    config = {
        "solver": "volume_qaoa_qiskit",
        "time_limit": 120,
        "seed": 42,
    }
    solver = VolumeQAOAQiskitSolver(config)
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="qaoa_result.json")
    else:
        print("Usage: python solver.py <instance_file>")