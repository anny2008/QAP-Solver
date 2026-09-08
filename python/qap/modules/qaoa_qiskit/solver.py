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
# Some docplex stacks still touch np.float_; make it an alias to be safe.
np.float_ = np.float64  # no-op if already defined

# ---------------- Project-specific imports (kept as in your code) ----------------
from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart

from qiskit_optimization import QuadraticProgram

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
 
from qiskit_ibm_runtime import QiskitRuntimeService

USE_HARDWARE = False
if USE_HARDWARE:
    from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
    from qiskit_ibm_runtime import SamplerV2 as Sampler
else:
    from qiskit_aer.primitives import Estimator, Sampler


# ============================================================================
# QAP -> QuadraticProgram
# ============================================================================

def build_qap_quadratic_program(A: np.ndarray, B: np.ndarray) -> QuadraticProgram:
    """
    Build the Quadratic Assignment Problem as a QuadraticProgram:

    minimize sum_{i,j,p,q} A[i,j] * B[p,q] * x_{i,p} * x_{j,q}
    s.t. sum_p x_{i,p} = 1  (each facility assigned once)
         sum_i x_{i,p} = 1  (each location used once)
         x_{i,p} in {0,1}
    """
    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1], \
        "A and B must be square matrices of same size."

    n = A.shape[0]
    qp = QuadraticProgram("QAP")

    # Binary variables x_{i,p}
    for i in range(n):
        for p in range(n):
            qp.binary_var(name=f"x_{i}_{p}")

    # Quadratic QAP objective
    quadratic = {}
    for i, j, p, q in product(range(n), repeat=4):
        coeff = float(A[i, j] * B[p, q])
        if coeff != 0.0:
            u = f"x_{i}_{p}"
            v = f"x_{j}_{q}"
            quadratic[(u, v)] = quadratic.get((u, v), 0.0) + coeff
    qp.minimize(quadratic=quadratic)

    # Assignment constraints
    for i in range(n):
        qp.linear_constraint(
            linear={f"x_{i}_{p}": 1.0 for p in range(n)},
            sense="==",
            rhs=1.0,
            name=f"assign_fac_{i}",
        )

    for p in range(n):
        qp.linear_constraint(
            linear={f"x_{i}_{p}": 1.0 for i in range(n)},
            sense="==",
            rhs=1.0,
            name=f"use_loc_{p}",
        )

    return qp

# ============================================================================
# QAP -> QUBO conversion
# ============================================================================

def choose_penalty(A: np.ndarray, B: np.ndarray, scale: float = 10.0) -> float:
    """
    Heuristic penalty strength λ for constraint violations.
    A safe rule of thumb: λ ~ scale * max|A| * max|B|
    You can increase if you see infeasible solutions dominating.
    """
    amax = float(np.max(np.abs(A))) if A.size else 1.0
    bmax = float(np.max(np.abs(B))) if B.size else 1.0
    lam = scale * amax * bmax
    return max(lam, 1.0)

def build_qap_qubo(
    A: np.ndarray,
    B: np.ndarray,
    penalty: float = 10.0,
) -> QuadraticProgram:
    """
    Build the Quadratic Assignment Problem as a QUBO (unconstrained).

    minimize:
        sum_{i,j,p,q} A[i,j] * B[p,q] * x_{i,p} * x_{j,q}
        + penalty * assignment penalties

    x_{i,p} ∈ {0,1}
    """

    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1]

    n = A.shape[0]
    qp = QuadraticProgram("QAP-QUBO")

    # Variables x_{i,p}
    for i in range(n):
        for p in range(n):
            qp.binary_var(name=f"x_{i}_{p}")

    linear = {}
    quadratic = {}

    # --- QAP objective ---
    for i, j, p, q in product(range(n), repeat=4):
        coeff = float(A[i, j] * B[p, q])
        if coeff != 0.0:
            u = f"x_{i}_{p}"
            v = f"x_{j}_{q}"
            quadratic[(u, v)] = quadratic.get((u, v), 0.0) + coeff

    # --- Penalty: each facility assigned once ---
    for i in range(n):
        vars_i = [f"x_{i}_{p}" for p in range(n)]

        # -sum x_k
        for u in vars_i:
            linear[u] = linear.get(u, 0.0) - penalty

        # +2 sum_{k<l} x_k x_l
        for p1 in range(n):
            for p2 in range(p1 + 1, n):
                u = f"x_{i}_{p1}"
                v = f"x_{i}_{p2}"
                quadratic[(u, v)] = quadratic.get((u, v), 0.0) + 2 * penalty

    # --- Penalty: each location used once ---
    for p in range(n):
        vars_p = [f"x_{i}_{p}" for i in range(n)]

        # -sum x_k
        for u in vars_p:
            linear[u] = linear.get(u, 0.0) - penalty

        # +2 sum_{k<l} x_k x_l
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                u = f"x_{i1}_{p}"
                v = f"x_{i2}_{p}"
                quadratic[(u, v)] = quadratic.get((u, v), 0.0) + 2 * penalty

    # Final QUBO objective
    qp.minimize(linear=linear, quadratic=quadratic)

    return qp
# ============================================================================
# QUBO -> Hamiltonian conversion
# ============================================================================
def qubo_to_hamiltonian(
    linear: Dict[int, float],
    quadratic: Dict[Tuple[int, int], float],
    num_vars: int,
) -> SparsePauliOp:
    """
    Convert a QUBO (from QuadraticProgram) to an Ising Hamiltonian.

    QUBO:
        f(x) = sum_i c_i x_i + sum_{i,j} Q_ij x_i x_j

    Mapping:
        x_i = (1 - Z_i) / 2

    Returns:
        SparsePauliOp with Z and ZZ terms (constant dropped).
    """

    h = defaultdict(float)   # Z_i coefficients
    J = defaultdict(float)   # Z_i Z_j coefficients

    # Linear terms
    for i, c in linear.items():
        h[i] -= c / 2.0

    # Quadratic terms
    for (i, j), q in quadratic.items():
        if i == j:
            # x_i^2 = x_i
            h[i] -= q / 2.0
        else:
            a, b = sorted((i, j))
            J[(a, b)] += q / 4.0
            h[i] -= q / 4.0
            h[j] -= q / 4.0

    # Build Pauli operators
    paulis = []
    coeffs = []

    # Z terms
    for i, c in h.items():
        if abs(c) > 1e-12:
            z = ["I"] * num_vars
            z[i] = "Z"
            paulis.append("".join(z))
            coeffs.append(c)

    # ZZ terms
    for (i, j), c in J.items():
        if abs(c) > 1e-12:
            z = ["I"] * num_vars
            z[i] = "Z"
            z[j] = "Z"
            paulis.append("".join(z))
            coeffs.append(c)

    return SparsePauliOp(paulis, coeffs)


# ---------------------------------------------------------------------
# 3) QAOA: build ansatz, optimize, sample
# ---------------------------------------------------------------------
def cost_func_estimator(params, ansatz, hamiltonian, estimator, objective_func_vals):
    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    print("Current params:", params)
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    print("Job submitted, waiting for result...")
    
    job = estimator.run(
        circuits=[ansatz],
        observables=[hamiltonian],
        parameter_values=[params],
    )

    print("Result received.")
    result = job.result()
    print("Result received.")
    cost = result.values[0]
 
    objective_func_vals.append(cost)
 
    return cost

# ============================================================================
# QAOA solve function (Qiskit 1.x)
# ============================================================================

def solve_qap_with_qaoa(
    qap_problem: Problem,
    penalty: Optional[float] = None,
):
    """
    End-to-end QAOA solver for QAP:
        - Build QUBO with penalties
        - Extract QUBO dicts (index-based)
        - Convert to Ising Hamiltonian
        - Optimize QAOA parameters (Estimator)
        - Sample and decode bitstrings
        - Return best feasible solution found (or best candidate)

    Returns:
        dict with keys:
          'x' : np.ndarray (binary solution vector)
          'lcost' : float (QUBO objective value)
    """
    A = qap_problem.D
    B = qap_problem.F
    n = A.shape[0]

    # --- Build QUBO (unconstrained)
    qp = build_qap_qubo(A, B, penalty=penalty)

    # Extract QUBO dicts (these are INDEX-based!)
    linear = qp.objective.linear.to_dict()          # Dict[int, float]
    quadratic = qp.objective.quadratic.to_dict()    # Dict[Tuple[int,int], float]
    num_vars = qp.get_num_vars()
    var_names = [v.name for v in qp.variables]
    print(f"QUBO has {num_vars} variables.")

    # --- QUBO → Hamiltonian
    H_cost = qubo_to_hamiltonian(linear, quadratic, num_vars)

    # --- Create circuit
    circuit = QAOAAnsatz(cost_operator=H_cost, reps=2)
    print(circuit.num_qubits, "qubits in the circuit.")
    circuit.measure_all()
    
    fig = circuit.draw("mpl")
    fig.savefig("qaoa_circuit.png")

    if USE_HARDWARE:
        service = QiskitRuntimeService()
        backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=144
        )
        print(backend)
    
        # Create pass manager for transpilation
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        print("Transpiling circuit...")
        candidate_circuit = pm.run(circuit)
        print("Transpilation done.")
    else:
        pm = generate_preset_pass_manager(optimization_level=3)
        candidate_circuit = pm.run(circuit)

    # fig = candidate_circuit.draw("mpl", fold=False, idle_wires=False)
    # print("Candidate circuit after transpilation:")
    # fig.savefig("qaoa_candidate_circuit.png")
    # print("saved candidate circuit figure.")
    
    initial_gamma = np.pi
    initial_beta = np.pi / 2
    init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]
    objective_func_vals = []  # Global variable
    if USE_HARDWARE:
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 1000
        
            # Set simple error suppression/mitigation options
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"
    else:
        estimator = Estimator()
    result = scipy.optimize.minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, H_cost, estimator, objective_func_vals),
        method="COBYLA",
        tol=1e-2,
    )
    plt.figure(figsize=(12, 6))
    plt.plot(objective_func_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig("objective_func_vals.png")
    return {'x': result.x, 'lcost': result.fun}
    

# ============================================================================
# Solver class
# ============================================================================

class QAOAQiskitSolver:
    """Solving QAP using QAOA algorithm (Qiskit 2.x)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Expected keys in config:
          - solver (str)
          - time_limit (float, optional)
          - seed (int, optional)
        """
        self.config = config
        self.solver_name = config.get("solver", "qaoa_qiskit")
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
        result = solve_qap_with_qaoa(
            qap_problem=problem,
            penalty=1,
        )
        print("QAOA result:", result)

        return Solution(
            instance="unknown",
            solver=self.solver_name,
            assignment=result['x'],
            objective=result['lcost'],
            lower_bound=result['lcost'],   # QUBO objective value
            time=time.time() - start,
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "QAOAQiskitSolver":
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
        "solver": "qaoa_qiskit",
        "time_limit": 120,
        "seed": 42,
    }
    solver = QAOAQiskitSolver(config)

    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="qaoa_result.json")
    else:
        print("Usage: python solver.py <instance_file>")