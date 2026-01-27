"""
Subgraph Flow Decomposition (SFDBD) Solver

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

# Handle imports for different calling contexts
try:
    from python.qap.core import Problem, Solution, write_result
    from python.qap.core.solution_io import read_warmstart
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from qap.core import Problem, Solution, write_result
    from qap.core.solution_io import read_warmstart


class SFDBDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    """
    Lazy constraint callback for SFDBD formulation.
    
    Implements valid inequality generation for improving the bound.
    """
    
    def __init__(self, env):
        """Initialize callback."""
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_lazy = 0
        self.eps = 1e-6

    def initialize(self, x_vars: Dict, qap_problem: Problem):
        """
        Initialize callback with model variables and problem data.
        
        Args:
            x_vars: Binary assignment variables x[i,u]
            qap_problem: QAP problem instance
        """
        self.x = x_vars
        self.qap_problem = qap_problem

    def __call__(self):
        """Called by CPLEX at each lazy constraint check."""
        # Build solution from current node
        sol = self.make_complete_solution()
        
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


class SFDBDSolver:
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
        self.solver_name = config.get("solver", "SFDBD")
        self.formulation = config.get("formulation", "SFDBD")
        self.decomposition = config.get("decomposition", "value_layer")
        self.use_cuts = config.get("use_cuts", True)
        self.time_limit = config.get("time_limit", 3600)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.is_relax = config.get("is_relax", False)

    def _create_model_master_problem(
        self,
        problem: Problem,
        subgraphs: Dict,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[Model, Dict]:
        """
        Create CPLEX model for SFDBD formulation.

        Args:
            problem: QAP problem instance
            subgraphs: Decomposed subgraphs {k: (f_k, G_k, G_n_k), ...}
            fixed_variables: List of (i, u) to fix x[i,u] = 1
            warmstart: Dictionary {(i,u): value} for warm-starting

        Returns:
            model: CPLEX model
            x_vars: Assignment variables
        """
        n = problem.n
        distances = problem.D
        flows = problem.F

        model = Model(name="QAP_SFDBD")
        
        # Variables
        x = model.binary_var_matrix(n, n, name="x")
        alpha = model.continuous_var(name="alpha", lb=0)

        # Objective function
        model.minimize(alpha)

        # Assignment constraints
        for i in range(n):
            model.add_constraint(
                model.sum(x[i, u] for u in range(n)) == 1,
                ctname=f"assign_facility_{i}"
            )
        for u in range(n):
            model.add_constraint(
                model.sum(x[i, u] for i in range(n)) == 1,
                ctname=f"assign_location_{u}"
            )

        # Fix variables if provided
        if fixed_variables is not None:
            for i, u in fixed_variables:
                model.add_constraint(x[i, u] == 1, ctname=f"fix_x_{i}_{u}")

        # Warm-start if provided
        if warmstart is not None:
            ws = model.new_solution()
            for (i, u), val in warmstart.items():
                ws.add_var_value(x[i, u], val)
            model.add_mip_start(ws)

        # Configure CPLEX parameters
        model.parameters.timelimit = self.time_limit
        model.parameters.threads = self.threads
        model.parameters.mip.tolerances.mipgap = 0.0
        
        return model, x
    
    def create_model_subproblem(
        self,
        problem: Problem,
        subgraphs: Dict,
        x_bar: Dict,
    ) -> Tuple[Model, Dict]:
        """
        Create CPLEX model for SFDBD formulation.

        Args:
            problem: QAP problem instance
            subgraphs: Decomposed subgraphs
            x_bar: Fixed assignment variables from master problem
        """
        n = problem.n
        distances = problem.D
        flows = problem.F

        model = Model(name="QAP_SFDBD_subproblem")
        lambd = model.continuous_var_dict([(i, u, k) for i in range(n) for u in range(n) for k in subgraphs], name="lambda", ub=float('inf'), lb=-float('inf'))
        pi = model.continuous_var_dict([(i, u, k) for i in range(n) for u in range(n) for k in subgraphs], name="pi", ub=float('inf'), lb=-float('inf'))
        theta = model.continuous_var_dict([k for k in subgraphs], name="theta", lb=0)

    def solve(
        self,
        problem: Problem,
        subgraphs: Dict,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Solution:
        """
        Solve QAP using SFDBD formulation.

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
        model, x = self._create_model(problem, subgraphs, fixed_variables, warmstart)

        # Register lazy callback if using cuts
        if self.use_cuts:
            cb = model.register_callback(SFDBDLazyCallback)
            cb.initialize(x, problem)

        # Solve
        solution = model.solve(log_output=self.log_output)

        elapsed_time = time.time() - start_time

        if solution:
            # Extract assignment
            assignment = [None] * problem.n
            for i in range(problem.n):
                for u in range(problem.n):
                    if solution.get_value(x[i, u]) > 0.5:
                        assignment[i] = u
                        break

            return Solution(
                instance="unknown",
                solver=self.solver_name,
                assignment=assignment,
                objective=solution.objective_value,
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
    print("SFDBD Solver Module")
    print("Subgraph Flow Decomposition formulation for QAP")
