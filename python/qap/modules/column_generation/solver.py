# solver.py (Updated)
# High-level Column Generation Solver (Class-Based)
# Pure CPLEX — integrates IncrementalRMP, PricingEngine, SeparationEngine
# Implements Algorithm 1: symmetry (hard) + two-sided Big-M balance + pricing
from pathlib import Path
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

# Project types
from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart
from qap.modules.local_search import LocalSearchSolver

from .incremental_rmp import IncrementalRMP
from .pricing import PricingEngine
from .separation import SeparationEngine

class ColumnGenerationCPLEXSolver:
    """
    High-level Column Generation solver for the QAP RTL1 relaxation.
    Integrates:
      - IncrementalRMP (pure CPLEX RMP)
      - PricingEngine (reduced-cost pricing; rc = c - A^T*pi)
      - SeparationEngine (symmetry constraints C7)
    """
    def __init__(self, config):
        self.config = config
        self.time_limit = config.get("time_limit", 120)
        self.add_most_negative = config.get("add_most_negative", True)
        self.use_bigM = config.get("use_bigM", False)
        self.bigM = 1e10
        self.eps = float(config.get("epsilon", 1e-7))
        self.max_iterations = int(config.get("max_iterations", 10000))
        self.log_output = config.get("log_output", False)
        self.sep_top_k = int(config.get("sep_top_k", 1))

    # --------------------------------------------------------------
    # Main entry point
    # --------------------------------------------------------------
    def solve(self, problem: Problem, fixed_variables: Optional[List[Tuple[int, int]]] = None, warmstart: Optional[Dict] = None) -> Solution:
        

        # Extract QAP data
        D = problem.D
        F = problem.F
        n = len(D)
        m = len(F)
        V = list(range(n))
        M = list(range(m))

        # Big-M scale (simple heuristic; adjust if needed)
        self.bigM = float(np.max(D) * np.max(F) * n*n)
        print(f"Using big-M value: {self.bigM}")

        fixed_assignments = self._merge_fixed_assignments(problem, fixed_variables)

        # Build phi cost coefficients
        phi = {(i,u,j,v): float(D[i][j]) * float(F[u][v]) for i in V for u in M for j in V for v in M}

        # Build initial Ω
        # Omega0 = self._build_initial_Omega(V, M, D, F, fixed_assignments)
        # Load Omega0 from file if exists (for warm start)
        omega_file = f"basis_cols_{problem.instance_name}.txt"
        if Path(omega_file).exists():
            print(f"Loading initial Omega from {omega_file}...")
            Omega0 = set()
            with open(omega_file, "r") as f:
                for line in f:
                    if line.startswith('y'): # y_i_u_j_v
                        (i, u, j, v) = map(int, line.strip()[2:].split('_'))
                        Omega0.add((i, u, j, v))
            print(f"Initial Omega size: {len(Omega0)}")
        else:            
            print(f"Building initial Omega...")
            Omega0 = self._build_initial_Omega(V, M, D, F, fixed_assignments)
            with open(omega_file, "w") as f:
                for (i,u,j,v) in Omega0:
                    f.write(f"{i},{u},{j},{v}\n")
            print(f"Initial Omega size: {len(Omega0)}")

        # Instantiate RMP
        rmp = IncrementalRMP(V, M, phi, bigM=self.bigM)
        
        rmp.ensure_all_rows()
        rmp.fix_x_assignments(fixed_assignments)
        # Add initial columns
        for (i, u, j, v) in Omega0:
            rmp.add_column(i, u, j, v, force=False)  # force=True to create all base rows for these columns
        

        # Engines
        pricing = PricingEngine(
            V,
            M,
            phi,
            add_most_negative=self.add_most_negative,
            eps=self.eps,
            fixed_assignments=fixed_assignments,
        )
        separation = SeparationEngine(V, M, eps=self.eps, top_k=self.sep_top_k)

        iteration = 0
        best_obj = None
        start_time_iteration = time.time()
        solving_start_time = time.time()
        while iteration < self.max_iterations:
            print(f"Last iteration time {time.time() - start_time_iteration:.2f}s")
            start_time_iteration = time.time()
            iteration += 1

            # Solve RMP
            t0 = time.time()
            status = rmp.solve()
            if self.log_output:
                print(f"RMP solved with status {status} in {time.time()-t0:.2f}s")
            if status == 3:  # Infeasible
                print(f"RMP is infeasible, adding warmstart columns and retrying...")
                if warmstart is not None:
                    for (i, u) in warmstart:
                        for (j, v) in warmstart:
                            if self._is_consistent_with_fixed(i, u, j, v, fixed_assignments):
                                rmp.add_column(i, u, j, v, force=True)
                else:
                    # Do a local search to find a good initial solution and add those columns to the RMP
                    # (This can help speed up convergence by starting with a better primal solution)
                    print(f"Running local search to find initial solution...")
                    local_solver = LocalSearchSolver({})
                    local_solution = local_solver.solve(problem, fixed_variables=fixed_assignments)
                    print(f"Local search initial solution: obj={local_solution.objective:.6f}")
                    if hasattr(problem, "fixed_assignments"):
                        for i, u in problem.fixed_assignments.items():
                            if local_solution.assignment[i] != u:
                                print(f"Warning: Local search solution violates fixed assignment at location {i}: assigned {local_solution.assignment[i]} vs fixed {u}")
                    
                    print(f"Adding initial columns from local search solution...")
                    for i, u in enumerate(local_solution.assignment):
                        for j, v in enumerate(local_solution.assignment):
                            if self._is_consistent_with_fixed(i, u, j, v, fixed_assignments):
                                rmp.add_column(i, u, j, v, force=True)
                            else:
                                print(f"Skipping column (i={i}, u={u}, j={j}, v={v}) due to fixed assignment conflict")
                
                

            # Current solution snapshot
            t0 = time.time()
            y_vals = rmp.get_y_values()
            print(f" -Retrieved RMP solution in {time.time()-t0:.2f}s; number of positive columns = {len(y_vals)}")
            
            # obj_val = sum(phi[k] * y_vals.get(k, 0.0) for k in rmp.Omega)
            added = False

            # 1) Symmetry separation (hard)
            t0 = time.time()
            new_cuts, new_cols = separation.separate_symmetry(rmp, y_vals)
            if self.log_output:
                print(f" -Symmetry separation took {time.time()-t0:.2f}s; added cuts={len(new_cuts)}, cols={len(new_cols)}")
            if new_cuts or new_cols:
                added = True
                if self.log_output:
                    print(f"  Added {len(new_cuts)} symmetry cuts and {len(new_cols)} columns")
                # re-solve next loop iteration
                continue
            
            if self.log_output:
                print('-' * 40)
                print(f"Iteration {iteration}:  obj={rmp.get_objective_value():.6f}")

            # 2) Pricing
            t0 = time.time()
            best_col, best_rc, negative_cols, most_negative_columns = pricing.price(rmp)
            if self.log_output:
                print(f" -Pricing took {time.time()-t0:.2f}s  ; best_rc={best_rc:.6f}, negative_cols={len(negative_cols)}, most_negative_columns={len(most_negative_columns)}")

            t0 = time.time()
            # if self.add_most_negative:
            #     if best_col is not None and best_rc < -self.eps:
            #         for idx, rc in most_negative_columns:
            #             (i, u, j, v) = idx
            #             if self._is_consistent_with_fixed(i, u, j, v, fixed_assignments):
            #                 rmp.add_column(i, u, j, v)
            #         added = True
            #         if self.log_output:
            #             print(f" Added {len(most_negative_columns)} most negative columns")
            # else:
            #     if negative_cols:
            #         for (idx, rc) in negative_cols:
            #             (i, u, j, v) = idx
            #             if self._is_consistent_with_fixed(i, u, j, v, fixed_assignments):
            #                 rmp.add_column(i, u, j, v)
            #         added = True
            #         if self.log_output:
            #             print(f" Added {len(negative_cols)} negative-rc columns")
            # print(f"Adding columns took {time.time()-t0:.2f}s")
            if len(most_negative_columns) > 1:
                # add all columns with rc within eps of best_rc
                for idx, rc in most_negative_columns:
                    (i, u, j, v) = idx
                    if self._is_consistent_with_fixed(i, u, j, v, fixed_assignments):
                        rmp.add_column(i, u, j, v)
                added = True
                if self.log_output:
                    print(f" Added {len(most_negative_columns)} most negative columns with rc within {self.eps} of best_rc={best_rc:.6f}")
            elif best_col is not None and best_rc < -self.eps:
                # add all negative columns
                for idx, rc in negative_cols:
                    (i, u, j, v) = idx
                    if self._is_consistent_with_fixed(i, u, j, v, fixed_assignments):
                        rmp.add_column(i, u, j, v)
                added = True
                if self.log_output:
                    print(f" Added {len(negative_cols)} negative-rc columns")
            
            
            if added:
                continue
            

            # Termination
            if not added:
                # for (i,u,j,v), val in y_vals.items():
                #     sym = (j,v,i,u)
                #     val_sym = y_vals.get(sym, 0.0)
                #     if abs(val - val_sym) > self.eps:
                #         print(f"Warning: C7 violated for (i={i}, u={u}, j={j}, v={v}), y={val:.6f} vs y_sym={val_sym:.6f}")
                #         return None
                break

        elapsed = time.time() - solving_start_time
        if self.log_output:
            obj_val = rmp.get_objective_value()
            print(f"Column Generation completed in {iteration} iterations, time={elapsed:.2f}s, obj={obj_val:.6f}")
        
        x_vals = rmp.get_x_values()
        assignment = [max(M, key=lambda u: x_vals.get((i,u), 0.0)) for i in V]
        
        
        
        # for i,u in x_vals:
        #     if x_vals[(i,u)] > self.eps:
        #         print(f"x[{i},{u}] = {x_vals[(i,u)]:.6f}")
                
        # check if x is valid:
        for i in V:
            sum_i = sum(x_vals.get((i,u), 0.0) for u in M)
            if abs(sum_i - 1.0) > self.eps:
                print(f"Warning: x variables for location {i} sum to {sum_i:.6f} (should be 1.0)")
        for u in M:
            sum_u = sum(x_vals.get((i,u), 0.0) for i in V)
            if abs(sum_u - 1.0) > self.eps:
                print(f"Warning: x variables for facility {u} sum to {sum_u:.6f} (should be 1.0)")

        # Check if solution is valid
        # C1
        for i in V:
            for j in V:
                if i != j:
                    s = sum(y_vals.get((i,u,j,v), 0.0) for u in M for v in M)
                    if s > 1.0 + self.eps:
                        print(f"Warning: C1 violated for (i={i}, j={j}), sum={s:.6f}")
        # C2
        for u in M:
            for v in M:
                if u != v:
                    s = sum(y_vals.get((i,u,j,v), 0.0) for i in V for j in V)
                    if s > 1.0 + self.eps:
                        print(f"Warning: C2 violated for (u={u}, v={v}), sum={s:.6f}")
        # C3
        for u in M:
            for j in V:
                s = sum(y_vals.get((i,u,j,v), 0.0) for i in V for v in M)
                if s > 1.0 + self.eps:
                    print(f"Warning: C3 violated for (u={u}, j={j}), sum={s:.6f}")
        # C4
        for i in V:
            for v in M:
                s = sum(y_vals.get((i,u,j,v), 0.0) for u in M for j in V)
                if s > 1.0 + self.eps:
                    print(f"Warning: C4 violated for (i={i}, v={v}), sum={s:.6f}")
        # C5
        for i in V:
            for u in M:
                for j in V:
                    s = sum(y_vals.get((i,u,j,v), 0.0) for v in M) - x_vals.get((i,u), 0.0)
                    if abs(s) > self.eps:
                        print(f"Warning: C5 violated for (i={i}, u={u}, j={j}), sum_v y - x = {s:.6f}")
                        return None
        # C6
        for i in V:
            for u in M:
                for v in M:
                    s = sum(y_vals.get((i,u,j,v), 0.0) for j in V) - x_vals.get((i,u), 0.0)
                    if abs(s) > self.eps:
                        print(f"Warning: C6 violated for (i={i}, u={u}, v={v}), sum_j y - x = {s:.6f}")
                        return None
        # C7
        for (i,u,j,v), val in y_vals.items():
            sym = (j,v,i,u)
            val_sym = y_vals.get(sym, 0.0)
            if abs(val - val_sym) > self.eps:
                print(f"Warning: C7 violated for (i={i}, u={u}, j={j}, v={v}), y={val:.6f} vs y_sym={val_sym:.6f}")
                return None
        
        basis_cols = rmp.get_basis()
        print(f"Basis columns: {len(basis_cols)}, number of columns in RMP: {len(rmp.Omega)}, number of rows: {len(rmp.c_index)}, number of C7 cuts: {len(rmp.c7_map)}")
        # Save basis cols to file for warm start
        with open(f"basis_cols_{problem.instance_name}.txt", "w") as f:
            for col in basis_cols:
                f.write(f"{col}\n")

        return Solution(
            instance="",
            assignment=assignment,
            objective=obj_val,
            time=elapsed,
            solver="column_generation_cplex",
            lower_bound=obj_val,
        )

    # --------------------------------------------------------------
    # Initial column set Ω
    # --------------------------------------------------------------
    def _build_initial_Omega(self, V, M, D, F, fixed_assignments):
        Omega0 = set()
        for i in V:
            for u in M:
                for j in V:
                    for v in M:
                        # structural validity: no mapping conflicts
                        if (i == j and u != v) or (i != j and u == v):
                            continue
                        if not self._is_consistent_with_fixed(i, u, j, v, fixed_assignments):
                            continue
                        if F[u][v] > 0 or F[v][u] > 0:
                            Omega0.add((i,u,j,v))
        return Omega0

    def _merge_fixed_assignments(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]],
    ) -> Dict[int, int]:
        merged: Dict[int, int] = {}
        fac_to_loc: Dict[int, int] = {}

        for loc, fac in getattr(problem, "fixed_assignments", {}).items():
            loc = int(loc)
            fac = int(fac)
            if loc in merged and merged[loc] != fac:
                raise ValueError(f"Conflicting fixed assignment for location {loc}: {merged[loc]} vs {fac}")
            if fac in fac_to_loc and fac_to_loc[fac] != loc:
                raise ValueError(f"Facility {fac} fixed to multiple locations: {fac_to_loc[fac]} and {loc}")
            merged[loc] = fac
            fac_to_loc[fac] = loc

        for loc, fac in (fixed_variables or []):
            loc = int(loc)
            fac = int(fac)
            if loc in merged and merged[loc] != fac:
                raise ValueError(f"Conflicting fixed assignment for location {loc}: {merged[loc]} vs {fac}")
            if fac in fac_to_loc and fac_to_loc[fac] != loc:
                raise ValueError(f"Facility {fac} fixed to multiple locations: {fac_to_loc[fac]} and {loc}")
            merged[loc] = fac
            fac_to_loc[fac] = loc

        return merged

    def _is_consistent_with_fixed(
        self,
        i: int,
        u: int,
        j: int,
        v: int,
        fixed_assignments: Dict[int, int],
    ) -> bool:
        if not fixed_assignments:
            return True
        fi = fixed_assignments.get(i)
        if fi is not None and fi != u:
            return False
        fj = fixed_assignments.get(j)
        if fj is not None and fj != v:
            return False
        return True

    # --------------------------------------------------------------
    # Instance runner
    # --------------------------------------------------------------
    def solve_instance(self, instance_path: str, output_path: Optional[str] = None, fixed_variables: Optional[List[Tuple[int, int]]] = None, warmstart_path: Optional[str] = None) -> Solution:
        print(f"Loading instance from {instance_path}...")
        if "QAPLIB" in instance_path:
            problem = Problem.from_qaplib(instance_path)
        else:
            problem = Problem.from_full_instance(
                matrix_file=instance_path,
                workstations_file=instance_path.replace(".txt", "_workstations.txt"),
                machines_file=instance_path.replace(".txt", "_machines.txt"),
                fixed_file=instance_path.replace(".txt", "_fixed.txt")
            )
        problem.instance_name = Path(instance_path).stem
        warmstart = read_warmstart(warmstart_path) if warmstart_path is not None else None
        solution = self.solve(problem, fixed_variables=fixed_variables, warmstart=warmstart)
        solution.instance = problem.instance_name
        if output_path:
            write_result(solution, output_path)
        return solution
