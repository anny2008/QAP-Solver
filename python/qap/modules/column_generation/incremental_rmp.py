# incremental_rmp.py — fully corrected (x, b, c + robust symmetry C7)
# Author: (updated for An)
# Python 3.10+, CPLEX 22.x+
#
# Key guarantees:
# - Pure LP (no integer vars), so duals are available.
# - C5/C6 implemented with x in [0,1] and Big-M slacks b, c.
# - Symmetry (C7) is enforced regardless of creation order:
#     * if C7 exists when a new y is added -> coefficient is set after creation
#     * if C7 is created later -> function populates coefficients for any existing columns
# - Includes a "y-only" symmetry separation helper for use in CG loop.

import cplex
from cplex import SparsePair


class IncrementalRMP:
    def __init__(self, V, M, phi, bigM=1e6, eps=1e-9, quiet=True, disable_presolve=True):
        """
        V: iterable of facilities indices
        M: iterable of location indices
        phi: dict {(i,u,j,v): cost}
        bigM: penalty for slacks b, c
        eps: numerical tolerance for separation checks
        """
        self.V = list(V)
        self.M = list(M)
        self.phi = dict(phi)
        self.bigM = float(bigM)
        self.eps = float(eps)

        # CPLEX model
        self.cpx = cplex.Cplex()
        self.cpx.set_problem_type(self.cpx.problem_type.LP)
        # Dual simplex is a good default for CG
        self.cpx.parameters.lpmethod.set(self.cpx.parameters.lpmethod.values.dual)

        # Silence streams if desired
        if quiet:
            self.cpx.set_log_stream(None)
            self.cpx.set_warning_stream(None)
            self.cpx.set_error_stream(None)
            self.cpx.set_results_stream(None)

        # Keep it a clean LP; presolve can sometimes move rows unexpectedly
        if disable_presolve:
            self.cpx.parameters.preprocessing.presolve.set(0)

        # Active y-columns
        self.Omega = set()       # set of (i,u,j,v)
        self.col_index = {}      # (i,u,j,v) -> var index

        # x, a, b, c indices
        self.x_index = {}        # (i,u)     -> var index
        self.a_index = {}        # (u,v)     -> var index
        self.b_index = {}        # (i,u,j)   -> var index
        self.c_index = {}        # (i,u,v)   -> var index

        # Row maps (lazy creation)
        self.c1_map = {}   # (i,j): sum_{u,v} y_{iujv} <= 1
        self.c2_map = {}   # (u,v): sum_{i,j} y_{iujv} = 1
        self.c3_map = {}   # (u,j): sum_{i,v} y_{iujv} <= 1
        self.c4_map = {}   # (i,v): sum_{u,j} y_{iujv} <= 1
        self.c5_map = {}   # (i,u,j): sum_v y_{iujv} - x_{iu} + b_{iuj} = 0
        self.c6_map = {}   # (i,u,v): sum_j y_{iujv} - x_{iu} + c_{iuv} = 0
        self.c7_map = {}   # canonical (i,u,j,v): y_{iujv} - y_{jviu} = 0

        self.num_rows = 0

    # -----------------------------
    # Variable creation (all CONT.)
    # -----------------------------
    def _ensure_x(self, i, u):
        key = (i, u)
        if key in self.x_index:
            return self.x_index[key]
        name = f"x_{i}_{u}"
        idx = self.cpx.variables.get_num()
        self.cpx.variables.add(names=[name], lb=[0.0], ub=[1.0], obj=[0.0], types=["C"])
        self.x_index[key] = idx
        return idx

    def _ensure_a(self, u, v):
        key = (u, v)
        if key in self.a_index:
            return self.a_index[key]
        name = f"a_{u}_{v}"
        idx = self.cpx.variables.get_num()
        self.cpx.variables.add(names=[name], lb=[0.0], ub=[1.0], obj=[self.bigM], types=["C"])
        self.a_index[key] = idx
        return idx

    def _ensure_b(self, i, u, j):
        key = (i, u, j)
        if key in self.b_index:
            return self.b_index[key]
        name = f"b_{i}_{u}_{j}"
        idx = self.cpx.variables.get_num()
        self.cpx.variables.add(names=[name],
                               lb=[0.0], ub=[cplex.infinity],
                               obj=[self.bigM], types=["C"])
        self.b_index[key] = idx
        return idx

    def _ensure_c(self, i, u, v):
        key = (i, u, v)
        if key in self.c_index:
            return self.c_index[key]
        name = f"c_{i}_{u}_{v}"
        idx = self.cpx.variables.get_num()
        self.cpx.variables.add(names=[name],
                               lb=[0.0], ub=[cplex.infinity],
                               obj=[self.bigM], types=["C"])
        self.c_index[key] = idx
        return idx

    # -----------------------------
    # Row creation
    # -----------------------------
    def ensure_all_rows(self):
        for i in self.V:
            for j in self.V:
                self._ensure_c1(i, j, force=True)
        for u in self.M:
            for v in self.M:
                self._ensure_c2(u, v, force=True)
        for u in self.M:
            for j in self.V:
                self._ensure_c3(u, j, force=True)
        for i in self.V:
            for v in self.M:
                self._ensure_c4(i, v, force=True)
        for i in self.V:
            for u in self.M:
                for j in self.V:
                    self._ensure_c5(i, u, j, force=True)
        for i in self.V:
            for u in self.M:
                for v in self.M:
                    self._ensure_c6(i, u, v, force=True)
    
    def _ensure_row(self, key, rhs, sense, row_map, force=False):
        if key in row_map:
            return row_map[key]
        else:
            if not force:
                raise ValueError(f"Row with key {key} does not exist and will be created with rhs={rhs}, sense={sense}.")
        self.cpx.linear_constraints.add(lin_expr=[SparsePair([], [])], senses=[sense], rhs=[rhs])
        r = self.num_rows
        row_map[key] = r
        self.num_rows += 1
        return r

    def _ensure_c1(self, i, j, force=False): return self._ensure_row((i, j), 1.0, 'L', self.c1_map, force=force)
    def _ensure_c3(self, u, j, force=False): return self._ensure_row((u, j), 1.0, 'L', self.c3_map, force=force)
    def _ensure_c4(self, i, v, force=False): return self._ensure_row((i, v), 1.0, 'L', self.c4_map, force=force)

    def _ensure_c2(self, u, v, force=False):
        """sum_v y_{i,u,j,v} - x_{i,u} + b_{i,u,j} = 0"""
        r = self._ensure_row((u, v), 1.0, 'E', self.c2_map, force=force)
        a_idx = self._ensure_a(u, v)
        self.cpx.linear_constraints.set_coefficients([(r, a_idx, 1.0)])
        return r
        
    def _ensure_c5(self, i, u, j, force=False):
        """sum_v y_{i,u,j,v} - x_{i,u} + b_{i,u,j} = 0"""
        r = self._ensure_row((i, u, j), 0.0, 'L', self.c5_map, force=force)
        x_idx = self._ensure_x(i, u)
        b_idx = self._ensure_b(i, u, j)
        self.cpx.linear_constraints.set_coefficients([(r, x_idx, -1.0), (r, b_idx, 1.0)])
        return r

    def _ensure_c6(self, i, u, v, force=False):
        """sum_j y_{i,u,j,v} - x_{i,u} + c_{i,u,v} = 0"""
        r = self._ensure_row((i, u, v), 0.0, 'E', self.c6_map, force=force)
        x_idx = self._ensure_x(i, u)
        c_idx = self._ensure_c(i, u, v)
        self.cpx.linear_constraints.set_coefficients([(r, x_idx, -1.0), (r, c_idx, 1.0)])
        return r

    # -----------------------------
    # Canonical helper for symmetry
    # -----------------------------
    @staticmethod
    def _canonical(i, u, j, v):
        t, s = (i, u, j, v), (j, v, i, u)
        return t if t <= s else s

    # -----------------------------
    # Add symmetry row (C7)
    # -----------------------------
    def add_symmetry_cut(self, i, u, j, v):
        """
        Add symmetry equality: y_{i,u,j,v} - y_{j,v,i,u} = 0
        Robust to creation order; after creating the row, we attach coeffs
        for any columns that already exist on either side of the pair.
        """
        can = self._canonical(i, u, j, v)
        if can in self.c7_map:
            return self.c7_map[can]

        # Create the equality row
        self.cpx.linear_constraints.add(lin_expr=[SparsePair([], [])], senses=['E'], rhs=[0.0])
        r = self.num_rows
        self.c7_map[can] = r
        self.num_rows += 1

        # Attach coefficients for any already-existing y columns on both sides
        t, s = (i, u, j, v), (j, v, i, u)
        if t in self.col_index:
            self.cpx.linear_constraints.set_coefficients([(r, self.col_index[t],  1.0 if t == can else -1.0)])
        if s in self.col_index:
            self.cpx.linear_constraints.set_coefficients([(r, self.col_index[s], -1.0 if t == can else  1.0)])
        return r

    # -----------------------------
    # Add a y-variable (column)
    # -----------------------------
    def add_column(self, i, u, j, v, force=False):
        """Add y(i,u,j,v) and attach all row coefficients (C1..C6). Also
        post-creation, if C7 row exists, enforce ±1 coefficients for this y
        and its mate (if present)."""
        key = (i, u, j, v)
        if key in self.col_index or key in self.Omega:
            return self.col_index[key]

        row_indices, row_values = [], []

        # Base rows: C1..C6 (+1 each)
        for f, args in [
            (self._ensure_c1, (i, j, force)),
            (self._ensure_c2, (u, v, force)),
            (self._ensure_c3, (u, j, force)),
            (self._ensure_c4, (i, v, force)),
            (self._ensure_c5, (i, u, j, force)),
            (self._ensure_c6, (i, u, v, force)),
        ]:
            r = f(*args)
            row_indices.append(r)
            row_values.append(1.0)

        # If C7 row already exists, tentatively include it
        can = self._canonical(i, u, j, v)
        if can in self.c7_map:
            r = self.c7_map[can]
            sign = 1.0 if key == can else -1.0
            row_indices.append(r)
            row_values.append(sign)

        # Create y variable FIRST
        name = f"y_{i}_{u}_{j}_{v}"
        idx = self.cpx.variables.get_num()
        self.cpx.variables.add(
            names=[name],
            obj=[self.phi[key]],
            lb=[0.0], ub=[cplex.infinity],
            columns=[SparsePair(row_indices, row_values)]
        )
        self.Omega.add(key)
        self.col_index[key] = idx

        return idx

    def fix_x_assignments(self, fixed_assignments):
        if not fixed_assignments:
            return
        items = fixed_assignments.items() if isinstance(fixed_assignments, dict) else fixed_assignments
        fixed_by_loc = dict(items)
        fixed_by_fac = {fac: loc for loc, fac in fixed_by_loc.items()}

        for i in self.V:
            for u in self.M:
                idx = self._ensure_x(i, u)
                lb = None
                ub = None

                if i in fixed_by_loc:
                    if u == fixed_by_loc[i]:
                        lb = 1.0
                        ub = 1.0
                    else:
                        lb = 0.0
                        ub = 0.0
                elif u in fixed_by_fac and fixed_by_fac[u] != i:
                    lb = 0.0
                    ub = 0.0

                if lb is not None:
                    self.cpx.variables.set_lower_bounds(idx, lb)
                if ub is not None:
                    self.cpx.variables.set_upper_bounds(idx, ub)
    # -----------------------------
    # Solve & accessors
    # -----------------------------
    def solve(self):
        # Make sure the model is an LP
        self.cpx.set_problem_type(self.cpx.problem_type.LP)
        self.cpx.solve()
        return self.cpx.solution.get_status()

    def get_y_values(self, positive_only=True):
        """Return dict {(i,u,j,v): value}; if positive_only, filter small values."""
        sol = self.cpx.solution
        vals = {}
        for key, idx in self.col_index.items():
            v = sol.get_values(idx)
            if not positive_only or v > self.eps:
                vals[key] = v
        return vals
    
    def get_x_values(self):
        sol = self.cpx.solution
        vals = {}
        for key, idx in self.x_index.items():
            vals[key] = sol.get_values(idx)
        return vals
    
    
    def get_row_dual(self, row_idx):
        """Safe dual getter for LP."""
        return self.cpx.solution.get_dual_values(row_idx)

    def get_objective_value(self):
        return self.cpx.solution.get_objective_value()

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def check_symmetry_violations(self, y_vals, top_k=20):
        """
        Return a list of the top_k most violated symmetry pairs (absolute difference).
        """
        diffs = []
        seen = set()
        for (i, u, j, v), val in y_vals.items():
            can = self._canonical(i, u, j, v)
            if can in seen:
                continue
            seen.add(can)
            sym = (j, v, i, u)
            val_sym = y_vals.get(sym, 0.0)
            diff = abs(val - val_sym)
            if diff > self.eps:
                diffs.append((diff, (i, u, j, v), val, sym, val_sym))
        diffs.sort(reverse=True, key=lambda x: x[0])
        return diffs[:top_k]

    def model_sizes(self):
        return dict(
            num_vars=self.cpx.variables.get_num(),
            num_rows=self.cpx.linear_constraints.get_num(),
            num_y=len(self.col_index),
            num_x=len(self.x_index),
            num_b=len(self.b_index),
            num_c=len(self.c_index),
            num_c7=len(self.c7_map),
        )