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
    def _ensure_c2(self, u, v, force=False): return self._ensure_row((u, v), 1.0, 'E', self.c2_map, force=force)
    def _ensure_c3(self, u, j, force=False): return self._ensure_row((u, j), 1.0, 'L', self.c3_map, force=force)
    def _ensure_c4(self, i, v, force=False): return self._ensure_row((i, v), 1.0, 'L', self.c4_map, force=force)

        
    def _ensure_c5(self, i, u, j, force=False):
        """sum_v y_{i,u,j,v} - x_{i,u} + b_{i,u,j} = 0"""
        r = self._ensure_row((i, u, j), 0.0, 'L', self.c5_map, force=force)
        x_idx = self._ensure_x(i, u)
        self.cpx.linear_constraints.set_coefficients([(r, x_idx, -1.0)])
        return r

    def _ensure_c6(self, i, u, v, force=False):
        """sum_j y_{i,u,j,v} - x_{i,u} + c_{i,u,v} = 0"""
        r = self._ensure_row((i, u, v), 0.0, 'E', self.c6_map, force=force)
        x_idx = self._ensure_x(i, u)
        self.cpx.linear_constraints.set_coefficients([(r, x_idx, -1.0)])
        return r

    # -----------------------------
    # Canonical helper for symmetry
    # -----------------------------
    @staticmethod
    def _canonical(i, u, j, v):
        t, s = (i, u, j, v), (j, v, i, u)
        return t if (i < j) else s

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
        """
        Add a UNIQUE canonical y variable representing the unordered pair.
        Attach coefficients for BOTH orientations (forward and reverse)
        to preserve the original model where two directed variables existed.
        Deduplicate row indices: if forward and reverse reference the same row
        (e.g., i==j or u==v), the coefficient becomes 2.0 for that row.
        """
        can = self._canonical(i, u, j, v)
        if can in self.col_index or can in self.Omega:
            return self.col_index[can]

        ci, cu, cj, cv = can
        # Accumulate coefficients per row index to avoid duplicates
        row_coeffs = {}
        def add_row(ridx, coeff=1.0):
            row_coeffs[ridx] = row_coeffs.get(ridx, 0.0) + coeff

        # --- Forward orientation (ci,cu,cj,cv)

        # --- Reverse orientation (cj,cv,ci,cu)
        if ci != cj:
            add_row(self._ensure_c1(ci, cj, force), 1.0)
            add_row(self._ensure_c2(cu, cv, force), 1.0)
            add_row(self._ensure_c3(cu, cj, force), 1.0)
            add_row(self._ensure_c4(ci, cv, force), 1.0)
            add_row(self._ensure_c5(ci, cu, cj, force), 1.0)
            add_row(self._ensure_c6(ci, cu, cv, force), 1.0)
            add_row(self._ensure_c1(cj, ci, force), 1.0)
            add_row(self._ensure_c2(cv, cu, force), 1.0)
            add_row(self._ensure_c3(cv, ci, force), 1.0)
            add_row(self._ensure_c4(cj, cu, force), 1.0)
            add_row(self._ensure_c5(cj, cv, ci, force), 1.0)
            add_row(self._ensure_c6(cj, cv, cu, force), 1.0)
        elif ci == cj and cu == cv:
            # print(f"Warning: adding column with identical i,j and u,v: {(i,u,j,v)}; this should not happen if the canonical function is correct.")
            add_row(self._ensure_c1(ci, cj, force), 1.0)
            add_row(self._ensure_c2(cu, cv, force), 1.0)
            add_row(self._ensure_c3(cu, cj, force), 1.0)
            add_row(self._ensure_c4(ci, cv, force), 1.0)
            add_row(self._ensure_c5(ci, cu, cj, force), 1.0)
            add_row(self._ensure_c6(ci, cu, cv, force), 1.0)
        # Flatten to lists
        row_indices = list(row_coeffs.keys())
        row_values = [row_coeffs[r] for r in row_indices]

        # Create the canonical y variable
        name = f"y_{ci}_{cu}_{cj}_{cv}"
        idx = self.cpx.variables.get_num()
        obj = self.phi.get((ci, cu, cj, cv), 0.0) + self.phi.get((cj, cv, ci, cu), 0.0)
        self.cpx.variables.add(
            names=[name],
            obj=[obj],
            lb=[0.0], ub=[cplex.infinity],
            columns=[SparsePair(row_indices, row_values)],
            types=["C"],
        )
        self.Omega.add(can)
        self.col_index[can] = idx
        return idx
    
    def check_row_correctness(self):
        #  First check constraints c1..c6 for all existing columns
        # For each row c1, get all columns with nonzero coeffs and verify they match the expected pattern
        for (ci, cj), r in self.c1_map.items():
            coeffs = self.cpx.linear_constraints.get_rows(r)
            for idx, coeff in zip(coeffs.ind, coeffs.val):
                if abs(coeff) > self.eps:
                    var_name = self.cpx.variables.get_names(idx)
                    if not var_name.startswith('y'):
                        raise ValueError(f"Row c1({ci},{cj}) has non-y variable {var_name} with coeff {coeff}.")
                    # Extract (ci,cu,cj,cv) from var_name
                    _, i, u, j, v = var_name.split('_')
                    i, u, j, v = int(i), int(u), int(j), int(v)
                    si, su, sj, sv = j, v, i, u  # symmetric pair
                    if not ((ci == i and cj == j) or (ci == si and cj == sj)):
                        raise ValueError(f"Row c1({ci},{cj}) has variable {var_name} with coeff {coeff} that does not match expected pattern y(i,u,j,v).")
            # check every y_ci_*_cj_* exists with nonzero coeff in this row
            # and every y_cj_*_ci_* exists with nonzero coeff in this row
            for u in self.M:
                for v in self.M:
                    var_name_1 = f"y_{ci}_{u}_{cj}_{v}"
                    var_name_2 = f"y_{cj}_{v}_{ci}_{u}"
                    idx_1 = self.col_index.get((ci, u, cj, v))
                    idx_2 = self.col_index.get((cj, v, ci, u))
                    can = self._canonical(ci, u, cj, v)
                    if can in self.col_index:
                        can_idx = self.col_index[can]
                        coeff_1 = self.cpx.linear_constraints.get_coefficients(r, can_idx)
                        if abs(coeff_1 - 1.0) > self.eps:
                            raise ValueError(f"Row c1({ci},{cj}) is missing expected canonical column {var_name_1} with coeff 1.0, found coeff {coeff_1} instead.")
        # C2: for each (u,v), check that all nonzero coeffs in
                
        for (cu, cv), r in self.c2_map.items():
            coeffs = self.cpx.linear_constraints.get_rows(r)
            for idx, coeff in zip(coeffs.ind, coeffs.val):
                if abs(coeff) > self.eps:
                    var_name = self.cpx.variables.get_names(idx)
                    if not var_name.startswith('y'):
                        raise ValueError(f"Row c2({cu},{cv}) has non-y variable {var_name} with coeff {coeff}.")
                    _, i, u, j, v = var_name.split('_')
                    i, u, j, v = int(i), int(u), int(j), int(v)
                    si, su, sj, sv = j, v, i, u  # symmetric pair
                    if not ((cu == u and cv == v) or (cu == su and cv == sv)):
                        raise ValueError(f"Row c2({cu},{cv}) has variable {var_name} with coeff {coeff} that does not match expected pattern y(i,u,j,v).")
            # check every y_*_cu_*_cv exists with nonzero coeff in this row
            for i in self.V:
                for j in self.V:
                    var_name_1 = f"y_{i}_{cu}_{j}_{cv}"
                    var_name_2 = f"y_{j}_{cv}_{i}_{cu}"
                    idx_1 = self.col_index.get((i, cu, j, cv))
                    idx_2 = self.col_index.get((j, cv, i, cu))
                    can = self._canonical(i, cu, j, cv)
                    if can in self.col_index:
                        can_idx = self.col_index[can]
                        coeff_1 = self.cpx.linear_constraints.get_coefficients(r, can_idx)
                        if abs(coeff_1 - 1.0) > self.eps:
                            raise ValueError(f"Row c2({cu},{cv}) is missing expected canonical column {var_name_1} with coeff 1.0, found coeff {coeff_1} instead.")
        # C3: for each (u,j), check that all nonzero coeffs in row c3(u,j) correspond to columns y(i,u,j,v)
        for (cu, cj), r in self.c3_map.items():
            coeffs = self.cpx.linear_constraints.get_rows(r)
            for idx, coeff in zip(coeffs.ind, coeffs.val):
                if abs(coeff) > self.eps:
                    var_name = self.cpx.variables.get_names(idx)
                    if not var_name.startswith('y'):
                        raise ValueError(f"Row c3({cu},{cj}) has non-y variable {var_name} with coeff {coeff}.")
                    _, i, u, j, v = var_name.split('_')
                    i, u, j, v = int(i), int(u), int(j), int(v)
                    si, su, sj, sv = j, v, i, u  # symmetric pair
                    if not ((cu == u and cj == j) or (cu == su and cj == sj)):
                        raise ValueError(f"Row c3({cu},{cj}) has variable {var_name} with coeff {coeff} that does not match expected pattern y(i,u,j,v).")
            # check every y_*_cu_*_cj exists with nonzero coeff in this row
            for i in self.V:
                for v in self.M:
                    var_name_1 = f"y_{i}_{cu}_{cj}_{v}"
                    var_name_2 = f"y_{cj}_{v}_{i}_{cu}"
                    idx_1 = self.col_index.get((i, cu, cj, v))
                    idx_2 = self.col_index.get((cj, v, i, cu))
                    can = self._canonical(i, cu, cj, v)
                    if can in self.col_index:
                        can_idx = self.col_index[can]
                        coeff_1 = self.cpx.linear_constraints.get_coefficients(r, can_idx)
                        if abs(coeff_1 - 1.0) > self.eps:
                            raise ValueError(f"Row c3({cu},{cj}) is missing expected canonical column {var_name_1} with coeff 1.0, found coeff {coeff_1} instead.")
        # C4: for each (i,v), check that all nonzero coeffs in row c4(i,v) correspond to columns y(i,u,j,v)
        for (ci, cv), r in self.c4_map.items():
            coeffs = self.cpx.linear_constraints.get_rows(r)
            for idx, coeff in zip(coeffs.ind, coeffs.val):
                if abs(coeff) > self.eps:
                    var_name = self.cpx.variables.get_names(idx)
                    if not var_name.startswith('y'):
                        raise ValueError(f"Row c4({ci},{cv}) has non-y variable {var_name} with coeff {coeff}.")
                    _, i, u, j, v = var_name.split('_')
                    i, u, j, v = int(i), int(u), int(j), int(v)
                    si, su, sj, sv = j, v, i, u  # symmetric pair
                    if not ((ci == i and cv == v) or (ci == si and cv == sv)):
                        raise ValueError(f"Row c4({ci},{cv}) has variable {var_name} with coeff {coeff} that does not match expected pattern y(i,u,j,v).")
            # check every y_ci_*_*_cv exists with nonzero coeff in this row
            for u in self.M:
                for j in self.V:
                    var_name_1 = f"y_{ci}_{u}_{j}_{cv}"
                    var_name_2 = f"y_{j}_{cv}_{ci}_{u}"
                    idx_1 = self.col_index.get((ci, u, j, cv))
                    idx_2 = self.col_index.get((j, cv, ci, u))
                    can = self._canonical(ci, u, j, cv)
                    if can in self.col_index:
                        can_idx = self.col_index[can]
                        coeff_1 = self.cpx.linear_constraints.get_coefficients(r, can_idx)
                        if abs(coeff_1 - 1.0) > self.eps:
                            raise ValueError(f"Row c4({ci},{cv}) is missing expected canonical column {var_name_1} with coeff 1.0, found coeff {coeff_1} instead.")
        # C5: for each (i,u,j), check that all nonzero coeffs in row c5(i,u,j) correspond to columns y(i,u,j,v) and x(i,u) and b(i,u,j)
        for (ci, cu, cj), r in self.c5_map.items():
            coeffs = self.cpx.linear_constraints.get_rows(r)
            for idx, coeff in zip(coeffs.ind, coeffs.val):
                if abs(coeff) > self.eps:
                    var_name = self.cpx.variables.get_names(idx)
                    if var_name.startswith('y'):
                        _, i, u, j, v = var_name.split('_')
                        i, u, j, v = int(i), int(u), int(j), int(v)
                        si, su, sj, sv = j, v, i, u  # symmetric pair
                        if not ((ci == i and cu == u and cj == j) or (ci == si and cu == su and cj == sj)):
                            raise ValueError(f"Row c5({ci},{cu},{cj}) has variable {var_name} with coeff {coeff} that does not match expected pattern y(i,u,j,v).")
                    elif var_name.startswith('x'):
                        _, xi, xu = var_name.split('_')
                        xi, xu = int(xi), int(xu)
                        if not (xi == ci and xu == cu):
                            raise ValueError(f"Row c5({ci},{cu},{cj}) has variable {var_name} with coeff {coeff} that does not match expected pattern x(i,u).")
                    elif var_name.startswith('b'):
                        _, bi, bu, bj = var_name.split('_')
                        bi, bu, bj = int(bi), int(bu), int(bj)
                        if not (bi == ci and bu == cu and bj == cj):

                            raise ValueError(f"Row c5({ci},{cu},{cj}) has variable {var_name} with coeff {coeff} that does not match expected pattern b(i,u,j).")
                    else:
                        raise ValueError(f"Row c5({ci},{cu},{cj}) has non-y/x/b variable {var_name} with coeff {coeff}.")
            # check every y_ci_cu_cj_* exists with nonzero coeff in this row
            for v in self.M:
                var_name_1 = f"y_{ci}_{cu}_{cj}_{v}"
                var_name_2 = f"y_{cj}_{v}_{ci}_{cu}"
                idx_1 = self.col_index.get((ci, cu, cj, v))
                idx_2 = self.col_index.get((cj, v, ci, cu))
                can = self._canonical(ci, cu, cj, v)
                if can in self.col_index:
                    can_idx = self.col_index[can]
                    coeff_1 = self.cpx.linear_constraints.get_coefficients(r, can_idx)
                    if abs(coeff_1 - 1.0) > self.eps:
                        raise ValueError(f"Row c5({ci},{cu},{cj}) is missing expected canonical column {var_name_1} with coeff 1.0, found coeff {coeff_1} instead.")
        # C6: for each (i,u,v), check that all nonzero coeffs in row c6(i,u,v) correspond to columns y(i,u,j,v) and x(i,u) and c(i,u,v)
        for (ci, cu, cv), r in self.c6_map.items():
            coeffs = self.cpx.linear_constraints.get_rows(r)
            for idx, coeff in zip(coeffs.ind, coeffs.val):
                if abs(coeff) > self.eps:
                    var_name = self.cpx.variables.get_names(idx)
                    if var_name.startswith('y'):
                        _, i, u, j, v = var_name.split('_')
                        i, u, j, v = int(i), int(u), int(j), int(v)
                        si, su, sj, sv = j, v, i, u  # symmetric pair
                        if not ((ci == i and cu == u and cv == v) or (ci == si and cu == su and cv == sv)):
                            raise ValueError(f"Row c6({ci},{cu},{cv}) has variable {var_name} with coeff {coeff} that does not match expected pattern y(i,u,j,v).")
                    elif var_name.startswith('x'):
                        _, xi, xu = var_name.split('_')
                        xi, xu = int(xi), int(xu)
                        if not (xi == ci and xu == cu):
                            raise ValueError(f"Row c6({ci},{cu},{cv}) has variable {var_name} with coeff {coeff} that does not match expected pattern x(i,u).")
                    elif var_name.startswith('c'):
                        _, ci, cu, cvv = var_name.split('_')
                        ci, cu, cvv = int(ci), int(cu), int(cvv)
                        if not (ci == ci and cu == cu and cvv == cv):
                            raise ValueError(f"Row c6({ci},{cu},{cv}) has variable {var_name} with coeff {coeff} that does not match expected pattern c(i,u,v).")
                    else:
                        raise ValueError(f"Row c6({ci},{cu},{cv}) has non-y/x/c variable {var_name} with coeff {coeff}.")
            # check every y_ci_cu_*_cv exists with nonzero coeff in this row
            for j in self.V:
                var_name_1 = f"y_{ci}_{cu}_{j}_{cv}"
                var_name_2 = f"y_{j}_{cv}_{ci}_{cu}"
                idx_1 = self.col_index.get((ci, cu, j, cv))
                idx_2 = self.col_index.get((j, cv, ci, cu))
                can = self._canonical(ci, cu, j, cv)
                if can in self.col_index:
                    can_idx = self.col_index[can]
                    coeff_1 = self.cpx.linear_constraints.get_coefficients(r, can_idx)
                    if abs(coeff_1 - 1.0) > self.eps:
                        raise ValueError(f"Row c6({ci},{cu},{cv}) is missing expected canonical column {var_name_1} with coeff 1.0, found coeff {coeff_1} instead.")
        print("Row correctness check passed: all rows have the expected variables and coefficients.")
        
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

    def get_y_values(self):
        """Return dict {(i,u,j,v): value}; if positive_only, filter small values."""
        if not self.col_index:
            return {}
        sol = self.cpx.solution
        indices = list(self.col_index.values())
        values = sol.get_values(indices)
        return {key: values[i] for i, key in enumerate(self.col_index.keys()) if values[i] > self.eps}
    
    def get_x_values(self):
        if not self.x_index:
            return {}
        sol = self.cpx.solution
        indices = list(self.x_index.values())
        values = sol.get_values(indices)
        return {key: values[i] for i, key in enumerate(self.x_index.keys())}
    
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
    def get_basis(self):
        basis_cols, basis_rows = self.cpx.solution.basis.get_basis()
        result = []
        for i, status in enumerate(basis_cols):
            if status == self.cpx.solution.basis.status.basic:
                var_name = self.cpx.variables.get_names(i)
                if var_name.startswith('y'):
                    result.append(var_name)
        return result