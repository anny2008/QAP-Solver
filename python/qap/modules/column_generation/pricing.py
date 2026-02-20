# pricing.py (Updated)
# Pure CPLEX pricing module for Column Generation (QAP RTL1)
# Uses a generic reduced-cost computation: rc = c - A^T * dual
# Compatible with IncrementalRMP (lazy row creation)
from math import inf

class PricingEngine:
    """
    Pricing engine for the pure-CPLEX column generation solver.
    Computes reduced costs and identifies entering columns.
    Implements a robust rc = c_y - sum_r dual[r] * a[r,y] formula against the
    actual RMP rows present in CPLEX (C1–C4, C7, and two-sided C5).
    """
    def __init__(self, V, M, phi, add_most_negative=True, eps=1e-7, fixed_assignments=None):
        self.V = V
        self.M = M
        self.phi = phi
        self.add_most_negative = add_most_negative
        self.eps = eps
        self.fixed_assignments = dict(fixed_assignments or {})

    def _is_consistent_with_fixed(self, i, u, j, v):
        if not self.fixed_assignments:
            return True
        fi = self.fixed_assignments.get(i)
        if fi is not None and fi != u:
            return False
        fj = self.fixed_assignments.get(j)
        if fj is not None and fj != v:
            return False
        return True

    # ---------------------------
    # Reduced cost for candidate y(i,u,j,v)
    # ---------------------------
    def reduced_cost(self, rmp, i, u, j, v):
        rc = self.phi[(i, u, j, v)] + self.phi[j,v,i,u]
        # C1..C4 (coeff=+1 if row exists)
        rc -= rmp.get_row_dual(rmp.c1_map[(i, j)])
        if i != j:
            rc -= rmp.get_row_dual(rmp.c1_map[(j, i)])
        rc -= rmp.get_row_dual(rmp.c2_map[(u, v)])
        if u != v:
            rc -= rmp.get_row_dual(rmp.c2_map[(v, u)])
        rc -= rmp.get_row_dual(rmp.c3_map[(u, j)])
        if (v, i) != (u, j):
            rc -= rmp.get_row_dual(rmp.c3_map[(v, i)])
        rc -= rmp.get_row_dual(rmp.c4_map[(i, v)])
        if (j, u) != (i, v):
            rc -= rmp.get_row_dual(rmp.c4_map[(j, u)])
        rc -= rmp.get_row_dual(rmp.c5_map[(i,u,j)])
        if (j,v,i) != (i,u,j):
            rc -= rmp.get_row_dual(rmp.c5_map[(j,v,i)])
        rc -= rmp.get_row_dual(rmp.c6_map[(i,u,v)])
        if (j,v,u) != (i,u,v):
            rc -= rmp.get_row_dual(rmp.c6_map[(j,v,u)])
        return rc

    # ---------------------------
    # Main pricing entry point
    # ---------------------------
    def price(self, rmp):
        """
        Return (best_column, best_rc, negative_cols)
        - best_column = (i,u,j,v) or None
        - best_rc = minimum reduced cost
        - negative_cols = list of ((i,u,j,v), rc) if add_most_negative=False
        """
        best_col = None
        best_rc = 0.0
        negative_cols = []
        most_negative_columns = []
        fixed = self.fixed_assignments
        if not fixed:
            for i in self.V:
                for u in self.M:
                    for j in self.V:
                        for v in self.M:
                            # Skip structurally invalid pairs
                            if (i == j and u != v) or (i != j and u == v):
                                continue
                            idx = rmp._canonical(i, u, j, v)
                            if idx in rmp.Omega:
                                continue
                            rc = self.reduced_cost(rmp, i, u, j, v)
                            if rc < -self.eps:
                                negative_cols.append((idx, rc))
                            if rc < best_rc:
                                best_rc = rc
                                best_col = idx
                                most_negative_columns = [(idx, rc)]
                            elif rc < best_rc + self.eps:
                                most_negative_columns.append((idx, rc))
            return best_col, best_rc, negative_cols, most_negative_columns

        for i in self.V:
            fixed_u = fixed.get(i)
            u_list = [fixed_u] if fixed_u is not None else self.M
            for u in u_list:
                for j in self.V:
                    fixed_v = fixed.get(j)
                    v_list = [fixed_v] if fixed_v is not None else self.M
                    for v in v_list:
                        # Skip structurally invalid pairs
                        if (i == j and u != v) or (i != j and u == v):
                            continue
                        idx = rmp._canonical(i, u, j, v)
                        if idx in rmp.Omega:
                            continue
                        rc = self.reduced_cost(rmp, i, u, j, v)
                        if rc < best_rc:
                            best_rc = rc
                            best_col = idx
                            most_negative_columns = [(idx, rc)]
                        elif rc < best_rc + self.eps:
                            most_negative_columns.append((idx, rc))
                        if rc < -self.eps:
                            negative_cols.append((idx, rc))
        return best_col, best_rc, negative_cols, most_negative_columns
