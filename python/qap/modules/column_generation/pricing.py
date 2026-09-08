from math import inf

class PricingEngine:
    """
    Pricing engine for the pure-CPLEX column generation solver.
    Computes reduced costs and identifies entering columns.
    Implements rc = c_y - sum_r dual[r] * a[r,y] for RMP rows.
    """

    def __init__(self, V, M, phi, add_most_negative=True, eps=1e-7, fixed_assignments=None):
        self.V = V
        self.M = M
        self.phi = phi
        self.add_most_negative = add_most_negative
        self.eps = eps
        self.fixed_assignments = dict(fixed_assignments or {})

        # Build caches only once
        self._caches_built = False

    # ----------------------------------------------------------------------
    # Build all expensive caches ONCE (canonical, symmetric phi, feasibility)
    # ----------------------------------------------------------------------
    def _build_caches(self, rmp):
        if self._caches_built:
            return

        V = self.V
        M = self.M
        phi = self.phi

        # Precompute symmetric phi term: phi(i,u,j,v) + phi(j,v,i,u)
        phi_sym = {}
        canonical = {}
        valid = {}

        for i in V:
            for u in M:
                for j in V:
                    for v in M:

                        # Canonical index (expensive, cache it)
                        idx = rmp._canonical(i, u, j, v)
                        canonical[(i, u, j, v)] = idx

                        # Structural validity
                        # Invalid if (i == j and u != v) or (i != j and u == v)
                        invalid = (i == j and u != v) or (i != j and u == v)
                        valid[idx] = not invalid

                        # φ symmetry
                        phi_sym[(i, u, j, v)] = phi[(i, u, j, v)] + phi[(j, v, i, u)]

        self.phi_sym = phi_sym
        self.canonical = canonical
        self.valid = valid
        self._caches_built = True

    # ---------------------------
    # Compute reduced cost using cached duals
    # ---------------------------
    def reduced_cost(self, rmp, i, u, j, v):
        # This function is still used but the optimized version bypasses it.
        rc = self.phi[(i, u, j, v)] + self.phi[(j, v, i, u)]

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
        rc -= rmp.get_row_dual(rmp.c5_map[(i, u, j)])
        if (j, v, i) != (i, u, j):
            rc -= rmp.get_row_dual(rmp.c5_map[(j, v, i)])
        rc -= rmp.get_row_dual(rmp.c6_map[(i, u, v)])
        if (j, v, u) != (i, u, v):
            rc -= rmp.get_row_dual(rmp.c6_map[(j, v, u)])
        return rc

    # ---------------------------
    # Main pricing entry point
    # ---------------------------
    def price(self, rmp):
        """
        Returns:
            best_col: (i,u,j,v) or None
            best_rc: minimum reduced cost
            negative_cols: list of (idx, rc)
            most_negative_columns: list of (idx, rc)
        """
        # Build expensive static caches on first use
        self._build_caches(rmp)

        # Preload expensive dual row queries ONCE per call
        rmp_dual_c1 = {k: rmp.get_row_dual(v) for k, v in rmp.c1_map.items()}
        rmp_dual_c2 = {k: rmp.get_row_dual(v) for k, v in rmp.c2_map.items()}
        rmp_dual_c3 = {k: rmp.get_row_dual(v) for k, v in rmp.c3_map.items()}
        rmp_dual_c4 = {k: rmp.get_row_dual(v) for k, v in rmp.c4_map.items()}
        rmp_dual_c5 = {k: rmp.get_row_dual(v) for k, v in rmp.c5_map.items()}
        rmp_dual_c6 = {k: rmp.get_row_dual(v) for k, v in rmp.c6_map.items()}

        # Localize variables for speed
        V = self.V
        M = self.M
        fixed = self.fixed_assignments
        Omega = rmp.Omega
        canonical = self.canonical
        valid = self.valid
        phi_sym = self.phi_sym

        best_col = None
        best_rc = 0.0
        negative_cols = []
        most_negative_cols = []

        eps = self.eps

        for i in V:
            u_list = [fixed[i]] if i in fixed else M
            for u in u_list:
                for j in V:
                    v_list = [fixed[j]] if j in fixed else M
                    for v in v_list:

                        idx = canonical[(i, u, j, v)]

                        # skip structurally invalid
                        if not valid[idx]:
                            continue

                        # skip existing columns
                        if idx in Omega:
                            continue

                        # -------------------------
                        # Fast reduced cost
                        # -------------------------
                        rc = phi_sym[(i, u, j, v)]

                        rc -= rmp_dual_c1[(i, j)]
                        if i != j:
                            rc -= rmp_dual_c1[(j, i)]

                        rc -= rmp_dual_c2[(u, v)]
                        if u != v:
                            rc -= rmp_dual_c2[(v, u)]

                        rc -= rmp_dual_c3[(u, j)]
                        if (v, i) != (u, j):
                            rc -= rmp_dual_c3[(v, i)]

                        rc -= rmp_dual_c4[(i, v)]
                        if (j, u) != (i, v):
                            rc -= rmp_dual_c4[(j, u)]

                        rc -= rmp_dual_c5[(i, u, j)]
                        if (j, v, i) != (i, u, j):
                            rc -= rmp_dual_c5[(j, v, i)]

                        rc -= rmp_dual_c6[(i, u, v)]
                        if (j, v, u) != (i, u, v):
                            rc -= rmp_dual_c6[(j, v, u)]

                        # -------------------------
                        # Track best and negatives
                        # -------------------------
                        if rc < best_rc:
                            best_rc = rc
                            best_col = idx
                            most_negative_cols = [(idx, rc)]
                        elif rc < best_rc + eps:
                            most_negative_cols.append((idx, rc))

                        if rc < -eps:
                            negative_cols.append((idx, rc))

        return best_col, best_rc, negative_cols, most_negative_cols