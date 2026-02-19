# separation.py (Updated)
# Pure CPLEX separation module for Column Generation (QAP RTL1)
# Compatible with IncrementalRMP and PricingEngine
# Implements symmetry separation (hard) and two-sided Big-M balance separation with top-K policy

class SeparationEngine:
    """
    Separation of:
      - Symmetry constraints (C7)
      - Big-M balance constraints (C5) if enabled (two-sided form)
    This module performs no solving; it only inserts new rows and
    returns lists of newly added cuts and columns.
    """
    def __init__(self, V, M, eps=1e-7, top_k=1):
        self.V = V
        self.M = M
        self.eps = eps
        self.top_k = top_k  # budget per iteration for balance cuts

    # ---------------------------
    # Symmetry separation (C7)
    # ---------------------------
    def separate_symmetry(self, rmp, y_vals):
        """
        Given current y-values, enforce y[i,u,j,v] == y[j,v,i,u].
        - If the partner column is missing, add it (and, if needed, add a C7 row).
        - If both exist but differ beyond eps, add the equality row.
        Returns: (new_cuts, new_cols)
        """
        new_cuts = []
        new_cols = []
        for (i, u, j, v), val in y_vals.items():
            sym = (j, v, i, u)
            can = rmp._canonical(i, u, j, v)
            if sym not in rmp.Omega:
                # Case 1: partner column missing -> add it and the symmetry row
                rmp.add_column(j, v, i, u)
                new_cols.append(sym)
                if can not in rmp.c7_map:
                    rmp.add_symmetry_cut(i, u, j, v)
                    new_cuts.append(can)
                else:
                    raise ValueError(f"Symmetry cut already exists for {can} but partner column was missing.")
            else:
                # Case 2: both exist but differ -> add equality if missing
                val_sym = y_vals.get(sym, 0.0)
                if abs(val - val_sym) > self.eps:
                    if can not in rmp.c7_map:
                        rmp.add_symmetry_cut(i, u, j, v)
                        new_cuts.append(can)
        return new_cuts, new_cols
