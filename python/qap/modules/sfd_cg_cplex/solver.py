"""
SFDCG CPLEX Solver Module

Relaxation-based Tightened Linear (SFDCG) formulation solved with CPLEX.
Supports binary variables, relaxed 0-1 variables, and warm-start solutions.

Reference: Based on solve_QAP_SFDCG_cplex from qap_new_formulation.py
"""

from __future__ import annotations
import time
import numpy as np

# Fix numpy 2.0 compatibility with docplex
np.float_ = np.float64
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from docplex.mp.model import Model
from cplex.exceptions import CplexSolverError

from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart
from typing import List, Dict, Tuple, Optional
import math
import random
import numpy as np

PathDict = Dict[str, object]  # {"machines": List[int], "arcs": List[Tuple[int,int]], "weight": float}

def decompose_flow_into_paths(
    F: np.ndarray,
    *,
    eps: float = 1e-9,
    prefer_sources: bool = True,
    strategy: str = "max_arc",          # "max_arc", "max_out_sum", or "max_out_degree"
    allow_revisit: bool = False,        # if False, path is simple (no repeated node)
    min_path_len: int = 2,              # minimum number of nodes in a path (≥2 -> at least one arc)
    max_paths: Optional[int] = None,
    seed: Optional[int] = None,
    return_zero_based: bool = False,    # if True, return machine ids in 0-based indexing
    aggregate_identical: bool = True,   # NEW: aggregate identical paths by summing weights
    agg_tol: float = 1e-12              # NEW: small tolerance when summing weights
) -> List[PathDict]:
    """
    Decompose a nonnegative flow matrix F into a list of path subgraphs (Variant A friendly).
    Each path k has:
        - "machines": sequence [u1, u2, ..., um],
        - "arcs":     directed arcs [(u1,u2), (u2,u3), ..., (u_{m-1}, u_m)],
        - "weight":   F_k (uniform weight subtracted from all arcs on the path).

    The algorithm repeatedly extracts a path, subtracts the min residual flow on its arcs,
    and continues until no positive flow remains (within eps).

    Parameters
    ----------
    F : np.ndarray
        Square (n x n) nonnegative flow matrix. Diagonal allowed but ignored.
    eps : float
        Tolerance for positive flows (values ≤ eps treated as 0).
    prefer_sources : bool
        If True, start paths from nodes with zero in-degree (w.r.t. positive residual arcs) when possible.
    strategy : str
        Heuristic to extend the path:
            - "max_arc": pick outgoing arc (u->v) with max residual R[u,v].
            - "max_out_sum": choose next node v with largest total outgoing residual from v.
            - "max_out_degree": choose v with the largest number of positive outgoing arcs.
    allow_revisit : bool
        If False, no repeated nodes in the current path (simple path). If True, cycles allowed.
    min_path_len : int
        Minimum number of nodes (>=2). If not met, fallback to best single arc.
    max_paths : Optional[int]
        If provided, stop after generating at most this many paths.
    seed : Optional[int]
        Random seed (tie-breaking only).
    return_zero_based : bool
        If True, return 0-based machine/arcs. Otherwise, 1-based (paper style).
    aggregate_identical : bool
        If True, merge identical machine sequences by summing weights before returning.
    agg_tol : float
        Very small tolerance when summing weights for aggregation.

    Returns
    -------
    List[PathDict]
        A list of dictionaries, each:
            {
              "machines": [u1, u2, ..., um],   # 1-based by default (or 0-based if return_zero_based=True)
              "arcs":     [(u1,u2), ..., (u_{m-1},u_m)],
              "weight":   float
            }

    Notes
    -----
    - Greedy heuristic: guarantees F equals the sum over all path-arc weights (within eps),
      but number/length of paths depends on the strategy and topology.
    - For column generation (Variant A), each returned path defines the machine order (U_k).
    """

    rng = random.Random(seed)
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix.")
    if np.any(F < -eps):
        raise ValueError("F must be nonnegative (within tolerance).")

    # Work on a residual copy; ignore diagonal arcs explicitly.
    R = F.copy()
    np.fill_diagonal(R, 0.0)
    raw_paths: List[PathDict] = []

    def out_neighbors(u: int) -> List[int]:
        return [v for v in range(n) if v != u and R[u, v] > eps]

    def total_out(u: int) -> float:
        return float(np.sum(R[u, :] * (R[u, :] > eps)))

    def out_degree(u: int) -> int:
        return int(np.count_nonzero(R[u, :] > eps))

    def residual_has_positive() -> bool:
        return bool(np.any(R > eps))

    def select_start_node() -> int:
        if prefer_sources:
            indeg_pos = (R > eps).sum(axis=0)
            outdeg_pos = (R > eps).sum(axis=1)
            candidates = [u for u in range(n) if indeg_pos[u] == 0 and outdeg_pos[u] > 0]
            if candidates:
                return max(candidates, key=lambda u: total_out(u))
        # Fallback: node with largest (outflow - inflow), but must have outflow
        out_sums = (R * (R > eps)).sum(axis=1)
        in_sums  = (R * (R > eps)).sum(axis=0)
        score = out_sums - in_sums
        candidates = [u for u in range(n) if out_sums[u] > eps]
        if not candidates:
            return -1
        return max(candidates, key=lambda u: (score[u], out_sums[u]))

    def choose_next(u: int, visited: set[int]) -> Optional[int]:
        nbrs = [v for v in out_neighbors(u) if (allow_revisit or v not in visited)]
        if not nbrs:
            return None
        if strategy == "max_arc":
            best_val = -1.0
            best_list = []
            for v in nbrs:
                val = R[u, v]
                if val > best_val + eps:
                    best_val = val
                    best_list = [v]
                elif abs(val - best_val) <= eps:
                    best_list.append(v)
            return rng.choice(best_list)
        elif strategy == "max_out_sum":
            best_val, best_list = -1.0, []
            for v in nbrs:
                val = total_out(v)
                if val > best_val + eps:
                    best_val, best_list = val, [v]
                elif abs(val - best_val) <= eps:
                    best_list.append(v)
            return rng.choice(best_list)
        elif strategy == "max_out_degree":
            best_val, best_list = -1, []
            for v in nbrs:
                val = out_degree(v)
                if val > best_val:
                    best_val, best_list = val, [v]
                elif val == best_val:
                    best_list.append(v)
            return rng.choice(best_list)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def build_path() -> Optional[Tuple[List[int], List[Tuple[int, int]]]]:
        start = select_start_node()
        if start < 0:
            return None  # no positive outgoing arcs remain
        path_nodes = [start]
        visited = {start}
        while True:
            u = path_nodes[-1]
            v = choose_next(u, visited)
            if v is None:
                break
            path_nodes.append(v)
            if not allow_revisit:
                if v in visited:
                    break
                visited.add(v)

        # Ensure minimum path length (>=2 nodes -> at least one arc)
        if len(path_nodes) < min_path_len:
            nbrs = out_neighbors(start)
            if not nbrs:
                return None
            v_best = max(nbrs, key=lambda vv: R[start, vv])
            path_nodes = [start, v_best]

        arcs = [(path_nodes[t], path_nodes[t + 1]) for t in range(len(path_nodes) - 1)]
        return path_nodes, arcs

    # -------- Main decomposition loop --------
    num_paths = 0
    while residual_has_positive():
        if max_paths is not None and num_paths >= max_paths:
            break
        built = build_path()
        if built is None:
            # No path could be built but residual positive remains -> take largest single arc.
            u, v = np.unravel_index(np.argmax(R), R.shape)
            if R[u, v] <= eps:
                break
            path_nodes = [u, v]
            arcs = [(u, v)]
        else:
            path_nodes, arcs = built

        # Path weight = minimum residual on selected arcs
        w = min(R[u, v] for (u, v) in arcs)
        if w <= eps:
            # Zero out and continue
            for (u, v) in arcs:
                R[u, v] = 0.0
            continue

        # Subtract residual and record path
        for (u, v) in arcs:
            R[u, v] -= w
            if R[u, v] < eps:
                R[u, v] = 0.0

        num_paths += 1

        # Prepare output indices (1-based by default)
        if return_zero_based:
            out_nodes = [int(u) for u in path_nodes]
            out_arcs  = [(int(u), int(v)) for (u, v) in arcs]
        else:
            out_nodes = [int(u) + 1 for u in path_nodes]
            out_arcs  = [(int(u) + 1, int(v) + 1) for (u, v) in arcs]

        raw_paths.append({
            "machines": out_nodes,
            "arcs": out_arcs,
            "weight": float(w)
        })

    # -------- Aggregation step (NEW) --------
    if aggregate_identical and raw_paths:
        from collections import OrderedDict
        aggregated: "OrderedDict[Tuple[int, ...], Dict[str, object]]" = OrderedDict()
        for p in raw_paths:
            key = tuple(p["machines"])  # identical machine order => same arcs
            if key not in aggregated:
                # store first occurrence to preserve order; copy arcs as they’re implied by machines
                aggregated[key] = {
                    "machines": p["machines"],
                    "arcs":     p["arcs"],
                    "weight":   float(p["weight"])
                }
            else:
                aggregated[key]["weight"] = float(aggregated[key]["weight"]) + float(p["weight"])
                # Clean very small noise
                if abs(aggregated[key]["weight"]) < agg_tol:
                    aggregated[key]["weight"] = 0.0

        # Drop zero-weight leftovers (paranoia, usually none)
        paths = [dict(machines=v["machines"], arcs=v["arcs"], weight=float(v["weight"]))
                 for v in aggregated.values() if abs(float(v["weight"])) > agg_tol]
        return paths

    # If no aggregation requested
    return raw_paths


def check_decomposition_reconstruction(
    F: np.ndarray,
    paths: List[PathDict],
    *,
    eps: float = 1e-7,
    zero_based_input: bool = False
) -> Tuple[bool, float]:
    """
    Reconstruct F_hat from the path list and check ||F - F_hat||_1.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    R = np.zeros_like(F, dtype=float)
    for p in paths:
        w = float(p["weight"])
        for (u, v) in p["arcs"]:
            uu, vv = (u, v) if zero_based_input else (u - 1, v - 1)
            if not (0 <= uu < n and 0 <= vv < n):
                raise ValueError("Arc index out of bounds in path.")
            R[uu, vv] += w
    err1 = float(np.sum(np.abs(F - R)))
    return (err1 <= eps, err1)

class SFDCGCPLEXSolver:
    """SFDCG formulation solver using CPLEX."""

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
        self.solver_name = config.get("solver", "SFDCG_cplex")
        self.formulation = config.get("formulation", "SFDCG")
        self.is_relax = config.get("is_relax", False)
        self.time_limit = config.get("time_limit", 120)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.preprocessing_symmetry = config.get("preprocessing_symmetry", 5)

        # Validate config
        assert self.formulation == "SFDCG", "SFDCG solver requires formulation='SFDCG'"

    @classmethod
    def from_config_file(cls, config_path: str) -> "SFDCGCPLEXSolver":
        """
        Create solver from config file.

        Args:
            config_path (str): Path to config JSON file

        Returns:
            SFDCGCPLEXSolver: Initialized solver
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    def solve(
        self,
        problem: "Problem",
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[Tuple[int, int], float]] = None,
        paths_agg: Optional[List["PathDict"]] = None,
    ) -> "Solution":
        """
        Column generation (Variant A: per-path, per-position arc columns) using Python CPLEX.
        - RMP: x (assignment) + lambda columns (added dynamically)
        - Constraints:
            * Assignment (rows by location and by machine)
            * For each path k, position t:
                - Convexity: sum_{i!=j} lambda_{k,t,i,j} = 1
                - Tail-link: sum_{j!=i} lambda_{k,t,i,j} - x_{i,u_t^k} = 0
                - Head-link: sum_{i!=j} lambda_{k,t,i,j} - x_{j,u_{t+1}^k} = 0
        - Pricing:
            rc(k,t,i,j) = F_k * d_{ij} - mu_{k,t} - phi_{k,t,i} - psi_{k,t,j}

        Args:
            problem: QAP instance with .n, .D (and optionally .F for auto-decomposition)
            fixed_variables: list of (i,u) to fix x_{iu}=1; all other x in row i and col u fixed to 0
            warmstart: dict {(i,u): value} to seed x
            paths_agg: precomputed list of paths (each with fields "machines", "arcs", "weight")

        Returns:
            Solution: filled with assignment (if integral or rounded), objective, LB, and runtime.
        """
        import time
        import math
        import numpy as np
        try:
            import cplex
            from cplex import SparsePair
        except Exception as e:
            raise RuntimeError(
                "CPLEX Python API not available. Please install and license IBM ILOG CPLEX."
            ) from e

        # -------------------------
        # Parameters you may tune
        # -------------------------
        rc_tol = 1e-8                            # reduced-cost tolerance for adding columns
        max_cg_iterations = 100                  # safety cap on CG iterations
        max_new_columns_per_it = 200             # cap per iteration to control LP growth
        price_top_per_block = None               # if not None, add up to K best columns per (k,t); else global bests
        do_final_mip_polish = True               # set x to binary and solve a final MIP over generated columns
        mip_time_limit = 300.0                   # seconds for final MIP
        display_cplex_output = False             # set True to see the solver's logs

        n = int(problem.n)
        D = np.asarray(problem.D, dtype=float)
        assert D.shape == (n, n), "Distance matrix D must be n x n."

        # -------------------------
        # Decomposition (paths_agg)
        # -------------------------
        if paths_agg is None:
            # If you have flow matrix F in the problem, you can auto-decompose here:
            if not hasattr(problem, "F"):
                raise ValueError("paths_agg is None and problem.F is not available for decomposition.")
            paths_agg = decompose_flow_into_paths(problem.F, strategy="max_arc", seed=0)

        # Preprocess paths: build for each k the machine order U_k and flow weight F_k
        # machines are assumed 1-based; convert to 0-based internally.
        K_paths = []
        for p in paths_agg:
            U = [u - 1 for u in p["machines"]]
            if len(U) < 2:
                continue
            L = len(U) - 1
            Fk = float(p["weight"])  # weight (flow) associated with this path
            K_paths.append({"U": U, "L": L, "F": Fk})
        if not K_paths:
            # Degenerate case: no flow => assignment cost is 0, any permutation works
            assignment = list(range(n))  # i <- u
            return Solution(
                instance=getattr(self, "instance", None),
                solver=getattr(self, "solver_name", "CPLEX-CG"),
                assignment=assignment,
                objective=0.0,
                lower_bound=0.0,
                time=0.0,
            )

        # -------------------------
        # Build the RMP (LP)
        # -------------------------
        t0 = time.time()
        cpx = cplex.Cplex()
        cpx.set_log_stream(None) if not display_cplex_output else None
        cpx.set_results_stream(None) if not display_cplex_output else None
        cpx.objective.set_sense(cpx.objective.sense.minimize)

        # --- Helper maps
        # x variable indices
        x_index = {}  # (i,u) -> var index
        # present lambda columns
        lam_index = {}  # (k, t, i, j) -> var index

        # --- Add x variables (continuous in CG; later we can switch to binary for MIP polish)
        # bounds and objective
        x_names, x_obj, x_lb, x_ub, x_types = [], [], [], [], []
        for i in range(n):
            for u in range(n):
                name = f"x_{i}_{u}"
                x_names.append(name)
                x_obj.append(0.0)
                x_lb.append(0.0)
                x_ub.append(1.0)
                x_types.append(cpx.variables.type.continuous)
        base = cpx.variables.get_num()
        cpx.variables.add(obj=x_obj, lb=x_lb, ub=x_ub, types=x_types, names=x_names)
        # Map indices
        for pos, name in enumerate(x_names, start=base):
            _, i, u = name.split("_")
            x_index[(int(i), int(u))] = pos

        # Apply fixed variables (if any)
        # If x_{i,u} fixed to 1, set it and force others in row i and col u to 0
        fixed_variables = fixed_variables or []
        fixed_set = set(fixed_variables)
        for (i, u) in fixed_set:
            vidx = x_index[(i, u)]
            cpx.variables.set_lower_bounds(vidx, 1.0)
            cpx.variables.set_upper_bounds(vidx, 1.0)
            # zero-out others in row i
            for uu in range(n):
                if uu == u: 
                    continue
                cpx.variables.set_upper_bounds(x_index[(i, uu)], 0.0)
            # zero-out others in column u
            for ii in range(n):
                if ii == i: 
                    continue
                cpx.variables.set_upper_bounds(x_index[(ii, u)], 0.0)

        # --- Constraints
        # We'll add all rows now so columns can be added with addcols later.
        row_names = []
        row_senses = []
        row_rhs = []
        row_exprs = []  # list of SparsePair

        # 1) Assignment by location: sum_u x_{i,u} = 1    (n rows)
        row_assign_loc = {}  # i -> row index
        for i in range(n):
            idxs = [x_index[(i, u)] for u in range(n)]
            vals = [1.0] * n
            row_exprs.append(SparsePair(ind=idxs, val=vals))
            row_senses.append("E")
            row_rhs.append(1.0)
            row_names.append(f"assign_loc_{i}")

        # 2) Assignment by machine: sum_i x_{i,u} = 1     (n rows)
        row_assign_mach = {}  # u -> row index
        for u in range(n):
            idxs = [x_index[(i, u)] for i in range(n)]
            vals = [1.0] * n
            row_exprs.append(SparsePair(ind=idxs, val=vals))
            row_senses.append("E")
            row_rhs.append(1.0)
            row_names.append(f"assign_mach_{u}")

        # Keep track of absolute row indices as we add more families
        # 3) For each path k and position t:
        #    Convexity: sum_{i!=j} lambda_{k,t,i,j} = 1
        #    Tail-link: sum_{j!=i} lambda_{k,t,i,j} - x_{i, u_t^k} = 0   for each i
        #    Head-link: sum_{i!=j} lambda_{k,t,i,j} - x_{j, u_{t+1}^k} = 0   for each j
        row_conv = {}   # (k,t) -> row index
        row_tail = {}   # (k,t,i) -> row index
        row_head = {}   # (k,t,j) -> row index

        # First add empty rows (no lambda yet); only x appear (with -1 in link rows)
        row_base = len(row_names)
        for k, pk in enumerate(K_paths):
            U, L, Fk = pk["U"], pk["L"], pk["F"]
            for t in range(L):
                # 3.a Convexity
                row_names.append(f"conv_{k}_{t}")
                row_senses.append("E")
                row_rhs.append(1.0)
                row_exprs.append(SparsePair(ind=[], val=[]))  # will receive lambda columns later
                # index after adding: row_base + ...
                # We'll fill indices after cpx.linear_constraints.add
                # 3.b Tail-link rows: for each i, include -1*x_{i,u_t}
                ut = U[t]
                for i in range(n):
                    vid = x_index[(i, ut)]
                    row_names.append(f"tail_{k}_{t}_{i}")
                    row_senses.append("E")
                    row_rhs.append(0.0)
                    row_exprs.append(SparsePair(ind=[vid], val=[-1.0]))
                # 3.c Head-link rows: for each j, include -1*x_{j,u_{t+1}}
                ut1 = U[t+1]
                for j in range(n):
                    vid = x_index[(j, ut1)]
                    row_names.append(f"head_{k}_{t}_{j}")
                    row_senses.append("E")
                    row_rhs.append(0.0)
                    row_exprs.append(SparsePair(ind=[vid], val=[-1.0]))

        # Add all rows to CPLEX
        cpx.linear_constraints.add(lin_expr=row_exprs, senses=row_senses, rhs=row_rhs, names=row_names)

        # Build row index maps
        name_to_row = {nm: ii for ii, nm in enumerate(cpx.linear_constraints.get_names())}
        # fill assignment maps
        for i in range(n):
            row_assign_loc[i] = name_to_row[f"assign_loc_{i}"]
        for u in range(n):
            row_assign_mach[u] = name_to_row[f"assign_mach_{u}"]
        # fill path row maps
        for k, pk in enumerate(K_paths):
            U, L, Fk = pk["U"], pk["L"], pk["F"]
            for t in range(L):
                row_conv[(k, t)] = name_to_row[f"conv_{k}_{t}"]
                for i in range(n):
                    row_tail[(k, t, i)] = name_to_row[f"tail_{k}_{t}_{i}"]
                for j in range(n):
                    row_head[(k, t, j)] = name_to_row[f"head_{k}_{t}_{j}"]

        # -------------------------
        # Seed columns for each (k,t)
        # -------------------------
        def add_lambda_column(k: int, t: int, i: int, j: int):
            """
            Add lambda_{k,t,i,j} >= 0 with:
            obj = F_k * d_{ij},
            coeff 1 in conv(k,t),
            coeff 1 in tail(k,t,i),
            coeff 1 in head(k,t,j).
            """
            pk = K_paths[k]
            obj = pk["F"] * float(D[i, j])
            ind = [
                row_conv[(k, t)],
                row_tail[(k, t, i)],
                row_head[(k, t, j)]
            ]
            val = [1.0, 1.0, 1.0]
            vname = f"lam_{k}_{t}_{i}_{j}"
            cpx.variables.add(
                obj=[obj],
                lb=[0.0],
                ub=[cplex.infinity],
                types=[cpx.variables.type.continuous],
                names=[vname],
                columns=[SparsePair(ind=ind, val=val)],
            )
            vidx = cpx.variables.get_indices(vname)
            lam_index[(k, t, i, j)] = vidx
            return vidx
        
        def _force_lp_relaxation(cpx):
            """
            Ensure the RMP is an LP from the perspective of dual availability:
            - all variables are continuous
            - no SOS constraints
            - no indicator constraints
            """
            # 1) Make all variables continuous
            types = cpx.variables.get_types()
            if isinstance(types, str):
                types_list = list(types)
            else:
                types_list = list(types)
            to_change = [(idx, cpx.variables.type.continuous)
                        for idx, t in enumerate(types_list)
                        if t != cpx.variables.type.continuous]
            if to_change:
                cpx.variables.set_types(to_change)

            # 2) SOS and indicator constraints must not be present for LP duals
            num_sos = 0
            try:
                num_sos = cpx.SOS.get_num()
            except Exception:
                # Some CPLEX versions/editions: SOS interface may not exist; ignore
                num_sos = 0

            num_ind = 0
            try:
                num_ind = cpx.indicator_constraints.get_num()
            except Exception:
                num_ind = 0

            if num_sos > 0 or num_ind > 0:
                raise RuntimeError(
                    f"RMP cannot be treated as LP for duals: SOS={num_sos}, indicator={num_ind}."
                )
        
        # If warmstart provided, try to seed lambda consistent with x warmstart.
        # Otherwise, add a cheap default arc for each (k,t).
        def seed_columns():
            for k, pk in enumerate(K_paths):
                U, L, Fk = pk["U"], pk["L"], pk["F"]
                for t in range(L):
                    # if warmstart for x exists, try to pick (i,j) where
                    # x[i, U[t]] ~ 1 and x[j, U[t+1]] ~ 1; else pick cheapest (i!=j)
                    i_star = None
                    j_star = None
                    if warmstart is not None:
                        # warmstart keys are (i,u) in your signature (likely 0-based)
                        # We'll pick argmax_i x[i, ut] and argmax_j x[j, ut1]
                        ut, ut1 = U[t], U[t+1]
                        best_xi, best_xj = -1.0, -1.0
                        for i in range(n):
                            val = float(warmstart.get((i, ut), 0.0))
                            if val > best_xi:
                                best_xi, i_star = val, i
                        for j in range(n):
                            val = float(warmstart.get((j, ut1), 0.0))
                            if val > best_xj:
                                best_xj, j_star = val, j
                        if i_star is not None and j_star is not None and i_star != j_star:
                            add_lambda_column(k, t, i_star, j_star)
                            continue
                    # fallback: pick cheapest arc (i!=j)
                    best = None
                    best_val = math.inf
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                continue
                            val = float(D[i, j])
                            if val < best_val:
                                best_val, best = val, (i, j)
                    i0, j0 = best
                    add_lambda_column(k, t, i0, j0)

        seed_columns()
        _force_lp_relaxation(cpx)   # <-- ensure LP before CG loop

        # -------------------------
        # Column generation loop
        # -------------------------

        def get_duals_blocks():
            """
            Fetch duals for (mu, phi, psi) with robust guards.
            If CPLEX still complains (1017), report a clear message with diagnostics.
            """
            # 0) Enforce LP conditions
            _force_lp_relaxation(cpx)

            # 1) (Optional) Solve if not solved, ensure it is an LP solve
            #    Some CPLEX versions require a (re)solve after type changes
            if cpx.solution.get_status() not in (1, 101, 102):  # optimal statuses for LP
                cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.dual)
                cpx.solve()

            # 2) Build row list
            mu, phi, psi = {}, {}, {}
            all_rows, row_tags = [], []
            for k, pk in enumerate(K_paths):
                for t in range(pk["L"]):
                    r = row_conv[(k, t)]
                    all_rows.append(r); row_tags.append(("mu", k, t))
                    for i in range(n):
                        r = row_tail[(k, t, i)]
                        all_rows.append(r); row_tags.append(("phi", k, t, i))
                    for j in range(n):
                        r = row_head[(k, t, j)]
                        all_rows.append(r); row_tags.append(("psi", k, t, j))

            # 3) Get duals with graceful fallback
            try:
                dual_values = cpx.solution.get_dual_values(all_rows)
            except CplexSolverError as e:
                # CPLEX 1017 => model considered mixed-integer at this moment
                # This happens if any var is integer or there are SOS/indicators.
                # We already forced types to continuous and checked SOS/indicators.
                # Print diagnostics and re-raise with guidance.
                var_types = cpx.variables.get_types()
                non_cont = [i for i, t in enumerate(var_types)
                            if t != cpx.variables.type.continuous]
                msg = (f"CPLEX refused to provide duals (likely 1017). "
                    f"Non-continuous vars count={len(non_cont)}; "
                    f"SOS={getattr(cpx.SOS, 'get_num', lambda: 'NA')() if hasattr(cpx, 'SOS') else 'NA'}; "
                    f"Indicator={getattr(cpx.indicator_constraints, 'get_num', lambda: 'NA')() if hasattr(cpx, 'indicator_constraints') else 'NA'}; "
                    f"Ensure no integrality/indicators before pricing.")
                raise RuntimeError(msg) from e

            # 4) Unpack duals
            for tag, d in zip(row_tags, dual_values):
                if tag[0] == "mu":
                    _, k, t = tag;  mu[(k, t)] = float(d)
                elif tag[0] == "phi":
                    _, k, t, i = tag;  phi[(k, t, i)] = float(d)
                else:
                    _, k, t, j = tag;  psi[(k, t, j)] = float(d)
            return mu, phi, psi
        
        
        
        # Pricing: compute best (k,t,i,j) with rc < -rc_tol (or top K per block)
        def price_and_add_columns():
            added = 0
            new_candidates = []  # (rc, k,t,i,j)
            ptype = cpx.get_problem_type()
            names = {
                cpx.problem_type.LP: "LP",
                getattr(cpx.problem_type, "QP", None): "QP",
                getattr(cpx.problem_type, "QCP", None): "QCP",
                getattr(cpx.problem_type, "MILP", None): "MILP",
                getattr(cpx.problem_type, "MIQP", None): "MIQP",
                getattr(cpx.problem_type, "MIQCP", None): "MIQCP",
            }
            print("Problem type just before duals:", names.get(ptype, ptype))
            mu, phi, psi = get_duals_blocks()
            # Explore all (k,t) blocks
            for k, pk in enumerate(K_paths):
                U, L, Fk = pk["U"], pk["L"], pk["F"]
                if Fk <= 0.0:
                    continue
                for t in range(L):
                    # Compute rc for all i!=j (but skip columns we already have)
                    best_block = []
                    for i in range(n):
                        mui = mu[(k, t)]
                        phi_kti = phi[(k, t, i)]
                        Dij = D[i, :]  # vectorized
                        for j in range(n):
                            if i == j: 
                                continue
                            if (k, t, i, j) in lam_index:
                                continue
                            rc = Fk * float(Dij[j]) - mui - phi_kti - psi[(k, t, j)]
                            if rc < -rc_tol:
                                if price_top_per_block is None:
                                    new_candidates.append((rc, k, t, i, j))
                                else:
                                    best_block.append((rc, k, t, i, j))
                    if price_top_per_block is not None and best_block:
                        best_block.sort(key=lambda x: x[0])  # ascending rc
                        new_candidates.extend(best_block[:price_top_per_block])

            # Global cap
            if not new_candidates:
                return 0
            new_candidates.sort(key=lambda x: x[0])  # most negative first
            new_candidates = new_candidates[:max_new_columns_per_it]

            # Add columns
            for (rc, k, t, i, j) in new_candidates:
                add_lambda_column(k, t, i, j)
                added += 1
            return added

        # Main CG loop
        it = 0
        while True:
            # Solve current RMP (LP)
            cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.primal)
            cpx.parameters.simplex.tolerances.optimality.set(1e-9)
            cpx.solve()
            if cpx.solution.get_status() not in (1, 101, 102):  # 1=optimal
                # Try dual simplex fallback
                cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.dual)
                cpx.solve()
            if cpx.solution.get_status() not in (1, 101, 102):
                raise RuntimeError(f"RMP LP did not solve to optimality. Status={cpx.solution.get_status()}.")

            # Pricing
            added_cols = price_and_add_columns()
            it += 1
            if added_cols == 0 or it >= max_cg_iterations:
                break

        lb = cpx.solution.get_objective_value()  # column generation lower bound (LP)

        # -------------------------
        # Optional final MIP polish
        # -------------------------
        assignment = None
        is_feasible = False
        if do_final_mip_polish:
            # Turn x into binary and solve a MIP with the existing columns
            for (i, u), vid in x_index.items():
                cpx.variables.set_types(vid, cpx.variables.type.binary)
            # limit time
            cpx.parameters.timelimit.set(float(mip_time_limit))
            # emphasize optimality or feasibility as you prefer
            cpx.parameters.mip.tolerances.mipgap.set(1e-4)
            if not display_cplex_output:
                # keep LP logs off, but let MIP log show if you want
                pass
            cpx.solve()
            status = cpx.solution.get_status()
            # 101-102: Optimal/Feasible
            if status in (101, 102):
                x_val = cpx.solution.get_values([x_index[(i, u)] for i in range(n) for u in range(n)])
                x_val = np.array(x_val, dtype=float).reshape(n, n)
                # recover assignment: pick argmax u for each i
                assignment = [-1] * n
                for i in range(n):
                    u_star = int(np.argmax(x_val[i, :]))
                    assignment[i] = u_star
                # Check feasibility (one per row and col)
                row_ok = all(abs(sum(int(u == assignment[i]) for u in range(n)) - 1) <= 0.5 for i in range(n))
                col_ok = all(abs(sum(int(assignment[i] == u) for i in range(n)) - 1) <= 0.5 for u in range(n))
                is_feasible = row_ok and col_ok
            else:
                # If MIP didn't finish, fall back to LP x and round greedily
                x_val = cpx.solution.get_values([x_index[(i, u)] for i in range(n) for u in range(n)])
                x_val = np.array(x_val, dtype=float).reshape(n, n)
                assignment = list(np.argmax(x_val, axis=1))
                is_feasible = True  # Typically satisfies row sums; columns may need repair in edge cases

        # Compute objective under the final (possibly integer) assignment:
        # NOTE: If you need the exact QAP objective, compute using original flows: sum_{u,v} f_uv d_{π(u),π(v)}.
        # Here, Variant A master objective equals sum_k F_k sum_t d_{i_t, i_{t+1}}, induced by the lambda variables.
        # We read the final objective from CPLEX:
        obj_value = float(cpx.solution.get_objective_value())

        elapsed_time = time.time() - t0

        result = Solution(
            instance=getattr(self, "instance", None),
            solver=getattr(self, "solver_name", "CPLEX-CG"),
            assignment=assignment if is_feasible else None,
            objective=obj_value,
            lower_bound=lb,
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
            paths_agg (list, optional): Precomputed path aggregations
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

        # Load warm-start if provided
        if warmstart_path is not None:
            warmstart = read_warmstart(warmstart_path)
        else:
            warmstart = None

        
        # Run decomposition with aggregation enabled (default)
        paths_agg = decompose_flow_into_paths(problem.F, strategy="max_arc", seed=7)
        print("Decomposed {} paths with aggregation:".format(len(paths_agg)))
        for p in paths_agg:
            print(p)

        ok, err1 = check_decomposition_reconstruction(problem.F, paths_agg)
        print("Reconstruction OK:", ok, "L1 error:", err1)

        # Solve
        solution = self.solve(
            problem,
            fixed_variables=fixed_variables,
            warmstart=warmstart,
            paths_agg=paths_agg,
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
        "solver": "SFDCG_cplex",
        "formulation": "SFDCG",
        "is_relax": False,  # Use binary variables
        "time_limit": 120,
        "threads": 8,
        "log_output": True,
    }

    solver = SFDCGCPLEXSolver(config)

    # Example: solve a single instance
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
        solver.solve_instance(instance_file, output_path="SFDCG_result.json")
    else:
        print("Usage: python solver.py <instance_file>")
        print(
            "Example: python solver.py /home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/chr12a.dat"
        )
