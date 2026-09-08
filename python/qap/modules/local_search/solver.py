"""Local search solver for QAP (lightweight 2-opt swap improvement)."""

from __future__ import annotations

import random
import time
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qap.core import Problem, Solution
from qap.core.io import write_result
from qap.core.solution_io import read_warmstart


class LocalSearchSolver:
    """Simple local search solver used for warm starts and heuristics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.config = config
        self.method = config.get("method", "two_opt")
        self.time_limit = float(config.get("time_limit", 10))
        self.max_iterations = int(config.get("max_iterations", 1000))
        self.initial_solution = config.get("initial_solution", "greedy")
        self.seed = config.get("seed", None)
        self.local_search_strategy = config.get("local_search_strategy", "best_improvement")
        self.eps = float(config.get("epsilon", 1e-9))

    def solve(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart: Optional[Dict[int, int]] = None,
    ) -> Solution:
        start_time = time.time()
        rng = random.Random(self.seed)

        merged_fixed = self._merge_fixed_variables(problem, fixed_variables)
        print(f"Merged fixed variables: {merged_fixed}")
        assignment = self._initial_assignment(problem, rng, warmstart, merged_fixed)
        # assignment = [-1] * problem.n
        # # read inital solution from warmstart
        # obj = 0
        # warmstartfile = "/home/local.isima.fr/antran/UFF/QAP-Solver/data/M/instance_318_LEFT.sln"
        # with open(warmstartfile, "r") as f:
        #     tokens = f.read().strip().split()
        #     u = 0
        #     print(tokens[0], tokens[1])
        #     for tokens in tokens[2:]:
        #         i = int(tokens)
        #         # print(f"Warmstart: location {i} assigned to facility {u}")
        #         assignment[i] = u
        #         u += 1
        #         # print(f"Warmstart: location {i} assigned to facility {assignment[i]}")
        current_obj = problem.evaluate_assignment(assignment)
        print(f"Initial solution: objective={current_obj:.4f}")
        # # check if initial solution is valid
        # for loc, fac in merged_fixed:
        #     if assignment[loc] != fac:
        #         print(f"Warning: initial solution does not satisfy fixed variable x[{loc}]={fac} (assigned {assignment[loc]})")

        current_obj = self._two_opt_search(
            problem,
            assignment,
            current_obj,
            start_time,
            merged_fixed,
        )

        elapsed = time.time() - start_time
        return Solution(
            instance="",
            solver="local_search",
            assignment=assignment,
            objective=current_obj,
            time=elapsed,
            lower_bound=None,
        )

    def _initial_assignment(
        self,
        problem: Problem,
        rng: random.Random,
        warmstart: Optional[Dict[int, int]] = None,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
    ) -> List[int]:
        n = problem.n
        m = problem.m

        assignment = [-1] * n
        used = set()
        for loc, fac in (fixed_variables or []):
            if 0 <= loc < n and 0 <= fac < m:
                assignment[loc] = fac
                used.add(fac)

        if warmstart:
            for loc, fac in warmstart.items():
                if 0 <= loc < n and 0 <= fac < m:
                    if assignment[loc] == -1:
                        assignment[loc] = fac
                        used.add(fac)
            remaining = [u for u in range(m) if u not in used]
            rng.shuffle(remaining)
            for i in range(n):
                if assignment[i] == -1:
                    if remaining:
                        assignment[i] = remaining.pop()
                    else:
                        assignment[i] = rng.randrange(m)
            return assignment

        if self.initial_solution == "random":
            perm = list(range(m))
            rng.shuffle(perm)
            for i in range(n):
                if assignment[i] == -1:
                    if perm:
                        assignment[i] = perm.pop()
                    else:
                        assignment[i] = rng.randrange(m)
            return assignment

        flow_score = [float(problem.F[u, :].sum() + problem.F[:, u].sum()) for u in range(m)]
        dist_score = [float(problem.D[i, :].sum() + problem.D[:, i].sum()) for i in range(n)]

        facilities = sorted(range(m), key=lambda u: flow_score[u], reverse=True)
        locations = sorted(range(n), key=lambda i: dist_score[i])

        for loc, fac in zip(locations, facilities):
            if assignment[loc] == -1:
                assignment[loc] = fac
                used.add(fac)
        remaining = [u for u in range(m) if u not in used]
        rng.shuffle(remaining)
        for i in range(n):
            if assignment[i] == -1:
                if remaining:
                    assignment[i] = remaining.pop()
                else:
                    assignment[i] = rng.randrange(m)
        return assignment

    def _two_opt_search(
        self,
        problem: Problem,
        assignment: List[int],
        current_obj: float,
        start_time: float,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[int], float]:
        n = problem.n
        fixed_locs = {loc: fac for loc, fac in (fixed_variables or [])}
        print(f"Fixed locations: {fixed_locs}")
        iteration = 0
        current_obj = problem.evaluate_assignment(assignment)
        while iteration < self.max_iterations:
            print(f"Iteration {iteration}, objective={current_obj:.4f}")
            if time.time() - start_time > self.time_limit:
                break
            iteration += 1

            best_delta = 0.0
            best_move = None
            improved = False
            # for i in tqdm(range(n)):
            for i in range(n):
                if i in fixed_locs:
                    continue
                u = assignment[i]
                for j in range(n):
                    if j in fixed_locs or i == j:
                        continue
                    v = assignment[j]
                    if v == u:
                        continue
                    # for loc, fac in fixed_locs.items():
                    #     if assignment[loc] != fac:
                    #         print(f"Error: before swap, fixed variable x[{loc}]={fac} is violated (assigned {assignment[loc]})")
                    # for u1 in range(problem.m):
                    #     found = False
                    #     for u2 in assignment:
                    #         if u1 == u2:
                    #             found = True        
                    #             break
                    #     if not found:
                    #         print(f"Error: before swap, facility {u1} is not assigned to any location")
                                
                    print(i,u,j,v)
                    print(assignment[i], assignment[j])
                    print(f"Before {assignment}")
                    new_assignment = assignment.copy()
                    # swap u and v
                    new_assignment[i] = v
                    new_assignment[j] = u
                    
                    # check if assignment is still valid (should always be valid since we only swap unfixed locations)
                    # for loc, fac in fixed_locs.items():
                    #     if new_assignment[loc] != fac:
                    #         print(f"Error: after swap, fixed variable x[{loc}]={fac} is violated (assigned {new_assignment[loc]})")
                    # for u1 in range(problem.m):
                    #     found = False
                    #     for u2 in new_assignment:
                    #         if u1 == u2:
                    #             found = True        
                    #             break
                    #     if not found:
                    #         print(i,u,j,v)
                    #         print(new_assignment[i], new_assignment[j])
                    #         print(f"Error: after swap, facility {u1} is not assigned to any location")

                    #         print(f"After {new_assignment}")
                                

                    # recompute objective
                    # start_time_comp = time.time()
                    new_obj = problem.evaluate_assignment(new_assignment)
                    # elapsed_comp = time.time() - start_time_comp
                    # print(f"    Swapping locations {i} and {j} changes objective to {new_obj:.4f} (computed in {elapsed_comp:.4f}s)")
                    if new_obj < current_obj:
                        print(f"    Swapping locations {i} and {j} improves objective to {current_obj:.4f} -> {new_obj:.4f}")
                        current_obj = new_obj
                        improved = True
                        assignment = new_assignment
                    else:
                        # swap back
                        print(f"    Swapping locations {i} and {j} worsens objective to {current_obj:.4f} -> {new_obj:.4f}, reverting")
                    
                        
                    # else:
                    #     print(f"    Swapping locations {i} and {j} does not change objective (still {current_obj:.4f} = {new_obj:.4f}), keeping swap")
                    #     current_obj = new_obj
                    #     improved = True

            if not improved:
                break

        return current_obj

    def _merge_fixed_variables(
        self,
        problem: Problem,
        fixed_variables: Optional[List[Tuple[int, int]]],
    ) -> List[Tuple[int, int]]:
        merged: Dict[int, int] = {}
        for loc, fac in getattr(problem, "fixed_assignments", {}).items():
            merged[int(loc)] = int(fac)
        items = fixed_variables.items() if isinstance(fixed_variables, dict) else (fixed_variables or [])
        for loc, fac in items:
            merged[int(loc)] = int(fac)
        return [(loc, fac) for loc, fac in merged.items()]

    def _delta_swap(self, problem: Problem, assignment: List[int], i: int, j: int) -> float:
        if i == j:
            return 0.0
        a = assignment[i]
        b = assignment[j]
        n = problem.n
        F = problem.F
        D = problem.D

        delta = (F[a, a] - F[b, b]) * (D[j, j] - D[i, i])
        delta += (F[a, b] - F[b, a]) * (D[j, i] - D[i, j])

        for k in range(n):
            if k == i or k == j:
                continue
            c = assignment[k]
            delta += (F[a, c] - F[b, c]) * (D[j, k] - D[i, k])
            delta += (F[c, a] - F[c, b]) * (D[k, j] - D[k, i])

        return float(delta)

    def solve_instance(
        self,
        instance_path: str,
        output_path: Optional[str] = None,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart_path: Optional[str] = None,
    ) -> Solution:
        if "QAPLIB" in instance_path:
            problem = Problem.from_qaplib(instance_path)
        else:
            problem = Problem.from_full_instance(
                matrix_file=instance_path,
                workstations_file=instance_path.replace(".txt", "_workstations.txt"),
                machines_file=instance_path.replace(".txt", "_machines.txt"),
                fixed_file=instance_path.replace(".txt", "_fixed.txt"),
            )
        warmstart = read_warmstart(warmstart_path) if warmstart_path else None
        solution = self.solve(problem, fixed_variables=fixed_variables, warmstart=warmstart)
        solution.instance = Path(instance_path).stem
        if output_path:
            write_result(solution, output_path)
        return solution
