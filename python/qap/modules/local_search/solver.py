"""Local search solver for QAP (lightweight 2-opt swap improvement)."""

from __future__ import annotations

import random
import time
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

        assignment = self._initial_assignment(problem, rng, warmstart, merged_fixed)
        current_obj = problem.evaluate_assignment(assignment)

        if self.method in {"tabu", "tabu_search", "two_opt", "local_search"}:
            assignment, current_obj = self._two_opt_search(
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
        fixed_locs = {loc for loc, _ in (fixed_variables or [])}
        iteration = 0
        while iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                break
            iteration += 1

            best_delta = 0.0
            best_move = None
            improved = False

            for i in range(n - 1):
                if i in fixed_locs:
                    continue
                for j in range(i + 1, n):
                    if j in fixed_locs:
                        continue
                    delta = self._delta_swap(problem, assignment, i, j)
                    if delta < -self.eps:
                        if self.local_search_strategy == "first_improvement":
                            assignment[i], assignment[j] = assignment[j], assignment[i]
                            current_obj += delta
                            improved = True
                            break
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)
                if improved and self.local_search_strategy == "first_improvement":
                    break

            if self.local_search_strategy != "first_improvement" and best_move is not None:
                i, j = best_move
                assignment[i], assignment[j] = assignment[j], assignment[i]
                current_obj += best_delta
                improved = True

            if not improved:
                break

        return assignment, current_obj

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
