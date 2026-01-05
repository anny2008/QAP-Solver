"""Solution class for QAP solver results."""

from pathlib import Path
from typing import Optional, List


class Solution:
    """Represents a QAP solver result."""

    def __init__(
        self,
        instance: str,
        solver: str,
        assignment: Optional[List[int]] = None,
        objective: Optional[float] = None,
        lower_bound: Optional[float] = None,
        time: float = 0.0,
    ):
        """
        Initialize a Solution.

        Args:
            instance (str): Instance name (without path or extension)
            solver (str): Solver/method identifier
            assignment (list, optional): Permutation π where π[i] = j means
                                        facility i → location j
            objective (float, optional): Best objective value found
            lower_bound (float, optional): Lower bound on optimal (None for heuristics)
            time (float): Wall-clock execution time in seconds
        """
        self.instance = instance
        self.solver = solver
        self.assignment = list(assignment) if assignment is not None else None
        self.objective = objective
        self.lower_bound = lower_bound
        self.time = time

    @property
    def gap(self) -> Optional[float]:
        """
        Calculate optimality gap (%).

        Returns:
            float: Gap = (objective - lower_bound) / lower_bound * 100
            None: If gap cannot be computed (heuristic or no solution)
        """
        if (
            self.objective is None
            or self.lower_bound is None
            or self.lower_bound <= 0
        ):
            return None
        return 100.0 * (self.objective - self.lower_bound) / self.lower_bound

    def to_dict(self) -> dict:
        """
        Convert solution to dictionary format.

        Returns:
            dict: Dictionary with instance, solver, objective, lower_bound, gap, time, assignment
        """
        return {
            "instance": self.instance,
            "solver": self.solver,
            "objective": self.objective,
            "lower_bound": self.lower_bound,
            "gap": self.gap,
            "time": self.time,
            "assignment": self.assignment,
        }

    def __repr__(self):
        status = "solved" if self.objective is not None else "infeasible"
        return (
            f"Solution({self.instance}, {self.solver}, status={status}, "
            f"obj={self.objective}, time={self.time:.2f}s)"
        )
