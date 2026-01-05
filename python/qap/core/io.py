"""Input/Output utilities for QAP solver."""

import json
from pathlib import Path
from typing import Union

from .solution import Solution


def write_result(
    solution: Solution, filepath: Union[str, Path], indent: int = 2
) -> None:
    """
    Write a solution to a JSON result file.

    Args:
        solution (Solution): The solution to write
        filepath (str or Path): Output file path
        indent (int): JSON indentation level (default: 2)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(solution.to_dict(), f, indent=indent)


def read_result(filepath: Union[str, Path]) -> Solution:
    """
    Read a solution from a JSON result file.

    Args:
        filepath (str or Path): Input file path

    Returns:
        Solution: The loaded solution
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        data = json.load(f)

    return Solution(
        instance=data.get("instance"),
        solver=data.get("solver"),
        assignment=data.get("assignment"),
        objective=data.get("objective"),
        lower_bound=data.get("lower_bound"),
        time=data.get("time", 0.0),
    )


# Result format specification
RESULT_FORMAT = {
    "instance": "str (instance name without path/extension)",
    "solver": "str (solver/method identifier)",
    "objective": "float or null (best objective value found)",
    "lower_bound": "float or null (lower bound on optimal; null for heuristics)",
    "gap": "float or null (optimality gap in %; null if not applicable)",
    "time": "float (wall-clock execution time in seconds)",
    "assignment": "array or null (permutation π where π[i] = j means facility i → location j)",
}
