"""QAP Solver Package"""

from .core import Problem, Solution, write_result, read_result, RESULT_FORMAT

__version__ = "0.1.0"
__author__ = "UFF Team"

__all__ = [
    "Problem",
    "Solution",
    "write_result",
    "read_result",
    "RESULT_FORMAT",
]
