"""Core QAP utilities: problem definitions, I/O, and common structures"""

from .problem import Problem
from .solution import Solution
from .io import write_result, read_result, RESULT_FORMAT

__all__ = ["Problem", "Solution", "write_result", "read_result", "RESULT_FORMAT"]
