"""
Unified QAP Solution I/O Module

Provides consistent utilities for reading and writing QAP solutions
in the legacy/current two-line QAPLIB format across all solvers.

Format (framework, matching .sln files in repo):
    Line 1: n objective_value
    Line 2: loc_1 loc_2 ... loc_n  (values are facilities, 1-indexed)

Example:
      12 9552
       7  5 12  2  1  3  9 11 10  6  8  4
"""

from pathlib import Path
from typing import Tuple, List, Optional


class Solution:
    """Represents a QAP solution in QAPLIB format."""
    
    def __init__(self, n: int, assignment: List[int], objective: float):
        """
        Initialize a solution.
        
        Args:
            n: Problem size
            assignment: Assignment array (0-indexed, facility i -> location assignment[i])
            objective: Objective value
        """
        self.n = n
        self.assignment = list(assignment)  # 0-indexed internally
        self.objective = objective
        
    def to_1indexed(self) -> List[int]:
        """Convert assignment to 1-indexed for file writing."""
        return [x + 1 for x in self.assignment]
    
    @staticmethod
    def from_1indexed(assignment_1indexed: List[int]) -> List[int]:
        """Convert 1-indexed assignment to 0-indexed."""
        return [x - 1 for x in assignment_1indexed]


def load_qaplib_solution(sln_filepath: str) -> Tuple[float, List[int]]:
    """
    Load a solution file and return assignment with location index -> facility value.
    
    Solution format (1-indexed values):
        Line 1: n objective_value
        Line 2: u_0 u_1 ... u_{n-1}
    where u_i is the facility placed at location i (1-indexed in file).
    
    Args:
        sln_filepath: Path to .sln file
        
    Returns:
        tuple: (optimal_value, assignment) where assignment[i] = facility at location i
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(sln_filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Solution file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        header = f.readline().strip().split()
        body = f.readline().strip().split()

    if len(header) < 2:
        raise ValueError("Invalid solution header: expected 'n objective'")

    n = int(header[0])
    optimal_value = float(header[1])

    perm_1indexed = [int(x) for x in body if x]
    if len(perm_1indexed) != n:
        raise ValueError(f"Invalid solution body: expected {n} entries, got {len(perm_1indexed)}")

    # Convert to 0-indexed assignment[i] = facility at location i
    assignment = [u - 1 for u in perm_1indexed]
    
    return optimal_value, assignment


def write_solution(filepath: str, n: int, assignment: List[int], objective: float) -> None:
    """
    Write a solution to QAPLIB format file.
    
    Args:
        filepath: Path to output .sln file
        n: Problem size
        assignment: Assignment array (0-indexed)
        objective: Objective value
        
    Raises:
        ValueError: If assignment size doesn't match n
    """
    if len(assignment) != n:
        raise ValueError(f"Assignment size {len(assignment)} doesn't match n={n}")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to 1-indexed for writing (loc -> facility)
    assignment_1indexed = [x + 1 for x in assignment]
    
    with open(filepath, 'w') as f:
        f.write(f"{n} {objective}\n")
        f.write(" ".join(str(x) for x in assignment_1indexed) + "\n")


def read_warmstart(filepath: str) -> dict:
    """
    Read a warm-start solution into a dictionary format.
    
    Args:
        filepath: Path to .sln file
        
    Returns:
        Dictionary {(i, u): 1.0} for warm-starting (0-indexed)
    """
    _, assignment = load_qaplib_solution(filepath)
    
    warm_start = {}
    for i, u in enumerate(assignment):
        warm_start[(i, u)] = 1.0
    
    return warm_start


if __name__ == "__main__":
    # Example usage
    print("QAP Solution I/O Module")
    
    # Example reading
    try:
        obj, assignment = load_qaplib_solution('/home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/nug12.sln')
        print(f"Read solution: n={len(assignment)}, objective={obj}")
        print(f"Assignment (0-indexed): {assignment}")
    except Exception as e:
        print(f"Error reading: {e}")
