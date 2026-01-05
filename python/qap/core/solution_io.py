"""
Unified QAP Solution I/O Module

Provides consistent utilities for reading and writing QAP solutions
in QAPLIB format across all solvers.

Format (QAPLIB):
  Line 1: n optimal_value
  Line 2: location_1 location_2 ... location_n (1-indexed)

Example:
  12  578
  12  7  9  3  4  8  11  1  5  6  10  2
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


def read_solution(filepath: str) -> Tuple[int, List[int], float]:
    """
    Read a solution from QAPLIB format file.
    
    Args:
        filepath: Path to .sln file
        
    Returns:
        Tuple of (n, assignment_0indexed, objective)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Solution file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        # Read first line: n optimal_value
        line1 = f.readline().strip().split()
        if len(line1) < 2:
            raise ValueError(f"Invalid format in {filepath}: first line should have 'n optimal_value'")
        
        try:
            n = int(line1[0])
            objective = float(line1[1])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid format in {filepath}: cannot parse 'n optimal_value': {e}")
        
        # Read second line: assignment (1-indexed)
        line2 = f.readline().strip().split()
        if len(line2) != n:
            raise ValueError(f"Invalid format in {filepath}: assignment should have {n} values, got {len(line2)}")
        
        try:
            assignment_1indexed = [int(x) for x in line2]
        except ValueError as e:
            raise ValueError(f"Invalid format in {filepath}: cannot parse assignment: {e}")
        
        # Convert to 0-indexed
        assignment = Solution.from_1indexed(assignment_1indexed)
    
    return n, assignment, objective


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
    
    # Convert to 1-indexed for writing
    assignment_1indexed = [x + 1 for x in assignment]
    
    with open(filepath, 'w') as f:
        # Write first line: n objective
        f.write(f"{n:6d} {objective:12.0f}\n")
        
        # Write second line: assignment (1-indexed, space-separated)
        f.write(" ".join(str(x) for x in assignment_1indexed) + "\n")


def read_warmstart(filepath: str) -> dict:
    """
    Read a warm-start solution into a dictionary format.
    
    Args:
        filepath: Path to .sln file
        
    Returns:
        Dictionary {(i, u): 1.0} for warm-starting (0-indexed)
    """
    n, assignment, _ = read_solution(filepath)
    
    warm_start = {}
    for i, u in enumerate(assignment):
        warm_start[(i, u)] = 1.0
    
    return warm_start


if __name__ == "__main__":
    # Example usage
    print("QAP Solution I/O Module")
    
    # Example reading
    try:
        n, assignment, obj = read_solution('/home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/nug12.sln')
        print(f"Read solution: n={n}, objective={obj}")
        print(f"Assignment (0-indexed): {assignment}")
    except Exception as e:
        print(f"Error reading: {e}")
