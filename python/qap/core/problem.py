"""QAP Problem class and utilities."""

import numpy as np
from pathlib import Path


class Problem:
    """Represents a Quadratic Assignment Problem instance."""

    def __init__(self, n, F, D):
        """
        Initialize a QAP problem.

        Args:
            n (int): Problem size (number of facilities/locations)
            F (ndarray): Flow matrix (n x n)
            D (ndarray): Distance matrix (n x n)
        """
        self.n = n
        self.F = np.array(F, dtype=float)
        self.D = np.array(D, dtype=float)

        # Validate dimensions
        assert self.F.shape == (n, n), f"Flow matrix must be {n}x{n}, got {self.F.shape}"
        assert self.D.shape == (n, n), f"Distance matrix must be {n}x{n}, got {self.D.shape}"

    @classmethod
    def from_qaplib(cls, filepath):
        """
        Load a QAP instance from QAPLIB format file.

        QAPLIB format:
            n
            f(0,0) f(0,1) ... f(0,n-1)
            ...
            f(n-1,0) ... f(n-1,n-1)
            d(0,0) d(0,1) ... d(0,n-1)
            ...
            d(n-1,0) ... d(n-1,n-1)

        Args:
            filepath (str or Path): Path to the QAPLIB file

        Returns:
            Problem: The loaded problem instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # First line is problem size
        n = int(lines[0])

        # Read flow matrix (n lines)
        F = []
        for i in range(1, n + 1):
            row = list(map(float, lines[i].split()))
            F.append(row)

        # Read distance matrix (next n lines)
        D = []
        for i in range(n + 1, 2 * n + 1):
            row = list(map(float, lines[i].split()))
            D.append(row)

        return cls(n, F, D)

    def evaluate(self, assignment):
        """
        Evaluate the objective value for a given assignment.

        Args:
            assignment (list or ndarray): Permutation π where π[i] = j means
                                         facility i is assigned to location j

        Returns:
            float: Objective value = sum(F[i,j] * D[π[i], π[j]])
        """
        assignment = np.array(assignment, dtype=int)
        assert len(assignment) == self.n, f"Assignment length must be {self.n}"

        obj = 0.0
        for i in range(self.n):
            for j in range(self.n):
                obj += self.F[i, j] * self.D[assignment[i], assignment[j]]

        return obj

    def __repr__(self):
        return f"Problem(n={self.n})"
