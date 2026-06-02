"""QAP Problem class and utilities."""
import re
import numpy as np
from pathlib import Path


class Problem:
    """Represents a Quadratic Assignment Problem instance."""

    def __init__(self, n, m, F, D):
        """
        Initialize a QAP problem.

        Args:
            n (int): Problem size (number of facilities/locations)
            m (int): Number of machines (for enriched instances)
            F (ndarray): Flow matrix (m x m)
            D (ndarray): Distance matrix (n x n)
        """
        self.n = n
        self.m = m
        self.F = np.array(F, dtype=float)
        self.D = np.array(D, dtype=float)
        self.F_positive = None  # Cache for positive flow entries
        self.D_positive = None  # Cache for positive distance entries
        assert self.F.shape == (m, m), f"Flow matrix F must be of shape ({m}, {m})"
        assert self.D.shape == (n, n), f"Distance matrix D must be of shape ({n}, {n})"

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
        
        # Read file and parse all tokens
        content = filepath.read_text().strip()
        tokens = content.split()

        # First token is problem size
        n = int(tokens[0])

        # Next n*n tokens are distance matrix (input: n, D, F)
        D = np.array([float(tokens[i]) for i in range(1, n*n + 1)]).reshape(n, n)

        # Next n*n tokens are flow matrix
        F = np.array([float(tokens[i]) for i in range(n*n + 1, 2*n*n + 1)]).reshape(n, n)

        return cls(n, n, F, D)

    @classmethod
    def from_raw_matrix_file(cls, filepath):
        """
        Load a QAP-like problem where the file begins with:
            <num_locations> <num_machines>
        followed by:
            D (num_locations x num_locations values)
            F (num_machines x num_machines values)
        """

        filepath = Path(filepath)
        tokens = filepath.read_text().split()

        L = int(tokens[0])  # number of locations
        M = int(tokens[1])  # number of machines

        expected_D = L * L
        expected_F = M * M

        values = [float(t) for t in tokens[2:]]

        assert len(values) == expected_D + expected_F, (
            f"File does not contain correct number of values: "
            f"expected {expected_D+expected_F}, got {len(values)}"
        )

        D = np.array(values[:expected_D]).reshape((L, L))
        F = np.array(values[expected_D:]).reshape((M, M))

        # Create problem
        problem = cls(L, M, F, D)   # n = number of locations, m = number of machines
        problem.num_locations = L

        return problem
    
    @classmethod
    def from_full_instance(cls, matrix_file, workstations_file,
                           machines_file, fixed_file=None):
        """
        Charge un problème complet :
        - matrices D et F
        - workstations (positions)
        - machines (facilities)
        - affectations fixes éventuelles

        Retourne : Problem enrichi
        """

        # === 1) Charger matrices ============================
        problem = cls.from_raw_matrix_file(matrix_file)

        # === 2) Charger workstations ========================
        workstations = {}
        for line in Path(workstations_file).read_text().strip().splitlines():
            tok = line.split()
            loc_id = int(tok[0])
            x, y = int(tok[1]), int(tok[2])
            side = tok[3]
            slot = int(tok[4])
            workstations[loc_id] = dict(x=x, y=y, side=side, slot=slot)

        problem.workstations = workstations

        # === 3) Charger machines ============================
        machines = {}
        for line in Path(machines_file).read_text().strip().splitlines():
            tok = line.split()
            machine_id = int(tok[0])
            name = tok[1]
            value = float(tok[2])
            machines[machine_id] = dict(name=name, value=value)

        problem.machines = machines

        # === 4) Charger fixed assignments ====================
        fixed_assignments = {}
        if fixed_file is not None:
            for line in Path(fixed_file).read_text().strip().splitlines():
                if "=" not in line:
                    continue
                if line.strip().endswith("= 1"):
                    # extraire machine_id et location_id
                    # format: x_<machine_id>_name= ... id=<loc> = 1
                    loc, mid = re.search(r"x_(\d+)_(\d+)", line).groups()
                    loc, mid = int(loc), int(mid)
                    fixed_assignments[loc] = mid

        problem.fixed_assignments = fixed_assignments

        return problem

    def evaluate_assignment(self, assignment):
        """
        Evaluate the objective value for a given assignment.

        Args:
            assignment (list or ndarray): Permutation π where π[i] = u means
                                         facility u is assigned to location i

        Returns:
            float: Objective value = sum(F[u,v] * D[i, j]) for all i,j
        """
        if self.F_positive is None:
            self.F_positive = {(u,v): self.F[u,v] for u in range(self.m) for v in range(self.m) if self.F[u,v] > 0}
        if self.D_positive is None:
            self.D_positive = {(i,j): self.D[i,j] for i in range(self.n) for j in range(self.n) if self.D[i,j] > 0}
        
        obj = 0.0
        if len(self.F_positive) > len(self.D_positive):
            assignment2 = {i: assignment[i] for i in range(self.n)}
            # compute sum over positive distances
            for (i,j) in self.D_positive:
                u = assignment[i]
                v = assignment[j]
                if u != -1 and v != -1:  # only consider pairs of distinct locations with assigned facilities
                    obj += self.F[u, v] * self.D[i, j]
        else:            # compute sum over positive flows
            # print(assignment)
            assignment2 = {u: i for i, u in enumerate(assignment) if u != -1}
            # print(assignment2)
            for (u,v) in self.F_positive:
                i = assignment2[u]
                j = assignment2[v]
                obj += self.F[u, v] * self.D[i, j]

        return obj

    def evaluate_assignment_dict(self, assignment):
        """
        Evaluate the objective value for a given assignment.

        Args:
            assignment (dict): Dict π where π[i] = u means
                               facility u is assigned to location i

        Returns:
            float: Objective value = sum(F[u,v] * D[i, j]) for all i,j
        """
        if self.F_positive is None:
            self.F_positive = {(u,v): self.F[u,v] for u in range(self.m) for v in range(self.m) if self.F[u,v] > 0}
        if self.D_positive is None:
            self.D_positive = {(i,j): self.D[i,j] for i in range(self.n) for j in range(self.n) if self.D[i,j] > 0}
        
        obj = 0.0
        if len(self.F_positive) > len(self.D_positive):
            assignment2 = {i: assignment[i] for i in range(self.n)}
            # compute sum over positive distances
            for (i,j) in self.D_positive:
                u = assignment[i]
                v = assignment[j]
                if u != -1 and v != -1:  # only consider pairs of distinct locations with assigned facilities
                    obj += self.F[u, v] * self.D[i, j]
        else:            # compute sum over positive flows
            # print(assignment)
            assignment2 = {u: i for i, u in assignment.items() if u != -1}
            # print(assignment2)
            for (u,v) in self.F_positive:
                i = assignment2[u]
                j = assignment2[v]
                obj += self.F[u, v] * self.D[i, j]

        return obj

    def cut_problem(self, size):
        """
        Create a subproblem of given size by taking the top-left submatrices.

        Args:
            size (int): Size of the subproblem

        Returns:
            Problem: The subproblem instance
        """
        assert size <= self.n, "Subproblem size must be less than or equal to original size"
        F_sub = self.F[:size, :size]
        D_sub = self.D[:size, :size]
        return Problem(size, F_sub, D_sub)
    
    def __repr__(self):
        return f"Problem(n={self.n})"
