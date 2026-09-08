# Pricing engine for SFD Branch-and-Price.
# A Column is a variable e^k_{ij}

import cplex


class Pricing:
    """
    Pricing engine for SFD Branch-and-Price.

    For each subgraph k:
        solve:
            
    """

    def __init__(self, problem, subgraphs, eps=1e-9):
        self.problem = problem
        self.subgraphs = subgraphs
        self.V = list(range(problem.n))    # locations
        self.M = list(range(problem.m))    # machines
        self.D = problem.D
        self.eps = eps

    # ------------------------------------------------------------
    # Solve pricing 
    # rc(k,i,j) = F_k*d_ij - sum_u (lambda_kiu + theta_kju)
    # ------------------------------------------------------------
    def price(self, Omega, dual_lambda, dual_theta):
        """
        dual_lambda:    dict (k,i,u) → λ_{k,i,u}
        dual_theta:     dict (k,j,u) → θ_{k,j,u}
        """
        negative_cols = {}
        most_negative_cols = {}
        most_negative_rc = {}

        for k in self.subgraphs:
            F_k, arcs, nodes = self.subgraphs[k]
            negative_cols[k] = []
            most_negative_cols[k] = []
            most_negative_rc[k] = float('inf')
            for i in self.V:
                for j in self.V:
                    if (k,i,j) in Omega[k]:
                        continue  # column already in RMP
                    # check if column is feasible with branching decisions
                    
                    rc = F_k * self.D[i][j]  # F_k * d_ij
                    for u,v in arcs:  # nodes of subgraph k
                        rc -= dual_lambda.get((k,i), 0)  # - λ_{k,i,u}
                        rc -= dual_theta.get((k,j), 0)   # - θ_{k,j,u}
                    if rc < -self.eps:
                        negative_cols[k].append((rc, (k,i,j)))
                        if rc < most_negative_rc[k]:
                            most_negative_rc[k] = rc
                            most_negative_cols[k] = [(rc,(k,i,j))]
                        elif rc == most_negative_rc[k]:
                            most_negative_cols[k].append((rc,(k,i,j)))
        # Return all columns with negative reduced cost
        return negative_cols, most_negative_cols, most_negative_rc