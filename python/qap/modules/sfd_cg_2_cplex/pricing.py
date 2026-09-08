# pricing_form3.py
#
# Pricing engine for FORM3 Branch-and-Price.
# For each subgraph k, solves a "mini-QAP":
#       min  F_k * cost(φ_k) - Σ π_{φ(u),u} - α_k
#
# Produces a Column object containing:
#   - subgraph index
#   - φ-map u → i
#   - sparse e^k_{ij}
#   - cost = Σ d[ φ(u), φ(v) ] over arcs (u,v) in A_k

import cplex
from .column import Column


class Pricing:
    """
    Pricing engine for FORM3.

    For each subgraph k:
        solve:
            min F_k * sum_{(u,v) in A_k} d[φ(u), φ(v)]
                - sum_{u in V_k} π[φ(u),u]
                - α_k
    """

    def __init__(self, problem, subgraphs, eps=1e-9):
        self.problem = problem
        self.subgraphs = subgraphs
        self.V = list(range(problem.n))    # locations
        self.M = list(range(problem.m))    # machines
        self.D = problem.D
        self.eps = eps

    # ------------------------------------------------------------
    # Solve pricing for all subgraphs, return list of negative columns
    # ------------------------------------------------------------
    def price_all_subgraphs(self, dual_pi, dual_alpha, branching_fixed_x):
        """
        dual_pi:    dict (i,u) → π_{iu}
        dual_alpha: dict k → α_k
        branching_fixed_x: dict (i,u) → 0 or 1
        """
        negative_cols = []

        for k in self.subgraphs:
            col = self._price_subgraph(k, dual_pi, dual_alpha[k], branching_fixed_x)
            if col is not None and col[0] < -self.eps:
                # col = (reduced_cost, Column_object)
                negative_cols.append(col)

        # Return all columns with negative reduced cost
        return negative_cols

    # ------------------------------------------------------------
    # Pricing per subgraph: build mini-QAP using the SFD formulation
    # and solve with CPLEX
    # ------------------------------------------------------------
    def _price_subgraph(self, k, dual_pi, alpha_k, branching_fixed_x):
        """
        Pricing MIP for layer k.
        Returns (reduced_cost, Column) or None if no improving column exists.
        """
        subgraph = self.subgraphs[k]
        Fk, arcs, nodes = subgraph             # arcs = list of (u,v), nodes = V_k
        V = self.V                             # full location set
        D = self.D                             # distance matrix

        # Trivial layer cannot produce a column
        if len(nodes) <= 1:
            return None

        import cplex
        mdl = cplex.Cplex()
        mdl.set_log_stream(None)
        mdl.set_error_stream(None)
        mdl.set_warning_stream(None)
        mdl.set_results_stream(None)

        # ============
        # VARIABLES
        # ============
        x_names = {}
        for u in nodes:
            for i in V:
                nm = f"x_{i}_{u}"
                x_names[(i,u)] = nm
                mdl.variables.add(names=[f"x_{i}_{u}"], types="B")

        e_names = {}
        for i in V:
            for j in V:
                nm = f"e_{k}_{i}_{j}"
                e_names[(i,j)] = nm
                mdl.variables.add(names=[f"e_{k}_{i}_{j}"], types="B")

        # ================================
        # Objective
        # ================================
        obj_terms = []

        # Fk * d[i][j] * e[i,j]
        for i in V:
            for j in V:
                # c = Fk * D[i,j]
                c = D[i,j]
                if c != 0 and (i,j) in e_names:
                    if (e_names[(i,j)], c) in obj_terms:
                        print(f"Warning: duplicate objective term for e[{i},{j}] with coefficient {c}")
                    obj_terms.append((e_names[(i,j)], c))
        # - π[i,u] * x[i,u]
        # for u in nodes:
        #     for i in V:
        #         c = -float(dual_pi.get((i,u), 0))
        #         if c != 0 and (i,u) in x_names:
        #             if (x_names[(i,u)], c) in obj_terms:
        #                 print(f"Warning: duplicate objective term for x[{i},{u}] with coefficient {c}")
        #             obj_terms.append((x_names[(i,u)], c))
        # - α_k (constant term)
        # obj_terms.append((None, -float(alpha_k)))
        # print(obj_terms)

        mdl.objective.set_linear(obj_terms)
        mdl.objective.set_sense(mdl.objective.sense.minimize)

        # ================================
        # Constraints
        # ================================

        # 1. Injectivity
        for u in nodes:
            vars_ = [x_names[(i,u)] for i in V]
            mdl.linear_constraints.add(
                lin_expr=[cplex.SparsePair(vars_, [1.0]*len(vars_))],
                senses=["E"], rhs=[1.0]
            )

        for i in V:
            vars_ = [x_names[(i,u)] for u in nodes]
            mdl.linear_constraints.add(
                lin_expr=[cplex.SparsePair(vars_, [1.0]*len(vars_))],
                senses=["L"], rhs=[1.0]
            )

        # # 2. Linking e and x: e[i,j] >= x[i,u] + x[j,v] - 1 for all (u,v) in arcs
        for (u,v) in arcs:
            for i in V:
                for j in V:
                    mdl.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            [e_names[(i,j)], x_names[(i,u)], x_names[(j,v)]],
                            [1.0, -1.0, -1.0]
                        )],
                        senses=["G"], rhs=[-1.0]
                    )
                    
        # 3. Degree constraints: sum_j e[i,j] = sum_{u in V_k} degree_out_u_Vk x[i,u] for all i,k
        degree_out = {u: sum(1 for (x,y) in arcs if x == u) for u in nodes}
        for i in V:
            vars_ = [e_names[(i,j)] for j in V if (i,j) in e_names]
            coefs_ = [1.0]*len(vars_)
            for u in nodes:
                vars_.append(x_names[(i,u)])
                coefs_.append(-float(degree_out[u]))
            mdl.linear_constraints.add(
                lin_expr=[cplex.SparsePair(vars_, coefs_)],
                senses=["E"], rhs=[0.0]
            )
        # 4. Degree constraints: sum_i e[i,j] = sum_{u in V_k} degree_in_u_Vk x[j,u] for all j,k
        degree_in = {u: sum(1 for (x,y) in arcs if y == u) for u in nodes}
        for j in V:
            vars_ = [e_names[(i,j)] for i in V if (i,j) in e_names]
            coefs_ = [1.0]*len(vars_)
            for u in nodes:
                vars_.append(x_names[(j,u)])
                coefs_.append(-float(degree_in[u]))
            mdl.linear_constraints.add(
                lin_expr=[cplex.SparsePair(vars_, coefs_)],
                senses=["E"], rhs=[0.0]
            )
        
        # ============
        # Solve MIP
        # ============
        mdl.parameters.mip.display.set(0)
        mdl.parameters.emphasis.mip.set(1)       # focus on finding feasibility quickly
        mdl.solve()

        if mdl.solution.get_status() not in [mdl.solution.status.MIP_optimal,
                                            mdl.solution.status.optimal]:
            print(f"Subgraph {k}: MIP not solved to optimality, status={mdl.solution.get_status_string()}")
            return None

        rc = mdl.solution.get_objective_value()  # subtract constant term to get true reduced cost

        # if rc >= -1e-9:
        #     print(f"Subgraph {k}: No improving column found (rc={rc:.6f})")
        #     return None  # No improving column

        # Build column (pattern)
        phi_map = {}
        e_ij = {}
        for u in nodes:
            for i in V:
                if mdl.solution.get_values(x_names[(i,u)]) > 0.5:
                    phi_map[u] = i
                    break
        for (u,v) in arcs:
            i = phi_map[u]
            j = phi_map[v]
            e_ij[(i,j)] = 1
        true_cost = sum(D[i][j] for (i,j) in e_ij.keys())


        col = Column(k, phi_map, e_ij, true_cost)
        print(f"Subgraph {k}, {Fk}: Found improving column with rc={rc:.6f}, cost={true_cost:.6f}, phi={phi_map}")
        return (rc, col)