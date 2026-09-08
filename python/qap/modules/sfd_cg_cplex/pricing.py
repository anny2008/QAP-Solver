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

import time

import cplex
from .column import Column
import numpy as np

np.float_ = np.float64
from docplex.mp.model import Model


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
        self.subDgraph = None  # will be built on demand

    # ------------------------------------------------------------
    # Solve pricing for all subgraphs, return list of negative columns
    # ------------------------------------------------------------
    def price_all_subgraphs(self, alpha, beta, branching_fixed_x):
        """
        alpha: dict k → dual of subgraph k constraint
        beta: dict (i,u) → dual of machine-location agreement constraint
        branching_fixed_x: dict (i,u) → 0 or 1
        """
        negative_cols = []

        for k in self.subgraphs:
            col = self._price_subgraph(k, alpha, beta, branching_fixed_x)
            if col is not None and col[0] < -self.eps:
                # col = (reduced_cost, Column_object)
                negative_cols.append(col)

        # Return all columns with negative reduced cost
        return negative_cols

    # ------------------------------------------------------------
    # Pricing per subgraph: build mini-QAP using the SFD formulation
    # and solve with CPLEX
    # ------------------------------------------------------------

    def _price_subgraph(self, k, alpha, beta, branching_fixed_x):
        _price_result = self._price_subgraph_SDD(k, alpha, beta, branching_fixed_x)
        return _price_result

    def _price_subgraph_SDD(self, k, alpha, beta, branching_fixed_x):

        subgraph = self.subgraphs[k]
        Fk, A_k, M_k = subgraph
        V = self.V
        D = self.D

        if len(M_k) <= 1:
            return None
        
        # decompose the distance graph into subgraphs
        def decompose_distance_graph(D):
            subDgraph_temp = {}
            for i in V:
                for j in V:
                    if i == j:
                        continue
                    if D[i][j] not in subDgraph_temp:
                        subDgraph_temp[D[i][j]] = []
                    subDgraph_temp[D[i][j]].append((i,j))
            subDgraph = {}
            didx = 0
            for D_s, arcs in subDgraph_temp.items():
                nodes = set()
                for (u,v) in arcs:
                    nodes.add(u)
                    nodes.add(v)
                subDgraph[didx] = (didx, D_s, arcs, nodes)
                didx += 1
                # print(f"Subgraph {k}: D_s={D_s}, arcs={arcs}, nodes={nodes}")
            return subDgraph
        
        if self.subDgraph is None:
            self.subDgraph = decompose_distance_graph(D)

        # =========================
        # MODEL
        # =========================
        mdl = Model(name=f"pricing_k_{k}", log_output=False)

        # =========================
        # VARIABLES
        # =========================
        x = mdl.binary_var_dict(
            ((i, u) for i in V for u in M_k),
            name="x"
        )
        e_vars = mdl.continuous_var_dict(
            ((s,u,v) for s in self.subDgraph for u in M_k for v in M_k if u < v),
            name="e"
        )
        e = {}
        for (s, u, v) in e_vars:
            if (s, u, v) in e_vars:
                e[s, u, v] = e_vars[s, u, v]
                e[s, v, u] = e_vars[s, u, v]  # symmetry

        # =========================
        # OBJECTIVE
        # =========================
        expr = -alpha[k]

        # linear: - sum beta[i,u] x[i,u]
        expr -= mdl.sum(beta[(i, u)] * x[i, u]
                        for i in V for u in M_k)

        # quadratic: Fk * sum_{s \in subDgraph}  D_s * sum_{(u,v) in A_k} e[s,u,v]
        expr += Fk * mdl.sum(
            D_s * e[s, u, v]
            for s, D_s, edges, nodes in self.subDgraph.values()
            for (u, v) in A_k
        )

        mdl.minimize(expr)

        # =========================
        # CONSTRAINTS
        # =========================
        # Each node u assigned to exactly one location
        for u in M_k:
            mdl.add_constraint(
                mdl.sum(x[i, u] for i in V) == 1,
                ctname=f"assign_node_{u}"
            )

        # Each location assigned to at most one node
        for i in V:
            mdl.add_constraint(
                mdl.sum(x[i, u] for u in M_k) <= 1,
                ctname=f"capacity_loc_{i}"
            )
        
        # # e[s,u,v] >= x_iu + x_jv - 1 for all s, (i,j) in E_s, u,v\in M_k
        # for s, D_s, edges, nodes in self.subDgraph.values():
        #     for u in M_k:
        #         for v in M_k:
        #             for (i, j) in edges:
        #                 mdl.add_constraint(
        #                     e[s, u, v] >= x[i, u] + x[j, v] - 1,
        #                     ctname=f"SDD_edge_{s}_{u}_{i}_{v}_{j}"
        #                 )
        
        #  e[s,u,v] >= x_iu + sum_{j \in delta_D_s(i)} x[j,v] - 1  for all s, i\in V_s, u,v\in M_k
        for s, D_s, edges, nodes in self.subDgraph.values():
            for u in M_k:
                for v in M_k:
                    if u == v:
                        continue
                    for i in nodes:
                        # sum_xjv = mdl.sum(x[j, v] for (j, i2) in edges if i2 == i)
                        sum_xjv = mdl.sum(x[j, v] for (i2, j) in edges if i2 == i)
                        mdl.add_constraint(
                            e[s, u, v] >= x[i, u] + sum_xjv - 1,
                            ctname=f"SDD_{s}_{u}_{i}_{v}"
                        )
        # # degree constraints: sum_{v} e[s,u,v] = sum_i Degree_out_i^s x[i,u] \forall s,u
        # for s, D_s, edges, nodes in self.subDgraph.values():
        #     degree_out = {i: 0 for i in nodes}
        #     for (i, j) in edges:
        #         if i not in nodes:
        #             print(f"Warning: edge ({i},{j}) in subDgraph {s} has i not in nodes")
        #         degree_out[i] += 1
        #     for u in M_k:
        #         mdl.add_constraint(
        #             mdl.sum(e[s, u, v] for v in M_k if v != u) == mdl.sum(degree_out[i] * x[i, u] for i in nodes),
        #             ctname=f"degree_out_{s}_{u}"
        #         )
        # # degree constraints: sum_{u} e[s,u,v] = sum_i Degree_in_i^s x[i,v] \forall s,v
        # for s, D_s, edges, nodes in self.subDgraph.values():
        #     degree_in = {i: 0 for i in nodes}
        #     for (i, j) in edges:
        #         degree_in[j] += 1
        #     for v in M_k:
        #         mdl.add_constraint(
        #             mdl.sum(e[s, u, v] for u in M_k if u != v) == mdl.sum(degree_in[i] * x[i, v] for i in nodes),
        #             ctname=f"degree_in_{s}_{v}"
        #         )
        # sum_s e[s,u,v] == 1 for all u,v
        for (u,v) in A_k:
            if u == v:
                continue
            mdl.add_constraint(
                mdl.sum(e[s, u, v] for s in self.subDgraph) <= 1,
                ctname=f"edge_cover_{u}_{v}"
            )

        # =========================
        # SOLVE
        # =========================
        start_time = time.time()
        sol = mdl.solve(log_output=True)
        elapsed_time = time.time() - start_time
        print(f"Subgraph {k}: solved in {elapsed_time:.2f} seconds")

        if sol is None:
            print(f"Subgraph {k}: no solution")
            return None

        rc = sol.objective_value

        # =========================
        # EXTRACT SOLUTION
        # =========================
        phi_map = {}
        for u in M_k:
            for i in V:
                if sol[x[i, u]] > 0.5:
                    phi_map[u] = i
                    break

        e_ij = {}
        for (u, v) in A_k:
            i = phi_map[u]
            j = phi_map[v]
            e_ij[(i, j)] = 1

        true_cost = sum(D[i][j] for (i, j) in e_ij)

        col = Column(k, phi_map, e_ij, true_cost)
        
        # e_uv = {}
        # for (u, v) in A_k:
        #     for s in self.subDgraph:
        #         if (s, u, v) in e:
        #             if sol[e[s, u, v]] > 0:
        #                 # check if e^s_uv is consistent with x_iu and x_jv
        #                 i = phi_map[u]
        #                 j = phi_map[v]
        #                 if (i, j) not in self.subDgraph[s][2]:  # edges
        #                     print(f"Warning: e[{s},{u},{v}] = {sol[e[s,u,v]]} but ({i},{j}) not in edges of subDgraph {s}")
                        
        #                 # print(f"e[{s},{u},{v}] = {sol[e[s,u,v]]}")
        #                 if (u, v) not in e_uv:
        #                     e_uv[(u, v)] = [(s, sol[e[s, u, v]])]
        #                 else:
        #                     e_uv[(u, v)].append((s, sol[e[s, u, v]]))
        #     print(f"e_uv[{u},{v}] = {e_uv.get((u,v), 'not set')}")
        # print(e_ij)

        # print(
        #     f"Subgraph {k}, Fk={Fk}: "
        #     f"rc={rc:.6f}, true_cost={true_cost:.6f}, phi={phi_map}"
        # )

        return rc, col

    def _price_subgraph_RLT1(self, k, alpha, beta, branching_fixed_x):

        subgraph = self.subgraphs[k]
        Fk, A_k, M_k = subgraph
        V = self.V
        D = self.D

        if len(M_k) <= 1:
            return None

        # =========================
        # MODEL
        # =========================
        mdl = Model(name=f"pricing_k_{k}", log_output=False)

        # =========================
        # VARIABLES
        # =========================
        x = mdl.binary_var_dict(
            ((i, u) for i in V for u in M_k),
            name="x"
        )
        y11 = mdl.continuous_var_dict(
            ((i, u, j, v) for i in V for u in M_k for j in V for v in M_k if i <= j),
            name="y"
        )
        y = {}
        for (i, u, j, v) in y11:
            if (i, u, j, v) in y11:
                y[i, u, j, v] = y11[i, u, j, v]
                y[j, v, i, u] = y11[i, u, j, v]  # symmetry

        # =========================
        # OBJECTIVE
        # =========================
        expr = -alpha[k]

        # linear: - sum beta[i,u] x[i,u]
        expr -= mdl.sum(beta[(i, u)] * x[i, u]
                        for i in V for u in M_k)

        # quadratic: Fk * sum_{(u,v) in A_k} sum_{i,j} D[i][j] x[i,u] x[j,v]
        expr += Fk * mdl.sum(
            D[i][j] * y[i, u, j, v]
            for (u, v) in A_k
            for i in V for j in V
            if (i, u, j, v) in y
        )

        mdl.minimize(expr)

        # =========================
        # CONSTRAINTS
        # =========================
        # Each node u assigned to exactly one location
        for u in M_k:
            mdl.add_constraint(
                mdl.sum(x[i, u] for i in V) == 1,
                ctname=f"assign_node_{u}"
            )

        # Each location assigned to at most one node
        for i in V:
            mdl.add_constraint(
                mdl.sum(x[i, u] for u in M_k) <= 1,
                ctname=f"capacity_loc_{i}"
            )
            
        # RLT1: sum_j y[i,u,j,v] = x[i,u] for all i,u,v
        for i in V:
            for u in M_k:
                for v in M_k:
                    mdl.add_constraint(
                        mdl.sum(y[i, u, j, v] for j in V if (i,u,j,v) in y) == x[i, u],
                        ctname=f"RLT1_1_{i}_{u}_{v}"
                    )
        # RLT1: sum_v y[i,u,j,v] <= x[j,v] for all i,u,j
        for i in V:
            for u in M_k:
                for j in V:
                    mdl.add_constraint(
                        mdl.sum(y[i, u, j, v] for v in M_k if (i,u,j,v) in y) <= x[i,u],
                        ctname=f"RLT1_2_{j}_{u}_{v}"
                    )

        # =========================
        # SOLVE
        # =========================
        sol = mdl.solve(log_output=False)

        if sol is None:
            print(f"Subgraph {k}: no solution")
            return None

        rc = sol.objective_value

        # =========================
        # EXTRACT SOLUTION
        # =========================
        phi_map = {}
        for u in M_k:
            for i in V:
                if sol[x[i, u]] > 0.5:
                    phi_map[u] = i
                    break

        e_ij = {}
        for (u, v) in A_k:
            i = phi_map[u]
            j = phi_map[v]
            e_ij[(i, j)] = 1

        true_cost = sum(D[i][j] for (i, j) in e_ij)

        col = Column(k, phi_map, e_ij, true_cost)

        # print(
        #     f"Subgraph {k}, Fk={Fk}: "
        #     f"rc={rc:.6f}, true_cost={true_cost:.6f}, phi={phi_map}"
        # )

        return rc, col

    def _price_subgraph_quadratic(self, k, alpha, beta, branching_fixed_x):

        subgraph = self.subgraphs[k]
        Fk, A_k, M_k = subgraph
        V = self.V
        D = self.D

        if len(M_k) <= 1:
            return None

        # =========================
        # MODEL
        # =========================
        mdl = Model(name=f"pricing_k_{k}", log_output=False)

        # =========================
        # VARIABLES
        # =========================
        x = mdl.binary_var_dict(
            ((i, u) for i in V for u in M_k),
            name="x"
        )

        # =========================
        # OBJECTIVE
        # =========================
        expr = -alpha[k]

        # linear: - sum beta[i,u] x[i,u]
        expr -= mdl.sum(beta[(i, u)] * x[i, u]
                        for i in V for u in M_k)

        # quadratic: Fk * sum_{(u,v) in A_k} sum_{i,j} D[i][j] x[i,u] x[j,v]
        expr += Fk * mdl.sum(
            D[i][j] * x[i, u] * x[j, v]
            for (u, v) in A_k
            for i in V for j in V
        )

        mdl.minimize(expr)

        # =========================
        # CONSTRAINTS
        # =========================
        # Each node u assigned to exactly one location
        for u in M_k:
            mdl.add_constraint(
                mdl.sum(x[i, u] for i in V) == 1,
                ctname=f"assign_node_{u}"
            )

        # Each location assigned to at most one node
        for i in V:
            mdl.add_constraint(
                mdl.sum(x[i, u] for u in M_k) <= 1,
                ctname=f"capacity_loc_{i}"
            )

        # =========================
        # SOLVE
        # =========================
        sol = mdl.solve(log_output=False)

        if sol is None:
            print(f"Subgraph {k}: no solution")
            return None

        rc = sol.objective_value

        # =========================
        # EXTRACT SOLUTION
        # =========================
        phi_map = {}
        for u in M_k:
            for i in V:
                if sol[x[i, u]] > 0.5:
                    phi_map[u] = i
                    break

        e_ij = {}
        for (u, v) in A_k:
            i = phi_map[u]
            j = phi_map[v]
            e_ij[(i, j)] = 1

        true_cost = sum(D[i][j] for (i, j) in e_ij)

        col = Column(k, phi_map, e_ij, true_cost)

        # print(
        #     f"Subgraph {k}, Fk={Fk}: "
        #     f"rc={rc:.6f}, true_cost={true_cost:.6f}, phi={phi_map}"
        # )

        return rc, col
