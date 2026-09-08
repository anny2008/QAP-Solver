import cplex

class RMP:
    """
    Restricted Master Problem for SFD Branch-and-Price.
    Holds:
      - assignment variables x[i,u]
      - one pattern-selection constraint per subgraph
      - Omega is a subset of columns e^k_{ij}
      - degree equalities sum_j e^k_{ij} = outdeg_Gk(u) * x[i,u] and sum_j e^k_{ji} = indeg_Gk(v) * x[i,v]
      - uplift constraints linking e^k_{ij} to x[i,u] for all u in V_k
    Provides:
      - add_column()
      - fix_x()
      - solve()
      - get_duals()
      - get_solution()
    """

    def __init__(self, problem, subgraphs):
        self.problem = problem
        self.subgraphs = subgraphs   # dict k → {nodes, arcs, Fk}

        self.V = list(range(problem.n))
        self.M = list(range(problem.m))

        self.cpx = cplex.Cplex()
        self.cpx.objective.set_sense(self.cpx.objective.sense.minimize)

        # Storage
        self.x_index = {}         # (i,u) → var index
        self.e_index = {}    # (k, i, j) → var index
        self.Omega = {k: [] for k in subgraphs} # k → list of columns currently in RMP for subgraph k
        self.subgraph_constraints = {}   # subgraph → constraint index
        self.degree_in_constraints = {}  # (k,i,u) → constraint index
        self.degree_out_constraints = {}  # (k,i,u) → constraint index
        self.branching_fixed = {}     # (i,u) → {0 or 1}

        self._build_base_model()

    # ----------------------------------------------------------
    # BUILD BASE MODEL (only x + subgraph constraints + degree eq.)
    # ----------------------------------------------------------
    def _build_base_model(self):
        V, M = self.V, self.M
        n, m = self.problem.n, self.problem.m

        # --------------------------
        # Create x[i,u] binary vars
        # --------------------------
        x_names = []
        for i in V:
            for u in M:
                name = f"x_{i}_{u}"
                x_names.append(name)
                self.x_index[(i,u)] = len(self.x_index)

        self.cpx.variables.add(
            names=x_names,
            lb=[0.0] * len(x_names),
            ub=[1.0] * len(x_names),
            types=[self.cpx.variables.type.continuous] * len(x_names)
        )

        # --------------------------
        # Assignment constraints
        # --------------------------
        # Each location i gets at most one machine: sum_u x[i,u] <= 1
        for i in V:
            indices = [self.x_index[(i,u)] for u in M]
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=[1]*len(indices))],
                senses=["L"],
                rhs=[1],
                names=[f"assign_loc_{i}"]
            )

        # every machine assigned to exactly one location: sum_i x[i,u] = 1
        for u in M:
            indices = [self.x_index[(i,u)] for i in V]
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=[1]*len(indices))],
                senses=["E"],
                rhs=[1],
                names=[f"assign_machine_{u}"]
            )
        # --------------------------
        # Degree equalities:
        # sum_j e^k_{ij} = sum_u outdeg_Gk(u) * x[i,u]
        # sum_i e^k_{ij} = sum_v indeg_Gk(v) * x[j,v]
        # Left side added when columns appear
        # --------------------------
        for k, subgraph in self.subgraphs.items():
            nodes = subgraph[2]
            outdeg = {u: sum(1 for (u2,v2) in subgraph[1] if u2 == u) for u in nodes}
            indeg = {u: sum(1 for (u2,v2) in subgraph[1] if v2 == u) for u in nodes}

            for i in self.V:
                # OUT constraint
                cname_out = f"deg_out_k{k}_i{i}"
                self.cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[self.x_index[(i,u)] for u in nodes],
                        val=[-outdeg.get(u,0) for u in nodes]
                    )],
                    senses=["E"],
                    rhs=[0],
                    names=[cname_out]
                )
                self.degree_out_constraints[(k,i)] = cname_out

                # IN constraint
                cname_in = f"deg_in_k{k}_i{i}"
                self.cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[self.x_index[(i,u)] for u in nodes],
                        val=[-indeg.get(u,0) for u in nodes]
                    )],
                    senses=["E"],
                    rhs=[0],
                    names=[cname_in]
                    )
                self.degree_in_constraints[(k,i)] = cname_in
        
        # set LP method to 2
        self.cpx.parameters.lpmethod.set(2)  # use dual simplex for faster re-optimization
        # disable CPLEX output (we will print our own logs)self.cpx.set_log_stream(None)
        self.cpx.set_warning_stream(None)
        self.cpx.set_error_stream(None)
        self.cpx.set_results_stream(None)
        

    # ----------------------------------------------------------
    # ADD COLUMN (add variable + coefficients in constraints)
    # ----------------------------------------------------------
    def add_column(self, k, column):
        """
        Add column e^k_{ij} (k,i,j). The only rows where e^k_{ij} appears are:
        - for each arc (u,v) in A_k with i=φ(u), j=φ(v):
                deg_out  (k,u,i):  +1
                deg_in   (k,v,j):  +1
        """
        kk, i, j = column
        if i == j:
            return False  # skip self-loops
        assert kk == k, "Column subgraph_id does not match k"
        # First, we need to check whether the column is already present to avoid duplicates (can happen if pricing returns same pattern multiple times)
        if column in self.Omega[k]:
            return False  # skip adding duplicate column
        self.Omega[k].append(column)
        # add column variable to cplex
        name = f"e_{k}_{i}_{j}"
        self.cpx.variables.add(
            names=[name],
            lb=[0.0],
            ub=[1.0],
            types=[self.cpx.variables.type.continuous]
        )
        var_index = self.cpx.variables.get_indices(name)
        self.e_index[(k,i,j)] = var_index
        # add to objective with cost = F_k*D_{ij}
        self.cpx.objective.set_linear(var_index, self.subgraphs[k][0] * self.problem.D[i][j])
        
        # appears in deg_out constraint for (k,i,u)
        # sum_j e^k_{ij} = \sum_u outdeg_Gk(u) * x[i,u] forall i
        cname_out = self.degree_out_constraints[(k,i)]
        self.cpx.linear_constraints.set_coefficients(cname_out, var_index, 1.0)
        # appears in deg_in constraint for (k,j,v)
        # sum_i e^k_{ij} = \sum_v outdeg_Gk(v) * x[j,v] forall j,v
        cname_in = self.degree_in_constraints[(k,j)]
        self.cpx.linear_constraints.set_coefficients(cname_in, var_index, 1.0)
        
        return True
            
    
    # ----------------------------------------------------------
    # SOLVE RMP
    # ----------------------------------------------------------
    def solve(self):
        # set problem type to LP for pricing (columns are continuous), but branching will fix x to 0/1
        self.cpx.set_problem_type(self.cpx.problem_type.LP)
        self.cpx.solve()
        return self.cpx.solution.get_status()

    # ----------------------------------------------------------
    # GET DUALS (needed for pricing)
    # ----------------------------------------------------------
    def get_duals(self):
        lamb = {}   # duals of degree constraints out
        theta = {}  # duals of degree constraints in
        for (k,i), cons_name in self.degree_out_constraints.items():
            lamb[(k,i)] = self.cpx.solution.get_dual_values(self.cpx.linear_constraints.get_indices(cons_name))
        for (k,j), cons_name in self.degree_in_constraints.items():
            theta[(k,j)] = self.cpx.solution.get_dual_values(self.cpx.linear_constraints.get_indices(cons_name))
        return lamb, theta

    # ----------------------------------------------------------
    # GET CURRENT X SOLUTION
    # ----------------------------------------------------------
    def get_x_solution(self):
        xvals = {}
        sol = self.cpx.solution
        for (i,u), idx in self.x_index.items():
            xvals[(i,u)] = sol.get_values(idx)
        return xvals

    # ----------------------------------------------------------
    # BRANCHING SUPPORT
    # ----------------------------------------------------------
    def fix_x(self, i, u, value):
        """
        Force x[i,u] = value (0 or 1).
        Used by branching.
        """
        idx = self.x_index[(i,u)]
        if value == 1:
            self.cpx.variables.set_lower_bounds(idx, 1.0)
            self.cpx.variables.set_upper_bounds(idx, 1.0)
        else:
            self.cpx.variables.set_lower_bounds(idx, 0.0)
            self.cpx.variables.set_upper_bounds(idx, 0.0)

        self.branching_fixed[(i,u)] = value
    
    def check_solution(self):
        """
        Check that the current solution is integral in x and λ.
        Used for debugging.
        """
        sol = self.cpx.solution
        # print objective value
        print(f"Current solution value: {sol.get_objective_value()}")
        # for (i,u), idx in self.x_index.items():
        #     val = sol.get_values(idx)
        #     if abs(val) > 1e-5 and abs(val-1) > 1e-5:
        #         print(f"Non-integral x[{i},{u}] = {val}")
        count_nonzero = 0
        for (k, i, j), idx in self.e_index.items():
            val = sol.get_values(idx)
            if val > 1e-5:
                # print(f"Non-zero e[{k},{i},{j}] = {val}")
                count_nonzero += 1
        print(f"Total non-zero e: {count_nonzero}")
        
    def report_num_columns(self):
        total_cols = sum(len(cols) for cols in self.Omega.values())
        print(f"Total columns in RMP: {total_cols}")
        for k, cols in self.Omega.items():
            print(f"  Subgraph {k}: {len(cols)} columns")