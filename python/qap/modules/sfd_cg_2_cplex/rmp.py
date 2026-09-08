import cplex
from .column import Column

class RMP:
    """
    Restricted Master Problem for SFD Branch-and-Price.
    Holds:
      - global assignment variables x[i,u]
      - one pattern-selection constraint per subgraph
      - columns λ_{k,φ}
      - degree-equality constraints linking x and λ
      - uplift constraints linking x and λ
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
        self.lambda_index = {}    # (k, col_id) → var index
        self.columns = {k: [] for k in subgraphs}
        self.subgraph_constraints = {}   # subgraph → constraint index
        self.degree_constraints = {}  # (k,u,i,'out'/'in') → constraint index
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
        # One-pattern-per-subgraph constraints
        # sum_{φ ∈ P_k} λ_{k,φ} = 1
        # --------------------------
        # One-pattern-per-subgraph constraint: sum λ = 1
        for k in self.subgraphs:
            cname = f"subgraph_{k}_one_pattern"
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[], val=[])],
                senses=["E"],
                rhs=[1],
                names=[cname]
            )
            self.subgraph_constraints[k] = cname

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
        # sum_j e^k_{ij} = outdeg_Gk(u) * x[i,u]
        # sum_j e^k_{ji} = indeg_Gk(v) * x[i,v]
        # Left side added when columns appear
        # --------------------------
        for k, subgraph in self.subgraphs.items():
            nodes = subgraph[2]
            outdeg = {u: sum(1 for (u2,v2) in subgraph[1] if u2 == u) for u in nodes}
            indeg = {u: sum(1 for (u2,v2) in subgraph[1] if v2 == u) for u in nodes}

            for u in nodes:
                for i in self.V:

                    # OUT constraint
                    cname_out = f"deg_out_k{k}_u{u}_i{i}"
                    self.cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[self.x_index[(i,u)]],
                            val=[-outdeg[u]]
                        )],
                        senses=["E"],
                        rhs=[0],
                        names=[cname_out]
                    )
                    self.degree_constraints[(k,u,i,'out')] = cname_out

                    # IN constraint
                    cname_in = f"deg_in_k{k}_u{u}_i{i}"
                    self.cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[self.x_index[(i,u)]],
                            val=[-indeg[u]]
                        )],
                        senses=["E"],
                        rhs=[0],
                        names=[cname_in]
                    )
                    self.degree_constraints[(k,u,i,'in')] = cname_in

    # ----------------------------------------------------------
    # ADD COLUMN (pattern φ for subgraph k)
    # ----------------------------------------------------------
    def add_column(self, k, column):
        """
        Add λ_{k,φ}. The only rows where λ appears are:
        - subgraph one-pattern row:  +1
        - for each arc (u,v) in A_k with i=φ(u), j=φ(v):
                deg_out  (k,u,i):  +1
                deg_in   (k,v,j):  +1
        """
        col_id = len(self.columns[k])
        # First, we need to check whether the column is already present to avoid duplicates (can happen if pricing returns same pattern multiple times)
        for existing_col in self.columns[k]:
            if existing_col.phi_map == column.phi_map:
                # Column already exists, skip adding
                return False
        self.columns[k].append(column)

        name = f"lambda_{k}_{col_id}"
        obj = self.subgraphs[k][0] * column.cost

        # create λ variable
        self.cpx.variables.add(
            names=[name], obj=[obj],
            lb=[0.0], ub=[1.0], types=[self.cpx.variables.type.continuous]
        )
        lam_idx = self.cpx.variables.get_num() - 1
        self.lambda_index[(k, col_id)] = lam_idx

        # subgraph one-pattern constraint
        self._add_to_constraint(self.subgraph_constraints[k], lam_idx, 1.0)

        # precise degree equalities: per arc (u,v) only
        arcs = self.subgraphs[k][1]
        phi  = column.phi_map

        for (u, v) in arcs:
            i = phi[u]
            j = phi[v]

            # out-degree of u at location i
            cname_out = self.degree_constraints[(k, u, i, 'out')]
            self._add_to_constraint(cname_out, lam_idx, 1.0)

            # in-degree of v at location j
            cname_in  = self.degree_constraints[(k, v, j, 'in')]
            self._add_to_constraint(cname_in,  lam_idx, 1.0)
        return True
            
    
    def _add_to_constraint(self, cons_name, var_index, value):
        cons_index = self.cpx.linear_constraints.get_indices(cons_name)

        # Read existing row
        row = self.cpx.linear_constraints.get_rows(cons_index)
        old_ind = list(row.ind)
        old_val = list(row.val)

        # MERGE WITH NEW ENTRY
        coef_map = {}

        # keep old coefficients
        for i, idx in enumerate(old_ind):
            coef_map[idx] = old_val[i]

        # add new (var_index, value)
        if var_index in coef_map:
            coef_map[var_index] += value
        else:
            coef_map[var_index] = value

        # build merged lists
        new_ind = list(coef_map.keys())
        new_val = list(coef_map.values())

        # Update constraint row SAFELY
        self.cpx.linear_constraints.set_linear_components(
            cons_index,
            cplex.SparsePair(new_ind, new_val)
        )

    # Generate required degree-constraint couplings for e^k_{ij}=1
    def _affected_degree_constraints(self, k, i, j):
        """
        Return [(u,i2,"out"/"in"), ...]
        specifying which degree eqs depend on e^k_{ij}.
        """
        out_list = []
        for u in self.subgraphs[k][2]:
            # if φ(u)=i => contributes to out-degree at location i
            out_list.append((u, i, 'out'))
            # if φ(u)=j => contributes to in-degree at location j
            out_list.append((u, j, 'in'))
        return out_list

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
        pi = {}     # duals of x-assignment constraints
        alpha = {}  # duals of subgraph constraints

        for k, idx in self.subgraph_constraints.items():
            alpha[k] = self.cpx.solution.get_dual_values(idx)

        # assignment constraints duals
        for i in self.V:
            idx = self.cpx.linear_constraints.get_indices(f"assign_loc_{i}")
            pi_loc = self.cpx.solution.get_dual_values(idx)

            for u in self.M:
                pi[(i,u)] = pi_loc  # symmetric, both loc/machine constraints contribute

        return pi, alpha

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
        for (k, col_id), idx in self.lambda_index.items():
            val = sol.get_values(idx)
            if val > 1e-5:
                print(f"Non-zero λ[{k},{col_id},{self.subgraphs[k][0]}] = {val}")
                count_nonzero += 1
        print(f"Total non-zero λ: {count_nonzero}")
        
    def report_num_columns(self):
        total_cols = sum(len(cols) for cols in self.columns.values())
        print(f"Total columns in RMP: {total_cols}")
        for k, cols in self.columns.items():
            print(f"  Subgraph {k}: {len(cols)} columns")