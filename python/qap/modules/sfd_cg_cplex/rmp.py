import cplex
from .column import Column
import numpy as np
np.float_ = np.float64
from docplex.mp.model import Model

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
        self.x_index = {}       # (i,u) → var index
        self.lambda_index = {}    # (k, col_id) → var index
        self.columns = {k: [] for k in subgraphs}
        self.alpha_constraints = {}   # k → constraint index
        self.beta_constraints = {}  # (i,u) → constraint index
        self.branching_fixed = {}     # (i,u) → {0 or 1}

        self._build_base_model()

    # ----------------------------------------------------------
    # BUILD BASE MODEL (only x + subgraph constraints + degree eq.)
    # ----------------------------------------------------------
    def _build_base_model(self):
        V, M = self.V, self.M
        n, m = self.problem.n, self.problem.m
        # x[i,u] = 1 if machine u assigned to location i
        for i in V:
            for u in M:
                name = f"x_{i}_{u}"
                self.cpx.variables.add(
                    names=[name], obj=[0.0],
                    lb=[0.0], ub=[1.0], types=[self.cpx.variables.type.continuous]
                )
                var_idx = self.cpx.variables.get_num() - 1
                self.x_index[(i,u)] = var_idx
        
        # --------------------------
        # One-pattern-per-subgraph constraints
        # sum_{φ ∈ P_k} λ_{k,φ} = 1
        # --------------------------
        # One-pattern-per-subgraph constraint: sum_p λ_{k,p} = 1
        for k in self.subgraphs:
            cname = f"subgraph_{k}_one_pattern"
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[], val=[])],
                senses=["E"],
                rhs=[1],
                names=[cname]
            )
            self.alpha_constraints[k] = cname
        # --------------------------
        # Subgraph agreement constraints
        # sum_{k \in S_u} sum_{p\in P_k} a_iu^pk λ_{k,p} = |S_u|x_iu for each (i,u)
        # --------------------------
        # For each machine u, we have a constraint linking x[i,u] to the patterns that assign machine u to some location
        for u in M:
            for i in V:
                cname = f"machine_{u}_loc_{i}_agreement"
                self.cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[self.x_index[(i, u)]], val=[-len([k for k in self.subgraphs if u in self.subgraphs[k][2]])])],
                    senses=["E"],
                    rhs=[0],
                    names=[cname]
                )
                self.beta_constraints[i,u] = cname

        # --------------------------
        # Assignment constraints
        # --------------------------
        # Each machine assigned to exactly one location: sum_i x[i,u] = 1 for each u
        for u in M:
            cname = f"assign_machine_{u}"
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[self.x_index[(i, u)] for i in V], val=[1.0] * len(V))],
                senses=["E"],
                rhs=[1],
                names=[cname]
            )
        # Each location assigned to exactly one machine: sum_u x[i,u] = 1 for each i
        for i in V:
            cname = f"assign_loc_{i}"
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[self.x_index[(i, u)] for u in M], val=[1.0] * len(M))],
                senses=["E"],
                rhs=[1],
                names=[cname]
            )
        


    # ----------------------------------------------------------
    # ADD COLUMN (pattern φ for subgraph k)
    # ----------------------------------------------------------
    def add_column(self, k, column):
        """
        Add λ_{k,φ}. The only rows where λ appears are:
        - subgraph k one-pattern constraint (coefficient 1)
        - for each (i,u) in φ, the machine-location agreement constraint (coefficient 1)
        """
        # print(f"Adding column for subgraph {k} with cost {column.cost} and pattern {column.phi_map}")
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
        self._add_to_constraint(self.alpha_constraints[k], lam_idx, 1.0)

        # assignment constraints
        # for each assignment i,u in column, we add +1 to the corresponding machine-location agreement constraint
        for u,i in column.phi_map.items():
            self._add_to_constraint(self.beta_constraints[(i, u)], lam_idx, 1.0)
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
        # export lp file for debugging
        self.cpx.write("rmp.lp")
        return self.cpx.solution.get_status()

    # ----------------------------------------------------------
    # GET DUALS (needed for pricing)
    # ----------------------------------------------------------
    def get_duals(self):
        beta = {}     # duals of x-assignment constraints
        alpha = {}  # duals of subgraph constraints

        for k, idx in self.alpha_constraints.items():
            alpha[k] = self.cpx.solution.get_dual_values(idx)
        for (i,u), idx in self.beta_constraints.items():
            beta[i,u] = self.cpx.solution.get_dual_values(idx)

        return alpha, beta

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
        # check assignment constraints
        for u in self.M:
            assign_sum = sum(sol.get_values(self.x_index[(i, u)]) for i in self.V)
            if abs(assign_sum - 1.0) > 1e-5:
                print(f"Assignment constraint violated for machine {u}: sum_i x[i,{u}] = {assign_sum}")
        for i in self.V:
            assign_sum = sum(sol.get_values(self.x_index[(i, u)]) for u in self.M)
            if abs(assign_sum - 1.0) > 1e-5:
                print(f"Assignment constraint violated for location {i}: sum_u x[{i},u] = {assign_sum}")
        # check subgraph constraints \sum_p λ_{k,p} = 1
        for k in self.subgraphs:
            lambda_sum = sum(sol.get_values(self.lambda_index[(k, col_id)]) for col_id in range(len(self.columns[k])))
            if abs(lambda_sum - 1.0) > 1e-5:
                print(f"Subgraph constraint violated for subgraph {k}: sum_p λ[{k},p] = {lambda_sum}")
        # check machine-location agreement constraints
        for (i,u), idx in self.beta_constraints.items():
            lhs = sum(sol.get_values(self.lambda_index[(k, col_id)]) for k in self.subgraphs for col_id, col in enumerate(self.columns[k]) if (u in col.phi_map and col.phi_map[u] == i))
            rhs = len([k for k in self.subgraphs if u in self.subgraphs[k][2]]) * sol.get_values(self.x_index[(i, u)])
            if abs(lhs - rhs) > 1e-5:
                print(f"Machine-location agreement constraint violated for (i={i}, u={u}): LHS={lhs}, RHS={rhs}")
        
        # for (i,u), idx in self.x_index.items():
        #     val = sol.get_values(idx)
        #     if abs(val) > 1e-5:
        #         print(f"Positive x[{i},{u}] = {val}")
        # count_nonzero = 0
        # for (k, col_id), idx in self.lambda_index.items():
        #     val = sol.get_values(idx)
        #     if val > 1e-5:
        #         print(f"Non-zero λ[{k},{col_id},{self.subgraphs[k][0]}] = {val}")
        #         count_nonzero += 1
        # print(f"Total non-zero λ: {count_nonzero}")
        
    def report_num_columns(self):
        total_cols = sum(len(cols) for cols in self.columns.values())
        # print(f"Total columns in RMP: {total_cols}")
        # for k, cols in self.columns.items():
        #     print(f"  Subgraph {k}: {len(cols)} columns")
            # for col_id, col in enumerate(cols):
            #     print(f"    Column {col_id}: cost={col.cost}, φ_map={col.phi_map}")
            
    def solve_integral(self):
        # create a copy of the model with integer constraints on λ and x
        int_cpx = self.cpx
        for (k, col_id), idx in self.lambda_index.items():
            int_cpx.variables.set_types(idx, int_cpx.variables.type.binary)
        for (i, u), idx in self.x_index.items():
            int_cpx.variables.set_types(idx, int_cpx.variables.type.binary)
        int_cpx.solve()
        print(f"Integral solution status: {int_cpx.solution.get_status()}, value: {int_cpx.solution.get_objective_value()}")
        return int_cpx.solution.get_status(), int_cpx.solution.get_objective_value()
    
                