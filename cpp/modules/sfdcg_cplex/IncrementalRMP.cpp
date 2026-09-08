#include "IncrementalRMP.h"

// constructor to initialize the RMP with the problem data and configuration
IncrementalRMP::IncrementalRMP(
    Problem& problem,
    SolverConfig& cfg,
    const std::vector<Subgraph>& subgraphs
)
    : problem(problem), cfg(cfg), subgraphs(subgraphs) {}

/**
 * Create the model with only x and all the rows, but no e columns; then we will add e^k_ij columns incrementally
 * Rows:
 *   C1(u):     sum_i x_iu   <= 1                          : alpha_u    (i in V, u in M)   (L)
 *   C2(i):     sum_u x_iu   == 1                          : beta_i     (i in V, u in M)   (E)
 *   C3(i,k):   sum_j e^k_ij = sum_u Degree_out^k_u * x_iu : lambda_ik  (i in V, k in subgraphs) (E)
 *   C4(i,k):   sum_i e^k_ij = sum_v Degree_in^k_v * x_jv  : theta_jv   (j in V, k in subgraphs) (E)
 *
 * Variables:
 *   - 0 <= e^k_ij  <= 1
 *   - x_{i,u} in [0,1]
 *
 * Objective:
 *   sum_kij F_k * d_ij * e^k_ij for e^k_ij in Omega
 */
void IncrementalRMP::createModel() {
    env = IloEnv();
    model = IloModel(env);
    cplex = IloCplex(model);
    objective = IloMinimize(env, 0.0);
    model.add(objective);
    int n = problem.n;
    int m = problem.m;


    e_vars = IloNumVarArray(env);
    model.add(e_vars);


    // create x_{i,u} variables
    x_vars = IloNumVarArray(env, n*m, 0.0, 1.0, ILOFLOAT);
    for (int i=0;i<n;++i) {
        for (int u=0;u<m;++u) {
            x_vars[i*m+u].setName(("x_" + std::to_string(i) + "_" + std::to_string(u)).c_str());
        }
    }

    // create rows C1 and C2
    for (int u=0;u<m;++u) {
        IloExpr expr(env);
        for (int i=0;i<n;++i) {
            expr += x_vars[i*m+u];
        }
        model.add(expr <= 1).setName(("C1_" + std::to_string(u)).c_str());
        expr.end();
    }
    for (int i=0;i<n;++i) {
        IloExpr expr(env);
        for (int u=0;u<m;++u) {
            expr += x_vars[i*m+u];
        }
        model.add(expr == 1).setName(("C2_" + std::to_string(i)).c_str());
        expr.end();
    }

    // create rows C3 and C4 with empty expression
    // create rows C3 and C4 with x terms already inside
    for (const auto& subgraph : subgraphs) {
        int k = subgraph.k;
        
        // ------- C3: sum_j e_kij  =  sum_u Degree_out[k][u] * x_iu -------
        for (int i = 0; i < n; ++i) {
            IloExpr expr(env);

            // RHS: sum_u Degree_out^k[u] * x_iu
            for (auto u: subgraph.nodes) { // only consider u in subgraph nodes since others have degree 0
                int w = subgraph.degree_out.count(u)? subgraph.degree_out.at(u): 0;
                if (w != 0)
                    expr += -w * x_vars[i*m + u];
            }

            // Build the range as equality:  expr == 0  (LHS empty, will add e later)
            IloRange c3(env, 0.0, expr, 0.0);

            c3.setName(("C3_" + std::to_string(k) + "_" + std::to_string(i)).c_str());
            model.add(c3);
            row_C3[{k, i}] = c3;

            expr.end();
        }

        // ------- C4: sum_i e_kij  =  sum_v Degree_in[k][v] * x_jv --------
        for (int j = 0; j < n; ++j) {
            IloExpr expr(env);

            // RHS: sum_v Degree_in^k[v] * x_jv
            for (auto v: subgraph.nodes) {
                double w = subgraph.degree_in.count(v)? subgraph.degree_in.at(v): 0;
                if (w != 0)
                    expr += -w * x_vars[j*m + v];
            }

            IloRange c4(env, 0.0, expr, 0.0);

            c4.setName(("C4_" + std::to_string(k) + "_" + std::to_string(j)).c_str());
            model.add(c4);
            row_C4[{k, j}] = c4;

            expr.end();
        }
    }


    // fix variable according to problem.fixed_assignments  // location -> facility
    for (const auto& [location, facility] : problem.fixed_assignments) {
        // fix x_{location, facility} = 1
        model.add(x_vars[location*m + facility] == 1);
    }
    // check all columns, add to not_in_omega if they can be
    // sort subgraphs by F_k in descending order to add more promising columns first
    // sort i,j by d_ij in ascending order to add more promising columns first
    auto subgraphs_sorted = subgraphs;
    std::sort(subgraphs_sorted.begin(), subgraphs_sorted.end(), [](const Subgraph& a, const Subgraph& b) {
        return a.F_k > b.F_k;
    });
    std::vector<PairKey> sorted_D;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sorted_D.emplace_back(i,j);
        }
    }
    std::sort(sorted_D.begin(), sorted_D.end(), [&](const PairKey& a, const PairKey& b) {
        return problem.D[a.a][a.b] < problem.D[b.a][b.b];
    });
    not_in_omega_pos.resize(subgraphs.size()*n*n, -1); // initialize all positions to -1 (not in not_in_omega)
    for (auto pair: sorted_D) {
        int i = pair.a;
        int j = pair.b;
        for (const auto& subgraph : subgraphs_sorted) {
            int k = subgraph.k;
            TripleKey key{k,i,j};
            if(canAddColumn(key)){
                auto id = k*n*n + i*n + j;
                not_in_omega_pos[id] = not_in_omega.size();
                not_in_omega.push_back(key);
            }
        }
    }
    omega.reserve(not_in_omega.size());
}

/**
 * Add columns for the given (k,i,j) keys;
 * - add triple key to Omega
 * - add the variable e^k_ij to the model, and update the corresponding C3 and C4 rows
 */
int IncrementalRMP::addColumns(const std::vector<TripleKey>& new_cols) {
    auto count = 0;
    for (const auto& col : new_cols) {
        auto added = addColumn(col);
        if (added) ++count;
    }
    return count;
}

/**
 * Add a single column for (k,i,j)
 */
bool IncrementalRMP::addColumn(const TripleKey& col) {
    auto k = col.a;
    auto i = col.b;
    auto j = col.c;
    auto n = problem.n;

    // ---- store column in omega ----
    int col_index = omega.size();
    omega.push_back(col);
    omega_index[col] = col_index;

    // ---- create variable ----
    IloNumVar e_var(env, 0.0, 1.0, ILOFLOAT);
    // Optional naming
    // e_var.setName(("e_"+std::to_string(k)+"_"+std::to_string(i)+"_"+std::to_string(j)).c_str());

    e_vars.add(e_var);

    // ---- C3 and C4 updates (VERY FAST NOW) ----
    row_C3[{k, i}].setLinearCoef(e_var, 1.0);
    row_C4[{k, j}].setLinearCoef(e_var, 1.0);

    // ---- objective ----
    double coeff = subgraphs[k].F_k * problem.D[i][j];
    objective.setLinearCoef(e_var, coeff);

    // ---- remove from unvisited ----
    removeColumn(col);   // O(1) if sparse-set

    return true;
}

void IncrementalRMP::removeColumn(const TripleKey& col) {
    auto idx = [&](int k, int i, int j) {
        return k*problem.n*problem.n + i*problem.n + j;
    };
    size_t id = idx(col.a, col.b, col.c);
    int p = not_in_omega_pos[id];
    if (p == -1) return;  // already removed

    int last = not_in_omega.size() - 1;

    // swap with last element
    std::swap(not_in_omega[p], not_in_omega[last]);

    // update the position of the moved element
    const TripleKey& moved = not_in_omega[p];
    size_t moved_id = idx(moved.a, moved.b, moved.c);
    not_in_omega_pos[moved_id] = p;

    // remove last
    not_in_omega.pop_back();
    not_in_omega_pos[id] = -1;
}

/**
 * Solve the LP RMP using primer dual method (2)
 */
bool IncrementalRMP::solve() {
    // Set LP method to Dual
    cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Algorithm::Dual);

    // Redirect CPLEX output to std::cout
    // cplex.setOut(std::cout);
    // cplex.setWarning(std::cout);
    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setError(env.getNullStream());

    // Solve
    // calculate and print solve time
    auto start = std::chrono::high_resolution_clock::now();
    bool ok = cplex.solve();
    auto end = std::chrono::high_resolution_clock::now();
    if(cfg.log_output) std::cout << " [RMP] CPLEX solve time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::chrono::duration<double> elapsed = end - start;
    if (!ok) {
        std::cout << " [RMP] CPLEX failed to solve. Status = "
                << cplex.getStatus() << std::endl;
        return ok;
    }


    if(cfg.log_output) {
        auto status = cplex.getStatus();
        std::cout << " [RMP] CPLEX status: " << status << std::endl;
        double obj = cplex.getObjValue();
        std::cout << " [RMP] Objective value = " << obj << std::endl;
        std::cout << " [RMP] Omega size " << omega.size() << std::endl;
    }
    return ok;
}

/**
 * fill the duals with lambda_ki then theta_kj
 * lambda_ki = duals[k*n + i]
 * theta_kj = duals[k*n+ j + n_subgraph*n]
 */
void IncrementalRMP::getDuals(std::vector<double>& duals) const {
    int n = problem.n;
    int K = subgraphs.size();
    duals.resize(2*n*K, 0);

    for (const auto& subgraph : subgraphs) {
        int k = subgraph.k;

        for (int i = 0; i < n; ++i) {
            // Dual of C3(k,i) → λ_{k,i}
            PairKey key3{k, i};
            duals[k*n + i] = cplex.getDual(row_C3.at(key3));
        }

        for (int j = 0; j < n; ++j) {
            // Dual of C4(k,j) → θ_{k,j}
            PairKey key4{k, j};
            duals[K*n + k*n + j] = cplex.getDual(row_C4.at(key4));
        }
    }

}

bool IncrementalRMP::canAddColumn(const TripleKey& col) const {
    int k = col.a;
    int i = col.b;
    int j = col.c;
    if (i==j || omega_index.count(col)) return false;

    return true;
}

// count the basis columns and positive columns in the current solution and print them (for debugging)
// if given a path, write all the basis columns to file

void IncrementalRMP::countBasisAndPositiveColumns(std::string basis_output_path) {
    const double POS_TOL = 1e-6;

    // Ensure solution is available
    if (!cplex.isPrimalFeasible()) {
        std::cerr << " [RMP][WARN] No primal feasible solution available. Skipping counts." << std::endl;
        return;
    }

    // Prepare output stream if needed
    std::ofstream out;
    if (!basis_output_path.empty()) {
        out.open(basis_output_path, std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << " [RMP][ERROR] Failed to open file for writing basis columns: "
                      << basis_output_path << std::endl;
            // Continue without writing to file
        }
    }

    // Bulk fetch basis statuses and values to avoid per-var calls
    IloEnv env = cplex.getEnv();
    IloCplex::BasisStatusArray statuses(env);
    IloNumArray values(env);

    try {
        cplex.getBasisStatuses(statuses, e_vars);  // statuses.size() == e_vars.getSize()
    } catch (const IloException& e) {
        std::cerr << " [RMP][ERROR] getBasisStatuses failed: " << e << std::endl;
        statuses.end();
        values.end();
        if (out.is_open()) out.close();
        return;
    }

    try {
        cplex.getValues(values, e_vars);           // values.size() == e_vars.getSize()
    } catch (const IloException& e) {
        std::cerr << " [RMP][ERROR] getValues failed: " << e << std::endl;
        statuses.end();
        values.end();
        if (out.is_open()) out.close();
        return;
    }

    int basis_count = 0;
    int positive_count = 0;

    // Iterate only over existing columns
    // omega_index: map<TripleKey,int>
    for (const auto& kv : omega_index) {
        const TripleKey& col = kv.first;
        const int col_index  = kv.second;

        // Bounds check (defensive)
        if (col_index < 0 || col_index >= e_vars.getSize()) {
            std::cerr << " [RMP][WARN] Column index out of bounds: " << col_index << std::endl;
            continue;
        }

        // Count basis vars
        const auto st = statuses[col_index];
        if (st == IloCplex::BasisStatus::Basic) {
            basis_count++;
        }

        // Count positive vars
        const double x = values[col_index];
        if (x > POS_TOL) {
            positive_count++;
            if (out.is_open()) {
                out << col.a << "," << col.b << "," << col.c << "\n";
            }
        }
    }

    // Print diagnostics
    double obj = IloInfinity;
    try {
        obj = cplex.getObjValue();
    } catch (const IloException&) {
        std::cerr << " [RMP][WARN] Unable to fetch objective value (not available)." << std::endl;
    }

    std::cout << " [RMP] Objective value = " << obj << std::endl;
    std::cout << " [RMP] Omega size = " << omega.size() << std::endl;
    std::cout << " [RMP] omega_index size = " << omega_index.size() << std::endl;
    std::cout << " [RMP] Basis columns: " << basis_count << std::endl;
    std::cout << " [RMP] Positive columns: " << positive_count << std::endl;

    // Cleanup
    if (out.is_open()) out.close();
    values.end();
    statuses.end();
}
