#include "ExactRefine.h"
#include <map>
#include <tuple>
#include <vector>
#include <set>
#include <string>

#define mu1(u) (pi[u])
#define mu2(i) (pi[n+i])
#define theta1(i, u, v) (pi[2*n + (i) * n * n + (u) * n + (v)])
#define theta2(i, u, j) (pi[2*n + n*n*n + (i) * n * n + (u) * n + (j)])
#define x(i, u) (psol[(i) * n + (u)])
#define y(i, u, j, v) (psol[n * n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])

// void ExactRefine::check_dual_warmstart() {
//     // Check if the warmstart vector
//     auto pi = warmstart;
//     auto n = problem.n;
//     auto m = problem.m;
//     // compute the dual objective value
//     double dual_obj = 0.0;
//     for (int i = 0; i < problem.n; ++i) {
//         dual_obj += mu1(i) + mu2(i);
//     }
//     std::cout << "[ExactRefine] Dual objective value = " << dual_obj << std::endl;

//     int violated_constraints = 0;
//     // check if the warmstart vector satisfies the constraints
//     //  mu1_i + mu2_u - sum_v theta1_{iuv} - sum_j theta2_{iuj} <= 0   \for all i in V, u in M
//     for (int i = 0; i < problem.n; ++i) {
//         for (int u = 0; u < problem.m; ++u) {
//             double lhs = mu1(i) + mu2(u);
//             for (int v = 0; v < problem.m; ++v) {
//                 if (u != v) {
//                     lhs -= theta1(i, u, v);
//                 }
//                 if (v != i) {
//                     lhs -= theta2(i, u, v);
//                 }
//             }
//             if (lhs > 1e-6) {
//                 violated_constraints++;
//             }
//         }
//     }
//     // theta2_{iuj} + theta2_{jvi} + theta1_{iuv} + theta1_{jvu} <= d_ij*f_uv + d_ji*f_vu for all i,j in V, u,v in M, i<j, u!=v
//     for (int i = 0; i < problem.n; ++i) {
//         for (int j = i + 1; j < problem.n; ++j) {
//             for (int u = 0; u < problem.m; ++u) {
//                 for (int v = 0; v < problem.m; ++v) {
//                     if (u != v) {
//                         double lhs = theta2(i, u, j) + theta2(j, v, i) + theta1(i, u, v) + theta1(j, v, u);
//                         double rhs = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
//                         if (lhs > rhs + 1e-6) {
//                             violated_constraints++;
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     std::cout << "[ExactRefine] Number of violated constraints = " << violated_constraints << std::endl;
// }

void ExactRefine::solve() {
    std::cout << "[ExactRefine] Solving the problem to optimality with CPLEX/Gurobi\n";
    int n = problem.n;
    int m = problem.m;

    try {
        // create a model
        GRBEnv env = GRBEnv();
        env.start();
        GRBModel model = GRBModel(env);
        model.set(GRB_IntParam_Method, GRB_METHOD_BARRIER);
        // model.set(GRB_IntParam_NodeMethod, GRB_METHOD_BARRIER);
        // set method to dual simplex
        // model.set(GRB_IntParam_Method, GRB_METHOD_DUAL);
        // model.set(GRB_IntParam_NodeMethod, GRB_METHOD_PRIMAL);
        // model.setObjective(objective, GRB_MINIMIZE);
        auto threshold = 0.001;
        auto count_fixed = 0;
        for (int i=0; i < n; ++i)
            for (int u=0; u < m; ++u)
                if (x(i, u) < threshold) {
                    count_fixed++;
                }
        std::cout << "[ExactRefine] Filtering " << count_fixed << " x variables with value < " << threshold << std::endl;
        create_model_RLT1_reduction(model, threshold);
        // create_model_RLT1(model, threshold);
        // create_model_SFD(model, threshold);
        model.update();
        

        // solve the model
        // print the objective value
        model.optimize();
    std::cout << "[ExactRefine] Optimal objective value = " << model.get(GRB_DoubleAttr_ObjVal) << ", status = " << model.get(GRB_IntAttr_Status) << std::endl;
    } catch (GRBException e) {
        std::cout << "[ExactRefine] Gurobi exception: " << e.getMessage() << std::endl;
    } catch (...) {
        std::cout << "[ExactRefine] Unknown exception during Gurobi optimization." << std::endl;
    }
}

void ExactRefine::create_model_RLT1(GRBModel& model, double theshold) {
    auto flows = problem.F;
    auto distances = problem.D;
    auto n = problem.n;
    auto m = problem.m;
    // variables 
    // 0 <= x_iu <= 1 for all i in V, u in M
    std::vector<GRBVar> x_vars(n * m);
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < m; ++u) {
            if (x(i, u) > theshold) {
                x_vars[i * m + u] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x_" + std::to_string(i) + "_" + std::to_string(u));
            }
        }
    }
        
    // y_iujv >= 0 for all i,j in V, u,v in M, i<j, u!=v
    std::map<std::tuple<int, int, int, int>, GRBVar> y_vars;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int u = 0; u < m; ++u) {
                for (int v = 0; v < m; ++v) {
                    if (u != v) {
                        if (x(i, u) > theshold && x(j, v) > theshold) 
                        {
                            auto cost = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                            auto y_var = model.addVar(0.0, GRB_INFINITY, cost, GRB_CONTINUOUS, "y_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j) + "_" + std::to_string(v));
                            y_vars[std::make_tuple(i, u, j, v)] = y_var;
                            y_vars[std::make_tuple(j, v, i, u)] = y_var; // add the symmetric variable
                        }
                    }
                }
            }
        }
    }
        

    // constraints
    std::vector<GRBConstr> constraints_mu1(n);
    // sum_u x_iu = 1 for all i in V
    for (int i = 0; i < n; ++i) {
        GRBLinExpr expr = 0;
        for (int u = 0; u < m; ++u) {
            if (x(i, u) > theshold) {
                expr += x_vars[i * m + u];
            }
        }
        constraints_mu1[i] = model.addConstr(expr == 1, "assign_" + std::to_string(i));
    }   

    std::vector<GRBConstr> constraints_mu2(m);
    // sum_i x_iu = 1 for all u in M
    for (int u = 0; u < m; ++u) {
        GRBLinExpr expr = 0;
        for (int i = 0; i < n; ++i) {
            if (x(i, u) > theshold) {
                expr += x_vars[i * m + u];
            }
        }
        constraints_mu2[u] = model.addConstr(expr == 1, "assign_" + std::to_string(u));
    }

    std::vector<GRBConstr> constraints_theta1(n * m * m);
    // sum_j y_iujv = x_iu for all i in V, u in M, v in M, u!=v
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < m; ++u) {
            if (x(i, u) > theshold) {
                for (int v = 0; v < m; ++v) {
                    if (u != v) {
                        GRBLinExpr expr = 0;
                        for (int j = 0; j < n; ++j) {
                            auto key = std::make_tuple(i, u, j, v);
                            if (y_vars.find(key) != y_vars.end()) {
                                expr += y_vars[key];
                            }
                        }
                        constraints_theta1[i * m * m + u * m + v] = model.addConstr(expr == x_vars[i * m + u], "flow_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
                    }
                }
            }
        }
    }

    std::vector<GRBConstr> constraints_theta2(n * m * n);
    // sum_v y_iujv = x_iu for all i in V, u in M, j in V, j!=i
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < m; ++u) {
            if (x(i, u) > theshold) {
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        GRBLinExpr expr = 0;
                        for (int v = 0; v < m; ++v) {
                            if (u != v) {
                                auto key = std::make_tuple(i, u, j, v);
                                if (y_vars.find(key) != y_vars.end()) {
                                    expr += y_vars[key];
                                }
                            }
                        }
                        constraints_theta2[i * m * n + u * n + j] = model.addConstr(expr == x_vars[i * m + u], "flow_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j));
                    }
                }
            }
        }
    }

}


void ExactRefine::create_model_RLT1_reduction(GRBModel& model, double theshold) {
    auto flows = problem.F;
    auto n = problem.n;
    auto m = problem.m;

    // -------------------------------------------------------------
    // Compute F0
    // -------------------------------------------------------------
    std::vector<std::pair<int,int>> F0;
    std::vector<std::pair<int,int>> Fn0;
    std::vector<std::pair<int,int>> Fn0_2;

    for (auto u=0; u < m; ++u) {
        for (auto v=0; v < m; ++v) {
            if (flows[u][v] + flows[v][u] > 0.) {
                Fn0_2.push_back(std::make_pair(u, v));
            }
            if (v > u) continue; // only consider pairs (u,v) with u < v to avoid duplicates
            if (flows[u][v] + flows[v][u] > 0) {
                Fn0.push_back(std::make_pair(u, v));
            } else {
                F0.push_back(std::make_pair(u, v));
            }
        }
    }

    // -------------------------------------------------------------
    // Compute I0
    // -------------------------------------------------------------
    std::vector<int> I0;
    std::vector<int> In0;

    for (auto u=0; u < m; ++u) {
        bool found = false;

        for (auto v=0; v < m; ++v) {
            if (v != u && flows[u][v]+flows[v][u] == 0) {
                found = true;
                break;
            }
        }

        if (found)
            I0.push_back(u);
        else
            In0.push_back(u);
    }

    // variables 
    // 0 <= x_iu <= 1 for all i in V, u in M
    std::vector<GRBVar> x_vars(n * m);
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < m; ++u) {
            if (x(i, u) < theshold) continue;
            x_vars[i * m + u] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x_" + std::to_string(i) + "_" + std::to_string(u));
        }
    }
        
    // y_iujv >= 0 for all i,j in V, u,v in M, i<j, u!=v
    std::map<std::tuple<int, int, int, int>, GRBVar> y_vars;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            for (auto pair : Fn0) {
                auto u = pair.first;
                auto v = pair.second;
                if (u == v) continue;
                if (x(i, u) < theshold || x(j, v) < theshold) continue;
                auto cost = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                auto y_var = model.addVar(0.0, GRB_INFINITY, cost, GRB_CONTINUOUS, "y_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j) + "_" + std::to_string(v));
                y_vars[std::make_tuple(i, u, j, v)] = y_var;
                y_vars[std::make_tuple(j, v, i, u)] = y_var; // add the symmetric variable
            }
        }
    }
        

    // constraints
    // sum_u x_iu = 1 for all i in V
    for (int i = 0; i < n; ++i) {
        GRBLinExpr expr = 0;
        for (int u = 0; u < m; ++u) {
            if (x(i, u) < theshold) continue;
            expr += x_vars[i * m + u];
        }
        model.addConstr(expr == 1, "assign_" + std::to_string(i));
    }   

    // sum_i x_iu = 1 for all u in M
    for (int u = 0; u < m; ++u) {
        GRBLinExpr expr = 0;
        for (int i = 0; i < n; ++i) {
            if (x(i, u) < theshold) continue;
            expr += x_vars[i * m + u];
        }
        model.addConstr(expr == 1, "assign_" + std::to_string(u));
    }

    // sum_j y_iujv = x_iu for all i in V, u in M, v in M, u!=v
    for (int i = 0; i < n; ++i) {
        for (auto pair: Fn0_2) {
            auto u = pair.first;
            auto v = pair.second;
            if (u == v) continue;
            if (x(i, u) < theshold) continue;
            GRBLinExpr expr = 0;
            for (int j = 0; j < n; ++j) {
                auto key = std::make_tuple(i, u, j, v);
                if (y_vars.find(key) != y_vars.end()) {
                    expr += y_vars[key];
                }
            }
            model.addConstr(expr == x_vars[i * m + u], "flow_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
        }
    }

    // sum_v y_iujv = x_iu for all i in V, u in M, j in V, j!=i
    for (int i = 0; i < n; ++i) {
        for (auto u : In0) {
            if (x(i, u) < theshold) continue;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                GRBLinExpr expr = 0;
                for (int v = 0; v < m; ++v) {
                    if (u != v) {
                        auto key = std::make_tuple(i, u, j, v);
                        if (y_vars.find(key) != y_vars.end()) {
                            expr += y_vars[key];
                        }
                    }
                }
                model.addConstr(expr == x_vars[i * m + u], "flow_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j));
                
            }
        }
    }
    // fixed variables
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        model.addConstr(x_vars[i * m + u] == 1);
        std::cout << "[ExactRefine] Fixed variable: x[" << i << "][" << u << "] = 1" << std::endl;
    }

}


struct SubgraphData {
    double f_k;
    std::vector<std::pair<int,int>> arcs;
    std::set<int> nodes;
};


std::map<int, SubgraphData> decomposeFlowgraphValueOnly(const Problem& problem) {
    std::map<int, SubgraphData> subgraphs;

    std::map<double, std::vector<std::pair<int,int>>> flow_to_arcs;
    for (int u = 0; u < problem.m; ++u) {
        for (int v = 0; v < problem.m; ++v) {
            if (u == v) continue;
            double flow = problem.F[u][v];
            //  if flow is not in flow_to_arcs, it will be default-initialized to empty vector
            flow_to_arcs[flow].emplace_back(u, v);
        }
    }
    int k = 0;
    for (const auto& kv : flow_to_arcs) {
        double f_k = kv.first;
        const auto& arcs = kv.second;
        std::set<int> nodes;
        for (const auto& uv : arcs) {
            nodes.insert(uv.first);
            nodes.insert(uv.second);
        }
        subgraphs.insert({k, {f_k, arcs, nodes}});
        ++k;
    }
    // for (const auto& kv : subgraphs) {
    //     int k = kv.first;
    //     const auto& G_k = kv.second;
    //     std::cout << "Subgraph " << k << ": flow=" << G_k.f_k
    //               << ", arcs=" << G_k.arcs.size()
    //               << ", nodes=" << G_k.nodes.size() << std::endl;
    // }
    return subgraphs;
}

void ExactRefine::create_model_SFD(GRBModel& model, double threshold) {
    auto m = problem.m;
    auto n = problem.n;
    auto flows = problem.F;
    auto distances = problem.D;
    auto subgraphs = decomposeFlowgraphValueOnly(problem);
    // -------------------------------------------------------------
    // x variables
    // -------------------------------------------------------------
    std::map<std::pair<int,int>, GRBVar> x_vars;

    for (auto i=0; i < n; ++i) {
        for (auto u=0; u < m; ++u) {
            x_vars[std::make_pair(i, u)] = model.addVar(
                0.0,
                1.0,
                0.0,
                GRB_BINARY,
                "x_" + std::to_string(i) + "_" + std::to_string(u)
            );
        }
    }
    // -------------------------------------------------------------
    // e variables (subgraph edge variables) and objective
    // -------------------------------------------------------------
    // detect symmetry of flow and distance matrices
    bool is_symmetric = true;
    for (int u = 0; u < m && is_symmetric; ++u) {
        for (int v = 0; v < m; ++v) {
            if (std::fabs(flows[u][v] - flows[v][u]) > 1e-9) { is_symmetric = false; break; }
        }
    }
    for (int i = 0; i < n && is_symmetric; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::fabs(distances[i][j] - distances[j][i]) > 1e-9) { is_symmetric = false; break; }
        }
    }

    // create reduced e vars when symmetric (only i<j), then map both directions to same var
    std::map<std::tuple<int,int,int>, GRBVar> e_temp; // temporary: only created keys
    for (const auto &kv : subgraphs) {
        int k = kv.first;
        for (int i=0; i < n; ++i) {
            for (int j=0; j < n; ++j) {
                if (i == j) continue;
                if (is_symmetric && j <= i) continue; // create only for i<j
                std::string name = "e_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j);
                e_temp[std::make_tuple(k,i,j)] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, name);
            }
        }
    }

    // final e map: if symmetric, map both (i,j) and (j,i) to same var
    std::map<std::tuple<int,int,int>, GRBVar> e;
    if (is_symmetric) {
        for (const auto &kv : e_temp) {
            int k,i,j; std::tie(k,i,j) = kv.first;
            GRBVar var = kv.second;
            e[std::make_tuple(k,i,j)] = var;
            e[std::make_tuple(k,j,i)] = var; // symmetry mapping
        }
    } else {
        e = e_temp;
    }

    // Objective as linear combination: sum_{k,i,j} D[i][j] * f_k * e_{k,i,j}
    GRBLinExpr obj = 0;
    for (const auto &kv : e) {
        int k,i,j; std::tie(k,i,j) = kv.first;
        auto f_k = subgraphs[k].f_k;
        // if (f_k < 1e-9) continue; // skip zero-flow subgraph
        GRBVar var = kv.second;
        obj += distances[i][j] * f_k * var;
    }
    model.setObjective(obj, GRB_MINIMIZE);

    // -------------------------------------------------------------
    // Assignment constraints
    // -------------------------------------------------------------
    for (auto i=0; i < n; ++i) {

        GRBLinExpr expr = 0;

        for (auto u=0; u < m; ++u)
            expr += x_vars[std::make_pair(i, u)];

        model.addConstr(
            expr == 1,
            "assign_fac_" + std::to_string(i)
        );
    }

    for (auto u=0; u < m; ++u) {

        GRBLinExpr expr = 0;

        for (auto i=0; i < n; ++i)
            expr += x_vars[std::make_pair(i, u)];

        model.addConstr(
            expr == 1,
            "assign_loc_" + std::to_string(u)
        );
    }

    // -------------------------------------------------------------
    // Subgraph constraints and flow conservation (using e variables)
    // -------------------------------------------------------------

    // sum_j e^k_ij == sum_u x_iu * degree_out(u) for all i in V, for all k
    // (and similarly for incoming if not symmetric)
    int total_subgraph_constraints = 0;
    for (const auto &kv : subgraphs) {
        int k = kv.first;
        auto G_k = kv.second;

        // compute degree sequences
        std::map<int,int> degree_out, degree_in;
        for (auto &uv : G_k.arcs) { degree_out[uv.first]++; degree_in[uv.second]++; }

        // flow conservation constraints
        for (int i=0; i < n; ++i) {
            GRBLinExpr lhs_out = 0;
            for (int j=0; j < n; ++j) if (i!=j) {
                auto key = std::make_tuple(k,i,j);
                lhs_out += e[key];
            }
            GRBLinExpr rhs_out = 0;
            for (int u=0; u < m; ++u) rhs_out += x_vars[std::make_pair(i,u)] * degree_out[u];
            model.addConstr(lhs_out == rhs_out, "flow_out_" + std::to_string(i) + "_" + std::to_string(k));

            if (!is_symmetric) {
                GRBLinExpr lhs_in = 0;
                for (int j=0; j < n; ++j) if (i!=j) {
                    auto key = std::make_tuple(k,j,i);
                    lhs_in += e[key];
                }
                GRBLinExpr rhs_in = 0;
                for (int u=0; u < m; ++u) rhs_in += x_vars[std::make_pair(i,u)] * degree_in[u];
                model.addConstr(lhs_in == rhs_in, "flow_in_" + std::to_string(i) + "_" + std::to_string(k));
            }
        }

        {
            for (int i=0; i < n; ++i) {
                for (int j=0; j < n; ++j) {
                    if (i==j) continue;
                    auto key_e = std::make_tuple(k,i,j);
                    for (int u=0; u < m; ++u) {
                        {
                            GRBLinExpr rhs = -1 + x_vars[std::make_pair(i,u)];
                            for (int v=0; v < m; ++v) {
                                if (std::find(G_k.arcs.begin(), G_k.arcs.end(), std::make_pair(u,v)) != G_k.arcs.end()) {
                                    rhs += x_vars[std::make_pair(j,v)];
                                }
                            }
                            model.addConstr(e[key_e] >= rhs, "edge_link_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(k));
                            total_subgraph_constraints++;
                        }
                        if(0)
                        {
                            GRBLinExpr rhs = 1 - x_vars[std::make_pair(i,u)];
                            for (int v=0; v < m; ++v) {
                                if (std::find(G_k.arcs.begin(), G_k.arcs.end(), std::make_pair(u,v)) != G_k.arcs.end()) {
                                    rhs += x_vars[std::make_pair(j,v)];
                                }
                            }
                            model.addConstr(e[key_e] <= rhs, "edge_link2_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(k));
                            total_subgraph_constraints++;
                        }
                    }
                }
            }
        }
    }
    std::cout << "Added " << total_subgraph_constraints << " subgraph constraints directly to the model." << std::endl;

    // sum_k e^k_ij == 1 for all i,j in V
    for (int i=0; i < n; ++i) {
        for (int j=0; j < n; ++j) {
            if (i == j) continue;
            GRBLinExpr lhs = 0;
            for (const auto &kv : subgraphs) {
                lhs += e[std::make_tuple(kv.first,i,j)];
            }
            model.addConstr(lhs == 1, "flow_value_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    // -------------------------------------------------------------
    // Fixed assignments
    // -------------------------------------------------------------
    if (!problem.fixed_assignments.empty()) {

        std::cout << "Adding "
                << problem.fixed_assignments.size()
                << " fixed assignment constraints"
                << std::endl;

        for (const auto& kv : problem.fixed_assignments) {

            int i = kv.first;
            int u = kv.second;

            model.addConstr(
                x_vars[std::make_pair(i, u)] == 1,
                "fixed_" +
                std::to_string(i) + "_" +
                std::to_string(u)
            );

            std::cout << "Fixed assignment: x["
                    << i << "," << u << "] = 1"
                    << std::endl;
        }
    }
    auto count_fixed = 0;
    for (int i=0; i < n; ++i)
        for (int u=0; u < m; ++u)
            if (x(i, u) < threshold) {
                // set the variable to 0 and fix it
                model.addConstr(x_vars[std::make_pair(i, u)] == 0);
                count_fixed++;
            }
    std::cout << "Fixed " << count_fixed << " x variables to 0 based on threshold " << threshold << std::endl;

    // lower bound constraint
    // std::cout << "[ExactRefine] Adding lower bound constraint: objective >= " << LB << std::endl;
    // model.addConstr(obj >= LB, "lower_bound");

    // if(cfg.use_lazy_constraints)
    //     model.set(GRB_IntParam_LazyConstraints, 1);
    model.update();
}
