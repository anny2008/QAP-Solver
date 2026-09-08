#include "ColumnGenSolver.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "../local_search/local_search.h"
#include "gurobi_c++.h"

ColumnGenSolver::ColumnGenSolver(const ColumnGenConfig& cfg_) : cfg(cfg_) {}

std::vector<double> ColumnGenSolver::buildPhi(const Problem& P) {
    int n = P.n;
    int m = P.n; // QAP: |V|==|M| == n
    std::cout << "[CG] Building phi matrix for n=" << n << ", m=" << m << "\n";
    std::vector<double> phi(((n*m)*n*m), 0.0);
    for (int u=0;u<m;++u) for (int v=0;v<m;++v) {
        if (P.F[u][v] == 0.0) {
            continue;
        }
        for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
            // D,F are P.D,P.F
            phi[(((i*m)+u)*n + j)*m + v] = P.D[i][j] * P.F[u][v];
        }
    }
    return phi;
}

std::vector<QuadKey> ColumnGenSolver::buildInitialOmega(
    const Problem& P, const std::unordered_map<int,int>& fixed)
{
    int n = P.n, m = P.n;
    std::vector<QuadKey> out;
    for (int i=0;i<n;++i) {
        for (int u=0;u<m;++u) {
            if (fixed.count(i) && fixed.at(i)!=u) continue;
            for (int j=i + 1;j<n;++j) {
                for (int v=0;v<m;++v) {
                    if(fixed.count(j) && fixed.at(j)!=v) continue;
                    if(u == v) continue; // no y-columns for u==v
                    // if (P.F[u][v] > 0.0 || P.F[v][u] > 0.0) 
                    {
                        out.push_back(IncrementalRMP::canonical(i,u,j,v));
                    }
                }
            }
        }
    }
    // deduplicate canonicals
    std::sort(out.begin(), out.end(), [](const QuadKey& a, const QuadKey& b){
        if (a.i!=b.i) return a.i<b.i; if (a.u!=b.u) return a.u<b.u;
        if (a.j!=b.j) return a.j<b.j; return a.v<b.v;
    });
    out.erase(std::unique(out.begin(), out.end(),
        [](const QuadKey& a, const QuadKey& b){
            return a.i==b.i && a.u==b.u && a.j==b.j && a.v==b.v;
        }), out.end());
    return out;
}

// std::vector<int> ColumnGenSolver::argmaxAssignment(
//     int n, int m, const std::unordered_map<PairKey,double,PairKeyHash>& xvals)
// {}
// {
//     std::vector<int> assign(n, 0);
//     for (int i=0;i<n;++i) {
//         double best = -1.0; int bestu = 0;
//         for (int u=0; u<m; ++u) {
//             auto it = xvals.find(PairKey{i,u});
//             double val = (it==xvals.end()) ? 0.0 : it->second;
//             if (val > best) { best = val; bestu = u; }
//         }
//         assign[i] = bestu;
//     }
//     return assign;
// }

void find_a_valid_dual_solution1(const Problem& problem, DualVector& duals) {
    // dualc4_{iuv} = \min_{j} d_{ij}f_{uv}
    // dualc3_{iuj} = 0
    int n = problem.n;
    int m = problem.m;
    duals.resize(n,m);
    duals.c3.assign(n*m*n, 0.0);
    duals.c4.assign(n*m*m, 0.0);
    int count_nonzero = 0;
    for (int i=0;i<n;++i) for (int u=0;u<m;++u)  for (int v=0;v<m;++v) {
        if (u==v) continue;
        double min_dual = std::numeric_limits<double>::infinity();
        for (int j=0;j<n;++j) {
            if (i==j) continue;
            double dual = problem.D[i][j] * problem.F[u][v];
            if (dual < min_dual) min_dual = dual;
        }
        duals.c4[(i * m + u) * m + v] = min_dual;
        if (min_dual > 1e-6) count_nonzero++;
    }
    std::cout << "[CG] Number of non-zero delta variables: " << count_nonzero << "\n";
    // \bar{\beta}_u = \min_{i} \sum_v \bar{\delta}_{iuv}
    std::vector<double> beta(m, std::numeric_limits<double>::infinity());
    for (int u=0;u<m;++u) {
        for (int i=0;i<n;++i) {
            double sum_dual = 0.0;
            for (int v=0;v<m;++v) {
                if (u==v) continue;
                sum_dual += duals.c4[(i * m + u) * m + v];
            }
            if (sum_dual < beta[u]) beta[u] = sum_dual;
        }
    }
    float dual_objective = 0.0;
    for (int u=0;u<m;++u) {
        dual_objective += beta[u];
    }
    std::cout << "[CG] Initial dual solution objective = " << dual_objective << "\n";
}

void find_a_valid_dual_solution_restricted1(const Problem& problem, DualVector& duals) {
    std::cout << "[CG] Finding a valid dual solution for the restricted problem 1\n";
    // solve resctricted LP of the dual problem using CPLEX to find a valid dual solution
    // maximize sum_u beta_u
    // subject to:
    // beta_u - sum_v delta_{iuv} <= 0   \for all i in V, u in M
    // delta_{iuv} + delta_{jvu} <= d_ij*f_uv + d_ji*f_vu \for all i,j in V, u,v in M, i<j, u!=v
    auto env = IloEnv();
    auto model = IloModel(env);
    int n = problem.n;
    int m = problem.m;
    // variables: beta_u for u in M, delta_{iuv} for i in V, u,v in M, i!=j
    IloNumVarArray beta(env, m, -IloInfinity, IloInfinity);
    IloNumVarArray delta(env, n*m*m, -IloInfinity, IloInfinity);
    // objective: maximize sum_u beta_u
    IloExpr obj_expr(env);
    for (int u=0;u<m;++u) {
        obj_expr += beta[u];
    }
    model.add(IloMaximize(env, obj_expr));
    // constraints
    // beta_u - sum_v delta_{iuv} <= 0   \for all i in V, u in M
    for (int i=0;i<n;++i) {
        for (int u=0;u<m;++u) {
            IloExpr con_expr(env);
            for (int v=0;v<m;++v) {
                if (u==v) continue;
                con_expr += delta[(i * m + u) * m + v];
            }
            model.add(beta[u] - con_expr <= 0);
        }
    }// delta_{iuv} + delta_{jvu} <= d_ij*f_uv + d_ji*f_vu \for all i,j in V, u,v in M, i<j, u!=v
    for (int i=0;i<n;++i) {
        for (int j=i+1;j<n;++j) {
            for (int u=0;u<m;++u) {
                for (int v=0;v<m;++v) {
                    if (u==v) continue;
                    model.add(delta[(i * m + u) * m + v] + delta[(j * m + v) * m + u] <= problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]);
                }
            }
        }
    }
    // solve the model
    IloCplex cplex(model);
    // set method barrier
    cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Barrier);
    // disable crossover
    cplex.setParam(IloCplex::Param::SolutionType, 2);
    cplex.solve();
    // count the non-zero delta variables
    int count_nonzero = 0;
    for (int i=0;i<n;++i) for (int u=0;u<m;++u)  for (int v=0;v<m;++v) {
        if (u==v) continue;
        if (cplex.getValue(delta[(i * m + u) * m + v]) != 0) {
            count_nonzero++;
        }
    }
    std::cout << "[CG] Initial dual solution objective = " << cplex.getObjValue() << ", non-zero delta variables = " << count_nonzero << "\n";
}

void find_a_valid_dual_solution_restricted2(const Problem& problem, DualVector& duals, std::vector<QuadKey>& tight_columns) {
    std::cout << "[CG] Finding a valid dual solution for the restricted problem 2\n";
    // solve restricted LP of the dual problem using CPLEX to find a valid dual solution
    // maximize sum_i alpha_i + sum_u beta_u
    // subject to:
    // alpha_i + beta_u - sum_v delta_{iuv} <= 0   \for all i in V, u in M
    // delta_{iuv} + delta_{jvu} <= d_ij*f_uv + d_ji*f_vu \for all i,j in V, u,v in M, i<j, u!=v
    auto env = IloEnv();
    auto model = IloModel(env);
    int n = problem.n;
    int m = problem.m;
    // variables: beta_u for u in M, delta_{iuv} for i in V, u,v in M, i!=j
    IloNumVarArray alpha(env, n, -IloInfinity, 0.0);
    IloNumVarArray beta(env, m, -IloInfinity, IloInfinity);
    IloNumVarArray delta(env, n*m*m, -IloInfinity, IloInfinity);
    // objective: maximize sum_i alpha_i + sum_u beta_u
    IloExpr obj_expr(env);
    for (int i=0;i<n;++i) {
        obj_expr += alpha[i];
    }
    for (int u=0;u<m;++u) {
        obj_expr += beta[u];
    }
    model.add(IloMaximize(env, obj_expr));
    // constraints
    for (int i=0;i<n;++i) {
        for (int u=0;u<m;++u) {
            IloExpr con_expr(env);
            for (int v=0;v<m;++v) {
                if (u==v) continue;
                con_expr += delta[(i * m + u) * m + v];
            }
            model.add(alpha[i] + beta[u] - con_expr <= 0);
        }
    }
    for (int i=0;i<n;++i) {
        for (int j=i+1;j<n;++j) {
            for (int u=0;u<m;++u) {
                for (int v=0;v<m;++v) {
                    if (u==v) continue;
                    model.add(delta[(i * m + u) * m + v] + delta[(j * m + v) * m + u] <= problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]);
                }
            }
        }
    }
    // solve the model
    IloCplex cplex(model);
    // set method barrier
    cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Barrier);
    // cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);
    // disable crossover
    cplex.setParam(IloCplex::Param::SolutionType, 2);
    cplex.solve();
    // // count the non-zero delta variables
    // int count_nonzero = 0;
    // for (int i=0;i<n;++i) for (int u=0;u<m;++u)  for (int v=0;v<m;++v) {
    //     if (u==v) continue;
    //     if (cplex.getValue(delta[(i * m + u) * m + v]) != 0) {
    //         count_nonzero++;
    //     }
    // }
    std::cout << "[CG] Initial dual solution objective = " << cplex.getObjValue() << ", status = " << cplex.getStatus() << "\n";
    
    // find column that have delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu
    for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) for (int u=0;u<m;++u) for (int v=0;v<m;++v) {
        if (u==v) continue;
        double lhs = cplex.getValue(delta[(i * m + u) * m + v]) + cplex.getValue(delta[(j * m + v) * m + u]);
        double rhs = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
        if (std::abs(lhs - rhs) < 1e-6) {
            // std::cout << "[CG] Column (" << i << "," << u << "," << j << "," << v << ") is tight with delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu\n";
            tight_columns.push_back(IncrementalRMP::canonical(i,u,j,v));
        }
    }


    // find column that have delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu and alpha_i + beta_u - sum_v delta_iuv = 0
    // for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) for (int u=0;u<m;++u) for (int v=0;v<m;++v) {
    //     if (u==v) continue;
    //     double lhs = cplex.getValue(delta[(i * m + u) * m + v]) + cplex.getValue(delta[(j * m + v) * m + u]);
    //     double rhs = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
    //     if (std::abs(lhs - rhs) < 1e-6) {

    //         double alpha_i = cplex.getValue(alpha[i]);
    //         double beta_u = cplex.getValue(beta[u]);
    //         double sum_delta_iuv = 0.0;
    //         for (int v2=0;v2<m;++v2) {
    //             if (v2==u) continue;
    //             sum_delta_iuv += cplex.getValue(delta[(i * m + u) * m + v2]);
    //         }
    //         if (std::abs(alpha_i + beta_u - sum_delta_iuv) < 1e-6) {
    //             // std::cout << "[CG] Column (" << i << "," << u << "," << j << "," << v << ") is tight with delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu and alpha_i + beta_u - sum_v delta_iuv = 0\n";
    //             tight_columns.push_back(IncrementalRMP::canonical(i,u,j,v));
    //         }
    //     }
    // }

    std::cout << "[CG] Number of tight columns: " << tight_columns.size() << "\n";
}

void find_a_valid_dual_solution_restricted2_Gurobi(const Problem& problem, DualVector& duals, std::vector<QuadKey>& tight_columns) {
    std::cout << "[CG] Finding a valid dual solution for the restricted problem 2\n";
    // solve restricted LP of the dual problem using Gurobi to find a valid dual solution
    // maximize sum_i alpha_i + sum_u beta_u
    // subject to:
    // alpha_i + beta_u - sum_v delta_{iuv} <= 0   \for all i in V, u in M
    // delta_{iuv} + delta_{jvu} <= d_ij*f_uv + d_ji*f_vu \for all i,j in V, u,v in M, i<j, u!=v
    auto env = GRBEnv();
    auto model = GRBModel(env);
    int n = problem.n;
    int m = problem.m;
    // variables: beta_u for u in M, delta_{iuv} for i in V, u,v in M, i!=j
    std::vector<GRBVar> alpha(n);
    std::vector<GRBVar> beta(m);
    std::vector<GRBVar> delta(n*m*m);
    for (int i=0;i<n;++i) {
        alpha[i] = model.addVar(-GRB_INFINITY, 0.0, 0.0, GRB_CONTINUOUS, "alpha_" + std::to_string(i));
    }
    for (int u=0;u<m;++u) {
        beta[u] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "beta_" + std::to_string(u));
    }
    for (int i=0;i<n;++i) for (int u=0;u<m;++u) for (int v=0;v<m;++v) {
        if (u==v) continue;
        delta[(i * m + u) * m + v] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "delta_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
    }
    
    // objective: maximize sum_i alpha_i + sum_u beta_u
    auto obj_expr = GRBLinExpr();
    for (int i=0;i<n;++i) {
        obj_expr += alpha[i];
    }
    for (int u=0;u<m;++u) {
        obj_expr += beta[u];
    }
    model.setObjective(obj_expr, GRB_MAXIMIZE);

    // constraints
    for (int i=0;i<n;++i) {
        for (int u=0;u<m;++u) {
            GRBLinExpr con_expr = 0.0;
            for (int v=0;v<m;++v) {
                if (u==v) continue;
                con_expr += delta[(i * m + u) * m + v];
            }
            model.addConstr(alpha[i] + beta[u] - con_expr <= 0);
        }
    }
    for (int i=0;i<n;++i) {
        for (int j=i+1;j<n;++j) {
            for (int u=0;u<m;++u) {
                for (int v=0;v<m;++v) {
                    if (u==v) continue;
                    model.addConstr(delta[(i * m + u) * m + v] + delta[(j * m + v) * m + u] <= problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]);
                }
            }
        }
    }
    // solve the model
    // set method barrier
    model.set(GRB_IntParam_Method, 2);
    // disable crossover
    model.set(GRB_IntParam_Crossover, 0);
    model.optimize();
    // // count the non-zero delta variables
    // int count_nonzero = 0;
    // for (int i=0;i<n;++i) for (int u=0;u<m;++u)  for (int v=0;v<m;++v) {
    //     if (u==v) continue;
    //     if (model.get(GRB_DoubleAttr_X, delta[(i * m + u) * m + v]) != 0) {
    //         count_nonzero++;
    //     }
    // }
    std::cout << "[CG] Initial dual solution objective = " << model.get(GRB_DoubleAttr_ObjVal) << ", status = " << model.get(GRB_IntAttr_Status) << "\n";
    
    // find column that have delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu
    for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) for (int u=0;u<m;++u) for (int v=0;v<m;++v) {
        if (u==v) continue;
        double lhs = delta[(i * m + u) * m + v].get(GRB_DoubleAttr_X) + delta[(j * m + v) * m + u].get(GRB_DoubleAttr_X);
        double rhs = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
        if (std::abs(lhs - rhs) < 1e-6) {
            // std::cout << "[CG] Column (" << i << "," << u << "," << j << "," << v << ") is tight with delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu\n";
            tight_columns.push_back(IncrementalRMP::canonical(i,u,j,v));
        }
    }


    // find column that have delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu and alpha_i + beta_u - sum_v delta_iuv = 0
    // for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) for (int u=0;u<m;++u) for (int v=0;v<m;++v) {
    //     if (u==v) continue;
    //     double lhs = delta[(i * m + u) * m + v].get(GRB_DoubleAttr_X) + delta[(j * m + v) * m + u].get(GRB_DoubleAttr_X);
    //     double rhs = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
    //     if (std::abs(lhs - rhs) < 1e-6) {

    //         double alpha_i = alpha[i].get(GRB_DoubleAttr_X);
    //         double beta_u = beta[u].get(GRB_DoubleAttr_X);
    //         double sum_delta_iuv = 0.0;
    //         for (int v2=0;v2<m;++v2) {
    //             if (v2==u) continue;
    //             sum_delta_iuv += delta[(i * m + u) * m + v2].get(GRB_DoubleAttr_X);
    //         }
    //         if (std::abs(alpha_i + beta_u - sum_delta_iuv) < 1e-6) {
    //             // std::cout << "[CG] Column (" << i << "," << u << "," << j << "," << v << ") is tight with delta_iuv + delta_jvu = d_ij*f_uv + d_ji*f_vu and alpha_i + beta_u - sum_v delta_iuv = 0\n";
    //             tight_columns.push_back(IncrementalRMP::canonical(i,u,j,v));
    //         }
    //     }
    // }

    std::cout << "[CG] Number of tight columns: " << tight_columns.size() << "\n";
}

void find_local_search_solution(const Problem& problem, std::vector<QuadKey>& ls_cols) {
    auto n = problem.n;
    auto m = problem.m;
    LocalSearch ls;
    Solution ls_sln("", "local_search");
    ls_sln.assignment = LocalSearch::initAssignment(problem.n - problem.fixed_assignments.size(), "random", "");
    // add fixed assignments to ls_sln.assignment
    ls_sln.assignment.resize(problem.n);
    for (const auto& [loc, fac] : problem.fixed_assignments) {
        ls_sln.assignment[loc] = fac;
    }
    auto max_iterations = 500;
    auto tabu_tenure = 10;
    ls.tabuSearch(problem, ls_sln, max_iterations, tabu_tenure, false);
    std::cout << "[CG] Local Search warm start objective: " << ls_sln.objective << "\n";
    ls_cols.clear();
    for (int i=0;i<n;++i) {
        int u = ls_sln.assignment[i];
        for (int j=0;j<n;++j) {
            if (i==j) continue;
            int v = ls_sln.assignment[j];
            ls_cols.push_back(IncrementalRMP::canonical(i,u,j,v));
        }
    }
}

void take_promising_columns(const Problem& problem, std::vector<QuadKey>& promising_cols) {
    // a promissing column is one with big flow but small distance, i.e. F[u][v] > 0 and D[i][j] < threshold
    promising_cols.clear();
    int n = problem.n;
    int m = problem.m;
    //  should have about 5-10% of the total possible columns
    int total_possible_columns = (n * (n - 1) / 2) * m * (m-1);
    int target_columns = std::max(1, (int)(total_possible_columns * 0.05));
    std::cout << "[CG] Total possible columns: " << total_possible_columns << ", Adding 5% = " << target_columns << "\n";
    // first sort (u,v) pairs with positive flow by flow value
    std::vector<std::pair<int,int>> positive_flow_pairs;
    for (int u=0;u<m;++u) {
        for (int v=0;v<m;++v) {
            if (u==v) continue;
            if (problem.F[u][v] > 0.0 || problem.F[v][u] > 0.0) {
                positive_flow_pairs.push_back({u,v});
            }
        }
    }
    std::sort(positive_flow_pairs.begin(), positive_flow_pairs.end(),
        [&problem](const std::pair<int,int>& a, const std::pair<int,int>& b) {
            double flow_a = problem.F[a.first][a.second] + problem.F[a.second][a.first];
            double flow_b = problem.F[b.first][b.second] + problem.F[b.second][b.first];
            return flow_a > flow_b;
        });
    // sort (i,j) pairs with small distance by distance value
    std::vector<std::pair<int,int>> small_distance_pairs;
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            if (i != j && problem.D[i][j] > 1e-6) {
                small_distance_pairs.push_back({i,j});
            }
        }
    }
    std::sort(small_distance_pairs.begin(), small_distance_pairs.end(),
        [&problem](const std::pair<int,int>& a, const std::pair<int,int>& b) {
            double dist_a = problem.D[a.first][a.second];
            double dist_b = problem.D[b.first][b.second];
            return dist_a < dist_b;
        });
    // create a mask to avoid adding duplicate columns
    std::vector<bool> added_mask(n*m*n*m, false);
    // keep adding promising columns until we reach target_columns
    while (promising_cols.size() < target_columns) {
        for (const auto& [u,v] : positive_flow_pairs) {
            for (const auto& [i,j] : small_distance_pairs) {
                QuadKey can = IncrementalRMP::canonical(i,u,j,v);
                auto index = can.to_index(n,m);
                if (!added_mask[index]) {
                    promising_cols.push_back(can);
                    added_mask[index] = true;
                    break;
                }
            }
            if (promising_cols.size() >= target_columns) break;
        }
    }
    std::cout << "[CG] Added " << promising_cols.size() << " promising columns based on flow and distance.\n";
}

Solution ColumnGenSolver::solve(
    const Problem& problem,
    const std::string& instance_path,
    const std::unordered_map<int,int>& fixed_variables
)
{
    int n = problem.n;
    int m = problem.m;

    // Build index sets
    auto phi = buildPhi(problem);

    IncrementalRMP rmp(phi, n, m, cfg.eps, true, true);
    std::cout << "[CG] Initialized IncrementalRMP with n=" << n << ", m=" << m << "\n";
    auto start_creation_time = std::chrono::steady_clock::now();
    rmp.createModel();
    auto elapsed_creation = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_creation_time).count();
    std::cout << "[CG] Created RMP model in " << elapsed_creation << " seconds.\n";

    // if problem has fixed variables, fix them in the RMP
    if(problem.fixed_assignments.size() > 0) {
        std::cout << "[CG] Fixing " << problem.fixed_assignments.size() << " variables based on problem input\n";
        rmp.fixXAssignments(problem.fixed_assignments);
    }

    if (!fixed_variables.empty()) {
        std::cout << "[CG] Fixing " << fixed_variables.size() << " variables based on input\n";
        rmp.fixXAssignments(fixed_variables);
    }

    // auto Omega0 = buildInitialOmega(problem, fixed_variables);
    std::vector<QuadKey> Omega0;
    std::cout << "[CG] Selecting promising columns based on flow and distance\n";
    take_promising_columns(problem, Omega0);

    // Load all positive columns from colfile
    Omega0 = buildInitialOmega(problem, fixed_variables);
    std::cout << "[CG] Initial Omega0 size = " << Omega0.size() << "\n";
    rmp.addColumns(Omega0);
    std::cout << "[CG] Added initial columns from Omega0, omega is now " << rmp.Omega.size() << "\n";
    // find a valid dual solution to initialize pi_in
    DualVector initial_pi_in;
    std::vector<QuadKey> tight_columns;
    find_a_valid_dual_solution_restricted2_Gurobi(problem, initial_pi_in, tight_columns);
    auto count_added_tight_columns = rmp.addColumns(tight_columns);
    std::cout << "[CG] Added " << count_added_tight_columns << " tight columns from restricted dual solution, omega is now " << rmp.Omega.size() << "\n";
    rmp.setPiIn(initial_pi_in);

    auto start_time = std::chrono::steady_clock::now();
    bool added_ls = false;
    int iteration = 0;
    float alpha = cfg.lambda_init;
    bool lastiter_swapped = false;
    while (true) {
        auto start_solving_time = std::chrono::steady_clock::now();
        auto solved = rmp.solve();
        auto elapsed_solving = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_solving_time).count();
        if (!solved) {
            // add Local search solution
            if (!added_ls) {
                std::vector<QuadKey> ls_cols;
                std::cout << "[CG] RMP solving failed, trying to add local search solution columns.\n";
                find_local_search_solution(problem, ls_cols);
                rmp.addColumns(ls_cols);
                added_ls = true;
                continue; // try solving again after adding local search columns
            } else {
                //  print status and break
                std::cerr << "[CG] Error: RMP solving status : " << rmp.cplex().getStatus() << ".\n";
                break;
            }
        }
        ++iteration;
        auto elapsed_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        std::cout << "=============================================================" << std::endl;
        std::cout << "[CG] Iteration " << iteration << " (" << elapsed_total << "s): RMP solved. Objective = " << rmp.getObjectiveValue()
                  << " in " << elapsed_solving << " seconds. Omega = "
                << rmp.Omega.size() << "\n";
        
        std::vector<PricingColumn> new_columns;
        PricingColumn mostnegative_column;
        double mostnegative_reduced_cost;
        rmp.retrieveCurrentDuals();
        // if (cfg.stabilize) 
        if(true)
        {
            bool found = rmp.pricingWithInOut(alpha, new_columns, mostnegative_column, mostnegative_reduced_cost);;
            int not_found_count = 0;
            while (!found) 
            {
                alpha = cfg.lambda_init; // reset alpha to initial value after no columns found
                std::cout << "[CG] No negative reduced cost columns found with stabilization and alpha=" << alpha << ".\n";
                bool swapped = rmp.swapInDualswPiStar();
                if(!swapped) {
                    std::cout << "[CG] No negative reduced cost columns found with stabilization and no swap possible. Terminating.\n";
                    // if(alpha > cfg.lambda_min) {
                    //     alpha *= 0.5;
                    // } else {
                    //     std::cout << "[CG] Alpha is already at minimum value. Terminating.\n";
                    //     break;
                    // }
                    break;
                } else {
                    std::cout << "[CG] No negative reduced cost columns found with stabilization. Swapped in pi_star and retrying.\n";
                }
                found = rmp.pricingWithInOut(alpha, new_columns, mostnegative_column, mostnegative_reduced_cost);
                
                if (!found && not_found_count > 5) {
                    rmp.pricingWithCurrentDuals(new_columns, mostnegative_column, mostnegative_reduced_cost);
                    break;
                }
                // if (!found && !lastiter_swapped) {
                //     // alpha *= 0.5;
                //     // std::cout << "[CG] No negative reduced cost columns found with alpha=" << alpha << ", reducing alpha and retrying.\n";
                //     rmp.swapInDualswPiStar();
                //     lastiter_swapped = true;
                //     std::cout << "[CG] No negative reduced cost columns found with alpha=" << alpha << ", swapping in pi_star and retrying.\n";
                //     continue;
                // } else if (!found && alpha > 1e-6) {
                //     alpha *= 0.5;
                //     std::cout << "[CG] No negative reduced cost columns found with alpha=" << alpha << " and already swapped in pi_star, reducing alpha and retrying.\n";
                //     lastiter_swapped = false;
                //     continue;
                // } else {
                //     lastiter_swapped = false;
                // }
            }
            if(!found) {
                std::cout << "[CG] No negative reduced cost columns found with stabilization. Terminating.\n";
                break;
            }
        } else {
            if (!rmp.pricingWithCurrentDuals(new_columns, mostnegative_column, mostnegative_reduced_cost)) {
                std::cout << "[CG] No negative reduced cost columns found. Terminating.\n";
                break;
            }
        }

        std::cout << "[CG] Found " << new_columns.size() << " negative reduced cost columns, most negative: (" 
                  << mostnegative_column.column.i << "," << mostnegative_column.column.u << ","
                  << mostnegative_column.column.j << "," << mostnegative_column.column.v
                  << ") with reduced cost " << mostnegative_reduced_cost << "\n";

        // if (cfg.add_most_negative) {
        //     auto added = rmp.addColumn(mostnegative_column);
        //     if (added) {
        //         std::cout << "[CG] Add most negative reduced cost column: (" 
        //               << mostnegative_column.column.i << "," << mostnegative_column.column.u << ","
        //               << mostnegative_column.column.j << "," << mostnegative_column.column.v
        //               << ") with reduced cost " << mostnegative_reduced_cost << "\n";
        //     } else {
        //         std::cout << "[CG] Most negative reduced cost column already exists in RMP.\n";
        //     }
        // } else {
        //     auto count = rmp.addColumns(new_columns);
        //     if (count > 0) {
        //         std::cout << "[CG] Added " << count << " new columns to RMP.\n";
        //     } else {
        //         std::cout << "[CG] No new columns were added to RMP (all already exist).\n";
        //     }
        // }

        {
            // add k smallest reduced cost columns
            int K = 1000;
            if (new_columns.size() > K) {
                auto cmp = [](const PricingColumn& a, const PricingColumn& b) {
                    return a.cost < b.cost;   // smaller cost first
                };
                std::nth_element(new_columns.begin(), new_columns.begin() + K, new_columns.end(), cmp);
                new_columns.resize(K);
                alpha *= 1.1; // increase alpha to stabilize more after adding only a few columns
                if (alpha > cfg.lambda_max) alpha = cfg.lambda_max;
                std::cout << "[CG] Added only " << K << " columns with smallest reduced cost. Increasing alpha to " << alpha << "\n";
            }
            auto count = rmp.addColumns(new_columns);
            if (count > 0) {
                std::cout << "[CG] Added " << count << " new columns to RMP.\n";
                // alpha = cfg.lambda_init; // reset alpha to initial value after adding new columns
            } else {
                std::cout << "[CG] No new columns were added to RMP (all already exist).\n";
            }

        }
    }

    Solution sln(instance_path, "column_generation_cplex");
    // sln.assignment = argmaxAssignment(n, m, rmp.getXValues());
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    sln.objective = rmp.getObjectiveValue();
    sln.time = elapsed;
    sln.lower_bound = rmp.getObjectiveValue();
    return sln;
}