#include "KBXYGurobi.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "../local_search/local_search.h"
#include "../../include/qap_solution_io.hpp"

KBXYGurobiSolver::KBXYGurobiSolver(const KBXYGurobiConfig& cfg_) : cfg(cfg_) {}

double l_iu(int i, int u, const Problem& problem) 
{
    // min sum_jv D[i][j] * F[u][v] * x[j][v]
    // st. sum_{v != u} x[j][v] == 1 for all j != i
    //  sum_{j != i} x[j][v] == 1 for all v != u
    //     x[j][v] in {0,1} for all j,v: j != i, v != u

    auto n = problem.n;
    GRBEnv* env = new GRBEnv();
    GRBModel* model = new GRBModel(*env);
    std::map<std::pair<int,int>, GRBVar> x;
    for (int j = 0; j < n; ++j) {
        for (int v = 0; v < n; ++v) {
            if (j == i || v == u) continue;
            x[std::make_pair(j, v)] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY);
        }
    }
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        GRBLinExpr expr = 0;
        for (int v = 0; v < n; ++v) {
            if (v == u) continue;
            expr += x[std::make_pair(j, v)];
        }
        model->addConstr(expr == 1);
    }
    for (int v = 0; v < n; ++v) {
        if (v == u) continue;
        GRBLinExpr expr = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            expr += x[std::make_pair(j, v)];
        }
        model->addConstr(expr == 1);
    }
    GRBLinExpr obj = 0;
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        for (int v = 0; v < n; ++v) {
            if (v == u) continue;
            obj += problem.D[i][j] * problem.F[u][v] * x[std::make_pair(j, v)];
        }
    }
    model->setObjective(obj, GRB_MINIMIZE);
    // turn off Gurobi output
    model->set(GRB_IntParam_OutputFlag, 0);
    model->optimize();
    // ensure that the model was solved to optimality
    if (model->get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
        std::cerr << "Error: Gurobi optimization did not finish with optimal status." << std::endl;
        delete model;
        delete env;
        throw std::runtime_error("Gurobi optimization failed");
    }
    double val = model->get(GRB_DoubleAttr_ObjVal);
    delete model;
    delete env;
    return val;
}

double a_iu(int i, int u, const Problem& problem) 
{
    // max sum_jv D[i][j] * F[u][v] * x[j][v]
    // st. sum_v x[j][v] == 1 for all
    //  sum_j x[j][v] == 1 for all 
    //     x[j][v] in {0,1} for all j, v

    auto n = problem.n;
    GRBEnv* env = new GRBEnv();
    GRBModel* model = new GRBModel(*env);
    std::map<std::pair<int,int>, GRBVar> x;
    for (int j = 0; j < n; ++j) {
        for (int v = 0; v < n; ++v) {
            x[std::make_pair(j, v)] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY);
        }
    }
    for (int j = 0; j < n; ++j) {
        GRBLinExpr expr = 0;
        for (int v = 0; v < n; ++v) {
            expr += x[std::make_pair(j, v)];
        }
        model->addConstr(expr == 1);
    }
    for (int v = 0; v < n; ++v) {
        GRBLinExpr expr = 0;
        for (int j = 0; j < n; ++j) {
            expr += x[std::make_pair(j, v)];
        }
        model->addConstr(expr == 1);
    }
    GRBLinExpr obj = 0;
    for (int j = 0; j < n; ++j) {
        for (int v = 0; v < n; ++v) {
            obj += problem.D[i][j] * problem.F[u][v] * x[std::make_pair(j, v)];
        }
    }
    model->setObjective(obj, GRB_MAXIMIZE);
    model->set(GRB_IntParam_OutputFlag, 0);
    model->optimize();
    // ensure that the model was solved to optimality
    if (model->get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
        std::cerr << "Error: Gurobi optimization did not finish with optimal status." << std::endl;
        delete model;
        delete env;
        throw std::runtime_error("Gurobi optimization failed");
    }
    double val = model->get(GRB_DoubleAttr_ObjVal);
    delete model;
    delete env;
    return val;
}

std::tuple< 
    GRBModel*,
    std::map<std::pair<int,int>, GRBVar>,
    std::map<std::pair<int,int>, GRBVar>
    >
KBXYGurobiSolver::buildModel(
    const Problem& problem,
    const std::unordered_map<int,int>& fixed_variables
) {
    auto n = problem.n;
    auto m = problem.m;

    const auto& distances = problem.D;
    const auto& flows = problem.F;

    std::vector<int> V(n);
    std::vector<int> M(m);

    for (int i = 0; i < n; ++i)
        V[i] = i;

    for (int u = 0; u < m; ++u)
        M[u] = u;

    // -------------------------------------------------------------
    // Compute l_iu and a_iu values
    // -------------------------------------------------------------
    std::map<std::pair<int,int>, double> l_iu_values;
    std::map<std::pair<int,int>, double> a_iu_values;
    for (auto i : V) {
        for (auto u : M) {
            l_iu_values[std::make_pair(i, u)] = l_iu(i, u, problem);
            a_iu_values[std::make_pair(i, u)] = a_iu(i, u, problem);
        }
    }

    // -------------------------------------------------------------
    // Create environment and model
    // -------------------------------------------------------------
    GRBEnv* env = new GRBEnv();
    GRBModel* model = new GRBModel(*env);

    model->set(GRB_StringAttr_ModelName, "QAP_RLT1");

    // -------------------------------------------------------------
    // x variables
    // -------------------------------------------------------------
    std::map<std::pair<int,int>, GRBVar> x;

    for (auto i : V) {
        for (auto u : M) {
            x[std::make_pair(i, u)] = model->addVar(
                0.0,
                1.0,
                0.0,
                cfg.is_relax ? GRB_CONTINUOUS : GRB_BINARY,
                "x_" + std::to_string(i) + "_" + std::to_string(u)
            );
        }
    }

    // -------------------------------------------------------------
    // z variables z_iu >= 0
    // -------------------------------------------------------------
    std::map<std::pair<int,int>, GRBVar> z;
    for (auto i : V) {
        for (auto u : M) {
            z[std::make_pair(i, u)] = model->addVar(
                0.0,
                GRB_INFINITY,
                0.0,
                GRB_CONTINUOUS,
                "z_" + std::to_string(i) + "_" + std::to_string(u)
            );
        }
    }
    std::cout << "Added " << z.size() << " z variables." << std::endl;

    // -------------------------------------------------------------
    // Objective: sum_{i,u} z_iu + (d_ij*f_uv + l_ij)*x_iu
    // -------------------------------------------------------------
    GRBLinExpr obj = 0;
    for (auto i : V) {
        for (auto u : M) {
            obj += z[std::make_pair(i, u)];
            obj += l_iu_values[std::make_pair(i, u)] * x[std::make_pair(i, u)];
        }
    }

    model->setObjective(obj, GRB_MINIMIZE);

    // -------------------------------------------------------------
    // Assignment constraints
    // -------------------------------------------------------------
    for (auto i : V) {

        GRBLinExpr expr = 0;

        for (auto u : M)
            expr += x[std::make_pair(i, u)];

        if (problem.n > problem.m) {
            model->addConstr(
                expr <= 1,
                "assign_fac_" + std::to_string(i)
            );
        }
        else {
            model->addConstr(
                expr == 1,
                "assign_fac_" + std::to_string(i)
            );
        }
    }

    for (auto u : M) {

        GRBLinExpr expr = 0;

        for (auto i : V)
            expr += x[std::make_pair(i, u)];

        model->addConstr(
            expr == 1,
            "assign_loc_" + std::to_string(u)
        );
    }

    // -------------------------------------------------------------
    // Linking constraints
    // -------------------------------------------------------------
    // a_iu = max_{x \in X}{sum_{j,v} d_ij*f_uv*x_jv}
    // z_iu >= sum_{j,v} d_ij*f_uv*x_jv - a_iu*(1-x_iu) -(l_iu+d_ij*f_uv)*x_iu
    for (auto i : V) {
        for (auto u : M) {
            double a = a_iu_values[std::make_pair(i, u)];
            double l = l_iu_values[std::make_pair(i, u)];
            auto x_iu = x[std::make_pair(i, u)];

            GRBLinExpr lhs = z[std::make_pair(i, u)];
            GRBLinExpr rhs = 0;

            for (auto j : V) {
                for (auto v : M) {
                    rhs += problem.D[i][j] * problem.F[u][v] * x[std::make_pair(j, v)];
                }
            }

            rhs -= a * (1 - x_iu);
            rhs -= (l + problem.D[i][i] * problem.F[u][u]) * x_iu;

            model->addConstr(
                lhs >= rhs,
                "link2_" +
                std::to_string(i) + "_" +
                std::to_string(u)
            );
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

            model->addConstr(
                x[std::make_pair(i, u)] == 1,
                "fixed_" +
                std::to_string(i) + "_" +
                std::to_string(u)
            );

            std::cout << "Fixed assignment: x["
                    << i << "," << u << "] = 1"
                    << std::endl;
        }
    }

    model->update();

    return {model, x, z};
}



Solution KBXYGurobiSolver::solve(const Problem& problem,
                                            const std::string& instance_path,
                                            const std::unordered_map<int,int>& fixed_variables) {
    try {
        // Build the model
        auto start_time = std::chrono::high_resolution_clock::now();
        auto [model, x, y] = buildModel(problem, fixed_variables);
        auto build_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();
        std::cout << "Model built in " << build_time << " seconds." << std::endl;

        if (cfg.lpmethod == "barrier") {
            model->set(GRB_IntParam_Method, GRB_METHOD_BARRIER);
        } else if (cfg.lpmethod == "dual") {
            model->set(GRB_IntParam_Method, GRB_METHOD_DUAL);
        } else if (cfg.lpmethod == "primal") {
            model->set(GRB_IntParam_Method, GRB_METHOD_PRIMAL);
        } else if (cfg.lpmethod == "concurrent") {
            model->set(GRB_IntParam_Method, GRB_METHOD_CONCURRENT);
        } else if (cfg.lpmethod == "auto") {
            std::cout << "Using Gurobi's automatic LP method selection." << std::endl;
        } else {
            std::cerr << "Unknown LP method: " << cfg.lpmethod << std::endl;
        }
        
        if(0)
        {
            auto m = problem.m;
            auto n = problem.n;
            double ws_obj;
            // read warm-start solution from file if provided
            int file_n;
            std::vector<int> loc_to_fac;
            qap::read_solution(cfg.warmstart, file_n, loc_to_fac, ws_obj);
            
            if (file_n != n) {
                std::cerr << "Warning: Warm-start file size mismatch (expected " << n << ", got " << file_n << ")" << std::endl;
                return Solution(instance_path, "RLT1_Gurobi");
            }
            // set warm-start values for x variables
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < m; ++u) {
                    if (loc_to_fac[i] == u) {
                        x[std::make_pair(i, u)].set(GRB_DoubleAttr_Start, 1.0);
                    } else {
                        x[std::make_pair(i, u)].set(GRB_DoubleAttr_Start, 0.0);
                    }
                }
            }
            // // set warm-start values for y variables based on x warm-start
            // for (int i = 0; i < n; ++i) {
            //     for (int j = 0; j < n; ++j) {
            //         if (i == j) continue; // skip diagonal terms
            //         for (int u = 0; u < m; ++u) {
            //             for (int v = 0; v < m; ++v) {
            //                 if (u == v) continue; // skip diagonal terms
            //                 auto key = std::make_tuple(i,u,j,v);
            //                 if (y.find(key) != y.end()) {
            //                     double val = (loc_to_fac[i] == u && loc_to_fac[j] == v) ? 1.0 : 0.0;
            //                     y[key].set(GRB_DoubleAttr_Start, val);
            //                 }
            //             }
            //         }
            //     }
            // }
        }
        
        {
            auto start_solve_time = std::chrono::high_resolution_clock::now();
            // Solve the model
            model->optimize();
            auto solve_time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_solve_time
            ).count();
            std::cout << "Model solved in " << solve_time << " seconds." << std::endl;
            
        }
        // print model to file for debugging
        // model->write("rlt1_gurobi_cpp.lp");

        // // Extract the solution
        Solution solution(instance_path, "RLT1_Gurobi");
        solution.assignment.resize(problem.n, -1);
        std::cout << "cfg.is_relax: " << cfg.is_relax << std::endl;
        if (!cfg.is_relax) {
            std::cout << "cfg.is_relax: " << cfg.is_relax << std::endl;
            for (const auto& kv : x) {
                auto [i, u] = kv.first;
                auto var = kv.second;
                if (var.get(GRB_DoubleAttr_X) > 0.5) {
                    solution.assignment[i] = u;
                    // std::cout << "Assigned facility " << i << " to location " << u << std::endl;
                }
            }
        }

        solution.objective = model->get(GRB_DoubleAttr_ObjVal);
        solution.lower_bound = model->get(GRB_DoubleAttr_ObjBound);
        // get non-zero y variables
        // std::map<std::tuple<int, int, int, int>, double> y_values;
        // // recompute objective from y values for verification
        // auto recomputed_obj = 0.0;
        // for (const auto& kv : y) {
        //     if (kv.second.get(GRB_DoubleAttr_X) > 1e-6) {
        //         auto [i, u, j, v] = kv.first;
        //         y_values[kv.first] = kv.second.get(GRB_DoubleAttr_X);
        //         recomputed_obj += problem.D[i][j] * problem.F[u][v] * y_values[kv.first];
        //     }
        // }
        // std::cout << "Non-zero y variables:" << y_values.size() << std::endl;
        std::cout << "Objective from model: " << solution.objective << std::endl;
        // std::cout << "Objective recomputed from y variables: " << recomputed_obj << std::endl;// recompute objective from x values for verification
        if (!cfg.is_relax) {
            auto recomputed_obj_from_x = 0.0;
            for (auto i = 0; i < solution.assignment.size(); ++i) {
                auto u = solution.assignment[i];
                for (auto j = 0; j < solution.assignment.size(); ++j) {
                    auto v = solution.assignment[j];
                    recomputed_obj_from_x += problem.D[i][j] * problem.F[u][v];
                }
            }
            std::cout << "Objective recomputed from x variables: " << recomputed_obj_from_x << std::endl;
        }

        return solution;
    }
    catch (GRBException e) {
        std::cerr << "Gurobi error code: " << e.getErrorCode() << std::endl;
        std::cerr << "Gurobi error message: " << e.getMessage() << std::endl;
        throw;
    }
    catch (...) {
        std::cerr << "Unknown error during optimization." << std::endl;
        throw;
    }
}