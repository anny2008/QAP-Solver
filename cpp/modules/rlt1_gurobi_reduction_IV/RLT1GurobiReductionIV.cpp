#include "RLT1GurobiReductionIV.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "../local_search/local_search.h"

RLT1GurobiReductionIVSolver::RLT1GurobiReductionIVSolver(const RLT1GurobiReductionIVConfig& cfg_) : cfg(cfg_) {}


std::tuple< 
    GRBModel*,
    std::map<std::pair<int,int>, GRBVar>,
    std::map<std::tuple<int,int,int,int>, GRBVar>
    >
RLT1GurobiReductionIVSolver::buildModel(
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
    // Compute F0
    // -------------------------------------------------------------
    std::vector<std::pair<int,int>> F0;
    std::vector<std::pair<int,int>> Fn0;
    std::vector<std::pair<int,int>> Fn0_2;

    for (auto u : M) {
        for (auto v : M) {
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

    for (auto u : M) {
        bool found = false;

        for (auto v : M) {
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

    std::cout << "I0 = ";
    for (auto u : I0)
        std::cout << u << " ";
    std::cout << std::endl;

    std::cout << "In0 = ";
    for (auto u : In0)
        std::cout << u << " ";
    std::cout << std::endl;

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
    // y variables
    // -------------------------------------------------------------
    std::map<std::tuple<int,int,int,int>, GRBVar> y;
    auto count_y = 0;
    for (auto i : V) {
        for (auto j : V) {
            if (i==j) continue; // skip diagonal terms
            for (auto pair : Fn0) {
                auto u = pair.first;
                auto v = pair.second;

                std::string name =
                    "y_" +
                    std::to_string(i) + "_" +
                    std::to_string(u) + "_" +
                    std::to_string(j) + "_" +
                    std::to_string(v);

                GRBVar var = model->addVar(
                    0.0,
                    1.0,
                    0.0,
                    GRB_CONTINUOUS,
                    name
                );
                count_y++;

                y[std::make_tuple(i, u, j, v)] = var;

                // symmetry
                y[std::make_tuple(j, v, i, u)] = var;
            }
        }
    }
    std::cout << "Added " << count_y << " y variables." << std::endl;

    // -------------------------------------------------------------
    // Objective
    // -------------------------------------------------------------
    GRBQuadExpr obj = 0;

    for (auto quad : y) {
        auto [i, u, j, v] = quad.first;
        auto var = quad.second;
        obj += (distances[i][j] * flows[u][v]) * var;
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
    // sum_j sum_v y[i,u,j,v] == x[i,u] for all i,u,v: u,v in Fn0
    for (auto i : V) {
        for (auto pair: Fn0_2) {
            auto u = pair.first;
            auto v = pair.second;

            GRBLinExpr lhs = 0;
            int count = 0;

            for (auto j : V) {

                auto key = std::make_tuple(i,u,j,v);

                if (y.find(key) != y.end()) {
                    lhs += y[key];
                    count++;
                }
            }

            if (count > 0) {
                model->addConstr(
                    lhs == x[std::make_pair(i, u)],
                    "link0_" +
                    std::to_string(i) + "_" +
                    std::to_string(u) + "_" +
                    std::to_string(v)
                );
            }
        }
    }

    // sum_v sum_i y[i,u,j,v] <= == x[i,u] for all i,u,j: u in Fn0
    for (auto i : V) {
        for (auto j : V) {
            if (i == j) continue; // skip diagonal terms
            for (auto u : In0) {

                GRBLinExpr lhs = 0;
                int count = 0;

                for (auto v : M) {
                    auto key = std::make_tuple(i,u,j,v);

                    if (y.find(key) != y.end()) {
                        lhs += y[key];
                        count++;
                    }
                }

                if (count > 0) {
                    if (problem.n > problem.m) {
                        model->addConstr(
                            lhs <= x[std::make_pair(i, u)],
                            "link1_" +
                            std::to_string(i) + "_" +
                            std::to_string(u) + "_" +
                            std::to_string(j)
                        );
                    }
                    else {
                        model->addConstr(
                            lhs == x[std::make_pair(i, u)],
                            "link1_" +
                            std::to_string(i) + "_" +
                            std::to_string(u) + "_" +
                            std::to_string(j)
                        );
                    }
                }
            }
            for (auto u : I0) {

                GRBLinExpr lhs = 0;
                int count = 0;

                for (auto v : M) {
                    auto key = std::make_tuple(i,u,j,v);

                    if (y.find(key) != y.end()) {
                        lhs += y[key];
                        count++;
                    }
                }

                if (count > 0) {
                    model->addConstr(
                        lhs <= x[std::make_pair(i,u)],
                        "link2_" +
                        std::to_string(i) + "_" +
                        std::to_string(u) + "_" +
                        std::to_string(j)
                    );
                }
            }
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

    // -------------------------------------------------------------
    // z_iujvkw if 
    //      f_uv + f_vu > 0 
    //  and f_uw + f_wu > 0 
    //  and f_vw + f_wv > 0
    // -------------------------------------------------------------
    std::map<std::tuple<int,int,int,int,int,int>, GRBVar> z;
    auto count_z = 0;
    for (auto i : V) {
        for (auto j : V) {
            if (i==j) continue; // skip diagonal terms
            for (auto k : V) {
                if (k==i || k==j) continue; // skip diagonal terms
                for (auto pair1 : Fn0_2) {
                    auto u = pair1.first;
                    auto v = pair1.second;
                    for (auto w = 0; w < m; ++w) {
                        if (w == u || w == v) continue;
                        if (flows[u][v] + flows[v][u] > 0 &&
                            flows[u][w] + flows[w][u] > 0 &&
                            flows[v][w] + flows[w][v] > 0) {
                            std::string name =
                                "z_" +
                                std::to_string(i) + "_" +
                                std::to_string(u) + "_" +
                                std::to_string(j) + "_" +
                                std::to_string(v) + "_" +
                                std::to_string(k) + "_" +
                                std::to_string(w);
                            // GRBVar var = model->addVar(
                            //     0.0,
                            //     1.0,
                            //     0.0,
                            //     GRB_CONTINUOUS,
                            //     name
                            // );
                            count_z++;
                            // z[std::make_tuple(i,u,j,v,k,w)] = var;
                        }
                    }
                }
            }
        }
    }
    std::cout << "Added " << count_z << " z variables." << std::endl;



    model->update();

    return {model, x, y};
}



Solution RLT1GurobiReductionIVSolver::solve(const Problem& problem,
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

        auto start_solve_time = std::chrono::high_resolution_clock::now();
        // Solve the model
        model->optimize();
        auto solve_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_solve_time
        ).count();
        std::cout << "Model solved in " << solve_time << " seconds." << std::endl;
        // print model to file for debugging
        // model->write("rlt1_gurobi_reduction_IV_cpp.lp");

        // // Extract the solution
        Solution solution(instance_path, "RLT1_Gurobi_Reduction_IV");
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
        std::map<std::tuple<int, int, int, int>, double> y_values;
        // recompute objective from y values for verification
        auto recomputed_obj = 0.0;
        for (const auto& kv : y) {
            if (kv.second.get(GRB_DoubleAttr_X) > 1e-6) {
                auto [i, u, j, v] = kv.first;
                y_values[kv.first] = kv.second.get(GRB_DoubleAttr_X);
                recomputed_obj += problem.D[i][j] * problem.F[u][v] * y_values[kv.first];
            }
        }
        std::cout << "Non-zero y variables:" << y_values.size() << std::endl;
        std::cout << "Objective from model: " << solution.objective << std::endl;
        std::cout << "Objective recomputed from y variables: " << recomputed_obj << std::endl;// recompute objective from x values for verification
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

        // // save y values to file for debugging
        // auto y_file_name = "y_values_318.json";
        // std::ofstream y_file(y_file_name);
        // for (const auto& kv : y_values) {
        //     auto [i, u, j, v] = kv.first;
        //     auto value = kv.second;
        //     y_file << "y[" << i << "," << u << "," << j << "," << v << "] = " << value << std::endl;
        // }
        // y_file.close();

        // save x values to file for debugging
        // auto x_file_name = "x_values_318.json";
        // std::ofstream x_file(x_file_name);
        // for (const auto& kv : x) {
        //     auto [i, u] = kv.first;
        //     auto value = kv.second.get(GRB_DoubleAttr_X);
        //     x_file << "x[" << i << "," << u << "] = " << value << std::endl;
        // }
        // x_file.close();

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