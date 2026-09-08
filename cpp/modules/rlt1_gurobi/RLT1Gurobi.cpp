#include "RLT1Gurobi.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "../local_search/local_search.h"
#include "../../include/qap_solution_io.hpp"

RLT1GurobiSolver::RLT1GurobiSolver(const RLT1GurobiConfig& cfg_) : cfg(cfg_) {}


std::tuple< 
    GRBModel*,
    std::map<std::pair<int,int>, GRBVar>,
    std::map<std::tuple<int,int,int,int>, GRBVar>
    >
RLT1GurobiSolver::buildModel(
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
            for (auto u : M) {
                for (auto v : M) {
                    if(u == v) continue; // skip diagonal terms

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
        for (auto u : M) {
            for (auto v : M) {
                if (u == v) continue; // skip diagonal terms

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
    }

    // sum_v sum_i y[i,u,j,v] <= == x[i,u] for all i,u,j: u in Fn0
    for (auto i : V) {
        for (auto j : V) {
            if (i == j) continue; // skip diagonal terms
            for (auto u : M) {

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

    return {model, x, y};
}



Solution RLT1GurobiSolver::solve(const Problem& problem,
                                            const std::string& instance_path,
                                            const std::unordered_map<int,int>& fixed_variables) {
    try {
        cfg.is_relax = true; // always solve the relaxation
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
        #if 0
        {
            // load primal solution from file if provided
            auto primal_path = "/home/local.isima.fr/antran/UFF/QAP-Solver/results/nug12_rlt1_volume.txt";
            int i,u;
            double val;
            std::ifstream primal_file(primal_path);
            if (primal_file.is_open()) {
                std::string line;
                while (std::getline(primal_file, line)) {
                    std::istringstream iss(line);
                    if (!(iss >> i >> u >> val)) {
                        std::cerr << "Warning: malformed line in primal solution file: " << line << std::endl;
                        continue;
                    }
                    if (x.find(std::make_pair(i,u)) != x.end()) {
                        x[std::make_pair(i,u)].set(GRB_DoubleAttr_PStart, val);
                    }
                }
                primal_file.close();
                std::cout << "Primal solution loaded from: " << primal_path << std::endl;
            } else {
                std::cerr << "Warning: Could not open primal solution file: " << primal_path << std::endl;
            }

        }
        #endif
        
        {
            auto start_solve_time = std::chrono::high_resolution_clock::now();
            // Solve the model
            model->optimize();
            auto solve_time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_solve_time
            ).count();
            std::cout << "Relaxation solved in " << solve_time << " seconds." << std::endl;
            // obtain solution from relaxation
            
        }


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
            // set warm-start values for y variables based on x warm-start
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue; // skip diagonal terms
                    for (int u = 0; u < m; ++u) {
                        for (int v = 0; v < m; ++v) {
                            if (u == v) continue; // skip diagonal terms
                            auto key = std::make_tuple(i,u,j,v);
                            if (y.find(key) != y.end()) {
                                double val = (loc_to_fac[i] == u && loc_to_fac[j] == v) ? 1.0 : 0.0;
                                y[key].set(GRB_DoubleAttr_Start, val);
                            }
                        }
                    }
                }
            }
        }
        
        {
            // change x variables to binary
            for (auto& kv : x) {
                kv.second.set(GRB_CharAttr_VType, GRB_BINARY);
            }
            
            // get all x variables that are zero in the relaxation solution, fix them to zero, and re-solve the model as an integer program
            std::vector<std::pair<int,int>> x_to_fix;
            for (const auto& kv : x) {
                if (kv.second.get(GRB_DoubleAttr_X) == 0.0) {
                    x_to_fix.push_back(kv.first);
                }
            }
            std::cout << "Fixing " << x_to_fix.size() << " x variables to zero based on relaxation solution." << std::endl;
            for (const auto& key : x_to_fix) {
                auto var = x[key];
                var.set(GRB_DoubleAttr_UB, 0.0);
            }
            model->update();
            std::cout << "===========================================" << std::endl;
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