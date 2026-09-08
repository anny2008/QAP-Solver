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


void load_warmstart_solution(const std::string& slnfile, const Problem& problem, std::vector<int>& assignment) {
    // n obj
    // i1 i2 i3 ...
    int n;
    double obj;
    int i = 0;
    int u = 0;
    std::ifstream fin(slnfile);
    if (!fin.is_open()) {
        std::cerr << "Cannot open solution file: " << slnfile << std::endl;
        return;
    }
    fin >> n >> obj;
    while (fin >> i) {
        if (i < 0 || u >= problem.m) {
            std::cerr << "Invalid facility index in solution file: " << i << std::endl;
            return;
        }
        assignment[u] = i;
        u++;
    }
}

void set_warmstart_solution(GRBModel* model, const std::map<std::pair<int,int>, GRBVar>& x, const std::map<std::tuple<int,int,int,int>, GRBVar>& y, const std::vector<int>& assignment) {
    for (const auto& kv : x) {
        auto [i, u] = kv.first;
        auto var = kv.second;
        if (assignment[u] == i) {
            var.set(GRB_DoubleAttr_Start, 1.0);
            std::cout << "Warmstart: setting x[" << i << "," << u << "] = 1" << std::endl;
        } else {
            var.set(GRB_DoubleAttr_Start, 0.0);
        }
    }
    for (const auto& kv : y) {
        auto [i, u, j, v] = kv.first;
        auto var = kv.second;
        if (assignment[u] == i && assignment[v] == j) {
            var.set(GRB_DoubleAttr_Start, 1.0);
        } else {
            var.set(GRB_DoubleAttr_Start, 0.0);
        }
    }
}


Solution RLT1GurobiReductionIVSolver::solve(const Problem& problem,
                                            const std::string& instance_path,
                                            const std::unordered_map<int,int>& fixed_variables) {
    try {
        cfg.is_relax = true; // first solve the relaxation to get a lower bound
        // Build the model
        auto start_time = std::chrono::high_resolution_clock::now();
        auto [model, x, y] = buildModel(problem, fixed_variables);
        auto build_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();
        std::cout << "Model built in " << build_time << " seconds." << std::endl;

        auto warmstart_sln_file = "/home/local.isima.fr/antran/UFF/QAP-Solver/data/M/instance_318_LEFT.sln";
        std::vector<int> warmstart_assignment(problem.m, -1);
        load_warmstart_solution(warmstart_sln_file, problem, warmstart_assignment);
        // set_warmstart_solution(model, x, y, warmstart_assignment);

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
        std::cout << "===========================================" << std::endl;
        if(0)
        {
            // Get all y variables that are zero in the relaxation solution, fix them to zero, and re-solve the model as an integer program
            std::vector<std::tuple<int,int,int,int>> y_to_fix;
            for (const auto& kv : y) {
                if (kv.second.get(GRB_DoubleAttr_X) == 0.0) {
                    y_to_fix.push_back(kv.first);
                }
            }

            std::cout << "Fixing " << y_to_fix.size() << " y variables to zero based on relaxation solution." << std::endl;
            for (const auto& key : y_to_fix) {
                auto var = y[key];
                var.set(GRB_DoubleAttr_UB, 0.0);
            }
            // change x variables to binary
            for (auto& kv : x) {
                kv.second.set(GRB_CharAttr_VType, GRB_BINARY);
            }
            // re-solve the model
            auto start_solve_time_2 = std::chrono::high_resolution_clock::now();
            model->optimize();
            auto solve_time_2 = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_solve_time_2
            ).count();
            std::cout << "Model re-solved with fixed y variables in " << solve_time_2 << " seconds." << std::endl;  
            std::cout << "Objective: " << model->get(GRB_DoubleAttr_ObjVal) << std::endl;
            // print model to file for debugging
            // model->write("rlt1_gurobi_reduction_IV_cpp.lp");
            std::cout << "===========================================" << std::endl;
        }
        if(0)
        {
            // change x variables to binary
            for (auto& kv : x) {
                kv.second.set(GRB_CharAttr_VType, GRB_BINARY);
            }
            //  change objective of y zero to 1 and other y to 0
            std::vector<std::tuple<int,int,int,int>> y_zeros;
            std::vector<std::tuple<int,int,int,int>> y_nonzeros;
            for (const auto& kv : y) {
                if (kv.second.get(GRB_DoubleAttr_X) == 0.0) {
                    y_zeros.push_back(kv.first);
                } else {
                    y_nonzeros.push_back(kv.first);
                }
            }
            std::cout << "Setting objective coefficient of " << y_zeros.size() << " zero y variables to 1 and " << y_nonzeros.size() << " non-zero y variables to 0." << std::endl;
            for (const auto& key : y_zeros) {
                auto var = y[key];
                var.set(GRB_DoubleAttr_Obj, 1.0);
            }
            for (const auto& key : y_nonzeros) {
                auto var = y[key];
                var.set(GRB_DoubleAttr_Obj, 0.0);
            }
            auto start_solve_time_2 = std::chrono::high_resolution_clock::now();
            model->optimize();
            auto solve_time_2 = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_solve_time_2
            ).count();
            std::cout << "Model re-solved with fixed y variables in " << solve_time_2 << " seconds." << std::endl;
            std::cout << "Objective: " << model->get(GRB_DoubleAttr_ObjVal) << std::endl;
            std::cout << "===========================================" << std::endl;
        }
        // if(0)
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

            set_warmstart_solution(model, x, y, warmstart_assignment);
            auto start_solve_time_2 = std::chrono::high_resolution_clock::now();
            model->optimize();
            auto solve_time_2 = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_solve_time_2
            ).count();
            std::cout << "Model re-solved with fixed x variables in " << solve_time_2 << " seconds." << std::endl;
            std::cout << "Objective: " << model->get(GRB_DoubleAttr_ObjVal) << std::endl;

            std::cout << "===========================================" << std::endl;
        }
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