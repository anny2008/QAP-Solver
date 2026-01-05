#include "rtl1_solver.h"
#include <iostream>
#include <cmath>
#include <chrono>

// Helper macro for SCIP error checking
#define SCIP_CHECK(x) do { \
    SCIP_RETCODE retcode = (x); \
    if (retcode != SCIP_OKAY) { \
        std::cerr << "SCIP Error: " << retcode << std::endl; \
        return 1; \
    } \
} while(0)

// ============================================================================
// PUBLIC METHODS
// ============================================================================

int RTL1Solver::buildModel() {
    // Create problem
    SCIP_CALL_ABORT(SCIPcreateProb(scip, "QAP_RTL1", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));

    // Create variables
    SCIP_CHECK(createXVariables());
    SCIP_CHECK(createYVariables());

    // Add constraints
    SCIP_CHECK(addAssignmentConstraints());
    SCIP_CHECK(addLinkingConstraints());
    SCIP_CHECK(addSymmetryConstraints());

    // Add fixed variables if any
    if (!fixed_variables.empty()) {
        SCIP_CHECK(addFixedVariables());
    }

    // Add warm-start if provided
    if (!warm_start_assignment.empty()) {
        SCIP_CHECK(addWarmStart());
    }

    // Add objective function: min Σ D[i,j] * F[u,v] * y[i,u,j,v]
    SCIP_CALL_ABORT(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int u = 0; u < n; u++) {
                for (int v = 0; v < n; v++) {
                    auto key = std::make_tuple(i, u, j, v);
                    if (y_vars.find(key) != y_vars.end()) {
                        double coeff = D[i][j] * F[u][v];
                        SCIP_CALL_ABORT(SCIPchgVarObj(scip, y_vars[key], coeff));
                    }
                }
            }
        }
    }

    return 0;
}

RTL1Solver::Solution RTL1Solver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();

    Solution sol;
    
    // Set parameters
    if (!config.log_output) {
        SCIP_CALL_ABORT(SCIPsetIntParam(scip, "display/verblevel", 0));
    }
    
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "parallel/nthreads", config.threads));
    SCIP_CALL_ABORT(SCIPsetRealParam(scip, "limits/time", config.time_limit));
    
    // Symmetry handling
    if (config.preprocessing_symmetry > 0) {
        SCIP_CALL_ABORT(SCIPsetIntParam(scip, "presolving/symmetryhandling/maxperms", 
                                       config.preprocessing_symmetry * 100));
    }

    // Build the model (objective and constraints already added)
    // Now we can solve

    // Solve
    SCIP_CALL_ABORT(SCIPsolve(scip));

    auto end_time = std::chrono::high_resolution_clock::now();
    sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    // Get status
    SCIP_STATUS status = SCIPgetStatus(scip);
    sol.status = (int)status;

    // Get solution
    SCIP_SOL* best_sol = SCIPgetBestSol(scip);
    
    if (best_sol != nullptr) {
        sol.feasible = SCIPsolIsPartial(scip, best_sol) == FALSE;
        sol.assignment = extractAssignment(best_sol);
        
        // Evaluate actual QAP objective if feasible
        if (sol.feasible && sol.assignment.size() == n) {
            sol.objective = evaluateObjective(sol.assignment);
        } else {
            // Use the RTL1 objective value
            sol.objective = SCIPgetSolOrigObj(scip, best_sol);
        }
    }

    // Get lower bound from root node
    sol.lower_bound = SCIPgetDualbound(scip);
    
    if (config.log_output) {
        printInfo();
        if (sol.feasible) {
            std::cout << "Best objective: " << sol.objective << std::endl;
        }
        std::cout << "Lower bound: " << sol.lower_bound << std::endl;
        std::cout << "Solve time: " << sol.solve_time << "s" << std::endl;
    }

    return sol;
}

double RTL1Solver::evaluateObjective(const std::vector<int>& assignment) const {
    double obj = 0.0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int u = assignment[i];
            int v = assignment[j];
            obj += D[i][j] * F[u][v];
        }
    }
    
    return obj;
}

void RTL1Solver::printInfo() const {
    std::cout << "\n=== RTL1 SCIP Solver Info ===" << std::endl;
    std::cout << "Problem size: " << n << std::endl;
    std::cout << "Relaxation: " << (config.is_relax ? "Yes" : "No") << std::endl;
    std::cout << "Time limit: " << config.time_limit << "s" << std::endl;
    std::cout << "Threads: " << config.threads << std::endl;
}

// ============================================================================
// PRIVATE METHODS
// ============================================================================

int RTL1Solver::createXVariables() {
    std::string var_type = config.is_relax ? "C" : "B";  // Continuous or Binary
    
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < n; u++) {
            SCIP_VAR* var;
            char varname[256];
            snprintf(varname, sizeof(varname), "x_%d_%d", i, u);
            
            if (config.is_relax) {
                SCIP_CALL_ABORT(SCIPcreateVarBasic(scip, &var, varname, 
                                                   0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
            } else {
                SCIP_CALL_ABORT(SCIPcreateVarBasic(scip, &var, varname, 
                                                   0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
            }
            SCIP_CALL_ABORT(SCIPaddVar(scip, var));
            x_vars[std::make_pair(i, u)] = var;
        }
    }
    
    return 0;
}

int RTL1Solver::createYVariables() {
    std::string var_type = config.is_relax ? "C" : "B";
    
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < n; u++) {
            for (int j = 0; j < n; j++) {
                for (int v = 0; v < n; v++) {
                    SCIP_VAR* var;
                    char varname[256];
                    snprintf(varname, sizeof(varname), "y_%d_%d_%d_%d", i, u, j, v);
                    
                    if (config.is_relax) {
                        SCIP_CALL_ABORT(SCIPcreateVarBasic(scip, &var, varname,
                                                           0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
                    } else {
                        SCIP_CALL_ABORT(SCIPcreateVarBasic(scip, &var, varname,
                                                           0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
                    }
                    SCIP_CALL_ABORT(SCIPaddVar(scip, var));
                    y_vars[std::make_tuple(i, u, j, v)] = var;
                }
            }
        }
    }
    
    return 0;
}

int RTL1Solver::addAssignmentConstraints() {
    // Each facility assigned to exactly one location
    for (int i = 0; i < n; i++) {
        SCIP_CONS* cons;
        char consname[256];
        snprintf(consname, sizeof(consname), "assign_fac_%d", i);
        
        std::vector<SCIP_VAR*> vars;
        std::vector<SCIP_Real> coefs;
        
        for (int u = 0; u < n; u++) {
            vars.push_back(x_vars[std::make_pair(i, u)]);
            coefs.push_back(1.0);
        }
        
        SCIP_CALL_ABORT(SCIPcreateConsLinear(scip, &cons, consname, 
                                            vars.size(), vars.data(), coefs.data(),
                                            1.0, 1.0, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE));
        SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
        SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
    }
    
    // Each location assigned to exactly one facility
    for (int u = 0; u < n; u++) {
        SCIP_CONS* cons;
        char consname[256];
        snprintf(consname, sizeof(consname), "assign_loc_%d", u);
        
        std::vector<SCIP_VAR*> vars;
        std::vector<SCIP_Real> coefs;
        
        for (int i = 0; i < n; i++) {
            vars.push_back(x_vars[std::make_pair(i, u)]);
            coefs.push_back(1.0);
        }
        
        SCIP_CALL_ABORT(SCIPcreateConsLinear(scip, &cons, consname,
                                            vars.size(), vars.data(), coefs.data(),
                                            1.0, 1.0, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE));
        SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
        SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
    }
    
    return 0;
}

int RTL1Solver::addLinkingConstraints() {
    // For each (i,u,v), sum over j of y[i,u,j,v] = x[i,u]
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                std::vector<SCIP_VAR*> vars;
                std::vector<SCIP_Real> coefs;
                
                // Σ_j y[i,u,j,v]
                for (int j = 0; j < n; j++) {
                    vars.push_back(y_vars[std::make_tuple(i, u, j, v)]);
                    coefs.push_back(1.0);
                }
                
                // - x[i,u]
                vars.push_back(x_vars[std::make_pair(i, u)]);
                coefs.push_back(-1.0);
                
                SCIP_CONS* cons;
                char consname[256];
                snprintf(consname, sizeof(consname), "link_iu_jv_%d_%d_%d", i, u, v);
                
                SCIP_CALL_ABORT(SCIPcreateConsLinear(scip, &cons, consname,
                                                    vars.size(), vars.data(), coefs.data(),
                                                    -SCIPinfinity(scip), 0.0, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE));
                SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
                SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
            }
        }
    }
    
    // For each (i,u,j), sum over v of y[i,u,j,v] = x[i,u]
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < n; u++) {
            for (int j = 0; j < n; j++) {
                std::vector<SCIP_VAR*> vars;
                std::vector<SCIP_Real> coefs;
                
                // Σ_v y[i,u,j,v]
                for (int v = 0; v < n; v++) {
                    vars.push_back(y_vars[std::make_tuple(i, u, j, v)]);
                    coefs.push_back(1.0);
                }
                
                // - x[i,u]
                vars.push_back(x_vars[std::make_pair(i, u)]);
                coefs.push_back(-1.0);
                
                SCIP_CONS* cons;
                char consname[256];
                snprintf(consname, sizeof(consname), "link_iu_jv2_%d_%d_%d", i, u, j);
                
                SCIP_CALL_ABORT(SCIPcreateConsLinear(scip, &cons, consname,
                                                    vars.size(), vars.data(), coefs.data(),
                                                    -SCIPinfinity(scip), 0.0, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE));
                SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
                SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
            }
        }
    }
    
    return 0;
}

int RTL1Solver::addSymmetryConstraints() {
    // y[i,u,j,v] == y[j,v,i,u]
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < n; u++) {
            for (int j = i; j < n; j++) {
                for (int v = (i == j ? u : 0); v < n; v++) {
                    if (i == j && u == v) continue;
                    
                    std::vector<SCIP_VAR*> vars;
                    std::vector<SCIP_Real> coefs;
                    
                    vars.push_back(y_vars[std::make_tuple(i, u, j, v)]);
                    coefs.push_back(1.0);
                    
                    vars.push_back(y_vars[std::make_tuple(j, v, i, u)]);
                    coefs.push_back(-1.0);
                    
                    SCIP_CONS* cons;
                    char consname[256];
                    snprintf(consname, sizeof(consname), "sym_%d_%d_%d_%d", i, u, j, v);
                    
                    SCIP_CALL_ABORT(SCIPcreateConsLinear(scip, &cons, consname,
                                                        vars.size(), vars.data(), coefs.data(),
                                                        0.0, 0.0, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE));
                    SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
                    SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
                }
            }
        }
    }
    
    return 0;
}

int RTL1Solver::addFixedVariables() {
    for (const auto& [i, u] : fixed_variables) {
        SCIP_CALL_ABORT(SCIPchgVarLbGlobal(scip, x_vars[std::make_pair(i, u)], 1.0));
        SCIP_CALL_ABORT(SCIPchgVarUbGlobal(scip, x_vars[std::make_pair(i, u)], 1.0));
    }
    
    return 0;
}

int RTL1Solver::addWarmStart() {
    // Create a partial solution with warm-start values
    SCIP_SOL* partial_sol = nullptr;
    SCIP_CALL_ABORT(SCIPcreatePartialSol(scip, &partial_sol, nullptr));
    
    // Set x variables from warm-start
    std::vector<SCIP_VAR*> vars;
    std::vector<double> values;
    
    for (const auto& [key, value] : warm_start_assignment) {
        auto [i, u] = key;
        vars.push_back(x_vars[std::make_pair(i, u)]);
        values.push_back(value);
    }
    
    if (!vars.empty()) {
        SCIP_CALL_ABORT(SCIPsetSolVals(scip, partial_sol, vars.size(), 
                                       vars.data(), values.data()));
    }
    
    // Try to add the solution
    SCIP_Bool success = FALSE;
    SCIP_CALL_ABORT(SCIPtrySol(scip, partial_sol, FALSE, FALSE, TRUE, TRUE, TRUE, &success));
    
    if (config.log_output && success) {
        std::cout << "Successfully added warm-start solution" << std::endl;
    }
    
    SCIP_CALL_ABORT(SCIPfreeSol(scip, &partial_sol));
    
    return 0;
}

std::vector<int> RTL1Solver::extractAssignment(SCIP_SOL* sol) const {
    std::vector<int> assignment(n, -1);
    
    for (int i = 0; i < n; i++) {
        double best_val = -1.0;
        int best_u = -1;
        
        for (int u = 0; u < n; u++) {
            double val = SCIPgetSolVal(scip, sol, x_vars.at(std::make_pair(i, u)));
            if (val > best_val) {
                best_val = val;
                best_u = u;
            }
        }
        
        assignment[i] = best_u;
    }
    
    return assignment;
}
