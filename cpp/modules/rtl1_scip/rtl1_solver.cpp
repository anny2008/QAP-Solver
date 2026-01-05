/**
 * RTL1 SCIP QAP Solver - Main Implementation
 * 
 * Implements the Reformulation-Linearization Technique (RLT1) for Quadratic Assignment Problem
 * using SCIP 9.2.0.
 * 
 * Based on: QAP_New_formulation/VolQAP copy 2/src/qap_scip.cpp
 */

#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <vector>
#include <map>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

// Type aliases for readability
using XVarMap = std::map<std::pair<int,int>, SCIP_VAR*>;
using YVarMap = std::map<std::tuple<int,int,int,int>, SCIP_VAR*>;

/**
 * Read QAP instance from QAPLIB format file
 * Format:
 *   n
 *   F[0,0] F[0,1] ... F[0,n-1]
 *   ...
 *   F[n-1,0] ... F[n-1,n-1]
 *   D[0,0] D[0,1] ... D[0,n-1]
 *   ...
 *   D[n-1,0] ... D[n-1,n-1]
 */
struct QAPInstance {
    int n;
    std::vector<std::vector<double>> F;  // Flow matrix
    std::vector<std::vector<double>> D;  // Distance matrix
    
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }
        
        file >> n;
        F.assign(n, std::vector<double>(n));
        D.assign(n, std::vector<double>(n));
        
        // Read flow matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> F[i][j];
            }
        }
        
        // Read distance matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> D[i][j];
            }
        }
        
        file.close();
        return true;
    }
};

/**
 * Create RLT1 x variables: x[i,u] ∈ {0,1} for facility i at location u
 */
void create_x_variables(SCIP* scip, int n, XVarMap& x) {
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            SCIP_VAR* var = nullptr;
            char varname[256];
            snprintf(varname, 255, "x_%d_%d", i, u);
            
            SCIP_CALL_ABORT(SCIPcreateVarBasic(scip, &var, varname, 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
            SCIP_CALL_ABORT(SCIPaddVar(scip, var));
            x[{i, u}] = var;
        }
    }
}

/**
 * Create RLT1 y variables: y[i,u,j,v] ∈ [0,1] (continuous) for linearization
 */
void create_y_variables(SCIP* scip, int n, YVarMap& y) {
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    SCIP_VAR* var = nullptr;
                    char varname[256];
                    snprintf(varname, 255, "y_%d_%d_%d_%d", i, u, j, v);
                    
                    // CONTINUOUS variables for RLT1 linearization
                    SCIP_CALL_ABORT(SCIPcreateVarBasic(scip, &var, varname, 0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
                    SCIP_CALL_ABORT(SCIPaddVar(scip, var));
                    y[{i, u, j, v}] = var;
                }
            }
        }
    }
}

/**
 * Add assignment constraints:
 * - Each facility to exactly one location: Σ_u x[i,u] = 1
 * - Each location to exactly one facility: Σ_i x[i,u] = 1
 */
void add_assignment_constraints(SCIP* scip, int n, const XVarMap& x) {
    // Facility constraints: each facility assigned to one location
    for (int i = 0; i < n; ++i) {
        SCIP_CONS* cons = nullptr;
        char consname[256];
        snprintf(consname, 255, "assign_fac_%d", i);
        
        SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 1.0, 1.0));
        for (int u = 0; u < n; ++u) {
            SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), 1.0));
        }
        SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
        SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
    }
    
    // Location constraints: each location gets one facility
    for (int u = 0; u < n; ++u) {
        SCIP_CONS* cons = nullptr;
        char consname[256];
        snprintf(consname, 255, "assign_loc_%d", u);
        
        SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 1.0, 1.0));
        for (int i = 0; i < n; ++i) {
            SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), 1.0));
        }
        SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
        SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
    }
}

/**
 * Add RLT1 linking constraints:
 * - Σ_v y[i,u,j,v] = x[i,u]  (for all i,u,j)
 * - Σ_j y[i,u,j,v] = x[i,u]  (for all i,u,v)
 * - y[i,u,j,v] = y[j,v,i,u]  (symmetry)
 */
void add_rlt1_constraints(SCIP* scip, int n, const XVarMap& x, const YVarMap& y) {
    // Link1: Σ_j y[i,u,j,v] = x[i,u]
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                SCIP_CONS* cons = nullptr;
                char consname[256];
                snprintf(consname, 255, "link1_%d_%d_%d", i, u, v);
                
                SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 0.0, 0.0));
                for (int j = 0; j < n; ++j) {
                    SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, y.at({i, u, j, v}), 1.0));
                }
                SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), -1.0));
                SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
                SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
            }
        }
    }
    
    // Link2: Σ_v y[i,u,j,v] = x[i,u]
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                SCIP_CONS* cons = nullptr;
                char consname[256];
                snprintf(consname, 255, "link2_%d_%d_%d", i, u, j);
                
                SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 0.0, 0.0));
                for (int v = 0; v < n; ++v) {
                    SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, y.at({i, u, j, v}), 1.0));
                }
                SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), -1.0));
                SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
                SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
            }
        }
    }
    
    // Link3 (Symmetry): y[i,u,j,v] = y[j,v,i,u]
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    SCIP_CONS* cons = nullptr;
                    char consname[256];
                    snprintf(consname, 255, "link3_%d_%d_%d_%d", i, u, j, v);
                    
                    SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 0.0, 0.0));
                    SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, y.at({i, u, j, v}), 1.0));
                    SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, y.at({j, v, i, u}), -1.0));
                    SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
                    SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
                }
            }
        }
    }
}

/**
 * Set objective function: min Σ F[i,j] * D[u,v] * y[i,u,j,v]
 */
void set_objective(SCIP* scip, int n, const QAPInstance& qap, const YVarMap& y) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    double coeff = qap.F[i][j] * qap.D[u][v];
                    SCIP_CALL_ABORT(SCIPchgVarObj(scip, y.at({i, u, j, v}), coeff));
                }
            }
        }
    }
}

/**
 * Extract solution from SCIP
 */
void extract_solution(SCIP* scip, int n, const XVarMap& x, const QAPInstance& qap, 
                      std::vector<int>& assignment, double& objective) {
    SCIP_SOL* bestsol = SCIPgetBestSol(scip);
    if (bestsol == nullptr) {
        std::cout << "No solution found" << std::endl;
        return;
    }
    
    assignment.resize(n);
    objective = 0.0;
    
    // Extract assignment from x variables
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            double val = SCIPgetSolVal(scip, bestsol, x.at({i, u}));
            if (val > 0.9) {  // Binary variable, close to 1
                assignment[i] = u;
                break;
            }
        }
    }
    
    // Compute objective from assignment
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            objective += qap.F[i][j] * qap.D[assignment[i]][assignment[j]];
        }
    }
}

/**
 * Main solver function
 */
int solve_qap_rtl1(const std::string& instance_file, double time_limit, 
                   std::vector<int>& assignment, double& objective, double& lower_bound) {
    // Load instance
    QAPInstance qap;
    if (!qap.load(instance_file)) {
        return 1;
    }
    
    int n = qap.n;
    std::cout << "Solving QAP instance with n=" << n << std::endl;
    
    // Create SCIP environment
    SCIP* scip = nullptr;
    SCIP_CALL_ABORT(SCIPcreate(&scip));
    SCIP_CALL_ABORT(SCIPincludeDefaultPlugins(scip));
    SCIP_CALL_ABORT(SCIPcreateProb(scip, "QAP_RTL1", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    
    // Create variables
    XVarMap x;
    YVarMap y;
    create_x_variables(scip, n, x);
    create_y_variables(scip, n, y);
    
    // Add constraints
    add_assignment_constraints(scip, n, x);
    add_rlt1_constraints(scip, n, x, y);
    
    // Set objective
    set_objective(scip, n, qap, y);
    
    // Configure solver
    SCIP_CALL_ABORT(SCIPsetRealParam(scip, "limits/time", time_limit));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "presolving/maxrounds", 1));
    
    // Solve
    std::cout << "Starting optimization..." << std::endl;
    SCIP_CALL_ABORT(SCIPsolve(scip));
    
    // Get results
    extract_solution(scip, n, x, qap, assignment, objective);
    lower_bound = SCIPgetDualbound(scip);
    
    // Print status
    SCIP_STATUS status = SCIPgetStatus(scip);
    std::cout << "SCIP Status: " << (int)status << std::endl;
    std::cout << "Objective: " << objective << std::endl;
    std::cout << "Lower Bound: " << lower_bound << std::endl;
    
    // Cleanup
    SCIP_CALL_ABORT(SCIPfree(&scip));
    
    return 0;
}

/**
 * Command line interface
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance.dat> [time_limit]" << std::endl;
        return 1;
    }
    
    std::string instance_file = argv[1];
    double time_limit = 3600.0;  // Default 1 hour
    
    if (argc > 2) {
        time_limit = std::stod(argv[2]);
    }
    
    std::vector<int> assignment;
    double objective, lower_bound;
    
    int result = solve_qap_rtl1(instance_file, time_limit, assignment, objective, lower_bound);
    
    if (result == 0 && !assignment.empty()) {
        std::cout << "\nFinal Assignment: [";
        for (int i = 0; i < (int)assignment.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << assignment[i];
        }
        std::cout << "]" << std::endl;
    }
    
    return result;
}
