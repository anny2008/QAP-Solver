/**
 * RLT1 SCIP QAP Solver - Main Implementation
 * 
 * Implements the Reformulation-Linearization Technique (RLT1) for Quadratic Assignment Problem
 * using SCIP 9.2.0.
 * 
 * Unified interface with JSON config, warmstart support, and JSON output.
 */

#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include "../../core/problem.h"
#include "../../include/qap_solution_io.hpp"
#include "../rlt1_volume/rlt1_volume_solver.hpp"


// Type aliases for readability
using XVarMap = std::map<std::pair<int,int>, SCIP_VAR*>;
using YVarMap = std::map<std::tuple<int,int,int,int>, SCIP_VAR*>;



struct VolumeRelaxData {
    explicit VolumeRelaxData(const Problem& p) : problem(p) {}
    Problem problem;
    int n = 0;
    int threads = 1;
    double time_limit = 0.0;
    bool verbose = false;
    bool relaxation_info = false;
    std::vector<SCIP_VAR*> x_vars;
    std::vector<SCIP_VAR*> y_vars;
    VOL_dvector last_dual;  // warm-start dual from previous iteration
    std::string formulation_name;
};

static SCIP_DECL_RELAXEXEC(relaxExecVolume);
static SCIP_DECL_RELAXFREE(relaxFreeVolume);
static SCIP_RETCODE includeVolumeRelax(SCIP* scip,
                                       const Problem& problem,
                                       int threads,
                                       double time_limit,
                                       bool verbose,
                                       const XVarMap& x,
                                       const YVarMap& y,
                                       bool relaxation_info,
                                       const std::string& formulation_name);

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
 * Release all created variables to avoid SCIP leak warnings
 */
void release_variables(SCIP* scip, XVarMap& x, YVarMap& y) {
    for (auto& kv : x) {
        SCIP_CALL_ABORT(SCIPreleaseVar(scip, &kv.second));
    }
    for (auto& kv : y) {
        SCIP_CALL_ABORT(SCIPreleaseVar(scip, &kv.second));
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
 * Set objective function: min Σ F[u,v] * D[i,j] * y[i,u,j,v]
 * Corrected formula matching Python RLT1 implementation
 */
void set_objective(SCIP* scip, int n, const Problem& problem, const YVarMap& y) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    // Objective: sum_{i,j,u,v} F[u][v] * D[i][j] * y[i,u,j,v]
                    double coeff = problem.F[u][v] * problem.D[i][j];
                    SCIP_CALL_ABORT(SCIPchgVarObj(scip, y.at({i, u, j, v}), coeff));
                }
            }
        }
    }
}

/**
 * Build FixedVariables structure from map of fixed variable indices and values
 */
 #define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v)) // key for y_fixed maps
FixedVariables build_fixed_from_map(const std::unordered_map<int, double> &map_fixed, int n) {
    FixedVariables fv;
    fv.map_fixed = const_cast<std::unordered_map<int, double>&>(map_fixed); // store reference to map
    for (const auto &kv : map_fixed) {
        int idx = kv.first;
        double val = kv.second;
        if (std::abs(val) > 1e-9 && std::abs(val - 1.0) > 1e-9) {
            continue; // ignore fractional fixes
        }
        int bval = (val >= 0.5) ? 1 : 0;
        if (idx < n * n) {
            int i = idx / n;
            int u = idx % n;
            if (bval == 1) fv.x_fixed_1[i] = u; else fv.x_fixed_0[i].insert(u);
        } else {
            int offset = idx - n * n;
            int i = offset / (n * n * n);
            int rem = offset % (n * n * n);
            int u = rem / (n * n);
            rem = rem % (n * n);
            int j = rem / n;
            int v = rem % n;
            int key = key_y(n, i, u, v);
            if (bval == 1) fv.y_fixed_1[key] = j; else fv.y_fixed_0[key].insert(j);
        }
    }
    return fv;
}

/**
 * Volume-based relaxation handler: gather fixings, call Volume, feed bound/primal to SCIP.
 */
static SCIP_DECL_RELAXEXEC(relaxExecVolume) {
    auto* data = reinterpret_cast<VolumeRelaxData*>(SCIPrelaxGetData(relax));
    if (data == nullptr) {
        *result = SCIP_DIDNOTRUN;
        return SCIP_OKAY;
    }

    const int n = data->n;
    const int num_x = n * n;
    const int num_y = n * n * n * n;

    std::unordered_map<int, double> map_fixed;

    // Collect x fixings
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            const int idx = i * n + u;
            SCIP_VAR* var = data->x_vars[idx];
            if (var == nullptr) continue;
            double lb = SCIPvarGetLbLocal(var);
            double ub = SCIPvarGetUbLocal(var);
            if (SCIPisEQ(scip, lb, ub)) {
                map_fixed[idx] = lb;
            }
        }
    }

    // Collect y fixings
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    const int idx = i * n * n * n + u * n * n + j * n + v;
                    SCIP_VAR* var = data->y_vars[idx];
                    if (var == nullptr) continue;
                    double lb = SCIPvarGetLbLocal(var);
                    double ub = SCIPvarGetUbLocal(var);
                    if (SCIPisEQ(scip, lb, ub)) {
                        map_fixed[num_x + idx] = lb;
                    }
                }
            }
        }
    }

    FixedVariables fixed = build_fixed_from_map(map_fixed, n);
    VolumeResult res = solve_rlt1_volume_relax(data->problem, fixed, false, data->last_dual, data->formulation_name);
    std::cout << "Volume solving status: " << res.status << std::flush;
    if (res.status != 0) {
        *result = SCIP_DIDNOTRUN;
        return SCIP_OKAY;
    }

    // Store dual for warm-start in next iteration
    if (res.dual.size() > 0) {
        data->last_dual = res.dual;
    }

    // // Populate relaxation solution (best effort if sizes match)
    // if (res.primal.size() >= static_cast<size_t>(num_x)) {
    //     SCIPclearRelaxSolVals(scip, relax);
    //     for (int idx = 0; idx < num_x && idx < (int)res.primal.size(); ++idx) {
    //         SCIP_VAR* var = data->x_vars[idx];
    //         if (var != nullptr) {
    //             SCIPsetRelaxSolVal(scip, relax, var, res.primal[idx]);
    //         }
    //     }
    //     const int yoffset = num_x;
    //     for (int idx = 0; idx < num_y && yoffset + idx < (int)res.primal.size(); ++idx) {
    //         SCIP_VAR* var = data->y_vars[idx];
    //         if (var != nullptr) {
    //             SCIPsetRelaxSolVal(scip, relax, var, res.primal[yoffset + idx]);
    //         }
    //     }
    //     SCIPmarkRelaxSolValid(scip, relax, 0u);
    // }

    *lowerbound = res.lower_bound;
    *result = SCIP_SUCCESS;
    std::cout << "\rVolume relaxation solved: lower bound = " << std::fixed << std::setprecision(6) << res.lower_bound << std::flush;
    return SCIP_OKAY;
}

static SCIP_DECL_RELAXFREE(relaxFreeVolume) {
    auto* data = reinterpret_cast<VolumeRelaxData*>(SCIPrelaxGetData(relax));
    delete data;
    return SCIP_OKAY;
}

static SCIP_RETCODE includeVolumeRelax(SCIP* scip,
                                       const Problem& problem,
                                       int threads,
                                       double time_limit,
                                       bool verbose,
                                       const XVarMap& x,
                                       const YVarMap& y,
                                       bool relaxation_info,
                                       const std::string& formulation_name) {
    const int n = problem.n;
    auto* data = new VolumeRelaxData(problem);
    data->n = n;
    data->threads = threads;
    data->time_limit = time_limit;
    data->verbose = verbose;
    data->relaxation_info = relaxation_info;
    data->x_vars.assign(n * n, nullptr);
    data->y_vars.assign(n * n * n * n, nullptr);
    data->formulation_name = formulation_name;
    
     // Map variables

    for (const auto& kv : x) {
        int i = kv.first.first;
        int u = kv.first.second;
        data->x_vars[i * n + u] = kv.second;
    }
    for (const auto& kv : y) {
        int i = std::get<0>(kv.first);
        int u = std::get<1>(kv.first);
        int j = std::get<2>(kv.first);
        int v = std::get<3>(kv.first);
        data->y_vars[i * n * n * n + u * n * n + j * n + v] = kv.second;
    }

    static bool g_relaxation_info = false;
    g_relaxation_info = relaxation_info;
    SCIP_RELAX* relax = nullptr;
    SCIP_CALL( SCIPincludeRelaxBasic(
        scip,
        &relax,
        "rlt1_volume",
        "RLT1 Volume relaxation",
        100000,   // high priority
        1,        // run at every node
        relaxExecVolume,
        (SCIP_RELAXDATA*)data) );

    SCIP_CALL( SCIPsetRelaxFree(scip, relax, relaxFreeVolume) );
    return SCIP_OKAY;
}

/**
 * Load warm-start from QAPLIB solution file (.sln)
 */
bool load_warmstart(const std::string& filepath, int n, std::vector<int>& assignment, double& ws_obj) {
    try {
        int file_n;
        std::vector<int> loc_to_fac;
        qap::read_solution(filepath, file_n, loc_to_fac, ws_obj);
        
        if (file_n != n) {
            std::cerr << "Warning: Warm-start file size mismatch (expected " << n << ", got " << file_n << ")" << std::endl;
            return false;
        }

        // Validate permutation (loc -> facility)
        assignment.assign(n, -1);
        for (int loc = 0; loc < n; ++loc) {
            int fac = loc_to_fac[loc];
            if (fac < 0 || fac >= n) {
                std::cerr << "Warning: Invalid facility " << fac << " in warm-start" << std::endl;
                return false;
            }
            assignment[loc] = fac;
        }
        
        std::cout << "Loaded warm-start assignment (location->facility): [";
        for (int i = 0; i < n; ++i) {
            std::cout << assignment[i];
            if (i < n-1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Warm-start objective (from file): " << ws_obj << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load warm-start: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Set warm-start solution in SCIP
 */
void set_warmstart(SCIP* scip, int n, const XVarMap& x, const YVarMap& y, 
                   const std::vector<int>& assignment) {
    SCIP_SOL* sol = nullptr;
    SCIP_CALL_ABORT(SCIPcreateSol(scip, &sol, nullptr));
    
    // Set x variables
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            double val = (assignment[i] == u) ? 1.0 : 0.0;
            SCIP_CALL_ABORT(SCIPsetSolVal(scip, sol, x.at({i, u}), val));
        }
    }
    
    // Set y variables: y[i,u,j,v] = x[i,u] * x[j,v]
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    double val = (assignment[i] == u && assignment[j] == v) ? 1.0 : 0.0;
                    SCIP_CALL_ABORT(SCIPsetSolVal(scip, sol, y.at({i, u, j, v}), val));
                }
            }
        }
    }
    
    // Check and add solution
    SCIP_Bool success;
    SCIP_CALL_ABORT(SCIPcheckSol(scip, sol, TRUE, FALSE, TRUE, TRUE, TRUE, &success));
    
    if (success) {
        std::cout << "✓ Warm-start solution is feasible" << std::endl;
        SCIP_Bool stored;
        SCIP_CALL_ABORT(SCIPaddSol(scip, sol, &stored));
        if (stored) {
            std::cout << "✓ Warm-start solution added to SCIP" << std::endl;
        }
    } else {
        std::cout << "⚠ Warm-start solution is infeasible (will be ignored)" << std::endl;
    }
    
    SCIP_CALL_ABORT(SCIPfreeSol(scip, &sol));
}

/**
 * Extract solution from SCIP
 */
void extract_solution(SCIP* scip, int n, const XVarMap& x, const Problem& problem, 
                      std::vector<int>& assignment, double& objective) {
    SCIP_SOL* bestsol = SCIPgetBestSol(scip);
    if (bestsol == nullptr) {
        std::cout << "No solution found" << std::endl;
        return;
    }
    
    assignment.resize(n);
    objective = 0.0;
    
    // Extract assignment from x variables (location -> facility)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            double val = SCIPgetSolVal(scip, bestsol, x.at({i, u}));
            if (val > 0.9) {  // Binary variable, close to 1
                assignment[i] = u;
                break;
            }
        }
    }
    
    // Compute objective: sum_{i,j} F[assignment[i]][assignment[j]] * D[i][j]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            objective += problem.F[assignment[i]][assignment[j]] * problem.D[i][j];
        }
    }
}

/**
 * Write JSON result file
 */
void write_json_result(const std::string& filepath, const std::string& instance_name,
                       const std::vector<int>& assignment, double objective, 
                       double lower_bound, double time_sec) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot write JSON output to " << filepath << std::endl;
        return;
    }
    
    double gap = 0.0;
    if (lower_bound > 0 && objective > lower_bound) {
        gap = 100.0 * (objective - lower_bound) / lower_bound;
    }
    
    file << "{\n";
    file << "  \"instance\": \"" << instance_name << "\",\n";
    file << "  \"solver\": \"rlt1_scip_cpp\",\n";
    file << "  \"formulation\": \"rlt1\",\n";
    
    if (!assignment.empty()) {
        file << "  \"objective\": " << std::fixed << std::setprecision(1) << objective << ",\n";
        file << "  \"assignment\": [";
        for (size_t i = 0; i < assignment.size(); ++i) {
            if (i > 0) file << ", ";
            file << assignment[i];
        }
        file << "],\n";
    } else {
        file << "  \"objective\": null,\n";
        file << "  \"assignment\": null,\n";
    }
    
    file << "  \"lower_bound\": " << std::fixed << std::setprecision(6) << lower_bound << ",\n";
    file << "  \"gap\": " << std::fixed << std::setprecision(2) << gap << ",\n";
    file << "  \"time\": " << std::fixed << std::setprecision(2) << time_sec << "\n";
    file << "}\n";
    
    std::cout << "Results written to " << filepath << std::endl;
}

/**
 * Main solver function with unified interface
 */
int solve_qap_rlt1(const std::string& instance_file, 
                   const std::string& warmstart_file,
                   double time_limit,
                   int threads,
                   bool log_output,
                   const std::string& relaxation_mode,
                   bool relaxation_info,
                   const std::string& formulation_name,
                   std::vector<int>& assignment, 
                   double& objective, 
                   double& lower_bound) {
    
    // Load problem
    Problem problem = Problem::fromQAPLIB(instance_file);
    int n = problem.n;
    
    if (log_output) {
        std::cout << "Solving QAP instance: n=" << n << std::endl;
        std::cout << "Time limit: " << time_limit << "s" << std::endl;
        std::cout << "Threads: " << threads << std::endl;
    }
    
    // Load warmstart if provided
    std::vector<int> ws_assignment;
    double ws_obj = 0.0;
    bool has_warmstart = false;
    
    if (!warmstart_file.empty()) {
        has_warmstart = load_warmstart(warmstart_file, n, ws_assignment, ws_obj);
    }
    
    // Create SCIP environment
    SCIP* scip = nullptr;
    SCIP_CALL_ABORT(SCIPcreate(&scip));
    SCIP_CALL_ABORT(SCIPincludeDefaultPlugins(scip));
    
    // Try to minimize LP solver workload (optional parameters, may not exist in SCIP)
    SCIP_CALL_ABORT(SCIPsetBoolParam(scip, "lp/alwaysgetduals", FALSE));  // Don't request LP duals
    
    SCIP_CALL_ABORT(SCIPcreateProb(scip, "QAP_RLT1", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    
    // Set output
    if (!log_output) {
        SCIP_CALL_ABORT(SCIPsetIntParam(scip, "display/verblevel", 0));
    }
    
    // Create variables
    XVarMap x;
    YVarMap y;
    create_x_variables(scip, n, x);
    create_y_variables(scip, n, y);
    
    // Add constraints
    add_assignment_constraints(scip, n, x);
    add_rlt1_constraints(scip, n, x, y);
    
    // Set objective
    set_objective(scip, n, problem, y);

    // Include custom relaxation handler if requested
    if (relaxation_mode == "volume") {
        SCIP_CALL_ABORT(includeVolumeRelax(scip, problem, threads, time_limit, log_output || relaxation_info, x, y, relaxation_info, formulation_name));
        if (log_output) {
            std::cout << "Using custom relaxation: volume" << std::endl;
        }
        SCIP_CALL_ABORT( SCIPsetIntParam(scip, "lp/solvefreq", -1) ); // Disable default LP relaxations
    } else if (log_output) {
        std::cout << "Using default SCIP relaxation (no custom relax)" << std::endl;
    }
    
    // Set warmstart if available
    if (has_warmstart) {
        set_warmstart(scip, n, x, y, ws_assignment);
    }
    
    // Configure solver
    SCIP_CALL_ABORT(SCIPsetRealParam(scip, "limits/time", time_limit));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "parallel/maxnthreads", threads));
    
    // Solve
    if (log_output) {
        std::cout << "\nStarting optimization...\n" << std::endl;
    }
    
    SCIP_CALL_ABORT(SCIPsolve(scip));
    
    // Get results
    extract_solution(scip, n, x, problem, assignment, objective);
    lower_bound = SCIPgetDualbound(scip);
    
    // Print status
    if (log_output) {
        SCIP_STATUS status = SCIPgetStatus(scip);
        std::cout << "\nSCIP Status: " << (int)status << std::endl;
        std::cout << "Objective: " << objective << std::endl;
        std::cout << "Lower Bound: " << lower_bound << std::endl;
        
        if (lower_bound > 0 && objective > lower_bound) {
            double gap = 100.0 * (objective - lower_bound) / lower_bound;
            std::cout << "Gap: " << std::fixed << std::setprecision(2) << gap << "%" << std::endl;
        }
    }
    
    // Cleanup
    release_variables(scip, x, y);
    SCIP_CALL_ABORT(SCIPfree(&scip));
    
    return 0;
}

/**
 * Command line interface with unified arguments
 * 
 * Usage: rlt1_solver <instance.dat> [options]
 *   --warmstart <file>  : Warm-start solution file (.sln)
 *   --output <file>     : JSON output file (default: none)
 *   --time <seconds>    : Time limit (default: 3600)
 *   --threads <n>       : Number of threads (default: 4)
 *   --log               : Enable detailed output
 *   --relaxation <mode> : Relaxation mode: 'default' or 'volume' (default: volume)
 *   --relaxation-info   : Print extra info from relaxation handler (value, fixed vars, violations)
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance.dat> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --warmstart <file>  : Warm-start solution file (.sln)\n";
        std::cerr << "  --output <file>     : JSON output file\n";
        std::cerr << "  --time <seconds>    : Time limit (default: 3600)\n";
        std::cerr << "  --threads <n>       : Number of threads (default: 4)\n";
        std::cerr << "  --log               : Enable detailed output\n";
        return 1;
    }
    std::cout.setf(std::ios::unitbuf);
    
    std::string instance_file = argv[1];
    std::string warmstart_file = "";
    std::string output_file = "";
    double time_limit = 3600.0;
    int threads = 4;
    bool log_output = false;
    std::string relaxation_mode = "volume"; // default keeps current behavior
    std::string formulation_name = "formulation4"; // default

    bool relaxation_info = false;
    
    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmstart" && i + 1 < argc) {
            warmstart_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--time" && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--log") {
            log_output = true;
        } else if (arg == "--relaxation" && i + 1 < argc) {
            relaxation_mode = argv[++i];
            if (relaxation_mode != "default" && relaxation_mode != "volume") {
                std::cerr << "Unknown relaxation mode: " << relaxation_mode << ". Use 'default' or 'volume'." << std::endl;
                return 1;
            }
        } else if (arg == "--relaxation-info") {
            relaxation_info = true;
        } else if (arg == "--formulation" && i + 1 < argc) {
            formulation_name = argv[++i];
        }
    }
    
    // Extract instance name
    std::string instance_name = instance_file;
    size_t last_slash = instance_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        instance_name = instance_name.substr(last_slash + 1);
    }
    size_t dot = instance_name.find_last_of(".");
    if (dot != std::string::npos) {
        instance_name = instance_name.substr(0, dot);
    }
    
    // Solve
    std::vector<int> assignment;
    double objective = 0.0, lower_bound = 0.0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int result = solve_qap_rlt1(instance_file, warmstart_file, time_limit, threads, log_output,
                                relaxation_mode, relaxation_info, formulation_name, assignment, objective, lower_bound);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    
    // Print final solution
    if (result == 0 && !assignment.empty()) {
        std::cout << "\nFinal Assignment: [";
        for (size_t i = 0; i < assignment.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << assignment[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "Objective: " << objective << std::endl;
        std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_sec << "s" << std::endl;
    }
    
    // Write JSON output if requested
    if (!output_file.empty()) {
        write_json_result(output_file, instance_name, assignment, objective, lower_bound, elapsed_sec);
    }
    
    return result;
}
