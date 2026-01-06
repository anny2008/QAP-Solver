/**
 * SFD SCIP QAP Solver - Main Implementation
 * 
 * Implements Subgraph Flow Decomposition (SFD/Form3) for Quadratic Assignment Problem
 * using SCIP 9.2.0.
 * 
 * Based on: QAP_New_formulation/FORM3/qap_new_formulation.py
 */

#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <optional>
#include "../../core/problem.h"
#include "../../include/qap_solution_io.hpp"

// Type aliases for readability
using XVarMap = std::map<std::pair<int,int>, SCIP_VAR*>;
using EVarMap = std::map<std::tuple<int,int,int>, SCIP_VAR*>;

/**
 * Subgraph structure for flow decomposition
 * Each subgraph k contains: (f_k, edge_set, node_set)
 * - f_k: flow value
 * - edge_set: edge set (u,v) pairs
 * - node_set: node set
 */
struct Subgraph {
    double f_k;                              // Flow value
    std::vector<std::pair<int,int>> edge_set;     // Edge set
    std::set<int> node_set;                     // Node set
    std::map<int, int> degree_out;               // Out-degree for each node
    std::map<int, int> degree_in;                // In-degree for each node
};

// Using Problem class from cpp/core/problem.h

/**
 * Decompose flow matrix by value layer
 * (Python: decompose_flow_by_value_layer)
 */
std::vector<Subgraph> decompose_flow_by_value_layer(const Problem& qap) {
    int n = qap.n;
    std::vector<std::vector<double>> flows = qap.F;
    std::vector<Subgraph> subgraphs;
    
    while (true) {
        // Find minimum non-zero flow
        double min_flow = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (flows[i][j] > 1e-9) {
                    if (min_flow < 1e-9 || flows[i][j] < min_flow) {
                        min_flow = flows[i][j];
                    }
                }
            }
        }
        
        if (min_flow <= 0) break;  // No more flow
        
        // Create subgraph with edges having flow >= min_flow
        Subgraph sg;
        sg.f_k = min_flow;
        
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                if (flows[u][v] >= min_flow - 1e-9) {
                    sg.edge_set.push_back({u, v});
                    sg.node_set.insert(u);
                    sg.node_set.insert(v);
                }
            }
        }
        
        // Compute degree sequences
        for (const auto& edge : sg.edge_set) {
            int u = edge.first;
            int v = edge.second;
            sg.degree_out[u]++;
            sg.degree_in[v]++;
        }
        
        subgraphs.push_back(sg);
        
        // Subtract min_flow from edges with flow >= min_flow
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (flows[i][j] >= min_flow - 1e-9) {
                    flows[i][j] -= min_flow;
                    if (flows[i][j] < 1e-9) {
                        flows[i][j] = 0.0;
                    }
                }
            }
        }
    }
    
    return subgraphs;
}

/**
 * Decompose flow matrix by value only
 * (Python: decompose_flow_by_value_only)
 */
std::vector<Subgraph> decompose_flow_by_value_only(const Problem& qap) {
    int n = qap.n;
    std::map<double, std::vector<std::pair<int,int>>> temp_subgraphs;
    
    // Group edges by flow value
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            if (qap.F[u][v] > 1e-9) {
                temp_subgraphs[qap.F[u][v]].push_back({u, v});
            }
        }
    }
    
    // Convert to Subgraph format
    std::vector<Subgraph> subgraphs;
    for (const auto& [f_k, edges] : temp_subgraphs) {
        Subgraph sg;
        sg.f_k = f_k;
        sg.edge_set = edges;
        
        for (const auto& edge : edges) {
            sg.node_set.insert(edge.first);
            sg.node_set.insert(edge.second);
            sg.degree_out[edge.first]++;
            sg.degree_in[edge.second]++;
        }
        
        subgraphs.push_back(sg);
    }
    
    return subgraphs;
}

/**
 * Create SFD x variables: x[i,u] ∈ {0,1} for facility i at location u
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
 * Create SFD e variables: e[i,j,k] ∈ [0,1] for edge flows in subgraph k
 */
void create_e_variables(SCIP* scip, int n, int num_subgraphs, EVarMap& e) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < num_subgraphs; ++k) {
                SCIP_VAR* var = nullptr;
                char varname[256];
                snprintf(varname, 255, "e_%d_%d_%d", i, j, k);
                
                SCIP_CALL_ABORT(SCIPcreateVarBasic(scip, &var, varname, 0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
                SCIP_CALL_ABORT(SCIPaddVar(scip, var));
                e[{i, j, k}] = var;
            }
        }
    }
}

/**
 * Add assignment constraints:
 * - Each facility to exactly one location: Σ_i x[i,u] = 1
 * - Each location to exactly one facility: Σ_u x[i,u] = 1
 */
void add_assignment_constraints(SCIP* scip, int n, const XVarMap& x) {
    // Facility constraints
    for (int i = 0; i < n; ++i) {
        SCIP_CONS* cons = nullptr;
        char consname[256];
        snprintf(consname, 255, "assign_loc_%d", i);
        
        SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 1.0, 1.0));
        for (int u = 0; u < n; ++u) {
            SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), 1.0));
        }
        SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
        SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
    }
    
    // Location constraints
    for (int u = 0; u < n; ++u) {
        SCIP_CONS* cons = nullptr;
        char consname[256];
        snprintf(consname, 255, "assign_fac_%d", u);
        
        SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 1.0, 1.0));
        for (int i = 0; i < n; ++i) {
            SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), 1.0));
        }
        SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
        SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
    }
}

/**
 * Add SFD subgraph constraints:
 * - Edge linking: e[i,j,k] >= x[i,u] + Σ_{v:(u,v)∈G_k} x[j,v] - 1
 * - Flow conservation: out-flow = in-flow for each node
 */
void add_subgraph_constraints(SCIP* scip, int n, const XVarMap& x, const EVarMap& e,
                               const std::vector<Subgraph>& subgraphs, bool is_relax) {
    for (size_t k = 0; k < subgraphs.size(); ++k) {
        const Subgraph& sg = subgraphs[k];
        
        // Edge linking constraints (optional if not relaxed)
        if (!is_relax) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    for (int u : sg.node_set) {
                        SCIP_CONS* cons = nullptr;
                        char consname[256];
                        snprintf(consname, 255, "edge_link_%d_%d_%d_%zu", i, j, u, k);
                        
                        // e[i,j,k] >= x[i,u] + Σ_{v:(u,v)∈G_k} x[j,v] - 1
                        SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 
                                                                   0, nullptr, nullptr, -1.0, SCIPinfinity(scip)));
                        
                        // e[i,j,k]
                        SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, e.at({i, j, (int)k}), 1.0));
                        
                        // -x[i,u]
                        SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), -1.0));
                        
                        // -Σ x[j,v] for (u,v) in edge_set
                        for (const auto& edge : sg.edge_set) {
                            if (edge.first == u) {
                                int v = edge.second;
                                SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({j, v}), -1.0));
                            }
                        }
                        
                        SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
                        SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
                    }
                }
            }
        }
        
        // Flow conservation constraints - Out-flow
        // Σ_j e[i,j,k] = Σ_u x[i,u] * degree_out[u]
        for (int i = 0; i < n; ++i) {
            SCIP_CONS* cons = nullptr;
            char consname[256];
            snprintf(consname, 255, "flow_out_%d_%zu", i, k);
            
            SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 0.0, 0.0));
            
            // Σ_j e[i,j,k]
            for (int j = 0; j < n; ++j) {
                SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, e.at({i, j, (int)k}), 1.0));
            }
            
            // -Σ_u x[i,u] * degree_out[u]
            for (const auto& [u, deg] : sg.degree_out) {
                SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, u}), -(double)deg));
            }
            
            SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
            SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
        }
        
        // Flow conservation constraints - In-flow
        // Σ_j e[j,i,k] = Σ_v x[i,v] * degree_in[v]
        for (int i = 0; i < n; ++i) {
            SCIP_CONS* cons = nullptr;
            char consname[256];
            snprintf(consname, 255, "flow_in_%d_%zu", i, k);
            
            SCIP_CALL_ABORT(SCIPcreateConsBasicLinear(scip, &cons, consname, 0, nullptr, nullptr, 0.0, 0.0));
            
            // Σ_j e[j,i,k]
            for (int j = 0; j < n; ++j) {
                SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, e.at({j, i, (int)k}), 1.0));
            }
            
            // -Σ_v x[i,v] * degree_in[v]
            for (const auto& [v, deg] : sg.degree_in) {
                SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, x.at({i, v}), -(double)deg));
            }
            
            SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
            SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
        }
    }
}

/**
 * Set objective function: min Σ_{i,j,k} D[i,j] * f_k * e[i,j,k]
 */
void set_objective(SCIP* scip, int n, const Problem& qap, const EVarMap& e,
                   const std::vector<Subgraph>& subgraphs) {
    // Debug: count objective terms
    int nonzero_coeff_count = 0;
    double max_coeff = 0.0;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < subgraphs.size(); ++k) {
                double coeff = qap.D[i][j] * subgraphs[k].f_k;
                if (coeff > 1e-9) {
                    nonzero_coeff_count++;
                    max_coeff = std::max(max_coeff, coeff);
                    SCIP_CALL_ABORT(SCIPchgVarObj(scip, e.at({i, j, k}), coeff));
                }
            }
        }
    }
    
    std::cout << "Objective: " << nonzero_coeff_count << " nonzero coefficients, max=" << max_coeff << std::endl;
}

/**
 * Load warm-start from QAPLIB solution file (.sln)
 * Keeps loc->fac layout (file format), converts internally when needed
 */
bool load_warmstart(const std::string& filepath, int n, std::vector<int>& loc_to_fac) {
    try {
        // qap::read_warmstart returns map {(location, facility): 1.0} (0-indexed)
        auto warmstart_map = qap::read_warmstart(filepath);

        if ((int)warmstart_map.size() != n) {
            std::cerr << "Warning: Warm-start file size mismatch (expected " << n << ", got " << warmstart_map.size() << ")" << std::endl;
            return false;
        }

        loc_to_fac.assign(n, -1);
        for (const auto& [key, val] : warmstart_map) {
            int loc = key.first;
            int facility = key.second;
            if (val > 0.5) {
                if (loc < 0 || loc >= n || facility < 0 || facility >= n) {
                    std::cerr << "Warning: Invalid loc/fac index in warm-start (loc=" << loc << ", fac=" << facility << ")" << std::endl;
                    return false;
                }
                if (loc_to_fac[loc] != -1) {
                    std::cerr << "Warning: Duplicate location " << loc << " in warm-start" << std::endl;
                    return false;
                }
                loc_to_fac[loc] = facility;
            }
        }

        for (int loc = 0; loc < n; ++loc) {
            if (loc_to_fac[loc] == -1) {
                std::cerr << "Warning: Warm-start missing location " << loc << std::endl;
                return false;
            }
        }

        // Debug: print loaded mapping (loc->fac)
        std::cout << "Loaded warm-start assignment (loc->fac): [";
        for (int loc = 0; loc < n; ++loc) {
            std::cout << loc_to_fac[loc];
            if (loc < n-1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load warm-start: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Set warm-start solution in SCIP
 */
void set_warmstart(SCIP* scip, int n, const Problem& qap, const XVarMap& x, const EVarMap& e, 
                   const std::vector<int>& loc_to_fac, const std::vector<Subgraph>& subgraphs) {
    SCIP_SOL* sol = nullptr;
    SCIP_CALL_ABORT(SCIPcreateSol(scip, &sol, nullptr));
    
    // Set x variables (the assignment)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            double val = (loc_to_fac[i] == u) ? 1.0 : 0.0;
            SCIP_CALL_ABORT(SCIPsetSolVal(scip, sol, x.at({i, u}), val));
        }
    }
    
    // Set e variables based on the assignment and subgraph structure
    // e[i,j,k] = 1 if edge (fac_to_loc[i], fac_to_loc[j]) exists in subgraph k pattern
    double ws_obj = 0.0;
    for (size_t k = 0; k < subgraphs.size(); ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // Check if edge (assignment[i], assignment[j]) exists in subgraph k pattern
                double e_val = 0.0;
                int u = loc_to_fac[i];
                int v = loc_to_fac[j];
                for (const auto& edge : subgraphs[k].edge_set) {
                    if (edge.first == u && edge.second == v) {
                        e_val = 1.0;
                        break;
                    }
                }
                
                if (e_val > 0.5) {
                    ws_obj += qap.D[i][j] * subgraphs[k].f_k;
                }
                
                SCIP_CALL_ABORT(SCIPsetSolVal(scip, sol, e.at({i, j, (int)k}), e_val));
            }
        }
    }
    
    std::cout << "Warmstart Form3 objective value: " << ws_obj << std::endl;
    
    // Check the solution and add it
    SCIP_Bool success;
    SCIP_CALL_ABORT(SCIPcheckSol(scip, sol, TRUE, FALSE, TRUE, TRUE, TRUE, &success));
    
    if (success) {
        std::cout << "Warm-start solution is feasible" << std::endl;
        SCIP_Bool stored;
        SCIP_CALL_ABORT(SCIPaddSol(scip, sol, &stored));
        if (stored) {
            std::cout << "Warm-start solution added to solution pool" << std::endl;
        }
    } else {
        std::cout << "Warning: Warm-start solution is infeasible!" << std::endl;
        std::cout << "  Checking constraint violations..." << std::endl;
        // The solution will print detailed violations when checksolcomplete is TRUE
    }
    
    SCIP_CALL_ABORT(SCIPfreeSol(scip, &sol));
}

/**
 * Extract solution from SCIP
 */
void extract_solution(SCIP* scip, int n, const XVarMap& x, const Problem& qap,
                      std::vector<int>& loc_to_fac, double& objective) {
    SCIP_SOL* bestsol = SCIPgetBestSol(scip);
    if (bestsol == nullptr) {
        std::cout << "No solution found" << std::endl;
        return;
    }
    
    // First collect fac->loc then invert to loc->fac for reporting
    std::vector<int> fac_to_loc(n, -1);
    for (int fac = 0; fac < n; ++fac) {
        for (int loc = 0; loc < n; ++loc) {
            double val = SCIPgetSolVal(scip, bestsol, x.at({loc, fac}));
            if (val > 0.5) {
                loc_to_fac[loc] = fac;
                break;
            }
        }
    }


    // Compute objective using loc->fac mapping (Problem::evaluate expects loc->fac)
    objective = qap.evaluate(loc_to_fac);   
}

/**
 * Main SFD solver function
 */
int solve_qap_sfd(const std::string& instance_file, 
                  const std::string& warmstart_file,
                  const std::string& decomposition,
                  double time_limit,
                  int threads,
                  bool is_relax,
                  bool log_output,
                  std::vector<int>& assignment, 
                  double& objective, 
                  double& lower_bound) {
    // Load instance
    Problem qap = Problem::fromQAPLIB(instance_file);
    
    int n = qap.n;
    std::cout << "Solving QAP instance with n=" << n << std::endl;
    
    // Decompose flow matrix
    std::vector<Subgraph> subgraphs;
    if (decomposition == "value_layer") {
        subgraphs = decompose_flow_by_value_layer(qap);
    } else {
        subgraphs = decompose_flow_by_value_only(qap);
    }
    
    std::cout << "Flow decomposed into " << subgraphs.size() << " subgraphs" << std::endl;
    
    // Create SCIP environment
    SCIP* scip = nullptr;
    SCIP_CALL_ABORT(SCIPcreate(&scip));
    SCIP_CALL_ABORT(SCIPincludeDefaultPlugins(scip));
    SCIP_CALL_ABORT(SCIPcreateProb(scip, "QAP_SFD", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    
    // Create variables
    XVarMap x;
    EVarMap e;
    create_x_variables(scip, n, x);
    create_e_variables(scip, n, subgraphs.size(), e);
    
    // Add constraints
    add_assignment_constraints(scip, n, x);
    add_subgraph_constraints(scip, n, x, e, subgraphs, is_relax);
    
    // Set objective
    set_objective(scip, n, qap, e, subgraphs);
    
    // Load and set warm-start if provided (loc->fac format from file)
    if (!warmstart_file.empty()) {
        std::vector<int> ws_loc_to_fac;
        if (load_warmstart(warmstart_file, n, ws_loc_to_fac)) {
            double ws_objective = qap.evaluate(ws_loc_to_fac);
            std::cout << "Warmstart assignment evaluated to QAP objective (loc->fac): " << ws_objective << std::endl;
            set_warmstart(scip, n, qap, x, e, ws_loc_to_fac, subgraphs);
        }
    }
    
    // Configure solver
    SCIP_CALL_ABORT(SCIPsetRealParam(scip, "limits/time", time_limit));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "parallel/maxnthreads", threads));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "lp/threads", threads));
    
    if (!log_output) {
        SCIP_CALL_ABORT(SCIPsetIntParam(scip, "display/verblevel", 0));
    }
    
    // Solve
    std::cout << "Starting optimization..." << std::endl;
    SCIP_CALL_ABORT(SCIPsolve(scip));
    
    // Get results
    extract_solution(scip, n, x, qap, assignment, objective);
    double scip_primal = SCIPgetPrimalbound(scip);
    lower_bound = SCIPgetDualbound(scip);
    
    // Print status
    SCIP_STATUS status = SCIPgetStatus(scip);
    std::cout << "\nSCIP Status: " << (int)status << std::endl;
    std::cout << "SCIP Primal (model objective): " << scip_primal << std::endl;
    std::cout << "SCIP Dual   (model objective): " << lower_bound << std::endl;
    std::cout << "QAP Objective (loc->fac): " << objective << std::endl;

    if (scip_primal > 0 && lower_bound > 0) {
        double gap = 100.0 * (scip_primal - lower_bound) / lower_bound;
        std::cout << "Gap: " << gap << "%" << std::endl;
    }
    
    // Release all variables before freeing SCIP
    for (auto& [key, var] : x) {
        if (var != nullptr) SCIP_CALL_ABORT(SCIPreleaseVar(scip, &var));
    }
    for (auto& [key, var] : e) {
        if (var != nullptr) SCIP_CALL_ABORT(SCIPreleaseVar(scip, &var));
    }
    
    // Cleanup
    SCIP_CALL_ABORT(SCIPfree(&scip));
    
    return 0;
}

/**
 * Command line interface
 */
int main(int argc, char** argv) {
    auto usage = [&](const std::string& prog){
        std::cerr << "Usage: " << prog << " <instance.dat> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --config <file.json>     : JSON config with keys (instance, warmstart, decomposition, time, threads, relax, log)\n";
        std::cerr << "  --warmstart <file.sln>   : Warm-start solution file\n";
        std::cerr << "  --decomposition <type>   : value_layer (default) or value_only\n";
        std::cerr << "  --time <seconds>         : Time limit (default: 3600)\n";
        std::cerr << "  --threads <n>            : Number of threads (default: 4)\n";
        std::cerr << "  --relax                  : Use relaxed constraints\n";
        std::cerr << "  --log                    : Enable solver output\n";
    };

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::string instance_file;
    std::string warmstart_file = "";
    std::string decomposition = "value_layer";
    double time_limit = 3600.0;
    int threads = 4;
    bool is_relax = false;
    bool log_output = false;
    std::optional<std::string> config_path;

    // Helper lambdas for naive JSON parsing
    auto read_file = [](const std::string& path) -> std::string {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open config file: " + path);
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };

    auto extract_string = [](const std::string& text, const std::string& key) -> std::optional<std::string> {
        auto pos = text.find("\"" + key + "\"");
        if (pos == std::string::npos) return std::nullopt;
        pos = text.find(':', pos);
        if (pos == std::string::npos) return std::nullopt;
        pos = text.find('"', pos);
        if (pos == std::string::npos) return std::nullopt;
        auto end = text.find('"', pos + 1);
        if (end == std::string::npos) return std::nullopt;
        return text.substr(pos + 1, end - pos - 1);
    };

    auto extract_number = [](const std::string& text, const std::string& key) -> std::optional<double> {
        auto pos = text.find("\"" + key + "\"");
        if (pos == std::string::npos) return std::nullopt;
        pos = text.find(':', pos);
        if (pos == std::string::npos) return std::nullopt;
        // skip spaces
        while (pos < text.size() && (text[pos] == ':' || std::isspace(text[pos]))) pos++;
        size_t end = pos;
        while (end < text.size() && (std::isdigit(text[end]) || text[end]=='-' || text[end]=='+' || text[end]=='.' || text[end]=='e' || text[end]=='E')) end++;
        if (end == pos) return std::nullopt;
        return std::stod(text.substr(pos, end - pos));
    };

    auto extract_bool = [](const std::string& text, const std::string& key) -> std::optional<bool> {
        auto pos = text.find("\"" + key + "\"");
        if (pos == std::string::npos) return std::nullopt;
        pos = text.find(':', pos);
        if (pos == std::string::npos) return std::nullopt;
        while (pos < text.size() && (text[pos] == ':' || std::isspace(text[pos]))) pos++;
        if (text.compare(pos, 4, "true") == 0) return true;
        if (text.compare(pos, 5, "false") == 0) return false;
        return std::nullopt;
    };

    // Initial positional instance if given
    if (argc >= 2 && argv[1][0] != '-') {
        instance_file = argv[1];
    }

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--warmstart" && i + 1 < argc) {
            warmstart_file = argv[++i];
        } else if (arg == "--decomposition" && i + 1 < argc) {
            decomposition = argv[++i];
        } else if (arg == "--time" && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--relax") {
            is_relax = true;
        } else if (arg == "--log") {
            log_output = true;
        }
    }

    // If config provided, load and apply (config can override defaults; CLI overrides config)
    if (config_path) {
        try {
            std::string cfg = read_file(*config_path);
            if (auto v = extract_string(cfg, "instance")) instance_file = *v;
            if (auto v = extract_string(cfg, "warmstart")) warmstart_file = *v;
            if (auto v = extract_string(cfg, "decomposition")) decomposition = *v;
            if (auto v = extract_number(cfg, "time")) time_limit = *v;
            if (auto v = extract_number(cfg, "threads")) threads = static_cast<int>(*v);
            if (auto v = extract_bool(cfg, "relax")) is_relax = *v;
            if (auto v = extract_bool(cfg, "log")) log_output = *v;
        } catch (const std::exception& e) {
            std::cerr << "Error reading config: " << e.what() << std::endl;
            return 1;
        }
    }

    if (instance_file.empty()) {
        usage(argv[0]);
        return 1;
    }
    
    std::vector<int> assignment;
    double objective, lower_bound;
    
    int result = solve_qap_sfd(instance_file, warmstart_file, decomposition,
                               time_limit, threads, is_relax, log_output,
                               assignment, objective, lower_bound);
    
    if (result == 0 && !assignment.empty()) {
        std::cout << "\nFinal Assignment (loc->fac): [";
        for (int i = 0; i < (int)assignment.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << assignment[i];
        }
        std::cout << "]" << std::endl;
    }
    
    return result;
}
