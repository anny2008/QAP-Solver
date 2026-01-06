/**
 * RTL1 Volume Algorithm Solver for QAP
 * 
 * Implements the Lagrangian relaxation-based Volume algorithm
 * for the RTL1 formulation of the Quadratic Assignment Problem.
 * 
 * Based on:
 * - Barahona & Anbil (1998): "The Volume algorithm: producing primal 
 *   solutions with a subgradient method"
 * - RTL1 formulation for QAP
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <omp.h>

#include "VolVolume.hpp"
#include "../../core/problem.h"
#include "../../include/qap_solution_io.hpp"

// Summary of primal violations for reporting at end of run
struct PrimalViolationSummary {
    double max_abs = 0.0;
    double avg_abs = 0.0;
    double assignment_max = 0.0;
    double assignment_avg = 0.0;
    double link_max = 0.0;
    double link_avg = 0.0;
    int n = 0;
};

static PrimalViolationSummary compute_primal_violation_summary(const VOL_dvector &psol, int n) {
    PrimalViolationSummary summary;
    summary.n = n;

    auto idx_x = [n](int i, int u) { return i * n + u; };
    auto idx_y = [n](int i, int u, int j, int v) {
        return n * n + i * n * n * n + u * n * n + j * n + v;
    };

    double total_abs = 0.0;
    long long total_cnt = 0;

    // Assignment: sum_i x[i,u] = 1 for all u
    double assign_abs_sum = 0.0;
    long long assign_cnt = 0;
    for (int u = 0; u < n; ++u) {
        double residual = 1.0;
        for (int i = 0; i < n; ++i) {
            residual -= psol[idx_x(i, u)];
        }
        double a = std::abs(residual);
        summary.assignment_max = std::max(summary.assignment_max, a);
        assign_abs_sum += a;
        ++assign_cnt;
    }
    summary.assignment_avg = assign_cnt > 0 ? assign_abs_sum / assign_cnt : 0.0;

    // Linking: sum_v y[i,u,j,v] = x[i,u] for all i,u,j
    double link_abs_sum = 0.0;
    long long link_cnt = 0;
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                double residual = psol[idx_x(i, u)];
                for (int v = 0; v < n; ++v) {
                    residual -= psol[idx_y(i, u, j, v)];
                }
                double a = std::abs(residual);
                summary.link_max = std::max(summary.link_max, a);
                link_abs_sum += a;
                ++link_cnt;
            }
        }
    }
    summary.link_avg = link_cnt > 0 ? link_abs_sum / link_cnt : 0.0;

    // Symmetry: y[i,u,j,v] = y[j,v,i,u] for all i,u,j,v
    double sym_abs_sum = 0.0;
    long long sym_cnt = 0;
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    double residual = psol[idx_y(i, u, j, v)] - psol[idx_y(j, v, i, u)];
                    double a = std::abs(residual);
                    sym_abs_sum += a;
                    ++sym_cnt;
                    summary.max_abs = std::max(summary.max_abs, a);
                }
            }
        }
    }

    total_abs = assign_abs_sum + link_abs_sum + sym_abs_sum;
    total_cnt = assign_cnt + link_cnt + sym_cnt;
    summary.max_abs = std::max(summary.max_abs, summary.assignment_max);
    summary.max_abs = std::max(summary.max_abs, summary.link_max);
    summary.avg_abs = total_cnt > 0 ? total_abs / static_cast<double>(total_cnt) : 0.0;

    return summary;
}

// Macro definitions for cleaner indexing
#define mu(u) (pi[u])
#define lambda(i, u, j, v) (pi[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define theta(i, u, j) (pi[n + n * n * n * n + (i) * n * n + (u) * n + (j)])
#define x(i, u) (psol[(i) * n + (u)])
#define y(i, u, j, v) (psol[n * n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_mu(u) (vio[u])
#define vio_lambda(i, u, j, v) (vio[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_theta(i, u, j) (vio[n + n * n * n * n + (i) * n * n + (u) * n + (j)])

/**
 * RTL1 Volume Hooks Implementation
 */
class RTL1VolumeHooks : public VOL_user_hooks {
private:
    const Problem& qap_data;
    int n;
    
    // Working arrays for subproblem
    std::vector<double> beta;       // beta[i,u,v] = min_j cost of y[i,u,j,v]=1
    std::vector<int> beta_j_ind;    // j that achieves beta[i,u,v]
    std::vector<double> alpha;      // alpha[i] = min_u cost of x[i,u]=1
    std::vector<int> alpha_u_ind;   // u that achieves alpha[i]
    
public:
    RTL1VolumeHooks(const Problem& data) 
        : qap_data(data), n(data.n) {
        beta.resize(n * n * n);
        beta_j_ind.resize(n * n * n);
        alpha.resize(n);
        alpha_u_ind.resize(n);
    }
    
    // Compute reduced costs (not used in RTL1 subproblem)
    virtual int compute_rc(const VOL_dvector& /*pi*/, VOL_dvector& rc) override {
        rc = 0;
        return 0;
    }
    
    /**
     * Solve RTL1 Lagrangian Subproblem
     * 
     * DUAL VARIABLES (pi):
     *   - mu[u]:           Lagrange multipliers for sum_i x[i,u] = 1
     *   - lambda[i,u,j,v]: Multipliers for y[i,u,j,v] = y[j,v,i,u]
     *   - theta[i,u,j]:    Multipliers for sum_v y[i,u,j,v] = x[i,u]
     * 
     * LAGRANGIAN:
     *   L = sum_{i,j,u,v} d[i,j]*f[u,v]*y[i,u,j,v]
     *       + sum_u mu[u] * (1 - sum_i x[i,u])
     *       + sum_{i,u,j,v} lambda[i,u,j,v] * (y[j,v,i,u] - y[i,u,j,v])
     *        + sum_{i,u,j} theta[i,u,j] * (x[i,u] - sum_v y[i,u,j,v])
     * 
     * ALGORITHM:
     *   1. Compute beta[i,u,v] = min_j {Lagrangian cost of y[i,u,j,v]=1}
     *   2. Compute alpha[i] = min_u {cost of x[i,u]=1 + sum_{j,v} beta[i,u,v]}
     *   3. Set x[i,u]=1 for u minimizing alpha[i]
     *   4. Set y[i,u,j,v]=1 for j minimizing beta[i,u,v]
     */
    virtual int solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                                double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                                double& pcost) override {
        
        // Initialize working arrays
        std::fill(beta.begin(), beta.end(), 0.0);
        std::fill(beta_j_ind.begin(), beta_j_ind.end(), 0);
        std::fill(alpha.begin(), alpha.end(), 0.0);
        std::fill(alpha_u_ind.begin(), alpha_u_ind.end(), 0);
        
        // STEP 1: Compute beta[i,u,v] = min_j {Lagrangian cost of y[i,u,j,v]=1}
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    double min_cost = DBL_MAX;
                    int best_j = 0;
                    
                    for (int j = 0; j < n; ++j) {
                        // Cost of setting y[i,u,j,v] = 1 (original objective)
                        double cost = qap_data.D[i][j] * qap_data.F[u][v];

                        // Symmetry dual counted once: -lambda(i,u,j,v) if i<=j, +lambda(j,v,i,u) if i>=j
                        if (i <= j) {
                            cost -= lambda(i, u, j, v);
                        }
                        if (i >= j) {
                            cost += lambda(j, v, i, u);
                        }

                        // Linking dual: -theta(i,u,j)
                        cost -= theta(i, u, j);

                        if (cost < min_cost) {
                            min_cost = cost;
                            best_j = j;
                        }
                    }
                    
                    int idx = i * n * n + u * n + v;
                    beta[idx] = min_cost;
                    beta_j_ind[idx] = best_j;
                }
            }
        }
        
        // STEP 2: Compute alpha[i] = min_u {cost of x[i,u]=1}
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            double min_cost = DBL_MAX;
            int best_u = 0;
            
            for (int u = 0; u < n; ++u) {
                // Assignment dual penalty: -mu[u]
                double cost = -mu(u);
                
                // Linking dual penalty: sum_j theta[i,u,j]
                for (int j = 0; j < n; ++j) {
                    cost += theta(i, u, j);
                }
                
                // Add beta costs for all v (third index) plus linking penalty already added above
                for (int v = 0; v < n; ++v) {
                    int idx = i * n * n + u * n + v;
                    cost += beta[idx];
                }
                
                if (cost < min_cost) {
                    min_cost = cost;
                    best_u = u;
                }
            }
            
            alpha[i] = min_cost;
            alpha_u_ind[i] = best_u;
        }
        
        // STEP 3: Compute Lagrangian lower bound
        lcost = 0.0;
        #pragma omp parallel for reduction(+:lcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            lcost += alpha[i];
        }
        
        // Add constant term: sum_u mu[u]
        for (int u = 0; u < n; ++u) {
            lcost += mu(u);
        }
        
        // STEP 4: Construct primal solution
        psol = 0.0;
        
        // Set x[i,u]=1 for chosen u
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int u = alpha_u_ind[i];
            x(i, u) = 1.0;
        }
        
        // Set y[i,u,j,v]=1 for chosen j at chosen u
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int v = 0; v < n; ++v) {
                int u = alpha_u_ind[i];
                int idx = i * n * n + u * n + v;
                int j = beta_j_ind[idx];
                y(i, u, j, v) = 1.0;
            }
        }
        
        // Compute primal objective (original QAP objective on fractional solution)
        pcost = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:pcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        pcost += qap_data.D[i][j] * qap_data.F[u][v] * y(i, u, j, v);
                    }
                }
            }
        }
        
        // STEP 5: Compute constraint violations
        vio = 0.0;
        
        // Violation of assignment constraints: 1 - sum_i x[i,u]
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < n; ++u) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += x(i, u);
            }
            vio_mu(u) = 1.0 - sum;
        }
        
        // Violation of linking constraints: x[i,u] - sum_v y[i,u,j,v]
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (int v = 0; v < n; ++v) {
                        sum += y(i, u, j, v);
                    }
                    vio_theta(i, u, j) = x(i, u) - sum;
                }
            }
        }
        
        // Violation of symmetry constraints: y[j,v,i,u] - y[i,u,j,v]
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        vio_lambda(i, u, j, v) = y(j, v, i, u) - y(i, u, j, v);
                    }
                }
            }
        }
        
        return 0;
    }
    
    // Simple heuristic: extract assignment from x variables
    virtual int heuristics(const VOL_problem& /*p*/, const VOL_dvector& psol,
                          double& heur_val) override {
        return 0;  // Found feasible solution
    }
};

/**
 * Main solver function
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <instance.dat> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --time <seconds>      Time limit (default: 3600)" << std::endl;
        std::cout << "  --threads <n>         Number of threads (default: 8)" << std::endl;
        std::cout << "  --log                 Enable detailed logging" << std::endl;
        std::cout << "  --output <file>       Output solution file" << std::endl;
        std::cout << "  --save-dual <file>    Save dual vector and primal solution" << std::endl;
        std::cout << "  --load-dual <file>    Load initial dual vector" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    std::string instance_path = argv[1];
    int time_limit = 3600;
    int num_threads = 8;
    bool verbose = false;
    std::string output_path = "";
    std::string save_dual_path = "";
    std::string load_dual_path = "";
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--time" && i + 1 < argc) {
            time_limit = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--log") {
            verbose = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--save-dual" && i + 1 < argc) {
            save_dual_path = argv[++i];
        } else if (arg == "--load-dual" && i + 1 < argc) {
            load_dual_path = argv[++i];
        }
    }
    
    // Set number of OpenMP threads
    omp_set_num_threads(num_threads);
    
    std::cout << "==================================================" << std::endl;
    std::cout << "RTL1 Volume Algorithm Solver for QAP" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Instance: " << instance_path << std::endl;
    std::cout << "Time limit: " << time_limit << " seconds" << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Read QAP instance
    auto qap_data = Problem::fromQAPLIB(instance_path);
    
    int n = qap_data.n;
    std::cout << "Problem size: n = " << n << std::endl;
    
    // Setup Volume problem
    VOL_problem vol_problem;
    
    // Set problem dimensions
    // Primal variables: x[i,u] (n*n) + y[i,u,j,v] (n^4)
    vol_problem.psize = n * n + n * n * n * n;
    
    // Dual variables: mu[u] (n) + lambda[i,u,j,v] (n^4) + theta[i,u,j] (n^3)
    vol_problem.dsize = n + n * n * n * n + n * n * n;
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;
    
    // Load dual vector if specified
    if (!load_dual_path.empty()) {
        std::ifstream dual_file(load_dual_path, std::ios::binary);
        if (dual_file.is_open()) {
            std::cout << "Loading dual vector from: " << load_dual_path << std::endl;
            for (int i = 0; i < vol_problem.dsize; ++i) {
                dual_file.read(reinterpret_cast<char*>(&vol_problem.dsol[i]), sizeof(double));
            }
            dual_file.close();
            std::cout << "Dual vector loaded successfully." << std::endl;
        } else {
            std::cerr << "Warning: Could not open dual file: " << load_dual_path << std::endl;
            std::cerr << "Starting with zero dual vector." << std::endl;
        }
    }
    
    // Set Volume algorithm parameters (from qap.par defaults)
    vol_problem.parm.lambdainit = 0.1;
    vol_problem.parm.alphainit = 0.1;
    vol_problem.parm.alphamin = 0.001;
    vol_problem.parm.alphafactor = 0.66;
    vol_problem.parm.alphaint = 50;
    
    vol_problem.parm.maxsgriters = 100000000;  // Approximate iterations from time
    vol_problem.parm.primal_abs_precision = 0.001;
    vol_problem.parm.gap_abs_precision = 0.0;
    vol_problem.parm.gap_rel_precision = 0.001;
    vol_problem.parm.granularity = 0.0;
    
    vol_problem.parm.ascent_first_check = 500;
    vol_problem.parm.ascent_check_invl = 500;
    vol_problem.parm.minimum_rel_ascent = 0.0001;
    
    vol_problem.parm.greentestinvl = 1;
    vol_problem.parm.yellowtestinvl = 4;
    vol_problem.parm.redtestinvl = 20;
    
    // Printing control
    vol_problem.parm.printflag = verbose ? 3 : 1;  // 1=iteration info, 3=add lambda info
    vol_problem.parm.printinvl = 100;
    vol_problem.parm.heurinvl = 100;
    
    // Create hooks
    RTL1VolumeHooks hooks(qap_data);
    
    // Determine if we should use loaded dual (warm-start)
    bool use_dual_warmstart = !load_dual_path.empty();
    
    // Solve
    std::cout << "\nStarting Volume algorithm..." << std::endl;
    if (use_dual_warmstart) {
        std::cout << "Using warm-start from loaded dual vector." << std::endl;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int retval = vol_problem.solve(hooks, use_dual_warmstart);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = duration.count() / 1000.0;
    
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Volume Algorithm Complete" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Status: " << (retval == 0 ? "Success" : "Error") << std::endl;
    std::cout << "Lower bound: " << vol_problem.value << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_seconds << " seconds" << std::endl;
    
    // Save dual vector and primal solution if requested
    if (!save_dual_path.empty()) {
        std::cout << "\nSaving dual vector and primal solution to: " << save_dual_path << std::endl;
        
        // Save dual vector
        std::ofstream dual_file(save_dual_path, std::ios::binary);
        if (dual_file.is_open()) {
            for (int i = 0; i < vol_problem.dsize; ++i) {
                dual_file.write(reinterpret_cast<const char*>(&vol_problem.dsol[i]), sizeof(double));
            }
            dual_file.close();
            std::cout << "Dual vector saved (" << vol_problem.dsize << " values)." << std::endl;
        } else {
            std::cerr << "Error: Could not save dual vector to: " << save_dual_path << std::endl;
        }
        
        // Save primal solution
        std::string primal_path = save_dual_path + ".primal";
        std::ofstream primal_file(primal_path, std::ios::binary);
        if (primal_file.is_open()) {
            for (int i = 0; i < vol_problem.psize; ++i) {
                primal_file.write(reinterpret_cast<const char*>(&vol_problem.psol[i]), sizeof(double));
            }
            primal_file.close();
            std::cout << "Primal solution saved to: " << primal_path << " (" << vol_problem.psize << " values)." << std::endl;
        } else {
            std::cerr << "Error: Could not save primal solution to: " << primal_path << std::endl;
        }
    }
    
    // Extract and save best solution if requested
    if (!output_path.empty() && vol_problem.psol.size() > 0) {
        // Extract assignment from x variables
        std::vector<int> assignment(n);
        for (int i = 0; i < n; ++i) {
            int best_u = 0;
            double best_val = vol_problem.psol[i * n];
            for (int u = 1; u < n; ++u) {
                if (vol_problem.psol[i * n + u] > best_val) {
                    best_val = vol_problem.psol[i * n + u];
                    best_u = u;
                }
            }
            assignment[i] = best_u;
        }
        
        // Evaluate objective
        double obj = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                obj += qap_data.D[i][j] * qap_data.F[assignment[i]][assignment[j]];
            }
        }
        
        std::cout << "Primal objective: " << obj << std::endl;

    }
    
    // Always report a summary of primal violations (assignment, link, symmetry)
    if (vol_problem.psol.size() > 0) {
        auto vsummary = compute_primal_violation_summary(vol_problem.psol, n);
        std::cout << "\nPrimal violation summary (n=" << vsummary.n << ")" << std::endl;
        std::cout << "  Max abs: " << std::setprecision(6) << vsummary.max_abs
                  << ", Avg abs: " << vsummary.avg_abs << std::endl;
        std::cout << "  Assignment  -> max: " << vsummary.assignment_max
                  << ", avg: " << vsummary.assignment_avg << std::endl;
        std::cout << "  Linking     -> max: " << vsummary.link_max
                  << ", avg: " << vsummary.link_avg << std::endl;
        std::cout << "  Symmetry    -> included in Max/Avg above" << std::endl;
    }
    
    std::cout << "==================================================" << std::endl;
    
    return retval;
}
