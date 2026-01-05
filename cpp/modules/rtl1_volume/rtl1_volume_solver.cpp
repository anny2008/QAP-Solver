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
#include "qap_solution_io.hpp"

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
 * QAP Problem Data Structure
 */
struct QAPData {
    int n;                        // Problem size
    std::vector<double> flows;    // Flow matrix (linearized, n*n)
    std::vector<double> distances;// Distance matrix (linearized, n*n)
    
    double flow(int i, int j) const { return flows[i * n + j]; }
    double dist(int u, int v) const { return distances[u * n + v]; }
};

/**
 * Read QAP instance from QAPLIB format
 */
bool read_qaplib(const std::string& filename, QAPData& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    file >> data.n;
    int n = data.n;
    
    data.distances.resize(n * n);
    data.flows.resize(n * n);
    
    // Read distance matrix (first matrix in QAPLIB format)
    for (int i = 0; i < n * n; ++i) {
        file >> data.distances[i];
    }
    
    // Read flow matrix (second matrix in QAPLIB format)
    for (int i = 0; i < n * n; ++i) {
        file >> data.flows[i];
    }
    
    file.close();
    return true;
}

/**
 * RTL1 Volume Hooks Implementation
 */
class RTL1VolumeHooks : public VOL_user_hooks {
private:
    const QAPData& qap_data;
    int n;
    
    // Working arrays for subproblem
    std::vector<double> beta;       // beta[i,u,v] = min_j cost of y[i,u,j,v]=1
    std::vector<int> beta_j_ind;    // j that achieves beta[i,u,v]
    std::vector<double> alpha;      // alpha[i] = min_u cost of x[i,u]=1
    std::vector<int> alpha_u_ind;   // u that achieves alpha[i]
    
public:
    RTL1VolumeHooks(const QAPData& data) 
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
     *       - sum_u mu[u] * (sum_i x[i,u] - 1)
     *       - sum_{i,u,j,v} lambda[i,u,j,v] * (y[i,u,j,v] - y[j,v,i,u])
     *       - sum_{i,u,j} theta[i,u,j] * (sum_v y[i,u,j,v] - x[i,u])
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
                        double cost = qap_data.dist(i, j) * qap_data.flow(u, v);

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
                        pcost += qap_data.dist(i, j) * qap_data.flow(u, v) * y(i, u, j, v);
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
        std::vector<int> assignment(n, -1);
        std::vector<bool> used(n, false);
        
        // Greedy assignment based on x values
        for (int i = 0; i < n; ++i) {
            int best_u = -1;
            double best_val = -1.0;
            
            for (int u = 0; u < n; ++u) {
                if (!used[u] && x(i, u) > best_val) {
                    best_val = x(i, u);
                    best_u = u;
                }
            }
            
            if (best_u >= 0) {
                assignment[i] = best_u;
                used[best_u] = true;
            }
        }
        
        // Check if we have a complete assignment
        bool is_complete = true;
        for (int i = 0; i < n; ++i) {
            if (assignment[i] < 0) {
                is_complete = false;
                break;
            }
        }
        
        if (!is_complete) {
            heur_val = DBL_MAX;
            return 0;
        }
        
        // Evaluate QAP objective
        heur_val = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int u = assignment[i];
                int v = assignment[j];
                heur_val += qap_data.dist(i, j) * qap_data.flow(u, v);
            }
        }
        
        return 1;  // Found feasible solution
    }
};

/**
 * Main solver function
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <instance.dat> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --time <seconds>    Time limit (default: 3600)" << std::endl;
        std::cout << "  --threads <n>       Number of threads (default: 8)" << std::endl;
        std::cout << "  --log               Enable detailed logging" << std::endl;
        std::cout << "  --output <file>     Output solution file" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    std::string instance_path = argv[1];
    int time_limit = 3600;
    int num_threads = 8;
    bool verbose = false;
    std::string output_path = "";
    
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
    QAPData qap_data;
    if (!read_qaplib(instance_path, qap_data)) {
        return 1;
    }
    
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
    
    // Set Volume algorithm parameters (from qap.par defaults)
    vol_problem.parm.lambdainit = 0.1;
    vol_problem.parm.alphainit = 1.0;
    vol_problem.parm.alphamin = 0.001;
    vol_problem.parm.alphafactor = 0.66;
    vol_problem.parm.alphaint = 50;
    
    vol_problem.parm.maxsgriters = time_limit * 10;  // Approximate iterations from time
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
    
    // Solve
    std::cout << "\nStarting Volume algorithm..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int retval = vol_problem.solve(hooks, false);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = duration.count() / 1000.0;
    
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Volume Algorithm Complete" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Status: " << (retval == 0 ? "Success" : "Error") << std::endl;
    std::cout << "Lower bound: " << vol_problem.value << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_seconds << " seconds" << std::endl;
    
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
                obj += qap_data.dist(i, j) * qap_data.flow(assignment[i], assignment[j]);
            }
        }
        
        std::cout << "Primal objective: " << obj << std::endl;
        
        // Save solution
        try {
            qap::write_solution(output_path, n, assignment, obj);
            std::cout << "Solution saved to: " << output_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not save solution: " << e.what() << std::endl;
        }
    }
    
    std::cout << "==================================================" << std::endl;
    
    return retval;
}
