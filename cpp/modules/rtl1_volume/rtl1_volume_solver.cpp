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
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <omp.h>

#include "VolVolume.hpp"
#include "../../core/problem.h"
#include "../../include/qap_solution_io.hpp"
// Fixed-variable structures (x and y)
struct FixedVariables {
    std::unordered_map<int, int> x_fixed_1;                    // i -> u
    std::unordered_map<int, std::unordered_set<int>> x_fixed_0; // i -> {u}
    std::unordered_map<int, int> y_fixed_1;                    // key(i,u,v) -> j
    std::unordered_map<int, std::unordered_set<int>> y_fixed_0; // key(i,u,v) -> {j}
};

static int key_y(int n, int i, int u, int v) { return i * n * n + u * n + v; }

// Parse fixed variables from a text file with lines:
//   x i u value   (value in {0,1})
//   y i u j v value (value in {0,1})
// Indices are 0-based; '#' starts a comment line.
static bool parse_fixed_file(const std::string &path, int n, FixedVariables &out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open fixed-variable file: " << path << std::endl;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        std::string trimmed = line;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](int ch) { return !std::isspace(ch); }));
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::istringstream iss(trimmed);
        char type;
        iss >> type;
        if (type == 'x') {
            int i, u, val;
            if (!(iss >> i >> u >> val)) {
                std::cerr << "Warning: malformed x-line at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || (val != 0 && val != 1)) {
                std::cerr << "Warning: invalid indices/value at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (val == 1) {
                out.x_fixed_1[i] = u;
            } else {
                out.x_fixed_0[i].insert(u);
            }
        } else if (type == 'y') {
            int i, u, j, v, val;
            if (!(iss >> i >> u >> j >> v >> val)) {
                std::cerr << "Warning: malformed y-line at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || j < 0 || j >= n || v < 0 || v >= n || (val != 0 && val != 1)) {
                std::cerr << "Warning: invalid indices/value at " << path << ":" << lineno << std::endl;
                continue;
            }
            int key = key_y(n, i, u, v);
            if (val == 1) {
                out.y_fixed_1[key] = j;
            } else {
                out.y_fixed_0[key].insert(j);
            }
        } else {
            std::cerr << "Warning: unknown line type at " << path << ":" << lineno << std::endl;
        }
    }

    std::cout << "Loaded fixed variables from " << path
              << " | x1=" << out.x_fixed_1.size()
              << " x0-rows=" << out.x_fixed_0.size()
              << " y1=" << out.y_fixed_1.size()
              << " y0-rows=" << out.y_fixed_0.size() << std::endl;
    return true;
}


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

struct FixedViolationReport {
    int x1_violations = 0;
    int x0_violations = 0;
    int y1_violations = 0;
    int y0_violations = 0;
    double x1_max_dev = 0.0;
    double x0_max_dev = 0.0;
    double y1_max_dev = 0.0;
    double y0_max_dev = 0.0;
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

static FixedViolationReport check_fixed_violations(const FixedVariables &fv, const VOL_dvector &psol, int n, double tol = 1e-6) {
    FixedViolationReport rep;

    // x fixed to 1
    for (const auto &kv : fv.x_fixed_1) {
        int i = kv.first;
        int u = kv.second;
        double val = psol[i * n + u];
        double dev = std::abs(val - 1.0);
        if (dev > tol) {
            ++rep.x1_violations;
            rep.x1_max_dev = std::max(rep.x1_max_dev, dev);
        }
    }

    // x fixed to 0
    for (const auto &kv : fv.x_fixed_0) {
        int i = kv.first;
        for (int u : kv.second) {
            double val = psol[i * n + u];
            double dev = std::abs(val);
            if (dev > tol) {
                ++rep.x0_violations;
                rep.x0_max_dev = std::max(rep.x0_max_dev, dev);
            }
        }
    }

    // y fixed to 1
    for (const auto &kv : fv.y_fixed_1) {
        int key = kv.first;
        int j = kv.second;
        int i = key / (n * n);
        int rem = key % (n * n);
        int u = rem / n;
        int v = rem % n;
        int idx = n * n + i * n * n * n + u * n * n + j * n + v;
        double val = psol[idx];
        double dev = std::abs(val - 1.0);
        if (dev > tol) {
            ++rep.y1_violations;
            rep.y1_max_dev = std::max(rep.y1_max_dev, dev);
        }
    }

    // y fixed to 0
    for (const auto &kv : fv.y_fixed_0) {
        int key = kv.first;
        int i = key / (n * n);
        int rem = key % (n * n);
        int u = rem / n;
        int v = rem % n;
        for (int j : kv.second) {
            int idx = n * n + i * n * n * n + u * n * n + j * n + v;
            double val = psol[idx];
            double dev = std::abs(val);
            if (dev > tol) {
                ++rep.y0_violations;
                rep.y0_max_dev = std::max(rep.y0_max_dev, dev);
            }
        }
    }

    return rep;
}

// Macro definitions for cleaner indexing
#define mu(u) (pi[u])
#define lambda(i, u, j, v) (pi[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define theta(i, u, j) (pi[n + n * n * n * n + (i) * n * n + (u) * n + (j)])
#define x(i, u) (psol[(i) * n + (u)])
#define y(i, u, j, v) (psol[n * n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define z(i, u, j, v) (zsol[(i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_mu(u) (vio[u])
#define vio_lambda(i, u, j, v) (vio[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_theta(i, u, j) (vio[n + n * n * n * n + (i) * n * n + (u) * n + (j)])
#define mu1(u) (pi[u])
#define mu2(i) (pi[n+i])
#define theta1(i, u, v) (pi[2*n + (i) * n * n + (u) * n + (v)])
#define theta2(i, u, j) (pi[2*n + n*n*n + (i) * n * n + (u) * n + (j)])
#define vio_mu1(u) (vio[u])
#define vio_mu2(i) (vio[n+i])
#define vio_theta1(i, u, v) (vio[2*n + (i) * n * n + (u) * n + (v)])
#define vio_theta2(i, u, j) (vio[2*n + n*n*n + (i) * n * n + (u) * n + (j)])

/**
 * RTL1 Volume Hooks Implementation
 */
class RTL1VolumeHooks1 : public VOL_user_hooks {
private:
    const Problem& qap_data;
    int n;
    
    // Working arrays for subproblem
    std::vector<double> beta;       // beta[i,u,v] = min_j cost of y[i,u,j,v]=1
    std::vector<int> beta_j_ind;    // j that achieves beta[i,u,v]
    std::vector<double> alpha;      // alpha[i] = min_u cost of x[i,u]=1
    std::vector<int> alpha_u_ind;   // u that achieves alpha[i]

    // Fixed variables
    FixedVariables fixed;
    std::unordered_map<int, int> x_fixed_1;                    // i -> u
    std::unordered_map<int, std::unordered_set<int>> x_fixed_0; // i -> {u}
    std::unordered_map<int, int> y_fixed_1;                    // key(i,u,v) -> j
    std::unordered_map<int, std::unordered_set<int>> y_fixed_0; // key(i,u,v) -> {j}
    
public:
    RTL1VolumeHooks1(const Problem& data) 
        : qap_data(data), n(data.n) {
        beta.resize(n * n * n);
        beta_j_ind.resize(n * n * n);
        alpha.resize(n);
        alpha_u_ind.resize(n);
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
    int solve_subproblem_1(const VOL_dvector& pi,
                                double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                                double& pcost) {
        
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

                    int ykey = key_y(n, i, u, v);
                    auto it_y1 = y_fixed_1.find(ykey);
                    if (it_y1 != y_fixed_1.end()) {
                        int j = it_y1->second;
                        double cost = qap_data.D[i][j] * qap_data.F[u][v];
                        if (i <= j) cost -= lambda(i, u, j, v);
                        if (i >= j) cost += lambda(j, v, i, u);
                        cost -= theta(i, u, j);
                        min_cost = cost;
                        best_j = j;
                    } else {
                        const auto it_y0 = y_fixed_0.find(ykey);
                        for (int j = 0; j < n; ++j) {
                            if (it_y0 != y_fixed_0.end() && it_y0->second.count(j)) {
                                continue; // skip fixed-to-zero y
                            }
                            double cost = qap_data.D[i][j] * qap_data.F[u][v];
                            if (i <= j) cost -= lambda(i, u, j, v);
                            if (i >= j) cost += lambda(j, v, i, u);
                            cost -= theta(i, u, j);
                            if (cost < min_cost) {
                                min_cost = cost;
                                best_j = j;
                            }
                        }
                    }

                    int idx = i * n * n + u * n + v;
                    if (min_cost == DBL_MAX) {
                        std::cerr << "Warning: no feasible j for y[" << i << "," << u << ",*," << v << "] under fixed variables; choosing j=0 with large cost." << std::endl;
                        beta[idx] = 1e30;
                        beta_j_ind[idx] = 0;
                    } else {
                        beta[idx] = min_cost;
                        beta_j_ind[idx] = best_j;
                    }
                }
            }
        }
        
        // STEP 2: Compute alpha[i] = min_u {cost of x[i,u]=1}
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            double min_cost = DBL_MAX;
            int best_u = 0;

            // If x[i,*] has a fixed 1, only evaluate that u
            auto it_x1 = x_fixed_1.find(i);
            if (it_x1 != x_fixed_1.end()) {
                int u = it_x1->second;
                double cost = -mu(u);
                for (int j = 0; j < n; ++j) cost += theta(i, u, j);
                for (int v = 0; v < n; ++v) {
                    int idx = i * n * n + u * n + v;
                    cost += beta[idx];
                }
                min_cost = cost;
                best_u = u;
            } else {
                const auto it_x0 = x_fixed_0.find(i);
                for (int u = 0; u < n; ++u) {
                    if (it_x0 != x_fixed_0.end() && it_x0->second.count(u)) {
                        continue; // fixed to zero
                    }
                    double cost = -mu(u);
                    for (int j = 0; j < n; ++j) cost += theta(i, u, j);
                    for (int v = 0; v < n; ++v) {
                        int idx = i * n * n + u * n + v;
                        cost += beta[idx];
                    }
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_u = u;
                    }
                }
            }

            if (min_cost == DBL_MAX) {
                std::cerr << "Warning: no feasible u for fixed variables at facility i=" << i << "; choosing u=0." << std::endl;
                alpha[i] = 1e30;
                alpha_u_ind[i] = 0;
            } else {
                alpha[i] = min_cost;
                alpha_u_ind[i] = best_u;
            }
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
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                for (int j = i; j < n; ++j) {
                        vio_lambda(i, u, j, v) = y(j, v, i, u) - y(i, u, j, v);
                    }
                }
            }
        }
        
        return 0;
    }

    void set_fixed_variables(const FixedVariables &fv) {
        fixed = fv;
        x_fixed_1 = fv.x_fixed_1;
        x_fixed_0 = fv.x_fixed_0;
        y_fixed_1 = fv.y_fixed_1;
        y_fixed_0 = fv.y_fixed_0;
    }
    
    // Compute reduced costs (not used in RTL1 subproblem)
    virtual int compute_rc(const VOL_dvector& /*pi*/, VOL_dvector& rc) override {
        rc = 0;
        return 0;
    }
    
    virtual int solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                                double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                                double& pcost) override {
                                    return solve_subproblem_1(pi, lcost, psol, vio, pcost);
                                }
    
    // Simple heuristic: extract assignment from x variables
    virtual int heuristics(const VOL_problem& /*p*/, const VOL_dvector& psol,
                          double& heur_val) override {
        return 0;  // Found feasible solution
    }
};
    
class RTL1VolumeHooks2 : public VOL_user_hooks {
private:
    const Problem& qap_data;
    int n;
    

    // Fixed variables
    FixedVariables fixed;
    std::unordered_map<int, int> x_fixed_1;                    // i -> u
    std::unordered_map<int, std::unordered_set<int>> x_fixed_0; // i -> {u}
    std::unordered_map<int, int> y_fixed_1;                    // key(i,u,v) -> j
    std::unordered_map<int, std::unordered_set<int>> y_fixed_0; // key(i,u,v) -> {j}
public:
    RTL1VolumeHooks2(const Problem& data): qap_data(data), n(data.n) {}

    void set_fixed_variables(const FixedVariables &fv) {
        fixed = fv;
        x_fixed_1 = fv.x_fixed_1;
        x_fixed_0 = fv.x_fixed_0;
        y_fixed_1 = fv.y_fixed_1;
        y_fixed_0 = fv.y_fixed_0;
    }
    
    virtual int solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                                double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                                double& pcost) override {
                                    return solve_subproblem_2(pi, lcost, psol, vio, pcost);
                                }
    
    
    int solve_subproblem_2(const VOL_dvector& pi,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
        // Alternative subproblem solver for RTL1 (Volume algorithm)
        // Step 1: Initialize costs and solution vectors
        lcost = 0.0;
        pcost = 0.0;
        psol = 0.0; // Set all primal variables to zero

        // Step 2: Set y variables by checking reduced cost for each (i,u,j,v) and its symmetric (j,v,i,u)
        // If the reduced cost (coeff_y) is negative, set both y(i,u,j,v) and y(j,v,i,u) to 1 (symmetry)
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    for (int j = i; j < n; ++j) {
                        // Compute reduced cost for y(i,u,j,v)
                        double coeff_y = qap_data.D[i][j] * qap_data.F[u][v]
                                     + qap_data.D[j][i] * qap_data.F[v][u]
                                     - theta1(i, u, v) - theta1(j, v, u)
                                     - theta2(i, u, j) - theta2(j, v, i);
                        // If negative, set y(i,u,j,v) and y(j,v,i,u) to 1 (enforce symmetry)
                        if (coeff_y < 0) {
                            y(i, u, j, v) = 1.0;
                            y(j, v, i, u) = 1.0;
                        }
                    }
                }
            }
        }

        // Step 3: Set x variables by checking reduced cost for each (i,u)
        // If the reduced cost (coeff_x) is negative, set x(i,u) = 1
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                // Compute reduced cost for x(i,u)
                double coeff_x = -mu1(u) - mu2(i);
                for (int v = 0; v < n; ++v) {
                    auto j = v;
                    coeff_x += theta1(i, u, v) + theta2(i, u, j);
                }
                // If negative, set x(i,u) = 1
                if (coeff_x < 0) {
                    x(i, u) = 1.0;
                }
            }
        }
        // Step 4: Compute constraint violations for the current solution
        vio = 0.0;

        // Assignment constraint 1: For each u, sum_i x[i,u] should be 1
        // Store violation in vio_mu1(u) = 1 - sum_i x[i,u]
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < n; ++u) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += x(i, u);
            }
            vio_mu1(u) = 1.0 - sum;
        }

        // Assignment constraint 2: For each i, sum_u x[i,u] should be 1
        // Store violation in vio_mu2(i) = 1 - sum_u x[i,u]
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int u = 0; u < n; ++u) {
                sum += x(i, u);
            }
            vio_mu2(i) = 1.0 - sum;
        }

        // Linking constraint 2: For each (i,u,v), x[i,u] = sum_j y[i,u,j,v]
        // Store violation in vio_theta1(i,u,v) = x[i,u] - sum_j y[i,u,j,v]
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    double sum = 0.0;
                    for (int j = 0; j < n; ++j) {
                        sum += y(i, u, j, v);
                    }
                    vio_theta1(i, u, v) = x(i, u) - sum;
                }
            }
        }

        // Linking constraint 1: For each (i,u,j), x[i,u] = sum_v y[i,u,j,v]
        // Store violation in vio_theta2(i,u,j) = x[i,u] - sum_v y[i,u,j,v]
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (int v = 0; v < n; ++v) {
                        sum += y(i, u, j, v);
                    }
                    vio_theta2(i, u, j) = x(i, u) - sum;
                }
            }
        }

        // Step 5: Compute primal cost (QAP objective value for current y)
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

        // Step 6: Compute Lagrangian cost (objective + penalty for constraint violations)
        lcost = pcost;
        // Add penalties for assignment constraint violations
        #pragma omp parallel for reduction(+:lcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            lcost += mu2(i)*vio_mu2(i) + mu1(i)*vio_mu1(i);
        }

        // Add penalties for linking constraint violations
        #pragma omp parallel for collapse(3) reduction(+:lcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    auto v = j;
                    lcost += theta2(i,u,j)*vio_theta2(i,u,j) + theta1(i,u,v)*vio_theta1(i,u,v);
                }
            }
        }

        // for (auto u= 0; u < n; ++u) {
        //     std::cout << "Vio u[" << u << "] = " << vio_mu1(u) << std::endl;
        // }
        // for (auto i= 0; i < n; ++i) {
        //     std::cout << "Vio i[" << i << "] = " << vio_mu2(i) << std::endl;
        // }

        auto total_vio = 0.0;
        #pragma omp parallel for reduction(+:total_vio) schedule(static)
        for (int k = 0; k < vio.size(); ++k) {
            total_vio += vio[k];
        }
        // std::cout << "Total violation: " << total_vio << std::endl;

        // Print costs for debugging
        // std::cout << "Primal cost: " << pcost << ", Lagrangian cost: " << lcost << std::endl;

        return 0;
    }

    // Compute reduced costs (not used in RTL1 subproblem)
    virtual int compute_rc(const VOL_dvector& /*pi*/, VOL_dvector& rc) override {
        rc = 0;
        return 0;
    }
    
    // Simple heuristic: extract assignment from x variables
    virtual int heuristics(const VOL_problem& /*p*/, const VOL_dvector& psol,
                          double& heur_val) override {
        return 0;  // Found feasible solution
    }
};

class RTL1VolumeHooks3 : public VOL_user_hooks {
private:
    const Problem& qap_data;
    int n;


    std::vector<double> zsol; // working array for z variables 
    // Fixed variables
    FixedVariables fixed;
    std::unordered_map<int, int> x_fixed_1;                    // i -> u
    std::unordered_map<int, std::unordered_set<int>> x_fixed_0; // i -> {u}
    std::unordered_map<int, int> y_fixed_1;                    // key(i,u,v) -> j
    std::unordered_map<int, std::unordered_set<int>> y_fixed_0; // key(i,u,v) -> {j}
public:
    RTL1VolumeHooks3(const Problem& data): qap_data(data), n(data.n) {
        zsol.resize(n * n * n * n);
    }

    void set_fixed_variables(const FixedVariables &fv) {
        fixed = fv;
        x_fixed_1 = fv.x_fixed_1;
        x_fixed_0 = fv.x_fixed_0;
        y_fixed_1 = fv.y_fixed_1;
        y_fixed_0 = fv.y_fixed_0;
    }
    
    virtual int solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                                double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                                double& pcost) override {
                                    return solve_subproblem_3(pi, lcost, psol, vio, pcost);
                                }
    int solve_subproblem_3(const VOL_dvector& pi,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
        // Alternative subproblem solver for RTL1 (Volume algorithm)
        // Step 1: Initialize costs and solution vectors
        lcost = 0.0;
        pcost = 0.0;
        psol = 0.0; // Set all primal variables to zero
        std::fill(zsol.begin(), zsol.end(), 0.0);

        // Step 2: Set y variables by checking reduced cost for each (i,u,j,v) and its symmetric (j,v,i,u)
        // If the reduced cost (coeff_y) is negative, set both y(i,u,j,v) and y(j,v,i,u) to 1 (symmetry)
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    for (int j = i; j < n; ++j) {
                        // Compute reduced cost for y(i,u,j,v)
                        double coeff_y = qap_data.D[i][j] * qap_data.F[u][v]
                                     + qap_data.D[j][i] * qap_data.F[v][u]
                                     - theta1(i, u, v) - theta1(j, v, u)
                                     - theta2(i, u, j) - theta2(j, v, i);
                        // If negative, set y(i,u,j,v) and y(j,v,i,u) to 1 (enforce symmetry)
                        if (coeff_y < 0) {
                            z(i, u, j, v) = 1.0;
                            z(j, v, i, u) = 1.0;
                        }
                    }
                }
            }
        }

        // Step 3: Set x variables by checking reduced cost for each (i,u)
        // If the reduced cost (coeff_x) is negative, set x(i,u) = 1
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                // Compute reduced cost for x(i,u)
                double coeff_x = -mu1(u) - mu2(i);
                for (int v = 0; v < n; ++v) {
                    auto j = v;
                    coeff_x += theta1(i, u, v) + theta2(i, u, j);
                }
                // If negative, set x(i,u) = 1
                if (coeff_x < 0) {
                    x(i, u) = 1.0;
                }
            }
        }
        // Step 4: Compute constraint violations for the current solution
        vio = 0.0;

        // Assignment constraint 1: For each u, sum_i x[i,u] should be 1
        // Store violation in vio_mu1(u) = 1 - sum_i x[i,u]
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < n; ++u) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += x(i, u);
            }
            vio_mu1(u) = 1.0 - sum;
        }

        // Assignment constraint 2: For each i, sum_u x[i,u] should be 1
        // Store violation in vio_mu2(i) = 1 - sum_u x[i,u]
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int u = 0; u < n; ++u) {
                sum += x(i, u);
            }
            vio_mu2(i) = 1.0 - sum;
        }

        // Linking constraint 2: For each (i,u,v), x[i,u] = sum_j y[i,u,j,v]
        // Store violation in vio_theta1(i,u,v) = x[i,u] - sum_j y[i,u,j,v]
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    double sum = 0.0;
                    for (int j = 0; j < n; ++j) {
                        sum += z(i, u, j, v);
                    }
                    vio_theta1(i, u, v) = x(i, u) - sum;
                }
            }
        }

        // Linking constraint 1: For each (i,u,j), x[i,u] = sum_v y[i,u,j,v]
        // Store violation in vio_theta2(i,u,j) = x[i,u] - sum_v y[i,u,j,v]
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (int v = 0; v < n; ++v) {
                        sum += z(i, u, j, v);
                    }
                    vio_theta2(i, u, j) = x(i, u) - sum;
                }
            }
        }

        // Step 5: Compute primal cost (QAP objective value for current y)
        pcost = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:pcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        pcost += qap_data.D[i][j] * qap_data.F[u][v] * z(i, u, j, v);
                    }
                }
            }
        }

        // Step 6: Compute Lagrangian cost (objective + penalty for constraint violations)
        lcost = pcost;
        // Add penalties for assignment constraint violations
        #pragma omp parallel for reduction(+:lcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            lcost += mu2(i)*vio_mu2(i) + mu1(i)*vio_mu1(i);
        }

        // Add penalties for linking constraint violations
        #pragma omp parallel for collapse(3) reduction(+:lcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    auto v = j;
                    lcost += theta2(i,u,j)*vio_theta2(i,u,j) + theta1(i,u,v)*vio_theta1(i,u,v);
                }
            }
        }

        // for (auto u= 0; u < n; ++u) {
        //     std::cout << "Vio u[" << u << "] = " << vio_mu1(u) << std::endl;
        // }
        // for (auto i= 0; i < n; ++i) {
        //     std::cout << "Vio i[" << i << "] = " << vio_mu2(i) << std::endl;
        // }

        // auto total_vio = 0.0;
        // #pragma omp parallel for reduction(+:total_vio) schedule(static)
        // for (int k = 0; k < vio.size(); ++k) {
        //     total_vio += vio[k];
        // }
        // std::cout << "Total violation: " << total_vio << std::endl;

        // Print costs for debugging
        // std::cout << "Primal cost: " << pcost << ", Lagrangian cost: " << lcost << std::endl;

        return 0;
    }

    // Compute reduced costs (not used in RTL1 subproblem)
    virtual int compute_rc(const VOL_dvector& /*pi*/, VOL_dvector& rc) override {
        rc = 0;
        return 0;
    }
    // Simple heuristic: extract assignment from x variables
    virtual int heuristics(const VOL_problem& /*p*/, const VOL_dvector& psol,
                          double& heur_val) override {
        return 0;  // Found feasible solution
    }
};


/**
 * RTL1 Volume Hooks Implementation
 */
class RTL1VolumeHooks4 : public VOL_user_hooks {
private:
    const Problem& qap_data;
    int n;
    
    // Working arrays for subproblem
    std::vector<double> beta;       // beta[i,u,v] = min_j cost of y[i,u,j,v]=1
    std::vector<int> beta_j_ind;    // j that achieves beta[i,u,v]
    std::vector<double> alpha;      // alpha[i] = min_u cost of x[i,u]=1
    std::vector<int> alpha_u_ind;   // u that achieves alpha[i]

    std::vector<double> zsol; // working array for z variables 
    // Fixed variables
    FixedVariables fixed;
    std::unordered_map<int, int> x_fixed_1;                    // i -> u
    std::unordered_map<int, std::unordered_set<int>> x_fixed_0; // i -> {u}
    std::unordered_map<int, int> y_fixed_1;                    // key(i,u,v) -> j
    std::unordered_map<int, std::unordered_set<int>> y_fixed_0; // key(i,u,v) -> {j}
    
public:
    RTL1VolumeHooks4(const Problem& data) 
        : qap_data(data), n(data.n) {
        beta.resize(n * n * n);
        beta_j_ind.resize(n * n * n);
        alpha.resize(n);
        alpha_u_ind.resize(n);
        zsol.resize(n * n * n * n);
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
    int solve_subproblem_4(const VOL_dvector& pi,
                                double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                                double& pcost) {
        
        // Initialize working arrays
        std::fill(beta.begin(), beta.end(), 0.0);
        std::fill(beta_j_ind.begin(), beta_j_ind.end(), 0);
        std::fill(alpha.begin(), alpha.end(), 0.0);
        std::fill(alpha_u_ind.begin(), alpha_u_ind.end(), 0);
        std::fill(zsol.begin(), zsol.end(), 0.0);
        
        // STEP 1: Compute beta[i,u,v] = min_j {Lagrangian cost of y[i,u,j,v]=1}
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    double min_cost = DBL_MAX;
                    int best_j = 0;

                    int ykey = key_y(n, i, u, v);
                    auto it_y1 = y_fixed_1.find(ykey);
                    if (it_y1 != y_fixed_1.end()) {
                        int j = it_y1->second;
                        double cost = qap_data.D[i][j] * qap_data.F[u][v];
                        if (i <= j) cost -= lambda(i, u, j, v);
                        if (i >= j) cost += lambda(j, v, i, u);
                        cost -= theta(i, u, j);
                        min_cost = cost;
                        best_j = j;
                    } else {
                        const auto it_y0 = y_fixed_0.find(ykey);
                        for (int j = 0; j < n; ++j) {
                            if (it_y0 != y_fixed_0.end() && it_y0->second.count(j)) {
                                continue; // skip fixed-to-zero y
                            }
                            double cost = qap_data.D[i][j] * qap_data.F[u][v];
                            if (i <= j) cost -= lambda(i, u, j, v);
                            if (i >= j) cost += lambda(j, v, i, u);
                            cost -= theta(i, u, j);
                            if (cost < min_cost) {
                                min_cost = cost;
                                best_j = j;
                            }
                        }
                    }

                    int idx = i * n * n + u * n + v;
                    if (min_cost == DBL_MAX) {
                        std::cerr << "Warning: no feasible j for y[" << i << "," << u << ",*," << v << "] under fixed variables; choosing j=0 with large cost." << std::endl;
                        beta[idx] = 1e30;
                        beta_j_ind[idx] = 0;
                    } else {
                        beta[idx] = min_cost;
                        beta_j_ind[idx] = best_j;
                    }
                }
            }
        }
        
        // STEP 2: Compute alpha[i] = min_u {cost of x[i,u]=1}
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            double min_cost = DBL_MAX;
            int best_u = 0;

            // If x[i,*] has a fixed 1, only evaluate that u
            auto it_x1 = x_fixed_1.find(i);
            if (it_x1 != x_fixed_1.end()) {
                int u = it_x1->second;
                double cost = -mu(u);
                for (int j = 0; j < n; ++j) cost += theta(i, u, j);
                for (int v = 0; v < n; ++v) {
                    int idx = i * n * n + u * n + v;
                    cost += beta[idx];
                }
                min_cost = cost;
                best_u = u;
            } else {
                const auto it_x0 = x_fixed_0.find(i);
                for (int u = 0; u < n; ++u) {
                    if (it_x0 != x_fixed_0.end() && it_x0->second.count(u)) {
                        continue; // fixed to zero
                    }
                    double cost = -mu(u);
                    for (int j = 0; j < n; ++j) cost += theta(i, u, j);
                    for (int v = 0; v < n; ++v) {
                        int idx = i * n * n + u * n + v;
                        cost += beta[idx];
                    }
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_u = u;
                    }
                }
            }

            if (min_cost == DBL_MAX) {
                std::cerr << "Warning: no feasible u for fixed variables at facility i=" << i << "; choosing u=0." << std::endl;
                alpha[i] = 1e30;
                alpha_u_ind[i] = 0;
            } else {
                alpha[i] = min_cost;
                alpha_u_ind[i] = best_u;
            }
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
                z(i, u, j, v) = 1.0;
            }
        }
        
        // Compute primal objective (original QAP objective on fractional solution)
        pcost = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:pcost) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        pcost += qap_data.D[i][j] * qap_data.F[u][v] * z(i, u, j, v);
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
                        sum += z(i, u, j, v);
                    }
                    vio_theta(i, u, j) = x(i, u) - sum;
                }
            }
        }
        
        // Violation of symmetry constraints: z[j,v,i,u] - z[i,u,j,v]
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                for (int j = i; j < n; ++j) {
                        vio_lambda(i, u, j, v) = z(j, v, i, u) - z(i, u, j, v);
                    }
                }
            }
        }
        
        return 0;
    }

    void set_fixed_variables(const FixedVariables &fv) {
        fixed = fv;
        x_fixed_1 = fv.x_fixed_1;
        x_fixed_0 = fv.x_fixed_0;
        y_fixed_1 = fv.y_fixed_1;
        y_fixed_0 = fv.y_fixed_0;
    }
    
    // Compute reduced costs (not used in RTL1 subproblem)
    virtual int compute_rc(const VOL_dvector& /*pi*/, VOL_dvector& rc) override {
        rc = 0;
        return 0;
    }
    
    virtual int solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                                double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                                double& pcost) override {
                                    return solve_subproblem_4(pi, lcost, psol, vio, pcost);
                                }
    
    // Simple heuristic: extract assignment from x variables
    virtual int heuristics(const VOL_problem& /*p*/, const VOL_dvector& psol,
                          double& heur_val) override {
        return 0;  // Found feasible solution
    }
};
    

void set_up_vol_problem_1(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

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
}

void set_up_vol_problem_2(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    // Primal variables: x[i,u] (n*n) + y[i,u,j,v] (n^4)
    vol_problem.psize = n * n + n * n * n * n;
    
    // Dual variables: mu1[u] (n) + mu2[i] (n) + theta1[i,u,v] (n^3) + theta2[i,u,j] (n^3)
    vol_problem.dsize = n*2 + n * n * n * 2;
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;
}

void set_up_vol_problem_3(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    // Placeholder for third formulation's dimensions
    vol_problem.psize = n*n;
    vol_problem.dsize = 2*n + 2*n*n*n;
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;
}

void set_up_vol_problem_4(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    // Placeholder for third formulation's dimensions
    vol_problem.psize = n*n;
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
}

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
        std::cout << "  --fixed <file>        Fixed variables file (x/y, 0-based indices)" << std::endl;
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
    std::string fixed_path = "";
    std::string formulation = "";
    
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
        } else if (arg == "--fixed" && i + 1 < argc) {
            fixed_path = argv[++i];
        } else if (arg == "--formulation" && i + 1 < argc) {
            formulation = argv[++i];
        }
    }
    std::cout << "Solving using formulation: " << formulation << std::endl;
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
    if (formulation == "formulation1") {
        set_up_vol_problem_1(vol_problem, qap_data, FixedVariables());
    } else if (formulation == "formulation2") {
        set_up_vol_problem_2(vol_problem, qap_data, FixedVariables());
    } else if (formulation == "formulation3") {
        set_up_vol_problem_3(vol_problem, qap_data, FixedVariables());
    } else if (formulation == "formulation4") {
        set_up_vol_problem_4(vol_problem, qap_data, FixedVariables());
    } else {
        std::cerr << "Error: Unknown formulation specified. Use --formulation <formulation1|formulation2|formulation3|formulation4>" << std::endl;
        return 1;
    }
    
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
    vol_problem.parm.alphainit = 0.01;
    vol_problem.parm.alphamin = 0.0001;
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
    RTL1VolumeHooks1 hooks1(qap_data);
    RTL1VolumeHooks2 hooks2(qap_data);
    RTL1VolumeHooks3 hooks3(qap_data);
    RTL1VolumeHooks4 hooks4(qap_data);
    // Load fixed variables if provided
    FixedVariables fv;
    bool has_fixed = false;
    if (!fixed_path.empty()) {
        if (parse_fixed_file(fixed_path, n, fv)) {
            if (formulation == "formulation1") {
                hooks1.set_fixed_variables(fv);
            } else if (formulation == "formulation2") {
                hooks2.set_fixed_variables(fv);
            } else if (formulation == "formulation3") {
                hooks3.set_fixed_variables(fv);
            } else if (formulation == "formulation4") {
                hooks4.set_fixed_variables(fv);
            }
            has_fixed = true;
        } else {
            std::cerr << "Error parsing fixed-variable file; continuing without fixed variables." << std::endl;
        }
    }
    
    // Determine if we should use loaded dual (warm-start)
    bool use_dual_warmstart = !load_dual_path.empty();
    
    // Solve
    std::cout << "\nStarting Volume algorithm..." << std::endl;
    if (use_dual_warmstart) {
        std::cout << "Using warm-start from loaded dual vector." << std::endl;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int retval;
    if (formulation == "formulation1") {
        retval = vol_problem.solve(hooks1, use_dual_warmstart);
    } else if (formulation == "formulation2") {
        retval = vol_problem.solve(hooks2, use_dual_warmstart);
    } else if (formulation == "formulation3") {
        retval = vol_problem.solve(hooks3, use_dual_warmstart);
    } else if (formulation == "formulation4") {
        retval = vol_problem.solve(hooks4, use_dual_warmstart);
    } else {
        std::cerr << "Error: Unknown formulation specified." << std::endl;
        return 1;
    }
    
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
    if (!output_path.empty() && vol_problem.psol.size() > 0 && formulation != "formulation4") {
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
    if (vol_problem.psol.size() > 0 and formulation != "formulation3" && formulation != "formulation4") {
        auto vsummary = compute_primal_violation_summary(vol_problem.psol, n);
        std::cout << "\nPrimal violation summary (n=" << vsummary.n << ")" << std::endl;
        std::cout << "  Max abs: " << std::setprecision(6) << vsummary.max_abs
                  << ", Avg abs: " << vsummary.avg_abs << std::endl;
        std::cout << "  Assignment  -> max: " << vsummary.assignment_max
                  << ", avg: " << vsummary.assignment_avg << std::endl;
        std::cout << "  Linking     -> max: " << vsummary.link_max
                  << ", avg: " << vsummary.link_avg << std::endl;
        std::cout << "  Symmetry    -> included in Max/Avg above" << std::endl;
        if (has_fixed) {
            auto frep = check_fixed_violations(fv, vol_problem.psol, n);
            std::cout << "  Fixed vars  -> x1: " << fv.x_fixed_1.size()
                      << " (viol " << frep.x1_violations << ", max dev " << frep.x1_max_dev << ")"
                      << ", x0-rows: " << fv.x_fixed_0.size()
                      << " (viol " << frep.x0_violations << ", max dev " << frep.x0_max_dev << ")"
                      << ", y1: " << fv.y_fixed_1.size()
                      << " (viol " << frep.y1_violations << ", max dev " << frep.y1_max_dev << ")"
                      << ", y0-rows: " << fv.y_fixed_0.size()
                      << " (viol " << frep.y0_violations << ", max dev " << frep.y0_max_dev << ")" << std::endl;
        }
    }
    
    std::cout << "==================================================" << std::endl;
    auto instance_name = instance_path.substr(instance_path.find_last_of("/\\") + 1);
    
    auto n_iterations = vol_problem.iter();
    std::cout << instance_name << " & " << formulation << " & " << vol_problem.value << " & " << n_iterations << " & " << std::fixed << std::setprecision(2) << elapsed_seconds << "s\\\\" << std::endl;
    
    return retval;
}
