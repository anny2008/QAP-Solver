/**
 * RLT1 Volume Algorithm Solver for QAP
 * 
 * Implements the Lagrangian relaxation-based Volume algorithm
 * for the RLT1 formulation of the Quadratic Assignment Problem.
 * 
 * Based on:
 * - Barahona & Anbil (1998): "The Volume algorithm: producing primal 
 *   solutions with a subgradient method"
 * - RLT1 formulation for QAP
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

#include "rlt1_volume_solver.hpp"
#include "../../include/qap_solution_io.hpp"

// Parse fixed variables from a text file with lines:
//   x i u value   (value in {0,1})
//   y i u j v value (value in {0,1})
// Indices are 0-based; '#' starts a comment line.
#define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v))
bool parse_fixed_file(const std::string &path, int n, FixedVariables &out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cout << "\rError: cannot open fixed-variable file: " << path << std::flush;
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
                std::cout << "\rWarning: malformed x-line at " << path << ":" << lineno << std::flush;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::flush;
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
                std::cout << "\rWarning: malformed y-line at " << path << ":" << lineno << std::flush;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || j < 0 || j >= n || v < 0 || v >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::flush;
                continue;
            }
            int key = key_y(n, i, u, v);
            if (val == 1) {
                out.y_fixed_1[key] = j;
            } else {
                out.y_fixed_0[key].insert(j);
            }
        } else {
            std::cout << "\rWarning: unknown line type at " << path << ":" << lineno << std::flush;
        }
    }

    std::cout << "\rLoaded fixed variables from " << path
              << " | x1=" << out.x_fixed_1.size()
              << " x0-rows=" << out.x_fixed_0.size()
              << " y1=" << out.y_fixed_1.size()
              << " y0-rows=" << out.y_fixed_0.size() << std::flush;
    return true;
}


PrimalViolationSummary compute_primal_violation_summary(const VOL_dvector &psol, int n) {
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

FixedViolationReport check_fixed_violations(const FixedVariables &fv, const VOL_dvector &psol, int n, double tol) {
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
 * RLT1 Volume Hooks Implementation
 */

RLT1VolumeHooks1::RLT1VolumeHooks1(const Problem& data) 
    : RLT1VolumeHooks(data) {
    beta.resize(n * n * n);
    beta_j_ind.resize(n * n * n);
    alpha.resize(n);
    alpha_u_ind.resize(n);
}

/**
    * Solve RLT1 Lagrangian Subproblem
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
int RLT1VolumeHooks1::solve_subproblem_1(const VOL_dvector& pi,
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
                auto it_y1 = fixed.y_fixed_1.find(ykey);
                if (it_y1 != fixed.y_fixed_1.end()) {
                    int j = it_y1->second;
                    double cost = qap_data.D[i][j] * qap_data.F[u][v];
                    if (i <= j) cost -= lambda(i, u, j, v);
                    if (i >= j) cost += lambda(j, v, i, u);
                    cost -= theta(i, u, j);
                    min_cost = cost;
                    best_j = j;
                } else {
                    const auto it_y0 = fixed.y_fixed_0.find(ykey);
                    for (int j = 0; j < n; ++j) {
                        if (it_y0 != fixed.y_fixed_0.end() && it_y0->second.count(j)) {
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
                    // std::cout << "Warning: no feasible j for y[" << i << "," << u << ",*," << v << "] under fixed variables; choosing j=0 with large cost." << std::flush;
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
        auto it_x1 = fixed.x_fixed_1.find(i);
        if (it_x1 != fixed.x_fixed_1.end()) {
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
            const auto it_x0 = fixed.x_fixed_0.find(i);
            for (int u = 0; u < n; ++u) {
                if (it_x0 != fixed.x_fixed_0.end() && it_x0->second.count(u)) {
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
            // std::cout << "Warning: no feasible u for fixed variables at facility i=" << i << "; choosing u=0." << std::flush;
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

int RLT1VolumeHooks1::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_1(pi, lcost, psol, vio, pcost);
}

RLT1VolumeHooks2::RLT1VolumeHooks2(const Problem& data): RLT1VolumeHooks(data) {}

int RLT1VolumeHooks2::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_2(pi, lcost, psol, vio, pcost);
                            }
    
    
int RLT1VolumeHooks2::solve_subproblem_2(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    // Alternative subproblem solver for RLT1 (Volume algorithm)
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
                    if (i == j && u == v) continue; // skip diagonal to avoid double counting
                    if (fixed.y_fixed_1.count(key_y(n, i, u, v))) {
                        int fixed_j = fixed.y_fixed_1.at(key_y(n, i, u, v));
                        y(i, u, fixed_j, v) = 1.0;
                        y(fixed_j, v, i, u) = 1.0;
                        continue;
                    }
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
            if (fixed.x_fixed_1.count(i)) {
                int fixed_u = fixed.x_fixed_1.at(i);
                if (u == fixed_u) {
                    x(i, u) = 1.0;
                }
                continue;
            }
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
    //     std::cout << "Vio u[" << u << "] = " << vio_mu1(u) << std::flush;
    // }
    // for (auto i= 0; i < n; ++i) {
    //     std::cout << "Vio i[" << i << "] = " << vio_mu2(i) << std::flush;
    // }

    auto total_vio = 0.0;
    #pragma omp parallel for reduction(+:total_vio) schedule(static)
    for (int k = 0; k < vio.size(); ++k) {
        total_vio += vio[k];
    }
    // std::cout << "Total violation: " << total_vio << std::flush;

    // Print costs for debugging
    // std::cout << "Primal cost: " << pcost << ", Lagrangian cost: " << lcost << std::flush;

    return 0;
}

RLT1VolumeHooks3::RLT1VolumeHooks3(const Problem& data): RLT1VolumeHooks(data) {
        zsol.resize(n * n * n * n);
    }

int RLT1VolumeHooks3::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_3(pi, lcost, psol, vio, pcost);
                            }

bool RLT1VolumeHooks3::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    for (const auto &kv : fixed.map_fixed) {
        int idx = kv.first;
        double val = kv.second;
        if (std::abs(val) > 1e-9 && std::abs(val - 1.0) > 1e-9) {
            continue; // ignore fractional fixes
        }
        if (idx >= n * n) {
            int offset = idx - n * n;
            int i = offset / (n * n * n);
            int rem = offset % (n * n * n);
            int u = rem / (n * n);
            rem = rem % (n * n);
            int j = rem / n;
            int v = rem % n;
            // y(i,u,j,v) fixed to val, need to check symmetry
            auto idx_symm = n * n + j * n * n * n + v * n * n + i * n + u;
            if (fixed.map_fixed.count(idx_symm)) {
                double symm_val = fixed.map_fixed.at(idx_symm);
                if (std::abs(symm_val - val) > 1e-9) {
                    std::cout << "Inconsistent fixed variables for y(" << i << "," << u << "," << j << "," << v << ") and its symmetric." << std::flush;
                    return false; // inconsistent fixed variables
                }
            }
        }
    }
    return true;
}

int RLT1VolumeHooks3::solve_subproblem_3(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    // Alternative subproblem solver for RLT1 (Volume algorithm)
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    std::fill(zsol.begin(), zsol.end(), 0.0);
    int retval = 0;

    // Step 2: Set y variables by checking reduced cost for each (i,u,j,v) and its symmetric (j,v,i,u)
    // If the reduced cost (coeff_y) is negative, set both y(i,u,j,v) and y(j,v,i,u) to 1 (symmetry)
    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                for (int j = i; j < n; ++j) {
                    // if y(i,u,j,v) is fixed
                    int key_y = n * n + i * n * n * n + u * n * n + j * n + v;
                    int key_y_symm = n * n + j * n * n * n + v * n * n + i * n + u;
                    if (fixed.map_fixed.count(key_y) || fixed.map_fixed.count(key_y_symm)) {
                        double val = fixed.map_fixed.at(key_y);
                        if (std::abs(val - 1.0) < 1e-9) {
                            z(i, u, j, v) = 1.0;
                            z(j, v, i, u) = 1.0;
                        } else if (std::abs(val) < 1e-9) {
                            z(i, u, j, v) = 0.0;
                            z(j, v, i, u) = 0.0;
                        }
                        continue;
                    }
                    
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
    if (retval == -1) {
        return retval; // infeasible fixed variables detected
    }

    // Step 3: Set x variables by checking reduced cost for each (i,u)
    // If the reduced cost (coeff_x) is negative, set x(i,u) = 1
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            // Check if x(i,u) is fixed
            if (fixed.map_fixed.count(i * n + u)) {
                double val = fixed.map_fixed.at(i * n + u);
                if (std::abs(val - 1.0) < 1e-9) {
                    x(i, u) = 1.0;
                } else if (std::abs(val) < 1e-9) {
                    x(i, u) = 0.0;
                }
                continue;
            }
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
    //     std::cout << "Vio u[" << u << "] = " << vio_mu1(u) << std::flush;
    // }
    // for (auto i= 0; i < n; ++i) {
    //     std::cout << "Vio i[" << i << "] = " << vio_mu2(i) << std::flush;
    // }

    // auto total_vio = 0.0;
    // #pragma omp parallel for reduction(+:total_vio) schedule(static)
    // for (int k = 0; k < vio.size(); ++k) {
    //     total_vio += vio[k];
    // }
    // std::cout << "Total violation: " << total_vio << std::flush;

    // Print costs for debugging
    // std::cout << "Primal cost: " << pcost << ", Lagrangian cost: " << lcost << std::flush;

    return 0;
}

RLT1VolumeHooks4::RLT1VolumeHooks4(const Problem& data) 
    : RLT1VolumeHooks(data) {
    beta.resize(n * n * n);
    beta_j_ind.resize(n * n * n);
    alpha.resize(n);
    alpha_u_ind.resize(n);
    zsol.resize(n * n * n * n);
}

/**
    * Solve RLT1 Lagrangian Subproblem
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
int RLT1VolumeHooks4::solve_subproblem_4(const VOL_dvector& pi,
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
                // If y[i,u,*,v] has a fixed 1, only evaluate that j
                auto it_y1 = fixed.y_fixed_1.find(key_y(n, i, u, v));
                if (it_y1 != fixed.y_fixed_1.end()) {
                    int j = it_y1->second;
                    double cost = qap_data.D[i][j] * qap_data.F[u][v];
                    if (i <= j) cost -= lambda(i, u, j, v);
                    if (i >= j) cost += lambda(j, v, i, u);
                    cost -= theta(i, u, j);
                    min_cost = cost;
                    best_j = j;
                } else {
                    // If y[i,u,*,v] has fixed zeros, skip those j
                    const auto it_y0 = fixed.y_fixed_0.find(key_y(n, i, u, v));
                    for (int j = 0; j < n; ++j) {
                        if (it_y0 != fixed.y_fixed_0.end() && it_y0->second.count(j)) {
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
                    // std::cout << "Warning: no feasible j for y[" << i << "," << u << ",*," << v << "] under fixed variables; choosing j=0 with large cost." << std::flush;
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
        auto it_x1 = fixed.x_fixed_1.find(i);
        if (it_x1 != fixed.x_fixed_1.end()) {
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
            // If x[i,*] has fixed zeros, skip those u
            const auto it_x0 = fixed.x_fixed_0.find(i);
            for (int u = 0; u < n; ++u) {
                if (it_x0 != fixed.x_fixed_0.end() && it_x0->second.count(u)) {
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
            // std::cout << "Warning: no feasible u for fixed variables at facility i=" << i << "; choosing u=0." << std::flush;
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
        double sum = 0.0;
        for (int v = 0; v < m_eff; ++v) sum += x(j,v);
        ind_real_j[j] = std::min(1.0, sum);
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

bool RLT1VolumeHooks4::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    // check fixed variables consistency
    // for (const auto &kv : fixed.y_fixed_1) {
        
    // }
    return true;
}

int RLT1VolumeHooks4::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_4(pi, lcost, psol, vio, pcost);
                            }

void set_up_volume_parameters(VOL_problem &vol_problem, bool verbose) {
    // Set Volume algorithm parameters (from qap.par defaults)
    vol_problem.parm.lambdainit = 0.01;
    vol_problem.parm.alphainit = 0.01;
    vol_problem.parm.alphamin = 0.0001;
    vol_problem.parm.alphafactor = 0.66;
    vol_problem.parm.alphaint = 50;
    
    vol_problem.parm.maxsgriters = 100000000; // effectively no limit
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
    vol_problem.parm.printflag = verbose ? 3 : 0;  // 1=iteration info, 3=add lambda info
    vol_problem.parm.printinvl = 100;
    vol_problem.parm.heurinvl = 100;
}

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


VolumeResult solve_rlt1_volume_relax(const Problem &problem,
                                     const FixedVariables &fixed,
                                     bool verbose,
                                     const VOL_dvector &initial_dual,
                                     std::string formulation) {
    VOL_problem vol_problem;
    RLT1VolumeHooks* hooks = nullptr; // default initialization
    set_up_vol_problem_4(vol_problem, problem, fixed);
    // Create hooks
    if (formulation == "formulation1") {
        hooks = new RLT1VolumeHooks1(problem);
    } else if (formulation == "formulation2") {
        hooks = new RLT1VolumeHooks2(problem);
    } else if (formulation == "formulation3") {
        hooks = new RLT1VolumeHooks3(problem);
    } else if (formulation == "formulation4") {
        hooks = new RLT1VolumeHooks4(problem);
    } else {
        std::cout << "\rError: Unknown formulation specified." << std::flush;
        throw std::invalid_argument("Unknown formulation");
    }
    // Load initial dual solution if provided
    bool use_dual_warmstart = false;
    if (initial_dual.size() > 0) {
        if (initial_dual.size() != vol_problem.dsize) {
            std::cout << "\rError: Initial dual vector size does not match problem dual size." << std::flush;
        }
        vol_problem.dsol = initial_dual;
        use_dual_warmstart = true;
    }
    set_up_volume_parameters(vol_problem, verbose);
    bool fixed_ok = hooks->set_fixed_variables(fixed);
    if (!fixed_ok) {
        if (verbose) std::cerr << "\rError: Inconsistent fixed variables provided." << std::flush;

        VolumeResult result;
        result.lower_bound = DBL_MAX;
        return result; // return infinite bound
    } else {
        if (verbose) std::cout << "\rFixed variables set successfully." << std::flush;
    }
    // Solve the Volume relaxation
    
    int retval =vol_problem.solve(*hooks, use_dual_warmstart);
    
    // Extract results
    VolumeResult result;
    result.status = retval;
    result.lower_bound = vol_problem.value;
    result.dual = vol_problem.dsol;
    return result;
}


