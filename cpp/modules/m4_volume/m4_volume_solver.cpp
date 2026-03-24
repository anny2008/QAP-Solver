/**
 * M4 Volume Algorithm Solver for QAP
 * 
 * Implements the Lagrangian relaxation-based Volume algorithm
 * for the M4 formulation of the Quadratic Assignment Problem.
 * 
 * Based on:
 * - Barahona & Anbil (1998): "The Volume algorithm: producing primal 
 *   solutions with a subgradient method"
 * - M4 formulation for QAP
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

#include "m4_volume_solver.hpp"
#include "../../include/qap_solution_io.hpp"

// Parse fixed variables from a text file with lines:
//   x i u value   (value in {0,1})
//   y i u j v value (value in {0,1})
// Indices are 0-based; '#' starts a comment line.
#define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v))
bool parse_fixed_file(const std::string &path, int n, FixedVariables &out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cout << "\rError: cannot open fixed-variable file: " << path << std::endl;
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
                std::cout << "\rWarning: malformed x-line at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::endl;
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
                std::cout << "\rWarning: malformed y-line at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || j < 0 || j >= n || v < 0 || v >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::endl;
                continue;
            }
            int key = key_y(n, i, u, v);
            if (val == 1) {
                out.y_fixed_1[key] = j;
            } else {
                out.y_fixed_0[key].insert(j);
            }
        } else {
            std::cout << "\rWarning: unknown line type at " << path << ":" << lineno << std::endl;
        }
    }

    std::cout << "\rLoaded fixed variables from " << path
              << " | x1=" << out.x_fixed_1.size()
              << " x0-rows=" << out.x_fixed_0.size()
              << " y1=" << out.y_fixed_1.size()
              << " y0-rows=" << out.y_fixed_0.size() << std::endl;
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
#define lambda1(i, j) (pi[(i) * n + (j)])
#define lambda2(i, v) (pi[n*n + (i) * n + (v)])
#define lambda3(j, u) (pi[2*n*n + (j) * n + (u)])
#define lambda4(u, v) (pi[3*n*n + (u) * n + (v)])
#define lambda(idx) (pi[4*n*n + (idx)])
#define y(i, u, j, v) (psol[(i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_lambda1(i, j) (vio[(i) * n + (j)])
#define vio_lambda2(i, v) (vio[n*n + (i) * n + (v)])
#define vio_lambda3(j, u) (vio[2*n*n + (j) * n + (u)])
#define vio_lambda4(u, v) (vio[3*n*n + (u) * n + (v)])
#define vio_lambda(idx) (vio[4*n*n + (idx)])
#define ind_y(i, u, j, v) ((i) * n * n * n + (u) * n * n + (j) * n + (v))

/**
 * M4 Volume Hooks Implementation
 */

M4VolumeHooks1::M4VolumeHooks1(const Problem& data) 
    : M4VolumeHooks(data){
}

inline std::tuple<int, int, int, int> decode_index_y(int ind_y, int n) {
    // ind_y = i * n * n * n + u * n * n + j * n + v
    int i = ind_y / (n * n * n);
    int rem1 = ind_y % (n * n * n);
    int u = rem1 / (n * n);
    int rem2 = rem1 % (n * n);
    int j = rem2 / n;
    int v = rem2 % n;
    return std::make_tuple(i, u, j, v);
}

/**
    * Solve M4 Lagrangian Subproblem
    * 
    * DUAL VARIABLES (pi):
    *   - lambda1[i,j]:    Lagrange multipliers for sum_uv y[i,u,j,v] = 1
    *   - lambda2[i,v]:    Lagrange multipliers for sum_uj y[i,u,j,v] = 1
    *   - lambda3[u,j]:    Lagrange multipliers for sum_iv y[i,u,j,v] = 1
    *   - lambda4[u,v]:    Lagrange multipliers for sum_ij y[i,u,j,v] = 1
    *   - lambda[i,u,j,v]: Multipliers for sum_k y[i,u,k,v] = sum_w y[i,u,j,w]
    * 
    * LAGRANGIAN:
    *   L = sum_{i,j,u,v} d[i,j]*f[u,v]*y[i,u,j,v]
    *       + sum_ij lambda1[i,j]* (1 - sum_uv y[i,u,j,v])
    *       + sum_iv lambda2[i,v]* (1 - sum_uj y[i,u,j,v])
    *       + sum_uj lambda3[u,j]* (1 - sum_iv y[i,u,j,v])
    *       + sum_ij lambda4[u,v]* (1 - sum_ij y[i,u,j,v])
    *       + sum_iujv lambda[i,u,j,v]* (sum_w y[i,u,j,w] - sum_k y[i,u,k,v])
    * 
    * ALGORITHM:
    *   1. Compute coefficient for y[i,u,j,v]: 
                            d[i,j]*f[u,v] 
                            -lambda1[i,j] - lambda2[i,v] - lambda3[j,u] - lambda4[u,v]
                            - sum_k lambda[i,u,k,v] + sum_w lambda[i,u,j,w]
    *   2. Set y[i,u,j,v]=1 and y[j,v,i,u]=1 if sum coefficient < 0
    *   3. Compute lagragian cost, primer cost, violations for all constraints
    */
int M4VolumeHooks1::solve_subproblem_1(const VOL_dvector& pi,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
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
                            y(i, u, j, v) = 1.0;
                            y(j, v, i, u) = 1.0;
                        } else if (std::abs(val) < 1e-9) {
                            y(i, u, j, v) = 0.0;
                            y(j, v, i, u) = 0.0;
                        }
                        continue;
                    }
                    
                    // Compute reduced cost for y(i,u,j,v)
                    double coeff_y = qap_data.D[i][j] * qap_data.F[u][v]
                                    + qap_data.D[j][i] * qap_data.F[v][u]
                                    - lambda1(i, j) - lambda1(j, i)
                                    - lambda2(i, v) - lambda2(j, u)
                                    - lambda3(j, u) - lambda3(i, v)
                                    - lambda4(u, v) - lambda4(v, u);
                    
                    for (int w = 0; w < n; ++w) {
                        auto key_iujw = ind_y(i, u, j, w);
                        auto it = violated_constraints.find(key_iujw);
                        if (it != violated_constraints.end()) { 
                            coeff_y += lambda(it->second);
                        }
                        auto key_jviw = ind_y(j, v, i, w);
                        it = violated_constraints.find(key_jviw);
                        if (it != violated_constraints.end()) { 
                            coeff_y += lambda(it->second);
                        }
                    }
                    for (int k = 0; k < n; ++k) {
                        auto key_iukv = ind_y(i, u, k, v);
                        auto it = violated_constraints.find(key_iukv);
                        if (it != violated_constraints.end()) { 
                            coeff_y -= lambda(it->second);
                        }
                        auto key_jvku = ind_y(j, v, k, u);
                        it = violated_constraints.find(key_jvku);
                        if (it != violated_constraints.end()) { 
                            coeff_y -= lambda(it->second);
                        }
                    }
                    if (coeff_y < 0) {
                        y(i, u, j, v) = 1.0;
                        y(j, v, i, u) = 1.0;
                    }
                }
            }
        }
    }
    if (retval == -1) {
        return retval; // infeasible fixed variables detected
    }

    // Step 4: Compute constraint violations for the current solution
    vio = 0.0;

    // Linking constraint 1: For each (i,j), 1 = sum_uv y[i,u,j,v]
    // Store violation in vio_lambda1(i,j) = 1 - sum_uv y[i,u,j,v]
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    sum += y(i, u, j, v);
                }
            }
            vio_lambda1(i, j) = 1.0 - sum;
        }
    }

    // Linking constraint 2: For each (i,v), 1 = sum_ju y[i,u,j,v]
    // Store violation in vio_lambda2(i,v) = 1 - sum_ju y[i,u,j,v]
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int v = 0; v < n; ++v) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                for (int u = 0; u < n; ++u) {
                    sum += y(i, u, j, v);
                }
            }
            vio_lambda2(i, v) = 1.0 - sum;
        }
    }

    // Linking constraint 3: For each (j,u), 1 = sum_iv y[i,u,j,v]
    // Store violation in vio_lambda3(j,u) = 1 - sum_iv y[i,u,j,v]
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < n; ++j) {
        for (int u = 0; u < n; ++u) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int v = 0; v < n; ++v) {
                    sum += y(i, u, j, v);
                }
            }
            vio_lambda3(j, u) = 1.0 - sum;
        }
    }

    // Linking constraint 4: For each (u,v), 1 = sum_ij y[i,u,j,v]
    // Store violation in vio_lambda4(u,v) = 1 - sum_ij y[i,u,j,v]
    #pragma omp parallel for collapse(2) schedule(static)
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    sum += y(i, u, j, v);
                }
            }
            vio_lambda4(u, v) = 1.0 - sum;
        }
    }
    
    // sum_w y[i,u,j,w] - sum_k y[i,u,k,v] = 0
    // Store violation in vio_lambda(idx)
    for (const auto &kv : violated_constraints) {
        int key = kv.first;
        int idx = kv.second;
        int i, u, j, v;
        std::tie(i, u, j, v) = decode_index_y(key, n);
        for (int w = 0; w < n; ++w) {
            vio_lambda(idx) += y(i, u, j, w);
        }
        for (int k = 0; k < n; ++k) {
            // if (k == j) continue;
            vio_lambda(idx) -= y(i, u, k, v);
        }
    }



    // Step 5: Compute primal cost (QAP objective value for current y)
    pcost = 0.0;
    #pragma omp parallel for collapse(4) reduction(+:pcost) schedule(static)
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
    #pragma omp parallel for collapse(2) reduction(+:lcost) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            lcost += lambda1(i, j) * vio_lambda1(i, j) 
                   + lambda2(i, j) * vio_lambda2(i, j)
                   + lambda3(i, j) * vio_lambda3(i, j)
                   + lambda4(i, j) * vio_lambda4(i, j);
        }
    }
    for (const auto &kv : violated_constraints) {
        int idx = kv.second;
        lcost += lambda(idx) * vio_lambda(idx);
    }

    return retval;
}

int M4VolumeHooks1::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_1(pi, lcost, psol, vio, pcost);
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

void set_up_vol_problem_1(VOL_problem &vol_problem, const Problem &qap_data, const int number_of_violated_constrains) {
    int n = qap_data.n;

    // Set problem dimensions
    // Primal variables: y[i,u,j,v] (n^4)
    vol_problem.psize = n * n * n * n; // M4 formulation
    
    // Dual variables: lambda1[i,j] (n^2) + lambda[i,v] (n^2) + lambda3[j,u] (n^2) + lambda4[u,v] (n^2) + lambda[idx] (number_of_violated_constrains)
    vol_problem.dsize = 4 * n * n + number_of_violated_constrains; // Adjusted for M4 formulation
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;
}

VolumeResult solve_m4_volume_relax(const Problem &problem,
                                     const FixedVariables &fixed,
                                     bool verbose,
                                     const VOL_dvector &initial_dual,
                                     std::string formulation,
                                    std::map<int, int>& violated_constraints) {
    VOL_problem vol_problem;
    // Create hooks
    M4VolumeHooks* hooks = new M4VolumeHooks1(problem);
    hooks->violated_constraints = violated_constraints;
    // Set up VOL_problem
    set_up_vol_problem_1(vol_problem, problem, violated_constraints.size()); // Placeholder for number_of_violated_constrains
    // Load initial dual solution if provided
    bool use_dual_warmstart = false;
    if (initial_dual.size() > 0) {
        if (initial_dual.size() != vol_problem.dsize) {
            // expand initial_dual to match vol_problem.dsize
            VOL_dvector expanded_dual;
            expanded_dual.allocate(vol_problem.dsize);
            expanded_dual = 0.0;
            int min_size = std::min((int)initial_dual.size(), vol_problem.dsize);
            for (int i = 0; i < min_size; ++i) {
                expanded_dual[i] = initial_dual[i];
            }
            vol_problem.dsol = expanded_dual;
            use_dual_warmstart = true;
        } else {
            vol_problem.dsol = initial_dual;
            use_dual_warmstart = true;
        }
    }
    set_up_volume_parameters(vol_problem, verbose);
    bool fixed_ok = hooks->set_fixed_variables(fixed);
    if (!fixed_ok) {
        if (verbose) std::cerr << "\rError: Inconsistent fixed variables provided." << std::endl;

        VolumeResult result;
        result.lower_bound = DBL_MAX;
        return result; // return infinite bound
    } else {
        if (verbose) std::cout << "\rFixed variables set successfully." << std::endl;
    }
    // Solve the Volume relaxation
    
    int retval = vol_problem.solve(*hooks, use_dual_warmstart);
    
    // Extract results
    VolumeResult result;
    result.status = retval;
    result.lower_bound = vol_problem.value;
    result.dual = vol_problem.dsol;
    result.primal = vol_problem.psol;
    return result;
}


