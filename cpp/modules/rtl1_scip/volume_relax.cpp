#include "volume_relax.hpp"


#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "../rtl1_volume/VolVolume.hpp"

static int key_y(int n, int i, int u, int v) { return i * n * n + u * n + v; }

FixedVariables build_fixed_from_map(const std::unordered_map<int, double> &map_fixed, int n) {
    FixedVariables fv;
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

// Volume hooks (trimmed from rtl1_volume_solver.cpp)
#define mu(u) (pi[u])
#define lambda(i, u, j, v) (pi[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define theta(i, u, j) (pi[n + n * n * n * n + (i) * n * n + (u) * n + (j)])
#define x(i, u) (psol[(i) * n + (u)])
#define y(i, u, j, v) (psol[n * n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_mu(u) (vio[u])
#define vio_lambda(i, u, j, v) (vio[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_theta(i, u, j) (vio[n + n * n * n * n + (i) * n * n + (u) * n + (j)])

class RTL1VolumeHooks : public VOL_user_hooks {
private:
    const Problem &qap_data;
    int n;

    std::vector<double> beta;
    std::vector<int> beta_j_ind;
    std::vector<double> alpha;
    std::vector<int> alpha_u_ind;

    FixedVariables fixed;
    std::unordered_map<int, int> x_fixed_1;
    std::unordered_map<int, std::unordered_set<int>> x_fixed_0;
    std::unordered_map<int, int> y_fixed_1;
    std::unordered_map<int, std::unordered_set<int>> y_fixed_0;
    bool first_iter = true;

public:
    double first_iter_lb = 0;
    RTL1VolumeHooks(const Problem &data, const FixedVariables &fv)
        : qap_data(data), n(data.n), fixed(fv) {
        beta.resize(n * n * n);
        beta_j_ind.resize(n * n * n);
        alpha.resize(n);
        alpha_u_ind.resize(n);
        x_fixed_1 = fv.x_fixed_1;
        x_fixed_0 = fv.x_fixed_0;
        y_fixed_1 = fv.y_fixed_1;
        y_fixed_0 = fv.y_fixed_0;
    }

    virtual int compute_rc(const VOL_dvector & /*pi*/, VOL_dvector &rc) override {
        rc = 0;
        return 0;
    }

    virtual int solve_subproblem(const VOL_dvector &pi, const VOL_dvector & /*rc*/,
                                 double &lcost, VOL_dvector &psol, VOL_dvector &vio,
                                 double &pcost) override {
        // STEP 1: beta
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
                            if (it_y0 != y_fixed_0.end() && it_y0->second.count(j)) continue;
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
                        beta[idx] = 1e30;
                        beta_j_ind[idx] = 0;
                    } else {
                        beta[idx] = min_cost;
                        beta_j_ind[idx] = best_j;
                    }
                }
            }
        }

        // STEP 2: alpha
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            double min_cost = DBL_MAX;
            int best_u = 0;
            auto it_x1 = x_fixed_1.find(i);
            if (it_x1 != x_fixed_1.end()) {
                int u = it_x1->second;
                double cost = -mu(u);
                for (int j = 0; j < n; ++j) cost += theta(i, u, j);
                for (int v = 0; v < n; ++v) cost += beta[i * n * n + u * n + v];
                min_cost = cost; best_u = u;
            } else {
                const auto it_x0 = x_fixed_0.find(i);
                for (int u = 0; u < n; ++u) {
                    if (it_x0 != x_fixed_0.end() && it_x0->second.count(u)) continue;
                    double cost = -mu(u);
                    for (int j = 0; j < n; ++j) cost += theta(i, u, j);
                    for (int v = 0; v < n; ++v) cost += beta[i * n * n + u * n + v];
                    if (cost < min_cost) { min_cost = cost; best_u = u; }
                }
            }
            if (min_cost == DBL_MAX) { min_cost = 1e30; best_u = 0; }
            alpha[i] = min_cost;
            alpha_u_ind[i] = best_u;
        }

        // STEP 3: lower bound
        lcost = 0.0;
        #pragma omp parallel for reduction(+:lcost) schedule(static)
        for (int i = 0; i < n; ++i) lcost += alpha[i];
        for (int u = 0; u < n; ++u) lcost += mu(u);

        // STEP 4: primal construction
        psol = 0.0;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) x(i, alpha_u_ind[i]) = 1.0;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int v = 0; v < n; ++v) {
                int u = alpha_u_ind[i];
                int j = beta_j_ind[i * n * n + u * n + v];
                y(i, u, j, v) = 1.0;
            }
        }

        // STEP 5: primal objective
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

        // STEP 6: constraint violations
        vio = 0.0;
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < n; ++u) {
            double sum = 0.0; for (int i = 0; i < n; ++i) sum += x(i, u); vio_mu(u) = 1.0 - sum; }
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < n; ++i) for (int u = 0; u < n; ++u) for (int j = 0; j < n; ++j) {
            double sum = 0.0; for (int v = 0; v < n; ++v) sum += y(i, u, j, v); vio_theta(i, u, j) = x(i, u) - sum; }
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < n; ++i) for (int u = 0; u < n; ++u) for (int j = 0; j < n; ++j) for (int v = 0; v < n; ++v) {
            vio_lambda(i, u, j, v) = y(j, v, i, u) - y(i, u, j, v); }
        
        if (first_iter) {
            first_iter_lb = lcost;
            first_iter = false;
        }
        return 0;
    }

    virtual int heuristics(const VOL_problem & /*p*/, const VOL_dvector & /*psol*/, double & /*heur_val*/) override {
        return 0;
    }
};

VolumeResult solve_rtl1_volume_relax(const Problem &problem,
                                     const FixedVariables &fixed,
                                     int threads,
                                     double time_limit,
                                     bool verbose,
                                     bool print_info,
                                     const std::vector<double> &seed_dual) {

    VolumeResult result;
    int n = problem.n;
    omp_set_num_threads(threads);

    VOL_problem vol_problem;
    vol_problem.psize = n * n + n * n * n * n;
    vol_problem.dsize = n + n * n * n * n + n * n * n;
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    vol_problem.dsol.allocate(vol_problem.dsize);

    // Warm-start with seed dual if provided
    if (seed_dual.size() == static_cast<size_t>(vol_problem.dsize)) {
        for (int i = 0; i < vol_problem.dsize; ++i) {
            vol_problem.dsol[i] = seed_dual[i];
        }
        std::cout << "[VolumeRelax] Warm-starting volume relaxation with provided dual solution." << std::endl;
    } else {
        vol_problem.dsol = 0.0;
    }

    vol_problem.parm.lambdainit = 0.1;
    vol_problem.parm.alphainit = 0.1;
    vol_problem.parm.alphamin = 0.001;
    vol_problem.parm.alphafactor = 0.66;
    vol_problem.parm.alphaint = 50;
    vol_problem.parm.maxsgriters = 100000000;
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
    vol_problem.parm.printflag = verbose ? 3 : 0;
    vol_problem.parm.printinvl = 100;
    vol_problem.parm.heurinvl = 100;

    RTL1VolumeHooks hooks(problem, fixed);

    // Timing
    auto relax_start = std::chrono::high_resolution_clock::now();

    int retval = vol_problem.solve(hooks, false);
    result.status = retval;
    result.lower_bound = vol_problem.value;
    result.primal.assign(vol_problem.psol.v, vol_problem.psol.v + vol_problem.psol.size());
    result.dual.assign(vol_problem.dsol.v, vol_problem.dsol.v + vol_problem.dsol.size());

    auto relax_end = std::chrono::high_resolution_clock::now();
    double relax_time = std::chrono::duration<double>(relax_end - relax_start).count();

    if (print_info) {
        int n = problem.n;
        int num_fixed_x = 0, num_fixed_y = 0;
        for (const auto& kv : fixed.x_fixed_1) num_fixed_x++;
        for (const auto& kv : fixed.x_fixed_0) num_fixed_x += kv.second.size();
        for (const auto& kv : fixed.y_fixed_1) num_fixed_y++;
        for (const auto& kv : fixed.y_fixed_0) num_fixed_y += kv.second.size();

        // Violation summary: compute max violation for each type and average violation
        double max_vio_mu = 0.0, max_vio_theta = 0.0, max_vio_lambda = 0.0;
        double sum_viol = 0.0;
        int count_viol = 0;
        // mu(u): n entries
        for (int u = 0; u < n; ++u) {
            double v = std::abs(vol_problem.viol[u]);
            max_vio_mu = std::max(max_vio_mu, v);
            sum_viol += v;
            count_viol++;
        }
        // theta(i,u,j): n^3 entries
        int theta_offset = n + n * n * n * n;
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    int idx = theta_offset + i * n * n + u * n + j;
                    double v = std::abs(vol_problem.viol[idx]);
                    max_vio_theta = std::max(max_vio_theta, v);
                    sum_viol += v;
                    count_viol++;
                }
            }
        }
        // lambda(i,u,j,v): n^4 entries
        int lambda_offset = n;
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        int idx = lambda_offset + i * n * n * n + u * n * n + j * n + v;
                        double val = std::abs(vol_problem.viol[idx]);
                        max_vio_lambda = std::max(max_vio_lambda, val);
                        sum_viol += val;
                        count_viol++;
                    }
                }
            }
        }
        double avg_viol = (count_viol > 0) ? (sum_viol / count_viol) : 0.0;
        std::cout << "[RelaxationInfo] lb: " << result.lower_bound
                  << ", fixed_x: " << num_fixed_x
                  << ", fixed_y: " << num_fixed_y
                  << ", max|mu|: " << max_vio_mu
                  << ", max|theta|: " << max_vio_theta
                  << ", max|lambda|: " << max_vio_lambda
                  << ", avg|viol|: " << avg_viol
                  << ", relax_time: " << relax_time
                  << ", first_iter_lb: " << hooks.first_iter_lb
                  << std::endl;
    }
    return result;
}
