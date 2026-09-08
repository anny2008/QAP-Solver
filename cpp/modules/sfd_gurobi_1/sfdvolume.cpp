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

#include "sfdvolume.hpp"
#include "../../include/qap_solution_io.hpp"
#include "SFDGurobi.h"


// Macro definitions for cleaner indexing
#define y(i, u, j, v) (psol[n * n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define alpha(i,u,v) (pi[(i)*n*n + (u)*n + (v)])
#define beta(i,u,j) (pi[n*n*n + (i)*n*n + (u)*n + (j)])
#define gamma(k,i,j) (pi[2*n*n*n + (k)*n*n + (i)*n + (j)])
#define vio_alpha(i,u,v) (vio[(i)*n*n + (u)*n + (v)])
#define vio_beta(i,u,j) (vio[n*n*n + (i)*n*n + (u)*n + (j)])
#define vio_gamma(k,i,j) (vio[2*n*n*n + (k)*n*n + (i)*n + (j)])
#define x_sol(i,u) (x_sol[(i)*n + (u)])
#define e_sol(k,i,j) (e_sol[(k)*n*n + (i)*n + (j)])

SFDVolumeHooks::SFDVolumeHooks(Problem& problem, std::map<int, SubgraphData>& subgraphs)
    : problem(problem), subgraphs(subgraphs), n(problem.n), K(subgraphs.size()) {
    e_sol.resize(K * n * n, 0.0);
    x_sol.resize(n * n, 0.0);
    uv_to_k.resize(n * n, -1);
    for (const auto& kv : subgraphs) {
        int k = kv.first;
        const auto& G_k = kv.second;
        for (const auto& uv : G_k.arcs) {
            int u = uv.first;
            int v = uv.second;
            uv_to_k[u * n + v] = k;
        }
    }
}
int SFDVolumeHooks::compute_rc(const VOL_dvector&, VOL_dvector& rc)
{
    rc = 0;
    return 0;
}
/* 
 * min (d_ij*f_uv + d_ji*f_vu) * y_iujv 
 *      + alpha(i,u,v) * (sum_j y_ijuv - x_iu) 
 *      + beta(i,u,j) * (sum_v y_ijuv - x_iu)
 *      + gamma(k,i,j) * (sum_uv y_ijuv - e_kij)
 * st. y_iujv = y_jvui
 * vio_alpha(i,u,v) = sum_j  y_ijuv - x_iu
 * vio_beta(i,u,j)  = sum_v  y_ijuv - x_iu
 * vio_gamma(k,i,j) = sum_uv y_ijuv - e_kij
 * lcost = pcost + alpha(i,u,v) * vio_alpha(i,u,v)
 *               + beta(i,u,j) * vio_beta(i,u,j)
 *               + gamma(k,i,j) * vio_gamma(k,i,j)
 */
int SFDVolumeHooks::solve_subproblem(const VOL_dvector &pi, const VOL_dvector &rc,
                                        double &lcost, VOL_dvector &psol, VOL_dvector &vio,
                                        double &pcost) 
{
    psol = 0;
    pcost = 0.0;
    lcost = 0.0;
    vio = 0;
    // solve subproblem based on the provided dual variables (pi)
    // coeff_y = d_ij*f_uv + d_ji*f_vu
    //           + alpha(i,u,v) + alpha(j,v,u)
    //           + beta(i,u,j) + beta(j,v,i)
    //           + gamma(k,i,j) + gamma(k,j,i)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = u + 1; v < n; ++v) {
                    if (i==j) continue; // skip invalid cases
                    auto k1 = uv_to_k[u * n + v];
                    auto k2 = uv_to_k[v * n + u];
                    double coeff_y = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]
                                    + alpha(i, u, v)  + alpha(j, v, u)
                                    + beta(i, u, j)   + beta(j, v, i)
                                    + gamma(k1, i, j) + gamma(k2, j, i)
                                    ;
                    if (coeff_y < 0) {
                        y(i, u, j, v) = 1.0;
                        pcost += problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                    }
                }
            }
        }
    }
    // compute lcost and pcost
    lcost = pcost;
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                if (i==j) 
                    continue;
                vio_alpha(i, u, j) = -x_sol(i, u);
                vio_beta(i, u, j) = -x_sol(i, u);
                for (int v = 0; v < n; ++v) {
                    if(u == v) 
                        continue;
                    vio_alpha(i, u, v) += y(i, u, j, v);
                    vio_beta(i, u, v)  += y(i, u, v, j);
                }
                lcost += alpha(i, u, j) * vio_alpha(i, u, j);
                lcost += beta(i, u, j)  * vio_beta(i, u, j);
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i==j) 
                    continue;
                double sum = 0.0;
                for (auto kv: subgraphs[k].arcs) {
                    int u = kv.first;
                    int v = kv.second;
                    sum += y(i, u, j, v);
                }
                vio_gamma(k, i, j) =  sum - e_sol(k, i, j);
                lcost += gamma(k,i,j) * vio_gamma(k,i,j);
            }
        }
    }
    if (lcost > UB && farkas_ray.empty()) {
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    if (j==u) {
                        farkas_ray["alpha_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j)] = 0;
                        continue;
                    }
                    if (j==i) {
                        farkas_ray["beta_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j)] = 0;
                        continue;
                    }
                    farkas_ray["alpha_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j)] = vio_alpha(i, u, j);
                    farkas_ray["beta_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j)] = vio_beta(i, u, j);
                }
            }
        }
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i==j) {
                        farkas_ray["gamma_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j)] = 0;
                        continue;
                    }
                    farkas_ray["gamma_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j)] = -vio_gamma(k, i, j);
                }
            }
        }
    }

    if (!farkas_ray.empty()) {
        bool verification_passed = true;
        // compare the computed Farkas ray with the stored one for verification
        double coeff = 0.0;
        for (const auto& kv : farkas_ray) {
            const std::string& key = kv.first;
            double stored_value = kv.second;
            double computed_value = 0.0;
            if (key.find("alpha_") == 0) {
                int i, u, v;
                sscanf(key.c_str(), "alpha_%d_%d_%d", &i, &u, &v);
                computed_value = -vio_alpha(i, u, v);
            } else if (key.find("beta_") == 0) {
                int i, u, j;
                sscanf(key.c_str(), "beta_%d_%d_%d", &i, &u, &j);
                computed_value = -vio_beta(i, u, j);
            } else if (key.find("gamma_") == 0) {
                int k, i, j;
                sscanf(key.c_str(), "gamma_%d_%d_%d", &k, &i, &j);
                computed_value = -vio_gamma(k, i, j); 
            }
            if (coeff == 0.0 && stored_value != 0.0) {
                coeff = computed_value/stored_value;
                // std::cout << "Setting coeff to " << coeff << " based on key: " << key 
                //           << ", stored value: " << stored_value 
                //           << ", computed value: " << computed_value << std::endl;
            } else if (stored_value != 0.0) {
                double ratio = computed_value/stored_value;
                if (std::abs(ratio - coeff) > 1e-6) {
                    // std::cerr << "Farkas ray verification failed for key: " << key 
                    //           << ", stored value: " << stored_value 
                    //           << ", computed value: " << computed_value 
                    //           << ", ratio: " << ratio 
                    //           << ", expected ratio: " << coeff << std::endl;
                    verification_passed = false;
                }
            }
        }
        if (!verification_passed) {
            UB = lcost*2; // Update UB to the current lcost if verification fails
        } else {
            UB = lcost-1; // Update UB to the current lcost if verification passes
            std::cout << "Farkas ray verification passed. Updated UB to: " << UB << std::endl;
        }
    }
    // if lcost is nan or inf, return -1 to indicate failure
    if (std::isnan(lcost) || std::isinf(lcost)) {
        return -1;
    }

    return 0;
}

int SFDVolumeHooks::heuristics(const VOL_problem&, const VOL_dvector&, double&)
{
    return 0;
}