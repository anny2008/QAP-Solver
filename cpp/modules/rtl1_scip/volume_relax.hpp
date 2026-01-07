#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "../../core/problem.h"

struct FixedVariables {
    std::unordered_map<int, int> x_fixed_1;                    // i -> u
    std::unordered_map<int, std::unordered_set<int>> x_fixed_0; // i -> {u}
    std::unordered_map<int, int> y_fixed_1;                    // key(i,u,v) -> j
    std::unordered_map<int, std::unordered_set<int>> y_fixed_0; // key(i,u,v) -> {j}
};

struct VolumeResult {
    int status = -1;           // 0 success, non-zero otherwise
    double lower_bound = 0.0;
    std::vector<double> primal; // size n*n + n^4
    std::vector<double> dual;   // size n + n^4 + n^3
};

/**
 * Solve RTL1 relaxation using Volume algorithm.
 * @param problem QAP instance
 * @param fixed fixed x/y variables (0/1)
 * @param threads number of threads (OpenMP)
 * @param time_limit seconds (used to set vol params heuristically)
 * @param verbose enable iteration logs (vol printflag=3)
 * @param seed_dual optional dual vector to warm-start (empty = cold start)
 */
VolumeResult solve_rtl1_volume_relax(const Problem &problem,
                                     const FixedVariables &fixed,
                                     int threads,
                                     double time_limit,
                                     bool verbose,
                                     bool print_info,
                                     const std::vector<double> &seed_dual = {});

/**
 * Build FixedVariables from flattened index->value map used in SCIP nodes.
 * x index: i*n + u
 * y index: n*n + i*n^3 + u*n^2 + j*n + v
 */
FixedVariables build_fixed_from_map(const std::unordered_map<int, double> &map_fixed, int n);
