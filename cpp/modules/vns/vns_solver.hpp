
#pragma once
#include "../../core/problem.h"
#include "../../core/solution.h"
#include <vector>
#include <random>
#include <string>

struct VNSOptions {
    int max_iters = 1000;
    int k_max = 3;
    int seed = 42;
    std::vector<std::string> neighborhoods = {"swap", "reverse", "insert"};
    std::string initial_solution = "identity";
    std::string local_search_strategy = "best_improvement";
};

class VNSSolver {
public:
    VNSSolver(const Problem& problem, const VNSOptions& options);
    Solution solve();
private:
    const Problem& problem;
    VNSOptions options;
    std::mt19937 rng;
    std::vector<int> shake(const std::vector<int>& perm, int neighborhood);
    Solution local_search(const Solution& sol, int neighborhood);
};
