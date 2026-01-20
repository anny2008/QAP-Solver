#pragma once
#include "../../core/problem.h"
#include "../../core/solution.h"
// Difference matrix type
#include <vector>
#include <random>
#include <string>

using DiffMatrix = std::vector<std::vector<double>>;

struct GAPopMember {
    Solution sol;
    double obj;
    DiffMatrix diff;
    GAPopMember(const Solution& s, double o, const DiffMatrix& d) : sol(s), obj(o), diff(d) {}
};

struct GAOptions {
    int population_size = 100;
    int generations = 500;
    double crossover_rate = 0.9;
    double mutation_rate = 0.2;
    int tournament_size = 3;
    int seed = 42;
    std::string crossover_type = "universal"; // "cohesive" or "universal"
    std::string mutation = "swap"; // or "insert", "reverse"
    bool use_local_search = true;
    bool elitism = true;
    // Tabu Search (HITS) parameters
    int ts_iterations = -1; // If <0, use default: 100 + 0.1*n*n
    int ts_idle_factor = 5; // gamma, idle iterations factor
    double ts_alpha = 0.01; // randomization parameter for tabu criterion
    int ts_hts_size = 10007; // hash table size (prime)
    std::vector<int> Q = {}; // HITS parameters (e.g., {3, 5})
};

class GASolver {
public:
    GASolver(const Problem& problem, const GAOptions& options);
    Solution solve();
    Solution cohesive_crossover(const Solution& p1, const Solution& p2);
    Solution universal_crossover(const Solution& p1, const Solution& p2);
    // Hierarchical Iterated Tabu Search (skeleton)
    Solution hierarchical_its(const Solution& sol, int level, const std::vector<int>& Q);
private:
    const Problem& problem;
    GAOptions options;
    std::mt19937 rng;
    std::vector<GAPopMember> population;
    Solution tournament_select();
    Solution crossover(const Solution& p1, const Solution& p2);
    void mutate(Solution& sol);
    void local_search(Solution& sol);
    DiffMatrix compute_diff_matrix(const Solution& sol);
};
