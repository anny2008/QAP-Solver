#include "vns_solver.hpp"
#include <algorithm>
#include <chrono>

VNSSolver::VNSSolver(const Problem& problem, const VNSOptions& options)
    : problem(problem), options(options), rng(options.seed) {}

std::vector<int> VNSSolver::shake(const std::vector<int>& perm, int neighborhood) {
    std::vector<int> new_perm = perm;
    int n = perm.size();
    const std::string& nh = options.neighborhoods[neighborhood % options.neighborhoods.size()];
    if (nh == "swap") {
        int i = rng() % n;
        int j = rng() % n;
        std::swap(new_perm[i], new_perm[j]);
    } else if (nh == "reverse") {
        int i = rng() % n;
        int j = rng() % n;
        if (i > j) std::swap(i, j);
        std::reverse(new_perm.begin() + i, new_perm.begin() + j + 1);
    } else if (nh == "insert") {
        int i = rng() % n;
        int j = rng() % n;
        int val = new_perm[i];
        new_perm.erase(new_perm.begin() + i);
        new_perm.insert(new_perm.begin() + j, val);
    }
    return new_perm;
}

Solution VNSSolver::local_search(const Solution& sol, int neighborhood) {
    // Local search: best or first improvement
    Solution best = sol;
    bool improved = true;
    if (options.local_search_strategy == "first_improvement") {
        while (improved) {
            improved = false;
            for (int i = 0; i < problem.n; ++i) {
                for (int j = i + 1; j < problem.n; ++j) {
                    Solution candidate = best;
                    std::swap(candidate.assignment[i], candidate.assignment[j]);
                    candidate.objective = problem.evaluate(candidate.assignment);
                    if (candidate.objective < best.objective) {
                        best = candidate;
                        improved = true;
                        goto next_iter;
                    }
                }
            }
            next_iter:;
        }
    } else { // best_improvement
        while (improved) {
            improved = false;
            Solution best_candidate = best;
            for (int i = 0; i < problem.n; ++i) {
                for (int j = i + 1; j < problem.n; ++j) {
                    Solution candidate = best;
                    std::swap(candidate.assignment[i], candidate.assignment[j]);
                    candidate.objective = problem.evaluate(candidate.assignment);
                    if (candidate.objective < best_candidate.objective) {
                        best_candidate = candidate;
                        improved = true;
                    }
                }
            }
            if (improved) best = best_candidate;
        }
    }
    return best;
}

Solution VNSSolver::solve() {
    // Initial solution
    std::vector<int> perm(problem.n);
    if (options.initial_solution == "random") {
        for (int i = 0; i < problem.n; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
    } else { // identity
        for (int i = 0; i < problem.n; ++i) perm[i] = i;
    }
    Solution current("unknown", "vns");
    current.assignment = perm;
    current.objective = problem.evaluate(current.assignment);
    Solution best = current;
    int k_max = options.k_max;
    auto t_start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < options.max_iters; ++iter) {
        int k = 0;
        while (k < k_max) {
            std::vector<int> shaken = shake(current.assignment, k);
            Solution shaken_sol("unknown", "vns");
            shaken_sol.assignment = shaken;
            shaken_sol.objective = problem.evaluate(shaken_sol.assignment);
            Solution local_opt = local_search(shaken_sol, k);
            if (local_opt.objective < best.objective) {
                best = local_opt;
                current = local_opt;
                k = 0;
            } else {
                ++k;
            }
        }
        // Logging: iteration, elapsed time, best objective
        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        printf("[VNS] Iteration %d, best_objective = %.8g, time = %.3fs\n", iter + 1, best.objective, elapsed);
    }
    return best;
}
