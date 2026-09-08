#include <fstream>
#include <random>
#ifndef QAP_LOCAL_SEARCH_H
#define QAP_LOCAL_SEARCH_H

#include "../../core/problem.h"
#include "../../core/solution.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>

/**
 * Implements local search for QAP.
 */
class LocalSearch {
public:
    /**
     * Iterated Tabu Search for QAP (as in literature, e.g., Misevičius 2008).
     * Repeatedly applies tabu search, perturbs the solution, and tracks the best found.
     * @param problem QAP instance
     * @param solution Initial solution (will be improved in-place)
     * @param outer_iters Number of outer iterations (tabu search runs)
     * @param tabu_iters Number of iterations for each tabu search
     * @param tabu_tenure Initial tabu tenure
     * @param perturb_strength Number of random swaps per perturbation
     * @param verbose If true, logs progress
     */
    static void iteratedTabuSearch(const Problem& problem, Solution& solution, int outer_iters = 10, int tabu_iters = 1000, int tabu_tenure = 10, int perturb_strength = 2, bool verbose = true) {
        int n = problem.n;
        std::vector<int> best_perm = solution.assignment;
        double best_obj = computeObjective(problem, best_perm);
        std::random_device rd;
        std::mt19937 gen(rd());
        auto its_start = std::chrono::high_resolution_clock::now();
        for (int its_iter = 1; its_iter <= outer_iters; ++its_iter) {
            Solution current = solution;
            tabuSearch(problem, current, tabu_iters, tabu_tenure, verbose);
            double current_obj = current.objective;
            if (current_obj < best_obj) {
                best_obj = current_obj;
                best_perm = current.assignment;
            }
            // Log with elapsed time
            if (verbose) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - its_start).count();
                std::cout << "[ITS] Iteration " << its_iter << ", objective = " << current_obj << ", time = " << elapsed << "s" << std::endl;
            }
            // Perturbation: random swaps
            for (int k = 0; k < perturb_strength; ++k) {
                int i = gen() % n;
                int j = gen() % n;
                if (i != j) std::swap(current.assignment[i], current.assignment[j]);
            }
            solution.assignment = current.assignment;
        }
        solution.assignment = best_perm;
        solution.objective = best_obj;
    }
    
    /**
     * Robust Tabu Search for QAP (swap-based neighborhood), following Taillard (1991).
     * Key features:
     *  - Adaptive tabu tenure: tabu list length is periodically adjusted to diversify search.
     *  - Aspiration criterion: tabu moves allowed if they yield a new best solution.
     *  - (To be added) Frequency-based memory and intensification/diversification.
     *
     * @param problem QAP instance
     * @param solution Initial solution (will be improved in-place)
     * @param max_iters Number of iterations
     * @param tabu_tenure Initial tabu tenure (will be adapted)
     * @param verbose If true, logs progress
     */
    static void tabuSearch(const Problem& problem, Solution& solution, int max_iters = 1000, int tabu_tenure = 10, bool verbose = true) {
        int n = problem.n;
        std::vector<int> perm = solution.assignment;
        double best_obj = computeObjective(problem, perm);
        std::vector<int> best_perm = perm;
        double current_obj = best_obj;
        std::vector<std::vector<int>> tabu(n, std::vector<int>(n, 0));
        int iter = 0;
        int it_since_best = 0;
        std::random_device rd;
        std::mt19937 gen(rd());
        auto start = std::chrono::high_resolution_clock::now();

        // --- Adaptive tabu tenure parameters (Taillard 1991) ---
        int min_tenure = std::max(5, n / 10); // lower bound for tabu tenure
        int max_tenure = std::max(10, n / 2); // upper bound for tabu tenure
        int adaptive_period = 100; // how often to adapt tenure
        int adaptive_counter = 0;
        int adaptive_direction = 1; // +1: increasing, -1: decreasing
        int adaptive_tenure = tabu_tenure;

        // --- Frequency-based memory (Taillard 1991) ---
        // freq[i][j] counts how many times swap (i,j) has been performed
        std::vector<std::vector<int>> freq(n, std::vector<int>(n, 0));
        double freq_penalty_weight = 0.1; // weight for penalizing frequent moves (can be tuned)

        // --- Assignment frequency memory for diversification (Taillard 1991) ---
        // assign_freq[i][k]: how many times facility i assigned to location k
        std::vector<std::vector<int>> assign_freq(n, std::vector<int>(n, 0));
        double assign_penalty_weight = 0.05; // weight for penalizing frequent assignments

        // Main search loop
        while (iter < max_iters) {
            ++iter;
            ++adaptive_counter;
            int best_i = -1, best_j = -1;
            double best_move_score = std::numeric_limits<double>::max();
            bool move_is_tabu = false;

            // Explore all swaps (neighborhood)
            for (int i = 0; i < n - 1; ++i) {
                // if i or j are fixed assignments, skip
                if (problem.fixed_assignments.count(i)) continue;
                for (int j = i + 1; j < n; ++j) {
                    // if i or j are fixed assignments, skip
                    if (problem.fixed_assignments.count(j)) continue;
                    std::swap(perm[i], perm[j]);
                    double obj = computeObjective(problem, perm);
                    // Frequency penalty: penalize moves that have been used often
                    double freq_penalty = freq_penalty_weight * freq[i][j];
                    // Assignment penalty: penalize assignments that are too frequent (diversification)
                    double assign_penalty = 0.0;
                    for (int k = 0; k < n; ++k) {
                        assign_penalty += assign_penalty_weight * assign_freq[k][perm[k]];
                    }
                    double move_score = obj + freq_penalty + assign_penalty;
                    bool is_tabu = tabu[i][j] > iter;
                    // Aspiration: allow tabu if improves best (ignore penalty for aspiration)
                    if ((is_tabu && obj < best_obj) || (!is_tabu && move_score < best_move_score)) {
                        best_move_score = move_score;
                        best_i = i;
                        best_j = j;
                        move_is_tabu = is_tabu;
                    }
                    std::swap(perm[i], perm[j]); // revert
                }
            }
            if (best_i == -1 || best_j == -1) break; // No move found

            // Apply best move
            std::swap(perm[best_i], perm[best_j]);
            current_obj = computeObjective(problem, perm);
            // Update frequency memory
            freq[best_i][best_j] += 1;
            // Update assignment frequency memory
            for (int k = 0; k < n; ++k) {
                assign_freq[k][perm[k]] += 1;
            }
            // Set tabu tenure for this move (adaptive)
            tabu[best_i][best_j] = iter + adaptive_tenure + (gen() % 5); // randomize slightly

            // Update best solution
            if (current_obj < best_obj) {
                best_obj = current_obj;
                best_perm = perm;
                it_since_best = 0;
            } else {
                ++it_since_best;
            }

            // Logging
            if (verbose) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start).count();
                std::cout << "[Tabu] Iteration " << iter << ", objective = " << current_obj << ", best = " << best_obj << ", tenure = " << adaptive_tenure << ", freq_penalty = " << freq[best_i][best_j] << ", time = " << elapsed << "s" << std::endl;
            }

            // --- Adaptive tabu tenure update (periodic) ---
            if (adaptive_counter >= adaptive_period) {
                adaptive_counter = 0;
                adaptive_tenure += adaptive_direction;
                // Reverse direction if bounds reached
                if (adaptive_tenure >= max_tenure) adaptive_direction = -1;
                if (adaptive_tenure <= min_tenure) adaptive_direction = 1;
            }

            // --- Intensification: restart from best solution if stuck (as in Taillard 1991) ---
            if (it_since_best > n * 5) {
                if (verbose) std::cout << "[Tabu] Intensification: Restarting from best solution so far." << std::endl;
                perm = best_perm;
                it_since_best = 0;
            }

            // --- Diversification: random restart if too many iterations without improvement ---
            // If no improvement for a long time, restart from a random solution
            if (it_since_best > n * 10) {
                if (verbose) std::cout << "[Tabu] Diversification: Random restart triggered." << std::endl;
                // Generate a random permutation
                for (int k = 0; k < n; ++k) perm[k] = k;
                std::shuffle(perm.begin(), perm.end(), gen);
                it_since_best = 0;
            }
        }
        // Write back best solution found
        solution.assignment = best_perm;
        solution.objective = best_obj;
    }
    /**
     * Iterated local search with 2-opt swap operation.
     * @param problem QAP instance
     * @param solution Initial solution (will be improved in-place)
     * @param max_iters Number of ILS iterations
     * @param perturb_strength Number of random swaps per perturbation
     */
    static void iteratedLocalSearch(const Problem& problem, Solution& solution, int max_iters = 100, int perturb_strength = 2) {
        int n = problem.n;
        std::vector<int> best_perm = solution.assignment;
        double best_obj = computeObjective(problem, best_perm);
        std::random_device rd;
        std::mt19937 gen(rd());
        auto ils_start = std::chrono::high_resolution_clock::now();
        for (int iter = 1; iter <= max_iters; ++iter) {
            // Local search (2-opt)
            Solution current = solution;
            improve(problem, current);
            double current_obj = current.objective;
            if (current_obj < best_obj) {
                best_obj = current_obj;
                best_perm = current.assignment;
            }
            // Log with elapsed time
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - ils_start).count();
            std::cout << "[ILS] Iteration " << iter << ", objective = " << current_obj << ", time = " << elapsed << "s" << std::endl;
            // Perturbation: random swaps
            for (int k = 0; k < perturb_strength; ++k) {
                int i = gen() % n;
                int j = gen() % n;
                if (i != j) std::swap(current.assignment[i], current.assignment[j]);
            }
            solution.assignment = current.assignment;
        }
        solution.assignment = best_perm;
        solution.objective = best_obj;
    }
    /**
     * Initialize solution assignment.
     * @param n Problem size
     * @param method "identity", "random", or "file"
     * @param sln_path Path to .sln file (if method == "file")
     * @return assignment vector
     */
    static std::vector<int> initAssignment(int n, const std::string& method, const std::string& sln_path = "") {
        std::vector<int> assignment(n);
        if (method == "identity") {
            for (int i = 0; i < n; ++i) assignment[i] = i;
        } else if (method == "random") {
            for (int i = 0; i < n; ++i) assignment[i] = i;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(assignment.begin(), assignment.end(), g);
        } else if (method == "file" && !sln_path.empty()) {
            std::ifstream fin(sln_path);
            if (!fin.is_open()) throw std::runtime_error("Cannot open solution file: " + sln_path);
            for (int i = 0; i < n; ++i) fin >> assignment[i];
            fin.close();
        } else {
            throw std::invalid_argument("Unknown initialization method: " + method);
        }
        return assignment;
    }
    /**
     * Perform a simple 2-opt local search on a QAP solution.
     *
     * @param problem QAP instance
     * @param solution Initial solution (will be improved in-place)
     * @return true if improved, false otherwise
     */
    static bool improve(const Problem& problem, Solution& solution) {
        bool improved = true;
        int n = problem.n;
        auto& perm = solution.assignment;
        double best_obj = computeObjective(problem, perm);
        int iter = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while(improved) {
            improved = false;
            ++iter;
            for (int i = 0; i < n - 1; ++i) {
                // if i or j are fixed assignments, skip
                if (problem.fixed_assignments.count(i)) continue;
                for (int j = i + 1; j < n; ++j) {
                    // if i or j are fixed assignments, skip
                    if (problem.fixed_assignments.count(j)) continue;
                    std::swap(perm[i], perm[j]);
                    double obj = computeObjective(problem, perm);
                    if (obj < best_obj) {
                        best_obj = obj;
                        solution.objective = obj;
                        improved = true;
                    } else {
                        std::swap(perm[i], perm[j]); // revert
                    }
                }
            }
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            std::cout << "\t[2-opt] Iteration " << iter << ", objective = " << best_obj << ", time = " << elapsed << "s" << std::endl;
        }
        return best_obj < solution.objective;
    }

    /**
     * Compute QAP objective value for a permutation.
     */
    static double computeObjective(const Problem& problem, const std::vector<int>& perm) {
        return problem.evaluate(perm);
    }
};

#endif // QAP_LOCAL_SEARCH_H
