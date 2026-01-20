
// ----------------------------------------------------------------------
// Genetic Algorithm (GA) for QAP - Misevičius et al. 2024
// This implementation follows the structure and parameterization described in:
//   - Table 1: Control parameters (population size, C, generations, DT, etc.)
//   - Section 2.7.1: Tabu Search Algorithm (HITS/ITS)
//   - Section 2.7.2: Perturbation (Mutation) Process
//   - Section 3/Table 1: Experimental setup and parameter ranges
//
// Key steps:
// 1. Auto-select all parameters as in the paper (or use config if set)
// 2. Create a primordial population (C * population_size), improve with HITS
// 3. Select a diverse initial population using distance threshold DT
// 4. Main GA loop: selection, crossover, mutation, HITS improvement, replacement
// 5. Restart if idle generations exceed L_idle_gen
// 6. All parameters are stored in the options struct for consistency
// ----------------------------------------------------------------------
#include <set>
#include "ga_solver.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>

// Hamming distance between two permutations
int permutation_distance(const std::vector<int> &a, const std::vector<int> &b)
{
    int d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i])
            ++d;
    return d;
}
// Simple candidate acceptance: always accept if better, else accept with small probability
Solution candidate_acceptance(const Solution &new_sol, const Solution &best_sol, std::mt19937 &rng)
{
    if (new_sol.objective < best_sol.objective)
        return new_sol;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (dist(rng) < 0.05)
        return new_sol; // 5% chance to accept worse
    return best_sol;
}

// Simple perturbation: random swap
void perturbation(Solution &sol, std::mt19937 &rng)
{
    int n = sol.assignment.size();
    std::uniform_int_distribution<int> dist(0, n - 1);
    int i = dist(rng), j = dist(rng);
    std::swap(sol.assignment[i], sol.assignment[j]);
}

// Recursive Hierarchical Iterated Tabu Search (HITS)
// --- 1-level HITS (Tabu Search) as per Misevičius et al. 2024 ---
Solution GASolver::hierarchical_its(const Solution &sol, int level, const std::vector<int> &Q)
{
    if (level > 0)
    {
        Solution best = sol;
        Solution current = sol;
        for (int q = 1; q <= Q[level]; ++q)
        {
            Solution improved = hierarchical_its(current, level - 1, Q);
            if (improved.objective < best.objective)
                best = improved;
            if (q < Q[level])
            {
                current = candidate_acceptance(improved, best, rng);
                perturbation(current, rng);
            }
        }
        return best;
    }

    // --- Begin 1-level Tabu Search (TS) ---
    int n = sol.assignment.size();
    int tau = 100 + 0.1 * n * n; // Number of TS iterations (can be parameterized)
    int hts_size = 10007;        // Hash table size (prime number)
    double alpha = 0.01;         // Randomization parameter for tabu criterion
    int gamma = 5;               // Idle iterations factor (can be parameterized)
    int L_idle_iter = gamma * tau;
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    // Tabu list (matrix)
    std::vector<std::vector<int>> T(n, std::vector<int>(n, 0));
    // Hash table for aspiration
    std::vector<bool> HTS(hts_size, false);
    // Secondary memory (archive of good solutions)
    std::vector<Solution> SM;
    Solution best = sol;
    Solution current = sol;
    int iter = 0;
    int idle = 0;
    double z_star = best.objective;
    int last_improve = 0;
    while (iter < tau)
    {
        double best_move_delta = std::numeric_limits<double>::max();
        int best_i = -1, best_j = -1;
        bool found = false;
        // Explore all pairwise swaps
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                Solution candidate = current;
                std::swap(candidate.assignment[i], candidate.assignment[j]);
                candidate.objective = problem.evaluate(candidate.assignment);
                double delta = candidate.objective - current.objective;
                // Tabu criterion (randomized)
                bool tabu = false;
                int q = iter;
                int tij = T[i][j];
                double zeta = urand(rng);
                bool TC = (tij >= q) || (zeta >= alpha);
                // Aspiration criterion
                bool AC = (candidate.objective < z_star);
                // Move acceptance
                bool MC = ((candidate.objective < z_star) && !TC) || AC;
                if (MC && delta < best_move_delta)
                {
                    best_move_delta = delta;
                    best_i = i;
                    best_j = j;
                    found = true;
                }
            }
        }
        if (!found)
            break; // No move found
        // Apply best move
        std::swap(current.assignment[best_i], current.assignment[best_j]);
        current.objective = problem.evaluate(current.assignment);
        // Update tabu list
        int h = std::max(1, int(0.05 * n));
        T[best_i][best_j] = iter + h;
        // Update best
        if (current.objective < best.objective)
        {
            best = current;
            z_star = best.objective;
            last_improve = iter;
            // Archive in SM
            SM.push_back(best);
        }
        // Idle detection
        if (iter - last_improve > L_idle_iter)
        {
            // Reset: wipe tabu list, pick from SM
            for (auto &row : T)
                std::fill(row.begin(), row.end(), 0);
            if (!SM.empty())
            {
                current = SM[rng() % SM.size()];
                current.objective = problem.evaluate(current.assignment);
            }
            last_improve = iter;
        }
        ++iter;
    }
    return best;
}

GASolver::GASolver(const Problem &problem, const GAOptions &options)
    : problem(problem), options(options), rng(options.seed) {}

Solution GASolver::tournament_select()
{
    std::uniform_int_distribution<int> dist(0, options.population_size - 1);
    GAPopMember best = population[dist(rng)];
    for (int i = 1; i < options.tournament_size; ++i)
    {
        GAPopMember challenger = population[dist(rng)];
        if (challenger.obj < best.obj)
            best = challenger;
    }
    return best.sol;
}

// Cohesive crossover as described in the paper
Solution GASolver::cohesive_crossover(const Solution &p1, const Solution &p2)
{
    int n = p1.assignment.size();
    // For QAP, we need the distance matrix. Assume problem.distances is available (n x n)
    std::vector<std::vector<double>> D = problem.D; // n x n
    std::vector<int> child(n, -1);
    std::vector<bool> assigned(n, false);

    // For each site, calculate median distance to all other sites
    std::vector<double> medians(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        std::vector<double> dists;
        for (int j = 0; j < n; ++j)
        {
            if (i != j)
                dists.push_back(D[i][j]);
        }
        std::sort(dists.begin(), dists.end());
        medians[i] = dists[dists.size() / 2];
    }

    // Step 2: Assign facilities from parent1 if closer than median, else from parent2
    for (int i = 0; i < n; ++i)
    {
        double dist_sum = 0.0;
        for (int j = 0; j < n; ++j)
            dist_sum += D[i][j];
        double median = medians[i];
        if (dist_sum < median * n)
        {
            child[i] = p1.assignment[i];
        }
        else
        {
            child[i] = p2.assignment[i];
        }
    }

    // Step 3: Repair - ensure each facility is assigned exactly once
    // Find assigned twice and unassigned
    std::vector<int> count(n, 0);
    for (int i = 0; i < n; ++i)
        count[child[i]]++;
    std::vector<int> unassigned, assigned_twice;
    for (int f = 0; f < n; ++f)
    {
        if (count[f] == 0)
            unassigned.push_back(f);
        if (count[f] > 1)
            assigned_twice.push_back(f);
    }
    // Replace sites assigned to a facility twice with unassigned
    int ua_idx = 0;
    for (int i = 0; i < n && ua_idx < (int)unassigned.size(); ++i)
    {
        if (count[child[i]] > 1)
        {
            child[i] = unassigned[ua_idx++];
            count[child[i]] = 1;
        }
    }

    Solution offspring("unknown", "ga");
    offspring.assignment = child;
    offspring.objective = problem.evaluate(offspring.assignment);
    return offspring;
}

Solution GASolver::crossover(const Solution &p1, const Solution &p2)
{
    if (options.crossover_type == "cohesive") {
        return cohesive_crossover(p1, p2);
    } else {
        return universal_crossover(p1, p2);
    }
}

Solution GASolver::universal_crossover(const Solution &p1, const Solution &p2)
{
    // ----------------------------------------------------------------------
    // Universal Crossover (UNIVX) - Misevičius et al. 2024
    // This operator uses a random mask to select genes from two parents.
    // Parameters:
    //   h    - mask length factor (α = h*n), typically close to 1 (e.g., 0.9)
    //   beta - controls number of 1's in mask (fraction of genes from parent1)
    //   gamma- controls mask arrangement (anytime algorithm, e.g., 0.9)
    //   chi  - starting position for transferred genes (random in [1, n])
    // Steps:
    //   1. Generate a random mask of length n with about beta*n ones
    //   2. Optionally sort and shuffle mask bits for additional randomness
    //   3. Copy genes from parent1 where mask is 1, starting at chi
    //   4. Fill remaining positions from parent2 in order, skipping duplicates
    //   5. Return the offspring
    // See attached images and Section 3.2.2 of the paper for details
    // ----------------------------------------------------------------------
    int n = p1.assignment.size();
    double h = 0.9;            // mask length factor (α = h*n)
    double beta = 4.0 / 9.0;   // controls number of 1's in mask
    double gamma = 0.9;        // controls mask arrangement
    int chi = 1 + (rng() % n); // starting position for transferred genes

    // Step 1: Generate mask
    int mask_len = int(h * n);
    std::vector<int> mask(n, 0);
    int ones = int(beta * n);
    std::vector<int> mask_bits(n, 0);
    for (int i = 0; i < ones; ++i)
        mask_bits[i] = 1;
    std::shuffle(mask_bits.begin(), mask_bits.end(), rng);
    mask = mask_bits;

    // Step 2: Optionally sort and shuffle mask bits (anytime algorithm)
    if (gamma < 1.0)
    {
        int z = int(gamma * n);
        std::sort(mask.begin(), mask.end(), std::greater<int>());
        std::shuffle(mask.begin() + z, mask.end(), rng);
    }

    // Step 3: Copy genes from parent1 where mask is 1
    std::vector<int> child(n, -1);
    std::vector<bool> used(n, false);
    int idx = chi - 1;
    for (int i = 0; i < n; ++i)
    {
        int pos = (idx + i) % n;
        if (mask[pos] == 1)
        {
            child[pos] = p1.assignment[pos];
            used[child[pos]] = true;
        }
    }
    // Step 4: Fill remaining positions from parent2 in order
    for (int i = 0; i < n; ++i)
    {
        int pos = (idx + i) % n;
        if (child[pos] == -1)
        {
            for (int j = 0; j < n; ++j)
            {
                int val = p2.assignment[(pos + j) % n];
                if (!used[val])
                {
                    child[pos] = val;
                    used[val] = true;
                    break;
                }
            }
        }
    }
    // Step 5: Return offspring
    Solution offspring("unknown", "ga");
    offspring.assignment = child;
    offspring.objective = problem.evaluate(offspring.assignment);
    return offspring;
}

void GASolver::mutate(Solution &sol)
{
    int n = sol.assignment.size();
    std::uniform_int_distribution<int> dist(0, n - 1);
    if (options.mutation == "swap")
    {
        int i = dist(rng), j = dist(rng);
        std::swap(sol.assignment[i], sol.assignment[j]);
    }
    else if (options.mutation == "insert")
    {
        int i = dist(rng), j = dist(rng);
        int val = sol.assignment[i];
        sol.assignment.erase(sol.assignment.begin() + i);
        sol.assignment.insert(sol.assignment.begin() + j, val);
    }
    else if (options.mutation == "reverse")
    {
        int i = dist(rng), j = dist(rng);
        if (i > j)
            std::swap(i, j);
        std::reverse(sol.assignment.begin() + i, sol.assignment.begin() + j + 1);
    }
    sol.objective = problem.evaluate(sol.assignment);
}

void GASolver::local_search(Solution &sol)
{
    // Simple best-improvement swap local search
    bool improved = true;
    while (improved)
    {
        improved = false;
        Solution best = sol;
        for (int i = 0; i < problem.n; ++i)
        {
            for (int j = i + 1; j < problem.n; ++j)
            {
                Solution candidate = sol;
                std::swap(candidate.assignment[i], candidate.assignment[j]);
                candidate.objective = problem.evaluate(candidate.assignment);
                if (candidate.objective < best.objective)
                {
                    best = candidate;
                    improved = true;
                }
            }
        }
        if (improved)
            sol = best;
    }
}

Solution GASolver::solve()
{
    int n = problem.n;
    std::uniform_int_distribution<int> dist(0, n - 1);
    // --- All parameters must be set in options before calling solve() ---
    int primordial_size = options.population_size * options.ts_idle_factor;
    int DT = std::max(2, n / 2);
    int L_idle_gen = std::max(3., 0.05 * options.generations);
    std::vector<int> Q = options.Q.empty() ? std::vector<int>{3, 5} : options.Q;

    // --- Primordial population creation ---
    // Step 2: Primordial population creation (C * population_size random solutions)
    // Each solution is improved by a 2-level HITS (Hierarchical Iterated Tabu Search)
    std::vector<GAPopMember> primordial;
    for (int i = 0; i < primordial_size; ++i)
    {
        std::vector<int> perm(n);
        for (int j = 0; j < n; ++j)
            perm[j] = j;
        auto rng = std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count() + i); // Different seed for each
        std::shuffle(perm.begin(), perm.end(), rng);
        Solution sol("unknown", "ga");
        sol.assignment = perm;
        sol.objective = problem.evaluate(sol.assignment);
        // HITS: 2-level (Q = {3, 5} by default, can be parameterized)
        Solution improved = hierarchical_its(sol, Q.size() - 1, Q);
        // auto improved = sol;
        primordial.emplace_back(improved, improved.objective, compute_diff_matrix(improved));
    }

    // Step 3: Select a diverse initial population using distance threshold DT
    // (see Table 1: DT = max{2, [0.5n]})
    std::sort(primordial.begin(), primordial.end(), [](const GAPopMember &a, const GAPopMember &b)
              { return a.obj < b.obj; });
    std::cout << "Primordial population created. Best objective: " << primordial[0].obj << ". Number of solutions: " << primordial.size() << std::endl;
    std::vector<GAPopMember> diverse;
    for (const auto &member : primordial)
    {
        bool far = true;
        for (const auto &d : diverse)
        {
            if (permutation_distance(member.sol.assignment, d.sol.assignment) < DT)
            {
                far = false;
                break;
            }
        }
        if (far || diverse.empty())
            diverse.push_back(member);
        if ((int)diverse.size() >= options.population_size)
            break;
    }
    // If not enough diverse, fill with best (as in the paper)
    while ((int)diverse.size() < options.population_size && !primordial.empty())
    {
        for (const auto &member : primordial)
        {
            if (std::find_if(diverse.begin(), diverse.end(), [&](const GAPopMember &d)
                             { return permutation_distance(member.sol.assignment, d.sol.assignment) < DT; }) == diverse.end())
            {
                diverse.push_back(member);
                if ((int)diverse.size() >= options.population_size)
                    break;
            }
        }
    }
    population = diverse;
    std::cout << "Diverse initial population selected. Size: " << population.size() << std::endl;
    // Step 4: Main GA loop (selection, crossover, mutation, HITS, replacement)
    // - Rank-based parent selection
    // - Always perform crossover and mutation
    // - Improve offspring with HITS
    // - Replace worst or best in population using distance threshold
    // - Restart if idle generations exceed L_idle_gen
    std::sort(population.begin(), population.end(), [](const GAPopMember &a, const GAPopMember &b)
              { return a.obj < b.obj; });
    Solution best = population[0].sol;
    auto t_start = std::chrono::steady_clock::now();
    int idle_gens = 0;
    double prev_best_obj = best.objective;
    int hits_levels = Q.size() - 1;
    std::cout << "Starting GA main loop for " << options.generations << " generations." << std::endl;
    for (int gen = 0; gen < options.generations; ++gen)
    {
        // Sort population by objective value
        std::sort(population.begin(), population.end(), [](const GAPopMember &a, const GAPopMember &b)
                  { return a.obj < b.obj; });
        // Rank-based parent selection (uniform random selection here)
        Solution parent1 = population[dist(rng) % population.size()].sol;
        Solution parent2 = population[dist(rng) % population.size()].sol;
        // Crossover (always)
        Solution child = crossover(parent1, parent2);
        // HITS improvement (recursive)
        Solution improved = hierarchical_its(child, hits_levels, Q);
        // Population replacement with distance threshold
        bool add_child = true;
        for (const auto &member : population)
        {
            if (permutation_distance(improved.assignment, member.sol.assignment) < DT)
            {
                add_child = false;
                break;
            }
        }
        if (add_child || improved.objective < best.objective)
        {
            // Replace worst if not best, else replace best
            if (improved.objective < best.objective)
            {
                best = improved;
                population[0] = GAPopMember(improved, improved.objective, compute_diff_matrix(improved));
            }
            else
            {
                // Remove worst
                auto worst_it = std::max_element(population.begin(), population.end(), [](const GAPopMember &a, const GAPopMember &b)
                                                 { return a.obj < b.obj; });
                if (worst_it != population.end())
                    *worst_it = GAPopMember(improved, improved.objective, compute_diff_matrix(improved));
            }
        }
        // Idle generation detection
        if (best.objective < prev_best_obj)
        {
            idle_gens = 0;
            prev_best_obj = best.objective;
        }
        else
        {
            idle_gens++;
        }
        // Step 5: Restart if idle generations exceed L_idle_gen
        if (idle_gens > L_idle_gen)
        {
            // Recreate primordial population
            primordial.clear();
            for (int i = 0; i < primordial_size; ++i)
            {
                std::vector<int> perm(n);
                for (int j = 0; j < n; ++j)
                    perm[j] = j;
                std::shuffle(perm.begin(), perm.end(), rng);
                Solution sol("unknown", "ga");
                sol.assignment = perm;
                sol.objective = problem.evaluate(sol.assignment);
                Solution improved = hierarchical_its(sol, Q.size() - 1, Q);
                primordial.emplace_back(improved, improved.objective, compute_diff_matrix(improved));
            }
            // Truncate to PS best and diverse
            std::sort(primordial.begin(), primordial.end(), [](const GAPopMember &a, const GAPopMember &b)
                      { return a.obj < b.obj; });
            diverse.clear();
            for (const auto &member : primordial)
            {
                bool far = true;
                for (const auto &d : diverse)
                {
                    if (permutation_distance(member.sol.assignment, d.sol.assignment) < DT)
                    {
                        far = false;
                        break;
                    }
                }
                if (far || diverse.empty())
                    diverse.push_back(member);
                if ((int)diverse.size() >= options.population_size)
                    break;
            }
            while ((int)diverse.size() < options.population_size && !primordial.empty())
            {
                for (const auto &member : primordial)
                {
                    if (std::find_if(diverse.begin(), diverse.end(), [&](const GAPopMember &d)
                                     { return permutation_distance(member.sol.assignment, d.sol.assignment) < DT; }) == diverse.end())
                    {
                        diverse.push_back(member);
                        if ((int)diverse.size() >= options.population_size)
                            break;
                    }
                }
            }
            population = diverse;
            std::sort(population.begin(), population.end(), [](const GAPopMember &a, const GAPopMember &b)
                      { return a.obj < b.obj; });
            best = population[0].sol;
            idle_gens = 0;
        }
        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        printf("[GA] Generation %d, best_objective = %.0f, time = %.3fs\n", gen + 1, best.objective, elapsed);
    }
    // Step 6: Return the best solution found
    return best;
}

// Dummy difference matrix computation (to be replaced with actual delta evaluation)
DiffMatrix GASolver::compute_diff_matrix(const Solution &sol)
{
    int n = sol.assignment.size();
    return DiffMatrix(n, std::vector<double>(n, 0.0));
}
