#include "ga_solver.hpp"
#include "../../core/problem.h"
#include "../../core/solution.h"
#include "../../core/include/json.hpp"
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    std::string problem_file;
    std::string config_file;
    GAOptions options;
    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            std::string seed_arg = argv[++i];
            if (seed_arg == "time") {
                options.seed = static_cast<int>(time(nullptr));
            } else {
                options.seed = std::stoi(seed_arg);
            }
        } else if (arg[0] != '-') {
            problem_file = arg;
        }
    }
    if (problem_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " <problem_file> [--config <config.json>] [--seed S]" << std::endl;
        return 1;
    }
    // Load config file if provided
    if (!config_file.empty()) {
        std::ifstream fin(config_file);
        if (fin) {
            nlohmann::json j;
            fin >> j;
            // --- Support 'auto' for all parameters ---
            int n = 0;
            if (j.contains("input_file")) {
                problem_file = j["input_file"].get<std::string>();
                std::ifstream pf(problem_file);
                if (pf) {
                    std::string line;
                    while (std::getline(pf, line)) {
                        if (line.find("DIM") != std::string::npos) {
                            size_t pos = line.find("DIM");
                            n = std::stoi(line.substr(pos + 3));
                            break;
                        }
                    }
                }
            }
            // Population size
            if (j.contains("population_size")) {
                if (j["population_size"].is_string() && j["population_size"].get<std::string>() == "auto")
                    options.population_size = std::min(40, std::max(2, n));
                else
                    options.population_size = j["population_size"].get<int>();
            }
            // Generations
            if (j.contains("generations")) {
                if (j["generations"].is_string() && j["generations"].get<std::string>() == "auto")
                    options.generations = std::min(100, std::max(1, n));
                else
                    options.generations = j["generations"].get<int>();
            }
            // Crossover rate
            if (j.contains("crossover_rate")) {
                if (j["crossover_rate"].is_string() && j["crossover_rate"].get<std::string>() == "auto")
                    options.crossover_rate = 0.5;
                else
                    options.crossover_rate = j["crossover_rate"].get<double>();
            }
            // Mutation rate
            if (j.contains("mutation_rate")) {
                if (j["mutation_rate"].is_string() && j["mutation_rate"].get<std::string>() == "auto")
                    options.mutation_rate = 0.5;
                else
                    options.mutation_rate = j["mutation_rate"].get<double>();
            }
            // Tournament size
            if (j.contains("tournament_size")) {
                if (j["tournament_size"].is_string() && j["tournament_size"].get<std::string>() == "auto")
                    options.tournament_size = std::min(500, std::max(1, 100 + int(0.1 * n * n)));
                else
                    options.tournament_size = j["tournament_size"].get<int>();
            }
            // Seed
            if (j.contains("seed")) {
                if (j["seed"].is_string()) {
                    std::string sval = j["seed"].get<std::string>();
                    if (sval == "time") options.seed = static_cast<int>(time(nullptr));
                    else if (sval == "auto") options.seed = std::max(int(0.1 * n), 1);
                } else if (j["seed"].is_number_integer()) {
                    options.seed = j["seed"].get<int>();
                }
            }
            // Crossover type ("cohesive" or "universal")
            if (j.contains("crossover_type")) options.crossover_type = j["crossover_type"].get<std::string>();
            // Mutation type
            if (j.contains("mutation")) options.mutation = j["mutation"].get<std::string>();
            // Use local search
            if (j.contains("use_local_search")) options.use_local_search = j["use_local_search"].get<bool>();
            // Elitism
            if (j.contains("elitism")) options.elitism = j["elitism"].get<bool>();
            // Instance
            if (j.contains("instance")) problem_file = j["instance"].get<std::string>();
        } else {
            std::cerr << "Could not open config file: " << config_file << std::endl;
            return 1;
        }
    }
    Problem problem = Problem::fromQAPLIB(problem_file);
    GASolver solver(problem, options);
    Solution best = solver.solve();
    printf("\n==== GA FINAL RESULT ====\n");
    printf("Best objective: %.0f\n", best.objective);
    std::cout << "Best assignment: ";
    for (int v : best.assignment) std::cout << v << " ";
    std::cout << std::endl;
    return 0;
}
