
#include "vns_solver.hpp"
#include "../../core/problem.h"
#include "../../core/solution.h"
#include "../../core/include/json.hpp"
#include <iostream>
#include <fstream>
#include <string>


int main(int argc, char* argv[]) {
    std::string problem_file;
    std::string config_file;
    VNSOptions options;
    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--max_iters" && i + 1 < argc) {
            options.max_iters = std::stoi(argv[++i]);
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
        std::cerr << "Usage: " << argv[0] << " <problem_file> [--config <config.json>] [--max_iters N] [--seed S]" << std::endl;
        return 1;
    }
    // Load config file if provided
    if (!config_file.empty()) {
        std::ifstream fin(config_file);
        if (fin) {
            nlohmann::json j;
            fin >> j;
            if (j.contains("max_iterations")) options.max_iters = j["max_iterations"].get<int>();
            if (j.contains("k_max")) options.k_max = j["k_max"].get<int>();
            if (j.contains("seed")) {
                if (j["seed"].is_string() && j["seed"].get<std::string>() == "time") {
                    options.seed = static_cast<int>(time(nullptr));
                } else if (j["seed"].is_number_integer()) {
                    options.seed = j["seed"].get<int>();
                }
            }
            if (j.contains("neighborhoods")) options.neighborhoods = j["neighborhoods"].get<std::vector<std::string>>();
            if (j.contains("initial_solution")) options.initial_solution = j["initial_solution"].get<std::string>();
            if (j.contains("local_search_strategy")) options.local_search_strategy = j["local_search_strategy"].get<std::string>();
            if (j.contains("input_file")) problem_file = j["input_file"].get<std::string>();
            if (j.contains("instance")) problem_file = j["instance"].get<std::string>();
        } else {
            std::cerr << "Could not open config file: " << config_file << std::endl;
            return 1;
        }
    }
    Problem problem = Problem::fromQAPLIB(problem_file);
    VNSSolver solver(problem, options);
    Solution best = solver.solve();
        printf("\n==== VNS FINAL RESULT ====\n");
        printf("Best objective: %.0f\n", best.objective);
    std::cout << "Best assignment: ";
    for (int v : best.assignment) std::cout << v << " ";
    std::cout << std::endl;
    return 0;
}
