#include "ColumnGenSolver.h"
#include "../../core/problem.h"

#include <iostream>
#include <fstream>
#include <include/json.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance.dat> [--config file.json]" << std::endl;
        return 1;
    }

    std::string instance_path = argv[1];
    std::string config_path = "";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }
    ColumnGenConfig cfg;

    if (!config_path.empty()) {
        std::ifstream in(config_path);
        json j;
        in >> j;

        if (j.contains("time_limit"))        cfg.time_limit_sec = j["time_limit"];
        if (j.contains("max_iterations"))    cfg.max_iterations = j["max_iterations"];
        if (j.contains("epsilon"))           cfg.eps = j["epsilon"];
        if (j.contains("add_most_negative")) cfg.add_most_negative = j["add_most_negative"];
        if (j.contains("log_output"))        cfg.log_output = j["log_output"];
        if (j.contains("use_bigM"))          cfg.use_bigM = j["use_bigM"];
    } else {
        std::cout << "No config file provided, using default settings.\n";
    }
    std::cout << "Max iterations: " << cfg.max_iterations << "\n";
    std::cout << "Epsilon: " << cfg.eps << "\n";
    std::cout << "Add most negative: " << (cfg.add_most_negative ? "true" : "false") << "\n";
    std::cout << "Log output: " << (cfg.log_output ? "true" : "false") << "\n";

    // If config provided, load it
    if (!config_path.empty()) {
        std::ifstream f(config_path);
        if (!f.is_open()) {
            std::cerr << "Cannot open config file: " << config_path << std::endl;
            return 1;
        }
        json j;
        try {
            f >> j;
            cfg = ColumnGenConfig::fromJson(j);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON config: " << e.what() << std::endl;
            return 1;
        }
    }

    Problem problem = Problem::fromQAPLIB(instance_path);

    ColumnGenSolver solver(cfg);
    Solution solution = solver.solve(problem, instance_path);
    std::cout << "CG-CPLEX finished.\n"
              << "Objective: " << solution.objective << "\n"
              << "Time: "      << solution.time      << " s\n";

    return 0;
}