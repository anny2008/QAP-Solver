#include "SFDGurobi.h"
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
    SFDGurobiConfig cfg;

    if (!config_path.empty()) {
        std::ifstream in(config_path);
        json j;
        in >> j;

        if (j.contains("log_output"))        cfg.log_output = j["log_output"];
        if (j.contains("is_relax"))         cfg.is_relax = j["is_relax"];
        if (j.contains("use_lazy_constraints")) cfg.use_lazy_constraints = j["use_lazy_constraints"];
        if (j.contains("use_cuts"))         cfg.use_cuts = j["use_cuts"];
        if (j.contains("lpmethod"))         cfg.lpmethod = j["lpmethod"];
        if (j.contains("warmstart"))        cfg.warmstart = j["warmstart"];
        if (j.contains("matrix_for_decomposition")) cfg.matrix_for_decomposition = j["matrix_for_decomposition"];
        if (j.contains("decomposition"))      cfg.decomposition = j["decomposition"];
    
    } else {
        std::cout << "No config file provided, using default settings.\n";
    }
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
            cfg = SFDGurobiConfig::fromJson(j);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON config: " << e.what() << std::endl;
            return 1;
        }
    }
    
    std::string auto_fixed_file;
    std::string base_name = instance_path;
    size_t dot_pos = base_name.find_last_of(".");
    if (dot_pos != std::string::npos) {
        std::string name_without_ext = base_name.substr(0, dot_pos);
        // Try several naming conventions
        std::vector<std::string> candidates = {
            name_without_ext + "_assignments.txt",
            name_without_ext + ".fixed",
            name_without_ext + "_fixed.txt"
        };
        for (const auto& candidate : candidates) {
            std::ifstream test(candidate);
            if (test.good()) {
                auto_fixed_file = candidate;
                std::cout << "Auto-detected fixed assignments file: " << auto_fixed_file << std::endl;
                break;
            }
        }
    }
    
    // Read QAP instance (with auto-detected fixed assignments if found)
    auto problem = Problem::fromAuto(instance_path, auto_fixed_file);

    SFDGurobiSolver solver(cfg);
    Solution solution = solver.solve(problem, instance_path);
    std::cout << "SFD-Gurobi finished.\n"
              << "Objective: " << solution.objective << "\n"
              << "Time: "      << solution.time      << " s\n";

    return 0;
}