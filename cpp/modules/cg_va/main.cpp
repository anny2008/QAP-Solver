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

    // Try to detect fixed assignments file
    // Pattern: if instance is "foo.raw", try "foo_assignments.txt" or "foo.fixed"
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
    auto qap_data = Problem::fromAuto(instance_path, auto_fixed_file);

    int n = qap_data.n;
    int m = qap_data.m;
    std::cout << "Problem size: n = " << n << ", m = " << m << std::endl;
    if (n > m) {
        std::cout << "Note: n > m. Adding " << (n - m) << " dummy facilities with zero flow." << std::endl;
        // add dummy facilities to qap_data.F if needed
        for (auto& row : qap_data.F) {
            row.resize(n, 0.0);
        }
        qap_data.F.resize(n, std::vector<double>(n, 0.0));
    }
    ColumnGenSolver solver(cfg);
    Solution solution = solver.solve(qap_data, instance_path);
    std::cout << "CG-VA finished.\n"
              << "Objective: " << solution.objective << "\n"
              << "Time: "      << solution.time      << " s\n";

    return 0;
}