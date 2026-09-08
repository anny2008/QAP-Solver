#include "gilmore_lawler.h"
#include "../../core/problem.h"
#include <iostream>
#include <fstream>
#include <include/json.hpp>
#include <iomanip>


/**
 * Example usage of Gilmore-Lawler bound for QAP.
 * Usage: ./gilmore_lawler_solver <instance.dat> [--config <file>]
 */
int main(int argc, char* argv[]) {
    std::string instance_path = "";
    std::string config_path = "";
    // Parse CLI options
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (instance_path.empty() && arg[0] != '-') {
            instance_path = arg;
        }
    }
    // If config file is provided, read and override instance path
    if (!config_path.empty()) {
        nlohmann::json config;
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            std::cerr << "Cannot open config file: " << config_path << std::endl;
            return 1;
        }
        try {
            config_file >> config;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON config: " << e.what() << std::endl;
            return 1;
        }
        config_file.close();
        if (config.contains("instance") && config["instance"].is_string()) {
            instance_path = config["instance"];
        }
    }
    if (instance_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " <instance.dat> [--config <file>]" << std::endl;
        return 1;
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
    auto problem = Problem::fromAuto(instance_path, auto_fixed_file);
    
    int n = problem.n;
    int m = problem.m;
    std::cout << "Problem size: n = " << n << ", m = " << m << std::endl;
    if (n > m) {
        std::cout << "Note: n > m. Adding " << (n - m) << " dummy facilities with zero flow." << std::endl;
        // add dummy facilities to qap_data.F if needed
        for (auto& row : problem.F) {
            row.resize(n, 0.0);
        }
        problem.F.resize(n, std::vector<double>(n, 0.0));
    }
    
    try {
        GilmoreLawler glb_solver;
        GLBResult res = glb_solver.compute(problem);

        std::cout << "GLB value: " << res.value << std::endl;
        // std::cout << "Assignment (facility -> location):" << std::endl;
        // for (int i = 0; i < (int)res.assignment.size(); ++i) {
        //     std::cout << "  " << i << " -> " << res.assignment[i] << std::endl;
        // }


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
