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
    Problem problem = Problem::fromQAPLIB(instance_path);
    
    try {
        GilmoreLawler glb_solver;
        GLBResult res = glb_solver.compute(problem.F, problem.D);

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
