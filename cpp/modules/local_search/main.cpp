#include "local_search.h"
#include "../../core/problem.h"
#include "../../core/solution.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <include/json.hpp>

/**
 * Example usage of LocalSearch for QAP.
 */
int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance.dat> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --config <file>         JSON config file" << std::endl;
        std::cerr << "  --initial-solution <m>  Initialization method (identity, random, file)" << std::endl;
        std::cerr << "  --sln-file <file>       Solution file for initialization" << std::endl;
        std::cerr << "  --output <file>         Output file (not implemented)" << std::endl;
        return 1;
    }

    std::string instance_path = argv[1];
    std::string config_path = "";
    std::string init_method = "identity";
    std::string sln_path = "";
    std::string output_path = "";
    int max_iterations = 100;
    int perturb_strength = 2;
    int tabu_tenure = 10;
    std::string method = "2opt";

    // Parse CLI options
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--initial-solution" && i + 1 < argc) {
            init_method = argv[++i];
        } else if (arg == "--sln-file" && i + 1 < argc) {
            sln_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--tabu-tenure" && i + 1 < argc) {
            tabu_tenure = std::stoi(argv[++i]);
        }
    }

    // If config file is provided, read and override options
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
        if (config.contains("initial_solution") && config["initial_solution"].is_string()) {
            init_method = config["initial_solution"];
        }
        if (config.contains("sln_file") && config["sln_file"].is_string()) {
            sln_path = config["sln_file"];
        }
        if (config.contains("output") && config["output"].is_string()) {
            output_path = config["output"];
        }
        if (config.contains("max_iterations") && config["max_iterations"].is_number_integer()) {
            max_iterations = config["max_iterations"];
        }
        if (config.contains("perturb_strength") && config["perturb_strength"].is_number_integer()) {
            perturb_strength = config["perturb_strength"];
        }
        if (config.contains("tabu_tenure") && config["tabu_tenure"].is_number_integer()) {
            tabu_tenure = config["tabu_tenure"];
        }
        if (config.contains("method") && config["method"].is_string()) {
            method = config["method"];
        }
    }

    // Always print method for debug
    std::cout << "Method selected: " << method << std::endl;
    Problem problem = Problem::fromQAPLIB(instance_path);
    Solution solution(instance_path, "local_search");
    solution.assignment = LocalSearch::initAssignment(problem.n, init_method, sln_path);
    solution.objective = LocalSearch::computeObjective(problem, solution.assignment);
    std::cout << "Initial objective: " << solution.objective << std::endl;
    if (method == "ils") {
        LocalSearch::iteratedLocalSearch(problem, solution, max_iterations, perturb_strength);
        std::cout << "ILS final objective: " << solution.objective << std::endl;
    } else if (method == "tabu") {
        LocalSearch::tabuSearch(problem, solution, max_iterations, tabu_tenure, true);
        std::cout << "Tabu final objective: " << solution.objective << std::endl;
    } else {
        bool improved = LocalSearch::improve(problem, solution);
        std::cout << "Improved: " << (improved ? "yes" : "no") << std::endl;
        std::cout << "Final objective: " << solution.objective << std::endl;
    }

    // Debug: print config values after parsing
    std::cout << "Config/CLI values:" << std::endl;
    std::cout << "  method: " << method << std::endl;
    std::cout << "  max_iterations: " << max_iterations << std::endl;
    std::cout << "  perturb_strength: " << perturb_strength << std::endl;
    std::cout << "  initial_solution: " << init_method << std::endl;
    std::cout << "  sln_file: " << sln_path << std::endl;
    std::cout << "  tabu_tenure: " << tabu_tenure << std::endl;
    std::cout << "  output: " << output_path << std::endl;

    // Optionally write output file here
    return 0;
}
