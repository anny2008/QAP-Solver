#include "local_search.h"
#include "../../core/problem.h"
#include "../../core/solution.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include "../../core/include/json.hpp"

// define solution that is easy to swap and compute objective
struct LocalSearchSolution {
    std::vector<int> assignment_i_u; // facility u -> location i assignment_i_u[i] = u
    std::vector<int> assignment_u_i; // facility u -> location i assignment_u_i[u] = i
    double objective;
    int n;
    int m;

    LocalSearchSolution(int n, int m) : n(n), m(m), assignment_i_u(n, -1), assignment_u_i(m, -1), objective(std::numeric_limits<double>::max()) {}

    bool assign(int u, int i) {
        // if location i is already assigned to some facility return false
        if (assignment_i_u[i] != -1) return false;
        // if facility u is already assigned to some location return false
        if (assignment_u_i[u] != -1) return false;
        // assign facility u to location i
        assignment_i_u[i] = u;
        assignment_u_i[u] = i;
        return true;
    }

    void swap(int i1, int i2) {
        int u1 = assignment_i_u[i1];
        int u2 = assignment_i_u[i2];
        std::swap(assignment_i_u[i1], assignment_i_u[i2]);
        if (u1 != -1) assignment_u_i[u1] = i2;
        if (u2 != -1) assignment_u_i[u2] = i1;
    }

    void set_assignment_i_u(const std::vector<int>& perm) {
        for (int i = 0; i < n; ++i) {
            int u = perm[i];
            assignment_u_i[u] = i;
            assignment_i_u[i] = u;
        }
    }

    void set_assignment_u_i(const std::vector<int>& perm) {
        for (int u = 0; u < m; ++u) {
            int i = perm[u];
            assignment_i_u[i] = u;
            assignment_u_i[u] = i;
        }
    }
};

struct ObjectiveComputer {
    Problem problem;
    std::vector<std::pair<int, int>> positive_flows;
    

    virtual double compute(std::vector<int> assignment_u_i) {
        double obj = 0.0;
        for (auto pair : positive_flows) {
            int u = pair.first;
            int v = pair.second;
            int i = assignment_u_i[u];
            int j = assignment_u_i[v];
            obj += problem.F[u][v] * problem.D[i][j];
        }
        return obj;
    }
};

struct ObjectiveComputerWithXBar: public ObjectiveComputer {
    std::vector<std::vector<double>> x_bar;

    double compute(std::vector<int> assignment_u_i) {
        double obj = 0.0;
        for (auto pair : positive_flows) {
            int u = pair.first;
            int v = pair.second;
            int i = assignment_u_i[u];
            int j = assignment_u_i[v];
            obj += problem.F[u][v] * problem.D[i][j] *(1 - x_bar[i][u]*x_bar[j][v]);
            // obj += problem.F[u][v] * problem.D[i][j] *(1 - x_bar[i][u])*(1 - x_bar[j][v]);
            // obj += problem.F[u][v] * problem.D[i][j];
        }
        return obj;
    }
};

struct ObjectiveComputerWithYBar: public ObjectiveComputer {
    std::vector<std::vector<std::vector<std::vector<double>>>> y_bar;

    double compute(std::vector<int> assignment_u_i) {
        double obj = 0.0;
        for (auto pair : positive_flows) {
            int u = pair.first;
            int v = pair.second;
            int i = assignment_u_i[u];
            int j = assignment_u_i[v];
            obj += problem.F[u][v] * problem.D[i][j] *(1 - y_bar[i][u][j][v]);
        }
        return obj;
    }
};


// pass computeObjective as a parameter to avoid code duplication and allow different objective computation methods
void two_opt(const Problem& problem, LocalSearchSolution& solution, const std::vector<std::pair<int, int>>& positive_flows, ObjectiveComputer* computeObjective) {
    bool improved = true;
    int n = problem.n;
    double best_obj = computeObjective->compute(solution.assignment_u_i);
    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while(improved) {
        improved = false;
        std::cout << "2-opt iteration " << iter << ", current objective = " << best_obj << std::endl;
        ++iter;
        for (int i = 0; i < n - 1; ++i) {
            //  if i is fixed, skip
            if (problem.fixed_assignments.find(i) != problem.fixed_assignments.end()) continue;
            for (int j = i + 1; j < n; ++j) {
                if (problem.fixed_assignments.find(j) != problem.fixed_assignments.end()) continue;
                solution.swap(i, j);
                double obj = computeObjective->compute(solution.assignment_u_i);
                if (obj < best_obj) {
                    best_obj = obj;
                    solution.objective = obj;
                    std::cout << "\tImproved by swapping locations " << i << " and " << j << ": new objective = " << best_obj << std::endl;
                    improved = true;
                } else {
                    solution.swap(i, j); // revert
                }
            }
        }
    }
}

bool isFeasible(const Problem& problem, const LocalSearchSolution& solution) {
    bool feasible = true;
    for (const auto& kv : problem.fixed_assignments) {
        int loc = kv.first;
        int facility = kv.second;
        if (solution.assignment_i_u[loc] != facility) {
            std::cerr << "Location " << loc << " is fixed to facility " << facility << " but assigned to facility " << solution.assignment_i_u[loc] << "." << std::endl;
            feasible = false;
        }
    }

    for (int u = 0; u < problem.m; ++u) {
        int assigned_loc = solution.assignment_u_i[u];
        if (assigned_loc == -1) {
            std::cerr << "Facility " << u << " is not assigned to any location." << std::endl;
            feasible = false;
        }
    }
    for (int i = 0; i < problem.n; ++i) {
        int assigned_facility = solution.assignment_i_u[i];
        if (assigned_facility == -1) {
            continue;
        }
        int i2 = solution.assignment_u_i[assigned_facility];
        if (i2 != i) {
            std::cerr << "Location " << i << " is assigned to facility " << assigned_facility << " but that facility is assigned to location " << i2 << "." << std::endl;
            feasible = false;
        }
    }
    return feasible;
}


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
    int ils_iterations = 10; // for iterated tabu
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
        } else if (arg == "--ils-iterations" && i + 1 < argc) {
            ils_iterations = std::stoi(argv[++i]);
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
        if (config.contains("ils_iterations") && config["ils_iterations"].is_number_integer()) {
            ils_iterations = config["ils_iterations"];
        }
        if (config.contains("method") && config["method"].is_string()) {
            method = config["method"];
        }
    }

    // Always print method for debug
    std::cout << "Method selected: " << method << std::endl;
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
    
    // Load fixed variables if specified
    FixedVariables fv;
    bool has_fixed = false;
    
    // First, load fixed assignments from the problem (if any)
    if (!problem.fixed_assignments.empty()) {
        std::cout << "Loading fixed assignments from problem..." << std::endl;
        for (const auto& kv : problem.fixed_assignments) {
            int loc = kv.first;
            int facility = kv.second;
            fv.x_fixed_1[loc] = facility;
        }
        has_fixed = true;
        std::cout << "Loaded " << problem.fixed_assignments.size() << " fixed assignments from problem." << std::endl;
    }
    // collect positive flows for quick objective computation
    std::vector<std::pair<int, int>> positive_flows;
    for (int u = 0; u < problem.m; ++u) {
        for (int v = 0; v < problem.m; ++v) {
            if (problem.F[u][v] > 0) {
                positive_flows.emplace_back(u, v);
                std::cout << "Positive flow: facility " << u << " -> facility " << v << " with flow " << problem.F[u][v] << std::endl;
            }
        }
    }
    std::cout << "Collected " << positive_flows.size() << " positive flow pairs for objective computation." << std::endl;
    
    auto objectiveComputer = ObjectiveComputer();
    objectiveComputer.problem = problem;
    objectiveComputer.positive_flows = positive_flows;
    // create initial solution
    LocalSearchSolution initial_assignment(problem.n, problem.m);
    std::cout << "Initializing solution : " << problem.n << " x " << problem.m << std::endl;
    // identical initialization
    // for (int u = 0; u < problem.m; ++u) {
    //     initial_assignment.assign(u, u);
    //     std::cout << "Initial assignment: facility " << u << " -> location " << u << std::endl;
    // }
    // load initial solution from file
    {
        // std::string init_sln_path = "/home/local.isima.fr/antran/UFF/QAP-Solver/data/M/instance_318_LEFT.sln";
        std::string init_sln_path = "/home/local.isima.fr/antran/UFF/QAP-Solver/data/M/instance_318_butterfly_LEFT.sln";
        std::ifstream fin(init_sln_path);
        if (!fin.is_open()) {
            std::cerr << "Cannot open solution file: " << init_sln_path << std::endl;
            return 1;
        }
        std::vector<int> perm;
        // n obj\n
        // i1 i2 i3 ...
        int n;
        double obj;
        fin >> n >> obj;
        std::cout << "Initializing solution from file: " << init_sln_path << std::endl;
        std::cout << "File header: n = " << n << ", objective = " << obj << std::endl;
        int kkk;
        int ind = 0;
        while (fin >> kkk) {
            // file format: i1 i2 i3 ... where entry u is the location assigned to facility u
            perm.push_back(kkk);
            ++ind;
        }
        // set internal assignment vectors from permutation if sizes match
        if ((int)perm.size() == problem.m) {
            initial_assignment.set_assignment_u_i(perm);
        } else {
            std::cerr << "Initial solution file has unexpected size: " << perm.size() << " (expected " << problem.m << ")" << std::endl;
            return 1;
        }
        fin.close();
    }
    if(0)
    {
        // initial_assignment.set_assignment_u_i(perm);
        if(isFeasible(problem, initial_assignment)) {
            std::cout << "Initial solution is feasible." << std::endl;
        } else {
            std::cerr << "Initial solution is not feasible!" << std::endl;
            return 1;
        }
        // compute objective for initial solution
        initial_assignment.objective = objectiveComputer.compute(initial_assignment.assignment_u_i);
        std::cout << "Initial solution objective: " << initial_assignment.objective << std::endl;
        std::cout << "===============================================" << std::endl;
        // two-opt local search
        two_opt(problem, initial_assignment, positive_flows, &objectiveComputer);
        std::cout << "Final solution objective: " << initial_assignment.objective << std::endl;

        if(isFeasible(problem, initial_assignment)) {
            std::cout << "Final solution is feasible." << std::endl;
        } else {
            std::cerr << "Final solution is not feasible!" << std::endl;
            return 1;
        }
    }
    if(0)
    {
        std::cout << "===============================================" << std::endl;
        // Load y_bar from file if needed for ObjectiveComputerWithYBar
        std::vector<std::vector<std::vector<std::vector<double>>>> y_bar(problem.n, std::vector<std::vector<std::vector<double>>>(problem.m, std::vector<std::vector<double>>(problem.n, std::vector<double>(problem.m, 0.0))));
        std::string y_bar_file = "/home/local.isima.fr/antran/UFF/QAP-Solver/y_values_318.txt";
        std::ifstream y_bar_fin(y_bar_file);
        int count_positive_y = 0;
        if (y_bar_fin.is_open()) {
            try {
                std::string line;
                while (std::getline(y_bar_fin, line)) {
                    // skip empty lines
                    if (line.empty()) continue;
                    // find '=' separator
                    auto eq = line.find('=');
                    if (eq == std::string::npos) continue;
                    std::string key = line.substr(0, eq);
                    std::string valstr = line.substr(eq + 1);
                    // trim spaces
                    auto trim = [](std::string &s) {
                        size_t a = s.find_first_not_of(" \t\r\n");
                        size_t b = s.find_last_not_of(" \t\r\n");
                        if (a == std::string::npos) { s.clear(); return; }
                        s = s.substr(a, b - a + 1);
                    };
                    trim(key);
                    trim(valstr);
                    // expect key like y[i,u,j,v]
                    if (key.size() < 4) continue;
                    size_t lb = key.find('[');
                    size_t rb = key.find(']');
                    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) continue;
                    std::string inside = key.substr(lb + 1, rb - lb - 1);
                    std::vector<int> idxs;
                    std::istringstream ss(inside);
                    std::string part;
                    while (std::getline(ss, part, ',')) {
                        trim(part);
                        if (part.empty()) break;
                        try {
                            idxs.push_back(std::stoi(part));
                        } catch (...) {
                            idxs.clear();
                            break;
                        }
                    }
                    if (idxs.size() == 4) {
                        int i = idxs[0];
                        int u = idxs[1];
                        int j = idxs[2];
                        int v = idxs[3];
                        if (i >= 0 && i < problem.n && u >= 0 && u < problem.m && j >= 0 && j < problem.n && v >= 0 && v < problem.m) {
                            try {
                                double val = std::stod(valstr);
                                y_bar[i][u][j][v] = val;
                                if (val > 1e-6) {
                                    count_positive_y++;
                                }
                            } catch (...) {
                                // ignore parse errors for this line
                            }
                        }
                    }
                }
                std::cout << "Loaded y_bar from file: " << y_bar_file << std::endl;
                std::cout << "Number of positive y_bar values: " << count_positive_y << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing y_bar file: " << e.what() << std::endl;
                return 1;
            }
            y_bar_fin.close();
        } else {
            std::cerr << "Cannot open y_bar file: " << y_bar_file << std::endl;
            return 1;
        }
        auto objectiveComputerWithYBar = ObjectiveComputerWithYBar();
        objectiveComputerWithYBar.problem = problem;
        objectiveComputerWithYBar.positive_flows = positive_flows;
        objectiveComputerWithYBar.y_bar = y_bar;
        // compute objective with y_bar for initial solution
        double obj_with_y_bar = objectiveComputerWithYBar.compute(initial_assignment.assignment_u_i);
        std::cout << "Initial solution objective with y_bar: " << obj_with_y_bar << std::endl;
        // two-opt local search with y_bar
        std::cout << "===============================================" << std::endl;
        two_opt(problem, initial_assignment, positive_flows, &objectiveComputerWithYBar);
        std::cout << "Final solution objective with y_bar: " << objectiveComputerWithYBar.compute(initial_assignment.assignment_u_i) << std::endl;
        // recompute original objective for final solution
        initial_assignment.objective = objectiveComputer.compute(initial_assignment.assignment_u_i);
        std::cout << "Final solution objective (original): " << initial_assignment.objective << std::endl;

        std::cout << "===============================================" << std::endl;
    }

    {
        std::cout << "===============================================" << std::endl;
        // Load x_bar from file if needed for ObjectiveComputerWithXBar
        auto x_bar_file = "/home/local.isima.fr/antran/UFF/QAP-Solver/x_values_318.txt";
        std::vector<std::vector<double>> x_bar(problem.n, std::vector<double>(problem.m, 0.0));
        std::ifstream x_bar_fin(x_bar_file);
        int count_positive_x = 0;
        if (x_bar_fin.is_open()) {
            std::string line;
            while (std::getline(x_bar_fin, line)) {
                // skip empty lines
                if (line.empty()) continue;
                // find '=' separator
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string key = line.substr(0, eq);
                std::string valstr = line.substr(eq + 1);
                // trim spaces
                auto trim = [](std::string &s) {
                    size_t a = s.find_first_not_of(" \t\r\n");
                    size_t b = s.find_last_not_of(" \t\r\n");
                    if (a == std::string::npos) { s.clear(); return; }
                    s = s.substr(a, b - a + 1);
                };
                trim(key);
                trim(valstr);
                // expect key like x[i,u]
                if (key.size() < 4) continue;
                size_t lb = key.find('[');
                size_t rb = key.find(']');
                if (lb == std::string::npos || rb == std::string::npos || rb <= lb) continue;
                std::string inside = key.substr(lb + 1, rb - lb - 1);
                std::vector<int> idxs;
                std::istringstream ss(inside);
                std::string part;
                while (std::getline(ss, part, ',')) {
                    trim(part);
                    if (part.empty()) break;
                    try {
                        idxs.push_back(std::stoi(part));
                    } catch (...) {
                        idxs.clear();
                        break;
                    }
                }
                if (idxs.size() == 2) {
                    int i = idxs[0];
                    int u = idxs[1];
                    if (i >= 0 && i < problem.n && u >= 0 && u < problem.m) {
                        try {
                            double val = std::stod(valstr);
                            x_bar[i][u] = val;
                            if (val > 1e-6) {
                                count_positive_x++;
                            }
                        } catch (...) {
                            // ignore parse errors for this line
                        }
                    }
                }
            }
            std::cout << "Loaded x_bar from file: " << x_bar_file << std::endl;
            std::cout << "Number of positive x_bar values: " << count_positive_x << std::endl;
        } else {
            std::cerr << "Cannot open x_bar file: " << x_bar_file << std::endl;
            return 1;
        }
        x_bar_fin.close();

        auto objectiveComputerWithXBar = ObjectiveComputerWithXBar();
        objectiveComputerWithXBar.problem = problem;
        objectiveComputerWithXBar.positive_flows = positive_flows;
        objectiveComputerWithXBar.x_bar = x_bar;
        // compute objective with x_bar for initial solution
        double obj_with_x_bar = objectiveComputerWithXBar.compute(initial_assignment.assignment_u_i);
        std::cout << "Initial solution objective with x_bar: " << obj_with_x_bar << std::endl;
        // two-opt local search with x_bar
        std::cout << "===============================================" << std::endl;
        two_opt(problem, initial_assignment, positive_flows, &objectiveComputerWithXBar);
        std::cout << "Final solution objective with x_bar: " << objectiveComputerWithXBar.compute(initial_assignment.assignment_u_i) << std::endl;
        // recompute original objective for final solution
        initial_assignment.objective = objectiveComputer.compute(initial_assignment.assignment_u_i);
        std::cout << "Final solution objective (original): " << initial_assignment.objective << std::endl;
    }
    return 0;
}
