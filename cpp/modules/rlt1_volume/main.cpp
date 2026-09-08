#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <omp.h>

#include "rlt1_volume_solver.hpp"
#include "../../include/qap_solution_io.hpp"
// #include "ExactRefine.h"

/**
 * Main solver function
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <instance.dat> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --time <seconds>      Time limit (default: 3600)" << std::endl;
        std::cout << "  --threads <n>         Number of threads (default: 8)" << std::endl;
        std::cout << "  --log                 Enable detailed logging" << std::endl;
        std::cout << "  --output <file>       Output solution file" << std::endl;
        std::cout << "  --save-dual <file>    Save dual vector and primal solution" << std::endl;
        std::cout << "  --load-dual <file>    Load initial dual vector" << std::endl;
        std::cout << "  --fixed <file>        Fixed variables file (x/y, 0-based indices)" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    std::string instance_path = argv[1];
    int time_limit = 3600;
    int num_threads = 8;
    bool verbose = false;
    std::string output_path = "";
    std::string save_dual_path = "";
    std::string load_dual_path = "";
    std::string fixed_path = "";
    std::string formulation = "";
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--time" && i + 1 < argc) {
            time_limit = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--log") {
            verbose = true;
        } else if (arg == "--output_path" && i + 1 < argc) {
            output_path = argv[++i];
            std::cout << "Output path set to: " << output_path << std::endl;
        } else if (arg == "--save-dual" && i + 1 < argc) {
            save_dual_path = argv[++i];
        } else if (arg == "--load-dual" && i + 1 < argc) {
            load_dual_path = argv[++i];
        } else if (arg == "--fixed" && i + 1 < argc) {
            fixed_path = argv[++i];
        } else if (arg == "--formulation" && i + 1 < argc) {
            formulation = argv[++i];
        }
    }
    std::cout << "Solving using formulation: " << formulation << std::endl;
    // Set number of OpenMP threads
    // if using OpenMP, set the number of threads
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    std::cout << "==================================================" << std::endl;
    std::cout << "RLT1 Volume Algorithm Solver for QAP" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Instance: " << instance_path << std::endl;
    std::cout << "Time limit: " << time_limit << " seconds" << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "==================================================" << std::endl;
    
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
        qap_data.m = n;
    }
    if (formulation == "formulation6" || formulation == "formulation7") {
        qap_data.shiftFixedAssignmentsToEnd(); // Shift fixed assignments to the end of the lists
    }
    
    // Load fixed variables if specified
    FixedVariables fv;
    bool has_fixed = false;
    
    // First, load fixed assignments from the problem (if any)
    if (!qap_data.fixed_assignments.empty()) {
        std::cout << "Loading fixed assignments from problem..." << std::endl;
        for (const auto& kv : qap_data.fixed_assignments) {
            int loc = kv.first;
            int facility = kv.second;
            fv.x_fixed_1[loc] = facility;
        }
        has_fixed = true;
        std::cout << "Loaded " << qap_data.fixed_assignments.size() << " fixed assignments from problem." << std::endl;
    }
    
    // Then, load additional fixed variables from file if specified
    if (!fixed_path.empty()) {
        if (parse_fixed_file(fixed_path, n, fv)) {
            has_fixed = true;
        } else {
            std::cerr << "Error parsing fixed-variable file; continuing without fixed variables." << std::endl;
        }
    }
    
    // Setup Volume problem with loaded fixed variables
    VOL_problem vol_problem;
    RLT1VolumeHooks* hooks = nullptr;
    if (formulation == "formulation1") {
        set_up_vol_problem_1(vol_problem, qap_data, fv);
        hooks = new RLT1VolumeHooks1(qap_data);
    } else if (formulation == "formulation2") {
        set_up_vol_problem_2(vol_problem, qap_data, fv);
        hooks = new RLT1VolumeHooks2(qap_data);
    } else if (formulation == "formulation3") {
        set_up_vol_problem_3(vol_problem, qap_data, fv);
        hooks = new RLT1VolumeHooks3(qap_data);
    } else if (formulation == "formulation4") {
        set_up_vol_problem_4(vol_problem, qap_data, fv);
        hooks = new RLT1VolumeHooks4(qap_data);
    } else if (formulation == "formulation5") {
        set_up_vol_problem_5(vol_problem, qap_data, fv);
        hooks = new RLT1VolumeHooks5(qap_data);
    } else if (formulation == "formulation6") {
        set_up_vol_problem_6(vol_problem, qap_data, fv);
        hooks = new RLT1VolumeHooks6(qap_data);
        for (const auto& kv : fv.x_fixed_1) {
            int loc = kv.first;
            int fac = kv.second;
            std::cout << "Fixed variable: x[" << loc << "," << fac << "] = 1" << std::endl;
        }
    } else if (formulation == "formulation7") {
        set_up_vol_problem_7(vol_problem, qap_data, fv);
        hooks = new RLT1VolumeHooks7(qap_data);
        for (const auto& kv : fv.x_fixed_1) {
            int loc = kv.first;
            int fac = kv.second;
            std::cout << "Fixed variable: x[" << loc << "," << fac << "] = 1" << std::endl;
        }
    } else {
        std::cerr << "Error: Unknown formulation specified. Use --formulation <formulation1|formulation2|formulation3|formulation4|formulation5|formulation6|formulation7>" << std::endl;
        return 1;
    }
    hooks->set_fixed_variables(fv);
    
    // Load dual vector if specified
    
    if (!load_dual_path.empty()) {
        std::ifstream dual_file(load_dual_path, std::ios::binary);
        if (dual_file.is_open()) {
            std::cout << "Loading dual vector from: " << load_dual_path << std::endl;
            for (int i = 0; i < vol_problem.dsize; ++i) {
                dual_file.read(reinterpret_cast<char*>(&vol_problem.dsol[i]), sizeof(double));
            }
            dual_file.close();
            std::cout << "Dual vector loaded successfully." << std::endl;
        } else {
            std::cerr << "Warning: Could not open dual file: " << load_dual_path << std::endl;
            std::cerr << "Starting with zero dual vector." << std::endl;
        }
    }

    set_up_volume_parameters(vol_problem, verbose);
    
    // Determine if we should use loaded dual (warm-start)
    bool use_dual_warmstart = !load_dual_path.empty();
    
    // Solve
    std::cout << "\nStarting Volume algorithm..." << std::endl;
    if (use_dual_warmstart) {
        std::cout << "Using warm-start from loaded dual vector." << std::endl;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int retval =vol_problem.solve(*hooks, use_dual_warmstart);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = duration.count() / 1000.0;
    
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Volume Algorithm Complete" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Status: " << (retval == 0 ? "Success" : "Error") << std::endl;
    std::cout << "Lower bound: " << vol_problem.value << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_seconds << " seconds" << std::endl;
    
    // Save dual vector and primal solution if requested
    if (!save_dual_path.empty()) {
        std::cout << "\nSaving dual vector and primal solution to: " << save_dual_path << std::endl;
        
        // Save dual vector
        std::ofstream dual_file(save_dual_path, std::ios::binary);
        if (dual_file.is_open()) {
            for (int i = 0; i < vol_problem.dsize; ++i) {
                dual_file.write(reinterpret_cast<const char*>(&vol_problem.dsol[i]), sizeof(double));
            }
            dual_file.close();
            std::cout << "Dual vector saved (" << vol_problem.dsize << " values)." << std::endl;
        } else {
            std::cerr << "Error: Could not save dual vector to: " << save_dual_path << std::endl;
        }
        
        // Save primal solution
        std::string primal_path = save_dual_path + ".primal";
        std::ofstream primal_file(primal_path, std::ios::binary);
        if (primal_file.is_open()) {
            for (int i = 0; i < vol_problem.psize; ++i) {
                primal_file.write(reinterpret_cast<const char*>(&vol_problem.psol[i]), sizeof(double));
            }
            primal_file.close();
            std::cout << "Primal solution saved to: " << primal_path << " (" << vol_problem.psize << " values)." << std::endl;
        } else {
            std::cerr << "Error: Could not save primal solution to: " << primal_path << std::endl;
        }
    }
    
    // Extract and save best solution if requested
    if (!output_path.empty() && vol_problem.psol.size() > 0) {
        std::cout << "\nExtracting best solution from primal variables and saving to: " << output_path << std::endl;
        // open output file
        std::ofstream out_file(output_path);
        if (!out_file.is_open()) {
            std::cerr << "Error: Could not open output file: " << output_path << std::endl;
        } else {
            // write solution: x_i_u = value
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    double val = vol_problem.psol[i * n + u];
                    out_file << i << " " << u << " " << val << "\n";
                }
            }
            out_file.close();
            std::cout << "Best solution extracted and saved to: " << output_path << std::endl;
        }
        // Extract assignment from x variables
        std::vector<int> assignment(n);
        for (int i = 0; i < n; ++i) {
            int best_u = 0;
            double best_val = vol_problem.psol[i * n];
            for (int u = 1; u < n; ++u) {
                if (vol_problem.psol[i * n + u] > best_val) {
                    best_val = vol_problem.psol[i * n + u];
                    best_u = u;
                }
            }
            assignment[i] = best_u;
        }
        
        // Evaluate objective
        double obj = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                obj += qap_data.D[i][j] * qap_data.F[assignment[i]][assignment[j]];
            }
        }
        
        std::cout << "Primal objective: " << obj << std::endl;

    }
    
    // Always report a summary of primal violations (assignment, link, symmetry)
    if (vol_problem.psol.size() > 0 and formulation != "formulation3" && formulation != "formulation4" && formulation != "formulation6" && formulation != "formulation2") {
        std::cout << "\nAnalyzing primal solution for constraint violations..." << std::endl;
        auto vsummary = compute_primal_violation_summary(vol_problem.psol, n);
        std::cout << "\nPrimal violation summary (n=" << vsummary.n << ")" << std::endl;
        std::cout << "  Max abs: " << std::setprecision(6) << vsummary.max_abs
                  << ", Avg abs: " << vsummary.avg_abs << std::endl;
        std::cout << "  Assignment  -> max: " << vsummary.assignment_max
                  << ", avg: " << vsummary.assignment_avg << std::endl;
        std::cout << "  Linking     -> max: " << vsummary.link_max
                  << ", avg: " << vsummary.link_avg << std::endl;
        std::cout << "  Symmetry    -> included in Max/Avg above" << std::endl;
        if (true) {
            std::cout << "\nChecking fixed variable violations..." << std::endl;
            auto frep = check_fixed_violations(fv, vol_problem.psol, n);
            std::cout << "  Fixed vars  -> x1: " << fv.x_fixed_1.size()
                      << " (viol " << frep.x1_violations << ", max dev " << frep.x1_max_dev << ")"
                      << ", x0-rows: " << fv.x_fixed_0.size()
                      << " (viol " << frep.x0_violations << ", max dev " << frep.x0_max_dev << ")"
                      << ", y1: " << fv.y_fixed_1.size()
                      << " (viol " << frep.y1_violations << ", max dev " << frep.y1_max_dev << ")"
                      << ", y0-rows: " << fv.y_fixed_0.size()
                      << " (viol " << frep.y0_violations << ", max dev " << frep.y0_max_dev << ")" << std::endl;
        }
    }

    // using ExactRefine to solve the problem to optimality with CPLEX/Gurobi
    // {
    //     ExactRefine exact_refine(qap_data, vol_problem.psol, vol_problem.value);
    //     exact_refine.solve();
    // }

    // if formulation 1 or 2, count the number of positive y
    // if (formulation == "formulation1" || formulation == "formulation2" || formulation == "formulation7") 
    // {
    //     int count_positive_y = 0;
    //     for (int i = 0; i < n; ++i) {
    //         for (int u = 0; u < n; ++u) {
    //             for (int j = i + 1; j < n; ++j) {
    //                 for (int v = 0; v < n; ++v) {
    //                     if (u == v) continue; // Skip if u == v, as y(i,u,j,v) is not defined for u == v
    //                     if (vol_problem.psol[n * n + i * n * n * n + u * n * n + j * n + v] > 1e-6) {
    //                         ++count_positive_y;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "Number of positive y variables: " << count_positive_y << std::endl;
    //     // save the positive y variables to a file
    //     // filename: instance_path/instance_name_positive_y.txt
    //     std::string instance_name = instance_path.substr(instance_path.find_last_of("/\\") + 1);
    //     std::string instance_dir = instance_path.substr(0, instance_path.find_last_of("/\\"));
    //     std::string positive_y_file = instance_dir + "/" + instance_name + "_positive_y.txt";
    //     std::ofstream py_file(positive_y_file);
    //     if (py_file.is_open()) {
    //         for (int i = 0; i < n; ++i) {
    //             for (int u = 0; u < n; ++u) {
    //                 for (int j = i + 1; j < n; ++j) {
    //                     for (int v = 0; v < n; ++v) {
    //                         if (u == v) continue; // Skip if u == v, as y(i,u,j,v) is not defined for u == v
    //                         double y_val = vol_problem.psol[n * n + i * n * n * n + u * n * n + j * n + v];
    //                         if (y_val > 0) {
    //                             py_file << i << " " << u << " " << j << " " << v << std::endl;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         py_file.close();
    //     }
    //     std::cout << "Positive y variables saved to: " << positive_y_file << std::endl;
    // }

    // if formulation 6, print best assignment and objective
    if (formulation == "formulation6" ) {
        // hooks must be RLT1VolumeHooks6
        auto* hooks6 = dynamic_cast<RLT1VolumeHooks6*>(hooks);
        if (hooks6) {
            std::cout << "\nBest assignment found: " << hooks6->best_assignment_cost << std::endl;
        }
    }
    std::cout << "==================================================" << std::endl;
    auto instance_name = instance_path.substr(instance_path.find_last_of("/\\") + 1);
    
    auto n_iterations = vol_problem.iter();
    std::cout << instance_name << " & " << formulation << " & " << vol_problem.value << " & " << n_iterations << " & " << std::fixed << std::setprecision(2) << elapsed_seconds << "s\\\\" << std::endl;
    
    return retval;
}
