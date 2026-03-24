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

#include "m4_volume_solver.hpp"
#include "../../include/qap_solution_io.hpp"

#define y(i, u, j, v) (psol[(i) * n * n * n + (u) * n * n + (j) * n + (v)])
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
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
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
    omp_set_num_threads(num_threads);
    
    std::cout << "==================================================" << std::endl;
    std::cout << "M4 Volume Algorithm Solver for QAP" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Instance: " << instance_path << std::endl;
    std::cout << "Time limit: " << time_limit << " seconds" << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Read QAP instance
    auto qap_data = Problem::fromQAPLIB(instance_path);
    
    int n = qap_data.n;
    std::cout << "Problem size: n = " << n << std::endl;
    // Load dual vector if specified
    VOL_dvector dsol;
    if (!load_dual_path.empty()) {
        std::ifstream dual_file(load_dual_path, std::ios::binary);
        if (dual_file.is_open()) {
            std::cout << "Loading dual vector from: " << load_dual_path << std::endl;
            while (dual_file.peek() != EOF) {
                dsol.allocate(dsol.size() + 1);
                dual_file.read(reinterpret_cast<char*>(&dsol[dsol.size() - 1]), sizeof(double));
            }
            dual_file.close();
            std::cout << "Dual vector loaded successfully." << std::endl;
        } else {
            std::cerr << "Warning: Could not open dual file: " << load_dual_path << std::endl;
            std::cerr << "Starting with zero dual vector." << std::endl;
        }
    }

    FixedVariables fv;
    bool has_fixed = false;
    if (!fixed_path.empty()) {
        parse_fixed_file(fixed_path, n, fv);
    }
    VOL_dvector psol;
    std::map<int, int> violated_constraints;



    for (auto i=0; i<n; ++i) {
                    bool violated = false;
        for (auto u=0; u<n; ++u) {
            for (auto j=0; j<n; ++j) {
                for (auto v=0; v<n; ++v) {
                    //  if psol is computed, check violation sum_w y[i,u,j,w] = sum_k y[i,u,k,v]
                    if (psol.size() > 0) {
                        double lhs = 0.0, rhs = 0.0;
                        for (int w = 0; w < n; ++w) {
                            lhs += y(i, u, j, w);
                        }
                        for (int k = 0; k < n; ++k) {
                            if (k == j) continue;
                            rhs += y(i, u, k, v);
                        }
                        double violation = lhs - rhs;
                        if (std::abs(violation) > 1e-6) {
                            int key = i * n * n * n + u * n * n + j * n + v;
                            violated_constraints[key] = violated_constraints.size(); // assign an index
                            violated = true;
                            std::cout << "Constraint violated for (i=" << i << ", u=" << u << ", j=" << j << ", v=" << v << "): "
                                      << "LHS = " << lhs << ", RHS = " << rhs << ", Violation = " << violation << std::endl;
                        }
                    }
                }
            }
        }
    if (psol.size() > 0 && (!violated)) {
                    
        continue;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    
    VolumeResult result = solve_m4_volume_relax(qap_data, fv, false, dsol, formulation, violated_constraints);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = duration.count() / 1000.0;
    
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Volume Algorithm Complete" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Status: " << (result.status == 0 ? "Success" : "Error") << std::endl;
    std::cout << "Lower bound: " << result.lower_bound << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_seconds << " seconds" << std::endl;
    if (result.status != 0) {
        return 1;
    }
    dsol = result.dual;
    psol = result.primal;
    
    // Always report a summary of primal violations (assignment, link, symmetry)
    if (result.primal.size() > 0 and formulation != "formulation3" && formulation != "formulation4") {
        auto vsummary = compute_primal_violation_summary(result.primal, n);
        std::cout << "\nPrimal violation summary (n=" << vsummary.n << ")" << std::endl;
        std::cout << "  Max abs: " << std::setprecision(6) << vsummary.max_abs
                  << ", Avg abs: " << vsummary.avg_abs << std::endl;
        std::cout << "  Assignment  -> max: " << vsummary.assignment_max
                  << ", avg: " << vsummary.assignment_avg << std::endl;
        std::cout << "  Linking     -> max: " << vsummary.link_max
                  << ", avg: " << vsummary.link_avg << std::endl;
        std::cout << "  Symmetry    -> included in Max/Avg above" << std::endl;
        if (has_fixed) {
            auto frep = check_fixed_violations(fv, result.primal, n);
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
    
    std::cout << "==================================================" << std::endl;
    auto instance_name = instance_path.substr(instance_path.find_last_of("/\\") + 1);
    
    }

    return 0;
}
