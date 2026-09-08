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

#include "../../include/qap_solution_io.hpp"
#include "Bundle.h"
#include "QPPnltMP.h"
#include "sdp_bundle_solver.h"



/**
 * Main solver function
 */
int main(int argc, char* argv[]) {
    auto retval = 0;
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
    int iteration_limit = 10;
    int num_threads = 8;
    bool verbose = false;
    std::string output_path = "";
    std::string save_dual_path = "";
    std::string load_dual_path = "";
    std::string fixed_path = "";
    std::string formulation = "";
    std::string eigen_solver = "lanczos"; // default eigen solver
    double mSS = 0.25; // SS condition: if DeltaFi >= | mSS | * Deltav, then a SS is declared
    
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
        } else if (arg == "--iteration-limit" && i + 1 < argc) {
            iteration_limit = std::atoi(argv[++i]);
        } else if (arg == "--mSS" && i + 1 < argc) {
            mSS = std::atof(argv[++i]);
        } else if (arg == "--eigen-solver" && i + 1 < argc) {
            eigen_solver = argv[++i];
        }
    }
    std::cout << "Solving using formulation: " << formulation << std::endl;
    // Set number of OpenMP threads
    // if using OpenMP, set the number of threads
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    std::cout << "OpenMP threads set to: " << omp_get_max_threads() << std::endl;
    #endif

    std::cout << "==================================================" << std::endl;
    std::cout << "SDP Bundle Algorithm Solver for QAP" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Instance: " << instance_path << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Iteration limit: " << iteration_limit << std::endl;
    std::cout << "Time limit: " << time_limit << std::endl;
    std::cout << "Formulation: " << formulation << std::endl;
    std::cout << "mSS: " << mSS << std::endl;
    std::cout << "Eigen solver: " << eigen_solver << std::endl;
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
    if (formulation == "formulation3") {
        qap_data.shiftFixedAssignmentsToEnd(); // Shift fixed assignments to the end of the lists
    }

    SDPOracle oracle(qap_data, formulation);
    oracle.eigen_solver = eigen_solver;
    
    
    
    // Determine if we should use loaded dual (warm-start)
    bool use_dual_warmstart = !load_dual_path.empty();
    
    // Solve
    std::cout << "\nStarting Bundle algorithm..." << std::endl;
    if (use_dual_warmstart) {
        std::cout << "Warning: dual warm-start is not wired for the bundle driver yet." << std::endl;
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    Bundle bundle;

    bundle.SetPar(Bundle::kBPar1, 10); // If an item has had a zero multiplier for the last BPar1 steps, it is eliminated
    bundle.SetPar(Bundle::kBPar2, oracle.dsize); // Maximum dimension of the bundle
    bundle.SetPar(Bundle::km1, mSS); // SS condition: if DeltaFi >= | m1 | * Deltav, then a SS is done.
    bundle.SetPar(Bundle::km3, 2.0); // A newly obtained subgradient is deemed "useless" if Alfa >= m3 * Sigma; in this case, if a NS has to be done,  t is decreased.
    bundle.SetPar(Bundle::ktMaior, 1e6); // Max t
    bundle.SetPar(Bundle::ktMinor, 1e-6); // Min t
    bundle.SetPar(Bundle::ktInit, 0.1); // initial value of t. These parameters may be critical, but they are not very difficult to set.
    
    QPPenaltyMP master;
    bundle.SetMPSolver(&master);
    bundle.SetFiOracle(&oracle);
    // set number of iterations
    bundle.SetPar(NDOSolver::kMaxItr, iteration_limit);
    bundle.SetPar(NDOSolver::kMaxTme, static_cast<double>(time_limit));
    bundle.SetPar(NDOSolver::kEpsLin, 1e-6);
    // bundle.SetNDOLog(verbose ? &std::cout : nullptr, verbose ? char(1) : char(0));
    bundle.SetNDOLog(&std::cout, char(1));
    
    oracle.start_time = start_time;
    auto bundle_status = bundle.Solve();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = duration.count() / 1000.0;

    auto average_subproblem_time = oracle.subproblem_solving_time.count() / oracle.iteration;
    
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Bundle Algorithm Complete" << std::endl;
    std::cout << "==================================================" << std::endl;
    retval = (bundle_status == NDOSolver::kOK) ? 0 : 1;
    std::cout << "Status: " << retval << std::endl;
    std::cout << "Best lower bound: " << -bundle.ReadBestFiVal() << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_seconds << " seconds" << std::endl;
    
    std::cout << "==================================================" << std::endl;
    auto instance_name = instance_path.substr(instance_path.find_last_of("/\\") + 1);
    
    // auto n_iterations = vol_problem.iter();
    std::cout << instance_name << " & " << formulation << " & " << eigen_solver << " & " << std::fixed << std::setprecision(4) << average_subproblem_time << "s & " << -bundle.ReadBestFiVal() << " & " << oracle.iteration << " & " << std::fixed << std::setprecision(2) << elapsed_seconds << "s\\\\" << std::endl;
    
    return retval;
}
