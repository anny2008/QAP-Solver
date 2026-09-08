#include "Solver.h"
#include "PricingEngine.h"
// include LocalSearchSolver from local search module
#include "../local_search/local_search.h"

// constructor
SFDCGSolver::SFDCGSolver(const SolverConfig& cfg) : cfg(cfg) {}

/**
 * Load the problem, and solve it using the following steps:
 * 1) Decompose the flow graph into subgraphs
 * 2) Using local search to find an initial solution, and add corresponding columns to the RMP
 * 3) Repeat until no negative reduced cost column can be found, or max iterations reached:
 *      a) Solve the RMP
 *      b) Get duals and call pricing engine to find negative reduced cost columns
 *      c) If negative reduced cost columns found, add them to the RMP; 
 *         if not, if add_most_negative is true and we found some most negative columns, add those;
 *         otherwise stop
 */
Solution SFDCGSolver::solve(Problem& problem, const std::string& instance_path) {
    auto n = problem.n;
    // 1) Decompose the flow graph into subgraphs
    // if decomposition file exists, load it; otherwise decompose and save it for future loading
    std::vector<Subgraph> subgraphs;
    std::ifstream decomp_input(cfg.decomposition_file);
    if (decomp_input.good()) {
        std::cout << "Loading decomposition from " << cfg.decomposition_file << std::endl;
        subgraphs = loadDecomposition(cfg.decomposition_file);
    } else {
        subgraphs = decomposeIntoSubgraphs(problem, cfg.decomposition);
        saveDecomposition(subgraphs, cfg.decomposition_file);
    }
    std::cout << "Decomposed flow graph into " << subgraphs.size() << " subgraphs using strategy " << cfg.decomposition << std::endl;
    // Create RMP
    IncrementalRMP rmp(problem, cfg, subgraphs);
    rmp.createModel();
    // Create a PricingEngine
    PricingEngine pricingEngine;

    std::string basis_output_path = cfg.basis_output_file; // path to write basis columns for debugging
    // Load from basis output path if it exists, otherwise create it during column generation
    // std::ifstream basis_input(basis_output_path);
    // if (basis_input.good()) {
    //     std::cout << "Loading basis columns from " << basis_output_path << std::endl;
    //     std::vector<TripleKey> basis_columns;
    //     std::string line;
    //     while (std::getline(basis_input, line)) {
    //         std::istringstream iss(line);
    //         std::string token;
    //         std::vector<int> values;
    //         while (std::getline(iss, token, ',')) {
    //             values.push_back(std::stoi(token));
    //         }
    //         if (values.size() == 3) {
    //             basis_columns.push_back({values[0], values[1], values[2]});
    //         }
    //     }
    //     // add basis columns to RMP
    //     rmp.addColumns(basis_columns);
    // } else {
    //     std::cout << "No basis columns file found at " << basis_output_path << ". It will be created during column generation.\n";
    

    // 2) Using Local Search to find a feasible Solution
    std::vector<int> warmstart;
    // LocalSearch ls;
    // Solution ls_sln(instance_path, "local_search");
    // ls_sln.assignment = LocalSearch::initAssignment(n, "random", "");
    // auto warmup_max_iterations = 500;
    // auto warmup_tabu_tenure = 10;
    // ls.tabuSearch(problem, ls_sln, warmup_max_iterations, warmup_tabu_tenure, false);
    // if (cfg.log_output)  std::cout << "[CG-VA] Local Search warmup objective: " << ls_sln.objective << std::endl;
    // // check if local search solution satisfies fixed assignments
    // // if not, create a random solution that satisfies fixed assignments
    // auto satisfies_fixed = true;
    // for (const auto& [location, facility] : problem.fixed_assignments) {
    //     if (ls_sln.assignment[location] != facility) {
    //         satisfies_fixed = false;
    //         break;
    //     }
    // }
    // if(!satisfies_fixed) {
    for (auto aiter = 0; aiter < 10; ++aiter)
    {
        if (cfg.log_output) std::cout << "[CG-VA] Local Search solution does not satisfy fixed assignments, creating a random solution that satisfies fixed assignments\n";
        std::vector<int> assignment(n, -1);
        std::vector<bool> facility_assigned(n, false);
        for (const auto& [location, facility] : problem.fixed_assignments) {
            assignment[location] = facility;
            facility_assigned[facility] = true;
        }
        // assign remaining locations randomly to unassigned facilities
        std::vector<int> unassigned_facilities;
        for (int u = 0; u < n; u++) {
            if (!facility_assigned[u]) {
                unassigned_facilities.push_back(u);
            }
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(unassigned_facilities.begin(), unassigned_facilities.end(), gen);
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (assignment[i] == -1) {
                assignment[i] = unassigned_facilities[idx++];
            }
        }
        warmstart = assignment;
    // } else {
    //     warmstart = ls_sln.assignment;
    // }

    // extract positive columns from local search solution
    auto initial_columns = extractColumnFromAssignment(warmstart, subgraphs);
    if (cfg.log_output) {
        std::cout << "[CG-VA] Adding " << initial_columns.size() << " initial columns from local search solution\n";
    }
    // add those columns to RMP
    // record adding time
    auto added = rmp.addColumns(initial_columns);
    if (cfg.log_output) {
        std::cout << "[CG-VA] Added " << added << " initial columns from local search solution\n";
    }
}
// }
    // solve RMP
    if (cfg.log_output) {
        std::cout << "[CG-VA] Starting column generation iterations...\n";
    }
    auto iter = 0;
    // record total column generation time
    auto start_cg = std::chrono::high_resolution_clock::now();
    auto lastObj = -1;
    rmp.nMaxColsPerSubgraph = 1;
    while(true) {
        // if (iter % 100 == 0) {
        //     cfg.log_output = true; // force log every 1000 iterations to track progress on long runs
        // } else {
        //     cfg.log_output = false;
        // }
        if (cfg.log_output) std::cout<< iter <<", time " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_cg).count() << " seconds ============================================" << std::endl;
        // solve the relax restricted master problem
        auto ok = rmp.solve();
        if (!ok)
        {
            if (cfg.log_output) std::cout<< "[CG-VA] Can't solve the rmp" << std::endl;
            break;
        }
        // auto curObj = rmp.getObjectiveValue();
        // if (curObj < lastObj) {
        //     rmp.nMaxColsPerSubgraph = std::max(1, rmp.nMaxColsPerSubgraph - 1); // if objective decreased, decrease the number of columns we add per subgraph to focus more on the most negative columns
        // }
        // lastObj = curObj;
        
        // if(iter % 1000 == 0) {
        //     // compute and sort unvisited columns by their reduced cost
        //     auto pricingResult = pricingEngine.all_pricing(rmp);
        //     // sort
        //     std::sort(pricingResult.begin(), pricingResult.end());
        //     std::cout << "[CG-VA] Iter " << iter << ": Best reduced cost among all columns is " << pricingResult[0].rc << std::endl;
        //     // create a new not_in_omega list sorted by reduced cost
        //     rmp.not_in_omega.clear();
        //     for (const auto& rc : pricingResult) {
        //         rmp.not_in_omega.emplace_back(rc.key);
        //     }
        //     if (iter == 0) { //add 10% most negative columns to Omega
        //         auto best_rc = pricingResult[0].rc;
        //         auto threshold = best_rc*0.99; // add a small epsilon to include columns with rc very close to best_rc
        //         std::vector<TripleKey> initial_negative_cols;
        //         for (const auto& rc : pricingResult) {
        //             if (rc.rc <= threshold) {
        //                 initial_negative_cols.emplace_back(rc.key);
        //             } else {
        //                 break; // since sorted, we can stop once we are above the threshold
        //             }
        //         }
        //         auto added = rmp.addColumns(initial_negative_cols);
        //         if (cfg.log_output) {
        //             std::cout << "[CG-VA] Iter " << iter << ": Added " << added <<"/"<< initial_negative_cols.size() << " initial negative columns with rc " << best_rc << std::endl;
        //         }
        //     }
        // }
        // do pricing and record time
        auto start_pricing = std::chrono::high_resolution_clock::now();
        auto pricingResult = pricingEngine.price(rmp);
        auto end_pricing = std::chrono::high_resolution_clock::now();
        if (cfg.log_output) std::cout << " [CG-VA] Pricing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_pricing - start_pricing).count() << " ms" << std::endl;
        auto added = 0;
        if (cfg.log_output) std::cout<< "[CG-VA] Best rc" << pricingResult.best_rc << std::endl;
        // record adding time
        auto start_adding = std::chrono::high_resolution_clock::now();
        if (pricingResult.best_rc > -1e-7) {
            if (cfg.log_output) {
                std::cout << "[CG-VA] No negative reduced cost column found, stopping column generation.\n";
            }
            break;
        } else {
            if (cfg.add_most_negative && !pricingResult.most_negative_cols.empty()) {
                added = rmp.addColumns(pricingResult.most_negative_cols);
                if (cfg.log_output) {
                    std::cout << "[CG-VA] Added " << added <<"/"<< pricingResult.most_negative_cols.size() << " most negative columns with rc " << pricingResult.best_rc << std::endl;
                }
            } else if (!pricingResult.negative_cols.empty()){
                added = rmp.addColumns(pricingResult.negative_cols);
                if (cfg.log_output) {
                    std::cout << "[CG-VA] Added " << added <<"/"<< pricingResult.negative_cols.size() << " negative columns with best rc " << pricingResult.best_rc << std::endl;
                }
            }
        }
        auto end_adding = std::chrono::high_resolution_clock::now();
        if (cfg.log_output) std::cout << " [CG-VA] Time to add columns: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_adding - start_adding).count() << " ms" << std::endl;
        iter++;
        if (added==0) break;
        // break;
    }
    auto end_cg = std::chrono::high_resolution_clock::now();
    // cfg.log_output = true; // re-enable log for final output
    if (cfg.log_output) std::cout << "[CG-VA] Total column generation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cg - start_cg).count() << " ms" << std::endl;
    // rmp.solve(); // solve one last time to get the final solution
    rmp.countBasisAndPositiveColumns(""); // count the basis columns and positive columns in the final solution and print them (for debugging)
}



/**
 * get all positive variable e from assignment
 * - assignment: Permutation π where π[u] = i means facility u → location i
 * - subgraphs: list of all subgraphs
 * returns list of TripleKey (k,i,j) for all e^k_ij that should be 1 according to the assignment and subgraphs
 */
std::vector<TripleKey> SFDCGSolver::extractColumnFromAssignment(std::vector<int>& assignment, std::vector<Subgraph>& subgraphs){
    std::vector<TripleKey> columns;
    std::vector<int> assignment_m;
    assignment_m.resize(assignment.size(), -1);
    for (auto i = 0; i < assignment.size(); i++)
    {
        assignment_m[assignment[i]] = i;
    }
    // for each subgraph k, for each arc (u,v) in that subgraph, get i = assignment[u], j = assignment[v], and add (k,i,j) to columns
    for (const auto& subgraph : subgraphs) {
        int k = subgraph.k;
        for (const auto& arc : subgraph.arcs) {
            int u = arc.first;
            int v = arc.second;
            int i = assignment_m[u];
            int j = assignment_m[v];
            columns.push_back({k,i,j});
        }
    }
    return columns;
}
