#include "ColumnGenSolver.h"
#include "StabilizedDuals.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "../local_search/local_search.h"

ColumnGenSolver::ColumnGenSolver(const ColumnGenConfig& cfg_) : cfg(cfg_) {}

std::vector<double> ColumnGenSolver::buildPhi(const Problem& P) {
    int n = P.n;
    int m = P.n; // QAP: |V|==|M| == n
    std::cout << "[CG] Building phi matrix for n=" << n << ", m=" << m << "\n";
    std::vector<double> phi(((n*m)*n*m), 0.0);
    for (int i=0;i<n;++i) for (int u=0;u<m;++u)
    for (int j=0;j<n;++j) for (int v=0;v<m;++v) {
        // D,F are P.D,P.F
        phi[(((i*m)+u)*n + j)*m + v] = P.D[i][j] * P.F[u][v];
    }
    return phi;
}

std::vector<QuadKey> ColumnGenSolver::buildInitialOmega(
    const Problem& P, const std::unordered_map<int,int>& fixed)
{
    int n = P.n, m = P.n;
    std::vector<QuadKey> out;
    for (int i=0;i<n;++i) {
        for (int u=0;u<m;++u) {
            if (fixed.count(i) && fixed.at(i)!=u) continue;
            for (int j=0;j<n;++j) {
                for (int v=0;v<m;++v) {
                    if (fixed.count(j) && fixed.at(j)!=v) continue;
                    // if ((i==j && u!=v) || (i!=j && u==v)) continue;
                    if (i == j || u == v) continue;
                    if (P.F[u][v] > 0.0 || P.F[v][u] > 0.0) {
                        out.push_back(IncrementalRMP::canonical(i,u,j,v));
                    }
                        // out.push_back(IncrementalRMP::canonical(i,u,j,v));
                }
            }
        }
    }
    // deduplicate canonicals
    std::sort(out.begin(), out.end(), [](const QuadKey& a, const QuadKey& b){
        if (a.i!=b.i) return a.i<b.i; if (a.u!=b.u) return a.u<b.u;
        if (a.j!=b.j) return a.j<b.j; return a.v<b.v;
    });
    out.erase(std::unique(out.begin(), out.end(),
        [](const QuadKey& a, const QuadKey& b){
            return a.i==b.i && a.u==b.u && a.j==b.j && a.v==b.v;
        }), out.end());
    return out;
}

std::vector<int> ColumnGenSolver::argmaxAssignment(
    int n, int m, const std::unordered_map<PairKey,double,PairKeyHash>& xvals)
{
    std::vector<int> assign(n, 0);
    for (int i=0;i<n;++i) {
        double best = -1.0; int bestu = 0;
        for (int u=0; u<m; ++u) {
            auto it = xvals.find(PairKey{i,u});
            double val = (it==xvals.end()) ? 0.0 : it->second;
            if (val > best) { best = val; bestu = u; }
        }
        assign[i] = bestu;
    }
    return assign;
}

Solution ColumnGenSolver::solve(
    const Problem& problem,
    const std::string& instance_path,
    const std::unordered_map<int,int>& fixed_variables
)
{
    int n = problem.n;
    int m = problem.n;

    // Build index sets
    std::vector<int> V(n); for (int i=0;i<n;++i) V[i]=i;
    std::vector<int> M = V;
    auto phi = buildPhi(problem);

    IncrementalRMP rmp(V, M, phi, n, m, cfg.eps, true, true);
    rmp.ensureAllRows();
    if(cfg.use_dual_box_restriction) {
        rmp.addAllTplusTminus(); // add all t_plus and t_minus variables for stabilization
    }

    if (!fixed_variables.empty()) {
        std::cout << "[CG] Fixing " << fixed_variables.size() << " variables based on input\n";
        rmp.fixXAssignments(fixed_variables);
    }

    // auto Omega0 = buildInitialOmega(problem, fixed_variables);
    std::vector<QuadKey> Omega0;

    // Load all positive columns from colfile
    // std::ifstream infile(instance_path + "_positive_columns.txt");
    // if (infile.is_open()) {
    //     std::string line;
    //     while (std::getline(infile, line)) {
    //         std::istringstream iss(line);
    //         int i, u, j, v;
    //         if (!(iss >> i >> u >> j >> v)) { break; }
    //         Omega0.push_back(IncrementalRMP::canonical(i,u,j,v));
    //     }
    //     infile.close();
    //     std::cout << "[CG] Loaded " << Omega0.size() << " positive columns from " << instance_path + "_positive_columns.txt" << "\n";
    // } else {
    //     std::cout << "[CG] No positive columns file found at " << instance_path + "_positive_columns.txt, starting with empty initial Omega\n";
    //     Omega0 = buildInitialOmega(problem, fixed_variables);
    // }
    std::cout << "[CG] Initial Omega0 size = " << Omega0.size() << "\n";
    rmp.addColumns(Omega0);
    std::cout << "[CG] Added initial columns from Omega0, omega is now " << rmp.omega().size() << "\n";

    PricingEngine pricing(V, M, phi, n, m, cfg.add_column_strategy, cfg.eps, fixed_variables);

    // StabilizedDuals for tracking duals and stabilization
    double lambda = cfg.lambda_init;
    DenseDuals tilde_du(n, m);      // \tilde{\pi}
    DenseDuals cur_du(n, m);        // \pi^k
    DenseDuals last_du(n, m);        // \pi^k
    DenseDuals pi_k1(n, m);         // Reuse this to avoid allocation in loop
    // Duals for box constraints
    DenseDuals hat_pi(n, m);     // box center
    DenseDuals delta(n, m);      // box radii
    bool have_hat_pi = false;
    double gamma = cfg.box_gamma;   // new parameter, e.g. 0.2
    double alpha = cfg.box_alpha;   // new parameter, e.g. 0.1


    int iter = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    bool have_tilde = false;
    std::cout << "[CG] Starting column generation iterations...\n";
    
    // // Initial solve to get starting point and duals
    double solve_time = 0.0;
    bool solving_status;
    // if (cfg.log_output) {
    //     auto t0 = std::chrono::high_resolution_clock::now();
    //     solving_status = rmp.solve();
    //     solve_time = std::chrono::duration<double>(
    //         std::chrono::high_resolution_clock::now() - t0).count();
    // } else {
    //     solving_status = rmp.solve();
    // }
    // std::cout << "[CG] Initial solve status: " << (solving_status ? "success" : "failure") << ", objective: " << rmp.getObjectiveValue() << "\n";

    // if (!solving_status) 
    {
        std::cerr << "[CG] CPLEX solving status: " << rmp.cplex().getStatus() << "\n";
        std::cerr << "[CG] CPLEX failed to solve in initial solve, using Local Search for warmstart\n";
        auto n_rows = rmp.cplex().getNrows();
        for (int lsiter = 0; lsiter < 1; ++lsiter)
        {
            LocalSearch ls;
            Solution ls_sln(instance_path, "local_search");
            ls_sln.assignment = LocalSearch::initAssignment(problem.n, "random", "");
            auto max_iterations = 1000; // minimum iterations
            auto tabu_tenure = 10;
            ls.tabuSearch(problem, ls_sln, max_iterations, tabu_tenure, false);
            std::cout << "[CG] Local Search warm start objective: " << ls_sln.objective << "\n";
            std::vector<QuadKey> ls_cols;
            for (int i=0;i<n;++i) {
                int u = ls_sln.assignment[i];
                for (int j=0;j<n;++j) {
                    if (i==j) continue;
                    int v = ls_sln.assignment[j];
                    ls_cols.push_back(IncrementalRMP::canonical(i,u,j,v));
                }
            }
            rmp.addColumns(ls_cols);
            if (rmp.omega().size() >= n_rows*0.8) break;
        }
        std::cout << "[CG] Added columns from local search warm start, omega is now "
                  << rmp.omega().size() << ", solving...\n";
        if (cfg.log_output) {
            auto t0 = std::chrono::high_resolution_clock::now();
            solving_status = rmp.solve();
            solve_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
        } else {
            solving_status = rmp.solve();
        }
        if (!solving_status) {
            std::cerr << "[CG] CPLEX failed to solve even after warm start, aborting\n";
            return Solution(instance_path, "column_generation_cplex");
        }
    }
    
    // Pre-define add_columns lambda outside loop to avoid recreation
    int cols_added_count = 0;
    std::function<bool(PricingEngine::Result&)> add_columns;
    if (cfg.add_column_strategy == "add_all_negative" ||
        cfg.add_column_strategy == "add_most_negative"){
        add_columns = [&](PricingEngine::Result& pr){
            if (pr.negative_cols.empty()) return false;
            rmp.addColumns(pr.negative_cols);
            cols_added_count = pr.negative_cols.size();
            return true;
        };
    }
    else if (cfg.add_column_strategy.find("add_k_negative") != std::string::npos){
        std::cout << cfg.add_column_strategy.substr(15) << std::endl;
        auto k = std::stoi(cfg.add_column_strategy.substr(15));
        std::cout << "k = " << k << std::endl;
        add_columns = [&](PricingEngine::Result& pr){
            if (pr.negative_cols.empty()) return false;
            std::vector<QuadKey> cols_to_add;
            cols_to_add.reserve(std::min(k, int(pr.negative_cols.size())));
            // randomly pick k cols
            // shuffle pr.negative_cols
            std::random_device rd;
            std::default_random_engine rng(rd());

            std::shuffle(pr.negative_cols.begin(), pr.negative_cols.end(), rng);

            for (int i=0; i<std::min(k, int(pr.negative_cols.size())); ++i) {
                cols_to_add.push_back(pr.negative_cols[i]);
            }
            rmp.addColumns(cols_to_add);
            cols_added_count = cols_to_add.size();
            return true;
        };
    }
    
    PricingEngine::Result priceRes;
    double z_k = rmp.getObjectiveValue();
    double realobj_k = rmp.getRealObjectiveValue();
    while (iter < cfg.max_iterations) {
        auto iter_t0 = std::chrono::high_resolution_clock::now();
        ++iter;

        // double z_k = rmp.getObjectiveValue();
        if (cfg.log_output) {
            std::cout << "========================================\n";
            std::cout << "Iter " << iter << " obj=" << z_k << " realobj=" << realobj_k << " time=" << solve_time << ", number of columns=" << rmp.omega().size() << "\n";
        }

        // Get current objective and duals (from previous solve)
        if (cfg.use_dual_box_restriction) {
            // if (iter == 1) {
            //     rmp.addAllTplusTminus(); // add all t_plus and t_minus variables for dual box restriction (for stabilization)
            // }
            {
                auto compute_delta = [&](const DenseDuals& hat_pi) {
                    for (size_t k = 0; k < delta.C1.size(); ++k)
                        delta.C1[k] = gamma * std::max(1.0, std::abs(hat_pi.C1[k]));

                    for (size_t k = 0; k < delta.C2.size(); ++k)
                        delta.C2[k] = gamma * std::max(1.0, std::abs(hat_pi.C2[k]));

                    for (size_t k = 0; k < delta.C3.size(); ++k)
                        delta.C3[k] = gamma * std::max(1.0, std::abs(hat_pi.C3[k]));

                    for (size_t k = 0; k < delta.C4.size(); ++k)
                        delta.C4[k] = gamma * std::max(1.0, std::abs(hat_pi.C4[k]));

                    for (size_t k = 0; k < delta.C5.size(); ++k)
                        delta.C5[k] = gamma * std::max(1.0, std::abs(hat_pi.C5[k]));
                };

                // Update box center
                if (!have_hat_pi) {
                    hat_pi = cur_du;
                    have_hat_pi = true;
                    std::cout << "[CG] Initialized dual box center with current duals\n";
                } else {
                    hat_pi.emaUpdate(cur_du, alpha);
                }
                // Compute δ from hat_pi
                compute_delta(hat_pi);

                // Update tp/tm objective coefficients
                rmp.updateStabilizationCoefficients(hat_pi, delta);
            }
        }
        
        // Solve with newly added columns
        double solve_time = 0.0;
        bool solving_status;
        if (cfg.log_output) {
            auto t2 = std::chrono::high_resolution_clock::now();
            solving_status = rmp.solve();
            solve_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t2).count();
            std::cout << "[CG] Solving after adding columns took " << solve_time
                      << " s, status = " << rmp.cplex().getStatus() << "\n";
        } else {
            solving_status = rmp.solve();
        }
        // rmp.checkLastAddedColumnBasis();
            
        
        if (!solving_status) {
            std::cerr << "[CG] Solve after adding columns failed at iter " << iter << "\n";
            break;
        }

        double z_k1 = rmp.getObjectiveValue();
        realobj_k = rmp.getRealObjectiveValue();

        // Adaptive lambda update (when columns came from perturbed pricing)
        if (cfg.stabilize) {
            bool from_perturbed = (priceRes.best_col.has_value() && priceRes.best_rc < -cfg.eps);
            if (from_perturbed) {
                bool progress;
                if (cfg.use_relative_improve) {
                    double rel = (z_k - z_k1) / std::max(1.0, std::abs(z_k));
                    progress = (rel > cfg.improve_rel_eps);
                } else {
                    progress = (z_k1 < z_k - cfg.eps);
                }
                if (progress) {
                    lambda = std::min(cfg.lambda_max, lambda + cfg.lambda_step_up);
                } else {
                    lambda = std::max(cfg.lambda_min, lambda * cfg.lambda_step_down);
                }
            }
            // Update \tilde{\pi}^{k+1} = lambda * \pi^{k+1} + (1 - lambda) * \tilde{\pi}^k
            double update_time = 0.0;
            auto t0 = cfg.log_output ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};
            pi_k1.captureFrom(rmp);
            tilde_du.emaUpdate(pi_k1, lambda);
            if (cfg.log_output) {
                update_time = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                std::cout << "[CG] Updated stabilized duals took " << update_time << " s, lambda = " << lambda << "\n";
            }
        }

        // Capture current duals
        cur_du.captureFrom(rmp);           // \pi^k
        // if (iter > 1) {
        //     // compare with last iteration's duals
        //     int count_diff = 0;
        //     double max_diff = 0.0;
        //     int count_diff_sign = 0;
        //     int count_zero_to_nonzero = 0;
        //     int count_nonzero_to_zero = 0;
        //     for (size_t k=0; k<cur_du.C1.size(); ++k) {
        //         auto diff = std::abs(cur_du.C1[k] - last_du.C1[k]);
        //         if (diff > cfg.eps) ++count_diff;
        //         if (diff > max_diff) max_diff = diff;
        //         if ((cur_du.C1[k] > 0 && last_du.C1[k] < 0) || (cur_du.C1[k] < 0 && last_du.C1[k] > 0)) ++count_diff_sign;
        //         if (std::abs(last_du.C1[k]) <= cfg.eps && std::abs(cur_du.C1[k]) > cfg.eps) ++count_zero_to_nonzero;
        //         if (std::abs(last_du.C1[k]) > cfg.eps && std::abs(cur_du.C1[k]) <= cfg.eps) ++count_nonzero_to_zero;
        //     }
        //     for (size_t k=0; k<cur_du.C2.size(); ++k) {
        //         auto diff = std::abs(cur_du.C2[k] - last_du.C2[k]);
        //         if (diff > cfg.eps) ++count_diff;
        //         if (diff > max_diff) max_diff = diff;
        //         if ((cur_du.C2[k] > 0 && last_du.C2[k] < 0) || (cur_du.C2[k] < 0 && last_du.C2[k] > 0)) ++count_diff_sign;
        //         if (std::abs(last_du.C2[k]) <= cfg.eps && std::abs(cur_du.C2[k]) > cfg.eps) ++count_zero_to_nonzero;
        //         if (std::abs(last_du.C2[k]) > cfg.eps && std::abs(cur_du.C2[k]) <= cfg.eps) ++count_nonzero_to_zero;
        //     }
        //     for (size_t k=0; k<cur_du.C3.size(); ++k) {
        //         auto diff = std::abs(cur_du.C3[k] - last_du.C3[k]);
        //         if (diff > cfg.eps) ++count_diff;
        //         if (diff > max_diff) max_diff = diff;
        //         if ((cur_du.C3[k] > 0 && last_du.C3[k] < 0) || (cur_du.C3[k] < 0 && last_du.C3[k] > 0)) ++count_diff_sign;
        //         if (std::abs(last_du.C3[k]) <= cfg.eps && std::abs(cur_du.C3[k]) > cfg.eps) ++count_zero_to_nonzero;
        //         if (std::abs(last_du.C3[k]) > cfg.eps && std::abs(cur_du.C3[k]) <= cfg.eps) ++count_nonzero_to_zero;
        //     }
        //     for (size_t k=0; k<cur_du.C4.size(); ++k) {
        //         auto diff = std::abs(cur_du.C4[k] - last_du.C4[k]);
        //         if (diff > cfg.eps) ++count_diff;
        //         if (diff > max_diff) max_diff = diff;
        //         if ((cur_du.C4[k] > 0 && last_du.C4[k] < 0) || (cur_du.C4[k] < 0 && last_du.C4[k] > 0)) ++count_diff_sign;
        //         if (std::abs(last_du.C4[k]) <= cfg.eps && std::abs(cur_du.C4[k]) > cfg.eps) ++count_zero_to_nonzero;
        //         if (std::abs(last_du.C4[k]) > cfg.eps && std::abs(cur_du.C4[k]) <= cfg.eps) ++count_nonzero_to_zero;
        //     }
        //     for (size_t k=0; k<cur_du.C5.size(); ++k) {
        //         auto diff = std::abs(cur_du.C5[k] - last_du.C5[k]);
        //         if (diff > cfg.eps) ++count_diff;
        //         if (diff > max_diff) max_diff = diff;
        //         if ((cur_du.C5[k] > 0 && last_du.C5[k] < 0) || (cur_du.C5[k] < 0 && last_du.C5[k] > 0)) ++count_diff_sign;
        //         if (std::abs(last_du.C5[k]) <= cfg.eps && std::abs(cur_du.C5[k]) > cfg.eps) ++count_zero_to_nonzero;
        //         if (std::abs(last_du.C5[k]) > cfg.eps && std::abs(cur_du.C5[k]) <= cfg.eps) ++count_nonzero_to_zero;
        //     }
        //     if (cfg.log_output) {
        //         std::cout << "[CG] Duals changed in " << count_diff << " components compared to last iteration\n";
        //         std::cout << "[CG] Maximum dual difference = " << max_diff << "\n";
        //         std::cout << "[CG] Number of dual sign changes = " << count_diff_sign << "\n";
        //         std::cout << "[CG] Number of duals changed from zero to nonzero = " << count_zero_to_nonzero << "\n";
        //         std::cout << "[CG] Number of duals changed from nonzero to zero = " << count_nonzero_to_zero << "\n";
        //     }
        // }
        // last_du.captureFrom(rmp);

        bool added = false;
        if (cfg.stabilize) {
            if (!have_tilde) { tilde_du = cur_du; have_tilde = true; } // \tilde{\pi}^0 := \pi^0
            PerturbedRMPView view(rmp, tilde_du);
            if (cfg.log_output) {
                auto t1 = std::chrono::high_resolution_clock::now();
                priceRes = pricing.price(view);
                double ptime = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t1).count();
                std::cout << "[CG] Pricing stabilized took " << ptime
                          << " s, best reduced cost = " << priceRes.best_rc << "\n";
            } else {
                priceRes = pricing.price(view);
            }
        } else {
            // legacy behavior
            if (cfg.log_output) {
                auto t1 = std::chrono::high_resolution_clock::now();
                priceRes = pricing.price(rmp);
                double ptime = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t1).count();
                std::cout << "[CG] Pricing (legacy) took " << ptime
                          << " s, best reduced cost = " << priceRes.best_rc << "\n";
            } else {
                priceRes = pricing.price(rmp);
            }
        }

        if (cfg.stabilize) {
            // First try with perturbed duals
            if (cfg.log_output) {
                auto t1 = std::chrono::high_resolution_clock::now();
                added = add_columns(priceRes);
                double add_time = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t1).count();
                std::cout << "[CG] Adding columns from perturbed pricing took " << add_time
                          << " s, added " << cols_added_count << " columns\n";
            } else {
                added = add_columns(priceRes);
            }
            if (!added) {
                // 3) Verification step with true duals
                PricingEngine::Result verRes;
                if (cfg.log_output) {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    verRes = pricing.price(rmp);
                    double ptime = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t1).count();
                    std::cout << "[CG] Pricing(true) verification took " << ptime
                              << " s, best reduced cost = " << verRes.best_rc << "\n";
                    t1 = std::chrono::high_resolution_clock::now();
                    added = add_columns(verRes);
                    double add_time = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t1).count();
                    std::cout << "[CG] Adding columns from true-dual verification took " << add_time
                              << " s, added " << cols_added_count << " columns\n";
                } else {
                    verRes = pricing.price(rmp);
                    added = add_columns(verRes);
                }
                if (!added) {
                    // Certified optimality (no negative RC with true duals)
                    break;
                } else {
                    // On columns from true-dual check: increase lambda
                    lambda = std::min(cfg.lambda_max, lambda + cfg.lambda_step_up);
                }
            }
        } else {
            // No stabilization; legacy termination rule
            if (cfg.log_output) {
                auto t1 = std::chrono::high_resolution_clock::now();
                added = add_columns(priceRes);
                double add_time = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t1).count();
                if (added) {
                    std::cout << "[CG] Adding columns took " << add_time
                            << " s, added " << cols_added_count << " columns\n";
                }
            } else {
                added = add_columns(priceRes);
            }
            if (!added && iter > 1) {
                std::cout << "[CG] No columns added in iteration " << iter
                          << " and not the first iteration, terminating.\n";
                break;
            }
        }


        if (cfg.log_output) {
            // std::cout << "[CG] Re-solved: z_k=" << z_k << " -> z_k+1=" << z_k1
            //           << " | lambda=" << lambda << "\n";
            double iter_wall = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - iter_t0).count();
            std::cout << "[CG] Iteration " << iter << " total wall time = " << iter_wall << " s\n";
        }
        z_k = z_k1;
        // rmp.checkBasicStatusChange(); // debug: analyze basis status change after adding columns
    } // while

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    
    auto xvals = rmp.getXValues();
    auto yvals = rmp.getYValues();
    rmp.verifySolution(yvals, xvals, fixed_variables); // sanity check
    auto assign = argmaxAssignment(n, m, xvals);

    rmp.columnsAnalysis();

    // save all positive columns to a file for analysis
    std::ofstream colfile(instance_path + "_positive_columns.txt");
    std::vector<QuadKey> positive_columns;
    for (const auto& kv : yvals) {
        if (kv.second > 1e-6) {
            positive_columns.push_back(kv.first);
        }
    }
    for (const auto& col : positive_columns) {
        colfile << col.i << " " << col.u << " " << col.j << " " << col.v << "\n";
    }
    colfile.close();


    Solution sln(instance_path, "column_generation_cplex");
    sln.assignment = assign;
    sln.objective = rmp.getObjectiveValue();
    sln.time = elapsed;
    sln.lower_bound = rmp.getObjectiveValue();
    return sln;
}