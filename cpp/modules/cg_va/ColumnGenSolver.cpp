#include "ColumnGenSolver.h"
#include "CGVolumeHooks.h"
#include "../local_search/local_search.h"
#include <iostream>
#include <unordered_set>
#include <chrono>

ColumnGenSolver::ColumnGenSolver(const ColumnGenConfig& cfg_) : cfg(cfg_) {}

Solution ColumnGenSolver::solve(const Problem& P,
    const std::string& instance_path,
    const std::unordered_map<int,int>& fixed)
{
    int n = P.n;
    int nx = n * n;

    std::unordered_map<QuadKey,int,QuadKeyHash> y_index;
    int ny = 0;

    auto add_column = [&](QuadKey q){ 
        auto canonical_q = canonicalize(q);
        // check problem's fixed assignments
        // - If i is fixed to u' != u, skip
        // - If j is fixed to v' != v, skip
        if (fixed.count(canonical_q.i) && fixed.at(canonical_q.i) != canonical_q.u) return;
        if (fixed.count(canonical_q.j) && fixed.at(canonical_q.j) != canonical_q.v) return;
        
        if ((canonical_q.i == canonical_q.j && canonical_q.u != canonical_q.v) ||
            (canonical_q.i != canonical_q.j && canonical_q.u == canonical_q.v)) {
            std::cerr << "Warning: skipping invalid diagonal column (" << q.i << "," << q.u << "," << q.j << "," << q.v << ")\n";
            return;
        }
        if (!y_index.count(canonical_q)) {
            y_index[canonical_q] = ny++;
        }
    };

    if (cfg.full_columns_at_once) {
        // Initialize with all canonical columns in Ω (full-column mode)
        // - Diagonal: (i,i,u,u)
        // - Cross canonical: i < j and u != v
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                add_column(QuadKey{i, u, i, u});
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        if (u == v) continue;
                        add_column(QuadKey{i, u, j, v});
                    }
                }
            }
        }
    } else {
        // add initial columns based on nonzero F entries (sparse-column mode)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (P.D[i][j] == 0.0) continue; // skip zero-distance pairs
                for (int u = 0; u < n; ++u) {
                    if (P.F[u][u] == 0.0) continue; // skip zero-flow self-assignments
                    add_column(QuadKey{i, u, i, u});
                    for (int v = 0; v < n; ++v) {
                        if (u == v) continue;
                        if (P.F[u][v] == 0.0) continue; // skip zero-flow pairs
                        add_column(QuadKey{i, u, j, v});
                    }
                }
            }
        }
        // if have some fixed assignments, also add corresponding columns to cover them
        for (const auto& kv : fixed) {
            int loc = kv.first;
            int fac = kv.second;
            add_column(QuadKey{loc, fac, loc, fac});
        }

        LocalSearch ls;
        Solution ls_sln(instance_path, "local_search");
        ls_sln.assignment = LocalSearch::initAssignment(n, "random", "");
        auto warmup_max_iterations = 500;
        auto warmup_tabu_tenure = 10;
        ls.tabuSearch(P, ls_sln, warmup_max_iterations, warmup_tabu_tenure, false);
        if (cfg.log_output) {
            std::cout << "[CG-VA] Local Search warmup objective: " << ls_sln.objective << std::endl;
        }

        // Extract columns from warmup solution
        std::vector<QuadKey> warmup_cols;
        for (int u = 0; u < n; ++u) {
            int i = ls_sln.assignment[u];
            for (int v = 0; v < n; ++v) {
                int j = ls_sln.assignment[v];
                if ((i == j && u != v) || (i != j && u == v)) continue;
                warmup_cols.push_back(QuadKey{i, u, j, v});
            }
        }
        // =============================
        // Local Search Warmup
        // =============================
        if (cfg.log_output) {
            std::cout << "[CG-VA] Starting local search warmup..." << std::endl;
        }
        
        // Add warmup columns to Omega
        for (const auto& q : warmup_cols) {
            add_column(q);
        }

        if (cfg.log_output) {
            std::cout << "[CG-VA] Added " << warmup_cols.size() << " columns from warmup, |Omega|=" << ny << std::endl;
        }

    }

    if (cfg.log_output) {
        std::cout << "[CG-VA] full_columns_at_once="
                  << (cfg.full_columns_at_once ? "true" : "false")
                  << ", |Omega|=" << ny << std::endl;
    }


    auto objective_from_x = [&](const VOL_dvector &xvec) -> double {
        std::vector<int> assign(n, -1);
        std::vector<char> used(n, 0);

        for (int i = 0; i < n; ++i) {
            int best_u = -1;
            double best_val = -1e100;
            for (int u = 0; u < n; ++u) {
                if (used[u]) continue;
                double val = xvec[i * n + u];
                if (val > best_val) {
                    best_val = val;
                    best_u = u;
                }
            }
            if (best_u < 0) {
                for (int u = 0; u < n; ++u) {
                    if (!used[u]) {
                        best_u = u;
                        break;
                    }
                }
            }
            assign[i] = best_u;
            used[best_u] = 1;
        }

        double obj = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                obj += P.D[i][j] * P.F[assign[i]][assign[j]];
            }
        }
        return obj;
    };

    bool done = false;
    double best_obj = 1e100;
    double best_lb = -1e100;
    auto start = std::chrono::high_resolution_clock::now();

    const int dsize = n*n + n*n + n*n + n*n + n*n*n + n*n*n;
    VOL_problem prob;
    prob.dsize = dsize;
    prob.psize = n*n + n*n*n*n; // primal variables correspond to x_iu
    prob.dual_lb.allocate(dsize);
    prob.dual_ub.allocate(dsize);
    prob.dual_lb = -1e12;
    prob.dual_ub = 1e12;
    prob.dsol.allocate(dsize);
    prob.dsol = 0.0; // start with zero duals   


    // Volume parameters from config
    prob.parm.lambdainit = 0.1;
    prob.parm.alphainit = 0.01;
    prob.parm.alphamin = 0.0001;
    prob.parm.alphafactor = 0.66;
    prob.parm.alphaint = 50;
    
    prob.parm.maxsgriters = 100000000; // effectively no limit
    prob.parm.primal_abs_precision = 0.001;
    prob.parm.gap_abs_precision = 0.0;
    prob.parm.gap_rel_precision = 0.001;
    prob.parm.granularity = 0.0;
    
    prob.parm.ascent_first_check = 500;
    prob.parm.ascent_check_invl = 500;
    prob.parm.minimum_rel_ascent = 0.0001;
    
    prob.parm.greentestinvl = 1;
    prob.parm.yellowtestinvl = 4;
    prob.parm.redtestinvl = 20;

    prob.parm.maxsgriters = std::max(1, cfg.max_iterations);
    prob.parm.printflag = cfg.full_columns_at_once ? 3 : 0;
    prob.parm.printinvl = 50;
    VOL_dvector prev_dsol;  // Keep previous dual solution for warm-starting

    int cg_iteration = 0;
    while(!done){
        // Create fresh VOL_problem each iteration to avoid indexing issues with growing psize
        
        // Warm-start with previous dual if available
        // if (cg_iteration > 0) {
        //     prob.dsol = prev_dsol;
        // } else {
        //     prob.dsol = 0.0;
        // }

        
        std::cout << "===============================" << std::endl;

        CGVolumeHooks hooks(P, y_index);
        bool use_warmstart = (cg_iteration > 0);  // Warm-start after first iteration
        int ret = prob.solve(hooks, use_warmstart);
        
        // Save dual solution for next iteration
        // prev_dsol = prob.dsol;
        cg_iteration++;
        if (ret < 0) {
            break;
        }

        if (cfg.log_output) {
            std::cout << "[CG-VA] VA value=" << prob.value << std::endl;
        }
        auto &pi = prob.dsol;

        int off_alpha = 0;
        int off_beta  = off_alpha + n*n;
        int off_gamma = off_beta  + n*n;
        int off_delta = off_gamma + n*n;
        int off_eta   = off_delta + n*n;
        int off_mu    = off_eta   + n*n*n;

        auto ALPHA = [=](int i,int j){ return pi[off_alpha + i*n + j]; };
        auto BETA  = [=](int u,int v){ return pi[off_beta  + u*n + v]; };
        auto GAMMA = [=](int u,int j){ return pi[off_gamma + u*n + j]; };
        auto DELTA = [=](int i,int v){ return pi[off_delta + i*n + v]; };
        auto ETA   = [=](int i,int u,int j){ return pi[off_eta + (i*n+u)*n + j]; };
        auto MU    = [=](int i,int u,int v){ return pi[off_mu + (i*n+u)*n + v]; };

        std::vector<QuadKey> new_cols;
        std::vector<QuadKey> best_cols;
        double bestC = 1e100;

        if (cfg.full_columns_at_once) {
            done = true;
        }

        // For all possible y candidates that is not in Omega, compute reduced cost and add if negative
        if (!cfg.full_columns_at_once) {
            for(int i=0;i<n;i++){
                for(int u=0;u<n;u++){
                    for(int j=i;j<n;j++){
                        for(int v=0;v<n;v++){

                        // diagonal
                        if(i == j && u == v){
                            QuadKey q{i,u,j,v};
                            if(y_index.count(q)) continue; // in Ω
                        }
                        // invalid
                        else if( (i==j && u!=v) || (i!=j && u==v) ){
                            continue;
                        }
                        QuadKey q = canonicalize(i,u,j,v);
                        if(y_index.count(q)) continue; // skip existing

                        // compute phi exactly once or twice
                        double phi = P.D[q.i][q.j]*P.F[q.u][q.v];
                        if(!(q.i==q.j && q.u==q.v))  // cross term
                            phi += P.D[q.j][q.i]*P.F[q.v][q.u];

                        // Reduced cost must match CGVolumeHooks: include both orientations
                        double C =
                            phi
                            - ALPHA(q.i,q.j) - ALPHA(q.j,q.i)
                            - BETA (q.u,q.v) - BETA (q.v,q.u)
                            - GAMMA(q.u,q.j) - GAMMA(q.v,q.i)
                            - DELTA(q.i,q.v) - DELTA(q.j,q.u)
                            - ETA  (q.i,q.u,q.j) - ETA  (q.j,q.v,q.i)
                            - MU   (q.i,q.u,q.v) - MU   (q.j,q.v,q.u);

                            if(C < -cfg.eps)
                                new_cols.push_back(q);
                            if(C < bestC){
                                bestC = C;
                                best_cols.clear();
                                best_cols.push_back(q);
                            } else if (C < bestC + cfg.eps) {
                                best_cols.push_back(q);
                            }
                        }
                    }
                }
            }
        }

        if(new_cols.empty()){
            done = true;
        } else {
            if(cfg.add_most_negative){
                for (const auto& q : best_cols) add_column(q);
                std::cout << "[CG-VA] Added " << best_cols.size() << " most negative columns, C=" << bestC << ", |Omega|=" << ny << std::endl;
            } else {
                // add at most k=100 most negative columns to control growth of Omega
                // random shuffle to avoid bias towards certain patterns, then take top k
                std::shuffle(new_cols.begin(), new_cols.end(), std::mt19937{std::random_device{}()});
                if (new_cols.size() > cfg.k) {
                    new_cols.resize(cfg.k);
                }
                for(const auto &q : new_cols) add_column(q);
                std::cout << "[CG-VA] Added " << new_cols.size() << " columns, best C=" << bestC << ", |Omega|=" << ny << std::endl;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();

    Solution sln(instance_path, "column_generation_cplex");

    sln.objective = prob.value;
    sln.time = elapsed;
    return sln;
}