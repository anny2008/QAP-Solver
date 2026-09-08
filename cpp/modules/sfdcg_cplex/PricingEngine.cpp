#include "PricingEngine.h"
#include <omp.h>

/**
 * Compute reduce cost for all e^k_ij
 * rc^k_ij = F_k * d_ij - lambda_ki - theta_kj
 */
PricingEngine::Result PricingEngine::price(const IncrementalRMP& rmp) {

    PricingEngine::Result result;

    double& best_rc = result.best_rc;
    std::vector<TripleKey>& negative_cols = result.negative_cols;
    std::vector<TripleKey>& most_negative_cols = result.most_negative_cols;

    auto vecresult = one_negative_pricing(rmp);
    best_rc = vecresult[0].rc;
    if(best_rc < -eps)
    {
        for (const auto& rc : vecresult) {
            negative_cols.emplace_back(rc.key);
            most_negative_cols.emplace_back(rc.key);
        }
    } else {
        // switch to sorting pricing if no negative column found, to find more negative columns and get better convergence
        std::cout << "No negative column found with one_negative_pricing, switching to sorting_pricing to find more negative columns..." << std::endl;
        vecresult = all_negative_pricing(rmp);
        // sorting by reduced cost
        std::sort(vecresult.begin(), vecresult.end());
        best_rc = vecresult[0].rc;
        
        auto threshold = best_rc*0.9; // add a small epsilon to include columns with rc very close to best_rc
        for (const auto& rc : vecresult) {
            negative_cols.emplace_back(rc.key);
            if (rc.rc <= threshold) {
                most_negative_cols.emplace_back(rc.key);
            } else {
                break; // since sorted, we can stop once we are above the threshold
            }
        }
        
    }
    return result;
    // return sorting_pricing(rmp);
}
/**
 * Compute reduce cost for all e^k_ij
 * rc^k_ij = F_k * d_ij - lambda_ki - theta_kj
 */
std::vector<PricingEngine::ReducedCost> PricingEngine::one_negative_pricing(const IncrementalRMP& rmp) {
    auto n = rmp.problem.n;
    auto K = rmp.subgraphs.size();
    auto subgraphs = rmp.subgraphs;
    auto D = rmp.problem.D;
    std::vector<ReducedCost> reduced_costs;

    std::vector<double> duals;
    rmp.getDuals(duals);
    // find at most one column for each subgraph
    // create a variable to track if we already found a negative column for each subgraph
    std::vector<int> found_negative_for_subgraph(K, 0);
    for (auto key: rmp.not_in_omega)
    {        
        auto k = key.a;
        if (found_negative_for_subgraph[k] > rmp.nMaxColsPerSubgraph) continue; // if we already found a negative column for this subgraph, skip
        auto i = key.b;
        auto j = key.c;
        auto lambda_ki = duals[k*n + i];
        auto theta_kj = duals[k*n + j + n*K];
        auto rc = subgraphs[k].F_k*D[i][j] - lambda_ki - theta_kj;
        if (rc < -eps)
        {
            reduced_costs.emplace_back(key, rc);
            found_negative_for_subgraph[k]++;
            if (reduced_costs.size() >= K*rmp.nMaxColsPerSubgraph) break; // if we already found a negative column for each subgraph, we can stop
        }
    }
    return reduced_costs;

}
/**
 * Compute reduce cost for all e^k_ij
 * rc^k_ij = F_k * d_ij - lambda_ki - theta_kj
 */
std::vector<PricingEngine::ReducedCost> PricingEngine::all_pricing(const IncrementalRMP& rmp) {
    auto n = rmp.problem.n;
    auto K = rmp.subgraphs.size();
    auto subgraphs = rmp.subgraphs;
    auto D = rmp.problem.D;

    std::vector<double> duals;
    rmp.getDuals(duals);
    // compute reduced cost for all (k,i,j) and save the negative ones
    std::vector<ReducedCost> reduced_costs;
    std::vector<double> rcs;
    rcs.resize(K*n*n, 0.0);
    // parallelize the loop with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (auto key:rmp.not_in_omega)
    {
        auto k = key.a;
        auto i = key.b;
        auto j = key.c;
        auto subgraph = subgraphs[k];
            continue;
        auto lambda_ki = duals[k*n + i];
        auto theta_kj = duals[k*n + j + n*K];
        auto rc = subgraph.F_k*D[i][j] - lambda_ki - theta_kj;
        rcs[k*n*n + i*n + j] = rc;
    }
    for (auto key:rmp.not_in_omega)
    {        
        auto k = key.a;
        auto i = key.b;
        auto j = key.c;
        auto subgraph = subgraphs[k];
        auto rc = rcs[k*n*n + i*n + j];
        reduced_costs.emplace_back(k,i,j, rc);
    }
    return reduced_costs;
}


std::vector<PricingEngine::ReducedCost> PricingEngine::all_negative_pricing(const IncrementalRMP& rmp) {
    auto n = rmp.problem.n;
    auto K = rmp.subgraphs.size();
    auto subgraphs = rmp.subgraphs;
    auto D = rmp.problem.D;

    std::vector<double> duals;
    rmp.getDuals(duals);
    // compute reduced cost for all (k,i,j) and save the negative ones
    std::vector<ReducedCost> reduced_costs;
    std::vector<double> rcs;
    rcs.resize(K*n*n, 0.0);
    // parallelize the loop with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (auto key:rmp.not_in_omega)
    {
        auto k = key.a;
        auto i = key.b;
        auto j = key.c;
        auto subgraph = subgraphs[k];
            continue;
        auto lambda_ki = duals[k*n + i];
        auto theta_kj = duals[k*n + j + n*K];
        auto rc = subgraph.F_k*D[i][j] - lambda_ki - theta_kj;
        rcs[k*n*n + i*n + j] = rc;
    }
    for (auto key:rmp.not_in_omega)
    {        
        auto k = key.a;
        auto i = key.b;
        auto j = key.c;
        auto subgraph = subgraphs[k];
        auto rc = rcs[k*n*n + i*n + j];
        if (rc < -eps)
        {
            reduced_costs.emplace_back(k,i,j, rc);
        }
    }
    return reduced_costs;
}