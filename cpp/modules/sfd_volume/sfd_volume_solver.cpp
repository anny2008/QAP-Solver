/**
 * SFD Volume Algorithm Solver for QAP (Form3)
 * Implements Lagrangian relaxation-based Volume algorithm for SFD formulation.
 * Based on sfd_solver.cpp and qap_form3.cpp
 */

#include "VolVolume.hpp"
#include "../../core/problem.h"
#include "../../include/qap_solution_io.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <omp.h>
#include <ilcplex/ilocplex.h>

// Macro definitions for variable access
#define x(i, u) (psol[(i) * n + (u)])
#define e(i, j, k) (psol[n * n + (i) * n * n + (j) * n + (k)])
#define v_x(i, u) (var_x[(i) * n + (u)])
#define v_e(i, j, k) (var_e[(i) * n * n + (j) * n + (k)])
#define vio_lambda(u) (vio[u])
#define lambda(u) (pi[u])
#define Beta(i, u) (beta[(i) * n + (u)])
#define Beta_ind(i, u) (beta_j_ind[(i) * n + (u)])

// SFD subgraph structure
struct Subgraph
{
    double f_k;
    std::vector<std::pair<int, int>> edge_set;
    std::set<int> node_set;
    std::unordered_map<int, int> degree_out;
    std::unordered_map<int, int> degree_in;
};

// Decompose flow matrix by value_layer
std::vector<Subgraph> decompose_flow_by_value_layer(const Problem &qap)
{
    int n = qap.n;
    std::vector<std::vector<double>> flows = qap.F;
    std::vector<Subgraph> subgraphs;
    while (true)
    {
        double min_flow = 0.0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (flows[i][j] > 1e-9)
                {
                    if (min_flow < 1e-9 || flows[i][j] < min_flow)
                    {
                        min_flow = flows[i][j];
                    }
                }
            }
        }
        if (min_flow <= 0)
            break;

        Subgraph sg;
        sg.f_k = min_flow;
        for (int u = 0; u < n; ++u)
        {
            for (int v = 0; v < n; ++v)
            {
                if (flows[u][v] >= min_flow - 1e-9)
                {
                    sg.edge_set.push_back({u, v});
                    sg.node_set.insert(u);
                    sg.node_set.insert(v);
                }
            }
        }
        for (const auto &edge : sg.edge_set)
        {
            sg.degree_out[edge.first]++;
            sg.degree_in[edge.second]++;
        }
        subgraphs.push_back(sg);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (flows[i][j] >= min_flow - 1e-9)
                {
                    flows[i][j] -= min_flow;
                    if (flows[i][j] < 1e-9)
                        flows[i][j] = 0.0;
                }
            }
        }
    }
    return subgraphs;
}

// Additional decomposition strategy: VALUE_ONLY
std::vector<Subgraph> decompose_flow_by_value_only(const Problem &qap)
{
    int n = qap.n;
    std::unordered_map<double, Subgraph> flow_map;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double flow = qap.F[i][j];
            if (flow > 1e-9)
            {
                if (flow_map.find(flow) == flow_map.end())
                {
                    flow_map[flow] = Subgraph{flow};
                }
                flow_map[flow].edge_set.push_back({i, j});
                flow_map[flow].node_set.insert(i);
                flow_map[flow].node_set.insert(j);
                flow_map[flow].degree_out[i]++;
                flow_map[flow].degree_in[j]++;
            }
        }
    }
    std::vector<Subgraph> subgraphs;
    for (auto &kv : flow_map)
    {
        subgraphs.push_back(kv.second);
    }
    return subgraphs;
}

// SFD Volume Hooks for Volume Algorithm
class SFDVolumeHooks : public VOL_user_hooks
{
private:
    const Problem &qap_data;
    int n;
    std::vector<Subgraph> subgraphs;
    std::vector<float> beta;
    std::vector<int> beta_j_ind;
    std::vector<float> alpha;
    std::vector<int> alpha_u_ind;

public:
    SFDVolumeHooks(const Problem &data, const std::vector<Subgraph> &subgs)
        : qap_data(data), n(data.n), subgraphs(subgs)
    {
        beta.resize(n * n, 0.0);
        beta_j_ind.resize(n * n, 0);
        alpha.resize(n, 0.0);
        alpha_u_ind.resize(n, 0);
    }

    // Reduced costs not used in SFD subproblem
    int compute_rc(const VOL_dvector & /*u*/, VOL_dvector &rc) override
    {
        rc = 0.0;
        return 0;
    }

    int solve_subproblem(const VOL_dvector &u, const VOL_dvector &rc,
                         double &lcost, VOL_dvector &x, VOL_dvector &v,
                         double &pcost) override
    {
        // Choose between row-only or column-only subproblem based on duals
        // Here we implement both and choose based on some criterion
        // For simplicity, we implement only column-only here
        return solve_subproblem_cplex(u, rc, lcost, x, v, pcost);
    }

    // SFD Lagrangian subproblem: assign facilities to locations, minimize cost
    int solve_subproblem_row(const VOL_dvector &dual, const VOL_dvector &rc, double &lcost, VOL_dvector &psol, VOL_dvector &vio, double &pcost)
    {
        auto n = qap_data.n;
        psol = 0.0;
        vio = 0.0;

        return 0;
    }

    int solve_subproblem_cplex(const VOL_dvector &dual, const VOL_dvector &rc, double &lcost, VOL_dvector &psol, VOL_dvector &vio, double &pcost)
    {
        auto n = qap_data.n;

        psol = 0.0;
        vio = 0.0;

        // Create CPLEX model for SFD subproblem
        // Decision variables: x[i][u] = 1 if location i assigned to facility u
        // Constraints: each location assigned to exactly one facility, and SFD constraints for each subgraph
        // Objective: minimize sum of assignment costs + dual penalties
        IloEnv env;
        IloModel model(env);
        IloNumVarArray x(env, n * n, 0, 1, ILOBOOL);
        // 0 <= e^k_ij <= 1 continuous variables for SFD constraints
        IloNumVarArray e(env, n * n * subgraphs.size(), 0, 1, ILOFLOAT);

        // Add constraints
        for (int i = 0; i < n; ++i)
        {
            IloExpr sum(env);
            for (int u = 0; u < n; ++u)
            {
                sum += x[i * n + u];
            }
            model.add(sum == 1);
        }

        // Add SFD constraints for each subgraph
        // e^k_ij >= x[i][u] + sum_{v | (u,v) \in G_k} x[j][v]  - 1 forall k, i,j
        for (const auto &subgraph : subgraphs)
        {
            // Implementation for SFD constraints
            for (auto i=0; i < n; ++i)
            {
                for (auto j=0; j < n; ++j)
                {
                    if (i != j)
                    {
                        IloExpr lhs(env);
                        lhs += e(i, j, &subgraph - &subgraphs[0]);
                        for (const auto &edge : subgraph.edge_set)
                        {
                            int u = edge.first;
                            int v = edge.second;
                            lhs += x(i, u) + x(j, v);
                        }
                        model.add(lhs >= 1);
                    }
                }
            }
        }

        // Set objective
        IloExpr obj(env);
        for (int i = 0; i < n; ++i)
        {
            for (int u = 0; u < n; ++u)
            {
                obj += qap_data.D[i][u] * x[i * n + u];
            }
        }
        model.add(IloMinimize(env, obj));

        // Solve
        IloCplex cplex(model);
        cplex.solve();

        // Extract solution
        for (int i = 0; i < n; ++i)
        {
            for (int u = 0; u < n; ++u)
            {
                psol[i * n + u] = cplex.getValue(x[i * n + u]);
            }
        }

        return 0;
    }

    int heuristics(const VOL_problem & /*p*/, const VOL_dvector & /*psol*/, double &heur_val) override
    {
        heur_val = DBL_MAX;
        return 0;
    }
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <instance.dat> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --config <file.json>     : JSON config with keys (instance, warmstart, decomposition, time_limit, threads, relax, log_output)" << std::endl;
        std::cout << "  --decomposition <type>   : value_layer (default) or value_only" << std::endl;
        std::cout << "  --time_limit <seconds>   : Time limit (default: 120)" << std::endl;
        std::cout << "  --threads <n>            : Number of threads (default: 8)" << std::endl;
        std::cout << "  --log_output             : Enable solver output" << std::endl;
        return 1;
    }

    // Parse CLI/config
    std::string instance_path = argv[1];
    std::string config_path = "";
    std::string decomposition = "value_layer";
    std::string relax_type = "facility"; // or "location"
    int time_limit = 120;
    int num_threads = 8;
    bool log_output = false;
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
        {
            config_path = argv[++i];
        }
        else if (arg == "--decomposition" && i + 1 < argc)
        {
            decomposition = argv[++i];
        }
        else if (arg == "--time_limit" && i + 1 < argc)
        {
            time_limit = std::atoi(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            num_threads = std::atoi(argv[++i]);
        }
        else if (arg == "--log_output")
        {
            log_output = true;
        }
    }

    // Load config if provided
    if (!config_path.empty())
    {
        std::ifstream cfg(config_path);
        if (cfg.is_open())
        {
            std::string json_str((std::istreambuf_iterator<char>(cfg)), std::istreambuf_iterator<char>());
            // Simple JSON parsing (assume flat keys)
            size_t pos = 0;
            while ((pos = json_str.find("\"instance\"")) != std::string::npos)
            {
                size_t start = json_str.find(':', pos) + 1;
                size_t end = json_str.find(',', start);
                instance_path = json_str.substr(start, end - start);
                instance_path.erase(std::remove(instance_path.begin(), instance_path.end(), '"'), instance_path.end());
                json_str.erase(pos, end - pos);
            }
        }
    }

    // Load QAP instance
    auto qap_data = Problem::fromQAPLIB(instance_path);
    int n = qap_data.n;
    std::cout << "SFD Volume Algorithm Solver for QAP (n=" << n << ")" << std::endl;

    // Decompose flow matrix
    std::vector<Subgraph> subgraphs;
    if (decomposition == "value_layer")
    {
        subgraphs = decompose_flow_by_value_layer(qap_data);
    }
    else if (decomposition == "value_only")
    {
        subgraphs = decompose_flow_by_value_only(qap_data);
    }
    else
    {
        std::cerr << "Unknown decomposition type: " << decomposition << std::endl;
        return 1;
    }
    std::cout << "Flow decomposed into " << subgraphs.size() << " subgraphs" << std::endl;

    // Set up Volume problem
    VOL_problem vol_problem;
    vol_problem.psize = n * n + n * n * n * subgraphs.size();
    if (relax_type == "facility")
    {
        vol_problem.dsize = n + n * n * n * subgraphs.size();
    }
    else if (relax_type == "location")
    {
        vol_problem.dsize = n;
    }
    else
    {
        std::cerr << "Unknown relaxation type: " << relax_type << std::endl;
        return 1;
    }
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;

    // Set Volume algorithm parameters
    vol_problem.parm.lambdainit = 0.1;
    vol_problem.parm.alphainit = 0.1;
    vol_problem.parm.alphamin = 0.001;
    vol_problem.parm.alphafactor = 0.66;
    vol_problem.parm.maxsgriters = 1000000;
    vol_problem.parm.primal_abs_precision = 0.001;
    vol_problem.parm.gap_abs_precision = 0.0;
    vol_problem.parm.gap_rel_precision = 0.001;
    vol_problem.parm.granularity = 0.0;
    vol_problem.parm.ascent_first_check = 500;
    vol_problem.parm.ascent_check_invl = 500;
    vol_problem.parm.minimum_rel_ascent = 0.0001;
    vol_problem.parm.greentestinvl = 1;
    vol_problem.parm.yellowtestinvl = 4;
    vol_problem.parm.redtestinvl = 20;
    vol_problem.parm.printflag = log_output ? 3 : 1;
    vol_problem.parm.printinvl = 100;
    vol_problem.parm.heurinvl = 100;

    // Create hooks
    SFDVolumeHooks hooks(qap_data, subgraphs);

    // Solve
    std::cout << "Starting Volume algorithm..." << std::endl;
    int retval = vol_problem.solve(hooks, false);

    std::cout << "Volume Algorithm Complete" << std::endl;
    std::cout << "Lower bound: " << vol_problem.value << std::endl;
    return retval;
}
