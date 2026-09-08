#pragma once
#include "../../core/problem.h"
#include "../../core/solution.h"
#include <optional>
#include <unordered_map>
#include "gurobi_c++.h"
#include <map>
#include <tuple>
#include <vector>
#include <set>
#include <string>

struct SFDGurobiConfig {
    bool log_output = false;
    bool is_relax = false;
    bool use_lazy_constraints = false;
    bool use_cuts = false;
    std::string lpmethod = "barrier";
    std::string warmstart = "";
    std::string matrix_for_decomposition = "flow"; // 'flow' or 'distance'
    std::string decomposition = "value_only";


    static SFDGurobiConfig fromJson(const json& j) {
        SFDGurobiConfig cfg;
        if (j.contains("log_output")) cfg.log_output = j["log_output"];
        if (j.contains("is_relax")) cfg.is_relax = j["is_relax"];
        if (j.contains("use_lazy_constraints")) cfg.use_lazy_constraints = j["use_lazy_constraints"];
        if (j.contains("use_cuts")) cfg.use_cuts = j["use_cuts"];
        if (j.contains("lpmethod")) cfg.lpmethod = j["lpmethod"];
        if (j.contains("warmstart")) cfg.warmstart = j["warmstart"];
        if (j.contains("matrix_for_decomposition")) cfg.matrix_for_decomposition = j["matrix_for_decomposition"];
        if (j.contains("decomposition")) cfg.decomposition = j["decomposition"];
        std::cout << "Config loaded: log_output=" << cfg.log_output
                  << ", is_relax=" << cfg.is_relax
                  << ", use_lazy_constraints=" << cfg.use_lazy_constraints
                  << ", matrix_for_decomposition=" << cfg.matrix_for_decomposition
                  << ", use_cuts=" << cfg.use_cuts
                  << ", lpmethod=" << cfg.lpmethod
                  << ", warmstart=" << cfg.warmstart
                  << std::endl;
        return cfg;
    }
};

struct SubgraphData {
    double f_k;
    std::vector<std::pair<int,int>> arcs;
    std::set<int> nodes;
};

class SFDGurobiSolver {
public:
    SFDGurobiSolver(const SFDGurobiConfig& cfg);

    // Build and solve for a Problem; fixed_variables is (location -> facility).
    Solution solve(Problem& problem,
                    const std::string& instance_path,
                    const std::unordered_map<int,int>& fixed_variables = {}
                    );

private:
    SFDGurobiConfig cfg;

    // Store subgraphs for callbacks
    std::map<int, SubgraphData> subgraphs;


    std::tuple< 
        GRBModel*,
        std::map<std::pair<int,int>, GRBVar>,
        std::map<std::tuple<int,int,int>, GRBVar>
    >
    buildModel(
        const Problem& problem,
        std::map<int, SubgraphData>& subgraphs,
        const std::unordered_map<int,int>& fixed_variables
    );
    

};