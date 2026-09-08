#pragma once
#include "../../core/problem.h"
#include "../../core/solution.h"
#include <optional>
#include <unordered_map>
#include "gurobi_c++.h"

struct RLT1GurobiReductionIVConfig {
    bool log_output = false;
    bool is_relax = false;
    std::string lpmethod = "barrier";
    std::string warmstart = "";


    static RLT1GurobiReductionIVConfig fromJson(const json& j) {
        RLT1GurobiReductionIVConfig cfg;
        if (j.contains("log_output")) cfg.log_output = j["log_output"];
        if (j.contains("is_relax")) cfg.is_relax = j["is_relax"];
        if (j.contains("lpmethod")) cfg.lpmethod = j["lpmethod"];
        if (j.contains("warmstart")) cfg.warmstart = j["warmstart"];
        std::cout << "Config loaded: log_output=" << cfg.log_output
                  << ", is_relax=" << cfg.is_relax
                  << ", lpmethod=" << cfg.lpmethod
                  << ", warmstart=" << cfg.warmstart
                  << std::endl;
        return cfg;
    }
};


class RLT1GurobiReductionIVSolver {
public:
    RLT1GurobiReductionIVSolver(const RLT1GurobiReductionIVConfig& cfg);

    // Build and solve for a Problem; fixed_variables is (location -> facility).
    Solution solve(const Problem& problem,
                    const std::string& instance_path,
                    const std::unordered_map<int,int>& fixed_variables = {}
                    );

private:
    RLT1GurobiReductionIVConfig cfg;

    std::tuple<GRBModel*,std::map<std::pair<int,int>, GRBVar>,std::map<std::tuple<int,int,int,int>, GRBVar>>
    buildModel(
        const Problem& problem,
        const std::unordered_map<int,int>& fixed_variables
    );

};