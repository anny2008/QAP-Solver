#pragma once
#include <string>

struct SolverConfig {
    bool add_most_negative = true;
    double eps = 1e-7;
    int max_iterations = 10000;
    bool log_output = false;
    std::string decomposition;
    std::string decomposition_file = "";
    std::string basis_output_file = "";

    static SolverConfig fromJson(const json& j) {
        SolverConfig cfg;
        if (j.contains("add_most_negative")) cfg.add_most_negative = j["add_most_negative"];
        if (j.contains("epsilon")) cfg.eps = j["epsilon"];
        if (j.contains("max_iterations")) cfg.max_iterations = j["max_iterations"];
        if (j.contains("log_output")) cfg.log_output = j["log_output"];
        if (j.contains("decomposition")) cfg.decomposition = j["decomposition"];
        return cfg;
    }
};

