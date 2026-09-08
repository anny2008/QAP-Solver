#pragma once
#include "IncrementalRMP.h"
#include <optional>
#include <unordered_map>

struct ColumnGenConfig {
    double time_limit_sec = 120.0;
    bool add_most_negative = true;
    bool use_bigM = false;
    double eps = 1e-7;
    int max_iterations = 10000;
    bool log_output = false;

    // --- NEW: stabilization knobs ---
    bool stabilize = true;       // enable perturbed dual pricing
    double lambda_min = 0.05;
    double lambda_max = 0.90;
    double lambda_init = 0.20;
    double lambda_step_up = 0.05; // increase when progress
    bool use_relative_improve = true; // relative vs absolute
    double improve_rel_eps = 1e-6;    // relative improvement threshold

    std::string instance_path = "";

    static ColumnGenConfig fromJson(const json& j) {
        ColumnGenConfig cfg;
        if (j.contains("instance_path")) cfg.instance_path = j["instance_path"];
        if (j.contains("time_limit")) cfg.time_limit_sec = j["time_limit"];
        if (j.contains("add_most_negative")) cfg.add_most_negative = j["add_most_negative"];
        if (j.contains("use_bigM")) cfg.use_bigM = j["use_bigM"];
        if (j.contains("epsilon")) cfg.eps = j["epsilon"];
        if (j.contains("max_iterations")) cfg.max_iterations = j["max_iterations"];
        if (j.contains("log_output")) cfg.log_output = j["log_output"];
        // NEW
        if (j.contains("stabilize")) cfg.stabilize = j["stabilize"];
        if (j.contains("lambda_min")) cfg.lambda_min = j["lambda_min"];
        if (j.contains("lambda_max")) cfg.lambda_max = j["lambda_max"];
        if (j.contains("lambda_init")) cfg.lambda_init = j["lambda_init"];
        if (j.contains("lambda_step_up")) cfg.lambda_step_up = j["lambda_step_up"];
        if (j.contains("use_relative_improve")) cfg.use_relative_improve = j["use_relative_improve"];
        if (j.contains("improve_rel_eps")) cfg.improve_rel_eps = j["improve_rel_eps"];
        return cfg;
    }
};


class ColumnGenSolver {
public:
    explicit ColumnGenSolver(const ColumnGenConfig& cfg);

    // Build and solve for a Problem; fixed_variables is (location -> facility).
    Solution solve(const Problem& problem,
                    const std::string& instance_path,
                    const std::unordered_map<int,int>& fixed_variables = {}
                    );

private:
    ColumnGenConfig cfg;

    // φ(i,u,j,v) flattened
    static std::vector<double> buildPhi(const Problem& P);
    static std::vector<QuadKey> buildInitialOmega(const Problem& P,
        const std::unordered_map<int,int>& fixed);

    // Greedy assignment from x values (argmax_u x_{i,u})
    // static std::vector<int> argmaxAssignment(int n, int m,
    //     const std::unordered_map<PairKey,double,PairKeyHash>& xvals);

};