#pragma once
#include "IncrementalRMP.h"
#include "PricingEngine.h"
#include <optional>
#include <unordered_map>

struct ColumnGenConfig {
    double time_limit_sec = 120.0;
    std::string add_column_strategy = "add_all_negative";
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
    double lambda_step_down = 0.5; // decrease when no progress
    bool use_relative_improve = true; // relative vs absolute
    double improve_rel_eps = 1e-6;    // relative improvement threshold

    bool use_dual_box_restriction = true;       // enable perturbed dual pricing
    double box_gamma = 0.2;
    double box_alpha = 0.1;

    static ColumnGenConfig fromJson(const json& j) {
        ColumnGenConfig cfg;
        if (j.contains("time_limit")) cfg.time_limit_sec = j["time_limit"];
        if (j.contains("add_column_strategy")) cfg.add_column_strategy = j["add_column_strategy"];
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
        if (j.contains("lambda_step_down")) cfg.lambda_step_down = j["lambda_step_down"];
        if (j.contains("use_relative_improve")) cfg.use_relative_improve = j["use_relative_improve"];
        if (j.contains("improve_rel_eps")) cfg.improve_rel_eps = j["improve_rel_eps"];
        if (j.contains("use_dual_box_restriction")) cfg.use_dual_box_restriction = j["use_dual_box_restriction"];
        if (j.contains("box_gamma")) cfg.box_gamma = j["box_gamma"];
        if (j.contains("box_alpha")) cfg.box_alpha = j["box_alpha"];
        if (cfg.stabilize) {
            std::cout << "| lambda_min | lambda_max | lambda_init | lambda_step_up | lambda_step_down" << std::endl;
            std::cout << "|------------|------------|-------------|----------------|------------------" << std::endl;
            std::cout << "| " << cfg.lambda_min << " | " << cfg.lambda_max << " | " << cfg.lambda_init << " | " << cfg.lambda_step_up << " | " << cfg.lambda_step_down << " |\n";
        }
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
    static std::vector<int> argmaxAssignment(int n, int m,
        const std::unordered_map<PairKey,double,PairKeyHash>& xvals);

};