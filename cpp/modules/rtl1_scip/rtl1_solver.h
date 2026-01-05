#ifndef QAP_RTL1_SCIP_SOLVER_H
#define QAP_RTL1_SCIP_SOLVER_H

#include <scip/scip.h>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include <cstdlib>

/**
 * RTL1 SCIP Solver for Quadratic Assignment Problem
 * 
 * Implements Relaxation-based Tightened Linear (RTL1) formulation with SCIP.
 * Supports warm-start solutions and fixed variables.
 * 
 * Variables:
 *   x[i,u] ∈ {0,1}: facility i assigned to location u
 *   y[i,u,j,v] ∈ {0,1}: both (i,u) and (j,v) assignments
 * 
 * Objective: min Σ D[i,j] * F[u,v] * y[i,u,j,v]
 */
class RTL1Solver {
public:
    struct Config {
        bool is_relax = false;              // Use relaxed 0-1 variables (continuous)
        double time_limit = 120.0;          // Time limit in seconds
        int threads = 8;                    // Number of threads
        bool log_output = false;            // Print SCIP output
        int preprocessing_symmetry = 5;     // Symmetry detection level (0-5)
    };

    struct Solution {
        bool feasible = false;
        std::vector<int> assignment;        // assignment[i] = u means facility i at location u
        double objective = -1.0;            // Actual QAP objective (if feasible)
        double lower_bound = -1.0;          // RTL1 lower bound
        double solve_time = 0.0;            // Wall-clock time
        int status = -1;                    // SCIP status
    };

    // Problem dimensions and matrices
    int n;
    std::vector<std::vector<double>> F;    // Flow matrix (n x n)
    std::vector<std::vector<double>> D;    // Distance matrix (n x n)
    
    // SCIP objects
    SCIP* scip;
    Config config;
    
    // Variables
    std::map<std::pair<int,int>, SCIP_VAR*> x_vars;  // x[i,u]
    std::map<std::tuple<int,int,int,int>, SCIP_VAR*> y_vars;  // y[i,u,j,v]
    
    // Warm-start and fixed variables
    std::map<std::pair<int,int>, double> warm_start_assignment;
    std::vector<std::pair<int,int>> fixed_variables;

    /**
     * Constructor
     * 
     * @param n Problem size
     * @param F Flow matrix (n x n)
     * @param D Distance matrix (n x n)
     * @param config Solver configuration
     */
    RTL1Solver(int n, const std::vector<std::vector<double>>& F,
               const std::vector<std::vector<double>>& D,
               const Config& config = Config())
        : n(n), F(F), D(D), config(config), scip(nullptr) {
        SCIP_CALL_ABORT(SCIPcreate(&scip));
        SCIP_CALL_ABORT(SCIPincludeDefaultPlugins(scip));
    }

    /**
     * Destructor - Clean up SCIP
     */
    ~RTL1Solver() {
        if (scip) {
            SCIP_CALL_ABORT(SCIPfree(&scip));
        }
    }

    /**
     * Set warm-start solution
     * 
     * @param assignment Map from (facility, location) to value
     */
    void setWarmStart(const std::map<std::pair<int,int>, double>& assignment) {
        warm_start_assignment = assignment;
    }

    /**
     * Add fixed variables
     * 
     * @param fixed List of (facility, location) pairs to fix to 1
     */
    void setFixedVariables(const std::vector<std::pair<int,int>>& fixed) {
        fixed_variables = fixed;
    }

    /**
     * Build the RTL1 model
     * 
     * @return 0 on success, non-zero on error
     */
    int buildModel();

    /**
     * Solve the RTL1 model
     * 
     * @return Solution object with results
     */
    Solution solve();

    /**
     * Evaluate QAP objective for a given assignment
     * 
     * @param assignment Vector where assignment[i] = u
     * @return QAP objective value
     */
    double evaluateObjective(const std::vector<int>& assignment) const;

    /**
     * Print solver information
     */
    void printInfo() const;

private:
    /**
     * Create x variables (n x n binary/continuous matrix)
     */
    int createXVariables();

    /**
     * Create y variables (n x n x n x n binary/continuous)
     */
    int createYVariables();

    /**
     * Add assignment constraints:
     * - Each facility assigned to exactly one location
     * - Each location assigned to exactly one facility
     */
    int addAssignmentConstraints();

    /**
     * Add linking constraints between x and y variables
     */
    int addLinkingConstraints();

    /**
     * Add symmetry constraints: y[i,u,j,v] == y[j,v,i,u]
     */
    int addSymmetryConstraints();

    /**
     * Add fixed variable constraints
     */
    int addFixedVariables();

    /**
     * Add warm-start solution to SCIP
     */
    int addWarmStart();

    /**
     * Extract assignment from solution
     * 
     * @param sol SCIP solution
     * @return Vector of assignments
     */
    std::vector<int> extractAssignment(SCIP_SOL* sol) const;
};

#endif // QAP_RTL1_SCIP_SOLVER_H
