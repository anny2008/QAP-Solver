#include "rtl1_solver.h"
#include <cstring>
#include <cstdlib>

/**
 * C Interface for RTL1 SCIP Solver
 * 
 * This provides C bindings that can be called from Python via ctypes.
 * Memory management is handled carefully to work across language boundaries.
 */

extern "C" {

// Opaque handle to RTL1Solver instance
typedef void* RTL1SolverHandle;

// Result structure that matches Python expectations
struct ResultData {
    bool feasible;
    double objective;
    double lower_bound;
    double solve_time;
    int status;
    int n;
    int* assignment;  // Array of size n
};

// ============================================================================
// Constructor and Destructor
// ============================================================================

/**
 * Create an RTL1 solver instance.
 * 
 * Returns: Opaque handle to RTL1Solver
 */
RTL1SolverHandle rtl1_create(int n,
                             double* F_data,  // Flow matrix, row-major order
                             double* D_data)  // Distance matrix, row-major order
{
    // Convert C arrays to C++ vectors
    std::vector<std::vector<double>> F(n, std::vector<double>(n));
    std::vector<std::vector<double>> D(n, std::vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            F[i][j] = F_data[i * n + j];
            D[i][j] = D_data[i * n + j];
        }
    }
    
    // Create solver with default config
    RTL1Solver::Config config;
    RTL1Solver* solver = new RTL1Solver(n, F, D, config);
    
    return (RTL1SolverHandle)solver;
}

/**
 * Create an RTL1 solver instance with configuration.
 * 
 * Parameters:
 *   n: Problem size
 *   F_data: Flow matrix (row-major)
 *   D_data: Distance matrix (row-major)
 *   is_relax: Use relaxed 0-1 variables
 *   time_limit: Time limit in seconds
 *   threads: Number of threads
 *   log_output: Print SCIP output
 *   preprocessing_symmetry: Symmetry detection level
 * 
 * Returns: Opaque handle to RTL1Solver
 */
RTL1SolverHandle rtl1_create_with_config(int n,
                                        double* F_data,
                                        double* D_data,
                                        int is_relax,
                                        double time_limit,
                                        int threads,
                                        int log_output,
                                        int preprocessing_symmetry)
{
    // Convert C arrays to C++ vectors
    std::vector<std::vector<double>> F(n, std::vector<double>(n));
    std::vector<std::vector<double>> D(n, std::vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            F[i][j] = F_data[i * n + j];
            D[i][j] = D_data[i * n + j];
        }
    }
    
    // Create config
    RTL1Solver::Config config;
    config.is_relax = (is_relax != 0);
    config.time_limit = time_limit;
    config.threads = threads;
    config.log_output = (log_output != 0);
    config.preprocessing_symmetry = preprocessing_symmetry;
    
    RTL1Solver* solver = new RTL1Solver(n, F, D, config);
    
    return (RTL1SolverHandle)solver;
}

/**
 * Destroy solver instance and free memory.
 */
void rtl1_destroy(RTL1SolverHandle handle)
{
    if (handle) {
        RTL1Solver* solver = (RTL1Solver*)handle;
        delete solver;
    }
}

// ============================================================================
// Configuration Methods
// ============================================================================

/**
 * Set warm-start solution.
 * 
 * Parameters:
 *   handle: Solver instance
 *   ws_data: Array of pairs [i1, u1, i2, u2, ..., i_n, u_n]
 *   ws_count: Number of (facility, location) pairs
 *   ws_values: Array of values for each pair
 */
void rtl1_set_warm_start(RTL1SolverHandle handle,
                        int* ws_data,        // [i1, u1, i2, u2, ...]
                        int ws_count,
                        double* ws_values)   // [v1, v2, ...]
{
    if (!handle) return;
    
    RTL1Solver* solver = (RTL1Solver*)handle;
    std::map<std::pair<int,int>, double> warm_start;
    
    for (int idx = 0; idx < ws_count; idx++) {
        int i = ws_data[2 * idx];
        int u = ws_data[2 * idx + 1];
        double val = ws_values[idx];
        warm_start[std::make_pair(i, u)] = val;
    }
    
    solver->setWarmStart(warm_start);
}

/**
 * Set fixed variables.
 * 
 * Parameters:
 *   handle: Solver instance
 *   fixed_data: Array of pairs [i1, u1, i2, u2, ...]
 *   fixed_count: Number of pairs
 */
void rtl1_set_fixed_variables(RTL1SolverHandle handle,
                             int* fixed_data,
                             int fixed_count)
{
    if (!handle) return;
    
    RTL1Solver* solver = (RTL1Solver*)handle;
    std::vector<std::pair<int,int>> fixed;
    
    for (int idx = 0; idx < fixed_count; idx++) {
        int i = fixed_data[2 * idx];
        int u = fixed_data[2 * idx + 1];
        fixed.push_back(std::make_pair(i, u));
    }
    
    solver->setFixedVariables(fixed);
}

// ============================================================================
// Solving
// ============================================================================

/**
 * Build and solve the RTL1 model.
 * 
 * Parameters:
 *   handle: Solver instance
 *   result: Pointer to ResultData structure to fill
 * 
 * Returns: 0 on success, non-zero on error
 */
int rtl1_solve(RTL1SolverHandle handle, ResultData* result)
{
    if (!handle || !result) return -1;
    
    RTL1Solver* solver = (RTL1Solver*)handle;
    
    // Build the model
    int build_status = solver->buildModel();
    if (build_status != 0) return build_status;
    
    // Solve
    RTL1Solver::Solution sol = solver->solve();
    
    // Fill result structure
    result->feasible = sol.feasible;
    result->objective = sol.objective;
    result->lower_bound = sol.lower_bound;
    result->solve_time = sol.solve_time;
    result->status = sol.status;
    result->n = solver->n;
    
    // Copy assignment
    result->assignment = (int*)malloc(solver->n * sizeof(int));
    if (sol.feasible && sol.assignment.size() == solver->n) {
        for (int i = 0; i < solver->n; i++) {
            result->assignment[i] = sol.assignment[i];
        }
    } else {
        for (int i = 0; i < solver->n; i++) {
            result->assignment[i] = -1;
        }
    }
    
    return 0;
}

/**
 * Evaluate objective for a given assignment.
 * 
 * Parameters:
 *   handle: Solver instance
 *   assignment: Array of size n where assignment[i] = location for facility i
 * 
 * Returns: QAP objective value
 */
double rtl1_evaluate_objective(RTL1SolverHandle handle, int* assignment)
{
    if (!handle) return -1.0;
    
    RTL1Solver* solver = (RTL1Solver*)handle;
    std::vector<int> assign(assignment, assignment + solver->n);
    
    return solver->evaluateObjective(assign);
}

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Free assignment array allocated by rtl1_solve.
 */
void rtl1_free_assignment(int* assignment)
{
    if (assignment) {
        free(assignment);
    }
}

// ============================================================================
// Utility
// ============================================================================

/**
 * Get solver info.
 */
void rtl1_print_info(RTL1SolverHandle handle)
{
    if (handle) {
        RTL1Solver* solver = (RTL1Solver*)handle;
        solver->printInfo();
    }
}

}  // extern "C"
