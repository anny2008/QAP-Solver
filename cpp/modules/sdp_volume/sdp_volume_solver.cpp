/**
 * SDP Volume Algorithm Solver for QAP
 * 
 * Implements the Lagrangian relaxation-based Volume algorithm
 * for the SDP formulation of the Quadratic Assignment Problem.
 * 
 * Based on:
 * - Barahona & Anbil (1998): "The Volume algorithm: producing primal 
 *   solutions with a subgradient method"
 * - SDP formulation for QAP
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <omp.h>

#include "sdp_volume_solver.hpp"
#include "../../include/qap_solution_io.hpp"
#include "Hungarian.h"
#include "ADMM.h"
#include <ilcplex/ilocplex.h>
// This MUST be included at the top of your file
#include <unsupported/Eigen/KroneckerProduct>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>



// Parse fixed variables from a text file with lines:
//   x i u value   (value in {0,1})
//   y i u j v value (value in {0,1})
// Indices are 0-based; '#' starts a comment line.
#define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v))
bool parse_fixed_file(const std::string &path, int n, FixedVariables &out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cout << "\rError: cannot open fixed-variable file: " << path << std::endl;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        std::string trimmed = line;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](int ch) { return !std::isspace(ch); }));
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::istringstream iss(trimmed);
        char type;
        iss >> type;
        if (type == 'x') {
            int i, u, val;
            if (!(iss >> i >> u >> val)) {
                std::cout << "\rWarning: malformed x-line at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (val == 1) {
                out.x_fixed_1[i] = u;
            } else {
                out.x_fixed_0[i].insert(u);
            }
        } else if (type == 'y') {
            int i, u, j, v, val;
            if (!(iss >> i >> u >> j >> v >> val)) {
                std::cout << "\rWarning: malformed y-line at " << path << ":" << lineno << std::endl;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || j < 0 || j >= n || v < 0 || v >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::endl;
                continue;
            }
            int key = key_y(n, i, u, v);
            if (val == 1) {
                out.y_fixed_1[key] = j;
            } else {
                out.y_fixed_0[key].insert(j);
            }
        } else {
            std::cout << "\rWarning: unknown line type at " << path << ":" << lineno << std::endl;
        }
    }

    std::cout << "\rLoaded fixed variables from " << path
              << " | x1=" << out.x_fixed_1.size()
              << " x0-rows=" << out.x_fixed_0.size()
              << " y1=" << out.y_fixed_1.size()
              << " y0-rows=" << out.y_fixed_0.size() << std::endl;
    return true;
}


PrimalViolationSummary compute_primal_violation_summary(const VOL_dvector &psol, int n) {
    PrimalViolationSummary summary;
    summary.n = n;

    auto idx_x = [n](int i, int u) { return i * n + u; };
    auto idx_y = [n](int i, int u, int j, int v) {
        return n * n + i * n * n * n + u * n * n + j * n + v;
    };

    double total_abs = 0.0;
    long long total_cnt = 0;

    // Assignment: sum_i x[i,u] = 1 for all u
    double assign_abs_sum = 0.0;
    long long assign_cnt = 0;
    for (int u = 0; u < n; ++u) {
        double residual = 1.0;
        for (int i = 0; i < n; ++i) {
            residual -= psol[idx_x(i, u)];
        }
        double a = std::abs(residual);
        summary.assignment_max = std::max(summary.assignment_max, a);
        assign_abs_sum += a;
        ++assign_cnt;
    }
    summary.assignment_avg = assign_cnt > 0 ? assign_abs_sum / assign_cnt : 0.0;

    // Linking: sum_v y[i,u,j,v] = x[i,u] for all i,u,j
    double link_abs_sum = 0.0;
    long long link_cnt = 0;
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                double residual = psol[idx_x(i, u)];
                for (int v = 0; v < n; ++v) {
                    residual -= psol[idx_y(i, u, j, v)];
                }
                double a = std::abs(residual);
                summary.link_max = std::max(summary.link_max, a);
                link_abs_sum += a;
                ++link_cnt;
            }
        }
    }
    summary.link_avg = link_cnt > 0 ? link_abs_sum / link_cnt : 0.0;

    // Symmetry: y[i,u,j,v] = y[j,v,i,u] for all i,u,j,v
    double sym_abs_sum = 0.0;
    long long sym_cnt = 0;
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    double residual = psol[idx_y(i, u, j, v)] - psol[idx_y(j, v, i, u)];
                    double a = std::abs(residual);
                    sym_abs_sum += a;
                    ++sym_cnt;
                    summary.max_abs = std::max(summary.max_abs, a);
                }
            }
        }
    }

    total_abs = assign_abs_sum + link_abs_sum + sym_abs_sum;
    total_cnt = assign_cnt + link_cnt + sym_cnt;
    summary.max_abs = std::max(summary.max_abs, summary.assignment_max);
    summary.max_abs = std::max(summary.max_abs, summary.link_max);
    summary.avg_abs = total_cnt > 0 ? total_abs / static_cast<double>(total_cnt) : 0.0;

    return summary;
}

FixedViolationReport check_fixed_violations(const FixedVariables &fv, const VOL_dvector &psol, int n, double tol) {
    FixedViolationReport rep;

    // x fixed to 1
    for (const auto &kv : fv.x_fixed_1) {
        int i = kv.first;
        int u = kv.second;
        double val = psol[i * n + u];
        double dev = std::abs(val - 1.0);
        if (dev > tol) {
            ++rep.x1_violations;
            rep.x1_max_dev = std::max(rep.x1_max_dev, dev);
        }
    }

    // x fixed to 0
    for (const auto &kv : fv.x_fixed_0) {
        int i = kv.first;
        for (int u : kv.second) {
            double val = psol[i * n + u];
            double dev = std::abs(val);
            if (dev > tol) {
                ++rep.x0_violations;
                rep.x0_max_dev = std::max(rep.x0_max_dev, dev);
            }
        }
    }

    // y fixed to 1
    for (const auto &kv : fv.y_fixed_1) {
        int key = kv.first;
        int j = kv.second;
        int i = key / (n * n);
        int rem = key % (n * n);
        int u = rem / n;
        int v = rem % n;
        int idx = n * n + i * n * n * n + u * n * n + j * n + v;
        double val = psol[idx];
        double dev = std::abs(val - 1.0);
        if (dev > tol) {
            ++rep.y1_violations;
            rep.y1_max_dev = std::max(rep.y1_max_dev, dev);
        }
    }

    // y fixed to 0
    for (const auto &kv : fv.y_fixed_0) {
        int key = kv.first;
        int i = key / (n * n);
        int rem = key % (n * n);
        int u = rem / n;
        int v = rem % n;
        for (int j : kv.second) {
            int idx = n * n + i * n * n * n + u * n * n + j * n + v;
            double val = psol[idx];
            double dev = std::abs(val);
            if (dev > tol) {
                ++rep.y0_violations;
                rep.y0_max_dev = std::max(rep.y0_max_dev, dev);
            }
        }
    }

    return rep;
}

/*
    * Solving the subproblem with MOSEK
    * min trace <L + W^T gamma W , R>
    * subject to:
    *   R is PSD
    *   R[0,0] = 1
    *   [1  y^T  ]  is PSD
    *   [y  WRW^T]  where y = diag(WRW^T)
    *  W = (1/n * e kron e, V kron V)
    *  V = (I_{n-1})
    *      (-e^T_{n-1})
    * L = W^T(B kron A)W
*/
SDPVolumeHooks1::SDPVolumeHooks1(const Problem& data, VOL_problem& vol_prob) : SDPVolumeHooks(data, vol_prob)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto n = data.n;
    Eigen::MatrixXd V(n, n - 1);
    V.topRows(n - 1).setIdentity();           // Top Block: I_{n-1}
    V.bottomRows(1).setConstant(-1.0);        // Bottom Block: -e^T_{n-1}

    // Left Component: 1/n * (e_n kron e_n) -> Size: (n^2 x 1)
    Eigen::VectorXd e = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd left_block = (1.0 / n) * kroneckerProduct(e, e).eval();
    Eigen::MatrixXd V_kron_V = kroneckerProduct(V, V).eval();
    int w_rows = n * n;
    int w_cols = 1 + (n - 1) * (n - 1);
    W.resize(w_rows, w_cols);
    W << left_block, V_kron_V;

    Eigen::MatrixXd A = convert_nested_vector_to_eigen(data.D);
    Eigen::MatrixXd B = convert_nested_vector_to_eigen(data.F);
    Eigen::MatrixXd B_kron_A = kroneckerProduct(B, A).eval();
    L = W.transpose() * B_kron_A * W;


    int r_dim = 1 + (n-1)*(n-1);         
    
    M = new Model("SDP_Bundle_Subproblem_Exact");
    M->setLogHandler([](const std::string & msg) {});

    // ==========================================
    // 1. VARIABLES (No large pi_param needed!)
    // ==========================================
    R = M->variable("R", Domain::inPSDCone(r_dim));

    // ==========================================
    // 2. CONSTRUCT INTERMEDIATE EXPRESSIONS
    // ==========================================
    RowMajorMatrix W_row_major = W;
    RowMajorMatrix L_row_major = L;
    // 2. Instantiate a blank Monty Array of the target shape
    auto W_monty = new_array_ptr<double, 2>(
        shape(static_cast<int>(W_row_major.rows()), static_cast<int>(W_row_major.cols()))
    );
    auto L_monty = new_array_ptr<double, 2>(
        shape(static_cast<int>(L_row_major.rows()), static_cast<int>(L_row_major.cols()))
    );

    // 3. Copy the contiguous raw Eigen data directly into the Monty layout
    std::copy(W_row_major.data(), W_row_major.data() + W_row_major.size(), W_monty->begin());
    std::copy(L_row_major.data(), L_row_major.data() + L_row_major.size(), L_monty->begin());


    Matrix::t W_mosek = Matrix::dense(W_monty);
    Expression::t WR = Expr::mul(W_mosek, R);
    Expression::t WRWT = Expr::mul(WR, W_mosek->transpose());
    auto diag_indices = new_array_ptr<int,2>({n*n, 2});
    for (int i = 0; i < n*n; ++i) {
        (*diag_indices)(i, 0) = i;
        (*diag_indices)(i, 1) = i;
    }
    // Pick elements explicitly if needed
    Expression::t x = WRWT->pick(diag_indices);

    // ==========================================
    // 3. OBJECTIVE INITIALIZATION
    // ==========================================
    std::cout << "Initial objective matrix L (size: " << L_monty->shape[0] << " x " << L_monty->shape[1] << ")" << std::endl;
    // Initialize with the exact baseline. We will overwrite this dynamically in the loop.
    M->objective(ObjectiveSense::Minimize, Expr::dot(L_monty, R));

    // ==========================================
    // 4. CONSTRAINTS (Your original exact formulation)
    // ==========================================
    M->constraint("R_00_unit", R->index(0, 0), Domain::equalsTo(1.0));

    // if(0)
    {
        // 1. Reshape the n^2 x n^2 matrix expression into an (n^2) x n x n matrix
        // This groups columns by (j * n + v) -> [j, v]
        Expression::t WRWT_3d = Expr::reshape(WRWT, new_array_ptr<int, 1>({ n * n, n, n }));

        // 2. Sum along the 'j' dimension (which is axis 1 of our 3D tensor)
        // This gives a matrix expression of shape [n^2, n], containing all your 'y_sum' blocks
        Expression::t y_sums = Expr::sum(WRWT_3d, 1);

        // 3. Reshape the 1D vector x (size n^2) into an n^2 x 1 matrix expression
        Expression::t x_col = Expr::reshape(x, n * n, 1);

        // 4. Replicate x_col horizontally n times using Expr::repeat to match the [n^2, n] shape of y_sums
        Expression::t x_broadcasted = Expr::repeat(x_col, n, 1);

        // 5. Generate all n^3 linking constraints simultaneously with a single matrix equation
        M->constraint("Linking_Vectorized", Expr::sub(y_sums, x_broadcasted), Domain::equalsTo(0.0));
    }

    // Expression::t top_left = Expr::constTerm(Matrix::eye(1));  
    // Expression::t top_right = Expr::transpose(x);              
    // Expression::t top_row = Expr::hstack(top_left, top_right); 
    // Expression::t bottom_row = Expr::hstack(x, WRWT);          
    // Expression::t composite_block = Expr::vstack(top_row, bottom_row); 

    // M->constraint("Composite_PSD", composite_block, Domain::inPSDCone());

    // compressed_L = W.transpose() * L * W;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "SDPVolumeHooks1 initialization completed in " << elapsed.count() << " seconds." << std::endl;
}

int SDPVolumeHooks1::solve_subproblem_1(const VOL_dvector& pi,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
    
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    vio = 0.0; // Set all dual variables to zero


    //  Step 2: Update the objective function in MOSEK with the current dual variables
    {
        // std::cout << "W matrix size: " << W.rows() << " x " << W.cols() << std::endl;

        // std::cout << "L matrix size: " << L.rows() << " x " << L.cols() << std::endl;
        // std::cout << "DensePi size: " << DensePi.size() << std::endl;
        // compute W^T * gamma * W and add it to L to form the new objective matrix
        Eigen::Map<Eigen::MatrixXd> gamma(pi.v, n*n, n*n); // Assuming DensePi is of size W.cols() * W.cols()
        auto W_gamma_W = W.transpose() * gamma * W; // Compute W^T * gamma * W
        Eigen::MatrixXd new_L = L - W_gamma_W; // New objective matrix

        RowMajorMatrix L_row_major = new_L;
        auto L_monty = new_array_ptr<double, 2>(
            shape(static_cast<int>(L_row_major.rows()), static_cast<int>(L_row_major.cols()))
        );

        std::copy(L_row_major.data(), L_row_major.data() + L_row_major.size(), L_monty->begin());
        // update the objective in MOSEK
        M->objective(ObjectiveSense::Minimize, Expr::dot(L_monty, R));
    }
    
    
    // Solve the problem using the existing optimization matrix structure
    // Measure the time taken for the optimization
    auto start_time = std::chrono::high_resolution_clock::now();
    M->solve();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    // std::cout << "MOSEK optimization time: " << elapsed.count() << " seconds" << std::endl;

    // Retrieve optimized matrices
    auto R_sol = R->level();
    lcost = M->primalObjValue();
    SolutionStatus status = M->getPrimalSolutionStatus();
    if (status != SolutionStatus::Optimal) {
        std::cerr << "Warning: MOSEK did not find an optimal solution. Status: " << static_cast<int>(status) << std::endl;
    // } else {
    //     std::cout << "  MOSEK found an optimal solution. Objective value: " << lcost << ", time: " << elapsed.count() << " seconds" << std::endl;
    }


    // update vio based on the solution
    //  vio [i*n^3 + u*n^2 + j*n + v] = WRWT[i*n + u, j*n + v]
    {
        int r_dim = W.cols();
        Eigen::Map<Eigen::MatrixXd> R_sol_eigen(R_sol->raw(), r_dim, r_dim);
        Eigen::MatrixXd result_matrix = - W * R_sol_eigen * W.transpose();
        // vio.assign(result_matrix.data(), result_matrix.data() + result_matrix.size());
        // vio = result_matrix.data(); // Copy the result into vio
        // copy the result into vio using std::copy
        std::copy(result_matrix.data(), result_matrix.data() + result_matrix.size(), vio.v);
        pcost = (L.transpose() * R_sol_eigen).trace();

    }

    return 0;
}

int SDPVolumeHooks1::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_1(pi, lcost, psol, vio, pcost);
}


SDPVolumeHooks2::SDPVolumeHooks2(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver)
    : SDPVolumeHooks(data, vol_prob), n(data.n), alpha(data.n), eigen_solver(eigen_solver)
{

}

int SDPVolumeHooks2::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_2(pi, lcost, psol, vio, pcost);
                            }
    
bool SDPVolumeHooks2::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    return true;
}

/**
 * @brief Computes the flat 1D array index for an upper-triangular packed 4D tensor.
 * @param offset The starting index of the y_iujv block in your global array.
 * @param i Row variable index (0 to n-1)
 * @param u Row assignment index (0 to n-1)
 * @param j Column variable index (0 to n-1)
 * @param v Column assignment index (0 to n-1)
 * @param n The dimension of the QAP problem.
 * @return The precise 1D absolute array index.
 */
inline int get_quad_index(int offset, int i, int u, int j, int v, int n) {
    // 1. Convert 4D coordinates to two 2D matrix linear indices
    int idx1 = i * n + u;
    int idx2 = j * n + v;

    // 2. Ensure idx1 is always the smaller or equal index (Row <= Col)
    if (idx1 > idx2) {
        std::swap(idx1, idx2);
    }

    // 3. Apply the dense triangular packing formula
    int n2 = n * n;
    int tri_idx = idx1 * n2 - (idx1 * (idx1 - 1)) / 2 + (idx2 - idx1);

    // 4. Return the absolute location inside the global array
    return offset + tri_idx;
}

#define pi_u(u) pi[(pi_u_offset) + (u)]
#define pi_i(i) pi[(pi_i_offset) + (i)]
#define pi_iuv(i,u,v) pi[(pi_iuv_offset) + (i)*(n)*(n) + (u)*(n) + (v)]
#define pi_iuj(i,u,j) pi[(pi_iuj_offset) + (i)*(n)*(n) + (u)*(n) + (j)]
#define pi_iujv(i,u,j,v) pi[(pi_iujv_offset) + (i)*(n)*(n)*(n) + (u)*(n)*(n) + (j)*(n) + (v)]


#define vio_u(u) vio[(vio_u_offset) + (u)]
#define vio_i(i) vio[(vio_i_offset) + (i)]
#define vio_iuv(i,u,v) vio[(vio_iuv_offset) + (i)*(n)*(n) + (u)*(n) + (v)]
#define vio_iuj(i,u,j) vio[(vio_iuj_offset) + (i)*(n)*(n) + (u)*(n) + (j)]
#define vio_iujv(i,u,j,v) vio[(vio_iujv_offset) + (i)*(n)*(n)*(n) + (u)*(n)*(n) + (j)*(n) + (v)]


/*
    * Solving the subproblem with Lanczos method (approximate)
    *   min <C,X> + <pi, b - A(X)> 
    * = min <C - ATpi, X> + <b,pi>
    * subject to:
    *   X is PSD, trace X = alpha
    * compute min eigenvalue of C - ATpi lambda
    * and corresponding eigenvector v
    * X* = alpha*vv^T
    * lcost = alpha*lambda + <b,pi> = <C - ATpi, X*> + <b,pi>
    * pcost = <C, X*> = <C, vv^T>
    * vio = b - A(X*) = b - A(vv^T)
    * pi [2*n + 2*n*n*n + n*n*n*n]
    * constraints:
    *   0                       -   n                               sum_i x_iu = 1 for all u
    *   n                       -   2*n                             sum_u x_iu = 1 for all i
    *   2*n                     -   2*n + n*n*n                     sum_j y[i,u,j,v] = x_iu for all i,u,v
    *   2*n + n*n*n             -   2*n + 2*n*n*n                   sum_v y[i,u,j,v] = x_iu for all i,u,j
    *   2*n + 2*n*n*n           -   2*n + 2*n*n*n + n*n*n*n         y_iujv >= 0 for all i,u,j,v
    * indexing:
    *   X is a matrix of size n^2 x n^2
    * C = B kron A
*/    
int SDPVolumeHooks2::solve_subproblem_2(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // Alternative subproblem solver for SDP (Volume algorithm)
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    vio = 0.0;  // Reset constraint violations for the current subproblem

    {
        int pi_u_offset = 0;
        int pi_i_offset = n;
        int pi_iuv_offset = 2*n;
        int pi_iuj_offset = 2*n + n*n*n;
        int pi_iujv_offset = 2*n + 2*n*n*n;
        // compute C - AT*pi in the lifted formulation X = [Y x; x^T 1],
        // where x stores the assignment variables and Y stores the y variables.
        // 1. Initialize to Zero
        Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n * n, n * n);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx1 = i * n + u;
                
                // Loop bounds strictly restricted to the upper triangle to save 50% effort
                for (int j = i; j < n; ++j) {
                    int v_start = (j == i) ? u : 0;
                    #pragma omp simd
                    for (int v = v_start; v < n; ++v) {
                        int idx2 = j * n + v;
                        
                        if (idx1 != idx2) {
                            // Retrieve the compressed 1D variable via our helper
                            int index = get_quad_index(pi_iujv_offset, i, u, j, v, n);
                            double current_pi_iujv = pi[index];

                            double AT_pi_f = pi_iuj(i, u, j) + pi_iuv(i, u, v);
                            double AT_pi_b = pi_iuj(j, v, i) + pi_iuv(j, v, u); 

                            // Symmetrize and apply the split factor for the off-diagonal multiplier
                            double val = qap_data.D[i][j] * qap_data.F[u][v] - 0.5 * (AT_pi_f + AT_pi_b) - 0.5 * current_pi_iujv;
                            
                            // Thread-safe writes: distinct (idx1, idx2) pairs belong to exactly one thread assignment
                            C_minus_AT_pi(idx1, idx2) = val;
                            C_minus_AT_pi(idx2, idx1) = val;
                        }
                    }
                }
            }
        }

        // 2. Build DIAGONAL elements in parallel
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx = i * n + u;
                
                int index = get_quad_index(pi_iujv_offset, i, u, i, u, n);
                double current_pi_iujv_diag = pi[index];

                // Every thread gets its own isolated local accumulator copy
                double AT_pi = pi_i(i) + pi_u(u) + pi_iuj(i, u, i) + pi_iuv(i, u, u) + current_pi_iujv_diag;
                
                #pragma omp simd reduction(+:AT_pi)
                for (int j = 0; j < n; ++j) {
                    AT_pi -= pi_iuj(i, u, j) + pi_iuv(i, u, j);
                }
                
                // Thread-safe write: each diagonal position is completely unique to the (i, u) pair
                C_minus_AT_pi(idx, idx) = qap_data.D[i][i] * qap_data.F[u][u] - AT_pi;
            }
        }


     
        // --- 3. EIGENVALUE DECOMPOSITION SELECTOR ---
        double lambda = 0.0;
	    Eigen::VectorXd v;
        if (eigen_solver == "power_iteration")
        {
            
            int matrix_dim = C_minus_AT_pi.rows();
            // Compute a fast, conservative upper bound using the infinity norm (max row sum)
            double max_ev_estimate = 0.0;
            #pragma omp parallel for reduction(max:max_ev_estimate)
            for (int i = 0; i < matrix_dim; ++i) {
                double row_sum = C_minus_AT_pi.row(i).cwiseAbs().sum();
                if (row_sum > max_ev_estimate) max_ev_estimate = row_sum;
            }
            
            // Apply spectral shift buffer to safely isolate the absolute smallest eigenvalue
            double mu = max_ev_estimate + 10.0; 

            v = Eigen::VectorXd::Random(matrix_dim);
            v.normalize();

            const int max_iter = 150;
            const double tol = 1e-5;
            double lambda_old = 0.0;
            bool converged = false;

            Eigen::VectorXd next_v = Eigen::VectorXd::Zero(matrix_dim);

            for (int iter = 0; iter < max_iter; ++iter) {
                // Shifted mapping: next_v = (mu * I - A) * v = mu * v - A * v
                // Eigen parallelizes this matrix-vector multiplication natively via OpenMP
                next_v.noalias() = mu * v - C_minus_AT_pi * v;

                double norm = next_v.norm();
                if (norm < 1e-12) break; 
                v = next_v / norm;

                // Evaluate Rayleigh Quotient for the unshifted target matrix
                double current_lambda = v.dot(C_minus_AT_pi * v);

                if (iter > 0 && std::abs(current_lambda - lambda_old) < tol) {
                    lambda = current_lambda;
                    converged = true;
                    break;
                }
                lambda_old = current_lambda;
            }

            // Secure production fallback if the power loop fails to hit target tolerance
            if (!converged) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else if (eigen_solver == "lanczos")
        {
            Spectra::DenseSymMatProd<double> op(C_minus_AT_pi);
            int matrix_dim = C_minus_AT_pi.rows();
            int ncv = std::min(2, matrix_dim);
            if (ncv <= 2) ncv = 3;

            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, ncv);
            eigs.init();

            // 1. CAP ITERATIONS AT 20 FOR BLAZING FAST SUBGRADIENT ESTIMATES
            // 2. We keep the fallback checking 'nconv > 0' because as long as it has computed 
            //    at least 1 rough estimate, it's valid for subproblem optimization bounds!
            int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 20, 1e-5);

            if (nconv > 0) {
                lambda = eigs.eigenvalues()(0);
                v = eigs.eigenvectors().col(0);
            } else {
                // Fail-safe fallback ONLY if the solver completely errors out (e.g., returns NaN)
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else // default/exact dense fallback option
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigenvalue decomposition failed!" << std::endl;
                return -1;
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                pcost += qap_data.D[i][i] * qap_data.F[u][u] * X(i*n + u, i*n + u);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += qap_data.D[i][j] * qap_data.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (qap_data.D[i][j] * qap_data.F[u][v] + qap_data.D[j][i] * qap_data.F[v][u]) * X(i*n + u, j*n + v);
                        }
                    }
                }
            }
        }
        // std::cout << "Min eigenvalue: " << lambda  << ", Pcost: " << pcost << std::endl;
        
        {
            int vio_u_offset = 0;
            int vio_i_offset = n;
            int vio_iuv_offset = 2*n;
            int vio_iuj_offset = 2*n + n*n*n;
            int vio_iujv_offset = 2*n + 2*n*n*n;
            // vio_u(u) = 1 - sum_i x_iu for all u
            #pragma omp parallel for schedule(static)
            for (int u = 0; u < n; ++u) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int i = 0; i < n; ++i) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_u(u) = acc;
            }
            // vio_i(i) = 1 - sum_u x_iu for all i
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int u = 0; u < n; ++u) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_i(i) = acc;
            }
            // vio_iuj(i, u, j) = x_iu - sum_v y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int j = 0; j < n; ++j) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int v = 0; v < n; ++v) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuj(i, u, j) = base_val - sum_y;
                    }
                }
            }
            // vio_iuv(i, u, v) = x_iu - sum_j y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int j = 0; j < n; ++j) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuv(i, u, v) = base_val - sum_y;
                    }
                }
            }
            // vio_iujv(i, u, j, v) = -y[i,u,j,v] ONLY for upper triangle (idx1 <= idx2)
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    int idx1 = i * n + u;
                    
                    for (int j = i; j < n; ++j) {
                        int v_start = (j == i) ? u : 0;
                        
                        #pragma omp simd
                        for (int v = v_start; v < n; ++v) {
                            int idx2 = j * n + v;
                            int index = get_quad_index(vio_iujv_offset, i, u, j, v, n);
                            vio[index] = -X(idx1, idx2);
                        }
                    }
                }
            }


        }
        lcost = lambda*alpha;
        // #pragma omp parallel for schedule(static) 
        for (int i= 0; i < n; ++i) {
            lcost += pi_i(i) + pi_u(i);
        }

    }

    auto end_time = std::chrono::high_resolution_clock::now();
    subproblem_solving_time += (end_time - start_time);

    return 0;
}


/**
 * @brief Computes the ultra-compressed flat 1D index for active y_iujv elements.
 * @return The absolute 1D array index if active, or -1 if the variable is pruned.
 */
inline int get_compressed_quad_index(int offset, int i, int u, int j, int v, int n, const int* quad_to_compact) {
    int idx1 = i * n + u;
    int idx2 = j * n + v;

    if (idx1 > idx2) {
        std::swap(idx1, idx2);
    }

    int n2 = n * n;
    int tri_idx = idx1 * n2 - (idx1 * (idx1 - 1)) / 2 + (idx2 - idx1);

    int compact_idx = quad_to_compact[tri_idx];
    
    // Return the compressed offset if active, otherwise flag it as -1
    return (compact_idx != -1) ? (offset + compact_idx) : -1;
}

SDPVolumeHooks3::SDPVolumeHooks3(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver)
    : SDPVolumeHooks(data, vol_prob), n(data.n), alpha(data.n), eigen_solver(eigen_solver)
{

    int total_upper_pairs = (n * n * (n * n + 1)) / 2;

    // Allocate a translation array initialized to -1 (meaning inactive/pruned)
    quad_to_compact.resize(total_upper_pairs, -1);
    int active_quad_count = 0;

    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            int idx1 = i * n + u;
            
            for (int j = i; j < n; ++j) {
                int v_start = (j == i) ? u : 0;
                for (int v = v_start; v < n; ++v) {
                    int idx2 = j * n + v;
                    
                    // Evaluate your condition
                    double cond = qap_data.D[i][j] * qap_data.F[u][v] + qap_data.D[j][i] * qap_data.F[v][u];
                    
                    // Get the standard uncompressed upper-triangular index
                    int n2 = n * n;
                    int tri_idx = idx1 * n2 - (idx1 * (idx1 - 1)) / 2 + (idx2 - idx1);
                    
                    if (cond > 0.0) {
                        quad_to_compact[tri_idx] = active_quad_count;
                        active_quad_count++;
                    }
                }
            }
        }
    }


    vol_problem.dsize = 2 * n + 2 * n * n * n + active_quad_count;
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = 0.0;
    vol_problem.dual_ub = DBL_MAX;
    // vol_problem.dual_lb = -DBL_MAX;
    // vol_problem.dual_ub = 0.0;

    for (int i = 0; i < 2*n + 2*n*n*n; ++i) {
        vol_problem.dual_lb[i] = -DBL_MAX; // first 2*n + 2*n*n*n are free variables
    }
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 1.0;
}

int SDPVolumeHooks3::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_3(pi, lcost, psol, vio, pcost);
                            }
    
bool SDPVolumeHooks3::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    return true;
}

/*
    * Solving the subproblem with Lanczos method (approximate)
    *   min <C,X> + <pi, b - A(X)> 
    * = min <C - ATpi, X> + <b,pi>
    * subject to:
    *   X is PSD, trace X = alpha
    * compute min eigenvalue of C - ATpi lambda
    * and corresponding eigenvector v
    * X* = vv^T
    * lcost = alpha*lambda + <b,pi> = <C - ATpi, X*> + <b,pi>
    * pcost = <C, X*> = <C, vv^T>
    * vio = b - A(X*) = b - A(vv^T)
    * pi [2*n + 2*n*n*n + n*n*n*n]
    * constraints:
    *   0                       -   n                               sum_i x_iu = 1 for all u
    *   n                       -   2*n                             sum_u x_iu = 1 for all i
    *   2*n                     -   2*n + n*n*n                     sum_j y[i,u,j,v] = x_iu for all i,u,v
    *   2*n + n*n*n             -   2*n + 2*n*n*n                   sum_v y[i,u,j,v] = x_iu for all i,u,j
    *   2*n + 2*n*n*n           -   2*n + 2*n*n*n + n*n*n*n         y_iujv >= 0 for all i,u,j,v
    * indexing:
    *   X is a matrix of size n^2 x n^2
    * C = B kron A
*/    
int SDPVolumeHooks3::solve_subproblem_3(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // Alternative subproblem solver for SDP (Volume algorithm)
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    vio = 0.0;  // Reset constraint violations for the current subproblem

    {
        int pi_u_offset = 0;
        int pi_i_offset = n;
        int pi_iuv_offset = 2*n;
        int pi_iuj_offset = 2*n + n*n*n;
        int pi_iujv_offset = 2*n + 2*n*n*n;
        // compute C - AT*pi in the lifted formulation X = [Y x; x^T 1],
        // where x stores the assignment variables and Y stores the y variables.
        // 1. Initialize to Zero
        Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n * n, n * n);
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx1 = i * n + u;
                for (int j = i; j < n; ++j) {
                    int v_start = (j == i) ? u : 0;
                    
                    // Note: Pruned lanes diverge, so we let the compiler optimize branching normally
                    for (int v = v_start; v < n; ++v) {
                        int idx2 = j * n + v;
                        if (idx1 != idx2) {
                            int index = get_compressed_quad_index(pi_iujv_offset, i, u, j, v, n, quad_to_compact.data());
                            
                            // If index is valid, constraint is active. Fetch from compact pi array.
                            double current_pi_iujv = (index != -1) ? pi[index] : 0.0;

                            double AT_pi_f = pi_iuj(i, u, j) + pi_iuv(i, u, v);
                            double AT_pi_b = pi_iuj(j, v, i) + pi_iuv(j, v, u); 

                            double val = qap_data.D[i][j] * qap_data.F[u][v] - 0.5 * (AT_pi_f + AT_pi_b) - 0.5 * current_pi_iujv;
                            
                            C_minus_AT_pi(idx1, idx2) = val;
                            C_minus_AT_pi(idx2, idx1) = val;
                        }
                    }
                }
            }
        }


        // 3. Build the DIAGONAL elements (idx1 == idx2)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx = i * n + u;
                
                // 1. For diagonal elements, j=i and v=u -> cond = 2 * d_ii * f_uu
                double cond = 2.0 * qap_data.D[i][i] * qap_data.F[u][u];
                
                int index = get_quad_index(pi_iujv_offset, i, u, i, u, n);
                double current_pi_iujv_diag = (cond > 0.0) ? pi[index] : 0.0;

                double AT_pi = pi_i(i) + pi_u(u) + pi_iuj(i, u, i) + pi_iuv(i, u, u) + current_pi_iujv_diag;
                
                #pragma omp simd reduction(+:AT_pi)
                for (int j = 0; j < n; ++j) {
                    AT_pi -= pi_iuj(i, u, j) + pi_iuv(i, u, j);
                }
                
                C_minus_AT_pi(idx, idx) = qap_data.D[i][i] * qap_data.F[u][u] - AT_pi;
            }
        }


     
        // --- 3. EIGENVALUE DECOMPOSITION SELECTOR ---
        double lambda = 0.0;
	    Eigen::VectorXd v;
        if (eigen_solver == "power_iteration")
        {
            
            int matrix_dim = C_minus_AT_pi.rows();
            // Compute a fast, conservative upper bound using the infinity norm (max row sum)
            double max_ev_estimate = 0.0;
            #pragma omp parallel for reduction(max:max_ev_estimate)
            for (int i = 0; i < matrix_dim; ++i) {
                double row_sum = C_minus_AT_pi.row(i).cwiseAbs().sum();
                if (row_sum > max_ev_estimate) max_ev_estimate = row_sum;
            }
            
            // Apply spectral shift buffer to safely isolate the absolute smallest eigenvalue
            double mu = max_ev_estimate + 10.0; 

            v = Eigen::VectorXd::Random(matrix_dim);
            v.normalize();

            const int max_iter = 150;
            const double tol = 1e-5;
            double lambda_old = 0.0;
            bool converged = false;

            Eigen::VectorXd next_v = Eigen::VectorXd::Zero(matrix_dim);

            for (int iter = 0; iter < max_iter; ++iter) {
                // Shifted mapping: next_v = (mu * I - A) * v = mu * v - A * v
                // Eigen parallelizes this matrix-vector multiplication natively via OpenMP
                next_v.noalias() = mu * v - C_minus_AT_pi * v;

                double norm = next_v.norm();
                if (norm < 1e-12) break; 
                v = next_v / norm;

                // Evaluate Rayleigh Quotient for the unshifted target matrix
                double current_lambda = v.dot(C_minus_AT_pi * v);

                if (iter > 0 && std::abs(current_lambda - lambda_old) < tol) {
                    lambda = current_lambda;
                    converged = true;
                    break;
                }
                lambda_old = current_lambda;
            }

            // Secure production fallback if the power loop fails to hit target tolerance
            if (!converged) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else if (eigen_solver == "lanczos")
        {
            Spectra::DenseSymMatProd<double> op(C_minus_AT_pi);
            int matrix_dim = C_minus_AT_pi.rows();
            int ncv = std::min(2, matrix_dim);
            if (ncv <= 2) ncv = 3;

            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, 2);
            eigs.init();

            // 1. CAP ITERATIONS AT 20 FOR BLAZING FAST SUBGRADIENT ESTIMATES
            // 2. We keep the fallback checking 'nconv > 0' because as long as it has computed 
            //    at least 1 rough estimate, it's valid for subproblem optimization bounds!
            int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 20, 1e-5);

            if (nconv > 0) {
                lambda = eigs.eigenvalues()(0);
                v = eigs.eigenvectors().col(0);
            } else {
                // Fail-safe fallback ONLY if the solver completely errors out (e.g., returns NaN)
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else // default/exact dense fallback option
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigenvalue decomposition failed!" << std::endl;
                return -1;
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                pcost += qap_data.D[i][i] * qap_data.F[u][u] * X(i*n + u, i*n + u);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += qap_data.D[i][j] * qap_data.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (qap_data.D[i][j] * qap_data.F[u][v] + qap_data.D[j][i] * qap_data.F[v][u]) * X(i*n + u, j*n + v);
                        }
                    }
                }
            }
        }
        // std::cout << "Min eigenvalue: " << lambda  << ", Pcost: " << pcost << std::endl;
        
        {
            int vio_u_offset = 0;
            int vio_i_offset = n;
            int vio_iuv_offset = 2*n;
            int vio_iuj_offset = 2*n + n*n*n;
            int vio_iujv_offset = 2*n + 2*n*n*n;
            
            // [vio_u, vio_i, vio_iuj, and vio_iuv loops remain unchanged]
            #pragma omp parallel for schedule(static)
            for (int u = 0; u < n; ++u) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int i = 0; i < n; ++i) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_u(u) = acc;
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int u = 0; u < n; ++u) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_i(i) = acc;
            }
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int j = 0; j < n; ++j) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        #pragma omp simd reduction(+:sum_y)
                        for (int v = 0; v < n; ++v) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuj(i, u, j) = base_val - sum_y;
                    }
                }
            }
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        #pragma omp simd reduction(+:sum_y)
                        for (int j = 0; j < n; ++j) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuv(i, u, v) = base_val - sum_y;
                    }
                }
            }

            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    int idx1 = i * n + u;
                    for (int j = i; j < n; ++j) {
                        int v_start = (j == i) ? u : 0;
                        
                        for (int v = v_start; v < n; ++v) {
                            int idx2 = j * n + v;
                            int index = get_compressed_quad_index(vio_iujv_offset, i, u, j, v, n, quad_to_compact.data());
                            
                            // Only write a violation if the constraint actually exists in memory
                            if (index != -1) {
                                vio[index] = -X(idx1, idx2);
                            }
                        }
                    }
                }
            }

        }
        lcost = lambda*alpha;
        // #pragma omp parallel for schedule(static) 
        for (int i= 0; i < n; ++i) {
            lcost += pi_i(i) + pi_u(i);
        }

    }

    auto end_time = std::chrono::high_resolution_clock::now();
    subproblem_solving_time += (end_time - start_time);

    return 0;
}


SDPVolumeHooks4::SDPVolumeHooks4(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver)
    : SDPVolumeHooks(data, vol_prob), n(data.n), alpha(data.n), eigen_solver(eigen_solver)
{
}

int SDPVolumeHooks4::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_4(pi, lcost, psol, vio, pcost);
                            }
    
bool SDPVolumeHooks4::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    return true;
}


#define pi_iu(i,u) pi[(pi_iu_offset) + (i)*(n) + (u)]
#define pi_iuiv(i,u,v) pi[(pi_iuiv_offset) + (i)*(n)*(n) + (u)*(n) + (v)]
#define pi_iuju(i,u,j) pi[(pi_iuju_offset) + (i)*(n)*(n) + (u)*(n) + (j)]
#define vio_iu(i,u) vio[(vio_iu_offset) + (i)*(n) + (u)]
#define vio_iuiv(i,u,v) vio[(vio_iuiv_offset) + (i)*(n)*(n) + (u)*(n) + (v)]
#define vio_iuju(i,u,j) vio[(vio_iuju_offset) + (i)*(n)*(n) + (u)*(n) + (j)]

/*
    * Solving the subproblem with Lanczos method (approximate)
    *   min <C,X> + <pi, b - A(X)> 
    * = min <C - ATpi, X> + <b,pi>
    * subject to:
    *   X is PSD, trace X = alpha
    * compute min eigenvalue of C - ATpi lambda
    * and corresponding eigenvector v
    * X* = vv^T
    * lcost = alpha*lambda + <b,pi> = <C - ATpi, X*> + <b,pi>
    * pcost = <C, X*> = <C, vv^T>
    * vio = b - A(X*) = b - A(vv^T)
    * pi [2*n + 2*n*n*n + n*n]
    * constraints:
    *   0                           -   n                               sum_i x_iu = 1 for all u
    *   n                           -   2*n                             sum_u x_iu = 1 for all i
    *   2*n                         -   2*n + n*n*n                     sum_j y[i,u,j,v] = x_iu for all i,u,v
    *   2*n + n*n*n                 -   2*n + 2*n*n*n                   sum_v y[i,u,j,v] = x_iu for all i,u,j
    *   2*n + 2*n*n*n               -   2*n + 2*n*n*n + n*n             y_iuiu >= 0 for all i,u
    *   2*n + 2*n*n*n + n*n         -   2*n + 2*n*n*n + n*n + n*n*n     y_iuiv == 0 for all i,u,v != u
    *   2*n + 2*n*n*n + n*n + n*n*n -   2*n + 2*n*n*n + n*n + 2*n*n*n   y_iuju == 0 for all i,u,j != u
    * indexing:
    *   X is a matrix of size n^2 x n^2
    * C = B kron A
*/    
int SDPVolumeHooks4::solve_subproblem_4(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // Alternative subproblem solver for SDP (Volume algorithm)
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    vio = 0.0;  // Reset constraint violations for the current subproblem

    {
        int pi_u_offset = 0;
        int pi_i_offset = n;
        int pi_iuv_offset = 2*n;
        int pi_iuj_offset = 2*n + n*n*n;
        int pi_iu_offset = 2*n + 2*n*n*n;
        int pi_iuiv_offset = 2*n + 2*n*n*n + n*n;
        int pi_iuju_offset = 2*n + 2*n*n*n + n*n + n*n*n;
        // compute C - AT*pi in the lifted formulation X = [Y x; x^T 1],
        // where x stores the assignment variables and Y stores the y variables.
        // 1. Initialize to Zero
        Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n * n, n * n);

        
        // 1. Build OFF-DIAGONAL elements in parallel
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx1 = i * n + u;
                
                // Loop bounds strictly restricted to the upper triangle to save 50% effort
                for (int j = i; j < n; ++j) {
                    int v_start = (j == i) ? u : 0;
                    #pragma omp simd
                    for (int v = v_start; v < n; ++v) {
                        int idx2 = j * n + v;
                        
                        if (idx1 != idx2) {
                            double AT_pi_f = pi_iuj(i, u, j) + pi_iuv(i, u, v);
                            double AT_pi_b = pi_iuj(j, v, i) + pi_iuv(j, v, u); 

                            // Apply y_iuiv == 0 constraints when within the same row block (j == i)
                            if (j == i) {
                                AT_pi_f += pi_iuiv(i, u, v);
                                AT_pi_b += pi_iuiv(i, v, u);
                            }

                            // Apply y_iuju == 0 constraints when within the same column block (v == u)
                            if (v == u) {
                                AT_pi_f += pi_iuju(i, u, j);
                                AT_pi_b += pi_iuju(j, u, i);
                            }

                            // Symmetrize and apply the split factor
                            double val = qap_data.D[i][j] * qap_data.F[u][v] - 0.5 * (AT_pi_f + AT_pi_b);
                            
                            // Thread-safe writes
                            C_minus_AT_pi(idx1, idx2) = val;
                            C_minus_AT_pi(idx2, idx1) = val;
                        }
                    }
                }
            }
        }

        // 2. Build DIAGONAL elements in parallel
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx = i * n + u;
                
                // Every thread gets its own isolated local accumulator copy
                // Includes the active diagonal non-negativity multiplier pi_iu(i, u)
                double AT_pi = pi_i(i) + pi_u(u) + pi_iuj(i, u, i) + pi_iuv(i, u, u) + pi_iu(i, u);
                
                #pragma omp simd reduction(+:AT_pi)
                for (int j = 0; j < n; ++j) {
                    AT_pi -= pi_iuj(i, u, j) + pi_iuv(i, u, j);
                }
                
                // Thread-safe write
                C_minus_AT_pi(idx, idx) = qap_data.D[i][i] * qap_data.F[u][u] - AT_pi;
            }
        }


     
        // --- 3. EIGENVALUE DECOMPOSITION SELECTOR ---
        double lambda = 0.0;
	    Eigen::VectorXd v;
        if (eigen_solver == "power_iteration")
        {
            
            int matrix_dim = C_minus_AT_pi.rows();
            // Compute a fast, conservative upper bound using the infinity norm (max row sum)
            double max_ev_estimate = 0.0;
            #pragma omp parallel for reduction(max:max_ev_estimate)
            for (int i = 0; i < matrix_dim; ++i) {
                double row_sum = C_minus_AT_pi.row(i).cwiseAbs().sum();
                if (row_sum > max_ev_estimate) max_ev_estimate = row_sum;
            }
            
            // Apply spectral shift buffer to safely isolate the absolute smallest eigenvalue
            double mu = max_ev_estimate + 10.0; 

            v = Eigen::VectorXd::Random(matrix_dim);
            v.normalize();

            const int max_iter = 150;
            const double tol = 1e-5;
            double lambda_old = 0.0;
            bool converged = false;

            Eigen::VectorXd next_v = Eigen::VectorXd::Zero(matrix_dim);

            for (int iter = 0; iter < max_iter; ++iter) {
                // Shifted mapping: next_v = (mu * I - A) * v = mu * v - A * v
                // Eigen parallelizes this matrix-vector multiplication natively via OpenMP
                next_v.noalias() = mu * v - C_minus_AT_pi * v;

                double norm = next_v.norm();
                if (norm < 1e-12) break; 
                v = next_v / norm;

                // Evaluate Rayleigh Quotient for the unshifted target matrix
                double current_lambda = v.dot(C_minus_AT_pi * v);

                if (iter > 0 && std::abs(current_lambda - lambda_old) < tol) {
                    lambda = current_lambda;
                    converged = true;
                    break;
                }
                lambda_old = current_lambda;
            }

            // Secure production fallback if the power loop fails to hit target tolerance
            if (!converged) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else if (eigen_solver == "lanczos")
        {
            Spectra::DenseSymMatProd<double> op(C_minus_AT_pi);
            int matrix_dim = C_minus_AT_pi.rows();
            int ncv = std::min(2, matrix_dim);
            if (ncv <= 2) ncv = 3;

            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, 2);
            eigs.init();

            // 1. CAP ITERATIONS AT 20 FOR BLAZING FAST SUBGRADIENT ESTIMATES
            // 2. We keep the fallback checking 'nconv > 0' because as long as it has computed 
            //    at least 1 rough estimate, it's valid for subproblem optimization bounds!
            int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 20, 1e-5);

            if (nconv > 0) {
                lambda = eigs.eigenvalues()(0);
                v = eigs.eigenvectors().col(0);
            } else {
                // Fail-safe fallback ONLY if the solver completely errors out (e.g., returns NaN)
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else // default/exact dense fallback option
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigenvalue decomposition failed!" << std::endl;
                return -1;
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                pcost += qap_data.D[i][i] * qap_data.F[u][u] * X(i*n + u, i*n + u);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += qap_data.D[i][j] * qap_data.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (qap_data.D[i][j] * qap_data.F[u][v] + qap_data.D[j][i] * qap_data.F[v][u]) * X(i*n + u, j*n + v);
                        }
                    }
                }
            }
        }
        // std::cout << "Min eigenvalue: " << lambda  << ", Pcost: " << pcost << std::endl;
       
        {
            int vio_u_offset = 0;
            int vio_i_offset = n;
            int vio_iuv_offset = 2*n;
            int vio_iuj_offset = 2*n + n*n*n;
            int vio_iu_offset = 2*n + 2*n*n*n;
            int vio_iuiv_offset = 2*n + 2*n*n*n + n*n;
            int vio_iuju_offset = 2*n + 2*n*n*n + n*n + n*n*n;
            
            // vio_u(u) = 1 - sum_i x_iu for all u
            #pragma omp parallel for schedule(static)
            for (int u = 0; u < n; ++u) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int i = 0; i < n; ++i) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_u(u) = acc;
            }
            // vio_i(i) = 1 - sum_u x_iu for all i
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int u = 0; u < n; ++u) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_i(i) = acc;
            }
            // vio_iuj(i, u, j) = x_iu - sum_v y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int j = 0; j < n; ++j) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int v = 0; v < n; ++v) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuj(i, u, j) = base_val - sum_y;
                    }
                }
            }
            // vio_iuv(i, u, v) = x_iu - sum_j y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int j = 0; j < n; ++j) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuv(i, u, v) = base_val - sum_y;
                    }
                }
            }
            
            // vio_iu(i, u) = -y[i,u,i,u] ONLY for diagonal elements
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    int idx = i * n + u;
                    vio_iu(i, u) = -X(idx, idx);
                }
            }

            // vio_iuiv(i, u, v) = -y[i,u,i,v] for all v != u (0 when v == u)
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        if (u != v) {
                            vio_iuiv(i, u, v) = -X(i * n + u, i * n + v);
                        } else {
                            vio_iuiv(i, u, v) = 0.0;
                        }
                    }
                }
            }

            // vio_iuju(i, u, j) = -y[i,u,j,u] for all j != i (0 when j == i)
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int j = 0; j < n; ++j) {
                        if (i != j) {
                            vio_iuju(i, u, j) = -X(i * n + u, j * n + u);
                        } else {
                            vio_iuju(i, u, j) = 0.0;
                        }
                    }
                }
            }
        }

        lcost = lambda*alpha;
        // #pragma omp parallel for schedule(static) 
        for (int i= 0; i < n; ++i) {
            lcost += pi_i(i) + pi_u(i);
        }

    }

    auto end_time = std::chrono::high_resolution_clock::now();
    subproblem_solving_time += (end_time - start_time);

    return 0;
}

SDPVolumeHooks5::SDPVolumeHooks5(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver)
    : SDPVolumeHooks(data, vol_prob), n(data.n), alpha(data.n), eigen_solver(eigen_solver), iter(0)
{
    n = data.n;
    Vhat = build_Vhat(n);
    auto N_total = 1 + n * n;
    auto n2 = n * n;
    In = Eigen::MatrixXd::Identity(n, n);
    En = Eigen::MatrixXd::Ones(n, n);
    en = Eigen::MatrixXd::Ones(n, 1);
    {
        // Define In and En matrices

        // 1. Compute the Kronecker product part
        Eigen::MatrixXd A = n * In - En;
        Eigen::MatrixXd kron_res = Eigen::kroneckerProduct(A, A);

        // 2. Compute the bottom-right matrix block
        Eigen::MatrixXd bottom_right = (1.0 / n2) * Eigen::MatrixXd::Ones(n2, n2) + 
                                    (1.0 / (n2 * (n - 1))) * kron_res;

        // 3. Initialize the full Yhat matrix with size (1 + n2) x (1 + n2)
        Y = Eigen::MatrixXd::Zero(1 + n2, 1 + n2);


        // 4. Populate the blocks (Note: C++ uses 0-based indexing)
        Y(0, 0) = 1.0;
        Y.block(0, 1, 1, n2)  = (1.0 / n) * Eigen::RowVectorXd::Ones(n2);
        Y.block(1, 0, n2, 1)  = (1.0 / n) * Eigen::VectorXd::Ones(n2);
        Y.block(1, 1, n2, n2) = bottom_right;
        J = Eigen::MatrixXd::Ones(N_total, N_total);
        // check if J is gangster
        for (int i = 0; i < n; ++i)
            for (int u = 0; u < n; ++u)
                for (int j = 0; j < n; ++j)
                    for (int v = 0; v < n; ++v)
                    {
                        bool is_gangster = (i != j && u == v) || (i == j && u != v);
                        if (is_gangster) J(i*n + u + 1, j*n + v + 1) = 0.0;
                        else J(i*n + u + 1, j*n + v + 1) = 1.0;
                    }

        
        Y = Y.cwiseProduct(J);
        Y(0, 0) = 1.0;
        R = Vhat.transpose() * Y * Vhat;
        Z = Y - Vhat * R * Vhat.transpose();
        std::copy(Z.data(), Z.data() + Z.size(), vol_problem.dsol.v);
    }
    beta = n/3.0;
    L = Eigen::MatrixXd::Zero(N_total, N_total);
    Eigen::MatrixXd A = convert_nested_vector_to_eigen(data.D);
    Eigen::MatrixXd B = convert_nested_vector_to_eigen(data.F);
    L.block(1, 1, n2, n2) = Eigen::kroneckerProduct(B, A);
    normL = L.norm();
    L = L / normL*n2;
}

int SDPVolumeHooks5::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_5(pi, lcost, psol, vio, pcost);
                            }
    
bool SDPVolumeHooks5::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    return true;
}


/*
    * Solving the subproblem with Lanczos method (approximate)
    *   min <WT C W,R> - <pi, Y> + pi0
    *       where Y = W R WT
    *   = min <WT (C - pi) W,R> + pi0
    * subject to:
    *   R is PSD, trace R = alpha
    * compute min eigenvalue of C - ATpi lambda
*/    
int SDPVolumeHooks5::solve_subproblem_5(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // Alternative subproblem solver for SDP (Volume algorithm)
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    vio = 0.0;  // Reset constraint violations for the current subproblem
    {
        auto N_total = 1 + n * n;
        auto n2 = n * n;
        bool is_low_rank = true;
        Z = Eigen::Map<const Eigen::MatrixXd>(pi.v, n*n+1, n*n+1);
        auto fW = Y + Z/beta;
        auto WVhat = Vhat.transpose() * fW * Vhat;
        Eigen::MatrixXd VRV;
        if(is_low_rank) {
            double lambda = 0.0;
	        Eigen::VectorXd v;
            Spectra::DenseSymMatProd<double> op(WVhat);
            int matrix_dim = WVhat.rows();
            int ncv = std::min(2, matrix_dim);
            if (ncv <= 2) ncv = 3;

            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, ncv);
            eigs.init();

            int nconv = eigs.compute(Spectra::SortRule::LargestAlge, 100, 1e-5);

            if (nconv > 0) {
                lambda = eigs.eigenvalues()(0);
                v = eigs.eigenvectors().col(0);
            }
            else 
            {
                // Fail-safe fallback ONLY if the solver completely errors out (e.g., returns NaN)
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(WVhat);
                Eigen::Index maxIndex;
                lambda = eigensolver.eigenvalues().maxCoeff(&maxIndex);
                v = eigensolver.eigenvectors().col(maxIndex);
                std::cout << "Warning: Lanczos solver failed to converge, using fallback eigenvalue decomposition." << std::endl;
            }
            if (lambda > 1e-9) {
                auto temp = Vhat * v;
                VRV = temp * lambda * temp.transpose();
            } else {
                VRV = Eigen::MatrixXd::Zero(Z.rows(), Z.cols());
            }
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(WVhat);
            Eigen::VectorXd evals = eigensolver.eigenvalues();
            Eigen::MatrixXd evecs = eigensolver.eigenvectors();
            int num_pos = (evals.array() > 1e-9).count();
            if (num_pos > 0) {
                // Slice the last 'num_pos' elements since they are sorted lowest -> highest
                Eigen::VectorXd positive_evals = evals.tail(num_pos);
                
                // Slice the corresponding columns from the eigenvector matrix
                Eigen::MatrixXd positive_evecs = evecs.rightCols(num_pos);
                auto temp = Vhat * positive_evecs;
                VRV = temp * positive_evals.asDiagonal() * temp.transpose();
            } else {
                VRV = Eigen::MatrixXd::Zero(Z.rows(), Z.cols());
            }
        }
        Eigen::MatrixXd Y_iter = (VRV - (L + Z) / beta).cwiseMin(1).cwiseMax(0).cwiseProduct(J).eval();
        Y_iter(0, 0) = 1.0;
        Eigen::MatrixXd pR = Y_iter - VRV;
        Eigen::MatrixXd dR = Y_iter - Y;
        Y = Y_iter;
        Eigen::MatrixXd  vio_matrix = (beta*pR).eval();
        std::copy(vio_matrix.data(), vio_matrix.data() + vio_matrix.size(), vio.v);
        auto obj = L.cwiseProduct(Y).sum();
        pcost = obj + 0.5*beta*(pR.cwiseProduct(pR)).sum();
        lcost = pcost + Z.cwiseProduct(pR).sum();
        // Z += (gamma * beta) * pR;
        // auto nrm_pR = pR.norm();
        // auto nrm_dR = beta*dR.norm();
        // auto feas = nrm_pR / Y.norm();
        // auto obj = L.cwiseProduct(Y).sum() * normL/n2;
        if(iter % 100 == 0) {
            std::cout << "Iter: " << iter << ", lcost: " << obj*normL/n2 << ", pcost: " << pcost << std::endl;
        }
        iter++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    subproblem_solving_time += (end_time - start_time);

    return 0;
}



SDPVolumeHooks6::SDPVolumeHooks6(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver)
    : SDPVolumeHooks(data, vol_prob), n(data.n), alpha(data.n), eigen_solver(eigen_solver),
        m_ncv(int(data.n*data.n*0.2)), m_tol(1.0)
{
    if(m_ncv < 3) m_ncv = 3;
    std::cout << "ncv: " << m_ncv << ", tol: " << m_tol << std::endl;

}

int SDPVolumeHooks6::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) 
{
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = solve_subproblem_6(pi, lcost, psol, vio, pcost);
    auto end_time = std::chrono::high_resolution_clock::now();
    subproblem_solving_time += (end_time - start_time);
    return result;
}
    
bool SDPVolumeHooks6::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    return true;
}

int SDPVolumeHooks6::post_solving() {
    // obtain pi_x = pi[2*n + 2*n*n*n:]
    // auto pi_x_offset = 2*n + 2*n*n*n;
    auto pi_x_offset = 0;
    auto pi = vol_problem.psol.v;
    // X_star = 0 n*n x n*n matrix
    Eigen::MatrixXd X_star = Eigen::MatrixXd::Zero(n*n, n*n);
    // X_star = -pi
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = i; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    int index = get_quad_index(pi_x_offset, i, u, j, v, n);
                    X_star(i*n + u, j*n + v) = pi[index];
                    X_star(j*n + v, i*n + u) =  X_star(i*n + u, j*n + v); // Symmetrize the matrix
                }
            }
        }
    }

    // std::cout << "X_star:\n" << X_star << std::endl;
    // check if X_star is PSD
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(X_star);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Eigenvalue decomposition failed!" << std::endl;
        return -1;
    }
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    // std::cout << "Eigenvalues of X_star:\n" << eigenvalues << std::endl;
    if ((eigenvalues.array() < -1e-9).any()) {
        std::cerr << "X_star is not positive semidefinite!" << std::endl;
        return -1;
    } else {
        std::cout << "X_star is positive semidefinite." << std::endl;
    }

    // comput objective value <C, X_star>
    double obj_value = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                for (int v = 0; v < n; ++v) {
                    obj_value += qap_data.D[i][j] * qap_data.F[u][v] * X_star(i*n + u, j*n + v);
                }
            }
        }
    }
    std::cout << "Objective value <C, X_star>: " << obj_value << std::endl;
    // compute constraint violations b - A(X_star)
    Eigen::VectorXd vio_i_vec = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd vio_u_vec = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd vio_iuj_vec = Eigen::VectorXd::Zero(n*n*n);
    Eigen::VectorXd vio_iuv_vec = Eigen::VectorXd::Zero(n*n*n);
    for (int i = 0; i < n; ++i) {
        double sum_x_iu = 0.0;
        for (int u = 0; u < n; ++u) {
            sum_x_iu += X_star(i*n + u, i*n + u);
        }
        vio_i_vec[i] = 1.0 - sum_x_iu;
    }
    for (int u = 0; u < n; ++u) {
        double sum_x_iu = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_x_iu += X_star(i*n + u, i*n + u);
        }
        vio_u_vec[u] = 1.0 - sum_x_iu;
    }
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                double sum_y = 0.0;
                for (int v = 0; v < n; ++v) {
                    sum_y += X_star(i*n + u, j*n + v);
                }
                vio_iuj_vec[i*n*n + u*n + j] = X_star(i*n + u, i*n + u) - sum_y;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                double sum_y = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum_y += X_star(i*n + u, j*n + v);
                }
                vio_iuv_vec[i*n*n + u*n + v] = X_star(i*n + u, i*n + u) - sum_y;
            }
        }
    }
    // print maximum absolute constraint violation and mean absolute constraint violation
    std::cout << "Constraint violations (b - A(X_star)):" << std::endl;
    std::cout << "Max absolute violation: " << vio_i_vec.lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << "Mean absolute violation: " << vio_i_vec.lpNorm<1>() / vio_i_vec.size() << std::endl;
    std::cout << "Max absolute violation: " << vio_u_vec.lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << "Mean absolute violation: " << vio_u_vec.lpNorm<1>() / vio_u_vec.size() << std::endl;
    std::cout << "Max absolute violation: " << vio_iuj_vec.lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << "Mean absolute violation: " << vio_iuj_vec.lpNorm<1>() / vio_iuj_vec.size() << std::endl;
    std::cout << "Max absolute violation: " << vio_iuv_vec.lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << "Mean absolute violation: " << vio_iuv_vec.lpNorm<1>() / vio_iuv_vec.size() << std::endl;




    return 0;
}

bool SDPVolumeHooks6::on_small_improvement() {
    if (eigen_solver != "lanczos") {
        return false; // Only switch to Lanczos if not already using it
    }

    auto new_ncv = std::min(std::max(int(m_ncv*1.5), m_ncv+1), n * n);
    auto new_tol = std::max(m_tol*0.8, 1e-6);
    if(new_ncv == m_ncv && new_tol == m_tol) {
        eigen_solver = "full";
    }
    else{

        m_ncv = new_ncv;
        m_tol = new_tol;
        std::cout << "Changing ncv = " << m_ncv << " and tol = " << m_tol << std::endl;
    }
    return true;

}

/*
    * Solving the subproblem with Lanczos method (approximate)
    *   min <C,X> + <pi, b - A(X)> 
    * = min <C - ATpi, X> + <b,pi>
    * subject to:
    *   X is PSD, trace X = alpha
    * compute min eigenvalue of C - ATpi lambda
    * and corresponding eigenvector v
    * X* = alpha*vv^T
    * lcost = alpha*lambda + <b,pi> = <C - ATpi, X*> + <b,pi>
    * pcost = <C, X*> = <C, vv^T>
    * vio = b - A(X*) = b - A(vv^T)
    * pi [2*n + 2*n*n*n + n*n*n*n]
    * constraints:
    *   0                       -   n                               sum_i x_iu = 1 for all u
    *   n                       -   2*n                             sum_u x_iu = 1 for all i
    *   2*n                     -   2*n + n*n*n                     sum_j y[i,u,j,v] = x_iu for all i,u,v
    *   2*n + n*n*n             -   2*n + 2*n*n*n                   sum_v y[i,u,j,v] = x_iu for all i,u,j
    *   2*n + 2*n*n*n           -   2*n + 2*n*n*n + n*n*n*n         y_iujv >= 0 for all i,u,j,v
    * indexing:
    *   X is a matrix of size n^2 x n^2
    * C = B kron A
*/    
int SDPVolumeHooks6::solve_subproblem_6(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    // Alternative subproblem solver for SDP (Volume algorithm)
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    vio = 0.0;  // Reset constraint violations for the current subproblem

    {
        int pi_u_offset = 0;
        int pi_i_offset = n;
        int pi_iuv_offset = 2*n;
        int pi_iuj_offset = 2*n + n*n*n;
        int pi_iujv_offset = 2*n + 2*n*n*n;
        // compute C - AT*pi in the lifted formulation X = [Y x; x^T 1],
        // where x stores the assignment variables and Y stores the y variables.
        // 1. Initialize to Zero
        Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n * n, n * n);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx1 = i * n + u;
                
                // Loop bounds strictly restricted to the upper triangle to save 50% effort
                for (int j = i; j < n; ++j) {
                    int v_start = (j == i) ? u : 0;
                    #pragma omp simd
                    for (int v = v_start; v < n; ++v) {
                        int idx2 = j * n + v;
                        
                        if (idx1 != idx2) {
                            // Retrieve the compressed 1D variable via our helper
                            int index = get_quad_index(pi_iujv_offset, i, u, j, v, n);
                            double current_pi_iujv = pi[index];

                            double AT_pi_f = pi_iuj(i, u, j) + pi_iuv(i, u, v);
                            double AT_pi_b = pi_iuj(j, v, i) + pi_iuv(j, v, u); 

                            // Symmetrize and apply the split factor for the off-diagonal multiplier
                            double val = qap_data.D[i][j] * qap_data.F[u][v] - 0.5 * (AT_pi_f + AT_pi_b) - 0.5 * current_pi_iujv;
                            
                            // Thread-safe writes: distinct (idx1, idx2) pairs belong to exactly one thread assignment
                            C_minus_AT_pi(idx1, idx2) = val;
                            C_minus_AT_pi(idx2, idx1) = val;
                        }
                    }
                }
            }
        }

        // 2. Build DIAGONAL elements in parallel
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx = i * n + u;
                
                int index = get_quad_index(pi_iujv_offset, i, u, i, u, n);
                double current_pi_iujv_diag = pi[index];

                // Every thread gets its own isolated local accumulator copy
                double AT_pi = pi_i(i) + pi_u(u) + pi_iuj(i, u, i) + pi_iuv(i, u, u) + current_pi_iujv_diag;
                
                #pragma omp simd reduction(+:AT_pi)
                for (int j = 0; j < n; ++j) {
                    AT_pi -= pi_iuj(i, u, j) + pi_iuv(i, u, j);
                }
                
                // Thread-safe write: each diagonal position is completely unique to the (i, u) pair
                C_minus_AT_pi(idx, idx) = qap_data.D[i][i] * qap_data.F[u][u] - AT_pi;
            }
        }


     
        // --- 3. EIGENVALUE DECOMPOSITION SELECTOR ---
        double lambda = 0.0;
	    Eigen::VectorXd v;
        if (eigen_solver == "power_iteration")
        {
            
            int matrix_dim = C_minus_AT_pi.rows();
            // Compute a fast, conservative upper bound using the infinity norm (max row sum)
            double max_ev_estimate = 0.0;
            #pragma omp parallel for reduction(max:max_ev_estimate)
            for (int i = 0; i < matrix_dim; ++i) {
                double row_sum = C_minus_AT_pi.row(i).cwiseAbs().sum();
                if (row_sum > max_ev_estimate) max_ev_estimate = row_sum;
            }
            
            // Apply spectral shift buffer to safely isolate the absolute smallest eigenvalue
            double mu = max_ev_estimate + 10.0; 

            v = Eigen::VectorXd::Random(matrix_dim);
            v.normalize();

            const int max_iter = 150;
            const double tol = 1e-5;
            double lambda_old = 0.0;
            bool converged = false;

            Eigen::VectorXd next_v = Eigen::VectorXd::Zero(matrix_dim);

            for (int iter = 0; iter < max_iter; ++iter) {
                // Shifted mapping: next_v = (mu * I - A) * v = mu * v - A * v
                // Eigen parallelizes this matrix-vector multiplication natively via OpenMP
                next_v.noalias() = mu * v - C_minus_AT_pi * v;

                double norm = next_v.norm();
                if (norm < 1e-12) break; 
                v = next_v / norm;

                // Evaluate Rayleigh Quotient for the unshifted target matrix
                double current_lambda = v.dot(C_minus_AT_pi * v);

                if (iter > 0 && std::abs(current_lambda - lambda_old) < tol) {
                    lambda = current_lambda;
                    converged = true;
                    break;
                }
                lambda_old = current_lambda;
            }

            // Secure production fallback if the power loop fails to hit target tolerance
            if (!converged) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else if (eigen_solver == "lanczos")
        {
            Spectra::DenseSymMatProd<double> op(C_minus_AT_pi);
            int matrix_dim = C_minus_AT_pi.rows();


            // 1. CAP ITERATIONS AT 20 FOR BLAZING FAST SUBGRADIENT ESTIMATES
            // 2. We keep the fallback checking 'nconv > 0' because as long as it has computed 
            //    at least 1 rough estimate, it's valid for subproblem optimization bounds!
            int nconv = 0;
            auto ncv = m_ncv;
            auto tol = m_tol;
            while (nconv <= 0 && ncv < matrix_dim) {
                Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, ncv);
                eigs.init();
                nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 1, tol);
                if (nconv <= 0) {
                    ncv = std::min(std::max(int(ncv*1.1), ncv+1), n * n);
                    // tol = std::min(tol*1.1, m_tol);
                    std::cout << "Lanczos solver failed to converge, increasing ncv to " << ncv  << std::endl;
                } else {
                    m_ncv = ncv;
                    lambda = eigs.eigenvalues()(0);
                    v = eigs.eigenvectors().col(0);
                    // std::cout << "Lanczos solver converged with ncv = " << ncv << " and tol = " << tol << std::endl;
                    break;
                }
            }
            if (v.size() == 0) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
                std::cout << "Warning: Lanczos solver failed to converge, using fallback eigenvalue decomposition." << std::endl;
            }
        }
        else // default/exact dense fallback option
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigenvalue decomposition failed!" << std::endl;
                return -1;
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                pcost += qap_data.D[i][i] * qap_data.F[u][u] * X(i*n + u, i*n + u);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += qap_data.D[i][j] * qap_data.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (qap_data.D[i][j] * qap_data.F[u][v] + qap_data.D[j][i] * qap_data.F[v][u]) * X(i*n + u, j*n + v);
                        }
                    }
                }
            }
        }
        // std::cout << "Min eigenvalue: " << lambda  << ", Pcost: " << pcost << std::endl;
        
        {
            int vio_u_offset = 0;
            int vio_i_offset = n;
            int vio_iuv_offset = 2*n;
            int vio_iuj_offset = 2*n + n*n*n;
            int vio_iujv_offset = 2*n + 2*n*n*n;
            // vio_u(u) = 1 - sum_i x_iu for all u
            #pragma omp parallel for schedule(static)
            for (int u = 0; u < n; ++u) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int i = 0; i < n; ++i) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_u(u) = acc;
            }
            // vio_i(i) = 1 - sum_u x_iu for all i
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int u = 0; u < n; ++u) {
                    acc -= X(i * n + u, i * n + u);
                }
                vio_i(i) = acc;
            }
            // vio_iuj(i, u, j) = x_iu - sum_v y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int j = 0; j < n; ++j) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int v = 0; v < n; ++v) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuj(i, u, j) = base_val - sum_y;
                    }
                }
            }
            // vio_iuv(i, u, v) = x_iu - sum_j y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        double base_val = X(i * n + u, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int j = 0; j < n; ++j) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuv(i, u, v) = base_val - sum_y;
                    }
                }
            }
            // vio_iujv(i, u, j, v) = -y[i,u,j,v] ONLY for upper triangle (idx1 <= idx2)
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    int idx1 = i * n + u;
                    
                    for (int j = i; j < n; ++j) {
                        int v_start = (j == i) ? u : 0;
                        
                        #pragma omp simd
                        for (int v = v_start; v < n; ++v) {
                            int idx2 = j * n + v;
                            int index = get_quad_index(vio_iujv_offset, i, u, j, v, n);
                            vio[index] = -X(idx1, idx2);
                            psol[index - vio_iujv_offset] = X(idx1, idx2);
                        }
                    }
                }
            }


        }
        lcost = lambda*alpha;
        // #pragma omp parallel for schedule(static) 
        for (int i= 0; i < n; ++i) {
            lcost += pi_i(i) + pi_u(i);
        }

    }
    // if(iteration % 100 == 0)
    // {
    //     // compute pcost from vol_problem.psol.v
    //     auto pcost_2 = 0.0;
    //     auto psol_ptr = vol_problem.psol.v;
    //     for (int i = 0; i < n; ++i) 
    //         for (int u = 0; u < n; ++u) 
    //             for (int j = 0; j < n; ++j) 
    //                 for (int v = 0; v < n; ++v) {
    //                     int index = get_quad_index(0, i, u, j, v, n);
    //                     pcost_2 += qap_data.D[i][j] * qap_data.F[u][v] * psol_ptr[index];
    //                 } 
    //     std::cout << "Pcost from psol: " << pcost_2 << ", Pcost from X: " << pcost << std::endl;
    // }
    // iteration++;

    return 0;
}


SDPVolumeHooks7::SDPVolumeHooks7(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver)
    : SDPVolumeHooks(data, vol_prob), n(data.n), alpha(data.n), eigen_solver(eigen_solver)
{

}

int SDPVolumeHooks7::solve_subproblem(const VOL_dvector& pi, const VOL_dvector& /*rc*/,
                            double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                            double& pcost) {
                                return solve_subproblem_7(pi, lcost, psol, vio, pcost);
                            }
    
bool SDPVolumeHooks7::set_fixed_variables(const FixedVariables &fv) {
    fixed = fv;
    return true;
}



#define pi_iuiu(i,u) pi[pi_iuiu_offset + (i)*n + (u)]
#define vio_iuiu(i,u) vio[vio_iuiu_offset + (i)*n + (u)]
/*
    * Solving the subproblem with Lanczos method (approximate)
    *   min <C,X> + <pi, b - A(X)> 
    * = min <C - ATpi, X> + <b,pi>
    * subject to:
    *   X is PSD, trace X = alpha
    * compute min eigenvalue of C - ATpi lambda
    * and corresponding eigenvector v
    * X* = alpha*vv^T
    * lcost = alpha*lambda + <b,pi> = <C - ATpi, X*> + <b,pi>
    * pcost = <C, X*> = <C, vv^T>
    * vio = b - A(X*) = b - A(vv^T)
    * pi [2*n + 2*n*n*n + n*n*n*n]
    * constraints:
    *   0                       -   n                               sum_i y_iuiu = 1 for all u
    *   n                       -   2*n                             sum_u y_iuiu = 1 for all i
    *   2*n                     -   2*n + n*n*n                     sum_j y[i,u,j,v] = y_iuiu for all i,u,v
    *   2*n + n*n*n             -   2*n + 2*n*n*n                   sum_v y[i,u,j,v] = y_iuiu for all i,u,j
    *   2*n + 2*n*n*n           -   2*n + 2*n*n*n + n*n             y[i,u,i,u] = x_iu for all i,u
    *   2*n + 2*n*n*n + n*n     -   2*n + 2*n*n*n + n*n + n*n*n*n   y_iujv >= 0 for all i,u,j,v
    * indexing:
    *   X is a matrix of size n^2+1 x n^2+1
    *   x_iu = X(n*n, i*n + u) = X(i*n + u, n*n)
    * C = [B kron A 0; 0 0]
*/    
int SDPVolumeHooks7::solve_subproblem_7(const VOL_dvector& pi,
                        double& lcost, VOL_dvector& psol, VOL_dvector& vio,
                        double& pcost) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // Alternative subproblem solver for SDP (Volume algorithm)
    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    psol = 0.0; // Set all primal variables to zero
    vio = 0.0;  // Reset constraint violations for the current subproblem

    {
        int pi_u_offset = 0;
        int pi_i_offset = n;
        int pi_iuv_offset = 2*n;
        int pi_iuj_offset = 2*n + n*n*n;
        int pi_iuiu_offset = 2*n + 2*n*n*n;
        int pi_iujv_offset = 2*n + 2*n*n*n + n*n;
        // compute C - AT*pi in the lifted formulation X = [Y x; x^T 1],
        // where x stores the assignment variables and Y stores the y variables.
        // 1. Initialize to Zero
        Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n*n+1, n*n+1);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx1 = i * n + u;
                
                // Loop bounds strictly restricted to the upper triangle to save 50% effort
                for (int j = i; j < n; ++j) {
                    int v_start = (j == i) ? u : 0;
                    #pragma omp simd
                    for (int v = v_start; v < n; ++v) {
                        int idx2 = j * n + v;
                        
                        if (idx1 != idx2) {
                            // Retrieve the compressed 1D variable via our helper
                            int index = get_quad_index(pi_iujv_offset, i, u, j, v, n);
                            double current_pi_iujv = pi[index];

                            double AT_pi_f = pi_iuj(i, u, j) + pi_iuv(i, u, v);
                            double AT_pi_b = pi_iuj(j, v, i) + pi_iuv(j, v, u); 

                            // Symmetrize and apply the split factor for the off-diagonal multiplier
                            double val = qap_data.D[i][j] * qap_data.F[u][v]
                                        - 0.5 * (AT_pi_f + AT_pi_b)
                                        - 0.5 * current_pi_iujv;
                            
                            // Thread-safe writes: distinct (idx1, idx2) pairs belong to exactly one thread assignment
                            C_minus_AT_pi(idx1, idx2) = val;
                            C_minus_AT_pi(idx2, idx1) = val;
                        }
                    }
                }
            }
        }

        // 2. Build DIAGONAL elements in parallel
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx = i * n + u;
                
                int index = get_quad_index(pi_iujv_offset, i, u, i, u, n);
                double current_pi_iujv_diag = pi[index];

                // Every thread gets its own isolated local accumulator copy
                double AT_pi = pi_i(i) + pi_u(u)
                                + pi_iuj(i, u, i) + pi_iuv(i, u, u)
                                - pi_iuiu(i, u)
                                + current_pi_iujv_diag ;
                
                #pragma omp simd reduction(+:AT_pi)
                for (int j = 0; j < n; ++j) {
                    AT_pi -= pi_iuj(i, u, j) + pi_iuv(i, u, j);
                }
                
                // Thread-safe write: each diagonal position is completely unique to the (i, u) pair
                C_minus_AT_pi(idx, idx) = qap_data.D[i][i] * qap_data.F[u][u] - AT_pi;
                C_minus_AT_pi(idx, n*n) = -0.5*pi_iuiu(i,u);
                C_minus_AT_pi(n*n, idx) = C_minus_AT_pi(idx, n*n);

                // C_minus_AT_pi(idx, n*n) = 0.5 * (qap_data.D[i][i] * qap_data.F[u][u] - AT_pi);
                // C_minus_AT_pi(n*n, idx) = C_minus_AT_pi(idx, n*n);
                // C_minus_AT_pi(idx, idx) = -pi_iuiu(i,u);
            }
        }
        C_minus_AT_pi(n*n, n*n) = 1.0;

     
        // --- 3. EIGENVALUE DECOMPOSITION SELECTOR ---
        double lambda = 0.0;
	    Eigen::VectorXd v;
        if (eigen_solver == "power_iteration")
        {
            
            int matrix_dim = C_minus_AT_pi.rows();
            // Compute a fast, conservative upper bound using the infinity norm (max row sum)
            double max_ev_estimate = 0.0;
            #pragma omp parallel for reduction(max:max_ev_estimate)
            for (int i = 0; i < matrix_dim; ++i) {
                double row_sum = C_minus_AT_pi.row(i).cwiseAbs().sum();
                if (row_sum > max_ev_estimate) max_ev_estimate = row_sum;
            }
            
            // Apply spectral shift buffer to safely isolate the absolute smallest eigenvalue
            double mu = max_ev_estimate + 10.0; 

            v = Eigen::VectorXd::Random(matrix_dim);
            v.normalize();

            const int max_iter = 150;
            const double tol = 1e-5;
            double lambda_old = 0.0;
            bool converged = false;

            Eigen::VectorXd next_v = Eigen::VectorXd::Zero(matrix_dim);

            for (int iter = 0; iter < max_iter; ++iter) {
                // Shifted mapping: next_v = (mu * I - A) * v = mu * v - A * v
                // Eigen parallelizes this matrix-vector multiplication natively via OpenMP
                next_v.noalias() = mu * v - C_minus_AT_pi * v;

                double norm = next_v.norm();
                if (norm < 1e-12) break; 
                v = next_v / norm;

                // Evaluate Rayleigh Quotient for the unshifted target matrix
                double current_lambda = v.dot(C_minus_AT_pi * v);

                if (iter > 0 && std::abs(current_lambda - lambda_old) < tol) {
                    lambda = current_lambda;
                    converged = true;
                    break;
                }
                lambda_old = current_lambda;
            }

            // Secure production fallback if the power loop fails to hit target tolerance
            if (!converged) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else if (eigen_solver == "lanczos")
        {
            Spectra::DenseSymMatProd<double> op(C_minus_AT_pi);
            int matrix_dim = C_minus_AT_pi.rows();
            int ncv = std::min(2, matrix_dim);
            if (ncv <= 2) ncv = 3;

            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, ncv);
            eigs.init();

            // 1. CAP ITERATIONS AT 20 FOR BLAZING FAST SUBGRADIENT ESTIMATES
            // 2. We keep the fallback checking 'nconv > 0' because as long as it has computed 
            //    at least 1 rough estimate, it's valid for subproblem optimization bounds!
            int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 20, 1e-5);

            if (nconv > 0) {
                lambda = eigs.eigenvalues()(0);
                v = eigs.eigenvectors().col(0);
            } else {
                // Fail-safe fallback ONLY if the solver completely errors out (e.g., returns NaN)
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
                Eigen::Index minIndex;
                lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
                v = eigensolver.eigenvectors().col(minIndex);
            }
        }
        else // default/exact dense fallback option
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_minus_AT_pi);
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigenvalue decomposition failed!" << std::endl;
                return -1;
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                // pcost += qap_data.D[i][i] * qap_data.F[u][u] * X(i*n + u, i*n + u);
                pcost += qap_data.D[i][i] * qap_data.F[u][u] * X(i*n + u, n*n);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += qap_data.D[i][j] * qap_data.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (qap_data.D[i][j] * qap_data.F[u][v] + qap_data.D[j][i] * qap_data.F[v][u]) * X(i*n + u, j*n + v);
                        }
                    }
                }
            }
        }
        // std::cout << "Min eigenvalue: " << lambda  << ", Pcost: " << pcost << std::endl;
        
        {
            int vio_u_offset = 0;
            int vio_i_offset = n;
            int vio_iuv_offset = 2*n;
            int vio_iuj_offset = 2*n + n*n*n;
            int vio_iuiu_offset = 2*n + 2*n*n*n;
            int vio_iujv_offset = 2*n + 2*n*n*n + n*n;
            // vio_u(u) = 1 - sum_i x_iu for all u
            #pragma omp parallel for schedule(static)
            for (int u = 0; u < n; ++u) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int i = 0; i < n; ++i) {
                    acc -= X(i * n + u, i * n + u);
                    // acc -= X(n*n, i * n + u);
                }
                vio_u(u) = acc;
            }
            // vio_i(i) = 1 - sum_u x_iu for all i
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double acc = 1.0;
                #pragma omp simd reduction(-:acc)
                for (int u = 0; u < n; ++u) {
                    acc -= X(i * n + u, i * n + u);
                    // acc -= X(n*n, i * n + u);
                }
                vio_i(i) = acc;
            }
            // vio_iuj(i, u, j) = x_iu - sum_v y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int j = 0; j < n; ++j) {
                        double base_val = X(i * n + u, i * n + u);
                        // double base_val = X(n*n, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int v = 0; v < n; ++v) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuj(i, u, j) = base_val - sum_y;
                    }
                }
            }
            // vio_iuv(i, u, v) = x_iu - sum_j y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        double base_val = X(i * n + u, i * n + u);
                        // double base_val = X(n*n, i * n + u);
                        double sum_y = 0.0;
                        
                        #pragma omp simd reduction(+:sum_y)
                        for (int j = 0; j < n; ++j) {
                            sum_y += X(i * n + u, j * n + v);
                        }
                        vio_iuv(i, u, v) = base_val - sum_y;
                    }
                }
            }
            // vio_iuiu(i, u) = x_iu - y[i,u,i,u]
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    vio_iuiu(i, u) = +X(n*n, i * n + u) - X(i * n + u, i * n + u);
                }
            }
            // vio_iujv(i, u, j, v) = -y[i,u,j,v] ONLY for upper triangle (idx1 <= idx2)
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    int idx1 = i * n + u;
                    
                    for (int j = i; j < n; ++j) {
                        int v_start = (j == i) ? u : 0;
                        
                        #pragma omp simd
                        for (int v = v_start; v < n; ++v) {
                            int idx2 = j * n + v;
                            int index = get_quad_index(vio_iujv_offset, i, u, j, v, n);
                            vio[index] = -X(idx1, idx2);
                        }
                    }
                }
            }


        }
        lcost = lambda*alpha;
        // #pragma omp parallel for schedule(static) 
        for (int i= 0; i < n; ++i) {
            lcost += pi_i(i) + pi_u(i);
        }

    }

    auto end_time = std::chrono::high_resolution_clock::now();
    subproblem_solving_time += (end_time - start_time);

    return 0;
}



void set_up_volume_parameters(VOL_problem &vol_problem, bool verbose) {
    // Set Volume algorithm parameters (from qap.par defaults)
    vol_problem.parm.lambdainit = 10.0;
    vol_problem.parm.alphainit = 0.9;
    vol_problem.parm.alphamin = 0.0001;
    vol_problem.parm.alphafactor = 0.66;
    vol_problem.parm.alphaint = 50;
    
    vol_problem.parm.maxsgriters = 300; // effectively no limit
    vol_problem.parm.primal_abs_precision = 0.001;
    vol_problem.parm.gap_abs_precision = 0.0;
    vol_problem.parm.gap_rel_precision = 0.001;
    vol_problem.parm.granularity = 0.0;
    
    vol_problem.parm.ascent_first_check = 5000;
    vol_problem.parm.ascent_check_invl = 5000;
    vol_problem.parm.minimum_rel_ascent = 0.0001;
    
    vol_problem.parm.greentestinvl = 1;
    vol_problem.parm.yellowtestinvl = 4;
    vol_problem.parm.redtestinvl = 20;
    
    // Printing control
    vol_problem.parm.printflag = verbose ? 3 : 0;  // 1=iteration info, 3=add lambda info
    vol_problem.parm.printinvl = 1;
    vol_problem.parm.heurinvl = 100;
}

void set_up_vol_problem_1(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    // Primal variables: x[i,u] (n*n) + y[i,u,j,v] (n^4)
    vol_problem.psize = 1;

    // Dual variables
    vol_problem.dsize = n * n * n * n;
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = 0.0;
    vol_problem.dual_ub = DBL_MAX;
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;
}

void set_up_vol_problem_2(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    vol_problem.psize = 1;
    
    // Dual variables:
    // vol_problem.dsize = n*n*n*n; 
    int n2 = n * n;
    int num_upper_tri_constraints = (n2 * (n2 + 1)) / 2;

    int total_size = 2 * n + 2 * n * n * n + num_upper_tri_constraints;
    vol_problem.dsize = total_size; 
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = 0.0;
    vol_problem.dual_ub = DBL_MAX;
    // vol_problem.dual_lb = -DBL_MAX;
    // vol_problem.dual_ub = 0.0;

    for (int i = 0; i < 2*n + 2*n*n*n; ++i) {
        vol_problem.dual_lb[i] = -DBL_MAX; // first 2*n + 2*n*n*n are free variables
    }
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 1.0;
    // for (int i = 2*n + 2*n*n*n; i < vol_problem.dsize; ++i) {
    //     vol_problem.dsol[i] = 10; // first 2*n + 2*n*n*n are free variables
    // }



}

void set_up_vol_problem_3(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    vol_problem.psize = 1;
    
    // Dual variables:
    // vol_problem.dsize = n*n*n*n; 
    int n2 = n * n;
}

void set_up_vol_problem_4(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    vol_problem.psize = 1;
    
    // Dual variables:
    // vol_problem.dsize = n*n*n*n; 
    int total_size = 2*n + 2*n*n*n + n*n + 2*n*n*n;
    vol_problem.dsize = total_size; 
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    // vol_problem.dual_lb = -DBL_MAX;
    // vol_problem.dual_ub = 0.0;

    for (int i = 2*n + 2*n*n*n; i < 2*n + 2*n*n*n + n*n; ++i) {
        vol_problem.dual_lb[i] = 0.0; 
    }
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;

}

void set_up_vol_problem_5(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    vol_problem.psize = 1;
    
    // Dual variables:
    int total_size = (n*n+1)*(n*n+1);
    vol_problem.dsize = total_size; 
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = -DBL_MAX;
    vol_problem.dual_ub = DBL_MAX;
    // vol_problem.dual_lb = -DBL_MAX;
    // vol_problem.dual_ub = 0.0;

    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 1.0;
    // for (int i = 2*n + 2*n*n*n; i < vol_problem.dsize; ++i) {
    //     vol_problem.dsol[i] = 10; // first 2*n + 2*n*n*n are free variables
    // }



}

void set_up_vol_problem_6(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Dual variables:
    // vol_problem.dsize = n*n*n*n; 
    int n2 = n * n;
    int num_upper_tri_constraints = (n2 * (n2 + 1)) / 2;

    int total_size = 2 * n + 2 * n * n * n + num_upper_tri_constraints;
    vol_problem.dsize = total_size; 
    // Set problem dimensions
    vol_problem.psize = num_upper_tri_constraints;
    
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = 0.0;
    vol_problem.dual_ub = DBL_MAX;
    // vol_problem.dual_lb = -DBL_MAX;
    // vol_problem.dual_ub = 0.0;

    for (int i = 0; i < 2*n + 2*n*n*n; ++i) {
        vol_problem.dual_lb[i] = -DBL_MAX; // first 2*n + 2*n*n*n are free variables
    }
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 1.0;
    // for (int i = 2*n + 2*n*n*n; i < vol_problem.dsize; ++i) {
    //     vol_problem.dsol[i] = 10; // first 2*n + 2*n*n*n are free variables
    // }



}

void set_up_vol_problem_7(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed) {
    int n = qap_data.n;

    // Set problem dimensions
    vol_problem.psize = 1;
    
    // Dual variables:
    // vol_problem.dsize = n*n*n*n; 
    int n2 = n * n;
    int num_upper_tri_constraints = (n2 * (n2 + 1)) / 2;

    int total_size = 2*n + 2*n*n*n + n*n + num_upper_tri_constraints;
    vol_problem.dsize = total_size; 
    
    // Set dual bounds (all free variables)
    vol_problem.dual_lb.allocate(vol_problem.dsize);
    vol_problem.dual_ub.allocate(vol_problem.dsize);
    vol_problem.dual_lb = 0.0;
    vol_problem.dual_ub = DBL_MAX;

    for (int i = 0; i < 2*n + 2*n*n*n; ++i) {
        vol_problem.dual_lb[i] = -DBL_MAX; // first 2*n + 2*n*n*n are free variables
    }
    
    // Initialize dual solution to zero
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 1.0;



}



Eigen::MatrixXd convert_nested_vector_to_eigen(const std::vector<std::vector<double>>& vec_2d) {
    if (vec_2d.empty() || vec_2d[0].empty()) return Eigen::MatrixXd(0, 0);

    int rows = vec_2d.size();
    int cols = vec_2d[0].size();

    // Allocate an explicit Row-Major target matrix in Eigen
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(rows, cols);

    // Map each vector row directly to its matching matrix row slot
    for (int i = 0; i < rows; ++i) {
        mat.row(i) = Eigen::Map<const Eigen::VectorXd>(vec_2d[i].data(), cols);
    }

    return mat; // Implicitly returns as standard Eigen::MatrixXd (Column-Major)
}
