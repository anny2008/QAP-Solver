#include "sdp_bundle_solver.h"
#include "Hungarian.h"
#include <omp.h>
#include "utils.h"
#include <vector>

void SDPOracle::SetUp()
{
    iteration = 0;
    best_fi = -std::numeric_limits<double>::infinity();
    last_checkpoint_fi = -std::numeric_limits<double>::infinity();
    if (formulation == "formulation1") {
        SetUp1();
    } else if (formulation == "formulation2") {
        SetUp2();
    } else if (formulation == "formulation3") {
        SetUp3();
    } else if (formulation == "formulation4") {
        SetUp4();
    } else {
        std::cerr << "Error: Unknown formulation type: " << formulation << std::endl;
        exit(1);
    }
}

bool SDPOracle::GetUC( cIndex index )
{
    if (formulation == "formulation1") {
        return GetUC1(index);
    } else if (formulation == "formulation2") {
        return GetUC2(index);
    } else if (formulation == "formulation3") {
        return GetUC3(index);
    } else if (formulation == "formulation4") {
        return GetUC4(index);
    } else {
        std::cerr << "Error: Unknown formulation type: " << formulation << std::endl;
        exit(1);
    }
}

bool SDPOracle::GetUC1( cIndex index )
{
    // all variable are non-negative
    return false;
}

bool SDPOracle::GetUC2( cIndex index )
{
    // size = 2 * n + 2 * n * n * n + num_upper_tri_constraints
    // The first 2*n + 2*n*n*n variables are free, the rest are non-negative
    return index < (2*n + 2*n*n*n);
}

bool SDPOracle::GetUC3( cIndex index )
{
    // The first 2*n + 2*n*n*n variables are free, the rest are non-negative
    return index < (2*n + 2*n*n*n);
}

bool SDPOracle::GetUC4( cIndex index )
{
    // size =2*n + 2*n*n*n + n*n + 2*n*n*n
    // The first 2*n + 2*n*n*n variables are free
    // The next n*n variables are non-negative
    // The last 2*n*n*n variables are free
    return (index < (2*n + 2*n*n*n) || index >= (2*n + 2*n*n*n + n*n));
    // return true; // all variables are free in formulation4
}


LMNum SDPOracle::GetUB( cIndex index )
{
    return Inf< LMNum >();
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
void SDPOracle::SetUp1()
{
    vio.resize(n*n*n*n, 0.0);
    NumVar = static_cast<Index>(vio.size());
    dsize = n*n*n*n;

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

    Eigen::MatrixXd A = convert_nested_vector_to_eigen(problem.D);
    Eigen::MatrixXd B = convert_nested_vector_to_eigen(problem.F);
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
    auto W_monty = std::make_shared<ndarray<double, 2>>(
        shape(static_cast<int>(W_row_major.rows()), static_cast<int>(W_row_major.cols()))
    );
    auto L_monty = std::make_shared<ndarray<double, 2>>(
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


}

void SDPOracle::SetUp2()
{
    int n2 = n * n;
    int num_upper_tri_constraints = (n2 * (n2 + 1)) / 2;
    int total_size = 2 * n + 2 * n * n * n + num_upper_tri_constraints;
    
    vio.resize(total_size, 0.0);
    NumVar = static_cast<Index>(vio.size());
    dsize = total_size;
    alpha = n;
}

void SDPOracle::SetUp3()
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
                    double cond = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                    
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


    dsize = 2*n + 2*n*n*n + active_quad_count;
    vio.resize(dsize, 0.0);
    NumVar = static_cast<Index>(vio.size());
    alpha = n;
}

void SDPOracle::SetUp4()
{
    dsize = 2*n + 2*n*n*n + n*n + 2*n*n*n;
    vio.resize(dsize, 0.0);
    NumVar = static_cast<Index>(vio.size());
    alpha = n;
}

HpNum SDPOracle::Fi(cIndex wFi)
{
    if (wFi == 0)
        return 0.0;

    if (needs_eval) {
        SolveSubproblem();
        needs_eval = false;
    }

    return -lcost;
}

Index SDPOracle::GetGi(
        SgRow SubG,
        cIndex_Set &SGBse,
        cIndex Name,
        cIndex strt,
        Index stp)
{
    if(Name == MaxName)
    {
        Index end = std::min(stp, NumVar);

        std::fill(SubG,
                  SubG + (end - strt),
                  0.0);

        SGBse = nullptr;
        return end - strt;
    }

    if (needs_eval) {
        SolveSubproblem();
        needs_eval = false;
    }

    Index begin = std::max<Index>(0, strt);
    Index end   = std::min<Index>(NumVar, stp);

    for(Index i = begin; i < end; ++i)
        SubG[i - begin] = -vio[i];

    SGBse = nullptr;
    LHasChgd = false;
    return end - begin;
}

void SDPOracle::SolveSubproblem()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    if (formulation == "formulation1") {
        SolveSubproblem1();
    } else if (formulation == "formulation2") {
        SolveSubproblem2();
    } else if (formulation == "formulation3") {
        SolveSubproblem3();
    } else if (formulation == "formulation4") {
        SolveSubproblem4();
    } else {
        std::cerr << "Error: Unknown formulation type: " << formulation << std::endl;
        exit(1);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    subproblem_solving_time += (end_time - start_time);
    // SolveSubproblemTest();
    if (lcost > best_fi) {
        best_fi = lcost;
    }
    // if after 500 iterations the cost has not improved, stop the algorithm
    if(iteration - last_checkpoint_iteration >= 1000) {
        double improvement = lcost - last_checkpoint_fi;
        if(improvement/(lcost+1e-16) < 1e-6) {
            std::cout << "No significant improvement in Fi for 1000 iterations. Stopping." << std::endl;
            mFiStatus = kFiStop;
        } else {
            last_checkpoint_iteration = iteration;
            last_checkpoint_fi = lcost;
        }
    } else if(iteration == 0) {
        last_checkpoint_iteration = iteration;
        last_checkpoint_fi = lcost;
    }

    if(iteration % 100 == 0) {
        auto ellapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << "Iteration: " << iteration << ", Fi: " << lcost << ", Best Fi: " << best_fi << ", Time: " << ellapsed << " seconds" << std::endl;
    }
    iteration++;
}


void SDPOracle::SolveSubproblem1() 
{
    SolveSubproblem1MOSEK();
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
void SDPOracle::SolveSubproblem1MOSEK()
{
    RetrieveDensePi(); // Ensure DensePi is up-to-date with current Lambda
    // auto pi = DensePi.data(); // Pointer to the dense dual variables

    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    
    //  Step 2: Update the objective function in MOSEK with the current dual variables
    {
        // std::cout << "W matrix size: " << W.rows() << " x " << W.cols() << std::endl;

        // std::cout << "L matrix size: " << L.rows() << " x " << L.cols() << std::endl;
        // std::cout << "DensePi size: " << DensePi.size() << std::endl;
        // compute W^T * gamma * W and add it to L to form the new objective matrix
        Eigen::Map<Eigen::MatrixXd> gamma(DensePi.data(), n*n, n*n); // Assuming DensePi is of size W.cols() * W.cols()
        auto W_gamma_W = W.transpose() * gamma * W; // Compute W^T * gamma * W
        Eigen::MatrixXd new_L = L - W_gamma_W; // New objective matrix

        RowMajorMatrix L_row_major = new_L;
        auto L_monty = std::make_shared<ndarray<double, 2>>(
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
        vio.assign(result_matrix.data(), result_matrix.data() + result_matrix.size());

    }


    
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
    *   C = B kron A
*/
void SDPOracle::SolveSubproblem2()
{
    
    RetrieveDensePi(); // Ensure DensePi is up-to-date with current Lambda
    auto pi = DensePi;
    lcost = 0.0;
    pcost = 0.0;
    dsize = 2*n + 2*n*n*n + n*n*n*n;
    std::fill(vio.begin(), vio.end(), 0.0); // Set all violation variables to zero
    
    Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n * n, n * n);

    {
        int pi_u_offset = 0;
        int pi_i_offset = n;
        int pi_iuv_offset = 2*n;
        int pi_iuj_offset = 2*n + n*n*n;
        int pi_iujv_offset = 2*n + 2*n*n*n;
        int pi_iuiu_offset = 2*n + 2*n*n*n + n*n*n*n;
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
                    
                    for (int v = v_start; v < n; ++v) {
                        int idx2 = j * n + v;
                        
                        if (idx1 != idx2) {
                            // Retrieve the compressed 1D variable via our helper
                            int index = get_quad_index(pi_iujv_offset, i, u, j, v, n);
                            double current_pi_iujv = pi[index];

                            double AT_pi_f = pi_iuj(i, u, j) + pi_iuv(i, u, v);
                            double AT_pi_b = pi_iuj(j, v, i) + pi_iuv(j, v, u); 

                            // Symmetrize and apply the split factor for the off-diagonal multiplier
                            double val = problem.D[i][j] * problem.F[u][v] - 0.5 * (AT_pi_f + AT_pi_b) - 0.5 * current_pi_iujv;
                            
                            // Thread-safe writes: distinct (idx1, idx2) pairs belong to exactly one thread assignment
                            C_minus_AT_pi(idx1, idx2) = val;
                            C_minus_AT_pi(idx2, idx1) = val;
                        }
                    }
                }
            }
        }


        // 3. Build the DIAGONAL elements (idx1 == idx2)
        // 2. Build DIAGONAL elements in parallel
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                int idx = i * n + u;
                
                int index = get_quad_index(pi_iujv_offset, i, u, i, u, n);
                double current_pi_iujv_diag = pi[index];

                // Every thread gets its own isolated local accumulator copy
                double AT_pi = pi_i(i) + pi_u(u) + pi_iuj(i, u, i) + pi_iuv(i, u, u) + current_pi_iujv_diag;
                
                for (int j = 0; j < n; ++j) {
                    AT_pi -= pi_iuj(i, u, j) + pi_iuv(i, u, j);
                }
                
                // Thread-safe write: each diagonal position is completely unique to the (i, u) pair
                C_minus_AT_pi(idx, idx) = problem.D[i][i] * problem.F[u][u] - AT_pi;
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
                throw std::runtime_error("Eigenvalue decomposition failed.");
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                pcost += problem.D[i][i] * problem.F[u][u] * X(i*n + u, i*n + u);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += problem.D[i][j] * problem.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]) * X(i*n + u, j*n + v);
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
            int vio_iuiu_offset = 2*n + 2*n*n*n + n*n*n*n;
            // vio_u(u) = 1 - sum_i x_iu for all u
            #pragma omp parallel for schedule(static)
            for (int u = 0; u < n; ++u) {
                vio_u(u) = 1.0;
                for (int i = 0; i < n; ++i) {
                    vio_u(u) += -X(i * n + u, i * n + u);
                }
            }
            // vio_i(i) = 1 - sum_u x_iu for all i
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                vio_i(i) = 1.0;
                for (int u = 0; u < n; ++u) {
                    vio_i(i) += -X(i * n + u, i * n + u);
                }
            }
            // vio_iuj(i, u, j) = x_iu - sum_v y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int j = 0; j < n; ++j) {
                        vio_iuj(i, u, j) = X(i * n + u, i * n + u);
                        for (int v = 0; v < n; ++v) {
                            vio_iuj(i, u, j) += -X(i * n + u, j * n + v);
                        }
                    }
                }
            }
            // vio_iuv(i, u, v) = x_iu - sum_j y[i,u,j,v]
            #pragma omp parallel for collapse(3) schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    for (int v = 0; v < n; ++v) {
                        vio_iuv(i, u, v) = X(i * n + u, i * n + u);
                        for (int j = 0; j < n; ++j) {
                            vio_iuv(i, u, v) += -X(i * n + u, j * n + v);
                        }
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
void SDPOracle::SolveSubproblem3()
{

    RetrieveDensePi(); // Ensure DensePi is up-to-date with current Lambda
    auto pi = DensePi;
    lcost = 0.0;
    pcost = 0.0;
    std::fill(psol.begin(), psol.end(), 0.0); // Set all primal variables to zero
    std::fill(vio.begin(), vio.end(), 0.0); // Set all violation variables to zero
    Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n * n, n * n);

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

                            double val = problem.D[i][j] * problem.F[u][v] - 0.5 * (AT_pi_f + AT_pi_b) - 0.5 * current_pi_iujv;
                            
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
                double cond = 2.0 * problem.D[i][i] * problem.F[u][u];
                
                int index = get_quad_index(pi_iujv_offset, i, u, i, u, n);
                double current_pi_iujv_diag = (cond > 0.0) ? pi[index] : 0.0;

                double AT_pi = pi_i(i) + pi_u(u) + pi_iuj(i, u, i) + pi_iuv(i, u, u) + current_pi_iujv_diag;
                
                #pragma omp simd reduction(+:AT_pi)
                for (int j = 0; j < n; ++j) {
                    AT_pi -= pi_iuj(i, u, j) + pi_iuv(i, u, j);
                }
                
                C_minus_AT_pi(idx, idx) = problem.D[i][i] * problem.F[u][u] - AT_pi;
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
                throw std::runtime_error("Eigenvalue decomposition failed.");
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                pcost += problem.D[i][i] * problem.F[u][u] * X(i*n + u, i*n + u);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += problem.D[i][j] * problem.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]) * X(i*n + u, j*n + v);
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
void SDPOracle::SolveSubproblem4()
{

    RetrieveDensePi(); // Ensure DensePi is up-to-date with current Lambda
    auto pi = DensePi;
    lcost = 0.0;
    pcost = 0.0;
    std::fill(psol.begin(), psol.end(), 0.0); // Set all primal variables to zero
    std::fill(vio.begin(), vio.end(), 0.0); // Set all violation variables to zero

    Eigen::MatrixXd C_minus_AT_pi = Eigen::MatrixXd::Zero(n * n, n * n);

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
                            double val = problem.D[i][j] * problem.F[u][v] - 0.5 * (AT_pi_f + AT_pi_b);
                            
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
                C_minus_AT_pi(idx, idx) = problem.D[i][i] * problem.F[u][u] - AT_pi;
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
                throw std::runtime_error("Eigenvalue decomposition failed.");
            }
            Eigen::Index minIndex;
            lambda = eigensolver.eigenvalues().minCoeff(&minIndex);
            v = eigensolver.eigenvectors().col(minIndex);
        }




        auto X = (alpha*v * v.transpose()).eval();
        // compute pcost = <C, X> = trace(C^T * X)
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                pcost += problem.D[i][i] * problem.F[u][u] * X(i*n + u, i*n + u);
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i != j || u != v) {
                            pcost += problem.D[i][j] * problem.F[u][v] * X(i*n + u, j*n + v);
                            // pcost += 0.5 * (problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]) * X(i*n + u, j*n + v);
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
}


void SDPOracle::SolveSubproblemTest() 
{
    // test using a simple problem min x^2 + y^2 subject to x + y = 1
    // The optimal solution is x = 0.5, y = 0.5, with cost = 0.5
    // Lagrangian: L(x,y,lambda) = x^2 + y^2 + lambda*(1 - x - y)
    // Lagrangian dual: max_lambda min_{x,y} L(x,y,lambda)
    // min_{x,y} L(x,y,lambda) = min_{x,y} (x^2 - lambda*x) + (y^2 - lambda*y) + lambda
    // = min_x (x^2 - lambda*x) + min_y (y^2 - lambda*y) + lambda
    // = -lambda^2/4 - lambda^2/4 + lambda = -lambda^2/2
    // max_lambda -lambda^2/2 = 0 at lambda = 0
    psol.resize(2, 0.0);
    vio.resize(1, 0.0);
    NumVar = 1;
    const double* pi = Lambda;
    std::vector<double> dense_pi(NumVar,0.0);

    if(LamBase)
    {
        std::cout << "Using LamBase to construct dense_pi." << std::endl;
        for(Index k=0;k<LamBDim;k++)
            dense_pi[LamBase[k]] = Lambda[k];

        pi = dense_pi.data();
    }
    else
    {
        pi = Lambda;
    }
    std::cout
    << "NumVar = " << NumVar
    << " LamBDim = " << LamBDim
    << " sparse = " << (LamBase != nullptr)
    << std::endl;

    // Step 1: Compute primal solution
    double lambda = pi[0];
    double x = lambda / 2.0;
    double y = lambda / 2.0;
    psol[0] = x;
    psol[1] = y;

    // Step 2: Compute primal cost
    pcost = x*x + y*y;

    // Step 3: Compute constraint violation
    vio[0] = 1.0 - x - y;

    // Step 4: Compute Lagrangian cost
    lcost = pcost + lambda * vio[0];

    std::cout << "Subproblem solved. lcost: " << lcost << ", pcost: " << pcost << ", vio: " << vio[0] << std::endl;
}
