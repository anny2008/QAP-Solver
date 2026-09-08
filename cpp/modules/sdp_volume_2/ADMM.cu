#include "ADMM.h"
#include <chrono>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/KroneckerProduct>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>
Eigen::MatrixXd convert_nested_vector_to_eigen(const std::vector<std::vector<double>>& vec_2d);

void ADMMSolver::initialize() 
{
    auto n = problem->n;
    Vhat = build_Vhat(problem->n);
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
    }
    beta = n/3.0;
    L = Eigen::MatrixXd::Zero(N_total, N_total);
    Eigen::MatrixXd A = convert_nested_vector_to_eigen(problem->D);
    Eigen::MatrixXd B = convert_nested_vector_to_eigen(problem->F);
    L.block(1, 1, n2, n2) = Eigen::kroneckerProduct(B, A);
    normL = L.norm();
    L = L / normL*n2;
}

void ADMMSolver::runADMM(bool is_low_rank)
{
    auto n = problem->n;
    auto N_total = 1 + n * n;
    auto n2 = n * n;
    double best_lb = 0.0;
    std::cout << "Max iterations: " << max_iter << ", gamma: " << gamma << ", beta: " << beta << std::endl;
    std::cout << "Starting ADMM iterations..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto iter = 0;
    int n_stall = 0;
    for (iter = 0; iter < max_iter; ++iter) {
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
        Z += (gamma * beta) * pR;
        auto nrm_pR = pR.norm();
        auto nrm_dR = beta*dR.norm();
        auto feas = nrm_pR / Y.norm();
        auto obj = L.cwiseProduct(Y).sum() * normL/n2;
        if (iter % 100 == 0) {
            std::cout << "Iteration: " << iter << ", Objective: " << obj << ", Feasibility: " << feas << ", Norm of pR: " << nrm_pR << ", Norm of dR: " << nrm_dR << std::endl;
        }
        if (nrm_pR < tol && nrm_dR < tol) {
            n_stall++;
        } else {
            n_stall = 0;
        }
        if (n_stall >= 5) {
            std::cout << "Converged at iteration: " << iter << ", Objective: " << obj << ", Feasibility: " << feas << std::endl;
            break;
        }
    }
    std::cout << "===================================================" << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = duration.count() / 1000.0;
    std::cout << "ADMM completed in " << elapsed_seconds << " seconds." << std::endl;
    auto average_iteration_time = elapsed_seconds / iter;
    std::cout << "Average time per iteration: " << average_iteration_time << " seconds" << std::endl;
    
}

double ADMMSolver::obtainLowerBound() {
    auto n = problem->n;
    auto N_total = 1 + n * n;
    auto n2 = n * n;
    
    Eigen::MatrixXd ones_col = -1.0 * Eigen::MatrixXd::Ones(2 * n, 1);
    Eigen::MatrixXd kron1 = Eigen::kroneckerProduct(In, en.transpose()).eval();
    Eigen::MatrixXd kron2 = Eigen::kroneckerProduct(en.transpose(), In).eval();
    
    Eigen::MatrixXd block_right(kron1.rows() + kron2.rows(), kron1.cols());
    block_right << kron1, 
                   kron2;
   

    Eigen::MatrixXd That(2 * n, 1 + block_right.cols());
    That << Eigen::MatrixXd::Ones(2 * n, 1) * -1.0, block_right;
    

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(That.transpose());
    Eigen::MatrixXd Q_full = qr.householderQ() * Eigen::MatrixXd::Identity(That.cols(), std::min(That.cols(), That.rows()));
    Eigen::MatrixXd Q = Q_full.leftCols(Q_full.cols() - 1);
    
    Eigen::MatrixXd Uloc(Vhat.rows(), Vhat.cols() + Q.cols());
    Uloc << Vhat, Q;
    Eigen::MatrixXd Zloc = Uloc.transpose() * Z * Uloc;
    
    
    int idx1 = (n - 1) * (n - 1) + 1; // Size of block 1
    
    Eigen::MatrixXd W11 = Zloc.block(0, 0, idx1, idx1);
    Eigen::MatrixXd W12 = Zloc.block(0, idx1, idx1, Zloc.cols() - idx1);
    Eigen::MatrixXd W22 = Zloc.block(idx1, idx1, Zloc.rows() - idx1, Zloc.cols() - idx1);
    W11 = 0.5 * (W11 + W11.transpose());
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(W11);
    Eigen::VectorXd dw = es.eigenvalues();
    Eigen::MatrixXd Uw = es.eigenvectors();
    
    Eigen::VectorXd dw_neg = Eigen::VectorXd::Zero(dw.size());
    int neg_count = 0;
    for (int i = 0; i < dw.size(); ++i) {
        if (dw(i) < 0) {
            dw_neg(i) = dw(i);
            neg_count++;
        }
    }
    
    // MATLAB: W11 = Uw(:,id)*diag(dw(id))*Uw(:,id)';
    // Equivalent to reconstructing with only negative eigenvalues zeroed out
    W11 = Uw * dw_neg.asDiagonal() * Uw.transpose();
    
    Eigen::MatrixXd W_block(Zloc.rows(), Zloc.cols());
    W_block << W11, W12,
               W12.transpose(), W22;
               
    Eigen::MatrixXd Zp = Uloc * W_block * Uloc.transpose();
    Zp = 0.5 * (Zp + Zp.transpose());
    
    // 8. Masking operations
    Eigen::MatrixXd L_plus_Zp = L + Zp;
    // Yp = 1 if L + Zp <= 0, else 0, but also gangster
    Eigen::MatrixXd Y_p = Eigen::MatrixXd::Zero(L_plus_Zp.rows(), L_plus_Zp.cols());
    for (int i = 0; i < L_plus_Zp.rows(); ++i) {
        for (int j = 0; j < L_plus_Zp.cols(); ++j) {
            if (L_plus_Zp(i, j) <= 0) {
                Y_p(i, j) = 1.0;
            }
        }
    }
    Y_p = Y_p.cwiseProduct(J);
    Y_p(0,0) = 1.0;
    Eigen::MatrixXd final_m = L_plus_Zp.cwiseProduct(Y_p);

    double sum_matrix = final_m.sum();
    double lbd_val = sum_matrix * normL / n2;
    std::cout << "Current lower bound: " << lbd_val << std::endl;

    return 0.0; // Placeholder
}

Eigen::MatrixXd build_Vchoice3(int n) {
    Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(n, n - 1);
    Eigen::VectorXd dscale = Eigen::VectorXd::Ones(n - 1);

    int iblk = 1;
    int ii = 0;

    std::vector<int> sizeblocks;
    std::vector<double> normalize;

    sizeblocks.push_back(n / 2); 
    normalize.push_back(std::sqrt(2.0)); 

    while (sizeblocks[iblk - 1] >= 1) {
        int s = sizeblocks[iblk - 1];
        int ncols = s;

        // Base vector for current block: [ones; -ones]
        int p1 = std::pow(2, iblk - 1);
        Eigen::VectorXd basevec(2 * p1);
        basevec.head(p1).setConstant(1.0);
        basevec.tail(p1).setConstant(-1.0);

        Eigen::MatrixXd block = Eigen::MatrixXd::Zero(n, ncols);

        for (int j = 0; j < ncols; ++j) {
            // Equivalent to np.kron(np.eye(s)[j], basevec)
            int row_offset = j * basevec.size();
            if (row_offset < n) {
                int length = std::min(static_cast<int>(basevec.size()), n - row_offset);
                block.block(row_offset, j, length, 1) = basevec.head(length);
            }
        }

        // Place into temp matrix
        temp.block(0, ii, n, ncols) = block;
        dscale.segment(ii, ncols).setConstant(normalize[iblk - 1]);

        ii += ncols;
        iblk += 1;

        int p_next = std::pow(2, iblk);
        sizeblocks.push_back(n / p_next);
        normalize.push_back(std::sqrt(static_cast<double>(p_next)));
    }

    // Fix zero columns by nullspace completion via SVD
    Eigen::VectorXd col_norms = temp.colwise().squaredNorm();
    std::vector<int> zero_ids;
    for (int i = 0; i < col_norms.size(); ++i) {
        if (col_norms(i) == 0.0) {
            zero_ids.push_back(i);
        }
    }

    if (!zero_ids.empty()) {
        Eigen::MatrixXd M(n, (n - 1) + 1);
        M.leftCols(n - 1) = temp;
        M.rightCols(1).setConstant(1.0);

        // Compute SVD of M.T
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M.transpose(), Eigen::ComputeFullV);
        Eigen::MatrixXd V_mat = svd.matrixV();

        // Extract trailing nullspace vectors
        for (size_t k = 0; k < zero_ids.size(); ++k) {
            int zero_col_idx = zero_ids[k];
            int v_col_idx = V_mat.cols() - zero_ids.size() + k;
            temp.col(zero_col_idx) = V_mat.col(v_col_idx);
        }
    }

    // Normalize columns
    for (int j = 0; j < temp.cols(); ++j) {
        double norm = temp.col(j).norm();
        if (norm > 0.0) {
            temp.col(j) /= norm;
        }
    }

    // Force values below 1e-12 absolute value to 0
    for (int r = 0; r < temp.rows(); ++r) {
        for (int c = 0; c < temp.cols(); ++c) {
            if (std::abs(temp(r, c)) < 1e-12) {
                temp(r, c) = 0.0;
            }
        }
    }

    return temp;
}

Eigen::MatrixXd build_Vhat(int n) {
    Eigen::MatrixXd V = build_Vchoice3(n);
    Eigen::MatrixXd KVV = kroneckerProduct(V, V);

    int n_sq = n * n;
    int w_rows = n_sq + 1;
    int w_cols = KVV.cols() + 1;

    Eigen::MatrixXd Vhat = Eigen::MatrixXd::Zero(w_rows, w_cols);

    // Configure the first column
    Vhat(0, 0) = std::sqrt(0.5);
    Vhat.block(1, 0, n_sq, 1).setConstant(std::sqrt(0.5) / n);

    // Place the Kronecker matrix block inside (leaving top row elements at index >=1 as 0)
    Vhat.block(1, 1, n_sq, KVV.cols()) = KVV;

    // Clean up numerical precision anomalies
    for (int r = 0; r < Vhat.rows(); ++r) {
        for (int c = 0; c < Vhat.cols(); ++c) {
            if (std::abs(Vhat(r, c)) < 1e-12) {
                Vhat(r, c) = 0.0;
            }
        }
    }

    return Vhat;
}
