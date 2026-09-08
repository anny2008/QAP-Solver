#pragma once

#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <fusion.h>
#include <monty.h>
#include <memory>

using namespace mosek::fusion;
using namespace monty;
// using namespace Eigen;

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

auto create_V_array(int n) 
{
    int k = n - 1; // Number of columns (and identity matrix dimension)
    int rows = n;
    int cols = k;

    // 1. Allocate a flat, uninitialized 2D ndarray of size n x (n-1)
    auto V = new_array_ptr<double, 2>(shape(rows, cols));
    
    // 2. Access the underlying raw pointer data buffer safely
    auto data = V->raw();

    // 3. Populate the upper block: I_{n-1}
    for (int r = 0; r < k; ++r) {
        for (int c = 0; c < k; ++c) {
            (*V)(r, c) = (r == c) ? 1.0 : 0.0;
        }
    }

    // 4. Populate the bottom block row: -e^T_{n-1} (index position row = n-1)
    int last_row = n - 1;
    for (int c = 0; c < k; ++c) {
        (*V)(last_row, c) = -1.0;
    }

    return V;
}

// Computes L = (A kron B) where A and B are std::vector<std::vector<double>> of size n x n
std::shared_ptr<ndarray<double, 2>> create_L_matrix(
    const std::vector<std::vector<double>>& A, 
    const std::vector<std::vector<double>>& B, 
    int n) 
{
    // Total dimension for the resulting matrix
    int L_dim = n * n;

    // 1. Allocate the flat 2D ndarray for L
    auto L = std::make_shared<ndarray<double, 2>>(shape(L_dim, L_dim));
    double* l_data = L->raw();

    // 2. Compute the Kronecker Product: A kron B
    for (int rA = 0; rA < n; ++rA) {
        for (int cA = 0; cA < n; ++cA) {
            double valA = A[rA][cA]; // Direct 2D lookup

            for (int rB = 0; rB < n; ++rB) {
                for (int cB = 0; cB < n; ++cB) {
                    double valB = B[rB][cB]; // Direct 2D lookup

                    // Calculate global matrix row and column indices
                    int rowL = rA * n + rB;
                    int colL = cA * n + cB;

                    // Convert to 1D flat row-major index for the raw data pointer
                    int flat_idx = rowL * L_dim + colL;

                    l_data[flat_idx] = valA * valB;
                }
            }
        }
    }

    return L;
}

std::shared_ptr<ndarray<double, 2>> compute_sandwich_L(
    std::shared_ptr<ndarray<double, 2>> W_raw,
    std::shared_ptr<ndarray<double, 2>> L_orig_raw) 
{
    int w_rows = W_raw->size(0); // 144
    int w_cols = W_raw->size(1); // 122
    
    // Allocate final 122 x 122 matrix
    auto L_final = std::make_shared<ndarray<double, 2>>(shape(w_cols, w_cols));
    double* l_out = L_final->raw();
    double* w = W_raw->raw();
    double* l_orig = L_orig_raw->raw();

    // Intermediate temporary matrix for (L_orig * W) -> Shape: 144 x 122
    std::vector<double> temp(w_rows * w_cols, 0.0);

    // Step 1: temp = L_orig * W
    for (int i = 0; i < w_rows; ++i) {
        for (int j = 0; j < w_cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < w_rows; ++k) {
                sum += l_orig[i * w_rows + k] * w[k * w_cols + j];
            }
            temp[i * w_cols + j] = sum;
        }
    }

    // Step 2: L_final = W^T * temp -> Shape: 122 x 122
    for (int i = 0; i < w_cols; ++i) {
        for (int j = 0; j < w_cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < w_rows; ++k) {
                // W^T[i, k] is W[k, i]
                sum += w[k * w_cols + i] * temp[k * w_cols + j];
            }
            l_out[i * w_cols + j] = sum;
        }
    }

    return L_final;
}

#include <memory>
#include <vector>
#include <stdexcept>

std::shared_ptr<ndarray<double, 2>> compute_sandwich_L(
    std::shared_ptr<ndarray<double, 2>> W_raw,
    const std::vector<double>& L_orig_raw) 
{
    int w_rows = W_raw->size(0); // 144
    int w_cols = W_raw->size(1); // 122
    
    // Safety check: L_orig must be a square matrix of size w_rows x w_rows
    if (L_orig_raw.size() != static_cast<size_t>(w_rows * w_rows)) {
        throw std::invalid_argument("L_orig_raw size does not match w_rows * w_rows.");
    }

    // Allocate final 122 x 122 matrix
    auto L_final = std::make_shared<ndarray<double, 2>>(shape(w_cols, w_cols));
    double* l_out = L_final->raw();
    double* w = W_raw->raw();
    
    // Direct pointer to the flat vector data
    const double* l_orig = L_orig_raw.data();

    // Intermediate temporary matrix for (L_orig * W) -> Shape: 144 x 122
    std::vector<double> temp(w_rows * w_cols, 0.0);

    // Step 1: temp = L_orig * W
    for (int i = 0; i < w_rows; ++i) {
        for (int j = 0; j < w_cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < w_rows; ++k) {
                sum += l_orig[i * w_rows + k] * w[k * w_cols + j];
            }
            temp[i * w_cols + j] = sum;
        }
    }

    // Step 2: L_final = W^T * temp -> Shape: 122 x 122
    for (int i = 0; i < w_cols; ++i) {
        for (int j = 0; j < w_cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < w_rows; ++k) {
                // W^T[i, k] is W[k, i]
                sum += w[k * w_cols + i] * temp[k * w_cols + j];
            }
            l_out[i * w_cols + j] = sum;
        }
    }

    return L_final;
}



auto create_W_matrix(int n) 
{
    int k = n - 1;
    
    // Calculate final structural dimensions of W
    int w_rows = n * n;
    int w_cols = 1 + (k * k);

    // 1. Allocate the flat 2D ndarray for W
    auto W = new_array_ptr<double, 2>(shape(w_rows, w_cols));
    double* w_data = W->raw();

    // 2. Pre-generate V in a flat buffer for easy indexing (Size: n rows * k columns)
    std::vector<double> V(n * k, 0.0);
    
    // Populate upper identity block: I_{n-1}
    for (int r = 0; r < k; ++r) {
        V[r * k + r] = 1.0; 
    }
    
    // Populate bottom row block: -e^T_{n-1} at row index (n - 1)
    int last_row = n - 1;
    for (int c = 0; c < k; ++c) {
        V[last_row * k + c] = -1.0; 
    }

    // 3. Populate W using multi-dimensional Kronecker indexing logic
    for (int r1 = 0; r1 < n; ++r1) {
        for (int r2 = 0; r2 < n; ++r2) {
            
            // Compute the unique target 1D row index for W
            int w_row_idx = r1 * n + r2;

            // --- Fill Left Block: 1/n * (e kron e) ---
            // Column index for this component is always 0
            int flat_idx_left = w_row_idx * w_cols + 0;
            w_data[flat_idx_left] = 1.0 / static_cast<double>(n);

            // --- Fill Right Block: V kron V ---
            for (int c1 = 0; c1 < k; ++c1) {
                for (int c2 = 0; c2 < k; ++c2) {
                    
                    // Column index in W shifts right by +1 to clear the left block
                    int w_col_idx = 1 + (c1 * k + c2);
                    int flat_idx_right = w_row_idx * w_cols + w_col_idx;

                    // Kronecker definition mapping: V[r1, c1] * V[r2, c2]
                    double v_val1 = V[r1 * k + c1];
                    double v_val2 = V[r2 * k + c2];

                    w_data[flat_idx_right] = v_val1 * v_val2;
                }
            }
        }
    }

    return W;
}


// Helper: Kronecker product of two matrices
Eigen::MatrixXd kroneckerProduct(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    int rowsA = A.rows(), colsA = A.cols();
    int rowsB = B.rows(), colsB = B.cols();
    Eigen::MatrixXd C(rowsA * rowsB, colsA * colsB);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            C.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
        }
    }
    return C;
}

// Construct matrix W (n² × m, where m = (n-1)² + 1)
Eigen::MatrixXd construct_W(int n) {
    int m = (n - 1) * (n - 1) + 1;
    int n2 = n * n;

    // Construct V: n × (n-1)
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, n - 1);
    V.topRows(n - 1) = Eigen::MatrixXd::Identity(n - 1, n - 1);
    V.bottomRows(1) = -Eigen::MatrixXd::Ones(1, n - 1);

    // Construct e: n × 1
    Eigen::VectorXd e = Eigen::VectorXd::Ones(n);

    // Construct (1/n) * e ⊗ e: n² × 1
    Eigen::MatrixXd e_outer = (1.0 / n) * e * e.transpose();
    Eigen::MatrixXd e_tensor = Eigen::Map<Eigen::MatrixXd>(e_outer.data(), n2, 1);

    // Construct V ⊗ V: n² × (n-1)²
    Eigen::MatrixXd V_tensor = kroneckerProduct(V, V);

    // W = [e_tensor, V_tensor] (horizontal concatenation)
    Eigen::MatrixXd W(n2, m);
    W << e_tensor, V_tensor;
    return W;
}

// Construct matrix L (m × m)
Eigen::MatrixXd construct_L(int n, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::VectorXd& c) {
    int n2 = n * n;
    int m = (n - 1) * (n - 1) + 1;

    // Compute B ⊗ A: n² × n²
    Eigen::MatrixXd B_kron_A = kroneckerProduct(B, A);

    // Compute Diag(c): n² × n²
    Eigen::MatrixXd Diag_c = Eigen::MatrixXd::Zero(n2, n2);
    for (int i = 0; i < n2; ++i) {
        Diag_c(i, i) = c(i);
    }

    // Compute B ⊗ A + Diag(c)
    Eigen::MatrixXd temp = B_kron_A + Diag_c;

    // Construct W
    Eigen::MatrixXd W = construct_W(n);

    // L = W^T * temp * W
    return W.transpose() * temp * W;
}

// Compute G_indices: zero pattern constraints
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>
compute_G_indices(int n) {
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> indices;

    // Row constraints: y_{(i,j),(i,k)} = 0 for j ≠ k
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (j != k) {
                    indices.push_back({{i, j}, {i, k}});
                }
            }
        }
    }

    // Column constraints: y_{(j,i),(k,i)} = 0 for j ≠ k
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (j != k) {
                    indices.push_back({{j, i}, {k, i}});
                }
            }
        }
    }
    return indices;
}

// Compute N_indices: nonnegativity constraints (excluding G_indices)
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>
compute_N_indices(int n, const std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>& G_indices) {
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> all_indices;
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> N_indices;

    // Generate all possible indices ((i1,j1), (i2,j2))
    for (int i1 = 0; i1 < n; ++i1) {
        for (int j1 = 0; j1 < n; ++j1) {
            for (int i2 = 0; i2 < n; ++i2) {
                for (int j2 = 0; j2 < n; ++j2) {
                    all_indices.push_back({{i1, j1}, {i2, j2}});
                }
            }
        }
    }

    // N_indices = all_indices - G_indices
    for (const auto& idx : all_indices) {
        if (std::find(G_indices.begin(), G_indices.end(), idx) == G_indices.end()) {
            N_indices.push_back(idx);
        }
    }
    return N_indices;
}

