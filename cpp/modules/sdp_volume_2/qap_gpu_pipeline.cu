#include "qap_gpu_pipeline.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <cstdlib>

// --- CUDA ERROR CHECKING UTILITY ---
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "❌ CUDA Error: " << cudaGetErrorString(code) << " in " << file << ":" << line << std::endl;
      if (abort) std::exit(code);
   }
}

// --- GLOBAL GPU ACCELERATION KERNELS ---

__device__ inline int get_quad_index_gpu(int offset, int i, int u, int j, int v, int n) {
    return offset + (i * n * n * n + u * n * n + j * n + v); 
}

__global__ void build_C_minus_AT_pi_kernel(
    double* d_matrix, const double* d_pi, const double* d_D, const double* d_F, 
    int n, int pi_iujv_offset) 
{
    int idx1 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = n * n;

    if (idx1 >= N2 || idx2 >= N2) return;

    int i = idx1 / n; int u = idx1 % n;
    int j = idx2 / n; int v = idx2 % n;

    if (idx1 != idx2) {
        int g_i = i, g_u = u, g_j = j, g_v = v;
        if (idx1 > idx2) { g_i = j; g_u = v; g_j = i; g_v = u; }

        int index = get_quad_index_gpu(pi_iujv_offset, g_i, g_u, g_j, g_v, n);
        double current_pi_iujv = d_pi[index];

        double AT_pi_f = d_pi[2*n + g_i*n*n + g_u*n + g_j] + d_pi[2*n + n*n*n + g_i*n*n + g_u*n + g_v]; 
        double AT_pi_b = d_pi[2*n + g_j*n*n + g_v*n + g_i] + d_pi[2*n + n*n*n + g_j*n*n + g_v*n + g_u]; 

        double val = d_D[g_i * n + g_j] * d_F[g_u * n + g_v] - 0.5 * (AT_pi_f + AT_pi_b) - 0.5 * current_pi_iujv;
        d_matrix[idx1 + idx2 * N2] = val;
    } 
    else {
        int index = get_quad_index_gpu(pi_iujv_offset, i, u, i, u, n);
        double current_pi_iujv_diag = d_pi[index];

        double AT_pi = d_pi[i] + d_pi[n + u] + d_pi[2*n + i*n*n + u*n + i] + d_pi[2*n + n*n*n + i*n + u*n + u] + current_pi_iujv_diag;
        for (int k = 0; k < n; ++k) {
            AT_pi -= (d_pi[2*n + i*n*n + u*n + k] + d_pi[2*n + n*n*n + i*n + u*n + k]);
        }
        d_matrix[idx1 + idx2 * N2] = d_D[i * n + i] * d_F[u * n + u] - AT_pi;
    }
}

__global__ void compute_X_and_pcost_kernel(
    double* d_X, const double* d_v, const double* d_D, const double* d_F, 
    double alpha, int n, double* d_pcost_blocks) 
{
    extern __shared__ double sdata[];
    int idx1 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = n * n;

    double thread_pcost = 0.0;
    if (idx1 < N2 && idx2 < N2) {
        double x_val = alpha * d_v[idx1] * d_v[idx2];
        d_X[idx1 + idx2 * N2] = x_val;

        int i = idx1 / n; int u = idx1 % n;
        int j = idx2 / n; int v = idx2 % n;

        if (i == j && u == v) {
            thread_pcost = d_D[i * n + i] * d_F[u * n + u] * x_val;
        } else {
            thread_pcost = d_D[i * n + j] * d_F[u * n + v] * x_val;
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = thread_pcost;
    __syncthreads();

    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) {
        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        d_pcost_blocks[blockId] = sdata[0]; // Fix shared memory root indexing syntax
    }
}

__global__ void compute_violations_kernel(const double* d_X, double* d_vio_u, double* d_vio_i, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double sum_u = 0.0; double sum_i = 0.0;
    for (int k = 0; k < n; ++k) {
        int diag_u_idx = (k * n + idx); sum_u += d_X[diag_u_idx + diag_u_idx * (n * n)];
        int diag_i_idx = (idx * n + k); sum_i += d_X[diag_i_idx + diag_i_idx * (n * n)];
    }
    d_vio_u[idx] = 1.0 - sum_u; d_vio_i[idx] = 1.0 - sum_i;
}

__global__ void compute_vio_iuj_kernel(const double* d_X, double* d_vio_iuj, int n) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || u >= n || j >= n) return;
    int N2 = n * n; int idx1 = i * n + u;
    double x_iu = d_X[idx1 + idx1 * N2];
    double sum_y = 0.0;
    for (int v = 0; v < n; ++v) { sum_y += d_X[idx1 + (j * n + v) * N2]; }
    d_vio_iuj[i * n * n + u * n + j] = x_iu - sum_y;
}

__global__ void compute_vio_iuv_kernel(const double* d_X, double* d_vio_iuv, int n) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || u >= n || v >= n) return;
    int N2 = n * n; int idx1 = i * n + u;
    double x_iu = d_X[idx1 + idx1 * N2];
    double sum_y = 0.0;
    for (int j = 0; j < n; ++j) { sum_y += d_X[idx1 + (j * n + v) * N2]; }
    d_vio_iuv[i * n * n + u * n + v] = x_iu - sum_y;
}

__global__ void compute_vio_iujv_kernel(const double* d_X, double* d_vio_iujv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_vio_iujv[idx] = -d_X[idx];
}

// --- FULL DEFINITION OF THE HIDDEN WORKER STRUCT (Solves your incomplete type error) ---
struct QapGpuSolverPipelineImpl {
    bool is_initialized = false;
    int n, N, pi_size, num_blocks;
    cusolverDnHandle_t cusolverH = nullptr;
    
    double *d_D = nullptr, *d_F = nullptr, *d_pi = nullptr;
    double *d_A = nullptr, *d_W = nullptr, *d_X = nullptr;
    double *d_vio_u = nullptr, *d_vio_i = nullptr, *d_vio_iuj = nullptr, *d_vio_iuv = nullptr, *d_vio_iujv = nullptr;
    double *d_work = nullptr, *d_pcost_blocks = nullptr;
    int *d_info = nullptr;
    int lwork = 0;

    double* h_pcost_blocks = nullptr;
};

// --- OUTER WRAPPER CLASS IMPLEMENTATION METHODS ---

QapGpuSolverPipeline::QapGpuSolverPipeline() {
    impl = new QapGpuSolverPipelineImpl();
}

QapGpuSolverPipeline::~QapGpuSolverPipeline() {
    if (impl->is_initialized) {
        cudaFree(impl->d_D); cudaFree(impl->d_F); cudaFree(impl->d_pi); cudaFree(impl->d_A); cudaFree(impl->d_W); cudaFree(impl->d_X);
        cudaFree(impl->d_vio_u); cudaFree(impl->d_vio_i); cudaFree(impl->d_vio_iuj); cudaFree(impl->d_vio_iuv); cudaFree(impl->d_vio_iujv);
        cudaFree(impl->d_work); cudaFree(impl->d_info); cudaFree(impl->d_pcost_blocks);
        std::free(impl->h_pcost_blocks);
        cusolverDnDestroy(impl->cusolverH);
    }
    delete impl;
}

void QapGpuSolverPipeline::initialize(int base_n, size_t total_pi_size, const std::vector<std::vector<double>>& raw_D, const std::vector<std::vector<double>>& raw_F) {
    if (impl->is_initialized) return;
    impl->n = base_n; impl->N = impl->n * impl->n; impl->pi_size = total_pi_size;

    double* h_D_flat = (double*)std::malloc(sizeof(double) * impl->N);
    double* h_F_flat = (double*)std::malloc(sizeof(double) * impl->N);
    for (int i = 0; i < impl->n; ++i) {
        for (int j = 0; j < impl->n; ++j) {
            h_D_flat[i * impl->n + j] = raw_D[i][j];
            h_F_flat[i * impl->n + j] = raw_F[i][j];
        }
    }

    CUDA_CHECK(cudaMalloc((void**)&impl->d_D, sizeof(double) * impl->N));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_F, sizeof(double) * impl->N));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_pi, sizeof(double) * impl->pi_size));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_A, sizeof(double) * impl->N * impl->N));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_W, sizeof(double) * impl->N));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_X, sizeof(double) * impl->N * impl->N));

    CUDA_CHECK(cudaMalloc((void**)&impl->d_vio_u, sizeof(double) * impl->n));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_vio_i, sizeof(double) * impl->n));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_vio_iuj, sizeof(double) * impl->n * impl->n * impl->n));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_vio_iuv, sizeof(double) * impl->n * impl->n * impl->n));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_vio_iujv, sizeof(double) * impl->N * impl->N));
    CUDA_CHECK(cudaMalloc((void**)&impl->d_info, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(impl->d_D, h_D_flat, sizeof(double) * impl->N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->d_F, h_F_flat, sizeof(double) * impl->N, cudaMemcpyHostToDevice));
    std::free(h_D_flat); std::free(h_F_flat);

    cusolverDnCreate(&impl->cusolverH);
    cusolverDnDsyevd_bufferSize(impl->cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, impl->N, impl->d_A, impl->N, impl->d_W, &impl->lwork);
    CUDA_CHECK(cudaMalloc((void**)&impl->d_work, sizeof(double) * impl->lwork));

    dim3 block(16, 16);
    dim3 grid((impl->N + block.x - 1) / block.x, (impl->N + block.y - 1) / block.y);
    impl->num_blocks = grid.x * grid.y;
    CUDA_CHECK(cudaMalloc((void**)&impl->d_pcost_blocks, sizeof(double) * impl->num_blocks));
    impl->h_pcost_blocks = (double*)std::malloc(sizeof(double) * impl->num_blocks);

    impl->is_initialized = true;
}

void QapGpuSolverPipeline::execute_iteration_step(
    const double* h_pi, double alpha, 
    double& lcost, double& pcost, 
    double* h_vio_u, double* h_vio_i, double* h_vio_iuj, double* h_vio_iuv, double* h_vio_iujv) 
{
    if (!impl->is_initialized) return;

    int pi_iujv_offset = 2 * impl->n + 2 * impl->n * impl->n * impl->n;

    CUDA_CHECK(cudaMemcpy(impl->d_pi, h_pi, sizeof(double) * impl->pi_size, cudaMemcpyHostToDevice));

    dim3 block2D(16, 16);
    dim3 grid2D((impl->N + block2D.x - 1) / block2D.x, (impl->N + block2D.y - 1) / block2D.y);
    
    build_C_minus_AT_pi_kernel<<<grid2D, block2D>>>(impl->d_A, impl->d_pi, impl->d_D, impl->d_F, impl->n, pi_iujv_offset);

    cusolverDnDsyevd(impl->cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, impl->N, impl->d_A, impl->N, impl->d_W, impl->d_work, impl->lwork, impl->d_info);
    CUDA_CHECK(cudaMemcpy(&lcost, impl->d_W, sizeof(double), cudaMemcpyDeviceToHost));
    size_t shared_mem_size = block2D.x * block2D.y * sizeof(double);
    compute_X_and_pcost_kernel<<<grid2D, block2D, shared_mem_size>>>(impl->d_X, impl->d_A, impl->d_D, impl->d_F, alpha, impl->n, impl->d_pcost_blocks);
    int block1D = 256;
    int grid1D = (impl->n + block1D - 1) / block1D;
    compute_violations_kernel<<<grid1D, block1D>>>(impl->d_X, impl->d_vio_u, impl->d_vio_i, impl->n);
    dim3 block3D(8, 8, 4);
    dim3 grid3D((impl->n + block3D.x - 1) / block3D.x, (impl->n + block3D.y - 1) / block3D.y, (impl->n + block3D.z - 1) / block3D.z);
    compute_vio_iuj_kernel<<<grid3D, block3D>>>(impl->d_X, impl->d_vio_iuj, impl->n);
    compute_vio_iuv_kernel<<<grid3D, block3D>>>(impl->d_X, impl->d_vio_iuv, impl->n);
    compute_vio_iujv_kernel<<<grid2D, block2D>>>(impl->d_X, impl->d_vio_iujv, impl->n);
    pcost = 0.0;
    CUDA_CHECK(cudaMemcpy(impl->h_pcost_blocks, impl->d_pcost_blocks, sizeof(double) * impl->num_blocks, cudaMemcpyDeviceToHost));
    for (int b = 0; b < impl->num_blocks; ++b) {
        pcost += impl->h_pcost_blocks[b];
    }
    CUDA_CHECK(cudaMemcpy(h_vio_u, impl->d_vio_u, sizeof(double) * impl->n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vio_i, impl->d_vio_i, sizeof(double) * impl->n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vio_iuj, impl->d_vio_iuj, sizeof(double) * impl->n * impl->n * impl->n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vio_iuv, impl->d_vio_iuv, sizeof(double) * impl->n * impl->n * impl->n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vio_iujv, impl->d_vio_iujv, sizeof(double) * impl->N * impl->N, cudaMemcpyDeviceToHost));
}