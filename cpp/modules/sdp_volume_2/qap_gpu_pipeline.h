#pragma once
#include <vector>

// Forward declaration of the hidden GPU worker struct
struct QapGpuSolverPipelineImpl;

class QapGpuSolverPipeline {
private:
    QapGpuSolverPipelineImpl* impl; // Pointer to implementation (PIMPL)

public:
    // Default constructor (fixes the previous "no default constructor exists" error)
    QapGpuSolverPipeline();
    
    // Destructor to clean up hidden GPU allocations safely
    ~QapGpuSolverPipeline();

    // Initialization routine accepting your nested CPU std::vectors natively
    void initialize(int base_n, size_t total_pi_size, 
                    const std::vector<std::vector<double>>& raw_D, 
                    const std::vector<std::vector<double>>& raw_F);

    // Core iteration step utilizing fast raw double pointers for math outputs
    void execute_iteration_step(
        const double* h_pi, double alpha, 
        double& lcost, double& pcost, 
        double* h_vio_u, double* h_vio_i, double* h_vio_iuj, double* h_vio_iuv, double* h_vio_iujv);
};
