#pragma once
// Forward declarations and includes
#include <vector>
#include <string>
#include <memory>
#include <set>
// ... add other necessary includes


#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <omp.h>
#include "VolVolumeCUDA.h"
#include "../../core/problem.h"
#include "qap_gpu_pipeline.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <Eigen/Dense>
#include <fusion.h>
#include <monty.h>

using Clique = std::set<int>;
using namespace mosek::fusion;
using namespace monty;
using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::MatrixXd convert_nested_vector_to_eigen(const std::vector<std::vector<double>>& vec_2d);


struct FixedVariables {
	std::unordered_map<int, int> x_fixed_1;
	std::unordered_map<int, std::unordered_set<int>> x_fixed_0;
	std::unordered_map<int, int> y_fixed_1;
	std::unordered_map<int, std::unordered_set<int>> y_fixed_0;
	std::unordered_map<int, double> map_fixed;
};

struct PrimalViolationSummary {
	double max_abs = 0.0;
	double avg_abs = 0.0;
	double assignment_max = 0.0;
	double assignment_avg = 0.0;
	double link_max = 0.0;
	double link_avg = 0.0;
	int n = 0;
};

struct FixedViolationReport {
	int x1_violations = 0;
	int x0_violations = 0;
	int y1_violations = 0;
	int y0_violations = 0;
	double x1_max_dev = 0.0;
	double x0_max_dev = 0.0;
	double y1_max_dev = 0.0;
	double y0_max_dev = 0.0;
};

class SDPVolumeHooks : public VOL_user_hooks {
public:
    const Problem& qap_data;
    int n;
    // Fixed variables
    FixedVariables fixed;
	VOL_problem& vol_problem;
	std::chrono::duration<double> subproblem_solving_time;
	int iteration;
public:
	SDPVolumeHooks(const Problem& data, VOL_problem& vol_prob): qap_data(data), n(data.n), vol_problem(vol_prob), subproblem_solving_time(0.0) {};
	virtual ~SDPVolumeHooks() {};
    virtual bool set_fixed_variables(const FixedVariables &fv) { fixed = fv; return true; };
	virtual int compute_rc(const VOL_dvector&, VOL_dvector& rc) override { rc=0;return 0; }
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) { return 0; };
	virtual int heuristics(const VOL_problem&, const VOL_dvector&, double&) override { return 0; }
};


class SDPVolumeHooks1 : public SDPVolumeHooks {
private:

    Eigen::MatrixXd L; // L matrix for the SDP relaxation
    Eigen::MatrixXd compressed_L; // L matrix for computing the primal cost
    Eigen::MatrixXd W; // W matrix for the SDP relaxation
    Model::t M; // MOSEK model for the SDP subproblem
    Parameter::t pi_param;
    Variable::t R; // PSD matrix variable for the SDP subproblem
    // Expression::t WRWT; // Expression for W*R*W^T
    
public:
	SDPVolumeHooks1(const Problem& data, VOL_problem& vol_prob);
	int solve_subproblem_1(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	};


class SDPVolumeHooks2 : public SDPVolumeHooks {
private:
	std::string eigen_solver; // "lanczos" or "full"
	double alpha; // trace constraint for the SDP relaxation
	int n; // problem size

    double* d_D;
    double* d_F;
    double* d_pi;


    double* d_A;
    double* d_W;
    double* d_v_gpu;
    double* d_X;
    int* d_info;
    cusolverDnHandle_t cusolverH;
	QapGpuSolverPipeline gpu_pipeline;

public:
	SDPVolumeHooks2(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver);
	~SDPVolumeHooks2();
    virtual bool set_fixed_variables(const FixedVariables &fv) override;
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	int solve_subproblem_2(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);

};

class SDPVolumeHooks3 : public SDPVolumeHooks {
private:
	std::string eigen_solver; // "lanczos" or "full"
	double alpha; // trace constraint for the SDP relaxation
	int n; // problem size
	std::vector<int> quad_to_compact;

public:
	SDPVolumeHooks3(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver);
	virtual bool set_fixed_variables(const FixedVariables &fv) override;
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	int solve_subproblem_3(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
};

class SDPVolumeHooks4 : public SDPVolumeHooks {
private:
	std::string eigen_solver; // "lanczos" or "full"
	double alpha; // trace constraint for the SDP relaxation
	int n; // problem size
public:
	SDPVolumeHooks4(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver);
	virtual bool set_fixed_variables(const FixedVariables &fv) override;
	int solve_subproblem_4(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
};

class SDPVolumeHooks5 : public SDPVolumeHooks {
private:
	std::string eigen_solver; // "lanczos" or "full"
	double alpha; // trace constraint for the SDP relaxation
	int n; // problem size

    int iter;
    double beta;
    double gamma;
    double tol;

	Eigen::MatrixXd Vhat;
	Eigen::MatrixXd En;
    Eigen::MatrixXd In;
	Eigen::MatrixXd en;
	Eigen::MatrixXd L;
	Eigen::MatrixXd Y;
	Eigen::MatrixXd Z;
	Eigen::MatrixXd R;
	Eigen::MatrixXd J;

    double normL;

public:
	SDPVolumeHooks5(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver);
	virtual bool set_fixed_variables(const FixedVariables &fv) override;
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	int solve_subproblem_5(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
};

class SDPVolumeHooks6 : public SDPVolumeHooks {
private:
	std::string eigen_solver; // "lanczos" or "full"
	double alpha; // trace constraint for the SDP relaxation
	int n; // problem size

	Eigen::MatrixXd W;
	Eigen::MatrixXd L;

public:
	SDPVolumeHooks6(const Problem& data, VOL_problem& vol_prob, std::string eigen_solver);
	virtual bool set_fixed_variables(const FixedVariables &fv) override;
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	int solve_subproblem_6(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);

};

class SDPVolumeHooks7 : public SDPVolumeHooks {
public:
	std::vector<int> best_assignment; // best assignment for x subproblem
	double best_assignment_cost; // cost of best assignment
	
	SDPVolumeHooks7(const Problem& data, VOL_problem& vol_prob);
	virtual bool set_fixed_variables(const FixedVariables &fv) override;
	int solve_subproblem_7(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
};

struct VolumeResult {
	int status = -1;           // 0 success, non-zero otherwise
	double lower_bound = 0.0;
	std::vector<double> primal;
	VOL_dvector dual;
};

VolumeResult solve_sdp_volume_relax(const Problem &problem,
									 const FixedVariables &fixed,
									 bool verbose,
									 const VOL_dvector &initial_dual,
									 std::string formulation = "formulation3");
bool parse_fixed_file(const std::string &path, int n, FixedVariables &out);
void set_up_vol_problem_1(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_2(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_3(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_4(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_5(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_6(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_7(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);

void set_up_volume_parameters(VOL_problem &vol_problem, bool verbose);
FixedViolationReport check_fixed_violations(const FixedVariables &fv, const VOL_dvector &psol, int n, double tol = 1e-6);
PrimalViolationSummary compute_primal_violation_summary(const VOL_dvector &psol, int n);
