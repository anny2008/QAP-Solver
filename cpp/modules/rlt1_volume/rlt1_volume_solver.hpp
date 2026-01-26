#pragma once
// Forward declarations and includes
#include <vector>
#include <string>
#include <memory>
// ... add other necessary includes


#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <omp.h>
#include "VolVolume.hpp"
#include "../../core/problem.h"

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

class RLT1VolumeHooks : public VOL_user_hooks {
public:
    const Problem& qap_data;
    int n;
    // Fixed variables
    FixedVariables fixed;
public:
	RLT1VolumeHooks(const Problem& data): qap_data(data), n(data.n) {};
	virtual bool set_fixed_variables(const FixedVariables &fv) { fixed = fv; return true; };
	virtual int compute_rc(const VOL_dvector&, VOL_dvector& rc) override { rc=0;return 0; }
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) { return 0; };
	virtual int heuristics(const VOL_problem&, const VOL_dvector&, double&) override { return 0; }
};


class RLT1VolumeHooks1 : public RLT1VolumeHooks {
private:
    // Working arrays for subproblem
    std::vector<double> beta;       // beta[i,u,v] = min_j cost of y[i,u,j,v]=1
    std::vector<int> beta_j_ind;    // j that achieves beta[i,u,v]
    std::vector<double> alpha;      // alpha[i] = min_u cost of x[i,u]=1
    std::vector<int> alpha_u_ind;   // u that achieves alpha[i]

public:
	RLT1VolumeHooks1(const Problem& data);
	int solve_subproblem_1(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	};

class RLT1VolumeHooks2 : public RLT1VolumeHooks {
public:
	RLT1VolumeHooks2(const Problem& data);
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	int solve_subproblem_2(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
};

class RLT1VolumeHooks3 : public RLT1VolumeHooks {
private:
	std::vector<double> zsol; // working array for z variables
public:
	RLT1VolumeHooks3(const Problem& data);
	virtual bool set_fixed_variables(const FixedVariables &fv) override;
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	int solve_subproblem_3(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
};

class RLT1VolumeHooks4 : public RLT1VolumeHooks {
private:
	// Working arrays for subproblem
	std::vector<double> beta;       // beta[i,u,v] = min_j cost of y[i,u,j,v]=1
	std::vector<int> beta_j_ind;    // j that achieves beta[i,u,v]
	std::vector<double> alpha;      // alpha[i] = min_u cost of x[i,u]=1
	std::vector<int> alpha_u_ind;   // u that achieves alpha[i]

	std::vector<double> zsol; // working array for z variables 
public:
	RLT1VolumeHooks4(const Problem& data);
	virtual bool set_fixed_variables(const FixedVariables &fv) override;
	int solve_subproblem_4(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
};

struct VolumeResult {
	int status = -1;           // 0 success, non-zero otherwise
	double lower_bound = 0.0;
	std::vector<double> primal;
	VOL_dvector dual;
};

VolumeResult solve_rlt1_volume_relax(const Problem &problem,
									 const FixedVariables &fixed,
									 bool verbose,
									 const VOL_dvector &initial_dual,
									 std::string formulation = "formulation3");
bool parse_fixed_file(const std::string &path, int n, FixedVariables &out);
void set_up_vol_problem_1(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_2(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_3(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_vol_problem_4(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_volume_parameters(VOL_problem &vol_problem, bool verbose);
FixedViolationReport check_fixed_violations(const FixedVariables &fv, const VOL_dvector &psol, int n, double tol = 1e-6);
PrimalViolationSummary compute_primal_violation_summary(const VOL_dvector &psol, int n);
