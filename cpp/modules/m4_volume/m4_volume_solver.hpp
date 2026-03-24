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
#include <map>
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

struct VolumeResult {
	int status = -1;           // 0 success, non-zero otherwise
	double lower_bound = 0.0;
	VOL_dvector primal;
	VOL_dvector dual;
};

class M4VolumeHooks : public VOL_user_hooks {
public:
    const Problem& qap_data;
    int n;
    // Fixed variables
    FixedVariables fixed;
	std::map<int, int> violated_constraints; // Store indices of violated constraints

public:
	M4VolumeHooks(const Problem& data): qap_data(data), n(data.n), violated_constraints() {};
	virtual bool set_fixed_variables(const FixedVariables &fv) { fixed = fv; return true; };
	virtual int compute_rc(const VOL_dvector&, VOL_dvector& rc) override { rc=0;return 0; }
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) { return 0; };
	virtual int heuristics(const VOL_problem&, const VOL_dvector&, double&) override { return 0; }
};


class M4VolumeHooks1 : public M4VolumeHooks {
private:

public:
	M4VolumeHooks1(const Problem& data);
	int solve_subproblem_1(const VOL_dvector& pi, double& lcost, VOL_dvector& psol, VOL_dvector& vio, double& pcost);
	virtual int solve_subproblem(const VOL_dvector&, const VOL_dvector&, double&, VOL_dvector&, VOL_dvector&, double&) override;
	};

VolumeResult solve_m4_volume_relax(const Problem &problem,
									 const FixedVariables &fixed,
									 bool verbose,
									 const VOL_dvector &initial_dual,
									 std::string formulation,
									std::map<int, int>& violated_constraints);
bool parse_fixed_file(const std::string &path, int n, FixedVariables &out);
void set_up_vol_problem_1(VOL_problem &vol_problem, const Problem &qap_data, const int number_of_violated_constrains);
void set_up_volume_parameters(VOL_problem &vol_problem, bool verbose);
FixedViolationReport check_fixed_violations(const FixedVariables &fv, const VOL_dvector &psol, int n, double tol = 1e-6);
PrimalViolationSummary compute_primal_violation_summary(const VOL_dvector &psol, int n);
