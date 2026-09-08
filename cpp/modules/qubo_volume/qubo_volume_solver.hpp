#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
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

struct QUBOViolation {
    double row_max = 0.0;
    double col_max = 0.0;
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

class QUBOVolumeHooks : public VOL_user_hooks {
public:
    const Problem& qap_data;
    const int n;

    // fixed variables support
    FixedVariables fixed;

    // working arrays for subproblem
    std::vector<char> xbin;      // x(i,u) in {0,1} stored as char
    std::vector<double> s;       // s_k = sum_l Q'_{k,l} x_l
    std::vector<double> c;       // c_k = mu(u) + nu(i)
    std::vector<char> locked;    // lock flips for fixed vars

public:
    QUBOVolumeHooks(const Problem& data);

    bool set_fixed_variables(const FixedVariables& fv);

    // VOL callbacks
    virtual int solve_subproblem(const VOL_dvector& pi,
                                 const VOL_dvector& /*rc*/,
                                 double& lcost,
                                 VOL_dvector& psol,
                                 VOL_dvector& vio,
                                 double& pcost) override;
    virtual int compute_rc(const VOL_dvector&, VOL_dvector& rc) override { rc=0;return 0; }
	virtual int heuristics(const VOL_problem&, const VOL_dvector&, double&) override { return 0; }

    // helper: index mapping
    inline int idx(int i, int u) const { return i * n + u; }

private:
    // Gain-based local search for UQBP
    void initialize_from_fixed_and_linear(const VOL_dvector& pi);
    double qubo_solve_with_cplex(const VOL_dvector& pi, VOL_dvector& psol);

    // compute pcost via matrix products O(n^3)
    double compute_primal_qap_cost(const VOL_dvector& psol) const;

    // Fill vio (size 2n) and return constant term sum(mu)+sum(nu)
    void fill_violations(const VOL_dvector& pi,
                                    const VOL_dvector& psol,
                                    VOL_dvector& vio) const;
};

// Problem setup + solver entry points
void set_up_vol_problem_qubo(VOL_problem& vol_problem, const Problem& qap_data);

VolumeResult solve_qap_qubo_volume_relax(const Problem &problem,
                                         const FixedVariables &fixed,
                                         bool verbose,
                                         const VOL_dvector &initial_dual);

bool parse_fixed_file(const std::string &path, int n, FixedVariables &out);
void set_up_vol_problem_qubo(VOL_problem &vol_problem, const Problem &qap_data, const FixedVariables &fixed);
void set_up_volume_parameters(VOL_problem &vol_problem, bool verbose);
QUBOViolation compute_qubo_violation(const VOL_dvector &psol, int n);