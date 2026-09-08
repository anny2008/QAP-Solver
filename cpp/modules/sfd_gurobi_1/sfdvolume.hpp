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

struct SubgraphData; 

class SFDVolumeHooks : public VOL_user_hooks {
public:
	int n;
	int K;
	std::vector<double> e_sol;
	std::vector<double> x_sol;
	std::vector<int> uv_to_k;
	Problem& problem;
    std::map<int, SubgraphData>& subgraphs;
	std::map<std::string, double> farkas_ray; // Store the Farkas ray for verification

public:
	SFDVolumeHooks(Problem& problem, std::map<int, SubgraphData>& subgraphs);
	virtual int compute_rc(const VOL_dvector&, VOL_dvector& rc) override;
	virtual int solve_subproblem(const VOL_dvector &pi, const VOL_dvector &rc,
                                double &lcost, VOL_dvector &psol, VOL_dvector &vio,
                                double &pcost) override;
	virtual int heuristics(const VOL_problem&, const VOL_dvector&, double&) override;
};
