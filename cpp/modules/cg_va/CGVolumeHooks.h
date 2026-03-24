#ifndef CGVOLUMEHOOKS_H
#define CGVOLUMEHOOKS_H

#include <unordered_map>
#include <vector>
#include "../../core/problem.h"
#include "../../core/solution.h"
#include "ColumnGenSolver.h"
#include "VolVolume.hpp"

class CGVolumeHooks : public VOL_user_hooks {
public:
    CGVolumeHooks(const Problem& prob,
                  const std::unordered_map<QuadKey,int,QuadKeyHash>& y_index_map);

    int solve_subproblem(const VOL_dvector& pi,
                         const VOL_dvector& rc,
                         double& lcost,
                         VOL_dvector& psol,
                         VOL_dvector& vio,
                         double& pcost) override;
    int compute_rc(const VOL_dvector& u, VOL_dvector& rc) override;
    int heuristics(const VOL_problem& p, const VOL_dvector& x, double& heur_val) override;

private:
    const Problem& P;
    const std::unordered_map<QuadKey,int,QuadKeyHash>& ymap;
    int iteration = 0;
    VOL_dvector prev_pi;
};

#endif