/*
 * Put the dual from the VA in CPLEX/Gurobi as warmstart to solve to optimal
*/
#include <gurobi_c++.h>
#include "rlt1_volume_solver.hpp"

class ExactRefine 
{
public:

    ExactRefine(Problem& problem, VOL_dvector& psol, double LB)
                : problem(problem), psol(psol), LB(LB)
    {
        // check_dual_warmstart();
    }

    // ExactRefine(Problem& problem, VOL_dvector& warmstart)
    //             : problem(problem), warmstart(warmstart)
    // {
    //     check_dual_warmstart();
    // }

    void solve();
    // void check_dual_warmstart();

private:
    Problem& problem;
    VOL_dvector& psol;
    double LB;
    // VOL_dvector& warmstart;

    void create_model_RLT1(GRBModel& model, double threshold);
    void create_model_RLT1_reduction(GRBModel& model, double threshold);
    void create_model_SFD(GRBModel& model, double threshold);
};