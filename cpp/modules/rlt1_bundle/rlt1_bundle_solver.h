#pragma once
// Forward declarations and includes
#include <vector>
#include <string>
#include <memory>
#include <chrono>
// ... add other necessary includes


#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <omp.h>
#include "../../core/problem.h"
#include "FiOracle.h"
#define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v))

using namespace NDO_di_unipi_it;

class RLT1Oracle : public FiOracle {
public:
    RLT1Oracle(const Problem& problem, const std::string& formulation)
        : problem(problem),
        n(problem.n),
        needs_eval(true), 
        pcost(0.0), 
        lcost(0.0), 
        formulation(formulation),
        iteration(0),
        mFiStatus(FiStatus::kFiNorm),
        last_checkpoint_iteration(0),
        last_checkpoint_fi(0),
        best_fi(0),
        start_time(std::chrono::high_resolution_clock::now())
    {
        SetUp();
    }

    void SetLamBase(
            cIndex_Set LmbdB,
            cIndex LmbdBD)
    {
        FiOracle::SetLamBase(LmbdB, LmbdBD);
    }

    void SetLambda(cLMRow L)
    {
        FiOracle::SetLambda(L);
        needs_eval = true;
    }

    FiStatus GetFiStatus( Index wFi = Inf< Index >() )
    {
        return mFiStatus;
    }

    void SetUp();
    void SetUp1();
    void SetUp2();
    void SetUp3();

    HpNum Fi(cIndex wFi = Inf< Index >()) override;
    Index GetGi(SgRow SubG, cIndex_Set &SGBse,
                cIndex Name = Inf< Index >(), cIndex strt = 0,
                Index stp = Inf< Index >()) override;
    void SolveSubproblem();
    void SolveSubproblemTest();
    void SolveSubproblem1();
    void SolveSubproblem2();
    void SolveSubproblem3();

    void RetrieveDensePi() {
        if (LamBase) {
            DensePi.assign(NumVar, 0.0);
            for (Index k = 0; k < LamBDim; ++k) {
                DensePi[LamBase[k]] = Lambda[k];
            }
        } else {
            DensePi.assign(Lambda, Lambda + NumVar);
        }
    }

    const Problem& problem;

    std::vector<double> DensePi; // Dense representation of the dual variables

    std::vector<double> psol;
    std::vector<double> vio;
    double lcost;
    double pcost;
    int n;
    std::vector<double> beta;       // beta[i,u,v] = min_j cost of y[i,u,j,v]=1
    std::vector<int> beta_j_ind;    // j that achieves beta[i,u,v]
    std::vector<double> alpha;      // alpha[i] = min_u cost of x[i,u]=1
    std::vector<int> alpha_u_ind;   // u that achieves alpha[i]

    std::vector<int> best_assignment; // best assignment for x subproblem
	double best_assignment_cost; // cost of best assignment
	

    // fixed variables
	std::unordered_map<int, int> x_fixed_1;
	std::unordered_map<int, std::unordered_set<int>> x_fixed_0;
	std::unordered_map<int, int> y_fixed_1;
	std::unordered_map<int, std::unordered_set<int>> y_fixed_0;
	std::unordered_map<int, double> map_fixed;

    std::string formulation;

    FiStatus mFiStatus;
    bool needs_eval;
    int iteration;
    double best_fi; // best Fi value found so far
    int last_checkpoint_iteration; // last iteration that improved best_fi
    double last_checkpoint_fi; // last Fi value that improved best_fi
    std::chrono::high_resolution_clock::time_point start_time;
};