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
#include <Eigen/Dense>
#include <fusion.h>
#include <monty.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>

using namespace mosek::fusion;
using namespace monty;
#define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v))

using namespace NDO_di_unipi_it;
using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * SDPOracle class implements the Lagrangian oracle for the SDP relaxation of the QAP.
 * It inherits from FiOracle and provides methods to evaluate the Lagrangian dual function,
 * compute subgradients, and solve the Lagrangian subproblem.
 * Formulations:
 * - formulation1: Zhao's relaxation, Lagragian of F.Rendl et al. 2008
 * - formulation2: 
 * - formulation3: 
 */


class SDPOracle : public FiOracle {
public:
    SDPOracle(const Problem& problem, const std::string& formulation)
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
        start_time(std::chrono::high_resolution_clock::now()),
        subproblem_solving_time(0.0)
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
    void SetUp4();

    HpNum Fi(cIndex wFi = Inf< Index >()) override;
    Index GetGi(SgRow SubG, cIndex_Set &SGBse,
                cIndex Name = Inf< Index >(), cIndex strt = 0,
                Index stp = Inf< Index >()) override;
                
    bool GetUC( cIndex i );
    bool GetUC1( cIndex i );
    bool GetUC2( cIndex i );
    bool GetUC3( cIndex i );
    bool GetUC4( cIndex i );
    LMNum GetUB( cIndex i );

    void SolveSubproblem();
    void SolveSubproblemTest();
    void SolveSubproblem1();
    void SolveSubproblem2();
    void SolveSubproblem3();
    void SolveSubproblem4();

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
    int dsize;

    std::vector<double> DensePi; // Dense representation of the dual variables

    std::vector<double> psol;
    std::vector<double> vio;
    double lcost;
    double pcost;
    int n;
    std::vector<int> best_assignment; // best assignment for x subproblem
	double best_assignment_cost; // cost of best assignment


    void SolveSubproblem1MOSEK(); 
	
    Eigen::MatrixXd L; // L matrix for the SDP relaxation
    Eigen::MatrixXd W; // W matrix for the SDP relaxation
    Model::t M; // MOSEK model for the SDP subproblem
    Parameter::t pi_param;
    Variable::t R; // PSD matrix variable for the SDP subproblem
    Expression::t WRWT; // Expression for W*R*W^T


	std::string eigen_solver; // "lanczos" or "full" or "power_iteration"
	double alpha; // trace constraint for the SDP relaxation
	std::vector<int> quad_to_compact;
    
    
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
	std::chrono::duration<double> subproblem_solving_time;
};