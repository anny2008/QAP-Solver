#pragma once

#include <Eigen/Dense>

#include "../../core/problem.h"
using MatrixXb = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

Eigen::MatrixXd build_Vchoice3(int n);
Eigen::MatrixXd build_Vhat(int n);

class ADMMSolver {
public:
    ADMMSolver() : max_iter(1000), beta(1.0), gamma(1.618), tol(1e-5), problem(nullptr) {}
    
    void setParameters(int max_iter, double beta, double gamma, double tol)
    {
        this->max_iter = max_iter;
        this->beta = beta;
        this->gamma = gamma;
        this->tol = tol;
    }

    void setProblem(Problem* prob) {
        problem = prob;
    };
    void initialize();
    void runADMM(bool is_low_rank);
    double obtainLowerBound();

private:
    int max_iter;
    double beta;
    double gamma;
    double tol;

    Problem* problem;

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
};