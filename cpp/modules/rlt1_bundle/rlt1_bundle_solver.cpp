#include "rlt1_bundle_solver.h"
#include <iostream>
#include <fstream>
#include "Hungarian.h"
#include <omp.h>

void RLT1Oracle::SetUp()
{
    if (formulation == "formulation1") {
        SetUp1();
    } else if (formulation == "formulation2") {
        SetUp2();
    } else if (formulation == "formulation3") {
        SetUp3();
    } else {
        std::cerr << "Error: Unknown formulation type: " << formulation << std::endl;
        exit(1);
    }
}

void RLT1Oracle::SetUp1()
{
    // Initialize psol and vio with the appropriate sizes
    psol.resize(n * n + n * n * n * n, 0.0); // x and y variables
    vio.resize(2 * n + 2 * n * n * n, 0.0); // vio_mu1, vio_mu2, vio_theta1, vio_theta2
    NumVar = static_cast<Index>(vio.size());
}

void RLT1Oracle::SetUp2()
{
    // Initialize psol and vio with the appropriate sizes
    psol.resize(n * n + n * n * n * n, 0.0); // x and y variables
    vio.resize(n + n * n * n + n * n * n * n, 0.0); // vio_mu1, vio_mu2, vio_theta1, vio_theta2
    NumVar = static_cast<Index>(vio.size());

    // Initialize beta and alpha arrays
    beta.resize(n * n * n, 0.0);
    beta_j_ind.resize(n * n * n, 0);
    alpha.resize(n, 0.0);
    alpha_u_ind.resize(n, 0);

    // Initialize fixed variable maps
    x_fixed_1.clear();
    x_fixed_0.clear();
    y_fixed_1.clear();
    y_fixed_0.clear();
    // Initialize fixed variable maps from the problem's fixed assignments
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        x_fixed_1[i] = u;
    }
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                x_fixed_0[i].insert(j);
            }
        }
    }
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                y_fixed_1[key_y(n, i, u, j)] = u; // Assuming fixed y corresponds to fixed x
            }
        }
    }
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                y_fixed_0[key_y(n, i, u, j)].insert(u); // Assuming fixed y corresponds to fixed x
            }
        }
    }
}

void RLT1Oracle::SetUp3()
{
    // Initialize psol and vio with the appropriate sizes
    psol.resize(n * n + n * n * n * n, 0.0); // x and y variables
    vio.resize(2 * n * n * n, 0.0); // vio_alpha, vio_beta
    NumVar = static_cast<Index>(vio.size());

    best_assignment.resize(n, -1); // Initialize best assignment with -1 (unassigned)
    best_assignment_cost = -1; // Initialize best assignment cost

    // Initialize fixed variable maps
    x_fixed_1.clear();
    x_fixed_0.clear();
    y_fixed_1.clear();
    y_fixed_0.clear();
    // Initialize fixed variable maps from the problem's fixed assignments
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        x_fixed_1[i] = u;
    }
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                x_fixed_0[i].insert(j);
            }
        }
    }
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                y_fixed_1[key_y(n, i, u, j)] = u; // Assuming fixed y corresponds to fixed x
            }
        }
    }
    for (const auto& kv : problem.fixed_assignments) {
        int i = kv.first;
        int u = kv.second;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                y_fixed_0[key_y(n, i, u, j)].insert(u); // Assuming fixed y corresponds to fixed x
            }
        }
    }
}
HpNum RLT1Oracle::Fi(cIndex wFi)
{
    if (wFi == 0)
        return 0.0;

    if (needs_eval) {
        SolveSubproblem();
        needs_eval = false;
    }

    return -lcost;
}

Index RLT1Oracle::GetGi(
        SgRow SubG,
        cIndex_Set &SGBse,
        cIndex Name,
        cIndex strt,
        Index stp)
{
    if(Name == MaxName)
    {
        Index end = std::min(stp, NumVar);

        std::fill(SubG,
                  SubG + (end - strt),
                  0.0);

        SGBse = nullptr;
        return end - strt;
    }

    if (needs_eval) {
        SolveSubproblem();
        needs_eval = false;
    }

    Index begin = std::max<Index>(0, strt);
    Index end   = std::min<Index>(NumVar, stp);

    for(Index i = begin; i < end; ++i)
        SubG[i - begin] = -vio[i];

    SGBse = nullptr;
    LHasChgd = false;

    return end - begin;
}

void RLT1Oracle::SolveSubproblem()
{
    if (formulation == "formulation1") {
        SolveSubproblem1();
    } else if (formulation == "formulation2") {
        SolveSubproblem2();
    } else if (formulation == "formulation3") {
        SolveSubproblem3();
    } else {
        std::cerr << "Error: Unknown formulation type: " << formulation << std::endl;
        exit(1);
    }
    if (lcost > best_fi) {
        best_fi = lcost;
    }
    // if after 500 iterations the cost has not improved, stop the algorithm
    if(best_fi > 0 && iteration - last_checkpoint_iteration >= 1000) {
        double improvement = lcost - last_checkpoint_fi;
        if(improvement/(lcost+1e-16) < 1e-6) {
            std::cout << "No significant improvement in Fi for 1000 iterations. Stopping." << std::endl;
            mFiStatus = kFiStop;
        } else {
            last_checkpoint_iteration = iteration;
            last_checkpoint_fi = lcost;
        }
    } else if(iteration == 0) {
        last_checkpoint_iteration = iteration;
        last_checkpoint_fi = lcost;
    }

    if(iteration % 100 == 0) {
        auto ellapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << "Iteration: " << iteration << ", Fi: " << lcost << ", Best Fi: " << best_fi << ", Time: " << ellapsed << " seconds" << std::endl;
    }
    iteration++;
}

#define x(i, u) (psol[(i) * n + (u)])
#define y(i, u, j, v) (psol[n * n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define mu1(u) (pi[u])
#define mu2(i) (pi[n+i])
#define theta1(i, u, v) (pi[2*n + (i) * n * n + (u) * n + (v)])
#define theta2(i, u, j) (pi[2*n + n*n*n + (i) * n * n + (u) * n + (j)])
#define vio_mu1(u) (vio[u])
#define vio_mu2(i) (vio[n+i])
#define vio_theta1(i, u, v) (vio[2*n + (i) * n * n + (u) * n + (v)])
#define vio_theta2(i, u, j) (vio[2*n + n*n*n + (i) * n * n + (u) * n + (j)])

void RLT1Oracle::SolveSubproblem1() 
{
    RetrieveDensePi(); // Ensure DensePi is up-to-date with current Lambda
    auto pi = DensePi;

    // Step 1: Initialize costs and solution vectors
    lcost = 0.0;
    pcost = 0.0;
    std::fill(psol.begin(), psol.end(), 0.0); // Set all primal variables to zero
	std::fill(vio.begin(), vio.end(), 0.0); // Set all violation variables to zero

    // Step 2: Set y variables by checking reduced cost for each (i,u,j,v) and its symmetric (j,v,i,u)
    // If the reduced cost (coeff_y) is negative, set both y(i,u,j,v) and y(j,v,i,u) to 1 (symmetry)
    #pragma omp parallel for collapse(2) reduction(+:pcost) schedule(static)
    for(int i=0;i<n;i++) {
        for(int u=0;u<n;u++) {
            auto it_fixed_x1_i = problem.fixed_assignments.find(i);
            if (it_fixed_x1_i != problem.fixed_assignments.end()) {
                auto uu = it_fixed_x1_i->second;
                if (uu != u) {
                    // x_iu = 0
                    continue;
                }
            }
            for(int j=i + 1;j<n;j++) { 
                auto it_fixed_x1_j = problem.fixed_assignments.find(j);
                for(int v=0;v<n;v++) {
                    if(u == v) {
                        continue;
                    }

                    // if y(i,u,j,v) is fixed
                    // if x_iu or x_jv is fixed to 0, then y(i,u,j,v) must be 0
                    // if x_iu and x_jv are both fixed to 1, then y(i,u,j,v) must be 1
                    if (it_fixed_x1_j != problem.fixed_assignments.end()) {
                        auto vv = it_fixed_x1_j->second;
                        if (vv != v) {
                            // x_jv = 0
                            continue;
                        } else if (it_fixed_x1_i != problem.fixed_assignments.end()) {
                            // x_iu = 1 and x_jv = 1, then y(i,u,j,v) must be 1
                            y(i, u, j, v) = 1.0;
                            y(j, v, i, u) = 1.0;
                            pcost += problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                            continue;
                        }
                    }
                    
                    // Compute reduced cost for y(i,u,j,v)
                    auto coeff_y = problem.D[i][j] * problem.F[u][v]
                                    + problem.D[j][i] * problem.F[v][u]
                                    - theta1(i, u, v) - theta1(j, v, u)
                                    - theta2(i, u, j) - theta2(j, v, i);
                    // If negative, set y(i,u,j,v) and y(j,v,i,u) to 1 (enforce symmetry)
                    if (coeff_y < 0) {
                        y(i, u, j, v) = 1.0;
                        y(j, v, i, u) = 1.0;
                        pcost += problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                    }
                }
            }
        }
    }


    // Step 3: Set x variables by checking reduced cost for each (i,u)
    // If the reduced cost (coeff_x) is negative, set x(i,u) = 1
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            // Check if x(i,u) is fixed
            auto it_x1 = problem.fixed_assignments.find(i);
            if (it_x1 != problem.fixed_assignments.end()) {
                int fixed_u = it_x1->second;
                if (u == fixed_u) {
                    x(i, u) = 1.0;
                }
                continue;
            }
            // Compute reduced cost for x(i,u)
            auto coeff_x = -mu1(u) - mu2(i);
            for (int v = 0; v < n; ++v) {
                coeff_x += theta1(i, u, v) + theta2(i, u, v);
            }
            // If negative or zero, set x(i,u) = 1
            if (coeff_x < 0) {
                x(i, u) = 1.0;
            // } else if (coeff_x < 1e-9) {
            //     std::cout << "Reduced cost for x(" << i << "," << u << ") is zero, setting x(i,u) = 1 for simplicity." << std::flush;
                // If reduced cost is zero, we can choose to set x(i,u) = 1 or 0
                // Here we choose to set it to 1 for simplicity
                // auto random_0_1 = (rand() % 2); // Randomly choose 0 or 1
                // x(i, u) = random_0_1; // Randomly choose 1 or 0
                // x(i,u) = 1.0; // For simplicity, we set it to 1
            }
        }
    }
    lcost = pcost;
    // Step 4: Compute constraint violations for the current solution
    // Assignment constraint 1: For each u, sum_i x[i,u] should be 1
    // Store violation in vio_mu1(u) = 1 - sum_i x[i,u]
    #pragma omp parallel for schedule(static) reduction(+:lcost)
    for (int u = 0; u < n; ++u) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (int i = 0; i < n; ++i) {
            sum1 += x(i, u);
            sum2 += x(u, i);
        }
        vio_mu1(u) = 1.0 - sum1;
        vio_mu2(u) = 1.0 - sum2;
        lcost += mu1(u)*vio_mu1(u) + mu2(u)*vio_mu2(u);
    }

    // Linking constraint 2: For each (i,u,v), x[i,u] = sum_j y[i,u,j,v]
    // Store violation in vio_theta1(i,u,v) = x[i,u] - sum_j y[i,u,j,v]
    #pragma omp parallel for collapse(3) schedule(static) reduction(+:lcost)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                if (v != u) {
                    double sum1 = 0.0;
                    for (int j = 0; j < n; ++j) {
                        if (j != i)
                        sum1 += y(i, u, j, v);
                    }
                    vio_theta1(i, u, v) = x(i, u) - sum1;
                    lcost += theta1(i,u,v)*vio_theta1(i,u,v);
                }
                if (v != i) {

                    double sum2 = 0.0;
                    for (int j = 0; j < n; ++j) {
                        if (j != u)
                        sum2 += y(i, u, v, j);
                    }
                    vio_theta2(i, u, v) = x(i, u) - sum2;
                    lcost += theta2(i,u,v)*vio_theta2(i,u,v);
                }
            }
        }
    }
}


#define mu(u) (pi[u])
#define lambda(i, u, j, v) (pi[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define theta(i, u, j) (pi[n + n * n * n * n + (i) * n * n + (u) * n + (j)])
#define vio_mu(u) (vio[u])
#define vio_lambda(i, u, j, v) (vio[n + (i) * n * n * n + (u) * n * n + (j) * n + (v)])
#define vio_theta(i, u, j) (vio[n + n * n * n * n + (i) * n * n + (u) * n + (j)])
#define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v))
// #define key_x(n, i, u) ((i)*(n) + (u))

/**
    * Solve RLT1 Lagrangian Subproblem
    * 
    * DUAL VARIABLES (pi):
    *   - mu[u]:           Lagrange multipliers for sum_i x[i,u] = 1
    *   - lambda[i,u,j,v]: Multipliers for y[i,u,j,v] = y[j,v,i,u]
    *   - theta[i,u,j]:    Multipliers for sum_v y[i,u,j,v] = x[i,u]
    * 
    * LAGRANGIAN:
    *   L = sum_{i,j,u,v} d[i,j]*f[u,v]*y[i,u,j,v]
    *       + sum_u mu[u] * (1 - sum_i x[i,u])
    *       + sum_{i,u,j,v} lambda[i,u,j,v] * (y[j,v,i,u] - y[i,u,j,v])
    *        + sum_{i,u,j} theta[i,u,j] * (x[i,u] - sum_v y[i,u,j,v])
    * 
    * ALGORITHM:
    *   1. Compute beta[i,u,v] = min_j {Lagrangian cost of y[i,u,j,v]=1}
    *   2. Compute alpha[i] = min_u {cost of x[i,u]=1 + sum_{j,v} beta[i,u,v]}
    *   3. Set x[i,u]=1 for u minimizing alpha[i]
    *   4. Set y[i,u,j,v]=1 for j minimizing beta[i,u,v]
    */
void RLT1Oracle::SolveSubproblem2()
{
    
    RetrieveDensePi(); // Ensure DensePi is up-to-date with current Lambda
    auto pi = DensePi;
    lcost = 0.0;
    pcost = 0.0;
    std::fill(psol.begin(), psol.end(), 0.0); // Set all primal variables to zero
    std::fill(vio.begin(), vio.end(), 0.0); // Set all violation variables to zero
    
    // Initialize working arrays
    std::fill(beta.begin(), beta.end(), 0.0);
    std::fill(beta_j_ind.begin(), beta_j_ind.end(), 0);
    std::fill(alpha.begin(), alpha.end(), 0.0);
    std::fill(alpha_u_ind.begin(), alpha_u_ind.end(), 0);
    
    // STEP 1: Compute beta[i,u,v] = min_j {Lagrangian cost of y[i,u,j,v]=1}
    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                double min_cost = DBL_MAX;
                int best_j = 0;

                int ykey = key_y(n, i, u, v);
                auto it_y1 = y_fixed_1.find(ykey);
                if (it_y1 != y_fixed_1.end()) {
                    int j = it_y1->second;
                    double cost = problem.D[i][j] * problem.F[u][v];
                        if (i <= j) cost -= lambda(i, u, j, v);
                        if (i >= j) cost += lambda(j, v, i, u);
                        cost -= theta(i, u, j);
                        min_cost = cost;
                        best_j = j;
                } else {
                    const auto it_y0 = y_fixed_0.find(ykey);
                    for (int j = 0; j < n; ++j) {
                        if (it_y0 != y_fixed_0.end() && it_y0->second.count(j)) {
                            continue; // skip fixed-to-zero y
                        }
                        double cost = problem.D[i][j] * problem.F[u][v];
                        if (i <= j) cost -= lambda(i, u, j, v);
                        if (i >= j) cost += lambda(j, v, i, u);
                        cost -= theta(i, u, j);
                        if (cost < min_cost) {
                            min_cost = cost;
                            best_j = j;
                        }
                    }
                }

                int idx = i * n * n + u * n + v;
                if (min_cost == DBL_MAX) {
                    // std::cout << "Warning: no feasible j for y[" << i << "," << u << ",*," << v << "] under fixed variables; choosing j=0 with large cost." << std::flush;
                    beta[idx] = 1e30;
                    beta_j_ind[idx] = 0;
                } else {
                    beta[idx] = min_cost;
                    beta_j_ind[idx] = best_j;
                }
            }
        }
    }
        
    // STEP 2: Compute alpha[i] = min_u {cost of x[i,u]=1}
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        double min_cost = DBL_MAX;
        int best_u = 0;

        // If x[i,*] has a fixed 1, only evaluate that u
        auto it_x1 = x_fixed_1.find(i);
        if (it_x1 != x_fixed_1.end()) {
            int u = it_x1->second;
            double cost = -mu(u);
            for (int j = 0; j < n; ++j) cost += theta(i, u, j);
            for (int v = 0; v < n; ++v) {
                int idx = i * n * n + u * n + v;
                cost += beta[idx];
            }
            min_cost = cost;
            best_u = u;
        } else {
            const auto it_x0 = x_fixed_0.find(i);
            for (int u = 0; u < n; ++u) {
                if (it_x0 != x_fixed_0.end() && it_x0->second.count(u)) {
                    continue; // fixed to zero
                }
                double cost = -mu(u);
                for (int j = 0; j < n; ++j) cost += theta(i, u, j);
                for (int v = 0; v < n; ++v) {
                    int idx = i * n * n + u * n + v;
                    cost += beta[idx];
                }
                if (cost < min_cost) {
                    min_cost = cost;
                    best_u = u;
                }
            }
        }

        if (min_cost == DBL_MAX) {
            // std::cout << "Warning: no feasible u for fixed variables at facility i=" << i << "; choosing u=0." << std::flush;
            alpha[i] = 1e30;
            alpha_u_ind[i] = 0;
        } else {
            alpha[i] = min_cost;
            alpha_u_ind[i] = best_u;
        }
    }
    
    // STEP 3: Compute Lagrangian lower bound
    #pragma omp parallel for reduction(+:lcost) schedule(static)
    for (int i = 0; i < n; ++i) {
        lcost += alpha[i];
    }
    
    // Add constant term: sum_u mu[u]
    for (int u = 0; u < n; ++u) {
        lcost += mu(u);
    }
    
    // STEP 4: Construct primal solution
    
    // Set x[i,u]=1 for chosen u
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int u = alpha_u_ind[i];
        x(i, u) = 1.0;
    }
    
    // Set y[i,u,j,v]=1 for chosen j at chosen u
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int v = 0; v < n; ++v) {
            int u = alpha_u_ind[i];
            int idx = i * n * n + u * n + v;
            int j = beta_j_ind[idx];
            y(i, u, j, v) = 1.0;
        }
    }
    
    // Compute primal objective (original QAP objective on fractional solution)
    #pragma omp parallel for collapse(2) reduction(+:pcost) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    pcost += problem.D[i][j] * problem.F[u][v] * y(i, u, j, v);
                }
            }
        }
    }
    
    // STEP 5: Compute constraint violations
    
    // Violation of assignment constraints: 1 - sum_i x[i,u]
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += x(i, u);
        }
        vio_mu(u) = 1.0 - sum;
    }
    
    // Violation of linking constraints: x[i,u] - sum_v y[i,u,j,v]
    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int v = 0; v < n; ++v) {
                    sum += y(i, u, j, v);
                }
                vio_theta(i, u, j) = x(i, u) - sum;
            }
        }
    }
    
    // Violation of symmetry constraints: y[j,v,i,u] - y[i,u,j,v]
    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
            for (int j = i; j < n; ++j) {
                    vio_lambda(i, u, j, v) = y(j, v, i, u) - y(i, u, j, v);
                }
            }
        }
    }
}

#define alpha(i,u,v) (pi[(i)*n*n + (u)*n + (v)])
#define beta(i,u,j) (pi[n*n*n + (i)*n*n + (u)*n + (j)])
#define vio_alpha(i,u,v) (vio[(i)*n*n + (u)*n + (v)])
#define vio_beta(i,u,j) (vio[n*n*n + (i)*n*n + (u)*n + (j)])


double solve_x_subproblem_by_hungarian(
                int N, int M,
                std::vector<double>& obj_coeff,
                std::vector<double>& pi, std::vector<int>& Assignment,
                std::unordered_map<int, int>& x_fixed_1)
{
    // assume that fixed locations are move to the end of the list of locations,
    //  and fixed facilities are move to the end of the list of facilities.
    int n = N - x_fixed_1.size();
    int m = M - x_fixed_1.size();
	double *distMatrixIn = new double[n * m];
	int *assignment = new int[n];
	double cost = 0.0;
    // because all cost is non positive, we convert to non-negative cost by adding a large enough constant to all costs. This does not change the optimal assignment, but ensures that the Hungarian algorithm works correctly.
    // double min_cost = 0.0;
	// Fill in the distMatrixIn. Mind the index is "i + nRows * j".
	// Here the cost matrix of size MxN is defined as a double precision array of N*M elements. 
	// In the solving functions matrices are seen to be saved MATLAB-internally in row-order.
	// (i.e. the matrix [1 2; 3 4] will be stored as a vector [1 3 2 4], NOT [1 2 3 4]).
    // cost(i,u) = -(sum_{j} beta[i,u,j] + sum_{v} alpha[i,u,v])
    auto bigM = 1000000; // a large constant to make the cost non-negative
    #pragma omp parallel for collapse(2) schedule(static)
	for (auto i = 0; i < n; i++)
    for (auto u = 0; u < m; u++)
    {
        distMatrixIn[i + n * u] = obj_coeff[i * M + u] + bigM; // add a large constant to make it non-negative
    }

	// call solving function
	HungarianAlgorithm hungarian;
	hungarian.assignmentoptimal(assignment, &cost, distMatrixIn, n, m);

    // std::cout << "Hungarian assignment cost1: " << cost - bigM*m << std::endl;
    // recompute the cost of the assignment in the original cost space
    cost = 0.0;

    // #pragma omp parallel for schedule(static) reduction(+:cost)
	for (auto i = 0; i < n; i++) {
        auto u = assignment[i];
        Assignment[i] = u;
        cost += obj_coeff[i * M + u];
    }
    // std::cout << "Hungarian assignment cost2: " << cost << std::endl;
    // add the cost of the fixed variables
    #pragma omp parallel for schedule(static) reduction(+:cost)
    for (auto i = n; i < N; i++) {
        auto it_fixed_x1_i = x_fixed_1.find(i);
        if (it_fixed_x1_i == x_fixed_1.end()) {
            std::cerr << "Error: fixed variable for facility " << i << " not found in x_fixed_1." << std::endl;
            continue;
        }
        auto u = it_fixed_x1_i->second;
        cost += obj_coeff[i * M + u];
        Assignment[i] = u;
    }

	delete[] distMatrixIn;
	delete[] assignment;
	return cost;
}


/**
    * Solve RLT1 Lagrangian Subproblem
    * 
    * DUAL VARIABLES (pi):
    *   beta[i,u,j]:     Multipliers for sum_v y[i,u,j,v] = x[i,u]
    *   alpha[i,u,v]:    Multipliers for sum_j y[i,u,j,v] = x[i,u]
    * 
    * LAGRANGIAN:
    *   L = sum_{i,j,u,v} d[i,j]*f[u,v]*y[i,u,j,v]
    *        - sum_{i,u,j} beta[i,u,j] * (x[i,u] - sum_v y[i,u,j,v])
    *        - sum_{i,u,v} alpha[i,u,v] * (x[i,u] - sum_j y[i,u,j,v])
    * 
    * ALGORITHM:
    *   1. Compute c_iujv = d[i,j]*f[u,v] + d[j,i]*f[v,u]
    *                        + beta[i,u,j] + alpha[i,u,v]
    *                        + beta[j,v,i] + alpha[j,v,u]
    *   2. Set y[i,u,j,v]= y[j,v,i,u] = 1 for c_iujv < 0, otherwise 0 
    *   3. Solve for x: 
    *       min -sum_{i,u} x[i,u]*(sum_{j} beta[i,u,j] + sum_{v} alpha[i,u,v])
    *       st. sum_{i} x[i,u] = 1 for all u
    *           sum_{u} x[i,u] <= 1 for all i
    *           x[i,u] in {0,1}
    *   4. Compute pcost = sum_{i,j,u,v} d[i,j]*f[u,v]*y[i,u,j,v]
    *   5. Compute violation for beta and alpha constraints:
    *        vio_beta[i,u,j] = x[i,u] - sum_v y[i,u,j,v]
    *        vio_alpha[i,u,v] = x[i,u] - sum_j y[i,u,j,v]
    *   6. Compute lcost = pcost - sum_{i,u,j} beta[i,u,j]*vio_beta[i,u,j]
    *                            - sum_{i,u,v} alpha[i,u,v]*vio_alpha[i,u,v]
    */
void RLT1Oracle::SolveSubproblem3()
{

    RetrieveDensePi(); // Ensure DensePi is up-to-date with current Lambda
    auto pi = DensePi;
    lcost = 0.0;
    pcost = 0.0;
    std::fill(psol.begin(), psol.end(), 0.0); // Set all primal variables to zero
    std::fill(vio.begin(), vio.end(), 0.0); // Set all violation variables to zero
    
    // auto locst_verify = 0.0;
    // Step 1: Compute c_iujv = d[i,j]*f[u,v] + d[j,i]*f[v,u] + beta[i,u,j] + alpha[i,u,v] + beta[j,v,i] + alpha[j,v,u]
    #pragma omp parallel for collapse(3) schedule(static) reduction(+:pcost, lcost)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                if (u == v) {
                    continue;
                }
                for (int j = i + 1; j < n; ++j) {
                    // fixed variables: if x_iu is fixed to 0, then y_iujv must be 0 for all j,v; if x_iu is fixed to 1, then y_iujv can be 1 for some j,v
                    auto it_xiu = x_fixed_1.find(i);
                    auto it_xjv = x_fixed_1.find(j);
                    if (it_xiu != x_fixed_1.end()) {
                        if (it_xiu->second != u)
                            continue; // x[i,u] is fixed to 0, so y[i,u,j,v] must be 0
                        else if (it_xjv != x_fixed_1.end() && it_xjv->second == v) {
                            // x[i,u] is fixed to 1, x[j,v] is fixed to 1, so y[i,u,j,v] = 1
                            y(i, u, j, v) = 1.0;
                            y(j, v, i, u) = 1.0; // enforce symmetry
                            double c_iujv = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]
                                            + beta(i, u, j) + alpha(i, u, v)
                                            + beta(j, v, i) + alpha(j, v, u);
                            pcost += problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                            lcost += c_iujv;
                            // locst_verify += c_iujv;
                            continue; 
                        }
                    } else if (it_xjv != x_fixed_1.end()) {
                        if (it_xjv->second != v)
                            continue; // x[j,v] is fixed to 0, so y[i,u,j,v] must be 0
                    }
                    double c_iujv = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u]
                                     + beta(i, u, j) + alpha(i, u, v)
                                     + beta(j, v, i) + alpha(j, v, u);
                    if (c_iujv < 0) {
                        y(i, u, j, v) = 1.0;
                        y(j, v, i, u) = 1.0; // enforce symmetry
                        pcost += problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                        lcost += c_iujv;
                        // locst_verify += c_iujv;
                    }
                    // assume that if c_iujv == 0, we can choose to set y[i,u,j,v] = 1 or 0; here we choose 0 for simplicity
                }
            }
        }
    }
    // Step 2: Solve for x
    // This is a linear assignment problem with costs c[i,u] = sum_j beta[i,u,j] + sum_v alpha[i,u,v]
    // Solving using hungarian algorithm
    std::vector<double> assignment_cost_matrix(n * n, 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            double cost = 0.0;
            for (int j = 0; j < n; ++j) {
                cost += beta(i, u, j);
            }
            for (int v = 0; v < n; ++v) {
                cost += alpha(i, u, v);
            }
            assignment_cost_matrix[i * n + u] = -cost; // negate for minimization
        }
    }
    std::vector<int> assignment(n, -1);
    double assignment_cost = 0.0;
    // if (x_fixed_1.size() > 0) 
    // {
    //     // std::cout << "Warning: fixed variables are present, using CPLEX to solve the assignment problem." << std::endl;
    //     assignment_cost = solve_x_subproblem_by_cplex(n, n, pi, assignment, x_fixed_1);
    // } else 
    // {
    //     assignment_cost = solve_x_subproblem_by_hungarian(n, n, pi, assignment, x_fixed_1);
    // }
    assignment_cost = solve_x_subproblem_by_hungarian(n, n, assignment_cost_matrix, pi, assignment, x_fixed_1);
    // double verify_assignment_cost = solve_x_subproblem_by_cplex(n, n, assignment_cost_matrix, pi, assignment, x_fixed_1);
    // if(abs(assignment_cost - verify_assignment_cost) > 1e-6) {
    //     std::cout << "Warning: Hungarian assignment cost (" << assignment_cost << ") does not match CPLEX assignment cost (" << verify_assignment_cost << ")." << std::endl;
    // }
    //  compute cost = sum_{i,u,j,v} d[i,j]*f[u,v]*x_iu*x_jv
    auto new_assignment_cost = 0.0;
    #pragma omp parallel for reduction(+:new_assignment_cost) schedule(static)
    for (int i = 0; i < n; ++i) {
        auto u = assignment[i];
        for (int j = 0; j < n; ++j) {
            auto v = assignment[j];
            new_assignment_cost += problem.D[i][j] * problem.F[u][v];
        }
    }
    if (best_assignment_cost < 0 || new_assignment_cost < best_assignment_cost) {
        best_assignment = assignment;
        best_assignment_cost = new_assignment_cost;
        std::cout << "New best assignment cost: " << best_assignment_cost << std::endl;
    }
    lcost += assignment_cost;
    // locst_verify += assignment_cost;
    // Set x[i,u] = 1 for assigned pairs
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int u = assignment[i];
        if (u >= 0 && u < n) {
            x(i, u) = 1.0;
        }
    }

    // Step 3: Compute violation for beta and alpha constraints
    // vio_beta[i,u,j] = x[i,u] - sum_v y[i,u,j,v]
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                double sum_v = 0.0;
                for (int v = 0; v < n; ++v) {
                    sum_v += y(i, u, j, v);
                }
                vio_beta(i, u, j) = -x(i, u) + sum_v;
                // lcost += beta(i, u, j) * vio_beta(i, u, j);
            }
            for (int v = 0; v < n; ++v) {
                if (u == v) {
                    continue;
                }
                double sum_j = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum_j += y(i, u, j, v);
                }
                vio_alpha(i, u, v) = -x(i, u) + sum_j;
                // lcost += alpha(i, u, v) * vio_alpha(i, u, v);
            }
        }
    }
}



void RLT1Oracle::SolveSubproblemTest() 
{
    // test using a simple problem min x^2 + y^2 subject to x + y = 1
    // The optimal solution is x = 0.5, y = 0.5, with cost = 0.5
    // Lagrangian: L(x,y,lambda) = x^2 + y^2 + lambda*(1 - x - y)
    // Lagrangian dual: max_lambda min_{x,y} L(x,y,lambda)
    // min_{x,y} L(x,y,lambda) = min_{x,y} (x^2 - lambda*x) + (y^2 - lambda*y) + lambda
    // = min_x (x^2 - lambda*x) + min_y (y^2 - lambda*y) + lambda
    // = -lambda^2/4 - lambda^2/4 + lambda = -lambda^2/2
    // max_lambda -lambda^2/2 = 0 at lambda = 0
    psol.resize(2, 0.0);
    vio.resize(1, 0.0);
    NumVar = 1;
    const double* pi = Lambda;
    std::vector<double> dense_pi(NumVar,0.0);

    if(LamBase)
    {
        std::cout << "Using LamBase to construct dense_pi." << std::endl;
        for(Index k=0;k<LamBDim;k++)
            dense_pi[LamBase[k]] = Lambda[k];

        pi = dense_pi.data();
    }
    else
    {
        pi = Lambda;
    }
    std::cout
    << "NumVar = " << NumVar
    << " LamBDim = " << LamBDim
    << " sparse = " << (LamBase != nullptr)
    << std::endl;

    // Step 1: Compute primal solution
    double lambda = pi[0];
    double x = lambda / 2.0;
    double y = lambda / 2.0;
    psol[0] = x;
    psol[1] = y;

    // Step 2: Compute primal cost
    pcost = x*x + y*y;

    // Step 3: Compute constraint violation
    vio[0] = 1.0 - x - y;

    // Step 4: Compute Lagrangian cost
    lcost = pcost + lambda * vio[0];

    std::cout << "Subproblem solved. lcost: " << lcost << ", pcost: " << pcost << ", vio: " << vio[0] << std::endl;
}
