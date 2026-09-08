#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <sstream>
#include "qubo_volume_solver.hpp"
#include <ilcplex/ilocplex.h>

#define key_y(n, i, u, v) ((i)*(n)*(n) + (u)*(n) + (v))
bool parse_fixed_file(const std::string &path, int n, FixedVariables &out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cout << "\rError: cannot open fixed-variable file: " << path << std::flush;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        std::string trimmed = line;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](int ch) { return !std::isspace(ch); }));
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::istringstream iss(trimmed);
        char type;
        iss >> type;
        if (type == 'x') {
            int i, u, val;
            if (!(iss >> i >> u >> val)) {
                std::cout << "\rWarning: malformed x-line at " << path << ":" << lineno << std::flush;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::flush;
                continue;
            }
            if (val == 1) {
                out.x_fixed_1[i] = u;
            } else {
                out.x_fixed_0[i].insert(u);
            }
        } else if (type == 'y') {
            int i, u, j, v, val;
            if (!(iss >> i >> u >> j >> v >> val)) {
                std::cout << "\rWarning: malformed y-line at " << path << ":" << lineno << std::flush;
                continue;
            }
            if (i < 0 || i >= n || u < 0 || u >= n || j < 0 || j >= n || v < 0 || v >= n || (val != 0 && val != 1)) {
                std::cout << "\rWarning: invalid indices/value at " << path << ":" << lineno << std::flush;
                continue;
            }
            int key = key_y(n, i, u, v);
            if (val == 1) {
                out.y_fixed_1[key] = j;
            } else {
                out.y_fixed_0[key].insert(j);
            }
        } else {
            std::cout << "\rWarning: unknown line type at " << path << ":" << lineno << std::flush;
        }
    }

    std::cout << "\rLoaded fixed variables from " << path
              << " | x1=" << out.x_fixed_1.size()
              << " x0-rows=" << out.x_fixed_0.size()
              << " y1=" << out.y_fixed_1.size()
              << " y0-rows=" << out.y_fixed_0.size() << std::flush;
    return true;
}

QUBOViolation compute_qubo_violation(const VOL_dvector &psol, int n)
{
    QUBOViolation v;

    // row constraints: sum_u x(i,u) = 1
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int u = 0; u < n; ++u)
            sum += psol[i*n + u];
        v.row_max = std::max(v.row_max, fabs(1 - sum));
    }

    // column constraints: sum_i x(i,u) = 1
    for (int u = 0; u < n; ++u) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += psol[i*n + u];
        v.col_max = std::max(v.col_max, fabs(1 - sum));
    }

    return v;
}
FixedViolationReport check_fixed_violations(const FixedVariables &fv, const VOL_dvector &psol, int n, double tol) {
    FixedViolationReport rep;

    // x fixed to 1
    for (const auto &kv : fv.x_fixed_1) {
        int i = kv.first;
        int u = kv.second;
        double val = psol[i * n + u];
        double dev = std::abs(val - 1.0);
        if (dev > tol) {
            ++rep.x1_violations;
            rep.x1_max_dev = std::max(rep.x1_max_dev, dev);
        }
    }

    // x fixed to 0
    for (const auto &kv : fv.x_fixed_0) {
        int i = kv.first;
        for (int u : kv.second) {
            double val = psol[i * n + u];
            double dev = std::abs(val);
            if (dev > tol) {
                ++rep.x0_violations;
                rep.x0_max_dev = std::max(rep.x0_max_dev, dev);
            }
        }
    }

    // y fixed to 1
    for (const auto &kv : fv.y_fixed_1) {
        int key = kv.first;
        int j = kv.second;
        int i = key / (n * n);
        int rem = key % (n * n);
        int u = rem / n;
        int v = rem % n;
        int idx = n * n + i * n * n * n + u * n * n + j * n + v;
        double val = psol[idx];
        double dev = std::abs(val - 1.0);
        if (dev > tol) {
            ++rep.y1_violations;
            rep.y1_max_dev = std::max(rep.y1_max_dev, dev);
        }
    }

    // y fixed to 0
    for (const auto &kv : fv.y_fixed_0) {
        int key = kv.first;
        int i = key / (n * n);
        int rem = key % (n * n);
        int u = rem / n;
        int v = rem % n;
        for (int j : kv.second) {
            int idx = n * n + i * n * n * n + u * n * n + j * n + v;
            double val = psol[idx];
            double dev = std::abs(val);
            if (dev > tol) {
                ++rep.y0_violations;
                rep.y0_max_dev = std::max(rep.y0_max_dev, dev);
            }
        }
    }

    return rep;
}

// Macros like your existing formulation-2/3
#define x(i,u) (psol[(i)*n + (u)])
#define v_x(i,u) (var_x[(i)*n + (u)])
#define mu1(u) (pi[(u)])
#define mu2(i) (pi[n + (i)])
#define vio_mu1(u) (vio[(u)])
#define vio_mu2(i) (vio[n + (i)])

// Q'_{(j,v),(i,u)} = 0.5*(D[j][i]*F[v][u] + D[i][j]*F[u][v])
static inline double Qsym_entry(const Problem& P, int n,
                                int j, int v, int i, int u) {
    return 0.5 * ( P.D[j][i]*P.F[v][u] + P.D[i][j]*P.F[u][v] );
}

// Diagonal Q'_{(i,u),(i,u)} = D[i][i]*F[u][u]
static inline double Qsym_diag(const Problem& P, int i, int u) {
    return P.D[i][i]*P.F[u][u];
}

QUBOVolumeHooks::QUBOVolumeHooks(const Problem& data)
: qap_data(data), n(data.n)
{
    xbin.resize(n*n, 0);
    s.resize(n*n, 0.0);
    c.resize(n*n, 0.0);
    locked.resize(n*n, 0);
}

bool QUBOVolumeHooks::set_fixed_variables(const FixedVariables& fv) {
    fixed = fv;
    // Prepare lock array for quick checks
    std::fill(locked.begin(), locked.end(), 0);
    for (const auto &kv : fixed.x_fixed_1) {
        int i = kv.first, u = kv.second;
        locked[idx(i,u)] = 1; // locked to 1
    }
    for (const auto &kv : fixed.x_fixed_0) {
        int i = kv.first;
        for (int u : kv.second) locked[idx(i,u)] = -1; // locked to 0
    }
    return true;
}

// In qubo_volume_solver.cpp (inside QUBOVolumeHooks)

double QUBOVolumeHooks::compute_primal_qap_cost(const VOL_dvector& psol) const {
    // pcost = sum_{i,j,u,v} D[i][j] * F[u][v] * x_{i,u} * x_{j,v}
    // Compute U = X F, then B = U X^T, pcost = sum_{i,j} D[i][j] * B[i][j]
    const int N = n * n;
    std::vector<double> U(N, 0.0), B(N, 0.0);

    // U(i, v) = sum_u X(i,u) * F(u,v)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int v = 0; v < n; ++v) {
            double sum = 0.0;
            for (int u = 0; u < n; ++u) {
                sum += x(i,u) * qap_data.F[u][v];
            }
            U[i*n + v] = sum;
        }
    }

    // B(i, j) = sum_v U(i,v) * X(j,v)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int v = 0; v < n; ++v) {
                sum += U[i*n + v] * x(j,v);
            }
            B[i*n + j] = sum;
        }
    }

    double pcost = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:pcost) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            pcost += qap_data.D[i][j] * B[i*n + j];
        }
    }
    return pcost;
}

void QUBOVolumeHooks::fill_violations(const VOL_dvector& pi,
                                                     const VOL_dvector& psol,
                                                     VOL_dvector& vio) const
{
    vio = 0.0;

    // Column constraints: for each u, sum_i x(i,u) = 1
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += x(i,u);
        vio_mu1(u) = 1.0 - sum;              // vio_mu1(u)
    }

    // Row constraints: for each i, sum_u x(i,u) = 1
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int u = 0; u < n; ++u) sum += x(i,u);
        vio_mu2(i) = 1.0 - sum;          // vio_mu2(i)
    }
}

double QUBOVolumeHooks::qubo_solve_with_cplex(const VOL_dvector& pi,
                                              VOL_dvector& psol)
{
    IloEnv env;
    double bestVal = 0.0;

    try {
        int N = n * n;
        IloModel model(env);
        env.setOut(std::cout); // suppress CPLEX output

        // Decision variables
        IloBoolVarArray var_x(env, N);
        // Build objective: x'Q'x - c'x
        IloExpr obj(env);

        // Linear term: -c[k] * x[k]
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                double ck = mu1(u) + mu2(i);
                obj += -ck * v_x(i,u);
            }
        }

        // Quadratic term
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < n; ++v) {
                        if (i > j || (i == j && u >= v)) continue; // upper triangle only
                        double Qkl = Qsym_entry(qap_data, n, i, u, j, v);
                        if (fabs(Qkl) > 1e-14)
                            obj += Qkl * v_x(i,u) * v_x(j,v);
                    }
                }
            }
        }

        // Fixed variables
        for (auto& kv : fixed.x_fixed_1) {
            model.add(v_x(kv.first, kv.second) == 1);
        }
        for (auto& kv : fixed.x_fixed_0) {
            int i = kv.first;
            for (int u : kv.second)
                model.add(v_x(i,u) == 0);
        }

        // Set objective
        model.add(IloMinimize(env, obj));
        obj.end();

        // Solve QUBO
        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());

        bool ok = cplex.solve();
        if (!ok) {
            std::cerr << "CPLEX failed to solve QUBO." << std::endl;
            env.end();
            return 0.0;
        }
        psol = 0.0; // initialize psol
        // Extract exact binary solution
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                double val = cplex.getValue(v_x(i,u));
                if (std::isnan(val)) val = 0.0; // safety check
                x(i,u) = val;
            }
        }
        // objective value is the QUBO objective + constant terms from duals
        bestVal = cplex.getObjValue();
        for (int u = 0; u < n; ++u) bestVal += mu1(u); // sum_u mu1(u)
        for (int i = 0; i < n; ++i) bestVal += mu2(i); // sum_i mu2(i)
        
    }
    catch (IloException& e) {
        std::cerr << "CPLEX exception: " << e.getMessage() << std::endl;
    }

    env.end();
    return bestVal;
}


int QUBOVolumeHooks::solve_subproblem(const VOL_dvector& pi,
                                         const VOL_dvector& /*rc*/,
                                         double& lcost,
                                         VOL_dvector& psol,
                                         VOL_dvector& vio,
                                         double& pcost)
{
    psol = 0.0; // initialize psol
    lcost = qubo_solve_with_cplex(pi, psol);
    pcost = compute_primal_qap_cost(psol);
    vio = 0.0;
    fill_violations(pi, psol, vio);
    std::cout << "VOL subproblem solved: lcost=" << lcost << ", pcost=" << pcost
              << ", max row vio=" << vio_mu2(0) << ", max col vio=" << vio_mu1(0) << std::endl;
    return 0;
}


// ---- VOL problem setup + driver ----

void set_up_volume_parameters(VOL_problem &vol_problem, bool verbose) {
    // Set Volume algorithm parameters (from qap.par defaults)
    vol_problem.parm.lambdainit = 0.01;
    vol_problem.parm.alphainit = 0.01;
    vol_problem.parm.alphamin = 0.0001;
    vol_problem.parm.alphafactor = 0.66;
    vol_problem.parm.alphaint = 50;
    
    vol_problem.parm.maxsgriters = 1000; // effectively no limit
    vol_problem.parm.primal_abs_precision = 0.001;
    vol_problem.parm.gap_abs_precision = 0.0;
    vol_problem.parm.gap_rel_precision = 0.001;
    vol_problem.parm.granularity = 0.0;
    
    vol_problem.parm.ascent_first_check = 500;
    vol_problem.parm.ascent_check_invl = 500;
    vol_problem.parm.minimum_rel_ascent = 0.0001;
    
    vol_problem.parm.greentestinvl = 1;
    vol_problem.parm.yellowtestinvl = 4;
    vol_problem.parm.redtestinvl = 20;
    
    // Printing control
    vol_problem.parm.printflag = verbose ? 3 : 0;  // 1=iteration info, 3=add lambda info
    vol_problem.parm.printinvl = 100;
    vol_problem.parm.heurinvl = 100;
}

void set_up_vol_problem_qubo(VOL_problem& vol_problem, const Problem& qap_data) {
    int n = qap_data.n;
    // primal size: x(i,u) only
    vol_problem.psize = n * n;

    // dual size: mu1[u] (n) + mu2[i] (n)
    vol_problem.dsize = 2 * n;

    // init duals
    vol_problem.dsol.allocate(vol_problem.dsize);
    vol_problem.dsol = 0.0;


    vol_problem.psol.allocate(vol_problem.psize);
    vol_problem.psol = 0.0;

    vol_problem.viol.allocate(vol_problem.dsize);
    vol_problem.viol = 0.0;

}

VolumeResult solve_qap_qubo_volume_relax(const Problem &problem,
                                         const FixedVariables &fixed,
                                         bool verbose,
                                         const VOL_dvector &initial_dual)
{
    VOL_problem vol_problem;
    set_up_vol_problem_qubo(vol_problem, problem);

    // Create hooks
    QUBOVolumeHooks hooks(problem);
    hooks.set_fixed_variables(fixed);

    // Warm start duals if provided
    bool use_dual_warmstart = false;
    if (initial_dual.size() > 0) {
        if (initial_dual.size() != vol_problem.dsize) {
            std::cerr << "\nError: Initial dual vector size mismatch for QUBO formulation." << std::flush;
        } else {
            vol_problem.dsol = initial_dual;
            use_dual_warmstart = true;
        }
    }

    // Reuse your parameter setup
    set_up_volume_parameters(vol_problem, verbose);

    // Solve
    int retval = vol_problem.solve(hooks, use_dual_warmstart);

    VolumeResult result;
    result.status = retval;
    result.lower_bound = vol_problem.value;
    result.dual = vol_problem.dsol;
    // Optional: you can also return the last primal x as a vector<double>
    // result.primal = ... (allocate and copy from last psol if VOL provides it)

    return result;
}