#include "IncrementalRMP.h"
#include <algorithm>
#include <iomanip>

using std::get;

// ------- Utilities -------
std::string IncrementalRMP::yName(const QuadKey& q) 
{
    std::ostringstream oss;
    oss << "y_" << q.i << "_" << q.u << "_" << q.j << "_" << q.v;
    return oss.str();
}

std::string IncrementalRMP::xName(int i, int u) 
{
    std::ostringstream oss;
    oss << "x_" << i << "_" << u;
    return oss.str();
}

GRBEnv&   IncrementalRMP::env()   { return _env; }
GRBModel& IncrementalRMP::model() { return _model; }

IncrementalRMP::IncrementalRMP(const std::vector<double>& phi_,
                               int n_, int m_, double eps_, bool quiet_, bool disable_presolve_)
                                : _env()
                                , _model(_env)
                                , phi(phi_)
                                , n(n_), m(m_)
                                , quiet(quiet_)
                                , c1_map(n_*m_)
                                , c2_map(m_*m_)
                                , c3_map(n_*m_*n_)
                                , c4_map(n_*m_*m_)
                                , x_index(n_*m_)
                                , is_in_Omega(n_*m_*n_*m_, false)
                                , y_index(n_*m_*n_*m_)
{
    _env.start();
    // set method to dual simplex, disable presolve, disable dual reductions, disable output
    // _model.set(GRB_IntParam_Method, 1); // dual simplex
    // _model.set(GRB_IntParam_Method, 0); // primal simplex
    _model.set(GRB_IntParam_Method, 2); // barriers
    // _model.set(GRB_IntParam_OutputFlag, quiet ? 0 : 1);
    // _model.set(GRB_IntParam_Presolve, disable_presolve_ ? 0 : 1);
    _model.set(GRB_IntParam_DualReductions, 0);
    _model.set(GRB_IntParam_OutputFlag, 1);
    // if (quiet) {
    //     _model.set(GRB_IntParam_OutputFlag, 0);
    // }
    // set minimization objective
    _model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    pi_in.resize(n, m);

}

void IncrementalRMP::createModel() 
{
    // Create x variables and C1–C4 rows (with no y-columns)
    x_index.resize(n*m);
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < m; u++) {
            // create x variable
            x_index[i*m + u] = _model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, xName(i,u));
        }
    }
    // C1(i):  sum_{u} x_{i,u} <= 1   (i in V)         (L)
    c1_map.resize(n);
    for (int i = 0; i < n; i++) {
        const char* name = ("C1_" + std::to_string(i)).c_str();
        GRBLinExpr expr = 0.0;
        for (int u = 0; u < m; u++) {
            expr += x_index[i*m + u];
        }
        c1_map[i] = _model.addConstr(expr <= 1.0, name);
    }
    // C2(u):  sum_{i} x_{i,u} == 1   (u in M)         (E)
    c2_map.resize(m);
    for (int u = 0; u < m; u++) {
        const char* name = ("C2_" + std::to_string(u)).c_str();
        GRBLinExpr expr = 0.0;
        for (int i = 0; i < n; i++) {
            expr += x_index[i*m + u];
        }
        c2_map[u] = _model.addConstr(expr == 1.0, name);
    }
    // C3(i,u,j): sum_v y_{i,u,j,v} - x_{i,u} <= 0   (i,u in VxM, j in V) (L)
    // c3_map.resize(n*m*n);
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < m; u++) {
            for (int j = 0; j < n; j++) {
                if(i==j) continue; // no y-columns for i==j
                const char* name = ("C3_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j)).c_str();
                GRBLinExpr expr = 0.0;
                expr -= x_index[i*m + u];
                c3_map[(i * m + u) * n + j] = _model.addConstr(expr <= 0.0, name);
            }
        }
    }
    // C4(i,u,v): sum_j y_{i,u,j,v} - x_{i,u}  == 0   (i,u in VxM, v in M) (E)
    // c4_map.resize(n*m*m);
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < m; u++) {
            for (int v = 0; v < m; v++) {
                if (u == v) continue; // no y-columns for u==v
                const char* name = ("C4_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v)).c_str();
                GRBLinExpr expr = 0.0;
                expr -= x_index[i*m + u];
                c4_map[(i * m + u) * m + v] = _model.addConstr(expr == 0.0, name);
            }
        }
    }
}

void IncrementalRMP::fixXAssignments(const std::unordered_map<int,int>& fixed_loc_to_fac)
{
    for (int i=0;i<n;++i) {
        for (int u = 0; u < m; u++) {
            auto x = x_index[i*m + u];
            auto it = fixed_loc_to_fac.find(i);
            if (it != fixed_loc_to_fac.end()) {
                int ufix = it->second;
                if (u == ufix) {
                    std::cout << "[RMP] Fixing x[" << i << "," << u << "] = 1\n";
                    x.set(GRB_DoubleAttr_LB, 1.0);
                    x.set(GRB_DoubleAttr_UB, 1.0);
                    
                } else {
                    x.set(GRB_DoubleAttr_LB, 0.0);
                    x.set(GRB_DoubleAttr_UB, 0.0);
                }
            }
        }
    }
}

bool IncrementalRMP::addColumn(const QuadKey& q) {
    return addColumn(q.i, q.u, q.j, q.v);
}

bool IncrementalRMP::addColumn(int i, int u, int j, int v) 
{
    if(i==j || u==v) {
        std::cerr << "Error: cannot add column for i==j or u==v, i=" << i << ", j=" << j << ", u=" << u << ", v=" << v << "\n";
        throw std::invalid_argument("Invalid column indices");
    }
    QuadKey can = canonical(i,u,j,v);
    auto index = can.to_index(n,m);
    if (is_in_Omega[index]) {
        // std::cerr << "Error: column already exists in Omega\n";
        // throw std::invalid_argument("Column already exists");
        return false; // return existing variable instead of throwing
    }

    auto ci = can.i, cu = can.u, cj = can.j, cv = can.v;

    // Objective coefficient
    double cost = (ci != cj)
        ? phiIUJV(ci,cu,cj,cv) + phiIUJV(cj,cv,ci,cu)
        : phiIUJV(ci,cu,cj,cv);

    
    // Create variable with column (column constructor: col, lb, ub, type, name)
    auto y = _model.addVar(0.0, 1.0, cost, GRB_CONTINUOUS, yName(can));
    y_index[index] = y;

    //  add y to constraints C3 and C4
    auto c3_i = c3_map[(ci * m + cu) * n + cj];
    auto c3_j = c3_map[(cj * m + cv) * n + ci];
    auto c4_i = c4_map[(ci * m + cu) * m + cv];
    auto c4_j = c4_map[(cj * m + cv) * m + cu];
    _model.chgCoeff(c3_i, y, 1.0);
    _model.chgCoeff(c3_j, y, 1.0);
    _model.chgCoeff(c4_i, y, 1.0);
    _model.chgCoeff(c4_j, y, 1.0);
    Omega.push_back(can);
    is_in_Omega[index] = true;
    // std::cout << "Added column " << yName(can) << " with cost " << cost << "\n";
    return true;
}

bool IncrementalRMP::pricingWithInOut(double alpha, std::vector<PricingColumn>& new_columns, PricingColumn& mostnegative_column, double& mostnegative_reduced_cost) 
{
    pi_star = pi_iter;
    pi_star.interpolate(pi_in, alpha);
    return pricing(pi_star, new_columns, mostnegative_column, mostnegative_reduced_cost);
}

bool IncrementalRMP::pricingWithCurrentDuals(std::vector<PricingColumn>& new_columns, PricingColumn& mostnegative_column, double& mostnegative_reduced_cost) 
{
    return pricing(pi_iter, new_columns, mostnegative_column, mostnegative_reduced_cost);
}

bool IncrementalRMP::pricing(DualVector& duals, std::vector<PricingColumn>& new_columns, PricingColumn& mostnegative_column, double& mostnegative_reduced_cost) 
{
    new_columns.clear();
    mostnegative_reduced_cost = 0.0;
    for (auto i = 0; i < n; ++i)
    for (auto u = 0; u < m; ++u) 
    for (auto v = 0; v < m; ++v)
    for (auto j = i + 1; j < n; ++j) {
        if (u == v ) continue;
        QuadKey can = canonical(i,u,j,v);
        auto index = can.to_index(n,m);
        // if (is_in_Omega[index]) continue; // already in Omega
        double reduced_cost = 
            phiIUJV(i,u,j,v) + phiIUJV(j,v,i,u)
            - duals.c3[(i * m + u) * n + j] - duals.c3[(j * m + v) * n + i]
            - duals.c4[(i * m + u) * m + v] - duals.c4[(j * m + v) * m + u];
        if (reduced_cost < -1) {
            new_columns.push_back({can, reduced_cost});
            if (reduced_cost < mostnegative_reduced_cost) {
                mostnegative_reduced_cost = reduced_cost;
                mostnegative_column = {can, reduced_cost};
            }
        }
        // if(is_in_Omega[index]) {
        //     auto rc = y_index[index].get(GRB_DoubleAttr_RC);
        //     // reduced_cost += rc;
        //     if(reduced_cost < -1e-6) {
        //         std::cout << "[RMP] Warning: column " << yName(can) << " already in Omega but has negative reduced cost " << reduced_cost << "\n";
        //     }
        //     if(abs(reduced_cost - rc) > 1e-6) {
        //         std::cout << "[RMP] Warning: column " << yName(can) << " already in Omega but has reduced cost " << rc << " from Gurobi and " << reduced_cost << " from calculation, difference = " << (reduced_cost - rc) << "\n";
        //     }
        //     // if(rc > 1e-6) {
        //     //     std::cout << "[RMP] Warning: column " << yName(can) << " already in Omega but has non-zero reduced cost " << rc << "\n";
        //     // }
        // }
    }
    return !new_columns.empty();
}

int IncrementalRMP::addColumns(const std::vector<QuadKey>& columns) 
{
    int count = 0;
    for (const auto& col : columns) {
        auto c =addColumn(col.i, col.u, col.j, col.v);
        if (c) count++;
    }
    return count;
}

int IncrementalRMP::addColumns(const std::vector<PricingColumn>& columns) 
{
    int count = 0;
    for (const auto& col : columns) {
        auto c = addColumn(col.column.i, col.column.u, col.column.j, col.column.v);
        if (c) count++;
    }
    return count;
}

bool IncrementalRMP::swapInDualswPiStar() {
    pi_in.swap(pi_star);
    double diff = 0.0;
    // check if pi_in and pi_star are the same, if so, do nothing
    for (size_t i=0;i<pi_in.c3.size();++i) {
        diff += abs(pi_in.c3[i] - pi_star.c3[i]);
    }
    for (size_t i=0;i<pi_in.c4.size();++i) {
        diff += abs(pi_in.c4[i] - pi_star.c4[i]);
    }
    if (diff < 1e-6) {
        std::cout << "[RMP] swapInDualswPiStar: pi_in and pi_star are the same, no swap performed.\n";
        return false;
    } else {
        std::cout << "[RMP] swapInDualswPiStar: pi_in and pi_star swapped, diff = " << diff << ".\n";
        return true;
    }
    return true;
}

bool IncrementalRMP::solve() 
{
    // _cplex.setOut(std::cout);
    // export model to file for debugging
    // _cplex.exportModel("rmp.lp");
    _model.optimize();
    
    // if current method is barrier, switch to dual simplex for next solve
    if (_model.get(GRB_IntParam_Method) == 2) {
        _model.set(GRB_IntParam_Method, 0);
        _model.optimize();
    }
    return _model.get(GRB_IntAttr_Status) == GRB_OPTIMAL || _model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL;
}

double IncrementalRMP::getObjectiveValue() const {
    return _model.get(GRB_DoubleAttr_ObjVal);
}

double IncrementalRMP::getDual(const GRBConstr& rng) const {
    return rng.get(GRB_DoubleAttr_Pi);
}

double IncrementalRMP::getRowDualC1(int i) const { return getDual(c1_map[i]); }
double IncrementalRMP::getRowDualC2(int u) const { return getDual(c2_map[u]); }
double IncrementalRMP::getRowDualC3(int i, int u, int j) const { return getDual(c3_map[(i * m + u) * n + j]); }
double IncrementalRMP::getRowDualC4(int i, int u, int v) const { return getDual(c4_map[(i * m + u) * m + v]); }

bool IncrementalRMP::hasColumn(const QuadKey& q) const {
    return hasColumn(q.i, q.u, q.j, q.v);
}

bool IncrementalRMP::hasColumn(int i, int u, int j, int v) const {
    return is_in_Omega[(((i * m) + u) * n + j) * m + v];
}

void IncrementalRMP::columnsAnalysis() {
    // IloCplex::BasisStatusArray varBasis(_env);
    // IloNumVarArray vars(_env);
    // // get status of var y
    // for (auto& kv : y_index) {
    //     vars.add(kv.second);
    // }
    // _cplex.getBasisStatuses(varBasis, vars);
    // auto count_basis = 0;
    // auto count_positive = 0;
    // for (int i = 0; i < vars.getSize(); i++) {
    //     switch (varBasis[i]) {
    //         case IloCplex::Basic:
    //             count_basis++;
    //             break;
    //         default:
    //             break;
    //     }
    //     double val = _cplex.getValue(vars[i]);
    //     if (val > 1e-5) {
    //         count_positive++;
    //     }
    // }
    // std::cout << "Total columns: " << vars.getSize() << ", in basis: " << count_basis << ", positive: " << count_positive << "\n";
}

void IncrementalRMP::retrieveCurrentDuals() {
    pi_iter.c3.resize(n*m*n);
    pi_iter.c4.resize(n*m*m);
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < m; u++) {
            for (int j = 0; j < n; j++) {
                if(i==j) continue;
                pi_iter.c3[(i * m + u) * n + j] = getRowDualC3(i,u,j);
            }
            for (int v = 0; v < m; v++) {
                if(u==v) continue;
                pi_iter.c4[(i * m + u) * m + v] = getRowDualC4(i,u,v);
            }
        }
    }
}
