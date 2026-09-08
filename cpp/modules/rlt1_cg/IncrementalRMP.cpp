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

IloEnv&   IncrementalRMP::env()   { return _env; }
IloModel& IncrementalRMP::model() { return _model; }
IloCplex& IncrementalRMP::cplex() { return _cplex; }

IncrementalRMP::IncrementalRMP(const std::vector<double>& phi_,
                               int n_, int m_, double eps_, bool quiet_, bool disable_presolve_)
                                : _env()
                                , _model(_env)
                                , _cplex(_model)
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
    // set method to dual simplex, disable presolve, disable dual reductions, disable output
    _cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Dual);
    _cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false);
    _cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
    if (quiet) {
        _cplex.setOut(_env.getNullStream());
        _cplex.setWarning(_env.getNullStream());
        _cplex.setError(_env.getNullStream());
        // _cplex.setOut(std::cout);

    }
    _obj = IloMinimize(_env, 0.0);
    _model.add(_obj);
    pi_in.resize(n, m);

}

void IncrementalRMP::createModel() 
{
    // Create x variables and C1–C4 rows (with no y-columns)
    x_index.resize(n*m);
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < m; u++) {
            // create x variable
            auto x = IloNumVar(_env, 0.0, 1.0, ILOFLOAT, xName(i,u).c_str());
            _model.add(x);
            x_index[i*m + u] = x;
        }
    }
    // C1(i):  sum_{u} x_{i,u} <= 1   (i in V)         (L)
    c1_map.resize(n);
    for (int i = 0; i < n; i++) {
        const char* name = ("C1_" + std::to_string(i)).c_str();
        IloRange rng(_env, -IloInfinity, 1.0, name);
        _model.add(rng);
        for (int u = 0; u < m; u++) {
            rng.setLinearCoef(x_index[i*m + u], 1.0);
        }
        c1_map[i] = rng;
    }
    // C2(u):  sum_{i} x_{i,u} == 1   (u in M)         (E)
    c2_map.resize(m);
    for (int u = 0; u < m; u++) {
        const char* name = ("C2_" + std::to_string(u)).c_str();
        IloRange rng(_env, 1.0, 1.0, name);
        _model.add(rng);
        for (int i = 0; i < n; i++) {
            rng.setLinearCoef(x_index[i*m + u], 1.0);
        }
        c2_map[u] = rng;
    }
    // C3(i,u,j): sum_v y_{i,u,j,v} - x_{i,u} <= 0   (i,u in VxM, j in V) (L)
    // c3_map.resize(n*m*n);
    for (int i = 0; i < n; i++) {
        for (int u = 0; u < m; u++) {
            for (int j = 0; j < n; j++) {
                if(i==j) continue; // no y-columns for i==j
                const char* name = ("C3_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j)).c_str();
                IloRange rng(_env, -IloInfinity, 0.0, name);
                _model.add(rng);
                rng.setLinearCoef(x_index[i*m + u], -1.0);
                c3_map[(i * m + u) * n + j] = rng;
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
                IloRange rng(_env, 0.0, 0.0, name);
                _model.add(rng);
                rng.setLinearCoef(x_index[i*m + u], -1.0);
                c4_map[(i * m + u) * m + v] = rng;
            }
        }
    }
}

void IncrementalRMP::fixXAssignments(const std::unordered_map<int,int>& fixed_loc_to_fac)
{
    for (int i=0;i<n;++i) {
        for (int u = 0; u < m; u++) {
            IloNumVar x = x_index[i*m + u];
            auto it = fixed_loc_to_fac.find(i);
            if (it != fixed_loc_to_fac.end()) {
                int ufix = it->second;
                if (u == ufix) {
                    std::cout << "[RMP] Fixing x[" << i << "," << u << "] = 1\n";
                    x.setBounds(1.0, 1.0);
                } else {
                    x.setBounds(0.0, 0.0);
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

    // Build column with all coefficients at once (more efficient than adding variable then setting coefficients)
    auto obj = _cplex.getObjective();
    IloNumColumn col = obj(cost);
    col += c3_map[(ci * m + cu) * n + cj](1.0);
    col += c4_map[(ci * m + cu) * m + cv](1.0);
    
    col += c3_map[(cj * m + cv) * n + ci](1.0);
    col += c4_map[(cj * m + cv) * m + cu](1.0);
    
    // Create variable with column (column constructor: col, lb, ub, type, name)
    IloNumVar y(col, 0.0, 1.0, ILOFLOAT, yName(can).c_str());
    _model.add(y);
    col.end();

    y_index[index] = y;
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
        if (is_in_Omega[index]) continue; // already in Omega
        double reduced_cost = 
            phiIUJV(i,u,j,v) + phiIUJV(j,v,i,u)
            - duals.c3[(i * m + u) * n + j] - duals.c3[(j * m + v) * n + i]
            - duals.c4[(i * m + u) * m + v] - duals.c4[(j * m + v) * m + u];
        if (reduced_cost < -1e-6) {
            new_columns.push_back({can, reduced_cost});
            if (reduced_cost < mostnegative_reduced_cost) {
                mostnegative_reduced_cost = reduced_cost;
                mostnegative_column = {can, reduced_cost};
            }
        }
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
    return _cplex.solve();
}

double IncrementalRMP::getObjectiveValue() const {
    return _cplex.getObjValue();
}

double IncrementalRMP::getDual(const IloRange& rng) const {
    return _cplex.getDual(rng);
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
