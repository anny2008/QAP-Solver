#include "IncrementalRMP.h"
#include <algorithm>
#include <iomanip>

using std::get;

// ------- Utilities -------
std::string IncrementalRMP::yName(const QuadKey& q) {
    std::ostringstream oss;
    oss << "y_" << q.i << "_" << q.u << "_" << q.j << "_" << q.v;
    return oss.str();
}
std::string IncrementalRMP::xName(int i, int u) {
    std::ostringstream oss;
    oss << "x_" << i << "_" << u;
    return oss.str();
}

IloEnv&   IncrementalRMP::env()   { return _env; }
IloModel& IncrementalRMP::model() { return _model; }
IloCplex& IncrementalRMP::cplex() { return _cplex; }

IncrementalRMP::IncrementalRMP(const VList& V_, const MList& M_,
                               const std::vector<double>& phi_,
                               int n_, int m_, double eps_, bool quiet_, bool disable_presolve_)
                                : _env()
                                , _model(_env)
                                , _cplex(_model)
                                , V(V_.begin(), V_.end())
                                , M(M_.begin(), M_.end())
                                , phi(phi_)
                                , n(n_), m(m_)
                                , eps(eps_), quiet(quiet_), disable_presolve(disable_presolve_)
                                , lastIterationAddedColumn{-1,-1,-1,-1}
                                , countAddedColumnsInBasis(0)
{
    _cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Dual);
    if (disable_presolve) {
        _cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false);
    }
    if (quiet) {
        // _cplex.setOut(_env.getNullStream());
        // _cplex.setWarning(_env.getNullStream());
        // _cplex.setError(_env.getNullStream());
        _cplex.setOut(std::cout);

    }
    _obj = IloMinimize(_env, 0.0);
    _model.add(_obj);

}

IloNumVar IncrementalRMP::ensureX(int i, int u) {
    PairKey key{i,u};
    auto it = x_index.find(key);
    if (it != x_index.end()) return it->second;

    IloNumVar x(_env, 0.0, 1.0, ILOFLOAT, xName(i,u).c_str());
    _model.add(x);
    x_index.emplace(key, x);
    return x;
}
template <class MapT, class KeyT>
static IloRange& ensureRowImpl(IloEnv& env, IloModel& model,
                               MapT& mp, const KeyT& key,
                               char sense, double rhs)
{
    // already exists
    auto it = mp.find(key);
    if (it != mp.end())
        return it->second;

    // create proper range inside env
    IloRange rng(env, -IloInfinity, rhs);

    if (sense == 'L')      rng.setUB(rhs);
    else if (sense == 'E') rng.setBounds(rhs, rhs);
    else if (sense == 'G') rng.setLB(rhs);

    model.add(rng);

    // insert and return stored handle
    auto inserted = mp.emplace(key, rng);
    return inserted.first->second;
}

IloRange& IncrementalRMP::ensureC1(int i, int j) {
    return ensureRowImpl(_env, _model, c1_map, PairKey{i,j}, 'L', 1.0);
}
IloRange& IncrementalRMP::ensureC2(int u, int v) {
    return ensureRowImpl(_env, _model, c2_map, PairKey{u,v}, 'E', 1.0);
}
IloRange& IncrementalRMP::ensureC3(int u, int j) {
    return ensureRowImpl(_env, _model, c3_map, PairKey{u,j}, 'L', 1.0);
}
IloRange& IncrementalRMP::ensureC4(int i, int v) {
    return ensureRowImpl(_env, _model, c4_map, PairKey{i,v}, 'L', 1.0);
}
IloRange& IncrementalRMP::ensureC5(int i, int u, int j) // sum_v y(i,u,j,v) <= x(i,u)
{
    TripleKey key{i,u,j};
    auto it = c5_map.find(key);
    if (it != c5_map.end()) return it->second;
    IloRange rng(_env, -IloInfinity, 0.0);
    rng.setUB(0.0);
    _model.add(rng);
    auto inserted = c5_map.emplace(key, rng);
    inserted.first->second.setLinearCoef(ensureX(i,u), -1.0);
    return inserted.first->second;

}
IloRange& IncrementalRMP::ensureC6(int i, int u, int v) // sum_j y(i,u,j,v) <= x(i,u)
{
    TripleKey key{i,u,v};
    auto it = c6_map.find(key);
    if (it != c6_map.end()) return it->second;
    IloRange rng(_env, -IloInfinity, 0.0);
    rng.setBounds(0.0, 0.0);
    _model.add(rng);
    auto inserted = c6_map.emplace(key, rng);
    inserted.first->second.setLinearCoef(ensureX(i,u), -1.0);

    return inserted.first->second;
}
void IncrementalRMP::ensureAllRows() {
    for (int i : V) for (int j : V) ensureC1(i,j);
    for (int u : M) for (int v : M) ensureC2(u,v);
    for (int u : M) for (int j : V) ensureC3(u,j);
    for (int i : V) for (int v : M) ensureC4(i,v);
    for (int i : V) for (int u : M) for (int j : V) ensureC5(i,u,j);
    for (int i : V) for (int u : M) for (int v : M) ensureC6(i,u,v);
}

void IncrementalRMP::fixXAssignments(const std::unordered_map<int,int>& fixed_loc_to_fac) {
    for (int i : V) {
        for (int u : M) {
            IloNumVar x = ensureX(i,u);
            auto it = fixed_loc_to_fac.find(i);
            if (it != fixed_loc_to_fac.end()) {
                int ufix = it->second;
                if (u == ufix) {
                    x.setBounds(1.0, 1.0);
                } else {
                    x.setBounds(0.0, 0.0);
                }
            }
        }
    }
}

IloNumVar IncrementalRMP::addColumn(int i, int u, int j, int v) {
    if (i==j && u!=v) {
        std::cerr << "Error: cannot add column for (i,u,j,v) = (" << i << "," << u << "," << j << "," << v << ") because i==j but u!=v\n";
        throw std::runtime_error("Invalid column");
    }
    if (i!=j && u==v) {
        std::cerr << "Error: cannot add column for (i,u,j,v) = (" << i << "," << u << "," << j << "," << v << ") because i!=j but u==v\n";
        throw std::runtime_error("Invalid column");
    }
    QuadKey can = canonical(i,u,j,v);
    if (y_index.count(can)) return y_index[can];

    auto ci = can.i, cu = can.u, cj = can.j, cv = can.v;

    // Objective coefficient
    double cost = (ci != cj)
        ? phiIUJV(ci,cu,cj,cv) + phiIUJV(cj,cv,ci,cu)
        : phiIUJV(ci,cu,cj,cv);

    // Build column with all coefficients at once (more efficient than adding variable then setting coefficients)
    auto obj = _cplex.getObjective();
    IloNumColumn col = obj(cost);
    col += ensureC1(ci,cj)(1.0);
    col += ensureC2(cu,cv)(1.0);
    col += ensureC3(cu,cj)(1.0);
    col += ensureC4(ci,cv)(1.0);
    col += ensureC5(ci,cu,cj)(1.0);
    col += ensureC6(ci,cu,cv)(1.0);
    
    col += ensureC1(cj,ci)(1.0);
    col += ensureC2(cv,cu)(1.0);
    col += ensureC3(cv,ci)(1.0);
    col += ensureC4(cj,cu)(1.0);
    col += ensureC5(cj,cv,ci)(1.0);
    col += ensureC6(cj,cv,cu)(1.0);
    
    // Create variable with column (column constructor: col, lb, ub, type, name)
    IloNumVar y(col, 0.0, IloInfinity, ILOFLOAT);
    _model.add(y);
    col.end();

    y_index.emplace(can, y);
    Omega.insert(can);
    // lastIterationAddedColumn = can; // for debugging
    return y;
}

void IncrementalRMP::addColumns(const std::vector<QuadKey>& columns) {
    if (columns.empty()) return;
    
    // Collect only new columns that need to be added
    std::vector<QuadKey> new_columns;
    new_columns.reserve(columns.size());
    for (const auto& col : columns) {
        QuadKey can = canonical(col.i, col.u, col.j, col.v);
        if (!y_index.count(can)) {
            new_columns.push_back(can);
        }
    }
    
    if (new_columns.empty()) return;
    
    // Use column-oriented construction for better performance
    auto obj = _cplex.getObjective();
    
    for (const auto& can : new_columns) {
        auto ci = can.i, cu = can.u, cj = can.j, cv = can.v;
        
        // Objective coefficient
        double cost = (ci != cj)
            ? phiIUJV(ci,cu,cj,cv) + phiIUJV(cj,cv,ci,cu)
            : phiIUJV(ci,cu,cj,cv);
        
        // Build column with all coefficients at once
        IloNumColumn col = obj(cost);
        col += ensureC1(ci,cj)(1.0);
        col += ensureC2(cu,cv)(1.0);
        col += ensureC3(cu,cj)(1.0);
        col += ensureC4(ci,cv)(1.0);
        col += ensureC5(ci,cu,cj)(1.0);
        col += ensureC6(ci,cu,cv)(1.0);
        
        col += ensureC1(cj,ci)(1.0);
        col += ensureC2(cv,cu)(1.0);
        col += ensureC3(cv,ci)(1.0);
        col += ensureC4(cj,cu)(1.0);
        col += ensureC5(cj,cv,ci)(1.0);
        col += ensureC6(cj,cv,cu)(1.0);
        
        // Create variable with column (no string name for performance)
        IloNumVar y(col, 0.0, IloInfinity, ILOFLOAT);
        _model.add(y);
        col.end();
        
        y_index.emplace(can, y);
        Omega.insert(can);
        // lastIterationAddedColumn = can; // for debugging
    }
}

bool IncrementalRMP::solve() {
    _cplex.setOut(quiet ? _env.getNullStream() : std::cout);
    return _cplex.solve();
}

double IncrementalRMP::getObjectiveValue() const {
    return _cplex.getObjValue();
}

double IncrementalRMP::getDual(const IloRange& rng) const {
    return _cplex.getDual(rng);
}

double IncrementalRMP::getRowDualC1(int i, int j) const { return getDual(c1_map.at(PairKey{i,j})); }
double IncrementalRMP::getRowDualC2(int u, int v) const { return getDual(c2_map.at(PairKey{u,v})); }
double IncrementalRMP::getRowDualC3(int u, int j) const { return getDual(c3_map.at(PairKey{u,j})); }
double IncrementalRMP::getRowDualC4(int i, int v) const { return getDual(c4_map.at(PairKey{i,v})); }
double IncrementalRMP::getRowDualC5(int i, int u, int j) const { return getDual(c5_map.at(TripleKey{i,u,j})); }
double IncrementalRMP::getRowDualC6(int i, int u, int v) const { return getDual(c6_map.at(TripleKey{i,u,v})); }

std::unordered_map<QuadKey, double, QuadKeyHash> IncrementalRMP::getYValues() const {
    std::unordered_map<QuadKey,double,QuadKeyHash> out;
    for (auto& kv : y_index) {
        double val = _cplex.getValue(kv.second);
        if (val > eps) out.emplace(kv.first, val);
    }
    return out;
}

std::unordered_map<PairKey, double, PairKeyHash> IncrementalRMP::getXValues() const {
    std::unordered_map<PairKey,double,PairKeyHash> out;
    for (auto& kv : x_index) {
        out.emplace(kv.first, _cplex.getValue(kv.second));
    }
    return out;
}

bool IncrementalRMP::hasColumn(int i, int u, int j, int v) const {
    return Omega.count(canonical(i,u,j,v)) > 0;
}

// check all the rows to see if the columns are correctly attached and if there any column should not be but is attached  (for debugging)
bool IncrementalRMP::debugCheckColumnAttachments() const {
    for (const auto& can : Omega) {
        if (!y_index.count(can)) {
            std::cerr << "Error: column " << yName(can) << " in Omega but not in y_index\n";
            return false;
        }
        IloNumVar y = y_index.at(can);
        auto ci = can.i, cu = can.u, cj = can.j, cv = can.v;
        // check C1
        IloRange c1 = c1_map.at(PairKey{ci,cj});
        bool found_in_c1 = false;
        for (IloExpr::LinearIterator it = c1.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0 && var.getName() == y.getName()) {
                found_in_c1 = true;
                break;
            }
        }        
        if (!found_in_c1) {
            std::cerr << "Error: column " << yName(can) << " in Omega but not attached to C1(" << ci << "," << cj << ")\n";
            return false;
        }
        // check C2
        IloRange c2 = c2_map.at(PairKey{cu,cv});
        bool found_in_c2 = false;
        for (IloExpr::LinearIterator it = c2.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0 && var.getName() == y.getName()) {
                found_in_c2 = true;
                break;
            }
        }        
        if (!found_in_c2) {
            std::cerr << "Error: column " << yName(can) << " in Omega but not attached to C2(" << cu << "," << cv << ")\n";
            return false;
        }
        // check C3
        IloRange c3 = c3_map.at(PairKey{cu,cj});
        bool found_in_c3 = false;
        for (IloExpr::LinearIterator it = c3.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0 && var.getName() == y.getName()) {
                found_in_c3 = true;
                break;
            }
        }
        if (!found_in_c3) {
            std::cerr << "Error: column " << yName(can) << " in Omega but not attached to C3(" << cu << "," << cj << ")\n";
            return false;
        }
        // check C4
        IloRange c4 = c4_map.at(PairKey{ci,cv});
        bool found_in_c4 = false;
        for (IloExpr::LinearIterator it = c4.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0 && var.getName() == y.getName()) {
                found_in_c4 = true;
                break;
            }
        }
        if (!found_in_c4) {
            std::cerr << "Error: column " << yName(can) << " in Omega but not attached to C4(" << ci << "," << cv << ")\n";
            return false;
        }
    }
    for (const auto& kv : c1_map) {
        const auto& key = kv.first;
        const auto& rng = kv.second;
        for (IloExpr::LinearIterator it = rng.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0) {
                // check if var corresponds to a column in Omega that should be attached to this row
                bool found = false;
                for (const auto& can : Omega) {
                    if (yName(can) == var.getName()) {
                        // check if can should be attached to this row
                        if ((can.i == key.a && can.j == key.b) || (can.i == key.b && can.j == key.a)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    std::cerr << "Error: variable " << var.getName() << " with nonzero coef in C1(" << key.a << "," << key.b << ") but no corresponding column in Omega\n";
                    return false;
                }
            }
        }
    }
    for (const auto& kv : c2_map) {
        const auto& key = kv.first;
        const auto& rng = kv.second;
        for (IloExpr::LinearIterator it = rng.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0) {
                // check if var corresponds to a column in Omega that should be attached to this row
                bool found = false;
                for (const auto& can : Omega) {
                    if (yName(can) == var.getName()) {
                        // check if can should be attached to this row
                        if ((can.u == key.a && can.v == key.b) || (can.u == key.b && can.v == key.a)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    std::cerr << "Error: variable " << var.getName() << " with nonzero coef in C2(" << key.a << "," << key.b << ") but no corresponding column in Omega\n";
                    return false;
                }
            }
        }
    }
    for (const auto& kv : c3_map) {
        const auto& key = kv.first;
        const auto& rng = kv.second;
        for (IloExpr::LinearIterator it = rng.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0) {
                // check if var corresponds to a column in Omega that should be attached to this row
                bool found = false;
                for (const auto& can : Omega) {
                    if (yName(can) == var.getName()) {
                        // check if can should be attached to this row
                        if ((can.u == key.a && can.j == key.b) || (can.v == key.a && can.i == key.b)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    std::cerr << "Error: variable " << var.getName() << " with nonzero coef in C3(" << key.a << "," << key.b << ") but no corresponding column in Omega\n";
                    return false;
                }
            }
        }
    }
    for (const auto& kv : c4_map) {
        const auto& key = kv.first;
        const auto& rng = kv.second;
        for (IloExpr::LinearIterator it = rng.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0) {
                // check if var corresponds to a column in Omega that should be attached to this row
                bool found = false;
                for (const auto& can : Omega) {
                    if (yName(can) == var.getName()) {
                        // check if can should be attached to this row
                        if ((can.i == key.a && can.v == key.b) || (can.j == key.a && can.u == key.b)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    std::cerr << "Error: variable " << var.getName() << " with nonzero coef in C4(" << key.a << "," << key.b << ") but no corresponding column in Omega\n";
                    return false;
                }
            }
        }
    }
    for (const auto& kv : c5_map) {
        const auto& key = kv.first;
        const auto& rng = kv.second;
        for (IloExpr::LinearIterator it = rng.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0) {
                // check if var corresponds to a column in Omega that should be attached to this row
                bool found = false;
                for (const auto& can : Omega) {
                    if (yName(can) == var.getName()) {
                        // check if can should be attached to this row
                        if ((can.i == key.a && can.u == key.b && can.j == key.c) || (can.j == key.a && can.v == key.b && can.i == key.c)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    std::cerr << "Error: variable " << var.getName() << " with nonzero coef in C5(" << key.a << "," << key.b << "," << key.c << ") but no corresponding column in Omega\n";
                    return false;
                }
            }
        }
    }
    for (const auto& kv : c6_map) {
        const auto& key = kv.first;
        const auto& rng = kv.second;
        for (IloExpr::LinearIterator it = rng.getLinearIterator(); it.ok(); ++it) {
            IloNumVar var = it.getVar();
            double coef = it.getCoef();
            if (coef != 0.0) {
                // check if var corresponds to a column in Omega that should be attached to this row
                bool found = false;
                for (const auto& can : Omega) {
                    if (yName(can) == var.getName()) {
                        // check if can should be attached to this row
                        if ((can.i == key.a && can.u == key.b && can.v == key.c) || (can.j == key.a && can.v == key.b && can.u == key.c)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    std::cerr << "Error: variable " << var.getName() << " with nonzero coef in C6(" << key.a << "," << key.b << "," << key.c << ") but no corresponding column in Omega\n";
                    return false;
                }
            }
        }
    }
    
    return true;
}

bool IncrementalRMP::verifySolution(const std::unordered_map<QuadKey,double,QuadKeyHash>& yvals,
                                    const std::unordered_map<PairKey,double,PairKeyHash>& xvals,
                                    const std::unordered_map<int,int>& fixed) const
{
    auto result = true;
    // check C1
    for (int i : V) for (int j : V) {
        double sum_ij = 0.0;
        for (int u=0;u<m;++u) {
            for (int v=0;v<m;++v) {
                if ((i==j && u!=v) || (i!=j && u==v)) continue;
                auto it = yvals.find(canonical(i,u,j,v));
                double val = (it==yvals.end()) ? 0.0 : it->second;
                sum_ij += val;
            }
        }
        if (sum_ij - 1.0 > 1e-5) {
            std::cerr << "Warning C1: solution for locations " << i << "," << j << " sums to " << sum_ij << " != 1.0\n";
            result = false;
        }
    }
    // check C2
    for (int u : M) for (int v : M) {
        double sum_uv = 0.0;
        for (int i=0;i<n;++i) {
            for (int j=0;j<n;++j) {
                if ((i==j && u!=v) || (i!=j && u==v)) continue;
                auto it = yvals.find(canonical(i,u,j,v));
                double val = (it==yvals.end()) ? 0.0 : it->second;
                sum_uv += val;
            }
        }
        if (std::abs(sum_uv - 1.0) > 1e-5) {
            std::cerr << "Warning C2: solution for facilities " << u << "," << v << " sums to " << sum_uv << " != 1.0\n";
            result = false;
        }
    }
    // check C3
    for (int u : M) for (int j : V) {
        double sum_uj = 0.0;
        for (int i=0;i<n;++i) {
            for (int v=0;v<m;++v) {
                if ((i==j && u!=v) || (i!=j && u==v)) continue;
                auto it = yvals.find(canonical(i,u,j,v));
                double val = (it==yvals.end()) ? 0.0 : it->second;
                sum_uj += val;
            }
        }
        if (sum_uj - 1.0 > 1e-5) {
            std::cerr << "Warning C3: solution for facility " << u << " and location " << j << " sums to " << sum_uj << " != 1.0\n";
            result = false;
        }
    }
    // check C4
    for (int i : V) for (int v : M) {
        double sum_iv = 0.0;
        for (int j=0;j<n;++j) {
            for (int u=0;u<m;++u) {
                if ((i==j && u!=v) || (i!=j && u==v)) continue;
                auto it = yvals.find(canonical(i,u,j,v));
                double val = (it==yvals.end()) ? 0.0 : it->second;
                sum_iv += val;
            }
        }
        if (sum_iv - 1.0 > 1e-5) {
            std::cerr << "Warning C4: solution for location " << i << " and facility " << v << " sums to " << sum_iv << " != 1.0\n";
            result = false;
        }
    }
    // check C5
    for (int i : V) for (int u : M) for (int j : V) {
        double sum_iuj = 0.0;
        for (int v=0;v<m;++v) {
            if ((i==j && u!=v) || (i!=j && u==v)) continue;
            auto it = yvals.find(canonical(i,u,j,v));
            double val = (it==yvals.end()) ? 0.0 : it->second;
            sum_iuj += val;
        }
        auto xit = xvals.find(PairKey{i,u});
        double xval = (xit==xvals.end()) ? 0.0 : xit->second;
        if (sum_iuj - xval > 1e-5) {
            std::cerr << "Warning C5: solution for location " << i << " and facility " << u << " and location " << j << " has sum " << sum_iuj << " > x(" << i << "," << u << ")=" << xval << "\n";
            result = false;
        }
    }
    // check C6
    for (int i : V) for (int u : M) for (int v : M) {
        double sum_iuv = 0.0;
        for (int j=0;j<n;++j) {
            if ((i==j && u!=v) || (i!=j && u==v)) continue;
            auto it = yvals.find(canonical(i,u,j,v));
            double val = (it==yvals.end()) ? 0.0 : it->second;
            sum_iuv += val;
        }
        auto xit = xvals.find(PairKey{i,u});
        double xval = (xit==xvals.end()) ? 0.0 : xit->second;
        if (sum_iuv - xval > 1e-5) {
            std::cerr << "Warning C6: solution for location " << i << " and facility " << u << " and facility " << v << " has sum " << sum_iuv << " > x(" << i << "," << u << ")=" << xval << "\n";
            result = false;
        }
    }
    // check x values are in [0,1]
    for (const auto& kv : xvals) {
        const auto& key = kv.first;
        double val = kv.second;
        if (val < -1e-5 || val > 1.0 + 1e-5) {
            std::cerr << "Warning: x(" << key.a << "," << key.b << ") has value " << val << " outside [0,1]\n";
            result = false;
        }
    }
    // check sum u x(i,u) <= 1 for each i
    for (int i : V) {
        double sum_iu = 0.0;
        for (int u : M) {
            auto it = xvals.find(PairKey{i,u});
            double val = (it==xvals.end()) ? 0.0 : it->second;
            sum_iu += val;
        }
        if (sum_iu > 1.0 + 1e-5) {
            std::cerr << "Warning: sum of x(" << i << ",*) = " << sum_iu << " > 1.0\n";
            result = false;
        }
    }
    // check sum i x(i,u) == 1 for each u
    for (int u : M) {
        double sum_iu = 0.0;
        for (int i : V) {
            auto it = xvals.find(PairKey{i,u});
            double val = (it==xvals.end()) ? 0.0 : it->second;
            sum_iu += val;
        }
        if (std::abs(sum_iu - 1.0) > 1e-5) {
            std::cerr << "Warning: sum of x(*," << u << ") = " << sum_iu << " != 1.0\n";
            result = false;
        }
    }
    // calculate objective value and compare to getObjectiveValue()
    double objval = 0.0;
    for (const auto& kv : yvals) {
        const auto& key = kv.first;
        double val = kv.second;
        double cost = (key.i != key.j)
            ? phiIUJV(key.i,key.u,key.j,key.v) + phiIUJV(key.j,key.v,key.i,key.u)
            : phiIUJV(key.i,key.u,key.j,key.v);
        objval += cost * val;
    }
    double reported_objval = getObjectiveValue();
    if (std::abs(objval - reported_objval) > 1e-5) {
        std::cerr << "Warning: calculated objective value " << objval << " differs from reported objective value " << reported_objval << "\n";
        result = false;
    } else {
        std::cout << "Objective value verified: " << objval << "\n";
    }
    // check fixed assignments
    for (const auto& kv : xvals) {
        const auto& key = kv.first;
        double val = kv.second;
        if (val > 1e-5) {
            auto it = fixed.find(key.a);
            if (it != fixed.end()) {
                int ufix = it->second;
                if (key.b != ufix) {
                    std::cerr << "Warning: x(" << key.a << "," << key.b << ") = " << val << " but location " << key.a << " is fixed to facility " << ufix << "\n";
                    result = false;
                }
            }
        }
    }
    return result;
}

void IncrementalRMP::columnsAnalysis() {
    IloCplex::BasisStatusArray varBasis(_env);
    IloNumVarArray vars(_env);
    // get status of var y
    for (auto& kv : y_index) {
        vars.add(kv.second);
    }
    _cplex.getBasisStatuses(varBasis, vars);
    auto count_basis = 0;
    auto count_positive = 0;
    for (int i = 0; i < vars.getSize(); i++) {
        switch (varBasis[i]) {
            case IloCplex::Basic:
                count_basis++;
                break;
            default:
                break;
        }
        double val = _cplex.getValue(vars[i]);
        if (val > 1e-5) {
            count_positive++;
        }
    }
    std::cout << "Total columns: " << vars.getSize() << ", in basis: " << count_basis << ", positive: " << count_positive << "\n";
}

void IncrementalRMP::checkLastAddedColumnBasis() {
    if (lastIterationAddedColumn.i == -1) {
        std::cerr << "No column added yet\n";
        return;
    }
    IloCplex::BasisStatusArray varBasis(_env);
    IloNumVarArray vars(_env);
    vars.add(y_index.at(lastIterationAddedColumn));
    _cplex.getBasisStatuses(varBasis, vars);
    for (int i = 0; i < vars.getSize(); i++) {
        if (varBasis[i] == IloCplex::Basic) {
            countAddedColumnsInBasis++;
            std::cout << "Column " << yName(lastIterationAddedColumn) << " is in the basis. The number of column added enter basis" << countAddedColumnsInBasis << "\n";
        } else {
            std::cout << "Column " << yName(lastIterationAddedColumn) << " is not in the basis.\n";
        }
    }
}
