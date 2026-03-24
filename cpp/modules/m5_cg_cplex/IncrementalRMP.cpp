#include "IncrementalRMP.h"
#include <algorithm>
#include <iomanip>
#include "StabilizedDuals.h"

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
                                , lastIterBasicStatus()
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

template <class MapRows, class KeyT>
IloRange& ensureRowImpl(
    IloEnv& env,
    IloModel& model,
    MapRows& rowmap,
    const KeyT& key,
    char sense,
    double rhs)
{
    // Already exists
    auto it = rowmap.find(key);
    if (it != rowmap.end()) return it->second;
    // Create the row
    IloRange rng(env, -IloInfinity, rhs);
    if (sense == 'L') rng.setUB(rhs);
    else if (sense == 'E') rng.setBounds(rhs, rhs);
    else if (sense == 'G') rng.setLB(rhs);

    model.add(rng);
    rowmap.emplace(key, rng);
    IloRange& row = rowmap.at(key);
    return row;
}

// C1 i,j, i < j: sum_{u,v} y(i,u,j,v) <= 1
IloRange& IncrementalRMP::ensureC1(int i, int j) {
    auto key = i < j?  PairKey{i,j}: PairKey{j,i};
    return ensureRowImpl(_env, _model, c1_map, key, 'L', 1.0);
}

//  C2 u,v u < v: sum_{i,j} y(i,u,j,v) == 1
IloRange& IncrementalRMP::ensureC2(int u, int v) {
    auto key = u < v?  PairKey{u,v}: PairKey{v,u};
    return ensureRowImpl(_env, _model, c2_map, key, 'E', 1.0);
}

// C3 u,j: sum_{i,v} y(i,u,j,v) <= 1
IloRange& IncrementalRMP::ensureC3(int u, int j) {
    return ensureRowImpl(_env, _model, c3_map, PairKey{u,j}, 'L', 1.0);
}

// C4 i,u,j, i < j: sum_{j,u} y(i,u,j,v) <= x_iu
// C4 i,u,j, i > j: sum_{j,u} y(j,v,i,u) <= x_iu
IloRange& IncrementalRMP::ensureC4(int i, int u, int j)
{
    TripleKey key{i,u,j};

    auto it = c4_map.find(key);
    if (it != c4_map.end())
        return it->second;

    // Create base row: sum y(...) - x(i,u) <= 0
    IloRange rng(_env, -IloInfinity, 0.0);
    rng.setUB(0.0);
    _model.add(rng);

    auto inserted = c4_map.emplace(key, rng);
    IloRange& row = inserted.first->second;

    // Add coefficient for x(i,u)
    row.setLinearCoef(ensureX(i,u), -1.0);

    return row;
}

// C5 i,u,v: sum_{j| i < j} y(i,u,j,v) + y(i,v,j,u) == x_iu
IloRange& IncrementalRMP::ensureC5(int i, int u, int v) 
{
    TripleKey key{i,u,v};
    auto it = c5_map.find(key);
    if (it != c5_map.end()) return it->second;
    IloRange rng(_env, -IloInfinity, 0.0);
    rng.setBounds(0.0, 0.0);
    _model.add(rng);
    auto inserted = c5_map.emplace(key, rng);
    IloRange& row = inserted.first->second;

    // Add coefficient for x(i,u)
    row.setLinearCoef(ensureX(i,u), -1.0);

    return row;
}
void IncrementalRMP::ensureAllRows() {
    for (int i : V) for (int j : V) if (i < j) ensureC1(i,j);
    for (int u : M) for (int v : M) if (u < v) ensureC2(u,v);
    for (int u : M) for (int j : V) ensureC3(u,j);
    for (int i : V) for (int u : M) for (int j : V) if (i != j) ensureC4(i,u,j);
    for (int i : V) for (int u : M) for (int v : M) if (u != v) ensureC5(i,u,v);

    // compute and print out the number of rows of each type for debugging
    std::cout << "[RMP] Total rows after ensureAllRows: " << c1_map.size() + c2_map.size() + c3_map.size() + c4_map.size() + c5_map.size() << " (C1: " << c1_map.size() << ", C2: " << c2_map.size() << ", C3: " << c3_map.size() << ", C4: " << c4_map.size() << ", C5: " << c5_map.size() << ")\n";
}

void IncrementalRMP::addAllTplusTminus()
{
    IloEnv& env = _env;

    // Helper lambda to attach t+ and t- to every row in a rowmap
    auto attach_tp_tm = [&](auto& rowmap,
                            auto& tplus_map,
                            auto& tminus_map)
    {
        for (auto& kv : rowmap)
        {
            const auto& key = kv.first;
            IloRange& row = kv.second;

            // Only add if not already present
            if (!tplus_map.count(key))
            {
                IloNumVar tp(env, 0.0, IloInfinity, ILOFLOAT);
                IloNumVar tm(env, 0.0, IloInfinity, ILOFLOAT);

                _model.add(tp);
                _model.add(tm);

                // Add linear coefficients
                row.setLinearCoef(tp, +1.0);
                row.setLinearCoef(tm, -1.0);

                // Store handles
                tplus_map[key]  = tp;
                tminus_map[key] = tm;

                // Initialization: objective contribution is 0
                _obj.setLinearCoef(tp, 0.0);
                _obj.setLinearCoef(tm, 0.0);
            }
        }
    };

    // auto attach_tm = [&](auto& rowmap, auto& tminus_map) {
    //     for (auto& kv : rowmap) {
    //         const auto& key = kv.first;
    //         IloRange& row = kv.second;

    //         if (!tminus_map.count(key)) {
    //             IloNumVar tm(env, 0.0, IloInfinity, ILOFLOAT);
    //             _model.add(tm);
    //             row.setLinearCoef(tm, -1.0);
    //             tminus_map[key] = tm;
    //             _obj.setLinearCoef(tm, 0.0);
    //         }
    //     }
    // };

    // auto attach_tp = [&](auto& rowmap, auto& tplus_map) {
    //     for (auto& kv : rowmap) {
    //         const auto& key = kv.first;
    //         IloRange& row = kv.second;

    //         if (!tplus_map.count(key)) {
    //             IloNumVar tp(env, 0.0, IloInfinity, ILOFLOAT);
    //             _model.add(tp);
    //             row.setLinearCoef(tp, +1.0);
    //             tplus_map[key] = tp;
    //             _obj.setLinearCoef(tp, 0.0);
    //         }
    //     }
    // };

    // Add t+/t- to every family of constraints
    // attach_tm(c1_map, t_minus_C1);
    // attach_tp(c1_map, t_plus_C1);
    attach_tp_tm(c1_map, t_plus_C1, t_minus_C1);
    attach_tp_tm(c2_map, t_plus_C2, t_minus_C2);
    attach_tp_tm(c3_map, t_plus_C3, t_minus_C3);
    attach_tp_tm(c4_map, t_plus_C4, t_minus_C4);
    // attach_tm(c3_map, t_minus_C3);
    // attach_tp(c3_map, t_plus_C3);
    // attach_tm(c4_map, t_minus_C4);
    // attach_tp(c4_map, t_plus_C4);
    attach_tp_tm(c5_map, t_plus_C5, t_minus_C5);

    std::cout << "[RMP] addAllTplusTminus(): attached t+/t- to "
              << (c1_map.size() + c2_map.size() + c3_map.size() +
                  c4_map.size() + c5_map.size())
              << " constraints." << std::endl;
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
    if (i==j) {
        std::cerr << "Error: cannot add column for (i,u,j,v) = (" << i << "," << u << "," << j << "," << v << ") because i==j but u!=v\n";
        throw std::runtime_error("Invalid column");
    }
    if (u==v) {
        std::cerr << "Error: cannot add column for (i,u,j,v) = (" << i << "," << u << "," << j << "," << v << ") because i!=j but u==v\n";
        throw std::runtime_error("Invalid column");
    }
    QuadKey can = canonical(i,u,j,v);
    if (y_index.count(can)) return y_index[can];

    auto ci = can.i, cu = can.u, cj = can.j, cv = can.v;

    // Objective coefficient
    double cost = phiIUJV(ci,cu,cj,cv) + phiIUJV(cj,cv,ci,cu);

    // Build column with all coefficients at once (more efficient than adding variable then setting coefficients)
    auto obj = _cplex.getObjective();
    IloNumColumn col = obj(cost);
    if (ci < cj) col += ensureC1(ci,cj)(1.0);
    else col += ensureC1(cj,ci)(1.0);
        
    if (cu < cv) col += ensureC2(cu,cv)(1.0);
    else col += ensureC2(cv,cu)(1.0);

    col += ensureC3(cu,cj)(1.0);
    col += ensureC3(cv,ci)(1.0);

    col += ensureC4(ci,cu,cj)(1.0);
    col += ensureC4(cj,cv,ci)(1.0);

    col += ensureC5(ci,cu,cv)(1.0);
    col += ensureC5(cj,cv,cu)(1.0);
    
    // Create variable with column (column constructor: col, lb, ub, type, name)
    IloNumVar y(col, 0.0, IloInfinity, ILOFLOAT);
    // add y to model and col end
    _model.add(y);
    col.end();
    y_index.emplace(can, y);
    Omega.insert(can);
    return y;
}

void IncrementalRMP::addColumns(const std::vector<QuadKey>& columns) {
    for (const auto& col : columns) {
        addColumn(col.i, col.u, col.j, col.v);
    }
}

bool IncrementalRMP::solve() {
    _cplex.setOut(quiet ? _env.getNullStream() : std::cout);
    return _cplex.solve();
}

double IncrementalRMP::getObjectiveValue() const {
    return _cplex.getObjValue();
}

double IncrementalRMP::getRealObjectiveValue() const {
    // compute the real objective value based on the current y variables, instead of relying on CPLEX's objective which may include stabilization terms
    double real_obj = 0.0;
    for (const auto& kv : y_index) {
        const QuadKey& q = kv.first;
        double val = _cplex.getValue(kv.second);
        if (val > eps) {
            real_obj += val * (phiIUJV(q.i,q.u,q.j,q.v) + phiIUJV(q.j,q.v,q.i,q.u));
        }
    }
    return real_obj;
}

double IncrementalRMP::getDual(const IloRange& rng) const {
    return _cplex.getDual(rng);
}

double IncrementalRMP::getRowDualC1(int i, int j) const { return getDual(c1_map.at(PairKey{i,j})); }
double IncrementalRMP::getRowDualC2(int u, int v) const { return getDual(c2_map.at(PairKey{u,v})); }
double IncrementalRMP::getRowDualC3(int u, int j) const { return getDual(c3_map.at(PairKey{u,j})); }
double IncrementalRMP::getRowDualC4(int i, int u, int j) const { return getDual(c4_map.at(TripleKey{i,u,j})); }
double IncrementalRMP::getRowDualC5(int i, int u, int v) const { return getDual(c5_map.at(TripleKey{i,u,v})); }

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

void IncrementalRMP::updateStabilizationCoefficients(
    const DenseDuals& hat_pi,
    const DenseDuals& delta)
{   

    auto delta_column_cost_min = std::numeric_limits<double>::max();
    auto delta_column_cost_max = 0.0;
    auto sum_phi_iujv = 0.0;
    for (int i=0;i<n;++i) for (int u=0;u<m;++u)
    for (int j=i+1;j<n;++j) for (int v=0;v<m;++v) {
        if ((i==j && u!=v) || (i!=j && u==v)) continue;
        double c = phiIUJV(i,u,j,v) + phiIUJV(j,v,i,u);
        if (c >= 1 && c < delta_column_cost_min) {
            delta_column_cost_min = c;
        }
        if (c > delta_column_cost_max) {
            delta_column_cost_max = c;
        }
        sum_phi_iujv += c;
    }
    auto delta_column_cost = (delta_column_cost_min + delta_column_cost_max) / 2.0;
    // std::cout << std::fixed << std::setprecision(6);
    // std::cout << "[RMP] updateStabilizationCoefficients: delta_i = " << delta_column_cost << std::endl;
    // -----------------------------
    // C1 : dual alpha(i,j)
    // -----------------------------
    for (const auto& kv : c1_map)
    {
        const PairKey& key = kv.first;
        int i = key.a;
        int j = key.b;

        IloNumVar tp = t_plus_C1.at(key);
        IloNumVar tm = t_minus_C1.at(key);

        double h = hat_pi.getC1(i, j);
        // double d = delta.getC1(i, j);
        double d = std::max(delta.getC1(i, j), delta_column_cost_min);
        d = std::min(d, delta_column_cost_max); // add a small positive lower bound to prevent stalling when all reduced costs are very small
        // double d = delta_column_cost;

        _obj.setLinearCoef(tp,  h + d);
        _obj.setLinearCoef(tm, -h + d);
    }

    // -----------------------------
    // C2 : dual beta(u,v)
    // -----------------------------
    for (const auto& kv : c2_map)
    {
        const PairKey& key = kv.first;
        int u = key.a;
        int v = key.b;

        IloNumVar tp = t_plus_C2.at(key);
        IloNumVar tm = t_minus_C2.at(key);

        double h = hat_pi.getC2(u, v);
        // double d = delta.getC2(u, v);
        double d = std::max(delta.getC2(u, v), delta_column_cost_min);
        d = std::min(d, delta_column_cost_max);
        // double d = delta_column_cost;

        _obj.setLinearCoef(tp,  h + d);
        _obj.setLinearCoef(tm, -h + d);
    }

    // -----------------------------
    // C3 : dual gamma(u,j)
    // -----------------------------
    for (const auto& kv : c3_map)
    {
        const PairKey& key = kv.first;
        int u = key.a;
        int j = key.b;

        IloNumVar tp = t_plus_C3.at(key);
        IloNumVar tm = t_minus_C3.at(key);

        double h = hat_pi.getC3(u, j);
        // double d = delta.getC3(u, j);
        double d = std::max(delta.getC3(u, j), delta_column_cost_min);
        d = std::min(d, delta_column_cost_max);
        // double d = delta_column_cost;

        _obj.setLinearCoef(tp,  h + d);
        _obj.setLinearCoef(tm, -h + d);
    }

    // -----------------------------
    // C4 : dual eta(i,u,j)
    // -----------------------------
    for (const auto& kv : c4_map)
    {
        const TripleKey& key = kv.first;
        int i = key.a;
        int u = key.b;
        int j = key.c;

        IloNumVar tp = t_plus_C4.at(key);
        IloNumVar tm = t_minus_C4.at(key);

        double h = hat_pi.getC4(i, u, j);
        // double d = delta.getC4(i, u, j);
        double d = std::max(delta.getC4(i, u, j), delta_column_cost_min);
        d = std::min(d, delta_column_cost_max);
        // double d = delta_column_cost;

        _obj.setLinearCoef(tp,  h + d);
        _obj.setLinearCoef(tm, -h + d);
    }

    // -----------------------------
    // C5 : dual mu(i,u,v)
    // -----------------------------
    for (const auto& kv : c5_map)
    {
        const TripleKey& key = kv.first;
        int i = key.a;
        int u = key.b;
        int v = key.c;

        IloNumVar tp = t_plus_C5.at(key);
        IloNumVar tm = t_minus_C5.at(key);

        double h = hat_pi.getC5(i, u, v);
        // double d = delta.getC5(i, u, v);
        double d = std::max(delta.getC5(i, u, v), delta_column_cost_min);
        d = std::min(d, delta_column_cost_max);
        // double d = delta_column_cost;

        _obj.setLinearCoef(tp,  h + d);
        _obj.setLinearCoef(tm, -h + d);
    }
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
        if (u == v) continue;
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
void IncrementalRMP::checkBasicStatusChange() {
    IloNumVarArray vars(_env);
    std::vector<QuadKey> allColumns;
    for (auto& kv : y_index) {
        vars.add(kv.second);
        allColumns.push_back(kv.first);
    }
    IloCplex::BasisStatusArray curBasis(_env);
    _cplex.getBasisStatuses(curBasis, vars);
    if (!lastIterBasicStatus.empty()) {
        int countEnteredBasis = 0;
        int countEnteredBasisNoneZero = 0;
        int countLeftBasis = 0;
        int countUnchanged = 0;
        int countBasis = 0;

        for (int i = 0; i < vars.getSize(); i++) {
            auto curStatus = curBasis[i];
            auto it = lastIterBasicStatus.find(allColumns[i]);
            if (it != lastIterBasicStatus.end()) {
                auto lastStatus = it->second;
                if (curStatus == IloCplex::Basic) countBasis++;
                if (curStatus != lastStatus) {
                    if (curStatus == IloCplex::Basic && lastStatus != IloCplex::Basic) {
                        countEnteredBasis++;
                        if (_cplex.getValue(vars[i]) > 1e-5) {
                            countEnteredBasisNoneZero++;
                        }
                    } else if (curStatus != IloCplex::Basic && lastStatus == IloCplex::Basic) {
                        countLeftBasis++;
                    }
                } else {
                    countUnchanged++;
                }
            } else if (curStatus == IloCplex::Basic) {
                countEnteredBasis++;
            }
        }
        std::cout << "Columns that entered the basis: " << countEnteredBasis << "\n";
        std::cout << "Columns that entered the basis with non-zero value: " << countEnteredBasisNoneZero << "\n";
        std::cout << "Columns that left the basis: " << countLeftBasis << "\n";
        std::cout << "Columns that remained unchanged status: " << countUnchanged << "\n";
        std::cout << "Columns in the basis: " << countBasis << "\n";
    }
        // Update lastIterBasicStatus for the next iteration
    lastIterBasicStatus.clear();
    for (int i = 0; i < vars.getSize(); i++) {
        lastIterBasicStatus.emplace(allColumns[i], curBasis[i]);
    }
}
