#pragma once

#include <ilcplex/ilocplex.h>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <optional>
#include <sstream>
#include <functional>

#include "../../core/problem.h"
#include "../../core/solution.h"

// A compact hash for (i,u,j,v) and (i,u) keys
struct QuadKey {
    int i, u, j, v;
    bool operator==(const QuadKey& o) const { return i==o.i && u==o.u && j==o.j && v==o.v; }
};
struct QuadKeyHash {
    std::size_t operator()(const QuadKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](int x) { h ^= std::hash<int>{}(x); h *= 1099511628211ull; };
        mix(k.i); mix(k.u); mix(k.j); mix(k.v);
        return h;
    }
};

struct PairKey {
    int a, b;
    bool operator==(const PairKey& o) const { return a==o.a && b==o.b; }
};
struct PairKeyHash {
    std::size_t operator()(const PairKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](int x) { h ^= std::hash<int>{}(x); h *= 1099511628211ull; };
        mix(k.a); mix(k.b);
        return h;
    }
};
struct TripleKey {
    int a, b, c;
    bool operator==(const TripleKey& o) const {
        return a==o.a && b==o.b && c==o.c;
    }
};

struct TripleKeyHash {
    std::size_t operator()(const TripleKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](int x) {
            h ^= (std::size_t)x;
            h *= 1099511628211ull;
        };
        mix(k.a); mix(k.b); mix(k.c);
        return h;
    }
};
/**
 * IncrementalRMP — LP RMP in Concert (C1–C6, dynamic y, x in [0,1]).
 * 
 * Rows:
 *   C1(i,j):  sum_{u,v} y_{i,u,j,v}            <= 1   (i,j in V)         (L)
 *   C2(u,v):  sum_{i,j} y_{i,u,j,v}            == 1   (u,v in M)         (E)
 *   C3(u,j):  sum_{i,v} y_{i,u,j,v}            <= 1   (u in M, j in V)   (L)
 *   C4(i,v):  sum_{u,j} y_{i,u,j,v}            <= 1   (i in V, v in M)   (L)
 *   C5(i,u,j): sum_v y_{i,u,j,v} - x_{i,u}     <= 0   (i,u in VxM, j in V) (L)
 *   C6(i,u,v): sum_j y_{i,u,j,v} - x_{i,u}     == 0   (i,u in VxM, v in M) (E)
 *
 * Variables:
 *   - y_{i,u,j,v}  (canonical orientation, nonneg, unbounded above)
 *   - x_{i,u} in [0,1]
 *
 * Objective:
 *   sum phi(i,u,j,v) (+ symmetric term when i!=j) * y_{canonical(i,u,j,v)}
 */
class IncrementalRMP {
public:
    using VList = std::vector<int>;
    using MList = std::vector<int>;

    IncrementalRMP(const VList& V, const MList& M,
                   const std::vector<double>& phi,  // flattened φ(i,u,j,v)
                   int n, int m,
                   double eps  = 1e-9,
                   bool quiet  = true,
                   bool disable_presolve = true);

    // Build all row shells upfront (as Python ensure_all_rows)  [1](https://michelingroup-my.sharepoint.com/personal/thi-thuy-an_tran_michelin_com/Documents/Fichiers%20Microsoft%20Copilot%20Chat/pricing.py)
    void ensureAllRows();

    // Fix x-assignments (loc -> fac).  [1](https://michelingroup-my.sharepoint.com/personal/thi-thuy-an_tran_michelin_com/Documents/Fichiers%20Microsoft%20Copilot%20Chat/pricing.py)
    void fixXAssignments(const std::unordered_map<int,int>& fixed_loc_to_fac);

    // Canonicalization helper (unordered pair at (i,j))  [1](https://michelingroup-my.sharepoint.com/personal/thi-thuy-an_tran_michelin_com/Documents/Fichiers%20Microsoft%20Copilot%20Chat/pricing.py)
    static QuadKey canonical(int i, int u, int j, int v) {
        if (i < j) return {i,u,j,v};
        else if (i == j and u <= v) return {i,u,j,v};
        else return {j,v,i,u};
    }

    // Add a canonical y-column if missing; attach to all rows (both orientations).  [1](https://michelingroup-my.sharepoint.com/personal/thi-thuy-an_tran_michelin_com/Documents/Fichiers%20Microsoft%20Copilot%20Chat/pricing.py)
    IloNumVar addColumn(int i, int u, int j, int v);
    
    // Add multiple columns at once (batch operation for performance)
    void addColumns(const std::vector<QuadKey>& columns);

    // Solve current LP
    bool solve(); // returns true iff optimal/feasible

    // Accessors (mirrors Python RMP API)  [1](https://michelingroup-my.sharepoint.com/personal/thi-thuy-an_tran_michelin_com/Documents/Fichiers%20Microsoft%20Copilot%20Chat/pricing.py)
    double getObjectiveValue() const;
    double getRowDualC1(int i, int j) const;
    double getRowDualC2(int u, int v) const;
    double getRowDualC3(int u, int j) const;
    double getRowDualC4(int i, int v) const;
    double getRowDualC5(int i, int u, int j) const;
    double getRowDualC6(int i, int u, int v) const;

    // Extract positive y values (> eps)
    std::unordered_map<QuadKey, double, QuadKeyHash> getYValues() const;
    // Get all x values
    std::unordered_map<PairKey, double, PairKeyHash> getXValues() const;

    // Direct dual getter (by IloRange)
    double getDual(const IloRange& rng) const;

    // Membership for Ω (existing canonical columns)
    bool hasColumn(int i, int u, int j, int v) const;
    const std::unordered_set<QuadKey, QuadKeyHash>& omega() const { return Omega; }

    // φ access: flattened index: (((i*m + u)*n + j)*m + v)
    inline double phiIUJV(int i, int u, int j, int v) const {
        return phi[(((i * m) + u) * n + j) * m + v];
    }

    // Concert handles
    IloEnv&       env();
    IloModel&     model();
    IloCplex&     cplex();

    // Dimensions
    int N() const { return n; }
    int Mdim() const { return m; }

    bool debugCheckColumnAttachments() const; // check all the rows to see if the columns are correctly attached and if there any column should not be but is attached  (for debugging)
    bool verifySolution(const std::unordered_map<QuadKey,double,QuadKeyHash>& yvals,
                        const std::unordered_map<PairKey,double,PairKeyHash>& xvals,
                        const std::unordered_map<int,int>& fixed) const; // verify that the given solution satisfies all constraints (for debugging)
    void columnsAnalysis(); // print analysis of current columns in RMP (for debugging)
    void checkLastAddedColumnBasis(); // check if the last added column is in the basis (for debugging)

private:
    // helpers to ensure/create rows and variables
    IloRange& ensureC1(int i, int j);
    IloRange& ensureC2(int u, int v);
    IloRange& ensureC3(int u, int j);
    IloRange& ensureC4(int i, int v);
    IloRange& ensureC5(int i, int u, int j);
    IloRange& ensureC6(int i, int u, int v);
    IloNumVar ensureX(int i, int u);

    // internal maps
    std::unordered_map<PairKey, IloRange, PairKeyHash> c1_map; // (i,j)
    std::unordered_map<PairKey, IloRange, PairKeyHash> c2_map; // (u,v)
    std::unordered_map<PairKey, IloRange, PairKeyHash> c3_map; // (u,j)
    std::unordered_map<PairKey, IloRange, PairKeyHash> c4_map; // (i,v)
    std::unordered_map<TripleKey, IloRange, TripleKeyHash> c5_map;
    std::unordered_map<TripleKey, IloRange, TripleKeyHash> c6_map;

    std::unordered_map<PairKey, IloNumVar, PairKeyHash> x_index; // (i,u)
    std::unordered_map<QuadKey, IloNumVar, QuadKeyHash> y_index; // canonical (i,u,j,v)


    // Concert objects
    IloEnv   _env;
    IloModel _model;
    IloCplex _cplex;
    IloObjective _obj;
    std::unordered_set<QuadKey, QuadKeyHash> Omega;

    // parameters / data
    std::vector<int> V;
    std::vector<int> M;
    std::vector<double> phi;
    int n, m;
    double eps;
    bool quiet;
    bool disable_presolve;

    // Name helpers
    static std::string yName(const QuadKey& q);
    static std::string xName(int i, int u);

    QuadKey lastIterationAddedColumn{-1,-1,-1,-1}; // for debugging: store the last added column's canonical key
    int countAddedColumnsInBasis = 0; // for debugging: count how many of the added columns are in the basis
};