#pragma once

#include <ilcplex/ilocplex.h>
#include "gurobi_c++.h"
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
    int i, u, j, v;  // i, j  \in V; u, v \in M; |V|=n, |M|=m
    bool operator==(const QuadKey& o) const { return i==o.i && u==o.u && j==o.j && v==o.v; }
    int to_index(int n, int m) const { return (((i * m) + u) * n + j) * m + v; }
    QuadKey (int index, int n, int m) {
        v = index % m; index /= m;
        j = index % n; index /= n;
        u = index % m; index /= m;
        i = index;
    }

    QuadKey() : i(0), u(0), j(0), v(0) {}
    QuadKey(int i_, int u_, int j_, int v_) : i(i_), u(u_), j(j_), v(v_) {} 
};

struct DualVector {
    // only store duals for c3 and c4
    std::vector<double> c3; // size n*m*n
    std::vector<double> c4; // size n*m*m

    // swap for quick assignment
    void swap(DualVector& other) {
        c3.swap(other.c3);
        c4.swap(other.c4);
    }

    void resize(int n, int m) {
        c3.resize(n*m*n);
        c4.resize(n*m*m);
    }

    void interpolate(const DualVector& other, double lambda) {
        for (size_t i=0;i<c3.size();++i) c3[i] = (1.0-lambda)*c3[i] + lambda*other.c3[i];
        for (size_t i=0;i<c4.size();++i) c4[i] = (1.0-lambda)*c4[i] + lambda*other.c4[i];
    }
};

struct PricingColumn {
    QuadKey column;
    double cost;
};

/**
 * IncrementalRMP — LP RMP in Concert (C1–C6, dynamic y, x in [0,1]).
 * 
 * Rows:
 *   C1(i):  sum_{u} x_{i,u}                    <= 1   (i in V)         (L)
 *   C2(u):  sum_{i} x_{i,u}                    == 1   (u in M)         (E)
 *   C3(i,u,j): sum_v y_{i,u,j,v} - x_{i,u}     <= 0   (i,u in VxM, j in V) (L)
 *   C4(i,u,v): sum_j y_{i,u,j,v} - x_{i,u}     == 0   (i,u in VxM, v in M) (E)
 *
 * Variables:
 *   - y_{i,u,j,v}  (canonical orientation, nonneg, unbounded above)
 *   - x_{i,u} in [0,1]
 *   if i > j: replace y_{i,u,j,v} with y_{j,v,i,u} in C3/C4
 *
 * Objective:
 *   sum phi(i,u,j,v) (+ symmetric term when i!=j) * y_{canonical(i,u,j,v)}
 */
class IncrementalRMP {
public:
    using VList = std::vector<int>;
    using MList = std::vector<int>;

    IncrementalRMP(const std::vector<double>& phi,  // flattened φ(i,u,j,v)
                   int n, int m,
                   double eps  = 1e-9,
                   bool quiet  = true,
                   bool disable_presolve = true);

    // Create the initial model (x, C1–C4 with no y-columns)
    void createModel();

    // Fix x-assignments (loc -> fac).
    void fixXAssignments(const std::unordered_map<int,int>& fixed_loc_to_fac);

    // Canonicalization helper (unordered pair at (i,j))
    static QuadKey canonical(int i, int u, int j, int v) {
        if (i < j) return {i,u,j,v};
        else if (i == j and u <= v) return {i,u,j,v};
        else return {j,v,i,u};
    }

    // Add a canonical y-column if missing; attach to all rows (both orientations).
    bool addColumn(int i, int u, int j, int v);
    bool addColumn(const QuadKey& q);
    
    // Add multiple columns at once (batch operation for performance)
    int addColumns(const std::vector<QuadKey>& columns);
    int addColumns(const std::vector<PricingColumn>& columns);

    // Solve current LP
    bool solve(); // returns true iff optimal/feasible

    // Accessors (mirrors Python RMP API) 
    double getObjectiveValue() const;
    double getRowDualC1(int i) const;
    double getRowDualC2(int u) const;
    double getRowDualC3(int i, int u, int j) const;
    double getRowDualC4(int i, int u, int v) const;

    // Direct dual getter (by GRBConstr)
    double getDual(const GRBConstr& rng) const;

    // Membership for Ω (existing canonical columns)
    bool hasColumn(const QuadKey& q) const;
    bool hasColumn(int i, int u, int j, int v) const;
    // φ access: flattened index: (((i*m + u)*n + j)*m + v)
    inline double phiIUJV(int i, int u, int j, int v) const {
        return phi[(((i * m) + u) * n + j) * m + v];
    }

    void retrieveCurrentDuals();
    bool swapInDualswPiStar();
    void setPiIn(const DualVector& pi) { pi_in = pi; }

    bool pricingWithCurrentDuals(std::vector<PricingColumn>& new_columns, PricingColumn& mostnegative_column, double& mostnegative_reduced_cost); // returns true if new columns were added

    bool pricingWithInOut(double alpha, std::vector<PricingColumn>& new_columns, PricingColumn& mostnegative_column, double& mostnegative_reduced_cost); // returns true if new columns were added


    bool pricing(DualVector& duals, std::vector<PricingColumn>& new_columns, PricingColumn& mostnegative_column, double& mostnegative_reduced_cost); // returns true if new columns were added

    // Concert handles
    GRBEnv&       env();
    GRBModel&     model();

    // Dimensions
    int N() const { return n; }
    int Mdim() const { return m; }

    void columnsAnalysis(); // print analysis of current columns in RMP (for debugging)

    std::vector<QuadKey> Omega;

private:

    // store constraints
    std::vector<GRBConstr> c1_map; // i
    std::vector<GRBConstr> c2_map; // u
    std::vector<GRBConstr> c3_map; // (i,u,j)
    std::vector<GRBConstr> c4_map; // (i,u,v)

    std::vector<GRBVar> x_index; // (i,u)
    std::vector<GRBVar> y_index; // canonical (i,u,j,v)



    // Concert objects
    GRBEnv   _env;
    GRBModel _model;
    std::vector<bool> is_in_Omega;

    // parameters / data
    std::vector<double> phi;
    int n, m;
    bool quiet;
    // Name helpers
    static std::string yName(const QuadKey& q);
    static std::string xName(int i, int u);

    DualVector pi_iter; // the dual vector at the current iteration, will be the pi_out
    DualVector pi_star; // dual vector used for pricing
    DualVector pi_in;  // a valid dual vector to interpolate from (for stabilization) 
};