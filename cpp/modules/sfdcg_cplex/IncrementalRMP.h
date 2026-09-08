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
#include "SubgraphDecomposition.h"
#include "Config.h"


struct PairKey {
    int a;
    int b;
    PairKey(int a, int b) : a(a), b(b) {}
    bool operator==(const PairKey& o) const noexcept {
        return a == o.a && b == o.b;
    }
};

struct PairKeyHash {
    std::size_t operator()(PairKey const& p) const noexcept {
        std::size_t h1 = std::hash<int>{}(p.a);
        std::size_t h2 = std::hash<int>{}(p.b);

        // boost::hash_combine pattern
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};
struct TripleKey {
    int a, b, c;
    TripleKey(int a, int b, int c) : a(a), b(b), c(c) {}
    bool operator==(const TripleKey& o) const noexcept {
        return a == o.a && b == o.b && c == o.c;
    }
};

struct TripleKeyHash {
    std::size_t operator()(TripleKey const& k) const noexcept {
        std::size_t h1 = std::hash<int>{}(k.a);
        std::size_t h2 = std::hash<int>{}(k.b);
        std::size_t h3 = std::hash<int>{}(k.c);

        std::size_t h = h1;
        h ^= h2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= h3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

/**
 * IncrementalRMP — LP RMP
 * 
 * Rows:
 *   C1(u):     sum_i x_iu   <= 1                          : alpha_u    (i in V, u in M)   (L)
 *   C2(i):     sum_u x_iu   == 1                          : beta_i     (i in V, u in M)   (E)
 *   C3(i,k):   sum_j e^k_ij = sum_u Degree_out^k_u * x_iu : lambda_ik  (i in V, k in subgraphs) (E)
 *   C4(i,k):   sum_i e^k_ij = sum_v Degree_in^k_v * x_jv  : theta_jv   (j in V, k in subgraphs) (E)
 *
 * Variables:
 *   - 0 <= e^k_ij  <= 1
 *   - x_{i,u} in [0,1]
 *
 * Objective:
 *   sum_kij F_k * d_ij * e^k_ij for e^k_ij in Omega
 */
class IncrementalRMP {
public:
    IncrementalRMP(Problem& problem, SolverConfig& cfg, const std::vector<Subgraph>& subgraphs);
    // create the model with only x and all the rows, but no e columns; then we will add e^k_ij columns incrementally
    void createModel();
    // add columns for the given (k,i,j) keys;
    int addColumns(const std::vector<TripleKey>& new_cols);
    // add a single column for (k,i,j)
    bool addColumn(const TripleKey& col);
    // solve the LP
    bool solve();
    // fill duals lambda_ik then theta_jv
    void getDuals(std::vector<double>& duals) const;
    // is this valid to add column for (k,i,j) (i.e. not already in Omega, and corresponding rows exist)
    bool canAddColumn(const TripleKey& col) const;
    // count the basis columns and positive columns in the current solution and print them (for debugging)
    // if given a path, write all the basis columns to file
    void countBasisAndPositiveColumns(std::string basis_output_path = ""); 

    double getObjectiveValue() const {
        return cplex.getObjValue();
    }
    
    Problem& problem; // reference to the problem data
    SolverConfig& cfg; // reference to the configuration
    std::vector<Subgraph> subgraphs; // list of subgraphs for this problem
    std::vector<TripleKey> omega; // list of (k,i,j) in current RMP
    std::vector<TripleKey> not_in_omega; // list of (k,i,j) not in current RMP but can be added (for debugging)
    int nMaxColsPerSubgraph = 1;
private:
    std::vector<int> not_in_omega_pos;                        // pos[id] = index in not_in_omega
    std::unordered_map<TripleKey, int, TripleKeyHash> omega_index; // maps (k,i,j) to column index in CPLEX
    IloEnv env; // CPLEX environment
    IloModel model;  // CPLEX model
    IloCplex cplex; // CPLEX solver interface
    IloObjective objective; // objective function
    IloNumVarArray e_vars; // e^k_ij variables
    IloNumVarArray x_vars; // x_{i,u} variables
    // store the row expressions for C3 and C4 to facilitate adding columns
    std::unordered_map<PairKey, IloRange, PairKeyHash> row_C3; // key = "k,i"
    std::unordered_map<PairKey, IloRange, PairKeyHash> row_C4; // key = "k,j"
    double lower_bound = 0.0; // current lower bound from RMP

    void removeColumn(const TripleKey& col); // remove column for (k,i,j) from the model and update corresponding rows and objective


};