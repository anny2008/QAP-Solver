#pragma once
#include <optional>
#include <unordered_map>
#include <vector>
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

inline QuadKey canonicalize(int i, int u, int j, int v) {
    if (i < j) return QuadKey{i,u,j,v};
    if (i > j) return QuadKey{j,v,i,u};
    // i == j: only diagonal valid
    return QuadKey{i,u,j,v};
}

inline QuadKey canonicalize(QuadKey q) {
    if (q.i < q.j) return QuadKey{q.i,q.u,q.j,q.v};
    if (q.i > q.j) return QuadKey{q.j,q.v,q.i,q.u};
    // i == j: only diagonal valid
    return QuadKey{q.i,q.u,q.j,q.v};
}

struct ColumnGenConfig {
    double time_limit_sec = 120.0;
    bool add_most_negative = true;
    bool use_bigM = false;
    double eps = 1e-7;
    int max_iterations = 10000;
    bool log_output = false;
    bool full_columns_at_once = false;
    int k = 100; // max columns to add per iteration when not adding all


    static ColumnGenConfig fromJson(const json& j) {
        ColumnGenConfig cfg;
        if (j.contains("time_limit")) cfg.time_limit_sec = j["time_limit"];
        if (j.contains("add_most_negative")) cfg.add_most_negative = j["add_most_negative"];
        if (j.contains("use_bigM")) cfg.use_bigM = j["use_bigM"];
        if (j.contains("epsilon")) cfg.eps = j["epsilon"];
        if (j.contains("max_iterations")) cfg.max_iterations = j["max_iterations"];
        if (j.contains("log_output")) cfg.log_output = j["log_output"];
        if (j.contains("full_columns_at_once")) cfg.full_columns_at_once = j["full_columns_at_once"];
        if (j.contains("k")) cfg.k = j["k"];
        return cfg;
    }
};


class ColumnGenSolver {
public:
    explicit ColumnGenSolver(const ColumnGenConfig& cfg);

    // Build and solve for a Problem; fixed_variables is (location -> facility).
    Solution solve(const Problem& problem,
                    const std::string& instance_path,
                    const std::unordered_map<int,int>& fixed_variables = {}
                    );

private:
    ColumnGenConfig cfg;

    // φ(i,u,j,v) flattened
    static std::vector<double> buildPhi(const Problem& P);
    static std::vector<QuadKey> buildInitialOmega(const Problem& P,
        const std::unordered_map<int,int>& fixed);

    // Greedy assignment from x values (argmax_u x_{i,u})
    static std::vector<int> argmaxAssignment(int n, int m,
        const std::unordered_map<PairKey,double,PairKeyHash>& xvals);

};