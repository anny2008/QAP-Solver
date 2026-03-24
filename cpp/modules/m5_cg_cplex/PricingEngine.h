#pragma once
#include "IncrementalRMP.h"
#include "StabilizedDuals.h"
#include <optional>

class PricingEngine {
public:
    struct Result {
        std::optional<QuadKey> best_col;
        double best_rc = 0.0;
        std::vector<QuadKey> negative_cols;
    };

    PricingEngine(const std::vector<int>& V_,
              const std::vector<int>& M_,
              const std::vector<double>& phi_,
              int n_, int m_,
              const std::string& add_column_strategy_,
              double eps_,
              const std::unordered_map<int,int>& fixed_assignments);

    Result price(const IncrementalRMP& rmp);
    Result price(const PerturbedRMPView& rmp);
    
    const std::vector<int>& getV() const { return V; }
    const std::vector<int>& getM() const { return M; }
    double getEps() const { return eps; }
    const std::unordered_map<int,int>& getFixed() const { return fixed; }

private:
    // caches
    std::vector<int> V, M;
    std::vector<double> phi; // flattened
    int n, m;
    std::string add_column_strategy;
    double eps;
    std::unordered_map<int,int> fixed;

    inline int idx(int i, int u, int j, int v) const {
        return (((i * m) + u) * n + j) * m + v;
    }
};