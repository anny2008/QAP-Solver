#pragma once
#include "IncrementalRMP.h"
#include "StabilizedDuals.h"
#include <optional>

class PricingEngine {
public:
    struct Result {
        std::optional<QuadKey> best_col;
        double best_rc = 0.0;
        std::vector<std::pair<QuadKey,double>> negative_cols;
        std::vector<std::pair<QuadKey,double>> most_negative_cols;
    };

    PricingEngine(const std::vector<int>& V_,
              const std::vector<int>& M_,
              const std::vector<double>& phi_,
              int n_, int m_,
              bool add_most_negative_,
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
    bool add_most_negative;
    double eps;
    std::unordered_map<int,int> fixed;

    inline int idx(int i, int u, int j, int v) const {
        return (((i * m) + u) * n + j) * m + v;
    }
};