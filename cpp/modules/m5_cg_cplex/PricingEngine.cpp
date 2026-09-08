// PricingEngine.cpp
#include "PricingEngine.h"
#include "StabilizedDuals.h"
#include <omp.h>


PricingEngine::PricingEngine(
    const std::vector<int>& V_,
    const std::vector<int>& M_,
    const std::vector<double>& phi_,
    int n_, int m_,
    const std::string& add_column_strategy_,
    double eps_,
    const std::unordered_map<int,int>& fixed_assignments)
    :
    V(V_),
    M(M_),
    phi(phi_),
    n(n_),
    m(m_),
    add_column_strategy(add_column_strategy_),
    eps(eps_),
    fixed(fixed_assignments)
{}

template <typename RMPType>
PricingEngine::Result runPricing(const PricingEngine& self, const RMPType& rmp, std::string add_column_strategy)
{
    const auto& V = self.getV();
    const auto& M = self.getM();
    const auto& fixed = self.getFixed();
    double eps = self.getEps();

    int NV = V.size();
    int NM = M.size();

    // Precompute domains
    std::vector<const int*> Ui_ptr(NV);
    std::vector<int> Ui_len(NV);
    std::vector<const int*> Vj_ptr(NV);
    std::vector<int> Vj_len(NV);

    static thread_local int oneval[1];

    for (int i : V) {
        if (fixed.count(i)) {
            oneval[0] = fixed.at(i);
            Ui_ptr[i] = oneval;
            Ui_len[i] = 1;
        } else {
            Ui_ptr[i] = M.data();
            Ui_len[i] = NM;
        }
    }

    for (int j : V) {
        if (fixed.count(j)) {
            oneval[0] = fixed.at(j);
            Vj_ptr[j] = oneval;
            Vj_len[j] = 1;
        } else {
            Vj_ptr[j] = M.data();
            Vj_len[j] = NM;
        }
    }

    PricingEngine::Result R;
    R.best_rc = 0.0;

    auto compute_rc_for_col = [&](int i, int u, int j, int v) {
        double rc = rmp.phiIUJV(i,u,j,v) + rmp.phiIUJV(j,v,i,u);

        rc -= i < j ? rmp.getRowDualC1(i,j) : rmp.getRowDualC1(j,i);
        rc -= u < v ? rmp.getRowDualC2(u,v) : rmp.getRowDualC2(v,u);
        rc -= rmp.getRowDualC3(u,j);
        rc -= rmp.getRowDualC3(v,i);
        rc -= rmp.getRowDualC4(i,u,j);
        rc -= rmp.getRowDualC4(j,v,i);
        rc -= rmp.getRowDualC5(i,u,v);
        rc -= rmp.getRowDualC5(j,v,u);
        R.all_rcs[QuadKey{i,u,j,v}] = rc; // store for debugging
        return rc;
    };

    #pragma omp parallel
    {
        double best_rc_local = 0.0;
        QuadKey best_col_local;
        std::vector<std::pair<QuadKey,double>> neg_local;

        #pragma omp for schedule(dynamic)
        for (int idx = 0; idx < NV; ++idx) {
            int i = V[idx];
            const int* Ui = Ui_ptr[i];
            int lenUi = Ui_len[i];

            for (int ui = 0; ui < lenUi; ++ui) {
                int u = Ui[ui];

                for (int j : V) {
                    const int* Vj = Vj_ptr[j];
                    int lenVj = Vj_len[j];

                    for (int vj = 0; vj < lenVj; ++vj) {
                        int v = Vj[vj];

                        if ((i >= j) || (u == v)) continue;

                        if (rmp.hasColumn(i,u,j,v)) continue;
                        double rc = compute_rc_for_col(i,u,j,v);
                        if (rc < 0) {
                            neg_local.emplace_back(QuadKey{i,u,j,v}, rc);
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (const auto& kv : neg_local) {
                if (kv.second < R.best_rc -eps) {
                    R.best_rc = kv.second;
                    R.best_col = kv.first;
                    if (add_column_strategy == "add_most_negative") {
                        R.negative_cols.clear();
                        R.negative_cols.push_back(kv.first);
                    }
                } else if (kv.second < -eps) {
                    if (add_column_strategy == "add_most_negative") {
                        if (kv.second < R.best_rc + eps) {
                            R.negative_cols.push_back(kv.first);
                        }
                    } else {
                        R.negative_cols.push_back(kv.first);
                    }
                }
            }
        }
    }

    return R;
}

PricingEngine::Result PricingEngine::price(const IncrementalRMP& rmp) {
    return runPricing(*this, rmp, add_column_strategy);
}

PricingEngine::Result PricingEngine::price(const PerturbedRMPView& rmp) {
    return runPricing(*this, rmp, add_column_strategy);
}