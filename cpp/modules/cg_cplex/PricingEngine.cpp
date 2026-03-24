// PricingEngine.cpp
#include "PricingEngine.h"
#include "StabilizedDuals.h"
#include <omp.h>


PricingEngine::PricingEngine(
    const std::vector<int>& V_,
    const std::vector<int>& M_,
    const std::vector<double>& phi_,
    int n_, int m_,
    bool add_most_negative_,
    double eps_,
    const std::unordered_map<int,int>& fixed_assignments)
    :
    V(V_),
    M(M_),
    phi(phi_),
    n(n_),
    m(m_),
    add_most_negative(add_most_negative_),
    eps(eps_),
    fixed(fixed_assignments)
{}

template <typename RMPType>
PricingEngine::Result runPricing(const PricingEngine& self, const RMPType& rmp)
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

#pragma omp parallel
    {
        double best_rc_local = 0.0;
        QuadKey best_col_local;
        std::vector<std::pair<QuadKey,double>> most_neg_local;
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

                    double C1ij = rmp.getRowDualC1(i,j);
                    double C1ji = (i != j ? rmp.getRowDualC1(j,i) : 0.0);

                    for (int vj = 0; vj < lenVj; ++vj) {
                        int v = Vj[vj];

                        if ((i == j) ^ (u == v)) continue;
                        if (i > j || (i == j && u > v)) continue;

                        if (rmp.hasColumn(i,u,j,v)) continue;

                        double rc =
                            (i != j ?
                             rmp.phiIUJV(i,u,j,v) + rmp.phiIUJV(j,v,i,u) :
                             rmp.phiIUJV(i,u,j,v));

                        rc -= C1ij + C1ji;
                        rc -= rmp.getRowDualC2(u,v);
                        if (u != v) rc -= rmp.getRowDualC2(v,u);
                        rc -= rmp.getRowDualC3(u,j);
                        if (i != j && u != v) rc -= rmp.getRowDualC3(v,i);
                        rc -= rmp.getRowDualC4(i,v);
                        if (i != j && u != v) rc -= rmp.getRowDualC4(j,u);
                        rc -= rmp.getRowDualC5(i,u,j);
                        if (i != j && u != v) rc -= rmp.getRowDualC5(j,v,i);
                        rc -= rmp.getRowDualC6(i,u,v);
                        if (i != j && u != v) rc -= rmp.getRowDualC6(j,v,u);

                        if (rc < best_rc_local) {
                            best_rc_local = rc;
                            best_col_local = {i,u,j,v};
                            most_neg_local.clear();
                            most_neg_local.emplace_back(best_col_local, rc);
                        } else if (rc < best_rc_local + eps) {
                            most_neg_local.emplace_back(QuadKey{i,u,j,v}, rc);
                        }

                        if (rc < -eps) neg_local.emplace_back(QuadKey{i,u,j,v}, rc);
                    }
                }
            }
        }

#pragma omp critical
        {
            if (best_rc_local < R.best_rc) {
                R.best_rc = best_rc_local;
                R.best_col = best_col_local;
                R.most_negative_cols = most_neg_local;
            } else if (best_rc_local < R.best_rc + eps) {
                R.most_negative_cols.insert(
                    R.most_negative_cols.end(),
                    most_neg_local.begin(),
                    most_neg_local.end());
            }
            R.negative_cols.insert(
                R.negative_cols.end(),
                neg_local.begin(),
                neg_local.end());
        }
    }

    return R;
}

PricingEngine::Result PricingEngine::price(const IncrementalRMP& rmp) {
    return runPricing(*this, rmp);
}

PricingEngine::Result PricingEngine::price(const PerturbedRMPView& rmp) {
    return runPricing(*this, rmp);
}