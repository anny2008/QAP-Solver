// StabilizedDuals.h
#pragma once
#include "IncrementalRMP.h"
#include <vector>
#include <cassert>

// Dense dual containers for the row families used in PricingEngine
// Indexing matches how PricingEngine queries duals:
//   C1(i,j), C2(u,v), C3(u,j), C4(i,v), C5(i,u,j), C6(i,u,v)
struct DenseDuals {
    int n{0}, m{0};
    std::vector<double> C1;      // n*n
    std::vector<double> C2;      // m*m
    std::vector<double> C3;      // m*n
    std::vector<double> C4;      // n*m
    std::vector<double> C5;      // n*m*n
    std::vector<double> C6;      // n*m*m

    DenseDuals() = default;
    DenseDuals(int n_, int m_) { reset(n_, m_); }

    void reset(int n_, int m_) {
        n = n_; m = m_;
        C1.assign(n*n, 0.0);
        C2.assign(m*m, 0.0);
        C3.assign(m*n, 0.0);
        C4.assign(n*m, 0.0);
        C5.assign(n*m*n, 0.0);
        C6.assign(n*m*m, 0.0);
    }

    // Flat index helpers
    inline int idxC1(int i,int j) const { return i*n + j; }
    inline int idxC2(int u,int v) const { return u*m + v; }
    inline int idxC3(int u,int j) const { return u*n + j; }
    inline int idxC4(int i,int v) const { return i*m + v; }
    inline int idxC5(int i,int u,int j) const { return ( (i*m + u)*n + j ); }
    inline int idxC6(int i,int u,int v) const { return ( (i*m + u)*m + v ); }

    // Accessors
    inline double getC1(int i,int j) const { return C1[idxC1(i,j)]; }
    inline double getC2(int u,int v) const { return C2[idxC2(u,v)]; }
    inline double getC3(int u,int j) const { return C3[idxC3(u,j)]; }
    inline double getC4(int i,int v) const { return C4[idxC4(i,v)]; }
    inline double getC5(int i,int u,int j) const { return C5[idxC5(i,u,j)]; }
    inline double getC6(int i,int u,int v) const { return C6[idxC6(i,u,v)]; }

    // Capture current duals from the RMP
    void captureFrom(const IncrementalRMP& rmp) {
        assert(n>0 && m>0);
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) C1[idxC1(i,j)] = rmp.getRowDualC1(i,j);
        for (int u=0;u<m;++u) for (int v=0;v<m;++v) C2[idxC2(u,v)] = rmp.getRowDualC2(u,v);
        for (int u=0;u<m;++u) for (int j=0;j<n;++j) C3[idxC3(u,j)] = rmp.getRowDualC3(u,j);
        for (int i=0;i<n;++i) for (int v=0;v<m;++v) C4[idxC4(i,v)] = rmp.getRowDualC4(i,v);
        for (int i=0;i<n;++i) for (int u=0;u<m;++u) for (int j=0;j<n;++j)
            C5[idxC5(i,u,j)] = rmp.getRowDualC5(i,u,j);
        for (int i=0;i<n;++i) for (int u=0;u<m;++u) for (int v=0;v<m;++v)
            C6[idxC6(i,u,v)] = rmp.getRowDualC6(i,u,v);
    }

    // this <- lambda * cur + (1-lambda) * this
    void emaUpdate(const DenseDuals& cur, double lambda) {
        const auto blend = [&](std::vector<double>& tgt, const std::vector<double>& src){
            const size_t L = tgt.size();
            for (size_t k=0;k<L;++k) tgt[k] = lambda*src[k] + (1.0 - lambda)*tgt[k];
        };
        blend(C1, cur.C1); blend(C2, cur.C2); blend(C3, cur.C3);
        blend(C4, cur.C4); blend(C5, cur.C5); blend(C6, cur.C6);
    }
};

// A lightweight view that forwards everything to the underlying RMP
// except dual getters, which are overridden by DenseDuals above.
class PerturbedRMPView {
public:
    PerturbedRMPView(const IncrementalRMP& base, const DenseDuals& dd)
        : rmp(base), du(dd) {}

    // Forwarders needed by PricingEngine
    inline bool hasColumn(int i,int u,int j,int v) const { return rmp.hasColumn(i,u,j,v); }
    inline double phiIUJV(int i,int u,int j,int v) const { return rmp.phiIUJV(i,u,j,v); }

    // Overridden dual getters
    inline double getRowDualC1(int i,int j) const { return du.getC1(i,j); }
    inline double getRowDualC2(int u,int v) const { return du.getC2(u,v); }
    inline double getRowDualC3(int u,int j) const { return du.getC3(u,j); }
    inline double getRowDualC4(int i,int v) const { return du.getC4(i,v); }
    inline double getRowDualC5(int i,int u,int j) const { return du.getC5(i,u,j); }
    inline double getRowDualC6(int i,int u,int v) const { return du.getC6(i,u,v); }

private:
    const IncrementalRMP& rmp;
    const DenseDuals& du;
};