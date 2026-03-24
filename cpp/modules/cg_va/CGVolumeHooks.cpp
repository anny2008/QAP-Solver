#include "CGVolumeHooks.h"

#include <limits>
#include <vector>

CGVolumeHooks::CGVolumeHooks(const Problem &prob,
                             const std::unordered_map<QuadKey, int, QuadKeyHash> &y_index_map)
    : P(prob), ymap(y_index_map) {}

int CGVolumeHooks::compute_rc(const VOL_dvector& /*u*/, VOL_dvector& /*rc*/) {
    return 0;
}

int CGVolumeHooks::heuristics(const VOL_problem& /*p*/, const VOL_dvector& x, double& heur_val) {
    const int n = P.n;
    std::vector<int> assign(n, -1);
    std::vector<char> used(n, 0);

    // Greedy permutation from x-block (first n*n entries): pick largest unused x(i,u)
    for (int i = 0; i < n; ++i) {
        int best_u = -1;
        double best_val = -std::numeric_limits<double>::infinity();
        for (int u = 0; u < n; ++u) {
            if (used[u]) continue;
            double val = x[i * n + u];
            if (val > best_val) {
                best_val = val;
                best_u = u;
            }
        }
        if (best_u < 0) {
            for (int u = 0; u < n; ++u) {
                if (!used[u]) {
                    best_u = u;
                    break;
                }
            }
        }
        assign[i] = best_u;
        used[best_u] = 1;
    }

    double obj = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            obj += P.D[i][j] * P.F[assign[i]][assign[j]];
        }
    }

    heur_val = obj;
    return 0;
}

int CGVolumeHooks::solve_subproblem(
                    const VOL_dvector& pi,
                    const VOL_dvector& /*rc*/,
                    double& lcost,
                    VOL_dvector& psol,
                    VOL_dvector& vio,
                    double& pcost)
{
    const int n = P.n;

    // =============================
    // 1) Initialize
    // =============================
    psol = 0.0;
    vio  = 0.0;

    // =============================
    // 2) Dual layout (dense α,β,γ,δ,η,μ)
    // =============================
    const int off_alpha = 0;                  // α(i,j): n*n (use only i<j)
    const int off_beta  = off_alpha + n*n;    // β(u,v): n*n (use only u!=v)
    const int off_gamma = off_beta  + n*n;    // γ(u,j): n*n
    const int off_delta = off_gamma + n*n;    // δ(i,v): n*n
    const int off_eta   = off_delta + n*n;    // η(i,u,j): n*n*n
    const int off_mu    = off_eta   + n*n*n;  // μ(i,u,v): n*n*n

    // -----------------------------
    // Macros for dual access
    // -----------------------------
    #define ALPHA(i,j) ( pi[ off_alpha + (i)*n + (j) ] )
    #define BETA(u,v)  ( pi[ off_beta  + (u)*n + (v) ] )
    #define GAMMA(u,j) ( pi[ off_gamma + (u)*n + (j) ] )
    #define DELTA(i,v) ( pi[ off_delta + (i)*n + (v) ] )
    #define ETA(i,u,j) ( pi[ off_eta   + ((i)*n + (u))*n + (j) ] )
    #define MU(i,u,v)  ( pi[ off_mu    + ((i)*n + (u))*n + (v) ] )

    // -----------------------------
    // Macros for primal access
    // -----------------------------
    #define X(i,u)     ( psol[ (i)*n + (u) ] )
    #define Y(k)       ( psol[ n*n + (k) ] )  // k corresponds to y_index_map


    // φ counted once for diagonal; twice for cross (symmetric)
    #define PHI_ONCE(i,u,j,v) ( P.D[(i)][(j)] * P.F[(u)][(v)] )
    #define PHI_FULL(i,u,j,v) ( ((i)==(j) && (u)==(v)) ? \
                                 PHI_ONCE(i,u,j,v) : \
                                 (PHI_ONCE(i,u,j,v) + PHI_ONCE(j,v,i,u)) )

    // =============================
    // 3) x-minimizers (independent 0-1)
    //     c_{iu} = sum_{j≠i} η_{iuj} + sum_{v≠u} μ_{iuv}
    // =============================
    std::vector<double> cx(n*n, 0.0); 
    // parallelize over i,u since independent; each cx[i*n+u] is sum of O(n) duals
    # pragma omp parallel for collapse(2) schedule(static)
    for (int i=0;i<n;i++){
        for (int u=0;u<n;u++){
            double c = 0.0;
            for (int j=0;j<n;j++) c += ETA(i,u,j);
            for (int v=0;v<n;v++) c += MU(i,u,v);
            cx[i*n+u] = c;
            if (c <= 0.0) X(i,u) = 1.0;
        }
    }

    // =============================
    // 4) y-minimizers over canonical Ω only
    //
    //   C_{iujv} = PHI_FULL(i,u,j,v)
    //            - α_{ij} - β_{uv} - γ_{uj} - δ_{iv} - η_{iuj} - μ_{iuv}
    // =============================
    auto ny = ymap.size();
    std::vector<double> Cy(ny, 0.0);
    for (const auto &kv : ymap){
        const QuadKey &q = kv.first;  // (i,u,j,v) canonical: (i<j,u!=v) or (i=j,u=v)
        int k = kv.second;
        // invalid should never be present in Ω; safe-guard:
        if ((q.i==q.j && q.u!=q.v) || (q.i!=q.j && q.u==q.v)) {
            Y(k) = 0.0; // keep off
            Cy[k] = 1e100; // large positive cost to avoid numerical issues
            continue;
        }

        double phi = PHI_FULL(q.i,q.u,q.j,q.v);
        double C   =  phi
                    - ALPHA(q.i,q.j) - ALPHA(q.j,q.i)
                    - BETA (q.u,q.v) - BETA (q.v,q.u)
                    - GAMMA(q.u,q.j) - GAMMA(q.v,q.i)
                    - DELTA(q.i,q.v) - DELTA(q.j,q.u)
                    - ETA  (q.i,q.u,q.j) - ETA  (q.j,q.v,q.i)
                    - MU   (q.i,q.u,q.v) - MU   (q.j,q.v,q.u);
        Cy[k] = C;
        if (Cy[k] <= 0.0) Y(k) = 1.0;
    }

    // =============================
    // 5) Primal cost (QAP) with symmetry
    //     diagonal once, cross twice
    // =============================
    pcost = 0.0;
    for (const auto &kv : ymap){
        const QuadKey &q = kv.first;
        int k = kv.second;
        pcost += PHI_FULL(q.i,q.u,q.j,q.v) * Y(k);
    }

    // =============================
    // 6) Subgradients (vio) for VA (exactly row-wise as in IncrementalRMP)
    //     Layout = [α (n*n)] [β (n*n)] [γ (n*n)] [δ (n*n)] [η (n^3)] [μ (n^3)]
    // =============================
    const int off_alpha_v = 0;
    const int off_beta_v  = off_alpha_v + n*n;
    const int off_gamma_v = off_beta_v  + n*n;
    const int off_delta_v = off_gamma_v + n*n;
    const int off_eta_v   = off_delta_v + n*n;
    const int off_mu_v    = off_eta_v   + n*n*n;

    auto VIO_ALPHA = [&](int i,int j) -> double& { return vio[off_alpha_v + i*n + j]; };
    auto VIO_BETA  = [&](int u,int v) -> double& { return vio[off_beta_v  + u*n + v]; };
    auto VIO_GAMMA = [&](int u,int j) -> double& { return vio[off_gamma_v + u*n + j]; };
    auto VIO_DELTA = [&](int i,int v) -> double& { return vio[off_delta_v + i*n + v]; };
    auto VIO_ETA   = [&](int i,int u,int j) -> double& { return vio[off_eta_v + ((i*n+u)*n + j)]; };
    auto VIO_MU    = [&](int i,int u,int v) -> double& { return vio[off_mu_v  + ((i*n+u)*n + v)]; };

    // RHS terms
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) {
        VIO_ALPHA(i,j) = 1.0;
        VIO_BETA (i,j) = 1.0;
        VIO_GAMMA(i,j) = 1.0;
        VIO_DELTA(i,j) = 1.0;
    }
    #pragma omp parallel for collapse(3) schedule(static)
    for (int i=0;i<n;i++) for (int u=0;u<n;u++) for (int j=0;j<n;j++) {
        const double xv = X(i,u);
        VIO_ETA(i,u,j) = xv;
        VIO_MU(i,u,j)  = xv;
    }


    // y contribution: each canonical y is attached to both orientations
    auto apply_oriented_y = [&](int i,int u,int j,int v,double yv) {
        VIO_ALPHA(i,j) -= yv;
        VIO_BETA(u,v)  -= yv;
        VIO_GAMMA(u,j) -= yv;
        VIO_DELTA(i,v) -= yv;
        VIO_ETA(i,u,j) -= yv;
        VIO_MU(i,u,v)  -= yv;
    };
    for (const auto &kv : ymap){
        const QuadKey &q = kv.first;
        int k = kv.second;
        double yv = Y(k);
        if (yv > 0.5){
            apply_oriented_y(q.i,q.u,q.j,q.v,yv);
            if (!(q.i==q.j && q.u==q.v))  // cross term: also apply other orientation
                apply_oriented_y(q.j,q.v,q.i,q.u,yv);
        }
    }
     
    // =============================
    // 7) Largrangean cost = pcost + sum(pi*vio)
    //     
    // =============================
    lcost = pcost;
    #pragma omp parallel for reduction(+:lcost) schedule(static)
    for (int idx=0; idx<pi.size(); idx++) {
        lcost += pi[idx] * vio[idx];
    }

    iteration++;
    prev_pi = pi; // store for potential debugging of dual changes
    
    return 0;
}