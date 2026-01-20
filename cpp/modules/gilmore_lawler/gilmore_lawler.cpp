#include "gilmore_lawler.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <bits/stdc++.h>

/*
  Hungarian algorithm (O(n^3)) for the min-cost assignment on a square matrix.
  Returns (min_cost, assignment), where assignment[i] = assigned column for row i (0-based).
  Cost type: double (works for nonnegative / moderate magnitudes).
*/
std::pair<double, std::vector<int>> hungarian_min(const std::vector<std::vector<double>>& a) {
    int n = (int)a.size();
    // 1-based arrays in the classic implementation
    std::vector<double> u(n + 1, 0), v(n + 1, 0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        double j0 = 0;
        std::vector<double> minv(n + 1, LLONG_MAX);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = LLONG_MAX;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = (int)j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        // Augmentation
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] != 0) assignment[p[j] - 1] = j - 1;
    }
    double min_cost = -v[0];
    return {min_cost, assignment};
}

/*
  Build the GLB cost matrix C for QAP(F, D).

  For every facility i and location k:
    1) Let A_i be the multiset { f_{i,j} : j != i } (row i of F without the diagonal),
       sorted ascending.
    2) Let B_k be the multiset { d_{k,l} : l != k } (row k of D without the diagonal),
       sorted descending.
    3) Define c_{i,k} = f_{ii}*d_{kk} + sum_{r=1}^{n-1} A_i^{↑}[r] * B_k^{↓}[r].
  Then GLB = min assignment on C.
*/

GLBResult GilmoreLawler::compute(const std::vector<std::vector<double>>& F,
                               const std::vector<std::vector<double>>& D) {
    int n = (int)F.size();
    if (n == 0 || (int)F[0].size() != n || (int)D.size() != n || (int)D[0].size() != n) {
        throw std::runtime_error("F and D must be square matrices of the same dimension.");
    }

    // Precompute sorted rows (excluding diagonals):
    // ascF[i] = sorted ascending list of { F[i][j] : j != i }
    // descD[k] = sorted descending list of { D[k][l] : l != k }
    std::vector<std::vector<double>> ascF(n), descD(n);
    ascF.reserve(n); descD.reserve(n);

    for (int i = 0; i < n; ++i) {
        ascF[i].reserve(n - 1);
        for (int j = 0; j < n; ++j) if (j != i) ascF[i].push_back(F[i][j]);
        std::sort(ascF[i].begin(), ascF[i].end()); // ascending
    }

    for (int k = 0; k < n; ++k) {
        descD[k].reserve(n - 1);
        for (int l = 0; l < n; ++l) if (l != k) descD[k].push_back(D[k][l]);
        sort(descD[k].begin(), descD[k].end(), std::greater<double>()); // descending
    }

    // Build C
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            __int128 acc = 0;
            // include potential diagonal self-term if present
            acc += (__int128)F[i][i] * (__int128)D[k][k];
            // rearrangement-based sum of off-diagonals
            for (int r = 0; r < n - 1; ++r) {
                acc += (__int128)ascF[i][r] * (__int128)descD[k][r];
            }
            // clamp to double (assuming values fit)
            double val;
            if (acc > (__int128)LLONG_MAX) val = LLONG_MAX;
            else if (acc < (__int128)LLONG_MIN) val = LLONG_MIN;
            else val = (double)acc;
            C[i][k] = val;
        }
    }

    // Solve LAP on C
    auto [glb_value, assign] = hungarian_min(C);

    return GLBResult{glb_value, assign, C};
}
