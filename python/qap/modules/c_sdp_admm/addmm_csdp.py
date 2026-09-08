# admm_c_sdp.py

import numpy as np

from .build_Ae import build_Ae
from .build_BB import build_BB
from .build_DD import build_DD

from .find_variables import find_variables
from .find_neighbours import find_neighbours


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def project_pos_cone(X):
    return np.maximum(X, 0.0)


def project_psd_cone(x):

    d = int(np.sqrt(len(x)))

    X = x.reshape((d, d), order="F")

    eigvals, eigvecs = np.linalg.eigh(X)

    eigvals[eigvals < 0] = 0.0

    Xp = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return Xp.reshape(-1, order="F")


def col_multiply(A_col, B_col):

    R = A_col[0].T @ B_col[0]

    for i in range(1, len(A_col)):
        R += A_col[i].T @ B_col[i]

    return R


def matpart(A, dim, vec):

    parts = []

    start = 0

    if dim == 2:

        for length in vec:
            parts.append(
                A[:, start:start + length]
            )
            start += length

    else:

        for length in vec:
            parts.append(
                A[start:start + length, :]
            )
            start += length

    return parts


# ----------------------------------------------------------------------
# Main Solver
# ----------------------------------------------------------------------

class CSDPADMM:

    def __init__(
        self,
        clique_size=2,
        max_iter=100,
        tol=1e-6,
        tau=1.568,
    ):
        self.clique_size = clique_size
        self.max_iter = max_iter
        self.tol = tol
        self.tau = tau

    def solve(self, A, B, P0=None):

        n = A.shape[0]

        if P0 is None:
            P0 = np.eye(n)

        # -------------------------------------------------------------
        # Normalize
        # -------------------------------------------------------------

        nA = np.linalg.norm(A)
        nB = np.linalg.norm(B)

        A = A / nA
        B = B / nB

        # -------------------------------------------------------------
        # Variables
        # -------------------------------------------------------------

        varlist = find_variables(
            B,
            self.clique_size
        )

        neighbors = find_neighbours(
            varlist,
            n
        )

        BLKSZ = varlist.shape[1]
        n_vars = varlist.shape[0]

        d = BLKSZ * n + 1

        print(f"n={n}")
        print(f"variables={n_vars}")
        print(f"block size={BLKSZ}")

        # -------------------------------------------------------------
        # Constraints
        # -------------------------------------------------------------

        Ae, invAeAeT, be = build_Ae(
            n,
            BLKSZ
        )

        BB_col = build_BB(
            n,
            BLKSZ
        )

        DD = build_DD(
            n,
            BLKSZ
        )

        # -------------------------------------------------------------
        # Cost
        # -------------------------------------------------------------

        tempC0 = np.kron(B, A)
        count = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                for k in range(n_vars):

                    row = varlist[k]

                    if (i in row) and (j in row):
                        count[i, j] += 1
        count[count == 0] = 1

        tempC = tempC0 / np.kron(
            count,
            np.ones((n, n))
        )

        C = np.zeros((d * d, n_vars))

        for k in range(n_vars):

            idx = []

            for node in varlist[k]:
                idx.extend(
                    range(
                        node * n,
                        (node + 1) * n
                    )
                )

            c = np.zeros((d, d))

            c[:-1, :-1] = tempC[
                np.ix_(idx, idx)
            ]

            c = 0.5 * (c + c.T)

            C[:, k] = c.reshape(
                -1,
                order="F"
            )

        # -------------------------------------------------------------
        # Initialize
        # -------------------------------------------------------------

        X = np.zeros((d * d, n_vars))

        p = P0.reshape(-1, order="F")

        for k in range(n_vars):

            idx = []

            for node in varlist[k]:
                idx.extend(
                    range(
                        node * n,
                        (node + 1) * n
                    )
                )

            ptemp = np.concatenate(
                [p[idx], [1.0]]
            )

            X[:, k] = np.outer(
                ptemp,
                ptemp
            ).reshape(-1, order="F")

        S = np.zeros_like(X)

        Z = np.zeros(
            ((BLKSZ * n) ** 2, n_vars)
        )

        Y = np.zeros(
            (Ae.shape[0], n_vars)
        )

        G = np.zeros((n, n))

        W_col = [
            np.zeros((n, n_vars))
            for _ in range(BLKSZ)
        ]

        BBtW = col_multiply(
            BB_col,
            W_col
        )

        # -------------------------------------------------------------
        # Precompute c_tilde_w
        # -------------------------------------------------------------

        c_tilde_w = []

        for node in range(n):

            nb = []

            for k in range(BLKSZ):
                nb.extend(
                    neighbors[k][node]
                )

            psize = len(nb)

            alpha = 1.5
            beta = 1.0

            x = 1.0 / (alpha - beta)

            y = (
                -beta
                / (alpha - beta)
                / (alpha - beta + psize * beta)
            )

            M = (
                x * np.eye(psize)
                + y * np.ones((psize, psize))
            )

            c_tilde_w.append(M)

        # -------------------------------------------------------------
        # ADMM
        # -------------------------------------------------------------

        rho = 1.0 / n

        for it in range(self.max_iter):

            # -------------------------------------------------
            # S update
            # -------------------------------------------------

            for i in range(n_vars):

                rhs = (
                    C[:, i]
                    - DD.T @ Z[:, i]
                    - Ae.T @ Y[:, i]
                    - BBtW[:, i]
                    - (1.0 / rho) * X[:, i]
                )

                S[:, i] = project_psd_cone(rhs)

            # -------------------------------------------------
            # T update
            # -------------------------------------------------

            Wsum = np.sum(
                np.hstack(W_col),
                axis=1
            )

            T = (
                (
                    Wsum
                    - (1.0 / rho)
                    * np.sum(G, axis=1)
                )
                / n
                + 1.0 / (rho * n)
            )

            # -------------------------------------------------
            # Y update
            # -------------------------------------------------

            rhs = (
                Ae @ (
                    C
                    - S
                    - DD.T @ Z
                    - BBtW
                    - (1.0 / rho) * X
                )
                + (1.0 / rho)
                * be[:, None]
            )

            Y = invAeAeT @ rhs

            # -------------------------------------------------
            # Z update
            # -------------------------------------------------

            Z = project_pos_cone(
                DD @ (
                    C
                    - S
                    - Ae.T @ Y
                    - (1.0 / rho) * X
                )
            )

            dMat = (
                C
                - S
                - Ae.T @ Y
                - (1.0 / rho) * X
            )

            # -------------------------------------------------
            # W and G updates
            # -------------------------------------------------

            for node in range(n):

                vec = [
                    len(neighbors[k][node])
                    for k in range(BLKSZ)
                ]

                nb_size = sum(vec)

                if nb_size == 0:
                    continue

                c_tilde = np.zeros(
                    (n, nb_size)
                )

                count = 0

                for k in range(BLKSZ):

                    ids = neighbors[k][node]

                    m = len(ids)

                    if m > 0:

                        c_tilde[:, count:count+m] = (
                            BB_col[k]
                            @ dMat[:, ids]
                        )

                    count += m

                c_tilde += (
                    T[:, None]
                    + (1.0 / rho)
                    * G[:, node, None]
                )

                res = (
                    c_tilde
                    @ c_tilde_w[node]
                )

                parts = matpart(
                    res,
                    2,
                    vec
                )

                for k in range(BLKSZ):

                    ids = neighbors[k][node]

                    if len(ids):
                        W_col[k][:, ids] = parts[k]

                G[:, node] += (
                    self.tau
                    * rho
                    * (
                        T
                        - np.sum(res, axis=1)
                    )
                )

            BBtW = col_multiply(
                BB_col,
                W_col
            )

            # -------------------------------------------------
            # second Y update
            # -------------------------------------------------

            rhs = (
                Ae @ (
                    C
                    - S
                    - DD.T @ Z
                    - BBtW
                    - (1.0 / rho) * X
                )
                + (1.0 / rho)
                * be[:, None]
            )

            Y = invAeAeT @ rhs

            # -------------------------------------------------
            # X update
            # -------------------------------------------------

            X += (
                self.tau
                * rho
                * (
                    -C
                    + S
                    + DD.T @ Z
                    + Ae.T @ Y
                    + BBtW
                )
            )

            primal = np.linalg.norm(
                Ae @ X - be[:, None]
            )

            dual = np.linalg.norm(
                -C
                + Ae.T @ Y
                + S
                + DD.T @ Z
                + BBtW
            )

            print(
                f"it={it:4d} "
                f"primal={primal:.3e} "
                f"dual={dual:.3e}"
            )

            if max(primal, dual) < self.tol:
                break
        
        print("sum count =", np.sum(count))
        print("count unique =", np.unique(count))

        print("tempC before", np.sum(np.abs(np.kron(B,A))))
        print("tempC after ", np.sum(np.abs(tempC)))

        count = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                for k in range(n_vars):

                    row = varlist[k]

                    if (i in row) and (j in row):
                        count[i,j] += 1

        print(count)

        print("k =", self.clique_size)
        print("edges =", np.sum(B > 0) // 2)
        print("n_vars =", len(varlist))


        print("objective =", np.sum(C * X))
        print("objective_S =", np.sum(C * S))

        print("nA =", nA)
        print("nB =", nB)
        print("nA*nB =", nA*nB)
        print("objective raw =", np.sum(C * X))
        print("objective scaled =", np.sum(C * X) * nA * nB)

        print("raw objective =", np.sum(C * X))

        print("raw objective * (-1) =", -np.sum(C * X))

        print(varlist[:10])
        print(tempC.shape)
        print(C.shape)

        print(np.min(C))
        print(np.max(C))

        optval = np.sum(C * X) * nA * nB

        print("SDP lower bound =", optval)

        return {
            "X": X,
            "Y": Y,
            "Z": Z,
            "S": S,
            "primal": primal,
            "dual": dual,
        }