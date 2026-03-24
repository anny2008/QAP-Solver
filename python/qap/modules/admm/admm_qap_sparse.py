import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def to_ndarray(X):
    """Convert csr_matrix or numpy.matrix to ndarray safely."""
    if hasattr(X, "toarray"):   # sparse
        return X.toarray()
    if isinstance(X, np.matrix):
        return np.asarray(X)
    return X     # already ndarray

def ADMM_QAPs(L, Vhat, J, opts):
    """
    True sparse version of ADMM for the SDP relaxation of QAP.
    - Y, Z, W remain sparse
    - R stays small (dense), size (k × k) where k = Vhat.shape[1]
    - VRV becomes sparse via Vhat * R * Vhat.T
    - WVhat becomes small dense matrix (required)
    """

    # ===========================================================
    # 1. Read parameters
    # ===========================================================
    maxit   = opts.get("maxit", 500)
    tol     = opts.get("tol", 1e-1)
    beta    = opts.get("beta", 50)
    low_rank = opts.get("low_rank", 0)
    K       = opts.get("K", 1)
    gamma   = opts.get("gamma", 1)
    cal_bd  = opts.get("cal_bd", 0)

    R = opts["R0"]                        # dense (k × k)
    Y = sp.csr_matrix(opts["Y0"])        # sparse (N × N) where N = n²+1
    Z = sp.csr_matrix(opts["Z0"])        # sparse (N × N)

    nrm_pR = 1
    nrm_dR = 1

    N = Vhat.shape[0]                    # full dimension = n²+1
    k = Vhat.shape[1]                    # reduced dimension = (n−1)²+1

    # Ensure Vhat is sparse
    if not sp.issparse(Vhat):
        Vhat = sp.csr_matrix(Vhat)

    # Ensure L is sparse
    if not sp.issparse(L):
        L = sp.csr_matrix(L)

    # For fast projection
    J_mask = J

    obj  = np.zeros(maxit)
    feas = np.zeros(maxit)
    pr   = np.zeros(maxit)
    dr   = np.zeros(maxit)

    start_time = 0.0

    # ===========================================================
    # 2. ADMM Iterations
    # ===========================================================
    for it in range(maxit):

        # -------------------------------------------
        # Step 1: R-update  (eigen projection)
        # -------------------------------------------
        # W = Y + (Z / beta)
        W = Y + (1.0 / beta) * Z          # sparse

        # WVhat = Vhat.T * W * Vhat  →  (k × k)
        WVhat = (Vhat.T @ (W @ Vhat))

        # convert WVhat to dense for eigendecomposition
        WVhat = WVhat.toarray()
        WVhat = 0.5 * (WVhat + WVhat.T)   # symmetrize

        if not low_rank:
            # Full PSD projection
            Svals, U = np.linalg.eigh(WVhat)
            idx = np.where(Svals > 0)[0]

            if len(idx) > 0:
                R = U[:, idx] @ np.diag(Svals[idx]) @ U[:, idx].T
            else:
                R = np.zeros((k, k))

        else:
            # Rank-K PSD projection via sparse eigs
            vals, vecs = spla.eigs(sp.csr_matrix(WVhat), k=K, which='LR')
            vals = np.real(vals)
            vecs = np.real(vecs)

            idx = vals > 0
            if np.any(idx):
                R = vecs[:, idx] @ np.diag(vals[idx]) @ vecs[:, idx].T
            else:
                R = np.zeros((k, k))

        # Symmetrize small R
        R = 0.5 * (R + R.T)

        # -------------------------------------------
        # Step 2: Y-update  (projection onto feasible set)
        # -------------------------------------------
        # VRV = Vhat * R * Vhat.T   → big sparse matrix
        VRV = (Vhat @ R) @ Vhat.T     # dense→sparse multiplications OK

        # Convert result to sparse
        VRV = sp.csr_matrix(VRV)

        # Y = VRV − (L + Z)/beta
        Y = VRV - (L + Z) * (1.0 / beta)
        Y = 0.5 * (Y + Y.T)           # enforce symmetry

        # Enforce gangster constraints
        Y = Y.multiply(~J_mask)       # zero entries
        Y = Y.tolil()
        Y[0, 0] = 1                    # Y(1,1) = 1
        Y = Y.maximum(0).minimum(1)    # clamp 0 ≤ Y ≤ 1
        Y = sp.csr_matrix(Y)

        # -------------------------------------------
        # Step 3: Z-update
        # -------------------------------------------
        pR = Y - VRV                   # sparse
        dR = Y - opts["Y0"]            # sparse
        opts["Y0"] = Y.copy()

        Z = Z + gamma * beta * pR
        Z = 0.5 * (Z + Z.T)

        # -------------------------------------------
        # Diagnostics
        # -------------------------------------------
        # Convert for norms
        pR_dense = to_ndarray(pR)
        dR_dense = to_ndarray(dR)
        Y_dense  = to_ndarray(Y)

        nrm_pR = np.linalg.norm(pR_dense)
        nrm_dR = beta * np.linalg.norm(dR_dense)

        feas[it] = nrm_pR / np.linalg.norm(Y_dense)
        pr[it]   = nrm_pR
        dr[it]   = nrm_dR

        # objectives <L,Y>
        obj[it] = (L.multiply(Y)).sum()

        # stopping test
        if nrm_pR < tol and nrm_dR < tol:
            break
        print(f"Iteration {it}: obj={obj[it]:.4f}")

    # ===========================================================
    # 3. Export outputs
    # ===========================================================
    Out = {
        "obj": obj[:it+1],
        "iter": it+1,
        "feas": feas[:it+1],
        "pr": pr[:it+1],
        "dr": dr[:it+1],
        "R": R,
        "Y": Y,
        "Z": Z
    }
    return R, Y, Out