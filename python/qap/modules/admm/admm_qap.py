import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eig, qr
from scipy.optimize import linprog

def ADMM_QAP(L, Vhat, J, opts):
    maxit = opts.get("maxit", 500)
    tol   = opts.get("tol", 1e-1)
    beta  = opts.get("beta", 50)
    low_rank = opts.get("low_rank", 0)
    K = opts.get("K", 1)
    gamma = opts.get("gamma", 1)
    cal_bd = opts.get("cal_bd", 0)

    if cal_bd:
        D = opts["B"]
        F = opts["A"]

    # Ensure symmetry
    if np.linalg.norm(L - L.T, 'fro') > 10 * np.finfo(float).eps:
        L = 0.5 * (L + L.T)

    R = opts["R0"]
    Y = opts["Y0"]
    Z = opts["Z0"]

    n2 = Vhat.shape[0] - 1
    n = int(np.sqrt(n2))
    In = np.eye(n)
    en = np.ones((n, 1))

    nrmL = np.linalg.norm(L, 'fro')
    L = (L / nrmL) * n2

    feas = np.zeros(maxit)
    obj = np.zeros(maxit)
    hist_pr = np.zeros(maxit)
    hist_dr = np.zeros(maxit)
    pos_eig = np.zeros(maxit)

    fVhat = Vhat

    for iter in range(maxit):
        # Step 1: update R
        W = Y + Z / beta
        WVhat = fVhat.T @ (W @ fVhat)
        WVhat = 0.5 * (WVhat + WVhat.T)

        # eigen-decomposition
        if not low_rank:
            Svals, U = np.linalg.eigh(WVhat)      # WVhat is 122×122
            idx = np.where(Svals > 0)[0]

            if len(idx) > 0:
                R = U[:, idx] @ np.diag(Svals[idx]) @ U[:, idx].T    # 122×122
            else:
                R = np.zeros((Vhat.shape[1], Vhat.shape[1]))          # 122×122
            pos_eig[iter] = len(idx)
        else:
            # rank-1 projection
            Svals, U = np.linalg.eigh(WVhat)
            if Svals[-1] > 0:
                u = U[:, [-1]]
                temp = fVhat @ u
                R = Svals[-1] * (temp @ temp.T)
            else:
                R = np.zeros_like(W)

        R = 0.5 * (R + R.T)
        # print("Vhat:", Vhat.shape)
        # print("R:", R.shape)
        # print("Vhat.T:", Vhat.T.shape)
        # Step 2: update Y
        VRV = fVhat @ R @ fVhat.T
        Y = VRV - (L + Z) / beta
        Y = 0.5 * (Y + Y.T)

        Y[J] = 0
        Y[0, 0] = 1
        Y = np.clip(Y, 0, 1)
        Y[J] = 0
        Y[0, 0] = 1

        # Step 3: update Z
        pR = Y - VRV
        dR = Y - opts["Y0"]
        opts["Y0"] = Y.copy()

        Z = Z + gamma * beta * pR
        Z = 0.5 * (Z + Z.T)

        nrm_pR = np.linalg.norm(pR, 'fro')
        nrm_dR = beta * np.linalg.norm(dR, 'fro')
        hist_pr[iter] = nrm_pR
        hist_dr[iter] = nrm_dR
        obj[iter] = np.sum(L * Y)
        feas[iter] = nrm_pR / np.linalg.norm(Y, 'fro')

        if nrm_pR < tol and nrm_dR < tol:
            break
        print(f"Iteration {iter}: obj={obj[iter]:.4f}")

    Out = {
        "obj": obj[:iter+1] * nrmL / n2,
        "iter": iter+1,
        "pr": hist_pr[:iter+1],
        "dr": hist_dr[:iter+1],
        "feas": feas[:iter+1],
        "Z": Z * nrmL / n2
    }

    return R, Y, Out