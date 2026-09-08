import numpy as np
from scipy import sparse

def fro_normalize(M):
    nrm = np.linalg.norm(M, "fro")
    return M.reshape(-1, order="F") / nrm, nrm


def build_Ae(n, blksz):

    n_cons = blksz * (blksz + 1) + 1 + 2 * blksz * n

    d = blksz * n + 1

    Ae = []
    be = []

    #
    # bottom-right corner = 1
    #
    temp = np.zeros((d, d))
    temp[-1, -1] = 1.0

    Ae.append(temp.reshape(-1, order="F"))
    be.append(1.0)

    #
    # diagonal blocks:
    # off-diagonal sum = 0
    #
    for blk in range(blksz):

        idx = slice(blk * n, (blk + 1) * n)

        temp = np.zeros((d, d))
        temp[idx, idx] = np.ones((n, n)) - np.eye(n)

        vec, nrm = fro_normalize(temp)

        Ae.append(vec)
        be.append(0.0)

    #
    # diagonal blocks:
    # diagonal sum = 1
    #
    for blk in range(blksz):

        idx = slice(blk * n, (blk + 1) * n)

        temp = np.zeros((d, d))
        temp[idx, -1] = 1

        vec, nrm = fro_normalize(temp)

        Ae.append(vec)
        be.append(1.0 / nrm)

    #
    # off-diagonal blocks:
    # diagonal entries sum to 0
    #
    for i in range(blksz):

        idx_i = slice(i * n, (i + 1) * n)

        for j in range(i + 1, blksz):

            idx_j = slice(j * n, (j + 1) * n)

            temp = np.zeros((d, d))

            temp[idx_i, idx_j] = np.eye(n)
            temp = temp + temp.T

            vec, nrm = fro_normalize(temp)

            Ae.append(vec)
            be.append(0.0)

    #
    # off-diagonal blocks:
    # off-diagonal entries sum to 1
    #
    for i in range(blksz):

        idx_i = slice(i * n, (i + 1) * n)

        for j in range(i + 1, blksz):

            idx_j = slice(j * n, (j + 1) * n)

            temp = np.zeros((d, d))

            temp[idx_i, idx_j] = np.ones((n, n)) - np.eye(n)
            temp = temp + temp.T

            vec, nrm = fro_normalize(temp)

            Ae.append(vec)
            be.append(2.0 / nrm)

    #
    # diagonal(Q)=u,v constraints
    #
    for iii in range(blksz * n):

        temp = np.zeros((d, d))

        temp[iii, iii] = 1
        temp[iii, -1] = -0.5
        temp[-1, iii] = -0.5

        vec, nrm = fro_normalize(temp)

        Ae.append(vec)
        be.append(0.0)

    #
    # symmetry constraints
    #
    for iii in range(blksz * n):

        temp = np.zeros((d, d))

        temp[iii, -1] = 1
        temp[-1, iii] = -1

        vec, nrm = fro_normalize(temp)

        Ae.append(vec)
        be.append(0.0)

    Ae = sparse.csr_matrix(np.vstack(Ae))
    be = np.asarray(be)

    invAeAeT = np.linalg.inv((Ae @ Ae.T).toarray())

    return Ae, invAeAeT, be