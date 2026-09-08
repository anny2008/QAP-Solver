import numpy as np
from scipy import sparse

def build_BB(n, blksz):
    d = blksz * n + 1

    BB_col = []

    for j in range(blksz):

        rows = []
        cols = []
        vals = []

        for iii in range(n):

            r = iii

            idx = iii + j * n

            # temp(idx,d)=1/2
            cols.append((d - 1) * d + idx)
            rows.append(r)
            vals.append(0.5)

            # temp(d,idx)=1/2
            cols.append(idx * d + (d - 1))
            rows.append(r)
            vals.append(0.5)

        BB = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n, d * d)
        )

        BB_col.append(BB)

    return BB_col