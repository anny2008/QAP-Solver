import numpy as np
from scipy import sparse

def build_DD(n, blksz):
    d = blksz * n + 1
    m = (blksz * n) ** 2

    rows = []
    cols = []
    data = []

    l = 0
    for iii in range(blksz * n):
        for jjj in range(blksz * n):
            rows.append(l)
            cols.append(jjj * d + iii)
            data.append(1.0)
            l += 1

    DD = sparse.csr_matrix((data, (rows, cols)),
                           shape=(m, d * d))
    return DD