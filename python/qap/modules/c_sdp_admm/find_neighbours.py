import numpy as np

def find_neighbours(varlist, n):
    """
    varlist: (n_vars, k)

    returns:
        neighbours[k][node]
    """

    n_vars, k = varlist.shape

    neighbours = []

    idx = np.arange(n_vars)

    for i in range(k):

        level = []

        for node in range(1, n + 1):

            level.append(
                idx[varlist[:, i] == node]
            )

        neighbours.append(level)

    return neighbours
