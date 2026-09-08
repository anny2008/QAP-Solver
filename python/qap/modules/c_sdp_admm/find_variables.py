import numpy as np

def find_pairs(B):

    iu = np.triu_indices_from(B, k=1)

    mask = B[iu] != 0

    return np.column_stack(
        [
            iu[0][mask],
            iu[1][mask]
        ]
    )

from itertools import combinations
import numpy as np

def merge_variables_aux(var1, var2):

    all_nodes = np.unique(
        np.concatenate([var1, var2])
    )

    k = len(var1)

    return np.array(
        list(combinations(all_nodes, k + 1))
    )

def merge_variables(varlist):

    n_vars = len(varlist)

    new_vars = []

    for i in range(n_vars - 1):

        cur_var = varlist[i]

        common = -np.ones(n_vars)

        for j in range(i + 1, n_vars):

            common[j] = np.intersect1d(
                cur_var,
                varlist[j]
            ).size

        j_best = np.argmax(common)

        merged = merge_variables_aux(
            cur_var,
            varlist[j_best]
        )

        new_vars.extend(merged)

    new_vars = np.asarray(new_vars)

    return np.unique(new_vars, axis=0)

def find_variables(B, k):

    B = B + B.T

    varlist = find_pairs(B)

    for _ in range(k - 2):
        varlist = merge_variables(varlist)

    return varlist