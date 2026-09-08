from collections import namedtuple
from functools import partial
import jax
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import numba as nb
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform  # type: ignore
from typing import Any, Dict, Tuple

from IPython import embed
from .loop import *
import argparse
from collections import namedtuple
from typing import Tuple

SDPState = namedtuple("SDPState",
                      ["C",
                       "A_indices",
                       "A_data",
                       "b",
                       "b_ineq_mask",
                       "X",
                       "P",
                       "Omega",
                       "y",
                       "z",
                       "tr_X",
                       "primal_obj",
                       "SCALE_C",
                       "SCALE_X",
                       "SCALE_A",
                       "best_ub",
                       "best_obj_gap",])


def scale_sdp_state(sdp_state: SDPState) -> SDPState:
    scaled_C = BCOO((sdp_state.C.data * sdp_state.SCALE_C, sdp_state.C.indices),
                    shape=sdp_state.C.shape)
    scaled_A_data = sdp_state.A_data * sdp_state.SCALE_A.at[sdp_state.A_indices[:,0]].get()
    scaled_b = sdp_state.b * sdp_state.SCALE_X * sdp_state.SCALE_A

    scaled_X = sdp_state.X
    scaled_P = sdp_state.P
    if sdp_state.X is not None:
        scaled_X = sdp_state.X * sdp_state.SCALE_X
    if sdp_state.P is not None:
        scaled_P = sdp_state.P * sdp_state.SCALE_X

    scaled_z = sdp_state.z * sdp_state.SCALE_A * sdp_state.SCALE_X
    scaled_tr_X = sdp_state.tr_X * sdp_state.SCALE_X
    scaled_primal_obj = sdp_state.primal_obj * sdp_state.SCALE_X * sdp_state.SCALE_C

    return SDPState(
        C=scaled_C,
        A_indices=sdp_state.A_indices,
        A_data=scaled_A_data,
        b=scaled_b,
        b_ineq_mask=sdp_state.b_ineq_mask,
        X=scaled_X,
        P=scaled_P,
        Omega=sdp_state.Omega,
        y=sdp_state.y,
        z=scaled_z,
        tr_X=scaled_tr_X,
        primal_obj=scaled_primal_obj,
        SCALE_C=sdp_state.SCALE_C,
        SCALE_X=sdp_state.SCALE_X,
        SCALE_A=sdp_state.SCALE_A,
        best_ub=sdp_state.best_ub,
        best_obj_gap=sdp_state.best_obj_gap)


def unscale_sdp_state(sdp_state: SDPState) -> SDPState:
    unscaled_C = BCOO((sdp_state.C.data / sdp_state.SCALE_C, sdp_state.C.indices),
                    shape=sdp_state.C.shape)
    unscaled_A_data = sdp_state.A_data / sdp_state.SCALE_A.at[sdp_state.A_indices[:,0]].get()
    unscaled_b = (sdp_state.b / sdp_state.SCALE_A) / sdp_state.SCALE_X

    unscaled_X = sdp_state.X
    unscaled_P = sdp_state.P
    if sdp_state.X is not None:
        unscaled_X = sdp_state.X / sdp_state.SCALE_X
    if sdp_state.P is not None:
        unscaled_P = sdp_state.P / sdp_state.SCALE_X

    unscaled_z = (sdp_state.z / sdp_state.SCALE_A ) / sdp_state.SCALE_X
    unscaled_tr_X = sdp_state.tr_X / sdp_state.SCALE_X
    unscaled_primal_obj = sdp_state.primal_obj / (sdp_state.SCALE_X * sdp_state.SCALE_C)

    return SDPState(
        C=unscaled_C,
        A_indices=sdp_state.A_indices,
        A_data=unscaled_A_data,
        b=unscaled_b,
        b_ineq_mask=sdp_state.b_ineq_mask,
        X=unscaled_X,
        P=unscaled_P,
        Omega=sdp_state.Omega,
        y=sdp_state.y,
        z=unscaled_z,
        tr_X=unscaled_tr_X,
        primal_obj=unscaled_primal_obj,
        SCALE_C=sdp_state.SCALE_C,
        SCALE_X=sdp_state.SCALE_X,
        SCALE_A=sdp_state.SCALE_A,
        best_ub=sdp_state.best_ub,
        best_obj_gap=sdp_state.best_obj_gap)


@jax.jit
def reconstruct_from_sketch(
    Omega: Array,
    P: Array,
    approx_eps: float = 1e-6
) -> Tuple[Array, Array]:
    n = Omega.shape[0]
    rho = jnp.sqrt(n) * approx_eps * jnp.linalg.norm(P, ord=2)
    P_rho = P + rho * Omega
    B = Omega.T @ P_rho
    B = 0.5 * (B + B.T)
    L = jnp.linalg.cholesky(B)
    E, Rho, _ = jnp.linalg.svd(
        jnp.linalg.lstsq(L, P_rho.T, rcond=None)[0].T,
        full_matrices=False,  # this compresses the output to be rank `R`
    )
    Lambda = jnp.clip(Rho ** 2 - rho, 0, jnp.inf)
    return E, Lambda


def apply_A_operator_mx(n: int, m: int, A_data: Array, A_indices: Array, X: Array) -> Array:
    A = BCOO((A_data, A_indices), shape=(m, n, n))
    return sparse.bcoo_reduce_sum(A * X[None, :, :], axes=[1,2]).todense()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@partial(jax.jit, static_argnames=["n"])
def munkres(n: int, cost_mx: Array) -> Array:
    """
    Implementation adapted from: http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
    """

    # create the working matrix
    M = jnp.copy(cost_mx)

    # Step 1: row reduction
    M = M - jnp.min(M, axis=1).reshape(-1, 1)

    StateStruct = namedtuple("StateStruct", ["M", "mask_mx", "row_cover", "col_cover"])

    # Step 2: initial starring of zeros
    def initial_star_body(i: int, state: StateStruct) -> StateStruct:
        row, col = i // n, i % n
        star_entry = jnp.logical_and(
            M[row, col] == 0, jnp.logical_and(state.row_cover[row] == 0, state.col_cover[col] == 0))
        next_mask_mx = jnp.copy(state.mask_mx)
        next_mask_mx = next_mask_mx.at[row, col].set(star_entry)
        next_row_cover = jnp.copy(state.row_cover)
        next_row_cover = next_row_cover.at[row].set(
                jnp.clip(
                    star_entry,
                    state.row_cover.at[row].get(),
                    None
                )
            )
        next_col_cover = jnp.copy(state.col_cover)
        next_col_cover = next_col_cover.at[col].set(
                jnp.clip(
                    star_entry,
                    state.col_cover.at[col].get(),
                    None
                )
            )
        return StateStruct(state.M, next_mask_mx, next_row_cover, next_col_cover)
    init_state = StateStruct(M, jnp.zeros_like(M), jnp.zeros((n,), dtype=bool), jnp.zeros((n,), dtype=bool))
    # TODO: convert this to scan?
    final_state = lax.fori_loop(0, n**2, initial_star_body, init_state)

    # hard assignment body_func helper functions
    def shift_zeros(state: StateStruct) -> StateStruct:
        # Step 6
        cover_mask = state.row_cover.reshape(-1, 1) | state.col_cover.reshape(1, -1)
        min_uncovered_val = jnp.min(jnp.where(~cover_mask, state.M, jnp.max(state.M)))
        sub_min_mask = jnp.ones_like(state.row_cover).reshape(-1, 1) & (~state.col_cover).reshape(1, -1)
        add_min_mask = state.row_cover.reshape(-1, 1) & jnp.ones_like(state.col_cover).reshape(1, -1)
        M_next = jnp.where(sub_min_mask, state.M - min_uncovered_val, state.M)
        M_next = jnp.where(add_min_mask, M_next + min_uncovered_val, M_next)
        return StateStruct(M_next, state.mask_mx, state.row_cover, state.col_cover)

    def adjust_cover(state: StateStruct, row: int, col: int) -> StateStruct:
        # part of Step 4
        col = jnp.argmax(state.mask_mx[row] == 1)
        row_cover_next = state.row_cover.at[row].set(True)
        col_cover_next = state.col_cover.at[col].set(False)
        return StateStruct(state.M, state.mask_mx, row_cover_next, col_cover_next)

    def aug_path(state: StateStruct, row: int, col: int) -> StateStruct:
        # Step 5 (and col_cover of Step 3)
        AugPathStateStruct = namedtuple("AugPathStateStruct", ["aug_path", "mask_mx", "row", "col"])
        aug_path = jnp.zeros_like(state.mask_mx).at[row, col].set(1)

        def cond_func(aug_path_state: AugPathStateStruct) -> bool:
            return jnp.sum(aug_path_state.mask_mx[:, aug_path_state.col] == 1) > 0

        def body_func(aug_path_state: AugPathStateStruct) -> AugPathStateStruct:
            row = jnp.argmax(aug_path_state.mask_mx[:, aug_path_state.col] == 1) 
            col = jnp.argmax(aug_path_state.mask_mx[row] == 2)
            aug_path_next = aug_path_state.aug_path.at[row, aug_path_state.col].set(1)
            aug_path_next = aug_path_next.at[row, col].set(1)
            return AugPathStateStruct(aug_path_next, aug_path_state.mask_mx, row, col)

        aug_path_state = AugPathStateStruct(aug_path, state.mask_mx, row, col)
        aug_path_state = while_loop(cond_func, body_func, aug_path_state, n, unroll=True, jit=True)
        mask_mx_next = state.mask_mx - aug_path_state.aug_path
        mask_mx_next = jnp.where(mask_mx_next == 2, 0, mask_mx_next)
        col_cover = jnp.sum(mask_mx_next == 1, axis=0).astype(bool)
        return StateStruct(state.M, mask_mx_next, jnp.zeros((n,), dtype=bool), col_cover)

    def find_hard_assignment(state: StateStruct) -> StateStruct:
        # Step 4 (Steps 5 & 6 nested)
        def cond_func(state: StateStruct) -> bool:
            # condition part of Step 3
            return jnp.sum(state.mask_mx == 1) < n

        def body_func(state: StateStruct) -> StateStruct:
            cover_mask = state.row_cover.reshape(-1, 1) | state.col_cover.reshape(1, -1)
            uncovered_zero_mask = (state.M == 0) & (~cover_mask)

            def prime_or_aug_path(state: StateStruct, uncovered_zero_mask: Array) -> StateStruct:
                flattened_idx = jnp.argmax(uncovered_zero_mask)
                row, col = flattened_idx // n, flattened_idx % n
                mask_mx_updated = state.mask_mx.at[row, col].set(2)
                next_state = StateStruct(state.M, mask_mx_updated, state.row_cover, state.col_cover)
                next_state = jax.tree.map(
                    lambda t, f: jax.lax.select(jnp.sum(mask_mx_updated[row] == 1) >= 1, t, f),
                    adjust_cover(next_state, row, col),
                    aug_path(next_state, row, col))
                return next_state

            next_state = jax.tree.map(
                lambda t, f: jax.lax.select(jnp.sum(uncovered_zero_mask) == 0, t, f),
                shift_zeros(state),
                prime_or_aug_path(state, uncovered_zero_mask))
            return next_state

        return while_loop(cond_func, body_func, state, n**2, unroll=True, jit=True).mask_mx

    state = StateStruct(
        final_state.M,
        final_state.mask_mx,
        jnp.zeros((n,), dtype=bool),
        jnp.sum(final_state.mask_mx == 1, axis=0).astype(bool))

    return find_hard_assignment(state)

@partial(jax.jit, static_argnames=["n", "n_fixed"])
def munkres_with_fixed(
        n: int,
        cost_mx: Array,
        n_fixed: int,
    ) -> Array:

    free_n = n - n_fixed

    free_perm = munkres(
        free_n,
        cost_mx[:free_n, :free_n],
    )

    perm = jnp.zeros((n, n), dtype=free_perm.dtype)

    perm = perm.at[:free_n, :free_n].set(free_perm)

    perm = perm.at[
        jnp.arange(free_n, n),
        jnp.arange(free_n, n)
    ].set(1)

    return perm




# def load_and_process_qap(fname: str, num_drop: int = 0) -> Tuple[Array, Array]:
#     with open(fname, "r") as f:
#         datastr = f.read().strip().split()
#         str_n, str_data = datastr[0], datastr[1:]
#         n = int(str_n)
#         M = jnp.array([float(v) for v in str_data]).reshape(2*n, n)
#         D = M[:n]
#         W = M[n:]

#     # by convention, W should be sparser than D -- this doesn't really matter
#     if jnp.count_nonzero(D) < jnp.count_nonzero(W):
#         D, W = W, D

#     n_out = n - num_drop
#     D = D[:n_out, :n_out]
#     W = W[:n_out, :n_out]

#     # return expanded and padded kronecker product
#     return n_out, D, W, build_objective_matrix(D, W)


def load_and_process_qap(problem) -> Tuple[Array, Array]:
    D = problem.D
    W = problem.F

    # return expanded and padded kronecker product
    return problem.n, D, W, build_objective_matrix(D, W)


def load_and_process_tsp(fname: str, num_drop: int = 0) -> Tuple[Array, Array]:
    # used documentation to implement: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
    spec_vars = {}
    with open(fname, "r") as f:
        for line in f:
            splitline = [ss.strip() for ss in line.split(":")]
            if len(splitline) == 2: # specification part
                spec_vars[splitline[0]] = splitline[1]
            elif len(splitline) == 1: # info part
                n = int(spec_vars["DIMENSION"])
                D = jnp.zeros((n, n))
                filled_D = False
                coords = None
                if splitline[0] == "NODE_COORD_SECTION":
                    coords = []
                    while True:
                        line = next(f).strip()
                        if line == "EOF":
                            assert len(coords) == n
                            break
                        _, _x, _y = tuple(line.split())
                        coords.append([float(_x), float(_y)])
                    break  # break out of initial for loop, we have what we need.
                elif splitline[0] == "EDGE_WEIGHT_SECTION":
                    edge_dist_str = ""
                    while True:
                        line = next(f).strip()
                        if line == "EOF" or line == "DISPLAY_DATA_SECTION":
                            break
                        edge_dist_str += " " + line
                    flat_edge_dists = jnp.array([int(v) for v in edge_dist_str.split()])
                    if spec_vars["EDGE_WEIGHT_FORMAT"] == "FULL_MATRIX":
                        indices = np.where(D == 0)
                        D = D.at[indices].set(flat_edge_dists)
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["UPPER_ROW", "LOWER_COL"]:
                        indices = jnp.triu_indices(n, k=1)
                        D = D.at[indices].set(flat_edge_dists)
                        D = D + D.T
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["LOWER_ROW", "UPPER_COL"]:
                        raise NotImplementedError("edge weight format type not implemented")  # no examples for this case?
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["UPPER_DIAG_ROW", "LOWER_DIAG_COL"]:
                        indices = jnp.triu_indices(n, k=0)
                        D = D.at[indices].set(flat_edge_dists)
                        D = D + D.T
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["LOWER_DIAG_ROW", "UPPER_DIAG_COL"]:
                        indices = jnp.tril_indices(n, k=0)
                        D = D.at[indices].set(flat_edge_dists)
                        D = D + D.T
                    else:
                        raise ValueError("Unsupported EDGE_WEIGHT_FORMAT.")
                    filled_D = True
                    break  # break out of initial for loop, we have what we need.
                else:
                    pass
            else:
                raise ValueError("Something went wrong when reading file.")

        if not filled_D:
            assert coords is not None
            coords = np.array(coords)
            if spec_vars["EDGE_WEIGHT_TYPE"] == "GEO":
                def geo_dist(p1, p2):
                    PI = 3.141592
                    RRR = 6378.388

                    # compute lat-long for each point in radians
                    lat1 = PI * ((deg := np.round(p1[0])) + 5.0 * (p1[0] - deg) / 3.0 ) / 180.0
                    long1 = PI * ((deg := np.round(p1[1])) + 5.0 * (p1[1] - deg) / 3.0 ) / 180.0
                    lat2 = PI * ((deg := np.round(p2[0])) + 5.0 * (p2[0] - deg) / 3.0 ) / 180.0
                    long2 = PI * ((deg := np.round(p2[1])) + 5.0 * (p2[1] - deg) / 3.0 ) / 180.0

                    # compute distance in kilometers
                    q1 = np.cos(long1 - long2)
                    q2 = np.cos(lat1 - lat2)
                    q3 = np.cos(lat1 + lat2)
                    return int(RRR * np.arccos(0.5*(q2 * (1.0 + q1) - q3 * (1.0 - q1))) + 1.0)
                D = squareform(pdist(coords, geo_dist))
            elif spec_vars["EDGE_WEIGHT_TYPE"] == "ATT":
                def att_dist(p1, p2):
                    xd = p1[0] - p2[0]
                    yd = p1[1] - p2[1]
                    return np.ceil(np.sqrt((xd**2 + yd**2) / 10.0))
                D = squareform(pdist(coords, att_dist))
            elif spec_vars["EDGE_WEIGHT_TYPE"] == "EUC_2D":
                D = np.round(squareform(pdist(coords, 'euclidean')))
            elif spec_vars["EDGE_WEIGHT_TYPE"] == "CEIL_2D":
                D = np.ceil(squareform(pdist(coords, 'euclidean')))
            else:
                raise ValueError("Unsupported EDGE_WEIGHT_TYPE.")
            D = jnp.array(D)
            

    n_out = n - num_drop
    D = D[:n_out, :n_out]

    # construct sparse symmetric canonical tour
    W_indices = []
    W_data = []
    for i in range(n_out):
        W_indices.append([i, (i + 1) % n_out])
        W_indices.append([(i + 1) % n_out, i])
        W_data += [0.5, 0.5]
    W_indices = jnp.array(W_indices)
    W_data = jnp.array(W_data)
    W = BCOO((W_data, W_indices), shape=(n_out, n_out)).todense()

    # return expanded and padded kronecker product
    return n_out, D, W, build_objective_matrix(D, W)


def build_objective_matrix(D: Array, W: Array) -> BCOO:
    n = D.shape[0]
    sparse_D = BCOO.fromdense(D)
    sparse_W = BCOO.fromdense(W)

    D_indices = sparse_D.indices.reshape(1, sparse_D.nse, 2)
    D_data = sparse_D.data.reshape(1, sparse_D.nse)
    W_indices = sparse_W.indices.reshape(sparse_W.nse, 1, 2)
    W_data = sparse_W.data.reshape(sparse_W.nse, 1)

    W_indices *= n
    W_kron_D_indices = (D_indices + W_indices).reshape(sparse_D.nse * sparse_W.nse, 2)
    W_kron_D_data = (D_data * W_data).reshape(sparse_D.nse * sparse_W.nse,)

    C = BCOO((W_kron_D_data, W_kron_D_indices + 1), shape=(n**2 + 1, n**2 + 1))
    return C


def get_all_problem_data(C: BCOO, fixed_assignment: Dict[int,int]) -> Tuple[BCOO, Array, Array, Array]:
    n = C.shape[0]
    l = int(jnp.sqrt(n - 1))

    # initialize with first constraint: X(0,0) = 1
    A_indices = jnp.array([[0, 0, 0]])
    A_data = jnp.array([1.0])
    b = jnp.array([1.0])
    b_ineq_mask = jnp.array([0.0])

    # constraint: diag(Y) = vec(B)
    # equivalent to the following:
    #   for j in range(1, n):
    #       _A_indices.append([_i, j, 0])
    #       _A_indices.append([_i, 0, j])
    #       _A_indices.append([_i, j, j])
    #       _A_data += [-0.5, -0.5, 1.0]
    #       _b.append(0.0)
    #       _b_ineq_mask.append(0.0)
    #       _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, n-1)[:, None], (1, 3)).reshape(-1,)
    coord_a = (jnp.tile(jnp.array([[1, 0, 1]]), (n-1, 1)) * jnp.arange(1, n)[:, None]).reshape(-1,)
    coord_b = (jnp.tile(jnp.array([[0, 1, 1]]), (n-1, 1)) * jnp.arange(1, n)[:, None]).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_data = jnp.concatenate(
        [A_data, jnp.tile(jnp.array([[-1.0, -1.0, 2.0]]), (n-1, 1)).reshape(-1,)], axis=0)
    b = jnp.concatenate([b, jnp.zeros((n-1,))], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((n-1,))], axis=0)

    # constraint: B1 = 1
    # equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           _A_indices.append([_i, j1*l + j2 + 1, 0])
    #           _A_indices.append([_i, 0, j1*l + j2 + 1])
    #           _A_data += [0.5, 0.5]
    #       _b.append(1.0)
    #       _b_ineq_mask.append(0.0)
    #       _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, l)[:, None], (1, 2*l)).reshape(-1,)
    coord_a = (jnp.tile(jnp.array([[1, 0]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).flatten()[:, None]).reshape(-1,)
    coord_b = (jnp.tile(jnp.array([[0, 1]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).flatten()[:, None]).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_data = jnp.concatenate(
        [A_data, jnp.tile(jnp.array([[1.0, 1.0]]), (n-1, 1)).reshape(-1,)], axis=0)
    b = jnp.concatenate([b, 2.0*jnp.ones((l,))], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((l,))], axis=0)


    # constraint: 1'B = 1'
    # equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           _A_indices.append([_i, j1 + j2*l + 1, 0])
    #           _A_indices.append([_i, 0, j1 + j2*l + 1])
    #           _A_data += [0.5, 0.5]
    #       _b.append(1.0)
    #       _b_ineq_mask.append(0.0)
    #       _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, l)[:, None], (1, 2*l)).reshape(-1,)
    coord_a = (jnp.tile(jnp.array([[1, 0]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).T.flatten()[:, None]).reshape(-1,)
    coord_b = (jnp.tile(jnp.array([[0, 1]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).T.flatten()[:, None]).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_data = jnp.concatenate(
        [A_data, jnp.tile(jnp.array([[1.0, 1.0]]), (n-1, 1)).reshape(-1,)], axis=0)
    b = jnp.concatenate([b, 2.0*jnp.ones((l,))], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((l,))], axis=0)

    ## constraint: tr_1(Y) = I
    ## equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           for diag_idx in range(l):
    #               _A_indices.append([_i, j1 + diag_idx*l + 1, j2 + diag_idx*l + 1])
    #               _A_data += [1.0]
    #           if j1 == j2:
    #               _b.append(1.0)
    #           else:
    #               _b.append(0.0)
    #           _b_ineq_mask.append(0.0)
    #           _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, n-1)[:, None], (1, l)).reshape(-1,)
    coord_a = 1 + jnp.tile(
        jnp.arange(l)[:, None, None]
        + l * jnp.arange(l)[None, None, :], (1, l, 1)).reshape(-1,)
    coord_b = 1 + jnp.tile(
        jnp.arange(l)[None, :, None]
        + l * jnp.arange(l)[None, None, :], (l, 1, 1)).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_b, coord_a]).T], axis=0)
    A_data = jnp.concatenate([A_data, jnp.ones_like(coord_a), jnp.ones_like(coord_a)], axis=0)
    b = jnp.concatenate([b, 2.0*(coord_a == coord_b).reshape(l**2, l).T[0].astype(float)], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((n-1,))], axis=0)

    # constraint: tr_2(Y) = I
    # equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           for diag_idx in range(l):
    #               _A_indices.append([_i, j1*l + diag_idx + 1, j2*l + diag_idx + 1])
    #               _A_data += [1.0]
    #           if j1 == j2:
    #               _b.append(1.0)
    #           else:
    #               _b.append(0.0)
    #           _b_ineq_mask.append(0.0)
    #           _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, n-1)[:, None], (1, l)).reshape(-1,)
    coord_a = 1 + jnp.tile(
        l * jnp.arange(l)[:, None, None]
        + jnp.arange(l)[None, None, :], (1, l, 1)).reshape(-1,)
    coord_b = 1 + jnp.tile(
        l * jnp.arange(l)[None, :, None]
        + jnp.arange(l)[None, None, :], (l, 1, 1)).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_b, coord_a]).T], axis=0)
    A_data = jnp.concatenate([A_data, jnp.ones_like(coord_a), jnp.ones_like(coord_a)], axis=0)
    b = jnp.concatenate([b, 2.0*(coord_a == coord_b).reshape(l**2, l).T[0].astype(float)], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((n-1,))], axis=0)
    
    if fixed_assignment is not None:
        for i, u in fixed_assignment.items():
            # fix i to u by adding B(i, u) = 1 and B(u, i) = 1
            k = i*l + u + 1
            A_indices = jnp.concatenate([A_indices, jnp.array([[b.shape[0], 0, k], [b.shape[0], k, 0]])], axis=0)
            A_data = jnp.concatenate([A_data, jnp.array([1.0, 1.0])], axis=0)
            b = jnp.concatenate([b, jnp.array([u])], axis=0)
            b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.array([0.0])], axis=0)

    # constraint: objective-relevant entries of Y >= 0, written as -Y <= 0
    triu_indices_mask = (C.indices[:, 0] <= C.indices[:, 1])
    constraint_indices = b.shape[0] + jnp.arange(jnp.sum(triu_indices_mask))
    constraint_triples = jnp.concatenate(
        [constraint_indices[:, None], C.indices[triu_indices_mask]], axis=1)
    constraint_triples = jnp.concatenate(
        [constraint_triples, constraint_triples[:, [0, 2, 1]]], axis=0)
    A_indices = jnp.concatenate([A_indices, constraint_triples], axis=0)
    A_data = jnp.concatenate([A_data, jnp.full((constraint_triples.shape[0],), -0.5)], axis=0)
    b = jnp.concatenate([b, jnp.full((constraint_indices.shape[0],), 0.0)], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.full((constraint_indices.shape[0],), 1.0)], axis=0)

    return A_data, A_indices, b, b_ineq_mask


def initialize_state(problem, C: BCOO, sketch_dim: int) -> SDPState:
    C = -C
    fixed_assignment = problem.fixed_assignment if hasattr(problem, "fixed_assignment") else None
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C, fixed_assignment)
    n = C.shape[0]
    m = b.shape[0]
    l = int(jnp.sqrt(n - 1))

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False, maxiter=np.iinfo(np.int32).max)[0])
    SCALE_A /= norm_A

    #SCALE_X = 1.0
    #SCALE_C = 1.0
    #SCALE_A = jnp.ones_like(SCALE_A)

    if sketch_dim == -1:
        X = jnp.zeros((n, n))
        Omega = None
        P = None
    elif sketch_dim > 0:
        X = None
        Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, sketch_dim))
        P = jnp.zeros_like(Omega)
    else:
        raise ValueError("Invalid value for sketch_dim")

    y = jnp.zeros((m,))
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A,
        best_ub=-1,
        best_obj_gap=jnp.inf,)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


@nb.njit
def _fill_constraint_index_map(old_A_indices, new_A_indices, constraint_index_map) -> None:
    old_idx = 0
    for curr_idx in range(new_A_indices.shape[0]):
        old_row = old_A_indices[old_idx]
        curr_row = new_A_indices[curr_idx]
        if curr_row[1] == old_row[1] and curr_row[2] == old_row[2]:
            constraint_index_map[old_row[0]] = curr_row[0]
            old_idx += 1


def get_implicit_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    assert sketch_dim == -1 or sketch_dim == old_sdp_state.Omega.shape[1]
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    old_l = int(jnp.sqrt(old_sdp_state.C.shape[0] - 1))
    old_m = old_sdp_state.b.shape[0]
    l = int(jnp.sqrt(C.shape[0] - 1))
    num_drop = l - old_l
    assert sketch_dim in [-1, l]
    
    C = -C
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    index_map = lambda a : a + num_drop * jnp.clip(((a - 1) // (l - num_drop)), a_min=0)

    X = old_sdp_state.X
    Omega = old_sdp_state.Omega
    P = old_sdp_state.P
    if old_sdp_state.X is not None:
        X = BCOO.fromdense(old_sdp_state.X)
        X = BCOO((X.data, jax.vmap(index_map)(X.indices)), shape=(n, n)).todense()
    if old_sdp_state.P is not None:
        Omega = jax.random.normal(jax.random.PRNGKey(n), shape=(n, l)).at[jax.vmap(index_map)(
            jnp.arange(old_sdp_state.Omega.shape[0]))].set(old_sdp_state.Omega)
        P = jnp.zeros_like(Omega).at[jax.vmap(index_map)(
            jnp.arange(old_sdp_state.P.shape[0]))].set(old_sdp_state.P)
    
    old_A_indices = old_sdp_state.A_indices.at[:, 1:].set(
        jax.vmap(index_map)(old_sdp_state.A_indices[:, 1:]))
    constraint_index_map = np.empty((old_m,), dtype=int)
    _fill_constraint_index_map(
        np.asarray(old_A_indices), np.asarray(A_indices), constraint_index_map)

    y = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.y)
    z = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.z)

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    maxiter = np.iinfo(np.int32).max + 1
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False, maxiter=np.iinfo(np.int32).max)[0])
    SCALE_A /= norm_A

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=old_sdp_state.tr_X,
        primal_obj=old_sdp_state.primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state

def get_explicit_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    old_l = int(jnp.sqrt(old_sdp_state.C.shape[0] - 1))
    old_m = old_sdp_state.b.shape[0]
    l = int(jnp.sqrt(C.shape[0] - 1))
    num_drop = l - old_l
    assert sketch_dim in [-1, l]
    
    C = -C
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    index_map = lambda a : a + num_drop * jnp.clip(((a - 1) // (l - num_drop)), a_min=0)

    old_A_indices = old_sdp_state.A_indices.at[:, 1:].set(
        jax.vmap(index_map)(old_sdp_state.A_indices[:, 1:]))
    constraint_index_map = np.empty((old_m,), dtype=int)
    _fill_constraint_index_map(
        np.asarray(old_A_indices), np.asarray(A_indices), constraint_index_map)

    X = old_sdp_state.X
    Omega = old_sdp_state.Omega
    P = old_sdp_state.P
    if old_sdp_state.X is not None:
        X = BCOO.fromdense(old_sdp_state.X)
        X = BCOO((X.data, jax.vmap(index_map)(X.indices)), shape=(n, n)).todense()
        z = apply_A_operator_mx(n, m, A_data, A_indices, X) 
        tr_X = jnp.trace(X)
        primal_obj = jnp.trace(C @ X)
    if old_sdp_state.P is not None:
        Omega = jax.random.normal(jax.random.PRNGKey(n), shape=(n, sketch_dim))
        E, Lambda = reconstruct_from_sketch(old_sdp_state.Omega, old_sdp_state.P)
        tr_offset = (old_sdp_state.tr_X - jnp.sum(Lambda)) / Lambda.shape[0]
        Lambda_tr_correct = Lambda + tr_offset
        E = jnp.zeros_like(Omega).at[jax.vmap(index_map)(jnp.arange(E.shape[0]))].set(E)
        sqrt_X_hat = E * jnp.sqrt(Lambda_tr_correct)[None, :]
        P = sqrt_X_hat @ (sqrt_X_hat.T @ Omega)
        z = apply_A_operator_batched(m, A_data, A_indices, sqrt_X_hat)
        tr_X = jnp.sum(Lambda_tr_correct)
        primal_obj = jnp.trace(sqrt_X_hat.T @ (C @ sqrt_X_hat))

    y = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.y)

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False, maxiter=np.iinfo(np.int32).max)[0])
    SCALE_A /= norm_A

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


def get_dual_only_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    C = -C
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    old_l = int(jnp.sqrt(old_sdp_state.C.shape[0] - 1))
    old_m = old_sdp_state.b.shape[0]
    l = int(jnp.sqrt(C.shape[0] - 1))
    num_drop = l - old_l
    assert sketch_dim in [-1, l]
    
    index_map = lambda a : a + num_drop * jnp.clip(((a - 1) // (l - num_drop)), a_min=0)

    old_A_indices = old_sdp_state.A_indices.at[:, 1:].set(
        jax.vmap(index_map)(old_sdp_state.A_indices[:, 1:]))
    constraint_index_map = np.empty((old_m,), dtype=int)
    _fill_constraint_index_map(
        np.asarray(old_A_indices), np.asarray(A_indices), constraint_index_map)

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False, maxiter=np.iinfo(np.int32).max)[0])
    SCALE_A /= norm_A

    if sketch_dim == -1:
        X = jnp.zeros((n, n))
        Omega = None
        P = None
    elif sketch_dim > 0:
        X = None
        Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, sketch_dim))
        P = jnp.zeros_like(Omega)
    else:
        raise ValueError("Invalid value for sketch_dim")

    y = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.y)
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


@partial(jax.jit, static_argnames=["callback_static_args"])
def qap_round(
    P: Array,
    Omega: Array,
    callback_static_args: bytes,
    callback_nonstatic_args: Any
) -> float:
    args_static = pickle.loads(callback_static_args)
    l = args_static["l"]
    n_fixed = args_static["n_fixed"]
    D = callback_nonstatic_args["D"]
    W = callback_nonstatic_args["W"]
    E, _ = reconstruct_from_sketch(Omega, P)
    
    def pos_body_func(i: int) -> float:
        cost_mx = E[1:, i].reshape(l, l)
        cost_mx = jnp.max(cost_mx) - cost_mx
        perm_mx = munkres_with_fixed(l, cost_mx, n_fixed)
        return jnp.clip(jnp.trace(W @ perm_mx @ D @ perm_mx.T))

    def neg_body_func(i: int) -> float:
        # try negative too
        cost_mx = -E[1:, i].reshape(l, l)
        cost_mx = jnp.max(cost_mx) - cost_mx
        perm_mx = munkres_with_fixed(l, cost_mx, n_fixed)
        return jnp.trace(W @ perm_mx @ D @ perm_mx.T)

    # want minimum
    min_pos = jnp.min(jax.vmap(pos_body_func)(jnp.arange(l, dtype=int)))
    min_neg = jnp.min(jax.vmap(neg_body_func)(jnp.arange(l, dtype=int)))
    best_assign_obj = lax.select(min_pos < min_neg, min_pos, min_neg)
    return best_assign_obj


@partial(jax.jit, static_argnames=["m"])
def apply_A_operator_slim(m: int, A_data: Array, A_indices: Array, u: Array) -> Array:
    outvec = jnp.zeros((m,))
    outvec = outvec.at[A_indices[:,0]].add(
        A_data * u.at[A_indices[:,1]].get() * u.at[A_indices[:,2]].get())
    return outvec


@partial(jax.jit, static_argnames=["n"])
def apply_A_adjoint_slim(n: int, A_data: Array, A_indices: Array, z: Array, u: Array) -> Array:
    outvec = jnp.zeros((n,))
    outvec = outvec.at[A_indices[:,1]].add(
        A_data * z.at[A_indices[:,0]].get() * u.at[A_indices[:,2]].get())
    return outvec


@partial(jax.jit, static_argnames=["m"])
def apply_A_operator_batched(m: int, A_data: Array, A_indices: Array, vecs: Array) -> Array:
    return jnp.sum(
        jax.vmap(apply_A_operator_slim, (None, None, None, 1), 1)(m, A_data, A_indices, vecs),
        axis=1)


@partial(jax.jit, static_argnames=["n"])
def apply_A_adjoint_batched(n: int, A_data: Array, A_indices: Array, z: Array, vecs: Array) -> Array:
    return jax.vmap(
        apply_A_adjoint_slim, (None, None, None, None, 1), 1)(n, A_data, A_indices, z, vecs)


def create_svec_matrix(k: int) -> BCOO:
    U = np.zeros((int(0.5*k*(k+1)), k**2))
    for a, (b, c) in enumerate(list(zip(*np.tril_indices(k)))):
        if b == c:
            U[a, b*k + c] = 1.0
        else:
            U[a, b*k + c] = 1.0 / np.sqrt(2.0)
            U[a, c*k + b] = U[a, b*k + c]
    U = coo_matrix(U)
    U = BCOO((U.data, jnp.stack((U.row, U.col)).T), shape=U.shape)
    return U


@partial(jax.jit, static_argnames=["m", "k"])
def create_Q_base(m: int, k: int, U: BCOO, A_data: Array, A_indices: Array, V: Array) -> Array:
    base_tensor = jnp.zeros((m, k, k))
    base_tensor = base_tensor.at[A_indices[:, 0]].add(
        A_data.reshape(-1, 1, 1)
        * jax.lax.batch_matmul(V.at[A_indices[:, 1]].get().reshape(-1, k, 1),
                               V.at[A_indices[:, 2]].get().reshape(-1, 1, k)))
    flat_base_tensor = base_tensor.reshape(m, k**2).T
    svec_proj = U @ flat_base_tensor
    svec_dim_size = int(k*(k+1)/2)
    expanded_mx = svec_proj.reshape(svec_dim_size, 1, m) * svec_proj.reshape(1, svec_dim_size, m)
    final_mx = jnp.sum(expanded_mx, axis=-1)
    return final_mx

