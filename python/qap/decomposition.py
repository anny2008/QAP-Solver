"""
Subgraph decomposition utilities for SFD formulation.
"""

from typing import Dict, List, Set, Tuple

import numpy as np


def decompose_value_layer(problem) -> Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]]:
    """
    Decompose flow matrix into value layers.

    Each layer contains edges with the minimum remaining flow value.

    Args:
        problem: QAP problem instance with F (flow matrix) and m

    Returns:
        dict: {k: (f_k, G_k, G_n_k)} where
            f_k: flow value for layer k
            G_k: list of edges (u,v) in layer k
            G_n_k: set of nodes in layer k
    """
    n = problem.m
    flows = problem.F.copy()
    subgraphs: Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]] = {}
    k = 0

    while True:
        min_flow = None
        for i in range(n):
            for j in range(n):
                if flows[i, j] > 1e-9:
                    if min_flow is None or flows[i, j] < min_flow:
                        min_flow = flows[i, j]

        if min_flow is None:
            break

        G_k: List[Tuple[int, int]] = []
        G_n_k: Set[int] = set()

        for u in range(n):
            for v in range(n):
                if flows[u, v] >= min_flow - 1e-9:
                    G_k.append((u, v))
                    G_n_k.add(u)
                    G_n_k.add(v)

        subgraphs[k] = (float(min_flow), G_k, G_n_k)
        k += 1

        for i in range(n):
            for j in range(n):
                if flows[i, j] >= min_flow - 1e-9:
                    flows[i, j] -= min_flow
                    if flows[i, j] < 1e-9:
                        flows[i, j] = 0.0

    return subgraphs


def decompose_value_only(problem) -> Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]]:
    """
    Decompose flow matrix by grouping edges with the same flow value.

    Args:
        problem: QAP problem instance with F (flow matrix) and m

    Returns:
        dict: {k: (f_k, G_k, G_n_k)} grouped by flow values
    """
    n = problem.m
    temp_subgraphs: Dict[float, List[Tuple[int, int]]] = {}

    for u in range(n):
        for v in range(n):
            if problem.F[u, v] > 1e-9:
                f_val = float(problem.F[u, v])
                temp_subgraphs.setdefault(f_val, []).append((u, v))

    subgraphs: Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]] = {}
    for k, (f_k, edges) in enumerate(sorted(temp_subgraphs.items())):
        G_n_k: Set[int] = set()
        for u, v in edges:
            G_n_k.add(u)
            G_n_k.add(v)
        subgraphs[k] = (f_k, edges, G_n_k)

    return subgraphs
