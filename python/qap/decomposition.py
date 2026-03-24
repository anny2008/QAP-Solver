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


def decompose_value_only_no_cycle3(problem) -> Dict[int, Tuple[float, List[Tuple[int, int]], Set[int]]]:
    """
    Decompose flow matrix by grouping edges with the same flow value.
    Remove cycles of length 3 from the subgraphs by moving arcs to new subgraphs.

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
    print(f"Initial decomposition into {len(subgraphs)} subgraphs based on flow values.")
    new_subgraphs = []
    for k, (f_k, edges, G_n_k) in subgraphs.items():
        cycles = set()
        # Build adjacency list for current subgraph
        adj: Dict[int, Set[int]] = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Detect and remove cycles of length 3
        new_edges: List[Tuple[int, int]] = []
        for u in adj:
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w != v and w in adj[u]:  # Found a cycle (u,v,w)
                        # add (u,v,w) to cycles, make sure that u <= v <= w to avoid duplicates
                        cycle = tuple(sorted((u, v, w)))
                        if cycle not in cycles:
                            cycles.add(cycle)
        # for each cycle (u,v,w), remove only one arc u->v
        for cycle in cycles:
            new_edges.append((cycle[0], cycle[1]))  # keep u->v
            edges.remove((cycle[0], cycle[1]))  # remove u->v
            # add new subgraph with only the removed arc
        new_subgraphs.append((f_k, new_edges))
        # recompute G_n_k for the remaining edges
        G_n_k = set()
        for u, v in edges:
            G_n_k.add(u)
            G_n_k.add(v)
        subgraphs[k] = (f_k, edges, G_n_k)
    # add new subgraphs for the removed arcs
    for f_k, edges in new_subgraphs:
        G_n_k: Set[int] = set()
        for u, v in edges:
            G_n_k.add(u)
            G_n_k.add(v)
        subgraphs[len(subgraphs)] = (f_k, edges, G_n_k)
    # verify no cycles of length 3 remain
    for k, (f_k, edges, G_n_k) in subgraphs.items():
        adj: Dict[int, Set[int]] = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        for u in adj:
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w != v and w in adj[u]:  # Found a cycle (u,v,w)
                        raise ValueError(f"Cycle of length 3 detected in subgraph {k} after decomposition")
    print(f"Decomposition after removing cycles of length 3 resulted in {len(subgraphs)} subgraphs.")
    return subgraphs
