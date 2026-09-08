# column.py
# Data structure representing a column for a subgraph G_k

class Column:
    """
    Represents a column for a subgraph G_k:
       - φ_map: machine → location mapping
       - e_ij: induced e^k_{ij} values as a sparse dict {(i,j): 1}
       - cost: Σ_{(u,v) in A_k} d_{φ(u), φ(v)}
    """

    __slots__ = ["subgraph_id", "phi_map", "e_ij", "cost"]

    def __init__(self, subgraph_id, phi_map, e_ij, cost):
        self.subgraph_id = subgraph_id
        self.phi_map = phi_map      # dict {u → i}
        self.e_ij = e_ij            # dict {(i,j) → 1}
        self.cost = cost            # float

    def __repr__(self):
        return f"Column(k={self.subgraph_id}, cost={self.cost}, phi={self.phi_map})"