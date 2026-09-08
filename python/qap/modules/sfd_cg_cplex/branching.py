# branching_form3.py
#
# Branching rule and node management for FORM3 Branch-and-Price.
#
# We branch on x[i,u] = 0 or 1.
# The branching decisions must be passed to both:
#   - RMP (fix x[i,u])
#   - Pricing (PricingFORM3 uses branching_fixed_x)
#
# This module provides:
#   - BranchingNode
#   - BranchingTree
#   - find_branching_variable()

import copy
import math


class BranchingNode:
    """
    Represents a node in the branch-and-bound tree.

    Stores:
        - fixed_x: dict (i,u) → 0 or 1
        - lower_bound: LP bound from solving RMP at this node
        - depth: depth in the search tree
        - id: unique ID
        - parent: optional link to parent
        - rmp_snapshot: optional (can store RMP state if needed)
    """

    _global_id = 0

    def __init__(self, fixed_x=None, lower_bound=0.0, depth=0, parent=None):
        BranchingNode._global_id += 1
        self.id = BranchingNode._global_id
        self.fixed_x = dict(fixed_x) if fixed_x else {}
        self.lower_bound = lower_bound
        self.depth = depth
        self.parent = parent

    def __repr__(self):
        return f"<Node {self.id} depth={self.depth} LB={self.lower_bound:.4f} fixes={len(self.fixed_x)}>"


# -------------------------------------------------------------------
# Branching tree manager (best-bound search)
# -------------------------------------------------------------------

class BranchingTree:
    """
    Maintains open nodes.
    Supports:
        - add_node()
        - get_best_node()
        - has_open_nodes()
    """

    def __init__(self):
        self.open_nodes = []

    def add_node(self, node: BranchingNode):
        self.open_nodes.append(node)

    def has_open_nodes(self):
        return len(self.open_nodes) > 0

    def get_best_node(self):
        """
        Best-bound strategy:
            pick node with smallest lower_bound (minimization)
        """
        if not self.open_nodes:
            return None

        best = min(self.open_nodes, key=lambda nd: nd.lower_bound)
        self.open_nodes.remove(best)
        return best


# -------------------------------------------------------------------
# Find branching candidate
# -------------------------------------------------------------------

def find_branching_variable(x_vals, eps=1e-6):
    """
    Given fractional x solution:
        x_vals[(i,u)] in [0,1]
    Return a variable (i,u) where x is fractional.

    Strategy:
        - Look for the most fractional (closest to 0.5).
        - If none found: return None (solution integer)
    """
    best_var = None
    best_dist = 1.0

    for (i,u), val in x_vals.items():
        if eps < val < 1.0 - eps:
            # fractional
            dist = abs(val - 0.5)
            if dist < best_dist:
                best_dist = dist
                best_var = (i,u)

    return best_var


# -------------------------------------------------------------------
# Create child nodes from a branching variable
# -------------------------------------------------------------------

def create_branch_children(parent_node: BranchingNode, i, u):
    """
    Creates two children:
        left:  x[i,u] = 1
        right: x[i,u] = 0

    They inherit parent's fixed_x.
    """
    fx_left = copy.deepcopy(parent_node.fixed_x)
    fx_left[(i,u)] = 1

    fx_right = copy.deepcopy(parent_node.fixed_x)
    fx_right[(i,u)] = 0

    left_node = BranchingNode(
        fixed_x=fx_left,
        lower_bound=math.inf,
        depth=parent_node.depth + 1,
        parent=parent_node
    )

    right_node = BranchingNode(
        fixed_x=fx_right,
        lower_bound=math.inf,
        depth=parent_node.depth + 1,
        parent=parent_node
    )

    return left_node, right_node