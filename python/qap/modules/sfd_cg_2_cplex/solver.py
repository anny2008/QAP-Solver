# solver.py
#
# FULL BRANCH-AND-PRICE FOR FORM3
#
# Integrates:
#   - RMP_FORM3             (master)
#   - PricingFORM3          (pricing)
#   - BranchingTree, Node   (branching)
#
# Usage:
#   from qap.solver_form3.solver import BranchAndPriceFORM3
#   sol = BranchAndPriceFORM3().solve(problem)

import time
import math

from .column import Column
from .rmp import RMP
from .pricing import Pricing
from .branching import (
    BranchingTree,
    BranchingNode,
    find_branching_variable,
    create_branch_children,
)
from qap.core import Problem, Solution, write_result
from qap.core.solution_io import read_warmstart
from qap.decomposition import decompose_value_layer, decompose_value_only

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
class BranchAndPriceSFD:
    """
    Full Branch-and-Price solver for FORM3.
    Uses:
        - RMP for LP relaxation
        - PricingFORM3 for column generation
        - branching_form3 for branching on x[i,u]
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize solver with configuration.

        Args:
            config (dict): Configuration dictionary
                Required keys: solver, formulation
                Optional keys:
                    - decomposition (str): 'value_subgraph' or 'value_only' (default: 'value_subgraph')
                    - use_cuts (bool): Use lazy constraint cuts (default: True)
                    - time_limit (float): Time limit in seconds (default: 3600)
                    - threads (int): Number of threads (default: 8)
                    - log_output (bool): Print solver output (default: False)
                    - is_relax (bool): Use relaxed constraints (default: False)
        """
        self.config = config
        self.solver_name = config.get("solver", "sfd")
        self.formulation = config.get("formulation", "sfd")
        self.decomposition = config.get("decomposition", "value_subgraph")
        self.use_cuts = config.get("use_cuts", True)
        self.time_limit = config.get("time_limit", 3600)
        self.threads = config.get("threads", 8)
        self.log_output = config.get("log_output", False)
        self.is_relax = config.get("is_relax", False)

        self.max_nodes = config.get("max_nodes", 10_000)
        self.max_iterations = config.get("max_iterations", 10_000)
        self.eps = config.get("eps", 1e-7)
        self.verbose = config.get("verbose", True)
        self.time_limit = config.get("time_limit", 3600)

        # GLOBAL incumbent
        self.best_obj = math.inf
        self.best_assignment = None
        

    # ------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------
    def solve(self, problem, subgraphs, warmstart):
        """
        problem:  Problem object (from problem.py)
        subgraphs:   dict k → {nodes, arcs, Fk}
        warmstart:   Path to warmstart file (optional)

        Returns:
            dict with:
                assignment: list π[i] = u
                objective: best solution
                time: CPU time
        """
        start = time.time()

        # INITIAL NODE (root)
        root = BranchingNode(fixed_x={}, lower_bound=-math.inf, depth=0)
        tree = BranchingTree()
        tree.add_node(root)

        # Explore nodes
        nodes_visited = 0
        while tree.has_open_nodes():
            if time.time() - start > self.time_limit:
                if self.verbose:
                    print("TIME LIMIT REACHED.")
                break

            node = tree.get_best_node()
            nodes_visited += 1

            if nodes_visited > self.max_nodes:
                if self.verbose:
                    print("MAX NODE LIMIT REACHED.")
                break

            if self.verbose:
                print(f"\n===== Exploring Node {node.id} (depth={node.depth}) =====")

            # Solve LP relaxation at this node
            result = self._solve_node(problem, subgraphs, node, warmstart=warmstart)
            if warmstart is not None:
                warmstart = None  # only use warmstart at root node
                if self.verbose:
                    print("Warmstart used at root node, not applying to child nodes.")
            if result is None:
                if self.verbose:
                    print("Infeasible LP → prune.")
                continue

            lb, x_vals = result

            # Bound check (prune)
            if lb >= self.best_obj - self.eps:
                if self.verbose:
                    print(f"LB {lb:.4f} >= incumbent {self.best_obj:.4f} → prune.")
                continue

            # Check integrality
            frac_var = find_branching_variable(x_vals, eps=self.eps)
            if frac_var is None:
                # Integer solution → feasible candidate
                self._update_incumbent(x_vals, problem)
                if self.verbose:
                    print(f"INTEGER SOLUTION FOUND: obj={self.best_obj:.4f}")
                continue

            # Branch further
            (i,u) = frac_var
            left, right = create_branch_children(node, i, u)

            if self.verbose:
                print(f"Branching on x[{i},{u}] fractional → left=1, right=0")

            # Add children to tree
            tree.add_node(left)
            tree.add_node(right)
            break

        # FINAL RESULT
        elapsed = time.time() - start
        if self.best_assignment is None:
            return {
                "assignment": None,
                "objective": None,
                "time": elapsed,
                "status": "No feasible solution found"
            }

        return {
            "assignment": self.best_assignment,
            "objective": self.best_obj,
            "time": elapsed,
            "status": "Optimal or best found"
        }
    
    
    def _build_initial_column_for_subgraph(self, problem, k, subgraph, branching_fixed_x):
        V = list(range(problem.n))
        nodes = list(subgraph[2])

        # Respect x-fixes: u→i if fixed 1; forbid u→i if fixed 0
        forced = {u: i for (i,u),val in branching_fixed_x.items() if val == 1 and u in nodes}
        forbidden = {(i,u) for (i,u),val in branching_fixed_x.items() if val == 0 and u in nodes}

        used_i = set()
        phi = {}

        # place forced first
        for u, i in forced.items():
            phi[u] = i
            used_i.add(i)

        # greedy fill remaining u
        for u in nodes:
            if u in phi:
                continue
            # pick the first i not used and not forbidden for this u
            i_choice = next(i for i in V if i not in used_i and (i,u) not in forbidden)
            phi[u] = i_choice
            used_i.add(i_choice)

        # build e_ij from arcs
        e_ij = {}
        for (u, v) in subgraph[1]:
            i = phi[u]
            j = phi[v]
            e_ij[(i, j)] = 1
        # compute cost
        cost = 0
        for (u, v) in subgraph[1]:
            i = phi[u]
            j = phi[v]
            cost += problem.D[i][j]
        return Column(subgraph_id=k, phi_map=phi, e_ij=e_ij, cost=cost)

    def _extract_columns_from_warmstart(self, problem, subgraphs, warmstart):
        columns = []
        for k, subgraph in subgraphs.items():
            col = self._build_initial_column_for_subgraph(problem, k, subgraph, warmstart)
            columns.append((k, col))
        return columns


    # ------------------------------------------------------------
    # SOLVE A NODE: column generation until LP convergence
    # ------------------------------------------------------------
    def _solve_node(self, problem, subgraphs, node, warmstart=None):
        """
        Solve LP relaxation at a B&P node:
            - build RMP
            - apply node's branching fixes
            - run column generation
            - return (lower_bound, x_solution)
        """
        rmp = RMP(problem, subgraphs)
        pricing = Pricing(problem, subgraphs)

        # FIX BRANCHING VALUES
        for (i,u), val in node.fixed_x.items():
            rmp.fix_x(i, u, val)
        if warmstart is not None:
            if self.verbose:
                print(f"Adding warmstart columns from {warmstart}...")
            warmstart_cols = self._extract_columns_from_warmstart(problem, subgraphs, warmstart)
            if self.verbose:
                print(f"  Extracted {len(warmstart_cols)} columns from warmstart.")
            for (k, col) in warmstart_cols:
                rmp.add_column(k, col)
                
        for k, subgraph in subgraphs.items():
            dummy_col = self._build_initial_column_for_subgraph(problem, k, subgraph, node.fixed_x)
            rmp.add_column(k, dummy_col)
        
        rmp.report_num_columns()

        # Iterative Column Generation
        for it in range(self.max_iterations):
            rmp.solve()
            status = rmp.cpx.solution.get_status()

            if status in [rmp.cpx.solution.status.infeasible,
                          rmp.cpx.solution.status.infeasible_or_unbounded]:
                return None

            lb = rmp.cpx.solution.get_objective_value()

            if self.verbose:
                print(f"  CG Iteration {it}: RMP obj={lb:.6f}")
                rmp.check_solution()
                
            

            if lb >= self.best_obj - self.eps:
                return None

            # Duals
            dual_pi, dual_alpha = rmp.get_duals()

            # Pricing
            branching_fixed_x = node.fixed_x
            neg_cols = pricing.price_all_subgraphs(dual_pi, dual_alpha, branching_fixed_x)

            if not neg_cols:
                # no improving column → LP optimal at this node
                x_vals = rmp.get_x_solution()
                node.lower_bound = lb
                return (lb, x_vals)

            # Add columns
            n_added = 0
            for (rc, col) in neg_cols:
                if rmp.add_column(col.subgraph_id, col):
                    n_added += 1
                if self.verbose:
                    print(f"    Adding column with reduced cost {rc:.6f} for subgraph {col.subgraph_id} with cost {col.cost:.6f}")
            
            if self.verbose:
                print(f"    Added {n_added} columns with negative reduced cost")
                
                rmp.report_num_columns()
                print("=============================================")
            if n_added == 0:
                # No new columns added, LP optimal at this node
                x_vals = rmp.get_x_solution()
                node.lower_bound = lb
                return (lb, x_vals)

        return None

    # ------------------------------------------------------------
    # Update global incumbent
    # ------------------------------------------------------------
    def _update_incumbent(self, x_vals, problem):
        # Convert fractional x to assignment vector π
        n = problem.n
        assignment = [-1] * n
        for (i,u), v in x_vals.items():
            if v > 0.5:
                assignment[i] = u

        # Compute objective
        val = problem.evaluate_assignment(assignment)
        if val < self.best_obj - self.eps:
            self.best_obj = val
            self.best_assignment = assignment
            if self.verbose:
                print(f"  >>> New incumbent: {val:.4f}")
                
                
    def solve_instance(
        self,
        instance_path: str,
        output_path: Optional[str] = None,
        fixed_variables: Optional[List[Tuple[int, int]]] = None,
        warmstart_path: Optional[str] = None,
    ) -> Solution:
        """
        Solve a problem instance from file.

        Args:
            instance_path (str): Path to QAPLIB instance file
            output_path (str, optional): Path to save result JSON
            fixed_variables (list, optional): List of (i, u) to fix
            warmstart_path (str, optional): Path to warmstart file (JSON dict)

        Returns:
            Solution: Solution object
        """
        # Load problem
        if "QAPLIB" in instance_path:
            problem = Problem.from_qaplib(instance_path)
        else:
            problem = Problem.from_full_instance(
                matrix_file=instance_path,
                workstations_file=instance_path.replace(".txt", "_workstations.txt"),
                machines_file=instance_path.replace(".txt", "_machines.txt"),
                fixed_file=instance_path.replace(".txt", "_fixed.txt")
            )

        # Load warm-start if provided
        if warmstart_path is not None:
            warmstart = read_warmstart(warmstart_path)
        else:
            # warm start with a random solution (or could use a heuristic)
            u_shuffled = np.random.permutation(problem.m)
            i_m = np.arange(problem.n)
            i_m = i_m[:problem.m]  # repeat if m > n
            warmstart = {(i, u): 1 for i,u in zip(i_m, u_shuffled)}  # assign each location to a random machine
            # calculate objective of this random solution
            random_obj = problem.evaluate_assignment([u for i, u in warmstart.keys()])
            if self.verbose:
                print(f"Random warmstart objective: {random_obj:.6f}")
            # # use LocalSearchSolver to find a warm-start solution
            # from qap.modules.local_search import LocalSearchSolver
            
            # print(f"Running local search to find initial solution...")
            # local_solver = LocalSearchSolver({})
            # local_solution = local_solver.solve(problem, fixed_variables=None)
            # print(f"Local search initial solution: obj={local_solution.objective:.6f}")
            # if hasattr(problem, "fixed_assignments"):
            #     for i, u in problem.fixed_assignments.items():
            #         if local_solution.assignment[i] != u:
            #             print(f"Warning: Local search solution violates fixed assignment at location {i}: assigned {local_solution.assignment[i]} vs fixed {u}")
                    
            # warmstart = {(i, u): 1.0 for i, u in enumerate(local_solution.assignment)}
        
        # Decompose problem into subgraphs
        if self.decomposition == "value_subgraph":
            subgraphs = decompose_value_layer(problem)
        elif self.decomposition == "value_only":
            subgraphs = decompose_value_only(problem)
        print(f"Decomposed into {len(subgraphs)} subgraphs using {self.decomposition} strategy.")
        # Solve
        solution = self.solve(
            problem,
            warmstart=warmstart,
            subgraphs=subgraphs
        )

        # Update instance name
        solution.instance = Path(instance_path).stem

        # Save result if output path provided
        if output_path:
            write_result(solution, output_path)

        return solution
