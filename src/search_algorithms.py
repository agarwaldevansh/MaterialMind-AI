"""
search_algorithms.py — Graph Search and Local Search for Material Substitution
MaterialMind AI — College Level AIML Project

PURPOSE:
Implements classic computer science search algorithms over a material similarity graph:
1. Breadth-First Search (BFS): Level-by-level exploration of direct and indirect material alternatives.
2. Depth-First Search (DFS): Deep branch exploration of substitution paths.
3. A* Search: Heuristic-guided search minimizing substitution friction and target property deficits.
4. Local Search (Hill Climbing): Iterative neighborhood optimization balancing multi-criteria trade-offs.

DEFINITIONS:
- Node: A specific steel alloy record in the dataset.
- Edge: A viable metallurgical substitution pathway between two alloys whose composition similarity exceeds threshold.
- Edge Cost: The composition distance or friction of transitioning between two alloys (1.0 - similarity).
"""

import heapq
from collections import deque
import numpy as np
import pandas as pd


class MaterialGraphSearch:
    """
    Executes BFS, DFS, A*, and Local Search on material similarity graphs.
    """

    def __init__(self, adj_list, df, similarity_calculator=None, cost_estimator=None):
        """
        Parameters
        ----------
        adj_list : dict
            Graph adjacency list {node_id: [(neighbor_id, edge_cost, similarity), ...]}
        df : pd.DataFrame
            Dataset records corresponding to graph nodes.
        similarity_calculator : MaterialSimilarityCalculator
            Helper for calculating pairwise similarities.
        cost_estimator : CostEstimator
            Helper for material pricing lookups.
        """
        self.adj_list = adj_list
        self.df = df
        self.similarity_calc = similarity_calculator
        self.cost_estimator = cost_estimator

    # -------------------------------------------------------------------------
    # A. BREADTH-FIRST SEARCH (BFS)
    # -------------------------------------------------------------------------
    def breadth_first_search(self, start_idx, goal_predicate, max_depth=3):
        """
        Explore the similarity graph level-by-level starting from start_idx.

        Parameters
        ----------
        start_idx : int
            Index of reference material.
        goal_predicate : callable
            Function taking (row) and returning True if candidate satisfies goal criteria.
        max_depth : int
            Maximum search depth / degrees of separation.

        Returns
        -------
        dict containing:
            - 'found_goals': list of (node_idx, depth, path)
            - 'visited_count': int
            - 'exploration_levels': dict mapping depth -> list of explored node indices
        """
        print(f"\n--- Running Breadth-First Search (BFS) from Node {start_idx} (max_depth={max_depth}) ---")
        visited = {start_idx}
        queue = deque([(start_idx, 0, [start_idx])])

        found_goals = []
        levels = {d: [] for d in range(max_depth + 1)}
        levels[0].append(start_idx)

        while queue:
            curr_idx, depth, path = queue.popleft()

            # Check goal condition (excluding start node)
            if curr_idx != start_idx and goal_predicate(self.df.iloc[curr_idx]):
                found_goals.append((curr_idx, depth, path))

            if depth >= max_depth:
                continue

            for neighbor, cost, sim in self.adj_list.get(curr_idx, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_depth = depth + 1
                    levels[next_depth].append(neighbor)
                    queue.append((neighbor, next_depth, path + [neighbor]))

        print(f"  BFS finished: Visited {len(visited)} nodes across {max_depth} levels. Found {len(found_goals)} qualifying goals.")
        return {
            "found_goals": found_goals,
            "visited_count": len(visited),
            "exploration_levels": levels
        }

    # -------------------------------------------------------------------------
    # B. DEPTH-FIRST SEARCH (DFS)
    # -------------------------------------------------------------------------
    def depth_first_search(self, start_idx, goal_predicate, max_depth=5):
        """
        Explore deeply along each branch of the similarity graph before backtracking.

        Parameters
        ----------
        start_idx : int
            Index of reference material.
        goal_predicate : callable
            Function taking (row) and returning True if candidate satisfies criteria.
        max_depth : int
            Maximum recursion depth limit to prevent infinite loops.

        Returns
        -------
        dict containing:
            - 'found_goals': list of (node_idx, depth, path)
            - 'visited_count': int
            - 'exploration_order': list of nodes in order visited
        """
        print(f"\n--- Running Depth-First Search (DFS) from Node {start_idx} (max_depth={max_depth}) ---")
        visited = set()
        found_goals = []
        exploration_order = []

        stack = [(start_idx, 0, [start_idx])]

        while stack:
            curr_idx, depth, path = stack.pop()

            if curr_idx in visited:
                continue

            visited.add(curr_idx)
            exploration_order.append(curr_idx)

            if curr_idx != start_idx and goal_predicate(self.df.iloc[curr_idx]):
                found_goals.append((curr_idx, depth, path))

            if depth < max_depth:
                # Add neighbors to stack in reverse order so first neighbor is visited first
                neighbors = self.adj_list.get(curr_idx, [])
                for neighbor, cost, sim in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append((neighbor, depth + 1, path + [neighbor]))

        print(f"  DFS finished: Visited {len(visited)} nodes. Found {len(found_goals)} qualifying goals.")
        return {
            "found_goals": found_goals,
            "visited_count": len(visited),
            "exploration_order": exploration_order
        }

    # -------------------------------------------------------------------------
    # C. A* SEARCH (HEURISTIC-GUIDED SEARCH)
    # -------------------------------------------------------------------------
    def a_star_search(self, start_idx, goal_predicate, required_min_ts, required_min_duct=None):
        """
        Find a minimal-cost substitution pathway from reference alloy to an acceptable target alloy.

        Definitions:
        - g(n): Cumulative path cost = sum of edge transition costs (1.0 - similarity)
        - h(n): Heuristic property deficit estimate = normalized shortfall to required strength and ductility.
        - f(n) = g(n) + h(n)

        Note on Optimality:
        Because metallurgical property relationships are complex and non-linear,
        this heuristic provides guidance rather than an absolute mathematical guarantee of global optimality.
        Formally documented as heuristic-guided search.

        Parameters
        ----------
        start_idx : int
            Reference material node.
        goal_predicate : callable
            Checks if goal requirements are met.
        required_min_ts : float
            Target tensile strength (MPa).
        required_min_duct : float, optional
            Target ductility (%).

        Returns
        -------
        dict containing:
            - 'target_node': int or None
            - 'path': list of node indices from start to target
            - 'total_cost': float g(n)
            - 'nodes_expanded': int
        """
        print(f"\n--- Running A* Search from Node {start_idx} (Target TS >= {required_min_ts} MPa) ---")

        def compute_heuristic(node_idx):
            """Estimate remaining distance to satisfy requirements."""
            row = self.df.iloc[node_idx]
            ts = row.get("tensile_strength_mpa", 0)
            duct = row.get("ductility_pct", 0)

            # Normalized shortfall (0 if requirement met, >0 if deficit exists)
            ts_deficit = max(0.0, (required_min_ts - ts) / max(required_min_ts, 1.0))
            duct_deficit = 0.0
            if required_min_duct is not None and required_min_duct > 0:
                duct_deficit = max(0.0, (required_min_duct - duct) / required_min_duct)

            return float(ts_deficit + duct_deficit)

        # Priority queue entries: (f_score, g_cost, node_idx, path)
        h0 = compute_heuristic(start_idx)
        open_set = []
        heapq.heappush(open_set, (h0, 0.0, start_idx, [start_idx]))

        g_costs = {start_idx: 0.0}
        nodes_expanded = 0
        closed_set = set()

        while open_set:
            f, g, curr_idx, path = heapq.heappop(open_set)

            if curr_idx in closed_set:
                continue

            closed_set.add(curr_idx)
            nodes_expanded += 1

            # Goal test (must not be the start node itself)
            if curr_idx != start_idx and goal_predicate(self.df.iloc[curr_idx]):
                print(f"  A* Goal Reached! Target Node: {curr_idx}, Total Path Cost: {g:.4f}, Nodes Expanded: {nodes_expanded}")
                return {
                    "target_node": curr_idx,
                    "path": path,
                    "total_cost": round(g, 4),
                    "nodes_expanded": nodes_expanded,
                    "heuristic_at_goal": compute_heuristic(curr_idx)
                }

            # Expand neighbors
            for neighbor, edge_cost, sim in self.adj_list.get(curr_idx, []):
                new_g = g + edge_cost

                if neighbor not in g_costs or new_g < g_costs[neighbor]:
                    g_costs[neighbor] = new_g
                    h = compute_heuristic(neighbor)
                    f_new = new_g + h
                    heapq.heappush(open_set, (f_new, new_g, neighbor, path + [neighbor]))

        print(f"  A* Search: Open set exhausted. No reachable goal found. (Nodes expanded: {nodes_expanded})")
        return {
            "target_node": None,
            "path": [],
            "total_cost": np.inf,
            "nodes_expanded": nodes_expanded,
            "heuristic_at_goal": None
        }

    # -------------------------------------------------------------------------
    # D. LOCAL SEARCH (HILL-CLIMBING OPTIMIZER)
    # -------------------------------------------------------------------------
    def local_search_optimization(self, initial_idx, ref_idx, weights=None, max_iterations=20):
        """
        Perform local neighborhood search to improve a candidate recommendation.

        Starting from initial_idx, evaluates all immediate neighbors in the similarity graph
        using a multi-objective balance score:
          Score = w_comp * CompSim + w_mech * MechSim + w_duct * DuctilityMargin - w_cost * CostPenalty

        Climbs to the neighbor that achieves the highest improvement.
        Terminates when no neighbor improves the score (local optimum).

        Parameters
        ----------
        initial_idx : int
            Initial candidate alloy node.
        ref_idx : int
            Reference material node to compare against.
        weights : dict, optional
            Importance weights for multi-objective scoring.
        max_iterations : int
            Maximum climbing steps.

        Returns
        -------
        dict containing:
            - 'best_node': int
            - 'best_score': float
            - 'trajectory': list of (node_idx, score) showing the optimization path
            - 'total_steps': int
        """
        if weights is None:
            weights = {
                "comp": 0.40,
                "mech": 0.35,
                "ductility": 0.15,
                "cost": 0.10
            }

        ref_row = self.df.iloc[ref_idx]

        def evaluate_node(node_idx):
            row = self.df.iloc[node_idx]

            # 1. Composition similarity
            if self.similarity_calc:
                comp_sim, _ = self.similarity_calc.compute_composition_similarity(ref_row, row)
            else:
                comp_sim = 0.5

            # 2. Mechanical property similarity
            if self.similarity_calc:
                mech_sim = self.similarity_calc.compute_mechanical_similarity(ref_row.to_dict(), row.to_dict())
                if mech_sim is None:
                    mech_sim = 0.5
            else:
                mech_sim = 0.5

            # 3. Ductility margin (normalized in [0, 1])
            duct_val = row.get("ductility_pct", 15.0)
            duct_score = min(1.0, max(0.0, duct_val / 35.0))

            # 4. Cost penalty (normalized in [0, 1])
            grade_name = row.get("name", "")
            cost_val = 1.0
            if self.cost_estimator:
                cost_info = self.cost_estimator.get_material_cost(grade_name)
                if cost_info["price_per_kg"] is not None:
                    cost_val = cost_info["price_per_kg"]
            cost_penalty = min(1.0, max(0.0, cost_val / 2.50))

            # Multi-objective balance score
            total_score = (
                weights["comp"] * comp_sim +
                weights["mech"] * mech_sim +
                weights["ductility"] * duct_score -
                weights["cost"] * cost_penalty
            )
            return round(float(total_score), 4)

        current_idx = initial_idx
        current_score = evaluate_node(current_idx)
        trajectory = [(current_idx, current_score)]

        print(f"\n--- Running Local Search from Initial Node {initial_idx} (Initial Score: {current_score:.4f}) ---")

        for step in range(1, max_iterations + 1):
            neighbors = self.adj_list.get(current_idx, [])
            if not neighbors:
                print(f"  Step {step}: No neighbors available. Stopping.")
                break

            # Evaluate all neighbors
            best_neighbor = None
            best_neighbor_score = current_score

            for neighbor_idx, cost, sim in neighbors:
                score = evaluate_node(neighbor_idx)
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor_idx

            # Check if an improvement was found
            if best_neighbor is not None and best_neighbor_score > current_score:
                print(f"  Step {step}: Improved to Node {best_neighbor} (Score: {current_score:.4f} -> {best_neighbor_score:.4f})")
                current_idx = best_neighbor
                current_score = best_neighbor_score
                trajectory.append((current_idx, current_score))
            else:
                print(f"  Step {step}: Local maximum reached (no neighbor improves score {current_score:.4f}).")
                break

        return {
            "best_node": current_idx,
            "best_score": current_score,
            "trajectory": trajectory,
            "total_steps": len(trajectory) - 1
        }


if __name__ == "__main__":
    from model_utils import load_dataset
    from material_similarity import MaterialSimilarityCalculator
    from cost_estimation import CostEstimator

    df_test = load_dataset("steel_db")
    calc = MaterialSimilarityCalculator("steel_db")
    graph = calc.build_similarity_graph(df_test, similarity_threshold=0.60)
    cost_est = CostEstimator()

    search = MaterialGraphSearch(graph, df_test, calc, cost_est)

    # Test predicate: alloy with tensile strength >= 1000 MPa
    predicate = lambda row: row.get("tensile_strength_mpa", 0) >= 1000

    bfs_res = search.breadth_first_search(start_idx=0, goal_predicate=predicate, max_depth=2)
    dfs_res = search.depth_first_search(start_idx=0, goal_predicate=predicate, max_depth=3)
    astar_res = search.a_star_search(start_idx=0, goal_predicate=predicate, required_min_ts=1000)
    local_res = search.local_search_optimization(initial_idx=bfs_res["found_goals"][0][0], ref_idx=0)

    print("\nSearch Algorithms Sanity Check: All 4 algorithms executed successfully!")
