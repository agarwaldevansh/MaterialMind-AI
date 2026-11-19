"""
material_similarity.py — Material Similarity Metrics and Graph Construction
MaterialMind AI — College Level AIML Project

PURPOSE:
Provides mathematically defensible similarity metrics between steel materials:
1. Chemical Composition Similarity: Normalized Euclidean distance on alloying elements.
2. Mechanical Property Similarity: Normalized fractional difference in measured properties.
3. Similarity Graph Construction: Nodes represent candidate materials; edges represent
   compositionally/metallurgically viable substitution transitions with explicit transition costs.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class MaterialSimilarityCalculator:
    """
    Computes compositional and mechanical similarities and constructs material graphs.
    """

    def __init__(self, dataset_name="steel_db"):
        self.dataset_name = dataset_name

        if dataset_name == "steel_db":
            self.composition_cols = [
                "c", "mn", "si", "cr", "ni", "mo", "cu", "v", "al", "ti",
                "s", "p", "fe", "co", "nb", "b", "n", "zr"
            ]
            self.id_col = "name"
            self.target_cols = ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"]
        else:  # matnavi
            self.composition_cols = [
                "c", "si", "mn", "p", "s", "ni", "cr", "mo", "cu", "v", "al", "n", "nb_ta"
            ]
            self.id_col = "alloy_code"
            self.target_cols = ["tensile_strength_mpa", "proof_stress_mpa", "elongation_pct"]

    def compute_composition_similarity(self, comp1_series, comp2_series):
        """
        Compute composition similarity between two materials in range [0, 1].

        Formula:
            d = Euclidean distance across available elemental weight percentages
            Similarity = 1.0 / (1.0 + d)
        """
        cols = [c for c in self.composition_cols if c in comp1_series and c in comp2_series]
        if not cols:
            return 0.0

        v1 = comp1_series[cols].fillna(0).to_numpy(dtype=float)
        v2 = comp2_series[cols].fillna(0).to_numpy(dtype=float)

        dist = np.linalg.norm(v1 - v2)
        sim = 1.0 / (1.0 + dist)
        return float(sim), float(dist)

    def compute_mechanical_similarity(self, prop1_dict, prop2_dict):
        """
        Compute mechanical property similarity based on fractional property divergence.

        Formula:
            Fractional delta for property P = |P1 - P2| / max(P1, P2)
            Average divergence = mean(fractional deltas)
            Similarity = 1.0 / (1.0 + divergence)
        """
        deltas = []
        for prop in self.target_cols:
            v1 = prop1_dict.get(prop)
            v2 = prop2_dict.get(prop)
            if v1 is not None and v2 is not None and not pd.isna(v1) and not pd.isna(v2):
                denom = max(abs(v1), abs(v2), 1e-5)
                deltas.append(abs(v1 - v2) / denom)

        if not deltas:
            return None  # Cannot evaluate

        mean_delta = np.mean(deltas)
        sim = 1.0 / (1.0 + mean_delta)
        return float(sim)

    def build_similarity_graph(self, df, similarity_threshold=0.65, max_neighbors=8):
        """
        Build an undirected material similarity graph as an adjacency list.

        Parameters
        ----------
        df : pd.DataFrame
            Materials dataset.
        similarity_threshold : float
            Minimum composition similarity (0 to 1) required to form an edge.
        max_neighbors : int
            Maximum number of nearest edges per node to maintain a clean, sparse graph.

        Returns
        -------
        dict
            Adjacency list mapping node_index -> list of (neighbor_index, edge_cost, similarity).
            Edge cost = 1.0 - similarity (lower cost means closer, more natural transition).
        """
        print(f"\nConstructing Material Similarity Graph ({self.dataset_name}, threshold={similarity_threshold})...")
        cols = [c for c in self.composition_cols if c in df.columns]

        X_comp = df[cols].fillna(0).to_numpy(dtype=float)
        n_samples = len(df)

        # Standardize for fair multi-element distance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_comp)

        adj_list = {i: [] for i in range(n_samples)}
        edge_count = 0

        # Build pairwise edges for nearest neighbors
        for i in range(n_samples):
            # Euclidean distance from node i to all nodes in scaled space
            diffs = X_scaled - X_scaled[i]
            dists = np.linalg.norm(diffs, axis=1)

            # Convert to similarity score in (0, 1]
            sims = 1.0 / (1.0 + dists)

            # Find candidates meeting threshold (excluding self)
            candidate_indices = np.where((sims >= similarity_threshold) & (np.arange(n_samples) != i))[0]

            # If candidates exist, sort by highest similarity and keep top max_neighbors
            if len(candidate_indices) > 0:
                cand_sims = sims[candidate_indices]
                sorted_cand = candidate_indices[np.argsort(-cand_sims)[:max_neighbors]]

                for neighbor in sorted_cand:
                    sim_val = float(sims[neighbor])
                    # Transition cost is difference: 1.0 - sim (admissible in [0, 1])
                    cost = round(1.0 - sim_val, 4)
                    adj_list[i].append((neighbor, cost, sim_val))
                    edge_count += 1

        print(f"  Graph created: {n_samples} nodes, {edge_count // 2} bidirectional edges.")
        return adj_list


if __name__ == "__main__":
    from model_utils import load_dataset
    df_sample = load_dataset("steel_db").head(50)
    calc = MaterialSimilarityCalculator("steel_db")
    graph = calc.build_similarity_graph(df_sample, similarity_threshold=0.60)
    print(f"Sample node 0 has {len(graph[0])} neighbors: {graph[0][:3]}")
