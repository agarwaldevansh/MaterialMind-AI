"""
material_substitution.py — Integrated Material Substitution Engine
MaterialMind AI — College Level AIML Project

PURPOSE:
Provides an end-to-end recommendation workflow for identifying substitute steel grades:
1. Formulates user requirements into hard constraints (CSP).
2. Filters candidate materials and records exact rejection reasons.
3. Computes chemical and mechanical similarity against the reference material.
4. Ranks feasible candidates using a transparent multi-objective scoring function.
5. Employs Step 2 trained ML models to generate predictions when measured properties are missing.
6. Clearly labels every property as "Measured", "Model Prediction", or "Unavailable".
7. Integrates cost estimation (User price, Jan 2024 illustrative estimate, or "Price unavailable").
8. Provides human-readable engineering explanations and safety disclaimers for every recommendation.
"""

import os
import sys
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_utils import load_dataset, load_model_artifact, ASSETS_DIR, PROCESSED_DIR
from cost_estimation import CostEstimator
from material_similarity import MaterialSimilarityCalculator
from constraint_satisfaction import MaterialCSPFilter
from search_algorithms import MaterialGraphSearch

SAFETY_DISCLAIMER = (
    "ENGINEERING SAFETY DISCLAIMER: This material substitution recommendation is generated strictly "
    "for educational and preliminary screening decision-support. Real-world engineering substitution "
    "requires certified laboratory qualification, metallurgical microstructure verification, fatigue "
    "and fracture toughness testing, corrosion assessment, and full compliance with governing engineering "
    "standards (e.g. ASTM, ASME, ISO, SAE)."
)


class MaterialSubstitutionEngine:
    """
    Main recommendation engine combining CSP filtering, ML predictions,
    similarity graph navigation, cost tracking, and natural language explanations.
    """

    def __init__(self, dataset_name="steel_db"):
        self.dataset_name = dataset_name
        self.df = load_dataset(dataset_name)

        # Reusable modules
        self.cost_estimator = CostEstimator()
        self.similarity_calc = MaterialSimilarityCalculator(dataset_name)
        self.csp_filter = MaterialCSPFilter(self.cost_estimator)

        # Pre-load Step 2 ML Decision Tree Regressor for tensile strength prediction
        self.ml_model = None
        self.ml_features = []
        self._load_step2_ml_model()

        # Cache similarity graph for fast search algorithm queries
        self.graph = None

    def _load_step2_ml_model(self):
        """Load trained Step 2 Decision Tree Regressor for tensile strength predictions."""
        try:
            model_file = f"{self.dataset_name}_tensile_strength_mpa_decision_tree_regressor.joblib"
            model, meta = load_model_artifact(model_file, subfolder="regression")
            self.ml_model = model
            self.ml_features = meta.get("feature_columns", [])
            print(f"Loaded Step 2 ML Model: {model_file} ({len(self.ml_features)} features)")
        except Exception as e:
            print(f"Note: Could not load Step 2 ML model ({e}). Continuing with measured data only.")
            self.ml_model = None
            self.ml_features = []

    def get_similarity_graph(self, threshold=0.60):
        """Lazy-load similarity graph adjacency list."""
        if self.graph is None:
            self.graph = self.similarity_calc.build_similarity_graph(self.df, similarity_threshold=threshold)
        return self.graph

    def find_substitutes(self, ref_identifier, constraints=None, weights=None, top_k=5, run_graph_search=False, allow_same_grade=True):
        """
        Execute full material substitution workflow.

        Parameters
        ----------
        ref_identifier : str or int
            Reference material name (e.g., 'AISI 4140 Steel') or row index.
        constraints : dict
            User hard constraints (min_tensile, min_yield, min_ductility, max_cost, etc.).
        weights : dict, optional
            Soft preference weights for ranking.
        top_k : int
            Number of recommendations to return.
        run_graph_search : bool
            Whether to also run BFS, DFS, and A* search demonstrations.
        allow_same_grade : bool
            If False, excludes different heat-treatment conditions of the SAME grade
            to force finding alternative steel grades (e.g., finding AISI 4340 instead of AISI 4140).

        Returns
        -------
        dict containing:
            - 'reference_material': dict
            - 'recommendations': list of dicts (top candidates with metrics & explanations)
            - 'total_candidates_evaluated': int
            - 'feasible_count': int
            - 'rejection_summary': list of sample rejections with reasons
            - 'graph_search_results': dict (if run_graph_search is True)
            - 'disclaimer': str
        """
        if constraints is None:
            constraints = {}

        # 1. Identify Reference Material
        ref_idx, ref_row = self._resolve_reference_material(ref_identifier)
        if ref_row is None:
            raise ValueError(f"Reference material '{ref_identifier}' could not be located in {self.dataset_name}.")

        print(f"\n==================================================================")
        print(f"MATERIALMIND AI: Material Substitution Search")
        print(f"Reference Material: {ref_row.get('name', ref_row.get('alloy_code', 'Ref'))} [Index {ref_idx}]")
        print(f"==================================================================")

        # Auto-populate constraints from reference if not explicitly set by user
        constraints = constraints.copy()
        constraints["exclude_index"] = ref_idx

        # 2. Hard Constraint Satisfaction Problem (CSP) Filtering
        feasible_df, rejected_records = self.csp_filter.filter_candidates(
            self.df, constraints, self.dataset_name
        )

        # Option to exclude records of the exact same grade
        if not allow_same_grade:
            ref_name = ref_row.get("name", ref_row.get("alloy_code", ""))
            name_col = "name" if "name" in feasible_df.columns else "alloy_code"
            feasible_df = feasible_df[feasible_df[name_col] != ref_name].copy()
            print(f"  Filtered out same-grade records. Remaining distinct alternative candidates: {len(feasible_df)}")

        total_eval = len(self.df) - 1
        feasible_count = len(feasible_df)
        print(f"CSP Hard Filtering: {feasible_count}/{total_eval} materials passed all constraints.")

        if feasible_df.empty:
            print("No candidate materials satisfied all mandatory constraints.")
            return {
                "reference_material": self._format_material_card(ref_row, ref_idx, is_reference=True),
                "recommendations": [],
                "total_candidates_evaluated": total_eval,
                "feasible_count": 0,
                "rejection_summary": rejected_records[:10],
                "graph_search_results": None,
                "disclaimer": SAFETY_DISCLAIMER
            }

        # 3. Soft Preference Multi-Criteria Scoring & Ranking
        scored_candidates = self._rank_candidates(feasible_df, ref_row, constraints, weights)

        # Select Top-K
        top_candidates = scored_candidates[:top_k]

        # 4. Optional Graph Search Demonstration (BFS, DFS, A*, Local Search)
        graph_results = None
        if run_graph_search:
            graph_results = self._execute_graph_searches(ref_idx, constraints)

        # 5. Format Output Recommendations with Explanations
        recommendations = []
        for cand in top_candidates:
            card = self._build_recommendation_card(cand, ref_row, constraints)
            recommendations.append(card)

        return {
            "reference_material": self._format_material_card(ref_row, ref_idx, is_reference=True),
            "recommendations": recommendations,
            "total_candidates_evaluated": total_eval,
            "feasible_count": feasible_count,
            "rejection_summary": rejected_records[:5],
            "graph_search_results": graph_results,
            "disclaimer": SAFETY_DISCLAIMER
        }

    def _resolve_reference_material(self, ref_identifier):
        """Find the reference material by index or string name."""
        if isinstance(ref_identifier, int):
            if 0 <= ref_identifier < len(self.df):
                return ref_identifier, self.df.iloc[ref_identifier]
            return None, None

        if isinstance(ref_identifier, str):
            id_col = "name" if "name" in self.df.columns else "alloy_code"
            # Exact match first
            matches = self.df[self.df[id_col].str.upper() == ref_identifier.strip().upper()]
            if not matches.empty:
                return matches.index[0], matches.iloc[0]

            # Substring match
            matches = self.df[self.df[id_col].str.upper().str.contains(ref_identifier.strip().upper(), na=False)]
            if not matches.empty:
                return matches.index[0], matches.iloc[0]

        return None, None

    def _rank_candidates(self, feasible_df, ref_row, constraints, weights=None):
        """
        Rank feasible candidates based on multi-criteria soft preferences.

        Score = w_comp * CompSim + w_mech * MechSim + w_margin * MarginScore - w_cost * CostPenalty
        """
        if weights is None:
            weights = {
                "comp_sim": 0.35,
                "mech_sim": 0.35,
                "duct_margin": 0.15,
                "cost_adv": 0.15
            }

        ref_ts = ref_row.get("tensile_strength_mpa", 700.0)
        ref_cost_info = self.cost_estimator.get_material_cost(ref_row.get("name", ""))
        ref_price = ref_cost_info.get("price_per_kg")

        scored = []
        for idx, cand_row in feasible_df.iterrows():
            # 1. Composition similarity
            comp_sim, comp_dist = self.similarity_calc.compute_composition_similarity(ref_row, cand_row)

            # 2. Mechanical property similarity
            mech_sim = self.similarity_calc.compute_mechanical_similarity(ref_row.to_dict(), cand_row.to_dict())
            if mech_sim is None:
                mech_sim = 0.5  # Neutral fallback

            # 3. Ductility / strength reserve margin
            cand_duct = cand_row.get("ductility_pct", cand_row.get("elongation_pct", 15.0))
            margin_score = min(1.0, max(0.0, cand_duct / 30.0))

            # 4. Cost penalty / advantage
            cand_grade = cand_row.get("name", cand_row.get("alloy_code", ""))
            cand_cost_info = self.cost_estimator.get_material_cost(cand_grade)
            cand_price = cand_cost_info.get("price_per_kg")

            if ref_price is not None and cand_price is not None and ref_price > 0:
                # Advantage if cheaper than reference
                cost_ratio = (ref_price - cand_price) / ref_price
                cost_score = min(1.0, max(-1.0, cost_ratio))
            else:
                cost_score = 0.0  # Neutral if unknown

            total_score = (
                weights["comp_sim"] * comp_sim +
                weights["mech_sim"] * mech_sim +
                weights["duct_margin"] * margin_score +
                weights["cost_adv"] * cost_score
            )

            scored.append({
                "index": idx,
                "row": cand_row,
                "total_score": round(float(total_score), 4),
                "comp_sim": round(float(comp_sim), 4),
                "mech_sim": round(float(mech_sim), 4),
                "cost_info": cand_cost_info
            })

        # Sort descending by total score
        scored.sort(key=lambda x: x["total_score"], reverse=True)
        return scored

    def _build_recommendation_card(self, cand_item, ref_row, constraints):
        """Construct a structured dictionary and natural-language explanation for a recommended material."""
        row = cand_item["row"]
        idx = cand_item["index"]
        score = cand_item["total_score"]
        comp_sim = cand_item["comp_sim"]
        mech_sim = cand_item["mech_sim"]
        cost_info = cand_item["cost_info"]

        name = row.get("name", row.get("alloy_code", f"Material_{idx}"))
        condition = row.get("processing_condition", "Standard Condition")

        # Measured properties
        ts = row.get("tensile_strength_mpa")
        ys = row.get("yield_strength_mpa", row.get("proof_stress_mpa"))
        duct = row.get("ductility_pct", row.get("elongation_pct"))

        # Model Prediction if required or as verification
        predicted_ts = None
        if self.ml_model and self.ml_features:
            try:
                features_df = pd.DataFrame([row[self.ml_features].to_dict()])
                predicted_ts = round(float(self.ml_model.predict(features_df)[0]), 1)
            except Exception:
                predicted_ts = None

        # Property Deltas vs Reference
        ref_ts = ref_row.get("tensile_strength_mpa")
        ref_ys = ref_row.get("yield_strength_mpa", ref_row.get("proof_stress_mpa"))
        ref_duct = ref_row.get("ductility_pct", ref_row.get("elongation_pct"))

        delta_ts = (ts - ref_ts) if (ts is not None and ref_ts is not None) else None
        delta_ys = (ys - ref_ys) if (ys is not None and ref_ys is not None) else None
        delta_duct = (duct - ref_duct) if (duct is not None and ref_duct is not None) else None

        # Key composition elements
        key_elements = {
            "C": row.get("c"),
            "Mn": row.get("mn"),
            "Si": row.get("si"),
            "Cr": row.get("cr"),
            "Ni": row.get("ni"),
            "Mo": row.get("mo")
        }

        # Build natural-language explanation
        explanation = self._generate_explanation(
            name, row, ref_row, constraints, score, comp_sim, mech_sim, delta_ts, delta_ys, delta_duct, cost_info
        )

        return {
            "material_name": name,
            "dataset_index": idx,
            "dataset_source": self.dataset_name,
            "processing_condition": condition,
            "suitability_score": score,
            "composition_similarity": comp_sim,
            "mechanical_similarity": mech_sim,
            "chemical_composition": {k: round(v, 3) for k, v in key_elements.items() if v is not None and not pd.isna(v)},
            "measured_properties": {
                "tensile_strength_mpa": ts,
                "yield_strength_mpa": ys,
                "ductility_pct": duct,
                "data_nature": "Measured experimental data from dataset"
            },
            "predicted_properties": {
                "tensile_strength_mpa": predicted_ts,
                "model_used": "Step 2 Decision Tree Regressor" if predicted_ts else "None",
                "data_nature": "Model prediction (distinguished from measured)" if predicted_ts else "N/A"
            },
            "property_deltas_vs_reference": {
                "delta_tensile_mpa": f"{delta_ts:+.1f} MPa" if delta_ts is not None else "N/A",
                "delta_yield_mpa": f"{delta_ys:+.1f} MPa" if delta_ys is not None else "N/A",
                "delta_ductility_pct": f"{delta_duct:+.1f}%" if delta_duct is not None else "N/A"
            },
            "cost_information": cost_info,
            "explanation": explanation
        }

    def _generate_explanation(self, name, row, ref_row, constraints, score, comp_sim, mech_sim, delta_ts, delta_ys, delta_duct, cost_info):
        """Generate human-readable engineering reasoning for this recommendation."""
        lines = []

        # 1. Why selected
        lines.append(f"• Selection Rationale: {name} satisfies all mandatory constraints with a suitability score of {score:.2f} (Composition similarity: {comp_sim*100:.1f}%, Mechanical similarity: {mech_sim*100:.1f}%).")

        # 2. Property comparisons
        ts = row.get("tensile_strength_mpa", 0)
        ys = row.get("yield_strength_mpa", row.get("proof_stress_mpa", 0))
        lines.append(f"• Mechanical Profile: Recorded tensile strength of {ts:.0f} MPa and yield strength of {ys:.0f} MPa.")

        if delta_ts is not None:
            if delta_ts >= 0:
                lines.append(f"• Strength Margin: Provides a {delta_ts:+.0f} MPa increase in tensile strength compared to the reference material.")
            else:
                lines.append(f"• Strength Trade-off: Tensile strength is {abs(delta_ts):.0f} MPa lower than the reference, but strictly within acceptable design bounds.")

        if delta_duct is not None:
            lines.append(f"• Ductility Margin: Ductility difference is {delta_duct:+.1f}% relative to reference.")

        # 3. Cost status
        if cost_info["price_status"] == "user_provided":
            lines.append(f"• Cost Status: User-entered live price is {cost_info['price_display']}.")
        elif cost_info["price_status"] == "illustrative_estimate":
            lines.append(f"• Cost Status: Estimated indicative range is {cost_info['price_display']} (Reference date: {cost_info['reference_date']}, approximate public data).")
        else:
            lines.append("• Cost Status: Verified price unavailable for this specific grade; commercial quotation required.")

        # 4. Critical Engineering Considerations
        cond = row.get("processing_condition")
        if cond and not pd.isna(cond):
            lines.append(f"• Processing Consideration: Property values correspond to heat treatment: '{cond}'. Heat treatment must be matched during fabrication.")

        return "\n".join(lines)

    def _format_material_card(self, row, idx, is_reference=False):
        """Format a concise overview card for reference material."""
        name = row.get("name", row.get("alloy_code", f"Material_{idx}"))
        cost = self.cost_estimator.get_material_cost(name)

        return {
            "name": name,
            "index": idx,
            "dataset": self.dataset_name,
            "processing_condition": row.get("processing_condition", "Standard"),
            "tensile_strength_mpa": row.get("tensile_strength_mpa"),
            "yield_strength_mpa": row.get("yield_strength_mpa", row.get("proof_stress_mpa")),
            "ductility_pct": row.get("ductility_pct", row.get("elongation_pct")),
            "cost_display": cost["price_display"],
            "cost_status": cost["price_status"]
        }

    def _execute_graph_searches(self, start_idx, constraints):
        """Run BFS, DFS, A*, and Local Search demonstrations on the similarity graph."""
        graph = self.get_similarity_graph(threshold=0.60)
        searcher = MaterialGraphSearch(graph, self.df, self.similarity_calc, self.cost_estimator)

        min_ts = constraints.get("min_tensile_strength", 600.0)
        predicate = lambda r: (r.get("tensile_strength_mpa", 0) >= min_ts and not r.get("is_outlier", False))

        # BFS
        bfs_out = searcher.breadth_first_search(start_idx, predicate, max_depth=2)

        # DFS
        dfs_out = searcher.depth_first_search(start_idx, predicate, max_depth=3)

        # A* Search
        astar_out = searcher.a_star_search(start_idx, predicate, required_min_ts=min_ts)

        # Local Search
        initial_node = None
        if bfs_out["found_goals"]:
            initial_node = bfs_out["found_goals"][0][0]
        elif astar_out["target_node"] is not None:
            initial_node = astar_out["target_node"]
        else:
            initial_node = (start_idx + 1) % len(self.df)

        local_out = searcher.local_search_optimization(initial_node, start_idx)

        return {
            "bfs": {
                "visited_nodes": bfs_out["visited_count"],
                "qualifying_goals_found": len(bfs_out["found_goals"]),
                "sample_goals": [g[0] for g in bfs_out["found_goals"][:3]]
            },
            "dfs": {
                "visited_nodes": dfs_out["visited_count"],
                "qualifying_goals_found": len(dfs_out["found_goals"]),
                "sample_goals": [g[0] for g in dfs_out["found_goals"][:3]]
            },
            "a_star": {
                "goal_found": astar_out["target_node"] is not None,
                "target_node": astar_out["target_node"],
                "path_length": len(astar_out["path"]),
                "total_path_cost": astar_out["total_cost"],
                "nodes_expanded": astar_out["nodes_expanded"]
            },
            "local_search": {
                "initial_node": initial_node,
                "best_node": local_out["best_node"],
                "initial_score": local_out["trajectory"][0][1],
                "optimized_score": local_out["best_score"],
                "improvement_steps": local_out["total_steps"]
            }
        }


if __name__ == "__main__":
    engine = MaterialSubstitutionEngine("steel_db")

    # Example query: Substitute for AISI 4140 with min TS = 950 MPa, min ductility = 15%
    req_constraints = {
        "min_tensile_strength": 950,
        "min_ductility": 15,
        "max_cost_usd_kg": 2.00
    }

    result = engine.find_substitutes("AISI 4140 Steel", req_constraints, top_k=3, run_graph_search=True)
    print("\n--- RESULTS PREVIEW ---")
    print(f"Reference: {result['reference_material']['name']} ({result['reference_material']['tensile_strength_mpa']} MPa)")
    print(f"Feasible candidates found: {result['feasible_count']}")

    for i, rec in enumerate(result["recommendations"], 1):
        print(f"\nRecommendation #{i}: {rec['material_name']} (Score: {rec['suitability_score']})")
        print(f"  Condition: {rec['processing_condition']}")
        print(f"  Measured: TS={rec['measured_properties']['tensile_strength_mpa']} MPa, YS={rec['measured_properties']['yield_strength_mpa']} MPa, Duct={rec['measured_properties']['ductility_pct']}%")
        if rec['predicted_properties']['tensile_strength_mpa']:
            print(f"  Model Prediction (Step 2 DT): {rec['predicted_properties']['tensile_strength_mpa']} MPa")
        print(f"  Cost: {rec['cost_information']['price_display']} [{rec['cost_information']['price_status']}]")
        print(f"  Explanation summary:\n{rec['explanation']}")
