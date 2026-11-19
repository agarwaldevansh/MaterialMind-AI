r"""
constraint_satisfaction.py — Constraint Satisfaction Problem (CSP) Engine for MaterialMind AI

CSP FORMULATION:
- Variables: Candidate materials $M \in \mathcal{D}$ with properties:
  {tensile_strength, yield_strength, ductility, cost, composition, temperature}
- Domain: All available records in the chosen processed dataset.
- Constraints:
  1. $C_1$: $\text{Tensile Strength} \ge \text{required\_min\_tensile}$
  2. $C_2$: $\text{Yield / Proof Stress} \ge \text{required\_min\_yield}$ (if specified)
  3. $C_3$: $\text{Ductility / Elongation} \ge \text{required\_min\_ductility}$ (if specified)
  4. $C_4$: $\text{Estimated Cost} \le \text{max\_acceptable\_cost}$ (if specified and cost available/mandatory)
  5. $C_5$: Composition restrictions (e.g. max_carbon, min_chromium, low_sulfur)
  6. $C_6$: Operating temperature compatibility (for MatNavi dataset)

RULE: A missing required property NEVER automatically satisfies a constraint.
Every rejected material is logged with the exact constraint(s) it failed.
"""

import pandas as pd
import numpy as np


class MaterialCSPFilter:
    """
    Executes hard constraint satisfaction filtering on candidate materials.
    """

    def __init__(self, cost_estimator=None):
        self.cost_estimator = cost_estimator

    def filter_candidates(self, df, constraints, dataset_name="steel_db"):
        """
        Evaluate all materials in df against specified constraints.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to search (steel_db or matnavi).
        constraints : dict
            Dictionary specifying active constraints:
            - 'min_tensile_strength': float (MPa)
            - 'min_yield_strength': float (MPa)
            - 'min_ductility': float (%)
            - 'max_cost_usd_kg': float ($/kg)
            - 'cost_mandatory': bool (if True, materials with unavailable price are rejected)
            - 'max_carbon': float (wt%)
            - 'min_chromium': float (wt%)
            - 'target_temperature_c': float (°C) [for MatNavi]
            - 'exclude_index': int (optional, index of reference material to exclude from candidates)
        dataset_name : str
            'steel_db' or 'matnavi'

        Returns
        -------
        feasible_df : pd.DataFrame
            Subset of materials that satisfy ALL mandatory hard constraints.
        rejected_records : list of dict
            List containing index, name, and exact human-readable reasons for rejection.
        """
        if dataset_name == "steel_db":
            ts_col = "tensile_strength_mpa"
            ys_col = "yield_strength_mpa"
            duct_col = "ductility_pct"
            name_col = "name"
        else:
            ts_col = "tensile_strength_mpa"
            ys_col = "proof_stress_mpa"
            duct_col = "elongation_pct"
            name_col = "alloy_code"

        feasible_indices = []
        rejected_records = []

        # Constraint parameters
        min_ts = constraints.get("min_tensile_strength")
        min_ys = constraints.get("min_yield_strength")
        min_duct = constraints.get("min_ductility")
        max_cost = constraints.get("max_cost_usd_kg")
        cost_mandatory = constraints.get("cost_mandatory", False)
        max_c = constraints.get("max_carbon")
        min_cr = constraints.get("min_chromium")
        target_temp = constraints.get("target_temperature_c")
        exclude_idx = constraints.get("exclude_index")

        for idx, row in df.iterrows():
            # Skip reference material if requested
            if exclude_idx is not None and idx == exclude_idx:
                continue

            # Skip flagged outliers in MatNavi
            if row.get("is_outlier", False):
                rejected_records.append({
                    "index": idx,
                    "identifier": row.get(name_col, f"Row_{idx}"),
                    "reasons": ["Flagged physical outlier in dataset (unreliable property data)"]
                })
                continue

            failures = []

            # 1. Tensile Strength Constraint
            if min_ts is not None:
                val = row.get(ts_col)
                if pd.isna(val) or val is None:
                    failures.append(f"Tensile strength data is missing (required >= {min_ts} MPa)")
                elif val < min_ts:
                    failures.append(f"Tensile strength ({val:.0f} MPa) is below required minimum ({min_ts} MPa)")

            # 2. Yield Strength Constraint
            if min_ys is not None:
                val = row.get(ys_col)
                if pd.isna(val) or val is None:
                    failures.append(f"Yield/Proof stress data is missing (required >= {min_ys} MPa)")
                elif val < min_ys:
                    failures.append(f"Yield/Proof stress ({val:.0f} MPa) is below required minimum ({min_ys} MPa)")

            # 3. Ductility Constraint
            if min_duct is not None:
                val = row.get(duct_col)
                if pd.isna(val) or val is None:
                    failures.append(f"Ductility/Elongation data is missing (required >= {min_duct}%)")
                elif val < min_duct:
                    failures.append(f"Ductility ({val:.1f}%) is below required minimum ({min_duct}%)")

            # 4. Cost Constraint
            if max_cost is not None and self.cost_estimator is not None:
                grade_name = row.get("name", "")
                cost_info = self.cost_estimator.get_material_cost(grade_name)
                price_val = cost_info["price_per_kg"]

                if price_val is None:
                    if cost_mandatory:
                        failures.append(f"Cost information is unavailable, but cost limit is mandatory (${max_cost:.2f}/kg)")
                else:
                    if price_val > max_cost:
                        failures.append(f"Estimated cost (${price_val:.2f}/kg) exceeds maximum limit (${max_cost:.2f}/kg)")

            # 5. Composition Restrictions: Max Carbon
            if max_c is not None:
                c_val = row.get("c")
                if pd.isna(c_val) or c_val is None:
                    failures.append(f"Carbon content is missing (required <= {max_c} wt%)")
                elif c_val > max_c:
                    failures.append(f"Carbon content ({c_val:.2f} wt%) exceeds allowable maximum ({max_c} wt%)")

            # 5. Composition Restrictions: Min Chromium
            if min_cr is not None:
                cr_val = row.get("cr")
                if pd.isna(cr_val) or cr_val is None:
                    failures.append(f"Chromium content is missing (required >= {min_cr} wt%)")
                elif cr_val < min_cr:
                    failures.append(f"Chromium content ({cr_val:.2f} wt%) is below required minimum ({min_cr} wt%)")

            # 6. Operating Temperature (MatNavi)
            if target_temp is not None and "temperature_c" in row:
                t_val = row.get("temperature_c")
                # Allow a small ±25°C temperature window
                if abs(t_val - target_temp) > 25:
                    failures.append(f"Test temperature ({t_val}°C) does not match required operating temperature ({target_temp}°C)")

            # Conclusion on candidate
            if not failures:
                feasible_indices.append(idx)
            else:
                rejected_records.append({
                    "index": idx,
                    "identifier": row.get(name_col, f"Row_{idx}"),
                    "reasons": failures
                })

        feasible_df = df.loc[feasible_indices].copy()
        return feasible_df, rejected_records


if __name__ == "__main__":
    from model_utils import load_dataset
    from cost_estimation import CostEstimator

    df_test = load_dataset("steel_db")
    cost_est = CostEstimator()
    csp = MaterialCSPFilter(cost_est)

    constraints = {
        "min_tensile_strength": 900,
        "min_ductility": 15,
        "max_cost_usd_kg": 1.50
    }

    feasible, rejected = csp.filter_candidates(df_test, constraints, "steel_db")
    print(f"Total candidates: {len(df_test)}")
    print(f"Feasible candidates satisfying all constraints: {len(feasible)}")
    print(f"Rejected candidates: {len(rejected)}")
    print("Sample rejection explanation:")
    print(rejected[0])
