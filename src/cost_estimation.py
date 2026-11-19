"""
cost_estimation.py — Material Cost Lookup and Comparison
MaterialMind AI — College Level AIML Project

PURPOSE:
Provides transparent, auditable cost estimation for steel grades:
1. Reuses illustrative market price ranges from data/raw/illustrative_supplier_reference.csv.
2. Supports user-provided price overrides.
3. Explicitly labels the provenance and date of every price estimate (e.g., "January 2024 Illustrative Estimate").
4. Returns "Price unavailable" when no reliable quotation exists, rather than guessing or fabricating numbers.
"""

import os
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "illustrative_supplier_reference.csv")


class CostEstimator:
    """
    Handles cost estimation, range parsing, and provenance tracking for steel materials.
    """

    def __init__(self, reference_csv_path=None):
        if reference_csv_path is None:
            reference_csv_path = REF_FILE

        self.reference_csv_path = reference_csv_path
        self.user_prices = {}  # In-memory store for user-entered overrides {standardized_grade: float}
        self.reference_df = self._load_reference_data()

    def get_all_reference_materials(self):
        """
        Return a copy of the reference materials table.
        """
        return self.reference_df.copy()

    def _load_reference_data(self):
        """Load and parse the reference supplier table."""
        if not os.path.exists(self.reference_csv_path):
            print(f"Warning: Cost reference file not found at {self.reference_csv_path}")
            return pd.DataFrame()

        df = pd.read_csv(self.reference_csv_path)
        # Standardize grade key: e.g. "AISI 4140" -> "4140"
        df["std_grade"] = df["grade"].apply(self._standardize_grade_name)
        return df

    @staticmethod
    def _standardize_grade_name(name):
        """
        Normalize material grade string to match reference keys.
        e.g. 'AISI 4140 Steel' -> '4140', 'AISI 1020' -> '1020'
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
        cleaned = name.upper().replace("STEEL", "").strip()
        # Extract alphanumeric core identifier (e.g. 4140, 1020, 12L14, A36, etc.)
        match = re.search(r'(?:AISI\s*|ASTM\s*|SAE\s*)?([0-9A-Z]+(?:L[0-9]+)?)', cleaned)
        if match:
            return match.group(1).strip()
        return cleaned

    def set_user_price(self, grade_name, price_usd_per_kg):
        """
        Register a user-provided price quotation for a material grade.
        """
        std_grade = self._standardize_grade_name(grade_name)
        if price_usd_per_kg is not None and price_usd_per_kg > 0:
            self.user_prices[std_grade] = float(price_usd_per_kg)
            print(f"Registered user price for {grade_name} (key: {std_grade}): ${price_usd_per_kg:.2f}/kg")
        else:
            if std_grade in self.user_prices:
                del self.user_prices[std_grade]

    def get_material_cost(self, grade_name):
        """
        Retrieve cost information for a steel grade.

        Returns
        -------
        dict containing:
            - 'price_per_kg': float or None (numeric midpoint or user quote)
            - 'price_display': str formatted display string (e.g. "$0.80 - $1.20 / kg")
            - 'price_status': 'user_provided', 'illustrative_estimate', or 'unavailable'
            - 'provenance_note': explanation of the estimate's origin
            - 'reference_date': date of reference quotation or 'Current'
        """
        std_grade = self._standardize_grade_name(grade_name)

        # 1. Check for user-provided price first
        if std_grade in self.user_prices:
            val = self.user_prices[std_grade]
            return {
                "price_per_kg": val,
                "price_display": f"${val:.2f} / kg",
                "price_status": "user_provided",
                "provenance_note": "User-entered current market price quotation",
                "reference_date": "User input (Current)"
            }

        # 2. Check in illustrative reference table
        if not self.reference_df.empty:
            match = self.reference_df[self.reference_df["std_grade"] == std_grade]
            if not match.empty:
                row = match.iloc[0]
                price_str = str(row["indicative_price_usd_per_kg"])
                # Parse range like '0.80-1.20'
                parts = [float(p.strip()) for p in price_str.split("-") if p.strip()]
                midpoint = np.mean(parts) if parts else None

                return {
                    "price_per_kg": midpoint,
                    "price_display": f"${price_str} / kg (est.)",
                    "price_status": "illustrative_estimate",
                    "provenance_note": str(row.get("estimate_nature", "Illustrative reference estimate")),
                    "reference_date": str(row.get("reference_date", "2024-01"))
                }

        # 3. Fallback: No cost data available
        return {
            "price_per_kg": None,
            "price_display": "Price unavailable",
            "price_status": "unavailable",
            "provenance_note": "No verified supplier price or public reference quote found for this grade",
            "reference_date": "N/A"
        }

    def compare_costs(self, ref_grade, candidate_grades):
        """
        Compare the cost of a reference material against a list of candidate substitutes.

        Returns
        -------
        pd.DataFrame
            Comparison table showing relative cost delta (%) and status.
        """
        ref_cost = self.get_material_cost(ref_grade)
        ref_price = ref_cost["price_per_kg"]

        records = []
        for cand in candidate_grades:
            cand_cost = self.get_material_cost(cand)
            cand_price = cand_cost["price_per_kg"]

            if ref_price is not None and cand_price is not None and ref_price > 0:
                delta_pct = ((cand_price - ref_price) / ref_price) * 100
                cost_impact = f"{delta_pct:+.1f}%"
            else:
                cost_impact = "Cannot calculate"

            records.append({
                "Candidate": cand,
                "Price_Display": cand_cost["price_display"],
                "Cost_Impact_vs_Ref": cost_impact,
                "Price_Status": cand_cost["price_status"],
                "Provenance": cand_cost["provenance_note"]
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    estimator = CostEstimator()
    print("Testing Cost Estimator:")
    for grade in ["AISI 1020 Steel", "AISI 4140 Steel", "AISI 4340", "Unknown Alloy X"]:
        c = estimator.get_material_cost(grade)
        print(f"  {grade:<18} -> {c['price_display']} [{c['price_status']}] ({c['provenance_note']})")
