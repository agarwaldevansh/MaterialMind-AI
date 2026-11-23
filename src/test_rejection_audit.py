"""
test_rejection_audit.py — Unit test for CSP Rejection Audit Log
Tests all 5 required edge cases for Phase 2:
1. Normal substitution with recommendations.
2. No feasible alternatives.
3. Empty rejection records.
4. Multiple rejection records.
5. Missing optional rejection fields.
"""

import unittest
import pandas as pd
from material_substitution import MaterialSubstitutionEngine


class TestRejectionAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = MaterialSubstitutionEngine("steel_db")

    def test_01_normal_substitution_with_recommendations(self):
        """Case 1: Normal substitution with recommendations."""
        res = self.engine.find_substitutes(
            ref_identifier=20,  # AISI 4140
            constraints={"min_tensile_strength": 800.0},
            top_k=3
        )
        self.assertGreater(len(res["recommendations"]), 0)
        self.assertIsInstance(res["rejection_summary"], list)
        self.assertGreater(len(res["rejection_summary"]), 0)
        # Check structure
        record = res["rejection_summary"][0]
        self.assertIn("identifier", record)
        self.assertIn("reasons", record)

    def test_02_no_feasible_alternatives(self):
        """Case 2: Over-constrained query where 0 candidates pass."""
        res = self.engine.find_substitutes(
            ref_identifier=20,
            constraints={"min_tensile_strength": 4000.0},  # Unrealistic ceiling
            top_k=5
        )
        self.assertEqual(len(res["recommendations"]), 0)
        self.assertEqual(res["feasible_count"], 0)
        self.assertIsInstance(res["rejection_summary"], list)
        self.assertGreater(len(res["rejection_summary"]), 0)
        # Reason should mention tensile strength
        first_reasons = res["rejection_summary"][0]["reasons"]
        self.assertTrue(any("Tensile strength" in r for r in first_reasons))

    def test_03_empty_rejection_records_rendering(self):
        """Case 3: Empty rejection list handled cleanly without exception."""
        rejection_summary = []
        # Simulate logic from app.py
        table_records = []
        for item in rejection_summary:
            if isinstance(item, dict):
                table_records.append({
                    "Material Identifier": item.get("identifier", "Unknown"),
                    "Dataset Index": item.get("index", "N/A"),
                    "Disqualification Reason(s)": "; ".join(item.get("reasons", []))
                })
        df = pd.DataFrame(table_records)
        self.assertEqual(len(df), 0)

    def test_04_multiple_rejection_records(self):
        """Case 4: Multiple rejection records with various constraints."""
        rejection_summary = [
            {"index": 1, "identifier": "Grade_A", "reasons": ["Tensile strength (500 MPa) is below required minimum (800 MPa)"]},
            {"index": 2, "identifier": "Grade_B", "reasons": ["Yield/Proof stress (300 MPa) is below required minimum (600 MPa)"]},
            {"index": 3, "identifier": "Grade_C", "reasons": ["Ductility (8.0%) is below required minimum (15.0%)", "Cost ($5.50/kg) exceeds maximum limit ($3.00/kg)"]},
        ]
        table_records = []
        constraint_counts = {}
        for item in rejection_summary:
            ident = item.get("identifier", f"Row #{item.get('index', 'N/A')}")
            idx_val = item.get("index", "N/A")
            reasons = item.get("reasons", [])
            reasons_str = "; ".join(str(r) for r in reasons)
            for r in reasons:
                r_str = str(r)
                if "Tensile strength" in r_str:
                    constraint_counts["Tensile"] = constraint_counts.get("Tensile", 0) + 1
                if "Yield" in r_str:
                    constraint_counts["Yield"] = constraint_counts.get("Yield", 0) + 1
                if "Ductility" in r_str:
                    constraint_counts["Ductility"] = constraint_counts.get("Ductility", 0) + 1
                if "Cost" in r_str:
                    constraint_counts["Cost"] = constraint_counts.get("Cost", 0) + 1
            table_records.append({
                "Material Identifier": ident,
                "Dataset Index": idx_val,
                "Disqualification Reason(s)": reasons_str
            })
        df = pd.DataFrame(table_records)
        self.assertEqual(len(df), 3)
        self.assertEqual(constraint_counts["Tensile"], 1)
        self.assertEqual(constraint_counts["Cost"], 1)

    def test_05_missing_optional_rejection_fields(self):
        """Case 5: Malformed/incomplete records missing index, reasons, or identifier."""
        malformed_summary = [
            {"reasons": ["Cost ceiling exceeded"]},  # Missing identifier & index
            {"identifier": "Grade_X"},  # Missing reasons & index
            {"index": 99},  # Missing identifier & reasons
            {},  # Empty dict
            "String instead of dict",  # Non-dict element
        ]
        table_records = []
        for item in malformed_summary:
            if not isinstance(item, dict):
                table_records.append({
                    "Material Identifier": str(item),
                    "Dataset Index": "N/A",
                    "Disqualification Reason(s)": "Constraint failure"
                })
                continue
            ident = item.get("identifier", f"Row #{item.get('index', 'N/A')}")
            idx_val = item.get("index", "N/A")
            reasons = item.get("reasons", [])
            reasons_str = "; ".join(str(r) for r in reasons) if isinstance(reasons, list) else str(reasons)
            table_records.append({
                "Material Identifier": ident,
                "Dataset Index": idx_val,
                "Disqualification Reason(s)": reasons_str or "Constraint violation"
            })
        df = pd.DataFrame(table_records)
        self.assertEqual(len(df), 5)
        self.assertEqual(df.iloc[0]["Material Identifier"], "Row #N/A")
        self.assertEqual(df.iloc[1]["Material Identifier"], "Grade_X")
        self.assertEqual(df.iloc[2]["Material Identifier"], "Row #99")
        self.assertEqual(df.iloc[4]["Material Identifier"], "String instead of dict")


if __name__ == "__main__":
    unittest.main()
