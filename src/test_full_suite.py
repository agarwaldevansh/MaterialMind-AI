"""
test_full_suite.py — Master Comprehensive Test Suite for MaterialMind AI
Validates all syllabus-aligned algorithms, models, and application functionality.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_utils import (
    load_dataset,
    load_model_artifact,
    PROCESSED_DIR,
    MODELS_DIR,
    ASSETS_DIR
)
from material_substitution import MaterialSubstitutionEngine
from search_algorithms import MaterialGraphSearch
from cost_estimation import CostEstimator


class TestComprehensiveMaterialMindAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steel_df = load_dataset("steel_db")
        cls.matnavi_df = load_dataset("matnavi")
        cls.sub_engine = MaterialSubstitutionEngine("steel_db")
        cls.cost_estimator = CostEstimator()

    # 1. Dataset Tests
    def test_01_dataset_integrity_and_separation(self):
        """Verify dataset row/col dimensions and strict physical separation."""
        self.assertEqual(self.steel_df.shape[0], 3234)
        self.assertEqual(self.matnavi_df.shape[0], 915)
        # Ensure separate schemas
        self.assertIn("carbon_equivalent", self.steel_df.columns)
        self.assertIn("temperature_c", self.matnavi_df.columns)

    # 2. Regression Model Artifacts Tests
    def test_02_all_8_regression_algorithms_exist_and_predict(self):
        """Verify all 8 approved regression algorithms load and predict for steel_db."""
        algorithms = [
            "simple_linear_regression",
            "multiple_linear_regression",
            "polynomial_regression_deg_2",
            "ridge_regression",
            "lasso_regression",
            "elastic_net_regression",
            "decision_tree_regressor",
            "knn_regressor"
        ]
        sample_input = pd.DataFrame([{
            "c": 0.40, "mn": 0.85, "si": 0.25, "cr": 1.00, "ni": 0.10, "mo": 0.20,
            "cu": 0.15, "v": 0.05, "al": 0.02, "ti": 0.005, "s": 0.02, "p": 0.025,
            "fe": 96.5, "co": 0.0, "nb": 0.0, "b": 0.0, "n": 0.008, "zr": 0.0,
            "total_alloying_pct": 3.128, "carbon_equivalent": 0.79,
            "mn_c_ratio": 2.125, "cr_mo_total": 1.15, "ni_cr_total": 1.15
        }])

        for algo in algorithms:
            fname = f"steel_db_tensile_strength_mpa_{algo}.joblib"
            model, meta = load_model_artifact(fname, subfolder="regression")
            self.assertIsNotNone(model, f"Failed to load {fname}")
            pred = model.predict(sample_input)[0]
            self.assertIsInstance(float(pred), float)
            self.assertGreater(float(pred), 0.0)

    # 3. Obsolete Models Removed
    def test_03_obsolete_models_strictly_removed(self):
        """Verify SGD Regressor and Voting Classifier do not exist in models directory."""
        reg_dir = os.path.join(MODELS_DIR, "regression")
        cls_dir = os.path.join(MODELS_DIR, "classification")

        for f in os.listdir(reg_dir):
            self.assertNotIn("sgd", f.lower())
            self.assertNotIn("gradient_descent", f.lower())

        for f in os.listdir(cls_dir):
            self.assertNotIn("voting", f.lower())

    # 4. Classification Models Tests
    def test_04_all_8_classification_models_predict_and_proba(self):
        """Verify all 8 classification algorithms (5 base + 3 ensembles) load and predict."""
        cls_models = [
            "steel_db_strength_logistic_regression.joblib",
            "steel_db_strength_gaussian_naive_bayes.joblib",
            "steel_db_strength_k-nearest_neighbors_knn.joblib",
            "steel_db_strength_decision_tree.joblib",
            "steel_db_strength_support_vector_machine_svm.joblib",
            "steel_db_ensemble_bagging_classifier.joblib",
            "steel_db_ensemble_adaboost_classifier.joblib",
            "steel_db_ensemble_stacking_classifier.joblib"
        ]
        sample_input = pd.DataFrame([{
            "c": 0.40, "mn": 0.85, "si": 0.25, "cr": 1.00, "ni": 0.10, "mo": 0.20,
            "cu": 0.15, "v": 0.05, "al": 0.02, "ti": 0.005, "s": 0.02, "p": 0.025,
            "fe": 96.5, "co": 0.0, "nb": 0.0, "b": 0.0, "n": 0.008, "zr": 0.0,
            "total_alloying_pct": 3.128, "carbon_equivalent": 0.79,
            "mn_c_ratio": 2.125, "cr_mo_total": 1.15, "ni_cr_total": 1.15
        }])

        expected_classes = {"Low Strength", "Medium Strength", "High Strength"}

        for fname in cls_models:
            model, meta = load_model_artifact(fname, subfolder="classification")
            self.assertIsNotNone(model, f"Failed to load {fname}")
            pred = model.predict(sample_input)[0]
            self.assertIn(pred, expected_classes)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample_input)[0]
                self.assertEqual(len(probs), 3)
                self.assertAlmostEqual(float(sum(probs)), 1.0, places=2)

    # 5. Substitution and CSP Hard Rejection Audit Log Tests
    def test_05_material_substitution_and_csp_audit(self):
        """Verify normal substitution and rejection audit log structure."""
        res = self.sub_engine.find_substitutes(
            ref_identifier="AISI 4140",
            constraints={"min_tensile_strength": 800.0},
            top_k=3
        )
        self.assertGreater(len(res["recommendations"]), 0)
        self.assertIsInstance(res["rejection_summary"], list)
        self.assertGreater(len(res["rejection_summary"]), 0)

        # Over-constrained query
        res_none = self.sub_engine.find_substitutes(
            ref_identifier="AISI 4140",
            constraints={"min_tensile_strength": 5000.0},
            top_k=3
        )
        self.assertEqual(len(res_none["recommendations"]), 0)
        self.assertEqual(res_none["feasible_count"], 0)
        self.assertGreater(len(res_none["rejection_summary"]), 0)

    # 6. Graph Search Algorithms
    def test_06_graph_search_algorithms(self):
        """Verify BFS, DFS, A*, and Local Search execution."""
        adj = self.sub_engine.get_similarity_graph(threshold=0.6)
        searcher = MaterialGraphSearch(adj, self.sub_engine.df, self.sub_engine.similarity_calc, self.cost_estimator)
        bfs = searcher.breadth_first_search(20, lambda r: r.get("tensile_strength_mpa", 0) >= 800, max_depth=2)
        dfs = searcher.depth_first_search(20, lambda r: r.get("tensile_strength_mpa", 0) >= 800, max_depth=2)
        astar = searcher.a_star_search(20, lambda r: r.get("tensile_strength_mpa", 0) >= 800, required_min_ts=800)
        local = searcher.local_search_optimization(20, 20, max_iterations=5)

        self.assertGreater(bfs["visited_count"], 0)
        self.assertGreater(dfs["visited_count"], 0)
        self.assertGreater(astar["nodes_expanded"], 0)
        self.assertGreaterEqual(local["total_steps"], 0)

    # 7. Comparison Tables Consistency
    def test_07_comparison_tables_consistency(self):
        """Verify regression and classification comparison tables."""
        reg_csv = os.path.join(PROCESSED_DIR, "regression_model_comparison.csv")
        cls_csv = os.path.join(PROCESSED_DIR, "classification_model_comparison.csv")

        self.assertTrue(os.path.exists(reg_csv))
        self.assertTrue(os.path.exists(cls_csv))

        reg_df = pd.read_csv(reg_csv)
        cls_df = pd.read_csv(cls_csv)

        # 8 algorithms * 5 targets = 40 rows
        self.assertEqual(len(reg_df), 40)
        # 8 classification models = 8 rows
        self.assertEqual(len(cls_df), 8)

        # Ensure no SGD or Voting
        self.assertNotIn("Gradient Descent (SGD)", reg_df["Algorithm"].values)
        self.assertNotIn("Voting Classifier (Soft)", cls_df["Algorithm"].values)

        # Ensure ROC-AUC column exists and has values
        self.assertIn("Test_ROC_AUC", cls_df.columns)
        self.assertFalse(cls_df["Test_ROC_AUC"].isna().all())


if __name__ == "__main__":
    unittest.main()
