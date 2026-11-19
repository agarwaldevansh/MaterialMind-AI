"""
train_all.py — Master Training Orchestrator for MaterialMind AI

Executes the complete Step 2 Machine Learning Pipeline in a single coordinated run:
1. Regression (Linear Regression, Polynomial Regression, Decision Tree, SGD Gradient Descent)
   - Evaluates MAE, MSE, RMSE, R²
   - Saves models to models/regression/
   - Writes data/processed/regression_model_comparison.csv
2. Classification (Decision Tree, KNN, SVM, Gaussian Naive Bayes)
   - Categorizes strength into Low, Medium, High tiers
   - Saves models to models/classification/
3. Ensemble Learning (Voting Classifier, AdaBoost, Stacking Classifier)
   - Evaluates on identical GroupShuffleSplit test partition
   - Writes data/processed/classification_model_comparison.csv
4. Clustering (K-Means and Agglomerative Hierarchical Clustering)
   - Elbow curve and Silhouette analysis
   - Dendrogram visualization
   - Writes data/processed/clustering_model_comparison.csv
   - Saves models to models/clustering/
5. Dimensionality Reduction (PCA)
   - 2D projection and Scree plot
   - Saves models to models/pca/
6. Verification: Loads saved models and executes sample inference.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure src/ is on python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from regression_models import run_all_regressions
from ensemble_models import run_classification_and_ensembles
from clustering_models import train_clustering_models
from dimensionality_reduction import run_pca_analysis
from model_utils import load_model_artifact, MODELS_DIR


def verify_saved_models():
    """
    Smoke test: Load sample models from disk and verify inference on mock input vectors.
    """
    print("\n" + "=" * 70)
    print("VERIFYING MODEL ARTIFACT INTEGRITY (INFERENCE SMOKE TEST)")
    print("=" * 70)

    # 1. Verify Regression model
    reg_filename = "steel_db_tensile_strength_mpa_linear_regression.joblib"
    print(f"Loading Regression Model: {reg_filename}...")
    reg_model, reg_meta = load_model_artifact(reg_filename, subfolder="regression")
    assert reg_model is not None, "Failed to load regression model!"

    # Create a realistic test composition vector (AISI 4140 normalized)
    # Fe = ~96.5%, C=0.40, Mn=0.85, Cr=0.95, Mo=0.20, Si=0.25
    sample_composition = pd.DataFrame([{
        "c": 0.40, "mn": 0.85, "si": 0.25, "cr": 0.95, "ni": 0.20, "mo": 0.20,
        "cu": 0.15, "v": 0.05, "al": 0.02, "ti": 0.005, "s": 0.02, "p": 0.025,
        "fe": 96.5, "co": 0.0, "nb": 0.0, "b": 0.0, "n": 0.008, "zr": 0.0,
        "total_alloying_pct": 3.128, "carbon_equivalent": 0.79,
        "mn_c_ratio": 2.125, "cr_mo_total": 1.15, "ni_cr_total": 1.15
    }])

    pred_ts = reg_model.predict(sample_composition)[0]
    print(f"  -> Sample Prediction for AISI 4140 Tensile Strength: {pred_ts:.1f} MPa (Expected: ~900-1100 MPa)")

    # 2. Verify Classification model
    cls_filename = "steel_db_strength_support_vector_machine_svm.joblib"
    print(f"\nLoading Classification Model: {cls_filename}...")
    cls_model, cls_meta = load_model_artifact(cls_filename, subfolder="classification")
    assert cls_model is not None, "Failed to load classification model!"

    pred_cat = cls_model.predict(sample_composition)[0]
    pred_prob = cls_model.predict_proba(sample_composition)[0]
    print(f"  -> Predicted Strength Tier: '{pred_cat}' (Probabilities: {dict(zip(cls_model.classes_, [round(p, 3) for p in pred_prob]))})")

    # 3. Verify Clustering model
    cluster_filename = "steel_db_kmeans_k4.joblib"
    print(f"\nLoading Clustering Model: {cluster_filename}...")
    cluster_artifact, cluster_meta = load_model_artifact(cluster_filename, subfolder="clustering")
    km_scaler = cluster_artifact["scaler"]
    km_model = cluster_artifact["kmeans"]

    scaled_sample = km_scaler.transform(sample_composition)
    assigned_cluster = km_model.predict(scaled_sample)[0]
    print(f"  -> Assigned K-Means Composition Cluster: Cluster #{assigned_cluster}")

    # 4. Verify PCA model
    pca_filename = "steel_db_pca.joblib"
    print(f"\nLoading PCA Model: {pca_filename}...")
    pca_artifact, pca_meta = load_model_artifact(pca_filename, subfolder="pca")
    pca_scaler = pca_artifact["scaler"]
    pca_model = pca_artifact["pca"]

    sample_coords = pca_model.transform(pca_scaler.transform(sample_composition))[0]
    print(f"  -> 2D PCA Projection Coordinates: (PC1 = {sample_coords[0]:.2f}, PC2 = {sample_coords[1]:.2f})")

    print("\n✅ All saved model artifacts loaded and verified successfully!")


def main():
    print("=" * 70)
    print("MATERIALMIND AI — STEP 2 COMPLETE MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # Stage 1: Regression
    reg_df = run_all_regressions()

    # Stage 2 & 3: Classification & Ensembles
    cls_df = run_classification_and_ensembles("steel_db")

    # Stage 4: Clustering
    clust_df, _, _, _, _ = train_clustering_models("steel_db", optimal_k=4)

    # Stage 5: Dimensionality Reduction (PCA)
    pca, scaler, X_pca, exp_var, cum_var = run_pca_analysis("steel_db", n_components=10)

    # Stage 6: Model Verification & Smoke Test
    verify_saved_models()

    print("\n" + "=" * 70)
    print("🎉 STEP 2 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
