"""
ensemble_models.py — Ensemble Learning for Material Strength Classification
MaterialMind AI — College Level AIML Project

Implements three ensemble architectures on the material strength classification task:
1. Voting Classifier:
   - Combines predictions from multiple distinct base classifiers (Decision Tree, KNN, SVM, GaussianNB).
   - Uses soft voting (combining predicted class probability distributions).
2. AdaBoost Classifier:
   - Sequential boosting using shallow decision stumps (max_depth=2).
   - Iteratively shifts sample weights toward previously misclassified material compositions.
3. Stacking Classifier:
   - Blends base models using out-of-fold cross-validation predictions (cv=5).
   - Meta-learner (LogisticRegression) learns how to weight each model's strengths.

Comparative Evaluation:
- Compares base vs. ensemble models on the exact same GroupShuffleSplit test set.
- Generates combined model comparison table (`data/processed/classification_model_comparison.csv`).
- Evaluates whether ensemble learning actually improves performance or causes unnecessary overhead.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, StackingClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_utils import (
    load_dataset,
    split_data_grouped,
    create_strength_categories,
    evaluate_classification,
    save_model_artifact,
    ASSETS_DIR,
    PROCESSED_DIR
)
from classification_models import train_and_evaluate_classifiers, STRENGTH_CLASSES


def build_ensemble_models():
    """
    Construct ensemble estimators using pre-configured pipelines.
    """
    # Base estimators for Voting and Stacking
    dt_base = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)

    knn_base = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ])

    svm_base = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=2.0, probability=True, random_state=42))
    ])

    gnb_base = Pipeline([
        ("scaler", StandardScaler()),
        ("gnb", GaussianNB())
    ])

    base_list = [
        ("dt", dt_base),
        ("knn", knn_base),
        ("svm", svm_base),
        ("gnb", gnb_base)
    ]

    ensembles = {
        # 1. Soft Voting Classifier
        "Voting Classifier (Soft)": VotingClassifier(
            estimators=base_list,
            voting="soft"
        ),

        # 2. AdaBoost Classifier with shallow decision tree
        "AdaBoost Classifier": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=50,
            learning_rate=0.8,
            random_state=42
        ),

        # 3. Stacking Classifier with out-of-fold CV predictions
        "Stacking Classifier": StackingClassifier(
            estimators=base_list,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            n_jobs=1
        )
    }

    return ensembles


def run_classification_and_ensembles(dataset_name="steel_db"):
    """
    Execute full classification experiment:
    1. Train 4 base classifiers.
    2. Train 3 ensemble models.
    3. Compare all 7 models side-by-side on identical test data.
    4. Save models, metrics, and visualization plots.
    """
    print("\n" + "=" * 70)
    print("STARTING FULL CLASSIFICATION & ENSEMBLE PIPELINE")
    print("=" * 70)

    # 1. Run Base Classifiers
    base_results, base_models, X_train, X_test, y_train, y_test, feature_cols = (
        train_and_evaluate_classifiers(dataset_name)
    )

    # 2. Build & Train Ensemble Models
    ensembles = build_ensemble_models()
    ensemble_results = []
    ensemble_preds = {}

    for name, ens in ensembles.items():
        print(f"\n  Training Ensemble: {name}...")
        ens.fit(X_train, y_train)

        y_train_pred = ens.predict(X_train)
        y_test_pred = ens.predict(X_test)

        train_metrics = evaluate_classification(y_train, y_train_pred, labels=STRENGTH_CLASSES)
        test_metrics = evaluate_classification(y_test, y_test_pred, labels=STRENGTH_CLASSES)

        ensemble_preds[name] = y_test_pred

        print(f"    Train Accuracy: {train_metrics['Accuracy']:.4f} | Macro F1: {train_metrics['F1_macro']:.4f}")
        print(f"    Test  Accuracy: {test_metrics['Accuracy']:.4f} | Macro F1: {test_metrics['F1_macro']:.4f} | Precision: {test_metrics['Precision_macro']:.4f} | Recall: {test_metrics['Recall_macro']:.4f}")

        # Save metadata and model artifact
        metadata = {
            "model_name": name,
            "dataset": dataset_name,
            "target": "strength_category",
            "group_column": "name",
            "feature_columns": feature_cols,
            "classes": STRENGTH_CLASSES,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
        safe_name = f"{dataset_name}_ensemble_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        save_model_artifact(ens, metadata, safe_name, subfolder="classification")

        ensemble_results.append({
            "Dataset": dataset_name,
            "Model_Type": "Ensemble",
            "Algorithm": name,
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision_macro": test_metrics["Precision_macro"],
            "Test_Recall_macro": test_metrics["Recall_macro"],
            "Test_F1_macro": test_metrics["F1_macro"],
            "Train_Accuracy": train_metrics["Accuracy"],
            "Train_F1_macro": train_metrics["F1_macro"]
        })

    # Combine Base and Ensemble results
    all_results = base_results + ensemble_results
    comparison_df = pd.DataFrame(all_results)

    # Save to CSV
    csv_path = os.path.join(PROCESSED_DIR, "classification_model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved classification & ensemble comparison to: {csv_path}")

    # Plot Comparison Charts
    plot_classification_comparison(comparison_df, dataset_name)
    plot_ensemble_confusion_matrices(y_test, ensemble_preds, dataset_name)

    return comparison_df


def plot_classification_comparison(df, dataset_name):
    """Plot comparative Accuracy and Macro F1 bar chart for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Color code: Blue for base, Orange for ensemble
    colors = ["#2b5c8f" if t == "Base Classifier" else "#d95f02" for t in df["Model_Type"]]

    # Accuracy Chart
    ax1.barh(df["Algorithm"], df["Test_Accuracy"], color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Test Accuracy Comparison", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Accuracy", fontsize=10)
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    for i, v in enumerate(df["Test_Accuracy"]):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")

    # Macro F1 Chart
    ax2.barh(df["Algorithm"], df["Test_F1_macro"], color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title("Test Macro F1-Score Comparison", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Macro F1-Score", fontsize=10)
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    for i, v in enumerate(df["Test_F1_macro"]):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")

    # Legend for Base vs Ensemble
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", edgecolor="black", label="Base Model"),
        Patch(facecolor="#d95f02", edgecolor="black", label="Ensemble Model")
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

    plt.suptitle("Classification & Ensemble Performance Comparison (Steel DB)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(ASSETS_DIR, f"classification_model_comparison_{dataset_name}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved classification comparison chart to: {chart_path}")


def plot_ensemble_confusion_matrices(y_true, preds_dict, dataset_name):
    """Plot confusion matrices for ensemble classifiers."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=STRENGTH_CLASSES)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            ax=ax
        )
        ax.set_title(f"{name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Category", fontsize=9)
        ax.set_ylabel("Actual Category", fontsize=9)

    plt.suptitle(f"Confusion Matrices — Ensemble Models ({dataset_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(ASSETS_DIR, f"confusion_matrices_ensemble_{dataset_name}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ensemble confusion matrices to: {chart_path}")


if __name__ == "__main__":
    df_comp = run_classification_and_ensembles("steel_db")
    print("\nFinal Model Comparison:")
    print(df_comp.to_string())
