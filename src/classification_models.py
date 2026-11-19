"""
classification_models.py — Material Strength Classification for MaterialMind AI

Implements multi-class classification to categorize steels into three strength tiers:
- Low Strength: < 600 MPa (general structural / mild steels)
- Medium Strength: 600 - 950 MPa (machinery / quenched & tempered alloy steels)
- High Strength: >= 950 MPa (high strength structural / spring / maraging steels)

STRICT ANTI-LEAKAGE RULE:
The target categories are derived from Tensile Strength.
Therefore, Tensile Strength, Yield Strength, Ductility, and Yield-to-Tensile Ratio
are STRICTLY EXCLUDED from input features.

Algorithms implemented:
1. Decision Tree Classifier (interpretable decision boundaries)
2. K-Nearest Neighbors Classifier (KNN with standard feature scaling)
3. Support Vector Machine Classifier (SVM with RBF kernel & standard scaling)
4. Gaussian Naive Bayes (probabilistic Bayes theorem baseline)

Evaluated using: Accuracy, Macro Precision, Macro Recall, Macro F1-score, Confusion Matrix.
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

STRENGTH_CLASSES = ["Low Strength", "Medium Strength", "High Strength"]


def get_base_classifiers():
    """Returns a dictionary of named base classification estimators."""
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "K-Nearest Neighbors (KNN)": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ]),
        "Support Vector Machine (SVM)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=2.0, probability=True, random_state=42))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("gnb", GaussianNB())
        ])
    }
    return classifiers


def train_and_evaluate_classifiers(dataset_name="steel_db"):
    """
    Train and evaluate all base classification models on Steel DB.
    """
    print("\n" + "=" * 70)
    print(f"CLASSIFICATION: Strength Tiers on Dataset '{dataset_name}'")
    print("=" * 70)

    df = load_dataset(dataset_name)

    # Create target strength categories based on tensile strength
    df["strength_category"] = create_strength_categories(df, target_col="tensile_strength_mpa")

    # Inspect class distribution
    class_counts = df["strength_category"].value_counts()[STRENGTH_CLASSES]
    print("\nClass Distribution across Dataset:")
    for cls_name, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  • {cls_name:<16}: {count:>4} samples ({pct:.1f}%)")

    # Plot Class Distribution
    plot_class_distribution(class_counts, dataset_name)

    # Input features strictly exclude ALL mechanical properties and display ratios
    group_col = "name"
    feature_cols = [
        "c", "mn", "si", "cr", "ni", "mo", "cu", "v", "al", "ti",
        "s", "p", "fe", "co", "nb", "b", "n", "zr",
        "total_alloying_pct", "carbon_equivalent", "mn_c_ratio", "cr_mo_total", "ni_cr_total"
    ]

    # Leakage check assertion
    forbidden_cols = ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct", "yield_tensile_ratio_DISPLAY_ONLY"]
    for col in forbidden_cols:
        assert col not in feature_cols, f"CRITICAL LEAKAGE: {col} found in classification input features!"

    # Group-based train/test split
    X_train, X_test, y_train, y_test, g_train, g_test = split_data_grouped(
        df, feature_cols, "strength_category", group_col, test_size=0.2, random_state=42
    )

    print(f"\nTraining Samples: {len(X_train)} ({g_train.nunique()} steel grades)")
    print(f"Testing  Samples: {len(X_test)} ({g_test.nunique()} steel grades)")

    classifiers = get_base_classifiers()
    results = []
    trained_models = {}
    test_predictions = {}

    for name, clf in classifiers.items():
        print(f"\n  Training: {name}...")
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_metrics = evaluate_classification(y_train, y_train_pred, labels=STRENGTH_CLASSES)
        test_metrics = evaluate_classification(y_test, y_test_pred, labels=STRENGTH_CLASSES)

        trained_models[name] = clf
        test_predictions[name] = y_test_pred

        print(f"    Train Accuracy: {train_metrics['Accuracy']:.4f} | Macro F1: {train_metrics['F1_macro']:.4f}")
        print(f"    Test  Accuracy: {test_metrics['Accuracy']:.4f} | Macro F1: {test_metrics['F1_macro']:.4f} | Precision: {test_metrics['Precision_macro']:.4f} | Recall: {test_metrics['Recall_macro']:.4f}")

        # Save metadata and model artifact
        metadata = {
            "model_name": name,
            "dataset": dataset_name,
            "target": "strength_category",
            "group_column": group_col,
            "feature_columns": feature_cols,
            "classes": STRENGTH_CLASSES,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
        safe_name = f"{dataset_name}_strength_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        save_model_artifact(clf, metadata, safe_name, subfolder="classification")

        results.append({
            "Dataset": dataset_name,
            "Model_Type": "Base Classifier",
            "Algorithm": name,
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision_macro": test_metrics["Precision_macro"],
            "Test_Recall_macro": test_metrics["Recall_macro"],
            "Test_F1_macro": test_metrics["F1_macro"],
            "Train_Accuracy": train_metrics["Accuracy"],
            "Train_F1_macro": train_metrics["F1_macro"]
        })

    # Plot Confusion Matrices
    plot_confusion_matrices(y_test, test_predictions, dataset_name)

    return results, trained_models, X_train, X_test, y_train, y_test, feature_cols


def plot_class_distribution(counts_series, dataset_name):
    """Plot bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2b5c8f", "#d95f02", "#2ca02c"]
    counts_series.plot(kind="bar", ax=ax, color=colors, edgecolor="black", alpha=0.85)

    ax.set_title("Material Strength Classification — Class Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Strength Category", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", rotation=0)

    for i, v in enumerate(counts_series):
        pct = v / counts_series.sum() * 100
        ax.text(i, v + 25, f"{v}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(ASSETS_DIR, f"class_distribution_{dataset_name}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution chart to: {chart_path}")


def plot_confusion_matrices(y_true, test_preds_dict, dataset_name):
    """Plot 2x2 grid of confusion matrices for base classifiers."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (name, y_pred) in zip(axes, test_preds_dict.items()):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=STRENGTH_CLASSES)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            ax=ax
        )
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Category", fontsize=9)
        ax.set_ylabel("Actual Category", fontsize=9)

    plt.suptitle(f"Confusion Matrices — Base Classifiers ({dataset_name})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(ASSETS_DIR, f"confusion_matrices_base_{dataset_name}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to: {chart_path}")


if __name__ == "__main__":
    results, _, _, _, _, _, _ = train_and_evaluate_classifiers("steel_db")
    df_res = pd.DataFrame(results)
    print("\nClassification Results:")
    print(df_res.to_string())
