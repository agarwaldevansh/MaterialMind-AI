"""
ensemble_models.py — Ensemble Learning for Material Strength Classification
MaterialMind AI — B.Tech Major AIML Project

Implements syllabus-aligned ensemble architectures (Unit 5) on the material strength classification task:
1. Bagging Classifier:
   - Bootstrap aggregating ensemble of decision trees to reduce prediction variance.
2. AdaBoost Classifier:
   - Sequential boosting using shallow decision stumps (max_depth=2).
   - Iteratively shifts sample weights toward previously misclassified material compositions.
3. Stacking Classifier:
   - Blends heterogeneous base models using out-of-fold cross-validation predictions (cv=5).
   - Meta-learner (LogisticRegression) learns how to optimally combine each model's strengths.

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
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier

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
    # Base estimators for Stacking
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
        # 1. Bagging Classifier
        "Bagging Classifier": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
            n_estimators=50,
            random_state=42
        ),

        # 2. AdaBoost Classifier with shallow decision tree
        "AdaBoost Classifier": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=60,
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
    1. Train 5 base classifiers (Logistic, Naive Bayes, KNN, Decision Tree, SVM).
    2. Train 3 ensemble models (Bagging, AdaBoost, Stacking).
    3. Compare all 8 models side-by-side on identical test data.
    4. Save models, metrics, and visualization plots.
    """
    print("\n" + "=" * 70)
    print("STARTING FULL CLASSIFICATION & ENSEMBLE PIPELINE (SYLLABUS ALIGNED)")
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

        # Probabilities for ROC-AUC
        y_test_prob = None
        if hasattr(ens, "predict_proba"):
            try:
                y_test_prob = ens.predict_proba(X_test)
            except Exception:
                pass

        prob_classes = list(ens.classes_) if hasattr(ens, "classes_") else STRENGTH_CLASSES
        train_metrics = evaluate_classification(y_train, y_train_pred, labels=STRENGTH_CLASSES, prob_classes=prob_classes)
        test_metrics = evaluate_classification(y_test, y_test_pred, y_prob=y_test_prob, labels=STRENGTH_CLASSES, prob_classes=prob_classes)

        ensemble_preds[name] = y_test_pred

        roc_display = f"{test_metrics['ROC_AUC']:.4f}" if test_metrics.get("ROC_AUC") is not None else "N/A"
        print(f"    Train Accuracy: {train_metrics['Accuracy']:.4f} | Macro F1: {train_metrics['F1_macro']:.4f}")
        print(f"    Test  Accuracy: {test_metrics['Accuracy']:.4f} | Macro F1: {test_metrics['F1_macro']:.4f} | ROC-AUC: {roc_display}")

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
            "Test_ROC_AUC": test_metrics.get("ROC_AUC"),
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

    # Plot Confusion Matrices for Ensembles
    plot_ensemble_confusion_matrices(y_test, ensemble_preds, dataset_name)

    # Plot Side-by-Side Accuracy and F1 Comparison Chart
    plot_overall_classification_comparison(comparison_df, dataset_name)

    return comparison_df


def plot_ensemble_confusion_matrices(y_true, ensemble_preds_dict, dataset_name):
    """Plot confusion matrices for ensemble architectures."""
    n_models = len(ensemble_preds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4.8 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]

    from sklearn.metrics import confusion_matrix
    short_labels = ["Low", "Med", "High"]

    for ax, (model_name, y_pred) in zip(axes, ensemble_preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred, labels=STRENGTH_CLASSES)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            ax=ax,
            xticklabels=short_labels,
            yticklabels=short_labels,
            annot_kws={"size": 10, "weight": "bold"}
        )
        ax.set_title(f"{model_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

    plt.suptitle(f"Confusion Matrices: Ensembles — {dataset_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_file = os.path.join(ASSETS_DIR, f"confusion_matrices_ensemble_{dataset_name}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ensemble confusion matrices to: {out_file}")


def plot_overall_classification_comparison(df, dataset_name):
    """Generate bar chart comparing Accuracy and Macro F1 score across all 8 models."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    width = 0.35

    rects1 = ax.bar(x - width/2, df["Test_Accuracy"], width, label="Test Accuracy", color="#2b5c8f", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, df["Test_F1_macro"], width, label="Test Macro F1", color="#d95f02", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=10)
    ax.set_title(f"Classification Benchmarks: Base Models vs Ensembles — {dataset_name}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Algorithm"], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")

    # Add labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_file = os.path.join(ASSETS_DIR, f"classification_model_comparison_{dataset_name}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved overall classification comparison plot to: {out_file}")


if __name__ == "__main__":
    df_cls = run_classification_and_ensembles()
    print("\nMaster Classification & Ensemble Summary:")
    print(df_cls[["Algorithm", "Model_Type", "Test_Accuracy", "Test_F1_macro", "Test_ROC_AUC"]].to_string())
