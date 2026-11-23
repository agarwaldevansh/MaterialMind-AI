"""
regression_models.py — Mechanical Property Prediction for MaterialMind AI

Implements syllabus-aligned regression approaches to predict continuous mechanical properties:
1. Simple Linear Regression (demonstrating single-feature baseline: CE / temperature)
2. Multiple Linear Regression (baseline multiple linear model across all features)
3. Polynomial Regression (degree 2 polynomial features to capture nonlinear interactions)
4. Ridge Regression (L2 regularization to stabilize collinear alloying coefficients)
5. Lasso Regression (L1 regularization providing automatic sparsity and feature selection)
6. Elastic Net Regression (combined L1 + L2 regularization for balanced stability)
7. Decision Tree Regressor (rule-based non-linear partitioning, Unit 5 Further Reading)
8. KNN Regressor (instance-based local metallurgical neighborhood averaging, Unit 4 Self Study)

Evaluation Metrics: MAE, MSE, RMSE, R²
Strict Data Splitting: GroupShuffleSplit on material grade to prevent data leakage.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_utils import (
    load_dataset,
    split_data_grouped,
    evaluate_regression,
    save_model_artifact,
    ASSETS_DIR,
    PROCESSED_DIR
)


def get_regression_models(dataset_name="steel_db"):
    """
    Returns a dictionary of named regression pipelines/estimators aligned with AIML syllabus.
    """
    single_feature = "carbon_equivalent" if dataset_name == "steel_db" else "temperature_c"
    simple_selector = ColumnTransformer(
        [("single_feat", StandardScaler(), [single_feature])],
        remainder="drop"
    )

    models = {
        "Simple Linear Regression": Pipeline([
            ("selector", simple_selector),
            ("regressor", LinearRegression())
        ]),
        "Multiple Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Polynomial Regression (Deg 2)": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("regressor", Ridge(alpha=50.0, random_state=42))
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=42))
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Lasso(alpha=0.5, max_iter=5000, random_state=42))
        ]),
        "Elastic Net Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, random_state=42))
        ]),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "KNN Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor(n_neighbors=7, weights="distance"))
        ])
    }
    return models


def train_and_evaluate_regressors(dataset_name="steel_db", target_col="tensile_strength_mpa"):
    """
    Train all 8 regression models on a specific dataset and target using GroupShuffleSplit.

    Parameters
    ----------
    dataset_name : str
        'steel_db' or 'matnavi'
    target_col : str
        e.g. 'tensile_strength_mpa', 'yield_strength_mpa', 'ductility_pct'

    Returns
    -------
    list of dict
        Evaluation summary for each model.
    """
    print(f"\n=======================================================")
    print(f"REGRESSION: Dataset '{dataset_name}' -> Target '{target_col}'")
    print(f"=======================================================")

    df = load_dataset(dataset_name)

    # Filter out outlier in MatNavi
    if dataset_name == "matnavi" and "is_outlier" in df.columns:
        initial_len = len(df)
        df = df[df["is_outlier"] == False].copy()
        print(f"  Filtered {initial_len - len(df)} flagged outlier record(s) from MatNavi.")

    # Select input features (strictly excluding all targets and display metrics)
    if dataset_name == "steel_db":
        group_col = "name"
        feature_cols = [
            "c", "mn", "si", "cr", "ni", "mo", "cu", "v", "al", "ti",
            "s", "p", "fe", "co", "nb", "b", "n", "zr",
            "total_alloying_pct", "carbon_equivalent", "mn_c_ratio", "cr_mo_total", "ni_cr_total"
        ]
    else:  # matnavi
        group_col = "alloy_code"
        feature_cols = [
            "c", "si", "mn", "p", "s", "ni", "cr", "mo", "cu", "v", "al", "n",
            "nb_ta", "ceq", "temperature_c", "total_alloying_pct",
            "carbon_equivalent_calc", "mn_c_ratio", "cr_mo_total", "is_elevated_temp"
        ]

    # Verify target column is NOT in features (strict anti-leakage guarantee)
    assert target_col not in feature_cols, f"LEAKAGE DETECTED: {target_col} in features!"

    # Group-based split
    X_train, X_test, y_train, y_test, g_train, g_test = split_data_grouped(
        df, feature_cols, target_col, group_col, test_size=0.2, random_state=42
    )

    print(f"  Train shape: {X_train.shape} across {g_train.nunique()} material groups")
    print(f"  Test shape:  {X_test.shape} across {g_test.nunique()} material groups")

    models = get_regression_models(dataset_name)
    results = []

    # Store predictions for plotting
    predictions_dict = {}

    for model_name, model in models.items():
        print(f"\n  Training: {model_name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = evaluate_regression(y_train, y_train_pred)
        test_metrics = evaluate_regression(y_test, y_test_pred)

        predictions_dict[model_name] = y_test_pred

        print(f"    Train R2: {train_metrics['R2']:.4f} | RMSE: {train_metrics['RMSE']:.2f}")
        print(f"    Test  R2: {test_metrics['R2']:.4f} | RMSE: {test_metrics['RMSE']:.2f} | MAE: {test_metrics['MAE']:.2f}")

        # Model metadata
        metadata = {
            "model_name": model_name,
            "dataset": dataset_name,
            "target": target_col,
            "group_column": group_col,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_columns": feature_cols,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }

        # Save model artifact
        safe_name = f"{dataset_name}_{target_col}_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        save_model_artifact(model, metadata, safe_name, subfolder="regression")

        results.append({
            "Dataset": dataset_name,
            "Target": target_col,
            "Algorithm": model_name,
            "Test_MAE": test_metrics["MAE"],
            "Test_MSE": test_metrics["MSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"],
            "Train_R2": train_metrics["R2"],
            "Train_RMSE": train_metrics["RMSE"]
        })

    # Generate Evaluation Visualizations
    plot_regression_actual_vs_predicted(
        y_test, predictions_dict, dataset_name, target_col
    )
    plot_regression_residuals(
        y_test, predictions_dict, dataset_name, target_col
    )

    return results


def plot_regression_actual_vs_predicted(y_true, preds_dict, dataset_name, target_col):
    """Generate 2x4 subplot of Actual vs. Predicted values for the 8 algorithms."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    min_val = min(y_true.min(), min(p.min() for p in preds_dict.values()))
    max_val = max(y_true.max(), max(p.max() for p in preds_dict.values()))

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#17becf"
    ]

    for ax, (model_name, y_pred), color in zip(axes, preds_dict.items(), colors):
        r2 = r2_score(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.45, color=color, edgecolors="black", linewidth=0.3, s=26)
        # 45-degree ideal prediction line
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit (y = x)")

        ax.set_title(f"{model_name}\n(Test R² = {r2:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Actual {target_col}", fontsize=8)
        ax.set_ylabel(f"Predicted {target_col}", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left")

    plt.suptitle(f"Regression: Actual vs. Predicted — {dataset_name} ({target_col})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_file = os.path.join(ASSETS_DIR, f"reg_actual_vs_pred_{dataset_name}_{target_col}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved actual vs predicted plot to: {out_file}")


def plot_regression_residuals(y_true, preds_dict, dataset_name, target_col):
    """Generate residual distribution plots (Residual = Actual - Predicted) for 8 algorithms."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#17becf"
    ]

    for ax, (model_name, y_pred), color in zip(axes, preds_dict.items(), colors):
        residuals = y_true - y_pred
        ax.hist(residuals, bins=25, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)

        mean_res = residuals.mean()
        std_res = residuals.std()
        ax.set_title(f"{model_name}\nMean: {mean_res:.1f}, Std: {std_res:.1f}", fontsize=9, fontweight="bold")
        ax.set_xlabel(f"Residual Error ({target_col})", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Residual Error Distributions — {dataset_name} ({target_col})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_file = os.path.join(ASSETS_DIR, f"reg_residuals_{dataset_name}_{target_col}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved residuals plot to: {out_file}")


def run_all_regressions():
    """
    Executes regression training across all 5 mechanical targets:
    Steel DB: Tensile Strength, Yield Strength, Ductility.
    MatNavi: Tensile Strength, Proof Stress.
    """
    print("\n" + "=" * 70)
    print("STARTING COMPLETE SYLLABUS-ALIGNED REGRESSION PIPELINE")
    print("=" * 70)

    all_results = []

    # 1. Steel DB Targets
    steel_targets = ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"]
    for target in steel_targets:
        res = train_and_evaluate_regressors(dataset_name="steel_db", target_col=target)
        all_results.extend(res)

    # 2. MatNavi Targets
    matnavi_targets = ["tensile_strength_mpa", "proof_stress_mpa"]
    for target in matnavi_targets:
        res = train_and_evaluate_regressors(dataset_name="matnavi", target_col=target)
        all_results.extend(res)

    # Compile Summary Comparison Table
    summary_df = pd.DataFrame(all_results)
    out_csv = os.path.join(PROCESSED_DIR, "regression_model_comparison.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved master regression comparison table to: {out_csv}")
    print("\nRegression Pipeline Completed Successfully!")
    return summary_df


if __name__ == "__main__":
    df_res = run_all_regressions()
    print("\nSummary Results Preview:")
    print(df_res[["Dataset", "Target", "Algorithm", "Test_RMSE", "Test_R2"]].to_string())
