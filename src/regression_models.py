"""
regression_models.py — Mechanical Property Prediction for MaterialMind AI

Implements four regression approaches to predict continuous mechanical properties:
1. Linear Regression (baseline multiple linear model)
2. Polynomial Regression (degree 2 polynomial features to capture nonlinear interactions)
3. Decision Tree Regressor (rule-based non-linear partitioning)
4. Gradient Descent Optimization (SGDRegressor minimizing MSE via iterative gradient descent)

Evaluation Metrics: MAE, MSE, RMSE, R²
Strict Data Splitting: GroupShuffleSplit on material grade to prevent data leakage.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
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


def get_regression_models():
    """
    Returns a dictionary of named regression pipelines/estimators.

    Notice:
    - Linear Regression: Scaler + LinearRegression
    - Polynomial Regression: Degree 2 polynomial features + Ridge (L2 penalty) to avoid multicollinear explosion
    - Decision Tree: max_depth=6 for good balance of accuracy and explainability
    - Gradient Descent: Scaler + SGDRegressor with squared error loss (optimization demonstration)
    """
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Polynomial Regression (Deg 2)": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("regressor", Ridge(alpha=50.0, random_state=42))
        ]),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "Gradient Descent (SGD)": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=0.01,
                max_iter=3000,
                tol=1e-4,
                random_state=42
            ))
        ])
    }
    return models


def train_and_evaluate_regressors(dataset_name="steel_db", target_col="tensile_strength_mpa"):
    """
    Train all 4 regression models on a specific dataset and target using GroupShuffleSplit.

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

    models = get_regression_models()
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

        print(f"    Train R2: {train_metrics['R2']:.4f} | RMSE: {train_metrics['RMSE']:.2f} MPa")
        print(f"    Test  R2: {test_metrics['R2']:.4f} | RMSE: {test_metrics['RMSE']:.2f} MPa | MAE: {test_metrics['MAE']:.2f}")

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
    """Generate 2x2 subplot of Actual vs. Predicted values for the 4 algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    min_val = min(y_true.min(), min(p.min() for p in preds_dict.values()))
    max_val = max(y_true.max(), max(p.max() for p in preds_dict.values()))

    colors = ["#2b5c8f", "#d95f02", "#2ca02c", "#7570b3"]

    for ax, (model_name, y_pred), color in zip(axes, preds_dict.items(), colors):
        r2 = r2_score(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.45, color=color, edgecolors="black", linewidth=0.3, s=28)
        # 45-degree ideal prediction line
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit (y = x)")

        ax.set_title(f"{model_name}\n(Test R² = {r2:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Actual {target_col}", fontsize=9)
        ax.set_ylabel(f"Predicted {target_col}", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(f"Regression: Actual vs. Predicted — {dataset_name} ({target_col})", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_file = os.path.join(ASSETS_DIR, f"reg_actual_vs_pred_{dataset_name}_{target_col}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved actual vs predicted plot to: {out_file}")


def plot_regression_residuals(y_true, preds_dict, dataset_name, target_col):
    """Generate residual distribution plots (Residual = Actual - Predicted)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    colors = ["#2b5c8f", "#d95f02", "#2ca02c", "#7570b3"]

    for ax, (model_name, y_pred), color in zip(axes, preds_dict.items(), colors):
        residuals = y_true - y_pred
        ax.hist(residuals, bins=25, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)

        mean_res = residuals.mean()
        std_res = residuals.std()
        ax.set_title(f"{model_name} Residuals\nMean: {mean_res:.1f}, Std: {std_res:.1f}", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Residual Error ({target_col})", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Residual Error Distributions — {dataset_name} ({target_col})", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_file = os.path.join(ASSETS_DIR, f"reg_residuals_{dataset_name}_{target_col}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved residuals plot to: {out_file}")


def run_all_regressions():
    """
    Run the full regression suite across both datasets:
    - Dataset 1 (Steel DB): tensile_strength_mpa, yield_strength_mpa, ductility_pct
    - Dataset 2 (MatNavi): tensile_strength_mpa, proof_stress_mpa
    """
    print("\n" + "=" * 70)
    print("STARTING COMPLETE REGRESSION EXPERIMENTS")
    print("=" * 70)

    all_results = []

    # Dataset 1 targets
    for target in ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"]:
        res = train_and_evaluate_regressors("steel_db", target)
        all_results.extend(res)

    # Dataset 2 targets
    for target in ["tensile_strength_mpa", "proof_stress_mpa"]:
        res = train_and_evaluate_regressors("matnavi", target)
        all_results.extend(res)

    # Save summary table
    summary_df = pd.DataFrame(all_results)
    out_csv = os.path.join(PROCESSED_DIR, "regression_model_comparison.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved regression model comparison table to: {out_csv}")
    print("\nSummary of Regression Performance:")
    print(summary_df.to_string())

    # Plot model comparison bar chart
    plot_regression_model_comparison_chart(summary_df)

    return summary_df


def plot_regression_model_comparison_chart(summary_df):
    """Plot comparative R2 and RMSE across all algorithms and targets."""
    # Filter for steel_db tensile strength for a clean focused comparison chart
    sub = summary_df[(summary_df["Dataset"] == "steel_db") & (summary_df["Target"] == "tensile_strength_mpa")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Test R2 comparison
    ax1.bar(sub["Algorithm"], sub["Test_R2"], color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax1.set_title("Test R² Score (Higher is Better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("R² Score", fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.tick_params(axis="x", rotation=30)
    for i, v in enumerate(sub["Test_R2"]):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # Test RMSE comparison
    ax2.bar(sub["Algorithm"], sub["Test_RMSE"], color="#d95f02", edgecolor="black", alpha=0.85)
    ax2.set_title("Test RMSE in MPa (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("RMSE (MPa)", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.tick_params(axis="x", rotation=30)
    for i, v in enumerate(sub["Test_RMSE"]):
        ax2.text(i, v + 5, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Regression Algorithm Comparison — Tensile Strength (Steel DB)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(ASSETS_DIR, "regression_model_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved regression comparison chart to: {chart_path}")


if __name__ == "__main__":
    run_all_regressions()
