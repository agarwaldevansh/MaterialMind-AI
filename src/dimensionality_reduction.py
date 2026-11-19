"""
dimensionality_reduction.py — Principal Component Analysis (PCA) for MaterialMind AI

Implements unsupervised dimensionality reduction using PCA:
1. Standardizes numerical chemical composition features.
2. Computes orthogonal principal component directions maximizing variance.
3. Calculates individual explained variance ratios and cumulative explained variance.
4. Generates:
   - 2D PCA projection scatter plot.
   - Scree plot (Individual and Cumulative Explained Variance).
5. Saves fitted PCA transformer and projection coordinates for Step 4 visualization.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_utils import (
    load_dataset,
    save_model_artifact,
    create_strength_categories,
    ASSETS_DIR,
    PROCESSED_DIR
)


def run_pca_analysis(dataset_name="steel_db", n_components=10):
    """
    Execute PCA on numerical composition features of the specified dataset.

    Parameters
    ----------
    dataset_name : str
        'steel_db' or 'matnavi'
    n_components : int
        Number of principal components to compute.
    """
    print("\n" + "=" * 70)
    print(f"DIMENSIONALITY REDUCTION: PCA on '{dataset_name}'")
    print("=" * 70)

    df = load_dataset(dataset_name)

    if dataset_name == "steel_db":
        feature_cols = [
            "c", "mn", "si", "cr", "ni", "mo", "cu", "v", "al", "ti",
            "s", "p", "fe", "co", "nb", "b", "n", "zr",
            "total_alloying_pct", "carbon_equivalent", "mn_c_ratio", "cr_mo_total", "ni_cr_total"
        ]
    else:
        df = df[df["is_outlier"] == False].copy() if "is_outlier" in df.columns else df
        feature_cols = [
            "c", "si", "mn", "p", "s", "ni", "cr", "mo", "cu", "v", "al", "n",
            "nb_ta", "ceq", "temperature_c", "total_alloying_pct",
            "carbon_equivalent_calc", "mn_c_ratio", "cr_mo_total", "is_elevated_temp"
        ]

    X = df[feature_cols].copy()
    print(f"Input features for PCA ({len(feature_cols)}): {feature_cols}")

    # Step 1: Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Step 3: Variance explained
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    print("\nPrincipal Component Variance Summary:")
    for i in range(n_components):
        print(f"  PC{i+1:>2}: Explained Variance = {exp_var[i]*100:>5.2f}% | Cumulative = {cum_var[i]*100:>5.2f}%")

    # Step 4: Plot 2D PCA Projection
    plot_2d_pca_projection(X_pca, df, dataset_name)

    # Step 5: Plot Explained Variance (Scree Plot)
    plot_pca_explained_variance(exp_var, cum_var, dataset_name)

    # Step 6: Save PCA Model Artifact
    metadata = {
        "model_type": "PCA",
        "dataset": dataset_name,
        "n_components": n_components,
        "feature_columns": feature_cols,
        "explained_variance_ratio": [round(float(v), 4) for v in exp_var],
        "cumulative_variance": [round(float(v), 4) for v in cum_var]
    }
    save_model_artifact(
        {"scaler": scaler, "pca": pca},
        metadata,
        f"{dataset_name}_pca.joblib",
        subfolder="pca"
    )

    # Save PCA coordinates to CSV for interactive dashboard in Step 4
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "PC3": X_pca[:, 2] if n_components >= 3 else 0
    }, index=df.index)

    if "name" in df.columns:
        pca_df["name"] = df["name"]
    if "tensile_strength_mpa" in df.columns:
        pca_df["tensile_strength_mpa"] = df["tensile_strength_mpa"]
        pca_df["strength_category"] = create_strength_categories(df, "tensile_strength_mpa")

    out_pca_csv = os.path.join(PROCESSED_DIR, f"{dataset_name}_pca_coordinates.csv")
    pca_df.to_csv(out_pca_csv, index=False)
    print(f"Saved PCA projected coordinates to: {out_pca_csv}")

    return pca, scaler, X_pca, exp_var, cum_var


def plot_2d_pca_projection(X_pca, df, dataset_name):
    """Plot 2D PCA projection colored by strength category."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Color code by strength category if tensile strength is available
    if "tensile_strength_mpa" in df.columns:
        categories = create_strength_categories(df, "tensile_strength_mpa")
        color_map = {
            "Low Strength": "#2b5c8f",
            "Medium Strength": "#e6550d",
            "High Strength": "#2ca02c"
        }
        for cat_name, color in color_map.items():
            mask = categories == cat_name
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=color, label=cat_name, alpha=0.55, edgecolors="none", s=30
            )
        ax.legend(title="Strength Tier", fontsize=9, loc="upper right")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c="#2b5c8f", alpha=0.5, s=25)

    ax.set_title(f"Principal Component Analysis (PCA) — 2D Projection ({dataset_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=10)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(ASSETS_DIR, f"pca_2d_scatter_{dataset_name}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D PCA scatter plot to: {chart_path}")


def plot_pca_explained_variance(exp_var, cum_var, dataset_name):
    """Plot individual and cumulative explained variance ratios (Scree plot)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    comps = range(1, len(exp_var) + 1)

    # Bar chart of individual variance
    bars = ax1.bar(comps, exp_var * 100, color="#2b5c8f", alpha=0.75, label="Individual Explained Variance (%)")
    ax1.set_xlabel("Principal Component Number", fontsize=10)
    ax1.set_ylabel("Individual Explained Variance (%)", color="#2b5c8f", fontsize=10)
    ax1.set_xticks(comps)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Line chart of cumulative variance on second y-axis
    ax2 = ax1.twinx()
    line = ax2.plot(comps, cum_var * 100, color="#d95f02", marker="o", linewidth=2, label="Cumulative Explained Variance (%)")
    ax2.set_ylabel("Cumulative Variance (%)", color="#d95f02", fontsize=10)
    ax2.set_ylim(0, 105)

    # 80% and 90% threshold lines
    ax2.axhline(80, color="gray", linestyle=":", alpha=0.7)
    ax2.axhline(90, color="gray", linestyle="--", alpha=0.7)

    plt.title(f"PCA Scree Plot — Explained Variance Ratios ({dataset_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()

    chart_path = os.path.join(ASSETS_DIR, f"pca_explained_variance_{dataset_name}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA scree plot to: {chart_path}")


if __name__ == "__main__":
    run_pca_analysis("steel_db", n_components=10)
