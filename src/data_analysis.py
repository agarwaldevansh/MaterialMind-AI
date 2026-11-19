"""
data_analysis.py - Exploratory Data Analysis (EDA) for MaterialMind AI
======================================================================
Generates clear, informative engineering visualizations for both cleaned steel datasets:
1. Target property distributions (Tensile, Yield/Proof Stress, Ductility/Elongation)
2. Correlation heatmaps (Composition vs Mechanical Properties)
3. Composition vs Strength scatter plots
4. Cluster distribution analysis
5. Temperature effect on mechanical properties (MatNavi dataset)
6. Missing value audit charts

All figures are saved to the assets/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")


def ensure_assets_dir():
    """Ensure the assets output directory exists."""
    os.makedirs(ASSETS_DIR, exist_ok=True)


def plot_ds1_target_distributions(df, save_path):
    """
    Plot distributions of mechanical properties for Dataset 1:
    - Tensile strength (MPa)
    - Yield strength (MPa)
    - Ductility (%)
    """
    print("Generating: DS1 Target Distributions...")
    targets = ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"]
    titles = ["Tensile Strength (MPa)", "Yield Strength (MPa)", "Ductility (%)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, col, title, color in zip(axes, targets, titles, colors):
        if col in df.columns:
            data = df[col].dropna()
            mean_val = data.mean()
            median_val = data.median()

            ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.75)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
            ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(title, fontsize=10)
            ax.set_ylabel("Frequency (Count)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)

    plt.suptitle("Dataset 1 (Steel DB) — Mechanical Property Distributions", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {save_path}")


def plot_ds1_correlation_heatmap(df, save_path):
    """
    Generate correlation heatmap of numerical composition elements and mechanical properties.
    """
    print("Generating: DS1 Correlation Heatmap...")
    numeric_cols = [
        "c", "si", "mn", "p", "s", "cr", "mo", "ni", "al", "cu", "v",
        "tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"
    ]
    cols_to_use = [c for c in numeric_cols if c in df.columns]
    corr = df[cols_to_use].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(cols_to_use)))
    ax.set_yticks(range(len(cols_to_use)))
    ax.set_xticklabels(cols_to_use, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(cols_to_use, fontsize=9)

    # Annotate values inside cells
    for i in range(len(cols_to_use)):
        for j in range(len(cols_to_use)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)

    plt.title("Dataset 1 (Steel DB) — Pearson Correlation Matrix", fontsize=13, fontweight="bold", pad=25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {save_path}")


def plot_ds1_composition_vs_tensile(df, save_path):
    """
    Scatter plots showing effect of key alloying elements on tensile strength.
    """
    print("Generating: DS1 Composition vs Tensile Strength...")
    elements = ["c", "mn", "cr", "ni", "mo", "si"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, elem in zip(axes, elements):
        if elem in df.columns:
            ax.scatter(df[elem], df["tensile_strength_mpa"], alpha=0.35, color="#1f77b4", edgecolors="none", s=25)
            ax.set_title(f"Tensile Strength vs {elem.upper()} wt%", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{elem.upper()} Content (wt%)", fontsize=9)
            ax.set_ylabel("Tensile Strength (MPa)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Dataset 1 — Key Alloying Elements vs Tensile Strength", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {save_path}")


def plot_ds1_cluster_distribution(df, save_path):
    """
    Horizontal bar chart showing sample frequency per pre-existing cluster label.
    """
    print("Generating: DS1 Cluster Distribution...")
    if "cluster_label" in df.columns:
        counts = df["cluster_label"].value_counts().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot(kind="barh", ax=ax, color="#4575b4", edgecolor="black", alpha=0.8)

        ax.set_title("Dataset 1 — Sample Count per Pre-Existing Cluster Category", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Materials", fontsize=10)
        ax.set_ylabel("Cluster Category", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for i, v in enumerate(counts):
            ax.text(v + 10, i, str(v), va="center", fontsize=8.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved to: {save_path}")


def plot_ds2_target_distributions(df, save_path):
    """
    Plot distributions of mechanical properties for Dataset 2 (excluding flagged outliers).
    """
    print("Generating: DS2 Target Distributions...")
    clean_df = df[df["is_outlier"] == False] if "is_outlier" in df.columns else df

    targets = ["tensile_strength_mpa", "proof_stress_mpa", "elongation_pct", "reduction_in_area_pct"]
    titles = ["Tensile Strength (MPa)", "0.2% Proof Stress (MPa)", "Elongation (%)", "Reduction in Area (%)"]
    colors = ["#3182bd", "#e6550d", "#31a354", "#756bb1"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, col, title, color in zip(axes, targets, titles, colors):
        if col in clean_df.columns:
            data = clean_df[col].dropna()
            mean_val = data.mean()
            median_val = data.median()

            ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.75)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
            ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel(title, fontsize=9)
            ax.set_ylabel("Frequency (Count)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8.5)

    plt.suptitle("Dataset 2 (MatNavi) — Mechanical Property Distributions (Cleaned)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {save_path}")


def plot_ds2_temperature_effect(df, save_path):
    """
    Plot the effect of test temperature on Tensile Strength and Proof Stress.
    """
    print("Generating: DS2 Temperature Effect...")
    clean_df = df[df["is_outlier"] == False] if "is_outlier" in df.columns else df

    grouped = clean_df.groupby("temperature_c").agg(
        ts_mean=("tensile_strength_mpa", "mean"),
        ts_std=("tensile_strength_mpa", "std"),
        ps_mean=("proof_stress_mpa", "mean"),
        ps_std=("proof_stress_mpa", "std")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(grouped["temperature_c"], grouped["ts_mean"], yerr=grouped["ts_std"],
                 fmt="-o", capsize=5, label="Tensile Strength (Mean ± 1 SD)", color="#d95f02", linewidth=2)
    ax.errorbar(grouped["temperature_c"], grouped["ps_mean"], yerr=grouped["ps_std"],
                 fmt="-s", capsize=5, label="0.2% Proof Stress (Mean ± 1 SD)", color="#2b5c8f", linewidth=2)

    ax.set_title("Dataset 2 — Elevated Temperature Degradation of Steel Strength", fontsize=12, fontweight="bold")
    ax.set_xlabel("Test Temperature (°C)", fontsize=10)
    ax.set_ylabel("Strength (MPa)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {save_path}")


def plot_ds2_correlation_heatmap(df, save_path):
    """
    Correlation heatmap for MatNavi dataset.
    """
    print("Generating: DS2 Correlation Heatmap...")
    clean_df = df[df["is_outlier"] == False] if "is_outlier" in df.columns else df
    numeric_cols = [
        "c", "si", "mn", "cr", "mo", "ni", "cu", "v",
        "temperature_c", "tensile_strength_mpa", "proof_stress_mpa", "elongation_pct", "reduction_in_area_pct"
    ]
    cols_to_use = [c for c in numeric_cols if c in clean_df.columns]
    corr = clean_df[cols_to_use].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(cols_to_use)))
    ax.set_yticks(range(len(cols_to_use)))
    ax.set_xticklabels(cols_to_use, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(cols_to_use, fontsize=9)

    for i in range(len(cols_to_use)):
        for j in range(len(cols_to_use)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)

    plt.title("Dataset 2 (MatNavi) — Correlation Matrix", fontsize=13, fontweight="bold", pad=25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {save_path}")


def plot_missing_value_summary(df1, df2, save_path):
    """
    Plot missing value comparison between the two cleaned datasets.
    """
    print("Generating: Missing Values Summary...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    m1 = df1.isnull().sum()
    m2 = df2.isnull().sum()

    m1_missing = m1[m1 > 0]
    m2_missing = m2[m2 > 0]

    if len(m1_missing) == 0:
        ax1.text(0.5, 0.5, "No Missing Values\n(100% Complete)", ha="center", va="center", fontsize=12, color="green", fontweight="bold")
        ax1.set_axis_off()
    else:
        m1_missing.plot(kind="bar", ax=ax1, color="salmon")
        ax1.set_ylabel("Count")

    ax1.set_title("Dataset 1 (Steel DB Cleaned)", fontsize=11, fontweight="bold")

    if len(m2_missing) == 0:
        ax2.text(0.5, 0.5, "No Missing Values\n(100% Complete)", ha="center", va="center", fontsize=12, color="green", fontweight="bold")
        ax2.set_axis_off()
    else:
        m2_missing.plot(kind="bar", ax=ax2, color="salmon")
        ax2.set_ylabel("Count")

    ax2.set_title("Dataset 2 (MatNavi Cleaned)", fontsize=11, fontweight="bold")

    plt.suptitle("Cleaned Datasets Data Quality Audit: Missing Values", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {save_path}")


def run_eda():
    """Main execution function for EDA generation."""
    print("========================================")
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("========================================")

    ensure_assets_dir()

    ds1_path = os.path.join(PROCESSED_DIR, "steel_db_cleaned.csv")
    ds2_path = os.path.join(PROCESSED_DIR, "matnavi_cleaned.csv")

    if not os.path.exists(ds1_path) or not os.path.exists(ds2_path):
        print("Cleaned datasets not found. Please run data_preprocessing.py first.")
        return

    df1 = pd.read_csv(ds1_path)
    df2 = pd.read_csv(ds2_path)

    # Dataset 1 plots
    plot_ds1_target_distributions(df1, os.path.join(ASSETS_DIR, "ds1_target_distributions.png"))
    plot_ds1_correlation_heatmap(df1, os.path.join(ASSETS_DIR, "ds1_correlation_heatmap.png"))
    plot_ds1_composition_vs_tensile(df1, os.path.join(ASSETS_DIR, "ds1_composition_vs_tensile.png"))
    plot_ds1_cluster_distribution(df1, os.path.join(ASSETS_DIR, "ds1_cluster_distribution.png"))

    # Dataset 2 plots
    plot_ds2_target_distributions(df2, os.path.join(ASSETS_DIR, "ds2_target_distributions.png"))
    plot_ds2_temperature_effect(df2, os.path.join(ASSETS_DIR, "ds2_temperature_effect.png"))
    plot_ds2_correlation_heatmap(df2, os.path.join(ASSETS_DIR, "ds2_correlation_heatmap.png"))

    # Missing value overview
    plot_missing_value_summary(df1, df2, os.path.join(ASSETS_DIR, "missing_value_summary.png"))

    print("\nEDA Generation complete. All plots saved to assets/")


if __name__ == "__main__":
    run_eda()
