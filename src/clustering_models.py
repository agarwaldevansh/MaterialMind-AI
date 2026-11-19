"""
clustering_models.py — Unsupervised Material Grouping for MaterialMind AI

Implements unsupervised clustering to group steel alloys by chemical composition:
1. K-Means Clustering:
   - Evaluates Elbow Method (inertia vs k).
   - Computes Silhouette Scores for k in [2..10].
   - Assigns materials to distinct composition clusters.
2. Agglomerative Hierarchical Clustering:
   - Bottom-up hierarchical tree using Ward's minimum variance linkage.
   - Truncated dendrogram visualization.
   - Compares cluster agreement with K-Means using Adjusted Rand Index (ARI).

STRICT INTEGRITY ENFORCEMENT:
Pre-existing cluster numbers and text labels from the raw Excel file are STRICTLY EXCLUDED
from the input features. They are not treated as ground truth.
Clustering similarity indicates compositional proximity, NOT certified engineering interchangeability.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_utils import (
    load_dataset,
    save_model_artifact,
    ASSETS_DIR,
    PROCESSED_DIR
)


def get_clustering_features(df):
    """
    Select numerical composition features for unsupervised clustering.
    Excludes identifiers, targets, and pre-existing cluster metadata.
    """
    feature_cols = [
        "c", "mn", "si", "cr", "ni", "mo", "cu", "v", "al", "ti",
        "s", "p", "fe", "co", "nb", "b", "n", "zr",
        "total_alloying_pct", "carbon_equivalent", "mn_c_ratio", "cr_mo_total", "ni_cr_total"
    ]
    # Filter only available columns
    available = [c for c in feature_cols if c in df.columns]

    # Verify no cluster labels or target columns are included
    forbidden = ["cluster_number", "cluster_label", "tensile_strength_mpa", "yield_strength_mpa"]
    for f in forbidden:
        assert f not in available, f"ERROR: Forbidden column {f} found in clustering features!"

    return df[available].copy(), available


def run_elbow_and_silhouette_analysis(X_scaled, k_range=range(2, 11)):
    """
    Compute inertia (Elbow method) and silhouette scores for k in k_range.
    """
    print("Evaluating K-Means cluster counts (k=2 to 10)...")
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouettes.append(score)
        print(f"  k={k:>2} | Inertia: {kmeans.inertia_:>10.1f} | Silhouette Score: {score:.4f}")

    return list(k_range), inertias, silhouettes


def plot_elbow_and_silhouette(k_vals, inertias, silhouettes, save_path):
    """Plot side-by-side Elbow Curve and Silhouette Scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Elbow Curve
    ax1.plot(k_vals, inertias, "bo-", linewidth=2, markersize=6)
    ax1.set_title("K-Means Elbow Method (Inertia)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Silhouette Scores
    ax2.plot(k_vals, silhouettes, "rs-", linewidth=2, markersize=6)
    best_k = k_vals[np.argmax(silhouettes)]
    best_score = max(silhouettes)
    ax2.axvline(best_k, color="green", linestyle="--", label=f"Max Silhouette (k={best_k})")
    ax2.set_title("Silhouette Score vs. k", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax2.set_ylabel("Average Silhouette Score", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.suptitle("Clustering Evaluation Metrics — Determining Optimal k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Elbow & Silhouette plot to: {save_path}")


def plot_hierarchical_dendrogram(linkage_matrix, save_path):
    """
    Plot truncated hierarchical clustering dendrogram.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=25,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax
    )
    ax.set_title("Hierarchical Agglomerative Clustering — Truncated Dendrogram (Ward Linkage)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cluster Size / Leaf Node Index", fontsize=10)
    ax.set_ylabel("Ward Distance Metric", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Hierarchical Dendrogram to: {save_path}")


def train_clustering_models(dataset_name="steel_db", optimal_k=4):
    """
    Fit K-Means and Agglomerative Hierarchical Clustering models.
    """
    print("\n" + "=" * 70)
    print(f"CLUSTERING: Material Grouping on '{dataset_name}'")
    print("=" * 70)

    df = load_dataset(dataset_name)
    X, feature_cols = get_clustering_features(df)
    print(f"Input features for clustering ({len(feature_cols)}): {feature_cols}")

    # Standardize composition features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Elbow and Silhouette analysis
    k_vals, inertias, silhouettes = run_elbow_and_silhouette_analysis(X_scaled, range(2, 11))
    plot_elbow_and_silhouette(
        k_vals, inertias, silhouettes,
        os.path.join(ASSETS_DIR, f"clustering_elbow_silhouette_{dataset_name}.png")
    )

    # 2. Fit K-Means with chosen k
    print(f"\nFitting K-Means with optimal k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=20, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    print(f"  K-Means Silhouette Score: {kmeans_silhouette:.4f}")

    # 3. Fit Agglomerative Hierarchical Clustering
    print(f"\nFitting Agglomerative Hierarchical Clustering with k={optimal_k}...")
    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    agg_silhouette = silhouette_score(X_scaled, agg_labels)
    print(f"  Agglomerative Silhouette Score: {agg_silhouette:.4f}")

    # Compute cluster agreement between K-Means and Agglomerative
    ari = adjusted_rand_score(kmeans_labels, agg_labels)
    print(f"  Adjusted Rand Index (ARI between K-Means and Hierarchical): {ari:.4f}")

    # 4. Generate Dendrogram on a sample to maintain computation efficiency
    print("\nGenerating hierarchical linkage matrix...")
    # Sample 500 records if dataset is large for clear linkage plotting
    sample_size = min(600, len(X_scaled))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample_scaled = X_scaled[sample_indices]
    linkage_matrix = linkage(X_sample_scaled, method="ward")

    plot_hierarchical_dendrogram(
        linkage_matrix,
        os.path.join(ASSETS_DIR, f"hierarchical_dendrogram_{dataset_name}.png")
    )

    # 5. Save model artifacts
    save_model_artifact(
        {"scaler": scaler, "kmeans": kmeans},
        {
            "model_type": "KMeans",
            "optimal_k": optimal_k,
            "silhouette_score": round(float(kmeans_silhouette), 4),
            "inertia": round(float(kmeans.inertia_), 2),
            "feature_columns": feature_cols
        },
        f"{dataset_name}_kmeans_k{optimal_k}.joblib",
        subfolder="clustering"
    )

    save_model_artifact(
        {"scaler": scaler, "agg": agg},
        {
            "model_type": "AgglomerativeClustering",
            "optimal_k": optimal_k,
            "silhouette_score": round(float(agg_silhouette), 4),
            "adjusted_rand_index_vs_kmeans": round(float(ari), 4),
            "feature_columns": feature_cols
        },
        f"{dataset_name}_hierarchical_k{optimal_k}.joblib",
        subfolder="clustering"
    )

    # Save comparison table
    results = [
        {
            "Dataset": dataset_name,
            "Algorithm": f"K-Means (k={optimal_k})",
            "Optimal_k": optimal_k,
            "Silhouette_Score": round(float(kmeans_silhouette), 4),
            "Inertia": round(float(kmeans.inertia_), 2),
            "ARI_Agreement": 1.0,
            "Key_Characteristic": "Centroid-based partition minimizing within-cluster variance"
        },
        {
            "Dataset": dataset_name,
            "Algorithm": f"Agglomerative Hierarchical (k={optimal_k})",
            "Optimal_k": optimal_k,
            "Silhouette_Score": round(float(agg_silhouette), 4),
            "Inertia": np.nan,
            "ARI_Agreement": round(float(ari), 4),
            "Key_Characteristic": "Tree-based hierarchical merge using Ward linkage"
        }
    ]

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(PROCESSED_DIR, "clustering_model_comparison.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved clustering model comparison table to: {csv_path}")

    return res_df, kmeans_labels, agg_labels, X_scaled, scaler


if __name__ == "__main__":
    res_df, km_lbl, agg_lbl, X_sc, sc = train_clustering_models("steel_db", optimal_k=4)
    print("\nClustering Performance Summary:")
    print(res_df.to_string())
