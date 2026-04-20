"""
Unsupervised Learning on Water Quality Data
=============================================
This script:
1. Loads and merges raw water quality data from multiple platform sensors (L0-L6).
2. Preprocesses the data (type coercion, missing-value handling, scaling).
3. Runs unsupervised learning methods:
   - PCA (dimensionality reduction & visualisation)
   - K-Means clustering (with elbow & silhouette analysis)
   - DBSCAN (density-based clustering)
   - Agglomerative (hierarchical) clustering
4. Saves all outputs (CSVs, plots, summary) to unsupervised/unsupervised_output/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "unsupervised_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features common to every platform file (the core water-quality measurements)
CORE_FEATURES = [
    "Temperature (C)",
    "Specific Conductance (uS/cm)",
    "Salinity (PPT)",
    "Pressure (psia)",
    "Depth (m)",
    "ODO (%Sat)",
    "ODO (mg/L)",
    "Turbidity (FNU)",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_merge_data() -> pd.DataFrame:
    """Load all platform CSVs, tag each with its platform id, and concatenate."""
    frames = []
    for filename in sorted(os.listdir(RAW_DATA_DIR)):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(RAW_DATA_DIR, filename)
        # extract platform label, e.g. "L0"
        platform = filename.replace("raw-data-platform", "").replace("_parameters.csv", "")
        df = pd.read_csv(filepath, low_memory=False)
        df["platform"] = platform
        frames.append(df)
        print(f"  Loaded {filename}: {df.shape[0]:,} rows, {df.shape[1]} cols")
    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Merged dataset: {merged.shape[0]:,} rows, {merged.shape[1]} cols\n")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Clean, select features, and scale the data.

    Returns
    -------
    df_clean : DataFrame with selected features (unscaled, after cleaning)
    df_scaled : DataFrame with the same columns, StandardScaler-transformed
    scaled_array : numpy array of scaled values (for sklearn estimators)
    """
    # Keep only core features that exist in the merged frame
    available = [c for c in CORE_FEATURES if c in df.columns]
    df_feat = df[available].copy()

    # Coerce everything to numeric (handles mixed-type columns in L4/L5)
    for col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    # Drop rows that are entirely NaN, then fill remaining NaNs with column median
    df_feat.dropna(how="all", inplace=True)
    for col in df_feat.columns:
        median_val = df_feat[col].median()
        df_feat[col].fillna(median_val, inplace=True)

    # Remove extreme outliers (beyond 1st/99th percentile) to stabilise clustering
    for col in df_feat.columns:
        lo, hi = df_feat[col].quantile(0.01), df_feat[col].quantile(0.99)
        df_feat = df_feat[(df_feat[col] >= lo) & (df_feat[col] <= hi)]

    df_feat.reset_index(drop=True, inplace=True)

    # Standardise
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_feat)
    df_scaled = pd.DataFrame(scaled_array, columns=df_feat.columns)

    print(f"  After preprocessing: {df_feat.shape[0]:,} rows, {df_feat.shape[1]} features")
    return df_feat, df_scaled, scaled_array


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PCA - DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pca(scaled_array: np.ndarray, df_clean: pd.DataFrame, n_components: int = 2):
    """Fit PCA, save explained-variance bar chart and 2-D scatter."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_array)

    # -- Explained variance bar chart --
    pca_full = PCA().fit(scaled_array)
    fig, ax = plt.subplots(figsize=(8, 4))
    var_ratio = pca_full.explained_variance_ratio_
    ax.bar(range(1, len(var_ratio) + 1), var_ratio, color="steelblue", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA - Explained Variance per Component")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pca_explained_variance.png"), dpi=150)
    plt.close(fig)

    # -- Cumulative explained variance --
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(var_ratio) + 1), np.cumsum(var_ratio), marker="o", color="darkorange")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA - Cumulative Explained Variance")
    ax.axhline(0.95, ls="--", color="grey", label="95 % threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pca_cumulative_variance.png"), dpi=150)
    plt.close(fig)

    # -- 2-D PCA scatter --
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(components[:, 0], components[:, 1],
                         c=df_clean["Temperature (C)"], cmap="coolwarm",
                         s=2, alpha=0.4)
    fig.colorbar(scatter, ax=ax, label="Temperature (C)")
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)")
    ax.set_title("PCA - 2-D Projection (coloured by Temperature)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pca_2d_scatter.png"), dpi=150)
    plt.close(fig)

    # -- Loadings heatmap --
    loadings = pd.DataFrame(
        pca_full.components_[:n_components],
        columns=df_clean.columns,
        index=[f"PC{i+1}" for i in range(n_components)],
    )
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("PCA Loadings (first 2 components)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pca_loadings_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"  PCA: first 2 components explain {var_ratio[:2].sum():.1%} of variance")
    return components, pca_full


# ═══════════════════════════════════════════════════════════════════════════════
# 4. K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def run_kmeans(scaled_array: np.ndarray, pca_components: np.ndarray,
               k_range: range = range(2, 11)):
    """Elbow plot, silhouette analysis, and best-k clustering."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled_array)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(scaled_array, labels, sample_size=10_000,
                                           random_state=42))
        print(f"    K={k}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"  Best K by silhouette: {best_k}")

    # -- Elbow plot --
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(list(k_range), inertias, "o-", color="steelblue", label="Inertia")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(list(k_range), sil_scores, "s--", color="darkorange", label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="darkorange")
    ax1.set_title("K-Means: Elbow & Silhouette Analysis")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "kmeans_elbow_silhouette.png"), dpi=150)
    plt.close(fig)

    # -- Final clustering with best K --
    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels_best = km_best.fit_predict(scaled_array)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1],
                         c=labels_best, cmap="tab10", s=2, alpha=0.4)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"K-Means Clusters (K={best_k}) in PCA Space")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "kmeans_pca_clusters.png"), dpi=150)
    plt.close(fig)

    return labels_best, best_k


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DBSCAN CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def run_dbscan(scaled_array: np.ndarray, pca_components: np.ndarray,
               eps: float = 0.8, min_samples: int = 15):
    """Run DBSCAN on the scaled features and visualise in PCA space."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled_array)
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"  DBSCAN: {n_clusters} clusters, {n_noise:,} noise points "
          f"({n_noise / len(labels):.1%})")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1],
                         c=labels, cmap="tab20", s=2, alpha=0.4)
    fig.colorbar(scatter, ax=ax, label="Cluster (-1 = noise)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"DBSCAN Clusters (eps={eps}, min_samples={min_samples})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dbscan_pca_clusters.png"), dpi=150)
    plt.close(fig)

    return labels, n_clusters


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AGGLOMERATIVE (HIERARCHICAL) CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def run_agglomerative(scaled_array: np.ndarray, pca_components: np.ndarray,
                      n_clusters: int = 4, sample_size: int = 5000):
    """Hierarchical clustering + dendrogram on a subsample."""
    # Subsample for the dendrogram (full data is too large)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(scaled_array), size=min(sample_size, len(scaled_array)),
                     replace=False)
    sample = scaled_array[idx]

    # Dendrogram
    Z = linkage(sample, method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, leaf_rotation=90,
               leaf_font_size=8, color_threshold=0)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward, subsample)")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "hierarchical_dendrogram.png"), dpi=150)
    plt.close(fig)

    # Full agglomerative clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = agg.fit_predict(scaled_array)
    sil = silhouette_score(scaled_array, labels, sample_size=10_000, random_state=42)
    print(f"  Agglomerative: {n_clusters} clusters, silhouette={sil:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1],
                         c=labels, cmap="Set2", s=2, alpha=0.4)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Agglomerative Clusters (n={n_clusters}) in PCA Space")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "agglomerative_pca_clusters.png"), dpi=150)
    plt.close(fig)

    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlation(df_clean: pd.DataFrame):
    """Save a feature-correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                square=True, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE DISTRIBUTION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_distributions(df_clean: pd.DataFrame):
    """Histogram + KDE for each feature."""
    n = len(df_clean.columns)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(df_clean.columns):
        axes[i].hist(df_clean[col], bins=60, density=True, alpha=0.6, color="steelblue")
        df_clean[col].plot.kde(ax=axes[i], color="darkorange", lw=1.5)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions (after cleaning)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CLUSTER PROFILE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def save_cluster_profiles(df_clean: pd.DataFrame, labels: np.ndarray,
                          method_name: str):
    """Compute per-cluster mean of original features and save as CSV."""
    df_tmp = df_clean.copy()
    df_tmp["cluster"] = labels
    profile = df_tmp.groupby("cluster").mean()
    path = os.path.join(OUTPUT_DIR, f"{method_name}_cluster_profiles.csv")
    profile.to_csv(path)
    print(f"  Saved cluster profiles -> {path}")
    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  UNSUPERVISED LEARNING - WATER QUALITY DATA")
    print("=" * 60)

    # 1. Load
    print("\n[1/7] Loading raw data ...")
    df_raw = load_and_merge_data()

    # 2. Preprocess
    print("[2/7] Preprocessing ...")
    df_clean, df_scaled, X = preprocess(df_raw)

    # Save cleaned & scaled data for reproducibility
    df_clean.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
    df_scaled.to_csv(os.path.join(OUTPUT_DIR, "scaled_data.csv"), index=False)
    print(f"  Saved cleaned & scaled CSVs -> {OUTPUT_DIR}")

    # 3. EDA plots
    print("\n[3/7] Generating EDA plots ...")
    plot_correlation(df_clean)
    plot_distributions(df_clean)

    # 4. PCA
    print("\n[4/7] Running PCA ...")
    pca_components, pca_model = run_pca(X, df_clean)

    # 5. K-Means
    print("\n[5/7] Running K-Means ...")
    km_labels, best_k = run_kmeans(X, pca_components)
    save_cluster_profiles(df_clean, km_labels, "kmeans")

    # 6. DBSCAN
    print("\n[6/7] Running DBSCAN ...")
    db_labels, db_n = run_dbscan(X, pca_components)
    if db_n > 0:
        save_cluster_profiles(df_clean, db_labels, "dbscan")

    # 7. Agglomerative
    print("\n[7/7] Running Agglomerative Clustering ...")
    agg_labels = run_agglomerative(X, pca_components, n_clusters=best_k)
    save_cluster_profiles(df_clean, agg_labels, "agglomerative")

    # Final labelled dataset
    df_final = df_clean.copy()
    df_final["kmeans_cluster"] = km_labels
    df_final["dbscan_cluster"] = db_labels
    df_final["agglomerative_cluster"] = agg_labels
    df_final.to_csv(os.path.join(OUTPUT_DIR, "clustered_data.csv"), index=False)

    # Summary text
    summary_lines = [
        "Unsupervised Learning Summary",
        "=" * 40,
        f"Total rows after cleaning: {len(df_clean):,}",
        f"Features used: {list(df_clean.columns)}",
        f"",
        f"PCA: first 2 components explain "
        f"{pca_model.explained_variance_ratio_[:2].sum():.1%} of variance",
        f"",
        f"K-Means best K (by silhouette): {best_k}",
        f"DBSCAN clusters: {db_n}  (eps=0.8, min_samples=15)",
        f"Agglomerative clusters: {best_k}  (Ward linkage)",
        f"",
        "Output files:",
        "  cleaned_data.csv            - preprocessed feature matrix",
        "  scaled_data.csv             - StandardScaler-transformed features",
        "  clustered_data.csv          - features + cluster labels from all methods",
        "  kmeans_cluster_profiles.csv - per-cluster feature means (K-Means)",
        "  dbscan_cluster_profiles.csv - per-cluster feature means (DBSCAN)",
        "  agglomerative_cluster_profiles.csv - per-cluster means (Agglomerative)",
        "  pca_explained_variance.png  - variance per component",
        "  pca_cumulative_variance.png - cumulative variance curve",
        "  pca_2d_scatter.png          - 2-D PCA scatter coloured by temperature",
        "  pca_loadings_heatmap.png    - component loadings",
        "  kmeans_elbow_silhouette.png - elbow & silhouette plots",
        "  kmeans_pca_clusters.png     - K-Means clusters in PCA space",
        "  dbscan_pca_clusters.png     - DBSCAN clusters in PCA space",
        "  hierarchical_dendrogram.png - Ward dendrogram",
        "  agglomerative_pca_clusters.png - agglomerative clusters in PCA space",
        "  correlation_heatmap.png     - feature correlation",
        "  feature_distributions.png   - per-feature histograms",
    ]
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n  Summary saved -> {summary_path}")
    print("\n✓ Done - all outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
