# ============================================================
# Cross-Correlation & PCA -- Biscayne Bay Subsystem Analysis
# ============================================================
# Two subsystems of interest:
#
#   NORTH of JFK Causeway:
#     Sources:  L0 (FIU Bay), L1 (Biscayne Canal)
#     Lagoon:   L2 (canal-bay junction), L5 (NBV north)
#
#   SOUTH of JFK Causeway:
#     Sources:  L3 (Little River A), L4 (Little River B)
#     Lagoon:   L6 (NBV south)
#
# Questions we're answering:
#   1. How strongly -- and with what time lag -- do the source
#      stations influence the lagoon stations?  (cross-correlation)
#   2. Do stations within a subsystem cluster together, and which
#      features drive the variation?  (PCA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
FEATURES = ["temperature_c", "specific_conductance_us_cm",
            "salinity_ppt", "odo_sat", "turbidity_fnu"]

FEATURE_LABELS = {
    "temperature_c":              "Temperature (deg C)",
    "specific_conductance_us_cm": "Conductance (uS/cm)",
    "salinity_ppt":               "Salinity (ppt)",
    "odo_sat":                    "DO Sat (%)",
    "turbidity_fnu":              "Turbidity (FNU)",
}

STATION_LABELS = {
    "l0": "L0 - FIU Bay",
    "l1": "L1 - Bisc. Canal",
    "l2": "L2 - Canal-Bay jct",
    "l3": "L3 - Little River A",
    "l4": "L4 - Little River B",
    "l5": "L5 - NBV North",
    "l6": "L6 - NBV South",
    "l7": "L7 - Miami River",
}

# The two subsystems
NORTH = {
    "name": "North of JFK Causeway",
    "sources": ["l0", "l1"],
    "lagoon":  ["l2", "l5"],
}
SOUTH = {
    "name": "South of JFK Causeway",
    "sources": ["l3", "l4"],
    "lagoon":  ["l6"],
}

# ------------------------------------------------------------------
# Step 1: Load data
# ------------------------------------------------------------------
raw = pd.read_csv("data/merged_keep.csv")

# The data is sampled every 5 minutes.  We use "datetime_5min" as
# the time index so we can reason about lags in real time.
raw["datetime_5min"] = pd.to_datetime(raw["datetime_5min"], format="mixed")
raw.set_index("datetime_5min", inplace=True)
raw.sort_index(inplace=True)
print(f"Data range: {raw.index.min()}  ->  {raw.index.max()}")
print(f"Rows: {len(raw)}\n")


# ==================================================================
# PART A -- CROSS-CORRELATION WITH TIME LAGS
# ==================================================================
# For every (source, lagoon) pair within a subsystem and for each
# feature, we compute the cross-correlation at lags from -6 h to +6 h.
#
# - The lag that yields the *peak* correlation tells you how long it
#   takes for a change at the source to show up at the lagoon.
# - A positive lag means the source LEADS the lagoon (expected).
# - A negative lag means the lagoon leads -- physically unlikely, but
#   can happen with tidal backflow.
#
# We use 5-min steps, so lag = 12 means 1 hour.
# ==================================================================

MAX_LAG_HOURS = 6
SAMPLES_PER_HOUR = 12   # 60 min / 5 min
MAX_LAG = MAX_LAG_HOURS * SAMPLES_PER_HOUR  # 72 steps = 6 hours

def compute_cross_corr(series_source, series_lagoon, max_lag):
    """
    Compute normalised cross-correlation between source and lagoon
    for lags from -max_lag to +max_lag.

    Positive lag -> source leads (source at time t correlates with
    lagoon at time t + lag).
    """
    # Align and drop NaN
    combined = pd.concat([series_source.rename("src"),
                          series_lagoon.rename("lag")], axis=1).dropna()
    if len(combined) < 100:
        return None, None

    src = combined["src"].values
    lag_series = combined["lag"].values

    # Normalise to zero-mean, unit-variance
    src = (src - src.mean()) / (src.std() + 1e-12)
    lag_series = (lag_series - lag_series.mean()) / (lag_series.std() + 1e-12)

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))

    for i, d in enumerate(lags):
        if d >= 0:
            corrs[i] = np.mean(src[:len(src) - d] * lag_series[d:])
        else:
            corrs[i] = np.mean(src[-d:] * lag_series[:len(lag_series) + d])

    return lags, corrs

print("=" * 60)
print("PART A: Cross-Correlation (source -> lagoon)")
print("=" * 60)

for subsystem in [NORTH, SOUTH]:
    print(f"\n{'─' * 50}")
    print(f"  {subsystem['name']}")
    print(f"  Sources: {', '.join(STATION_LABELS[s] for s in subsystem['sources'])}")
    print(f"  Lagoon:  {', '.join(STATION_LABELS[s] for s in subsystem['lagoon'])}")
    print(f"{'─' * 50}")

    pairs = [(src, lag)
             for src in subsystem["sources"]
             for lag in subsystem["lagoon"]]

    n_pairs = len(pairs)
    n_feats = len(FEATURES)

    fig, axes = plt.subplots(n_pairs, n_feats, figsize=(4 * n_feats, 3.5 * n_pairs),
                             squeeze=False, sharey="row")

    for row, (src_station, lag_station) in enumerate(pairs):
        for col, feat in enumerate(FEATURES):
            src_col = f"{feat}_{src_station}"
            lag_col = f"{feat}_{lag_station}"

            ax = axes[row][col]

            if src_col not in raw.columns or lag_col not in raw.columns:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(FEATURE_LABELS[feat], fontsize=9)
                continue

            lags, corrs = compute_cross_corr(raw[src_col], raw[lag_col], MAX_LAG)

            if lags is None:
                ax.text(0.5, 0.5, "insufficient\ndata", ha="center",
                        va="center", transform=ax.transAxes)
                ax.set_title(FEATURE_LABELS[feat], fontsize=9)
                continue

            lag_hours = lags / SAMPLES_PER_HOUR
            ax.plot(lag_hours, corrs, linewidth=1)
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

            # Mark peak
            peak_idx = np.argmax(corrs)
            peak_lag_h = lag_hours[peak_idx]
            peak_corr = corrs[peak_idx]
            ax.axvline(peak_lag_h, color="red", linewidth=0.8, linestyle="--",
                       alpha=0.7)
            ax.scatter([peak_lag_h], [peak_corr], color="red", s=30, zorder=5)
            ax.annotate(f"{peak_lag_h:+.1f}h\nr={peak_corr:.2f}",
                        xy=(peak_lag_h, peak_corr), fontsize=7,
                        textcoords="offset points", xytext=(8, -8))

            ax.set_title(FEATURE_LABELS[feat], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{STATION_LABELS[src_station]}\n-> {STATION_LABELS[lag_station]}",
                              fontsize=8)
            if row == n_pairs - 1:
                ax.set_xlabel("Lag (hours)", fontsize=8)

            print(f"  {STATION_LABELS[src_station]:20s} -> {STATION_LABELS[lag_station]:20s} | "
                  f"{FEATURE_LABELS[feat]:25s} | peak r = {peak_corr:.3f} at lag = {peak_lag_h:+.1f} h")

    fig.suptitle(f"Cross-Correlation -- {subsystem['name']}\n"
                 f"(positive lag = source leads lagoon)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f"xcorr_{subsystem['name'].lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  -> saved {fname}")


# ==================================================================
# PART A2 -- FOCUSED CROSS-CORRELATION: SALINITY & DO ONLY
# ==================================================================
FOCUS_FEATURES = ["salinity_ppt", "odo_sat"]
FOCUS_LABELS = {k: FEATURE_LABELS[k] for k in FOCUS_FEATURES}
FOCUS_COLORS = {"salinity_ppt": "#1f77b4", "odo_sat": "#d62728"}

print("\n" + "=" * 60)
print("PART A2: Focused Cross-Correlation -- Salinity & DO Only")
print("=" * 60)

for subsystem in [NORTH, SOUTH]:
    pairs = [(src, lag)
             for src in subsystem["sources"]
             for lag in subsystem["lagoon"]]
    n_pairs = len(pairs)

    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs),
                             squeeze=False)

    for row, (src_station, lag_station) in enumerate(pairs):
        for col, feat in enumerate(FOCUS_FEATURES):
            src_col = f"{feat}_{src_station}"
            lag_col = f"{feat}_{lag_station}"
            ax = axes[row][col]

            if src_col not in raw.columns or lag_col not in raw.columns:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(FOCUS_LABELS[feat], fontsize=11, fontweight="bold")
                continue

            lags_arr, corrs = compute_cross_corr(raw[src_col], raw[lag_col], MAX_LAG)

            if lags_arr is None:
                ax.text(0.5, 0.5, "insufficient\ndata", ha="center",
                        va="center", transform=ax.transAxes, fontsize=11)
                ax.set_title(FOCUS_LABELS[feat], fontsize=11, fontweight="bold")
                continue

            lag_hours = lags_arr / SAMPLES_PER_HOUR
            ax.fill_between(lag_hours, corrs, alpha=0.15, color=FOCUS_COLORS[feat])
            ax.plot(lag_hours, corrs, linewidth=1.5, color=FOCUS_COLORS[feat])
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.3)

            # Mark peak
            peak_idx = np.argmax(corrs)
            peak_lag_h = lag_hours[peak_idx]
            peak_corr = corrs[peak_idx]
            ax.axvline(peak_lag_h, color="red", linewidth=1.2, linestyle="--",
                       alpha=0.8)
            ax.scatter([peak_lag_h], [peak_corr], color="red", s=60, zorder=5,
                       edgecolors="black", linewidths=0.5)
            ax.annotate(f"peak: {peak_lag_h:+.1f} h\nr = {peak_corr:.3f}",
                        xy=(peak_lag_h, peak_corr), fontsize=9, fontweight="bold",
                        textcoords="offset points", xytext=(10, -12),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="gray", alpha=0.85))

            ax.set_title(FOCUS_LABELS[feat], fontsize=11, fontweight="bold")
            ax.set_xlabel("Lag (hours)", fontsize=9)
            ax.set_xlim(-MAX_LAG_HOURS, MAX_LAG_HOURS)
            if col == 0:
                ax.set_ylabel(
                    f"{STATION_LABELS[src_station]}\n-> {STATION_LABELS[lag_station]}",
                    fontsize=9, fontweight="bold")

            print(f"  {STATION_LABELS[src_station]:20s} -> {STATION_LABELS[lag_station]:20s} | "
                  f"{FOCUS_LABELS[feat]:20s} | peak r = {peak_corr:.3f} at lag = {peak_lag_h:+.1f} h")

    fig.suptitle(f"Salinity & DO Cross-Correlation -- {subsystem['name']}\n"
                 f"(positive lag = source leads lagoon)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"xcorr_focus_{subsystem['name'].lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  -> saved {fname}")


# ==================================================================
# PART B -- PCA (Principal Component Analysis)
# ==================================================================
# We stack all station readings into one long table (same approach
# as water_knn.py), run PCA, and project each sample into the first
# 2-3 principal components.  Stations that cluster together share
# similar water-quality fingerprints.
#
# We do this SEPARATELY for each subsystem so the components capture
# the variance *within* that subsystem, and then once for ALL stations
# to see the big picture.
# ==================================================================

STATIONS_ALL = [f"l{i}" for i in range(8)]

def stack_stations(raw_df, stations, features):
    """Reshape wide -> long: one row per station-reading."""
    frames = []
    for s in stations:
        cols = {f"{f}_{s}": f for f in features}
        existing = {k: v for k, v in cols.items() if k in raw_df.columns}
        if not existing:
            continue
        sub = raw_df[list(existing.keys())].rename(columns=existing).copy()
        sub["station"] = s
        frames.append(sub)
    stacked = pd.concat(frames, ignore_index=True).dropna()
    return stacked

print("\n" + "=" * 60)
print("PART B: PCA -- Station Clustering")
print("=" * 60)

# ---- B1: Global PCA (all stations) ----
df_all = stack_stations(raw, STATIONS_ALL, FEATURES)
X_all = df_all[FEATURES].values
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

pca_all = PCA(n_components=3)
pc_all = pca_all.fit_transform(X_all_scaled)

# How much variance each component explains
print(f"\nGlobal PCA -- explained variance:")
for i, ev in enumerate(pca_all.explained_variance_ratio_):
    print(f"  PC{i+1}: {ev:.1%}")
print(f"  Total (PC1-PC3): {sum(pca_all.explained_variance_ratio_):.1%}")

# What each component is made of (loadings)
print(f"\nGlobal PCA -- loadings (which features drive each component):")
loadings = pd.DataFrame(pca_all.components_.T,
                        index=[FEATURE_LABELS[f] for f in FEATURES],
                        columns=["PC1", "PC2", "PC3"])
print(loadings.to_string(float_format="{:.3f}".format))

# Plot PC1 vs PC2 coloured by station
fig, ax = plt.subplots(figsize=(10, 7))
unique_stations = sorted(df_all["station"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_stations)))

for i, s in enumerate(unique_stations):
    mask = df_all["station"].values == s
    ax.scatter(pc_all[mask, 0], pc_all[mask, 1],
               c=[colors[i]], label=STATION_LABELS[s],
               alpha=0.15, s=8, edgecolors="none")

ax.set_xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("PCA -- All Stations (L0-L7)", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, markerscale=3)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("pca_all_stations.png", dpi=150)
plt.show()
print("  -> saved pca_all_stations.png")

# ---- B2: Subsystem PCA ----
for subsystem in [NORTH, SOUTH]:
    stations = subsystem["sources"] + subsystem["lagoon"]
    df_sub = stack_stations(raw, stations, FEATURES)
    X_sub = df_sub[FEATURES].values
    scaler_sub = StandardScaler()
    X_sub_scaled = scaler_sub.fit_transform(X_sub)

    pca_sub = PCA(n_components=3)
    pc_sub = pca_sub.fit_transform(X_sub_scaled)

    print(f"\n{'─' * 50}")
    print(f"  PCA -- {subsystem['name']}")
    print(f"  Stations: {', '.join(STATION_LABELS[s] for s in stations)}")
    print(f"{'─' * 50}")
    for i, ev in enumerate(pca_sub.explained_variance_ratio_):
        print(f"  PC{i+1}: {ev:.1%}")

    loadings_sub = pd.DataFrame(pca_sub.components_.T,
                                index=[FEATURE_LABELS[f] for f in FEATURES],
                                columns=["PC1", "PC2", "PC3"])
    print(f"\n  Loadings:")
    print(loadings_sub.to_string(float_format="  {:.3f}".format))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    unique_sub = sorted(df_sub["station"].unique())
    colors_sub = plt.cm.Set1(np.linspace(0, 0.8, len(unique_sub)))

    # PC1 vs PC2
    for i, s in enumerate(unique_sub):
        mask = df_sub["station"].values == s
        # Mark sources with circles, lagoon with triangles
        marker = "o" if s in subsystem["sources"] else "^"
        axes[0].scatter(pc_sub[mask, 0], pc_sub[mask, 1],
                        c=[colors_sub[i]], label=STATION_LABELS[s],
                        alpha=0.2, s=15, marker=marker, edgecolors="none")
        axes[1].scatter(pc_sub[mask, 0], pc_sub[mask, 2],
                        c=[colors_sub[i]], label=STATION_LABELS[s],
                        alpha=0.2, s=15, marker=marker, edgecolors="none")

    axes[0].set_xlabel(f"PC1 ({pca_sub.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca_sub.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("PC1 vs PC2")
    axes[0].legend(fontsize=8, markerscale=3)
    axes[0].grid(True, alpha=0.2)

    axes[1].set_xlabel(f"PC1 ({pca_sub.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC3 ({pca_sub.explained_variance_ratio_[2]:.1%})")
    axes[1].set_title("PC1 vs PC3")
    axes[1].legend(fontsize=8, markerscale=3)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(f"PCA -- {subsystem['name']}\n(○ = source, △ = lagoon)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fname = f"pca_{subsystem['name'].lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  -> saved {fname}")


# ==================================================================
# PART C -- PCA Loadings Heatmap
# ==================================================================
# A compact view of which features matter most in each component,
# for each subsystem.

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
titles = ["All stations", NORTH["name"], SOUTH["name"]]

for idx, (title, stations) in enumerate([
    ("All Stations", STATIONS_ALL),
    (NORTH["name"], NORTH["sources"] + NORTH["lagoon"]),
    (SOUTH["name"], SOUTH["sources"] + SOUTH["lagoon"]),
]):
    df_tmp = stack_stations(raw, stations, FEATURES)
    X_tmp = StandardScaler().fit_transform(df_tmp[FEATURES].values)
    pca_tmp = PCA(n_components=3).fit(X_tmp)

    load_df = pd.DataFrame(
        pca_tmp.components_.T,
        index=[FEATURE_LABELS[f] for f in FEATURES],
        columns=[f"PC{i+1} ({v:.0%})" for i, v in enumerate(pca_tmp.explained_variance_ratio_)]
    )

    sns.heatmap(load_df, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=axes[idx],
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[idx].set_title(title, fontsize=10, fontweight="bold")
    axes[idx].tick_params(axis="y", labelsize=8)

plt.suptitle("PCA Loadings -- What Drives Each Component",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("pca_loadings_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> saved pca_loadings_heatmap.png")

print("\n✓ All done!")
