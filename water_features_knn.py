# ============================================================
# Feature Correlation Matrices -- L0 to L7
# ============================================================
# For each water-quality feature (temperature, conductance,
# salinity, dissolved oxygen, turbidity), we compute the
# Pearson correlation between every pair of stations.
#
# High correlation (close to 1) means two stations rise and
# fall together for that feature -- they respond similarly to
# environmental conditions.  Low or negative correlation means
# they behave independently or oppositely.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Step 1: Load the raw wide-format data
# ------------------------------------------------------------------
raw = pd.read_csv("data/merged_keep.csv")

# ------------------------------------------------------------------
# Step 2: Define the features and stations
# ------------------------------------------------------------------
FEATURES = {
    "temperature_c":              "Temperature (deg C)",
    "specific_conductance_us_cm": "Specific Conductance (uS/cm)",
    "salinity_ppt":               "Salinity (ppt)",
    "odo_sat":                    "Dissolved O₂ Saturation (%)",
    "turbidity_fnu":              "Turbidity (FNU)",
}

STATIONS = [f"l{i}" for i in range(8)]

STATION_PRETTY = {
    "l0": "L0\nFIU Bay",
    "l1": "L1\nBisc. Canal",
    "l2": "L2\nCanal-Bay",
    "l3": "L3\nLittle Riv A",
    "l4": "L4\nLittle Riv B",
    "l5": "L5\nNBV North",
    "l6": "L6\nNBV South",
    "l7": "L7\nMiami Riv",
}

# ------------------------------------------------------------------
# Step 3: Build one correlation heatmap per feature
# ------------------------------------------------------------------
# For each feature we pull every station's column into one dataframe,
# then compute .corr() (Pearson) across stations.

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()  # easier to index

for idx, (feat_key, feat_label) in enumerate(FEATURES.items()):
    # Gather the station columns that exist for this feature
    col_map = {}
    for s in STATIONS:
        col_name = f"{feat_key}_{s}"
        if col_name in raw.columns:
            col_map[col_name] = STATION_PRETTY[s]

    if len(col_map) < 2:
        axes[idx].set_visible(False)
        continue

    subset = raw[list(col_map.keys())].rename(columns=col_map)

    # Compute pairwise Pearson correlation (drops NaN pairs automatically)
    corr = subset.corr()

    # Plot as a heatmap
    ax = axes[idx]
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,          # show the numbers in each cell
        fmt=".2f",           # two decimal places
        cmap="RdBu_r",       # red = negative, blue = positive
        vmin=-1, vmax=1,     # correlation always in [-1, 1]
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(feat_label, fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)

# Hide any leftover subplot (we have 5 features in a 2x3 grid)
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Station-to-Station Correlation per Feature (L0 - L7)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("water_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved water_feature_correlations.png")

# ------------------------------------------------------------------
# Step 4: Also print the numeric correlation tables
# ------------------------------------------------------------------
for feat_key, feat_label in FEATURES.items():
    col_map = {}
    for s in STATIONS:
        col_name = f"{feat_key}_{s}"
        if col_name in raw.columns:
            col_map[col_name] = s.upper()

    if len(col_map) < 2:
        continue

    corr = raw[list(col_map.keys())].rename(columns=col_map).corr()
    print(f"\n=== {feat_label} ===")
    print(corr.to_string())
