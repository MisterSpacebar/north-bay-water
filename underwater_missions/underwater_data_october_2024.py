# ============================================================
# Exploratory Data Analysis -- Underwater Mission, October 25 2024
# ============================================================
# This script performs an EDA on the underwater mission dataset
# collected on October 25, 2024 in Biscayne Bay.
#
# Single continuous mission -- 678 data points.
#
# Sensor columns (24 total):
#   Lat, Lon, Date, Time, Chlorophyll RFU, Conductivity, Depth,
#   nLF Conductivity, ODO (% sat, % CB, mg/L), Pressure,
#   Salinity, SpCond, TAL PC RFU, TDS, Turbidity, TSS,
#   pH, pH mV, Temperature, Vertical Position,
#   Altitude, Barometer
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# ── paths ────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "October 25th 2024"
OUT_DIR  = Path(__file__).parent / "eda_plots_october_2024"
OUT_DIR.mkdir(exist_ok=True)

# ── style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 140

# Columns of interest (nice name -> csv col, unit)
FEATURE_MAP = {
    "Chlorophyll":   ("Chlorophyll RFU",   "RFU"),
    "Conductivity":  ("Cond µS/cm",        "uS/cm"),
    "Depth":         ("Depth m",           "m"),
    "DO Saturation": ("ODO % sat",         "%"),
    "DO Conc.":      ("ODO mg/L",          "mg/L"),
    "Salinity":      ("Sal psu",           "psu"),
    "SpConductance": ("SpCond µS/cm",      "uS/cm"),
    "Turbidity":     ("Turbidity FNU",     "FNU"),
    "pH":            ("pH",                ""),
    "Temperature":   ("Temp °C",           "deg C"),
    "TAL PC":        ("TAL PC RFU",        "RFU"),
    "Pressure":      ("Pressure psi a",    "psia"),
    "Altitude":      ("Altitude m",        "m"),
    "Barometer":     ("Barometer mmHg",    "mmHg"),
}

# Ecological thresholds
DO_STRESS_MGL  = 4.0   # mg/L -- fish stress begins
DO_HYPOXIC_MGL = 2.0   # mg/L -- hypoxic / lethal
TURB_HIGH_FNU  = 25.0  # elevated turbidity

ACCENT = "#1f77b4"

# ==============================================================
# 1. LOAD DATA
# ==============================================================
print("=" * 60)
print("Loading October 25, 2024 mission data ...")
print("=" * 60)

fp = DATA_DIR / "oct25-2024.csv"
df = pd.read_csv(fp)

# Build datetime column
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%m/%d/%Y %H:%M:%S",
)
df.sort_values("Datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

# Elapsed time in minutes from start (useful for colour mapping)
t0 = df["Datetime"].min()
df["Elapsed_min"] = (df["Datetime"] - t0).dt.total_seconds() / 60.0

print(f"Total rows: {len(df)}")
print(f"Time span: {df['Datetime'].min()} -> {df['Datetime'].max()}")
duration_min = df["Elapsed_min"].max()
print(f"Duration:  {duration_min:.1f} minutes ({duration_min/60:.1f} hours)")
print(f"Lat range: {df['Latitude'].min():.5f} - {df['Latitude'].max():.5f}")
print(f"Lon range: {df['Longitude'].min():.5f} - {df['Longitude'].max():.5f}")
print(f"Depth range: {df['Depth m'].min():.3f} - {df['Depth m'].max():.3f} m")
print()

# ==============================================================
# 2. DESCRIPTIVE STATISTICS
# ==============================================================
print("=" * 60)
print("Descriptive Statistics")
print("=" * 60)

numeric_cols = [v[0] for v in FEATURE_MAP.values()]
desc = df[numeric_cols].describe().T
desc["missing"] = df[numeric_cols].isna().sum()
desc["missing%"] = (desc["missing"] / len(df) * 100).round(1)
print(desc.to_string())
print()

# ==============================================================
# 3. CORRELATION HEAT MAP
# ==============================================================
print("Generating correlation heatmap ...")
fig, ax = plt.subplots(figsize=(12, 10))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    annot_kws={"size": 7},
)
ax.set_title("Feature Correlation -- October 25, 2024 Mission", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "01_correlation_heatmap.png")
plt.close(fig)

# ==============================================================
# 4. DISTRIBUTION PLOTS (histograms + KDE)
# ==============================================================
print("Generating distribution plots ...")
features_to_plot = [
    "Chlorophyll", "Depth", "DO Saturation", "DO Conc.",
    "Salinity", "Turbidity", "pH", "Temperature",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for i, feat in enumerate(features_to_plot):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    vals = df[col].dropna()
    ax.hist(vals, bins=30, alpha=0.6, color=ACCENT, edgecolor="white")
    # overlay KDE
    if len(vals) > 2:
        ax2 = ax.twinx()
        vals.plot.kde(ax=ax2, color="black", linewidth=1.2)
        ax2.set_ylabel("")
        ax2.set_yticks([])
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_xlabel(label_str, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

fig.suptitle("Feature Distributions -- October 25, 2024", fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(OUT_DIR / "02_distributions.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 5. BOX PLOTS (single mission -- show spread & outliers)
# ==============================================================
print("Generating box plots ...")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for i, feat in enumerate(features_to_plot):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    bp = ax.boxplot(df[col].dropna().values, patch_artist=True,
                    medianprops=dict(color="black"))
    bp["boxes"][0].set_facecolor(ACCENT)
    bp["boxes"][0].set_alpha(0.6)
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_ylabel(label_str, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.set_xticks([])

fig.suptitle("Feature Box Plots -- October 25, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "03_boxplots.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 6. TIME-SERIES PLOTS (key sensors over time)
# ==============================================================
print("Generating time-series plots ...")
ts_features = ["DO Conc.", "Temperature", "Salinity", "Turbidity", "pH", "Depth"]

fig, axes = plt.subplots(len(ts_features), 1, figsize=(14, 3.0 * len(ts_features)),
                         sharex=True)
for i, feat in enumerate(ts_features):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    ax.plot(df["Datetime"], df[col], marker=".", markersize=2,
            linewidth=0.8, color=ACCENT, alpha=0.8)
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_ylabel(label_str, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    # ecological thresholds
    if feat == "DO Conc.":
        ax.axhline(DO_STRESS_MGL, ls="--", color="orange", lw=1,
                    label=f"Stress ({DO_STRESS_MGL} mg/L)")
        ax.axhline(DO_HYPOXIC_MGL, ls="--", color="red", lw=1,
                    label=f"Hypoxic ({DO_HYPOXIC_MGL} mg/L)")
        ax.legend(fontsize=7)
    if feat == "Turbidity":
        ax.axhline(TURB_HIGH_FNU, ls="--", color="red", lw=1,
                    label=f"High ({TURB_HIGH_FNU} FNU)")
        ax.legend(fontsize=7)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[-1].set_xlabel("Time (HH:MM)", fontsize=9)
fig.suptitle("Sensor Time Series -- October 25, 2024", fontsize=13, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_DIR / "04_timeseries.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 7. GPS TRACK MAP (coloured by elapsed time)
# ==============================================================
print("Generating GPS track map ...")
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["Elapsed_min"],
                cmap="plasma", s=18, alpha=0.85, edgecolors="none")
ax.plot(df["Longitude"], df["Latitude"], linewidth=0.4, color="grey", alpha=0.4)
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Elapsed Time (min)", fontsize=10)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Mission GPS Track -- October 25, 2024", fontsize=12)
ax.ticklabel_format(useOffset=False, style="plain")
plt.tight_layout()
fig.savefig(OUT_DIR / "05_gps_track.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 8. GPS + DEPTH SCATTER (colour = depth)
# ==============================================================
print("Generating depth map ...")
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(
    df["Longitude"], df["Latitude"],
    c=df["Depth m"], cmap="viridis_r", s=20,
    edgecolors="none", alpha=0.85,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Depth (m)", fontsize=10)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Depth Along GPS Track -- October 25, 2024", fontsize=12)
ax.ticklabel_format(useOffset=False, style="plain")
plt.tight_layout()
fig.savefig(OUT_DIR / "06_depth_map.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 9. PAIR PLOT (key water quality features)
# ==============================================================
print("Generating pair plot ...")
pair_cols = ["ODO mg/L", "Sal psu", "Temp °C", "Turbidity FNU", "pH", "Depth m"]
pair_df = df[pair_cols].dropna()

g = sns.pairplot(pair_df, diag_kind="kde",
                 plot_kws={"s": 12, "alpha": 0.5, "color": ACCENT},
                 diag_kws={"linewidth": 1, "color": ACCENT})
g.figure.suptitle("Pair Plot -- Key Water Quality Features (October 25, 2024)",
                   y=1.01, fontsize=13)
g.savefig(OUT_DIR / "07_pairplot.png", bbox_inches="tight")
plt.close(g.figure)

# ==============================================================
# 10. DEPTH PROFILES (sensor vs depth)
# ==============================================================
print("Generating depth profiles ...")
depth_features = ["DO Conc.", "Salinity", "Temperature", "Turbidity"]

fig, axes = plt.subplots(1, len(depth_features), figsize=(16, 6))
for i, feat in enumerate(depth_features):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    sc = ax.scatter(df[col], df["Depth m"], c=df["Elapsed_min"],
                    cmap="plasma", s=14, alpha=0.7)
    ax.invert_yaxis()
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_xlabel(label_str, fontsize=9)
    if i == 0:
        ax.set_ylabel("Depth (m)", fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.02)
cbar.set_label("Elapsed Time (min)", fontsize=9)
fig.suptitle("Depth Profiles -- October 25, 2024", fontsize=13)
plt.tight_layout(rect=[0, 0, 0.92, 0.96])
fig.savefig(OUT_DIR / "08_depth_profiles.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 11. DISSOLVED OXYGEN DETAIL ANALYSIS
# ==============================================================
print("Generating DO analysis ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 11a. DO mg/L vs DO % sat -- coloured by time
ax = axes[0]
sc = ax.scatter(df["ODO % sat"], df["ODO mg/L"], c=df["Elapsed_min"],
                cmap="plasma", s=18, alpha=0.7)
ax.axhline(DO_STRESS_MGL, ls="--", color="orange", lw=1)
ax.axhline(DO_HYPOXIC_MGL, ls="--", color="red", lw=1)
ax.set_xlabel("DO Saturation (%)", fontsize=10)
ax.set_ylabel("DO Concentration (mg/L)", fontsize=10)
ax.set_title("DO Concentration vs Saturation", fontsize=11)
fig.colorbar(sc, ax=ax, shrink=0.85).set_label("Elapsed (min)", fontsize=9)

# 11b. DO mg/L vs Temperature -- coloured by salinity
ax = axes[1]
sc = ax.scatter(df["Temp °C"], df["ODO mg/L"],
                c=df["Sal psu"], cmap="YlGnBu", s=18, alpha=0.7)
ax.axhline(DO_STRESS_MGL, ls="--", color="orange", lw=1, label="Stress")
ax.axhline(DO_HYPOXIC_MGL, ls="--", color="red", lw=1, label="Hypoxic")
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Salinity (psu)", fontsize=9)
ax.set_xlabel("Temperature (deg C)", fontsize=10)
ax.set_ylabel("DO Concentration (mg/L)", fontsize=10)
ax.set_title("DO vs Temperature (coloured by Salinity)", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("Dissolved Oxygen Analysis -- October 25, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "09_do_analysis.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 12. SALINITY ANALYSIS
# ==============================================================
print("Generating salinity analysis ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 12a. Salinity over time
ax = axes[0]
ax.plot(df["Datetime"], df["Sal psu"], marker=".", markersize=2,
        linewidth=0.8, color=ACCENT, alpha=0.8)
ax.set_ylabel("Salinity (psu)", fontsize=10)
ax.set_xlabel("Time", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("Salinity Over Time", fontsize=11, fontweight="bold")

# 12b. Salinity spatial map
ax = axes[1]
sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["Sal psu"],
                cmap="YlGnBu", s=20, edgecolors="none", alpha=0.85)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Salinity (psu)", fontsize=9)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Salinity Along GPS Track", fontsize=11, fontweight="bold")
ax.ticklabel_format(useOffset=False, style="plain")

# 12c. Salinity vs Specific Conductance
ax = axes[2]
ax.scatter(df["SpCond µS/cm"], df["Sal psu"], s=18,
           c=df["Elapsed_min"], cmap="plasma", alpha=0.7)
ax.set_xlabel("Specific Conductance (uS/cm)", fontsize=10)
ax.set_ylabel("Salinity (psu)", fontsize=10)
ax.set_title("Salinity vs Specific Conductance", fontsize=11, fontweight="bold")

fig.suptitle("Salinity Analysis -- October 25, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "10_salinity_analysis.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 13. TURBIDITY / TSS RELATIONSHIP
# ==============================================================
print("Generating turbidity-TSS plot ...")
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(df["Turbidity FNU"], df["TSS mg/L"], c=df["Elapsed_min"],
                cmap="plasma", s=18, alpha=0.7)
fig.colorbar(sc, ax=ax, shrink=0.85).set_label("Elapsed (min)", fontsize=9)
ax.set_xlabel("Turbidity (FNU)", fontsize=10)
ax.set_ylabel("TSS (mg/L)", fontsize=10)
ax.set_title("Turbidity vs Total Suspended Solids -- October 25, 2024", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "11_turbidity_tss.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 14. CHLOROPHYLL & PHYCOCYANIN
# ==============================================================
print("Generating pigment analysis ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Chlorophyll over time
ax = axes[0]
ax.plot(df["Datetime"], df["Chlorophyll RFU"], marker=".", markersize=2,
        linewidth=0.8, color=ACCENT, alpha=0.8)
ax.set_ylabel("Chlorophyll (RFU)", fontsize=10)
ax.set_xlabel("Time", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("Chlorophyll RFU Over Time", fontsize=11)

# Chlorophyll vs TAL PC (phycocyanin proxy)
ax = axes[1]
sc = ax.scatter(df["Chlorophyll RFU"], df["TAL PC RFU"], c=df["Elapsed_min"],
                cmap="plasma", s=18, alpha=0.7)
fig.colorbar(sc, ax=ax, shrink=0.85).set_label("Elapsed (min)", fontsize=9)
ax.set_xlabel("Chlorophyll (RFU)", fontsize=10)
ax.set_ylabel("TAL PC / Phycocyanin (RFU)", fontsize=10)
ax.set_title("Chlorophyll vs Phycocyanin", fontsize=11)

fig.suptitle("Algal Pigment Analysis -- October 25, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "12_pigments.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 15. ALTITUDE & BAROMETER (unique to Oct dataset)
# ==============================================================
print("Generating altitude & barometer plots ...")
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax = axes[0]
ax.plot(df["Datetime"], df["Altitude m"], marker=".", markersize=2,
        linewidth=0.8, color="#2ca02c", alpha=0.8)
ax.set_ylabel("Altitude (m)", fontsize=10)
ax.set_title("Altitude Over Time", fontsize=11, fontweight="bold")
ax.tick_params(labelsize=8)

ax = axes[1]
ax.plot(df["Datetime"], df["Barometer mmHg"], marker=".", markersize=2,
        linewidth=0.8, color="#d62728", alpha=0.8)
ax.set_ylabel("Barometer (mmHg)", fontsize=10)
ax.set_xlabel("Time", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("Barometric Pressure Over Time", fontsize=11, fontweight="bold")
ax.tick_params(labelsize=8)

fig.suptitle("Altitude & Barometer -- October 25, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "13_altitude_barometer.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# SUMMARY
# ==============================================================
print()
print("=" * 60)
print("EDA COMPLETE -- plots saved to:", OUT_DIR.resolve())
print("=" * 60)
print()
print("Plots generated:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  * {f.name}")
print()

# Quick flag summary
below_stress = df[df["ODO mg/L"] < DO_STRESS_MGL]
below_hypoxic = df[df["ODO mg/L"] < DO_HYPOXIC_MGL]
high_turb = df[df["Turbidity FNU"] > TURB_HIGH_FNU]
print(f"⚠  Rows with DO < {DO_STRESS_MGL} mg/L (stress):   {len(below_stress)} / {len(df)}")
print(f"⚠  Rows with DO < {DO_HYPOXIC_MGL} mg/L (hypoxic):  {len(below_hypoxic)} / {len(df)}")
print(f"⚠  Rows with Turbidity > {TURB_HIGH_FNU} FNU:        {len(high_turb)} / {len(df)}")
