# ============================================================
# Exploratory Data Analysis -- Underwater Missions, March 15 2024
# ============================================================
# This script performs an EDA on 8 underwater mission datasets
# collected on March 15, 2024 in Biscayne Bay.
#
# Missions: 140, 151, 153, 156, 201, 211, 237, 309
#
# Sensor columns (23 total):
#   Date, Time, Chlorophyll RFU, Conductivity, Depth,
#   nLF Conductivity, ODO (% sat, % CB, mg/L), Pressure,
#   Salinity, SpCond, TAL PC RFU, TDS, Turbidity, TSS,
#   pH, pH mV, Temperature, Vertical Position, Lat, Lon
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
DATA_DIR = Path(__file__).parent / "data" / "March 15th 2024"
OUT_DIR  = Path(__file__).parent / "eda_plots_march_2024"
OUT_DIR.mkdir(exist_ok=True)

# ── style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 140

# ── mission metadata ─────────────────────────────────────────
MISSION_IDS = [140, 151, 153, 156, 201, 211, 237, 309]

MISSION_COLORS = {
    140: "#1f77b4",
    151: "#ff7f0e",
    153: "#2ca02c",
    156: "#d62728",
    201: "#9467bd",
    211: "#8c564b",
    237: "#e377c2",
    309: "#17becf",
}

# Columns of interest for the analysis (nice name -> csv col, unit)
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
}

# Ecological thresholds
DO_STRESS_MGL  = 4.0   # mg/L -- fish stress begins
DO_HYPOXIC_MGL = 2.0   # mg/L -- hypoxic / lethal
TURB_HIGH_FNU  = 25.0  # elevated turbidity

# ==============================================================
# 1. LOAD & COMBINE DATA
# ==============================================================
print("=" * 60)
print("Loading mission data ...")
print("=" * 60)

frames = []
for mid in MISSION_IDS:
    fp = DATA_DIR / f"mission{mid}-complete.csv"
    df = pd.read_csv(fp)
    df["Mission"] = mid
    # Build a datetime column
    df["Datetime"] = pd.to_datetime(
        df["Date (MM/DD/YYYY)"] + " " + df["Time (HH:mm:ss)"],
        format="%m/%d/%Y %H:%M:%S",
    )
    frames.append(df)

all_missions = pd.concat(frames, ignore_index=True)
all_missions.sort_values("Datetime", inplace=True)

print(f"Total rows across all missions: {len(all_missions)}")
print(f"Time span: {all_missions['Datetime'].min()} -> {all_missions['Datetime'].max()}")
print()

# ==============================================================
# 2. PER-MISSION SUMMARY TABLE
# ==============================================================
print("=" * 60)
print("Per-Mission Summary")
print("=" * 60)

summary_rows = []
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    t0 = mdf["Datetime"].min()
    t1 = mdf["Datetime"].max()
    dur = (t1 - t0).total_seconds()
    summary_rows.append({
        "Mission": mid,
        "Rows": len(mdf),
        "Start": t0.strftime("%H:%M:%S"),
        "End": t1.strftime("%H:%M:%S"),
        "Duration (s)": int(dur),
        "Lat range": f"{mdf['Latitude'].min():.5f} - {mdf['Latitude'].max():.5f}",
        "Lon range": f"{mdf['Longitude'].min():.5f} - {mdf['Longitude'].max():.5f}",
        "Depth max (m)": round(mdf["Depth m"].max(), 3),
        "Temp mean (deg C)": round(mdf["Temp °C"].mean(), 2),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
print()

# ==============================================================
# 3. GLOBAL DESCRIPTIVE STATISTICS
# ==============================================================
print("=" * 60)
print("Descriptive Statistics (all missions combined)")
print("=" * 60)

numeric_cols = [v[0] for v in FEATURE_MAP.values()]
desc = all_missions[numeric_cols].describe().T
desc["missing"] = all_missions[numeric_cols].isna().sum()
desc["missing%"] = (desc["missing"] / len(all_missions) * 100).round(1)
print(desc.to_string())
print()

# ==============================================================
# 4. CORRELATION HEAT MAP
# ==============================================================
print("Generating correlation heatmap ...")
fig, ax = plt.subplots(figsize=(11, 9))
corr = all_missions[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    annot_kws={"size": 7},
)
ax.set_title("Feature Correlation -- All Missions (March 15, 2024)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "01_correlation_heatmap.png")
plt.close(fig)

# ==============================================================
# 5. DISTRIBUTION PLOTS  (histograms + KDE per mission)
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
    for mid in MISSION_IDS:
        subset = all_missions.loc[all_missions["Mission"] == mid, col].dropna()
        if len(subset) > 1:
            ax.hist(subset, bins=20, alpha=0.35, color=MISSION_COLORS[mid],
                    label=str(mid), edgecolor="none")
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_xlabel(label_str, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

# single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8,
           title="Mission", title_fontsize=9)
fig.suptitle("Feature Distributions by Mission -- March 15, 2024", fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT_DIR / "02_distributions.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 6. BOX PLOTS (per mission)
# ==============================================================
print("Generating box plots ...")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for i, feat in enumerate(features_to_plot):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    data_list, labels_list = [], []
    for mid in MISSION_IDS:
        vals = all_missions.loc[all_missions["Mission"] == mid, col].dropna()
        data_list.append(vals.values)
        labels_list.append(str(mid))
    bp = ax.boxplot(data_list, tick_labels=labels_list, patch_artist=True,
                    medianprops=dict(color="black"))
    for patch, mid in zip(bp["boxes"], MISSION_IDS):
        patch.set_facecolor(MISSION_COLORS[mid])
        patch.set_alpha(0.6)
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_ylabel(label_str, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Mission", fontsize=9)

fig.suptitle("Feature Box Plots by Mission -- March 15, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "03_boxplots.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 7. TIME-SERIES PLOTS (key sensors over time)
# ==============================================================
print("Generating time-series plots ...")
ts_features = ["DO Conc.", "Temperature", "Salinity", "Turbidity", "pH", "Depth"]

fig, axes = plt.subplots(len(ts_features), 1, figsize=(14, 3.0 * len(ts_features)),
                         sharex=True)
for i, feat in enumerate(ts_features):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    for mid in MISSION_IDS:
        mdf = all_missions[all_missions["Mission"] == mid]
        ax.plot(mdf["Datetime"], mdf[col], marker=".", markersize=3,
                linewidth=1, color=MISSION_COLORS[mid], label=str(mid), alpha=0.8)
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_ylabel(label_str, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    # add threshold lines
    if feat == "DO Conc.":
        ax.axhline(DO_STRESS_MGL, ls="--", color="orange", lw=1, label=f"Stress ({DO_STRESS_MGL} mg/L)")
        ax.axhline(DO_HYPOXIC_MGL, ls="--", color="red", lw=1, label=f"Hypoxic ({DO_HYPOXIC_MGL} mg/L)")
    if feat == "Turbidity":
        ax.axhline(TURB_HIGH_FNU, ls="--", color="red", lw=1, label=f"High ({TURB_HIGH_FNU} FNU)")

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[-1].set_xlabel("Time (HH:MM)", fontsize=9)
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in labels:
            handles.append(hi)
            labels.append(li)
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
           title="Mission / Threshold", title_fontsize=9)
fig.suptitle("Sensor Time Series -- March 15, 2024", fontsize=13, y=1.0)
plt.tight_layout(rect=[0, 0.04, 1, 0.98])
fig.savefig(OUT_DIR / "04_timeseries.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 8. GPS TRACK MAP (scatter: Lat vs Lon, coloured by mission)
# ==============================================================
print("Generating GPS track map ...")
fig, ax = plt.subplots(figsize=(9, 7))
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    ax.scatter(mdf["Longitude"], mdf["Latitude"], s=18,
               color=MISSION_COLORS[mid], label=str(mid), alpha=0.8, edgecolors="none")
    ax.plot(mdf["Longitude"], mdf["Latitude"], linewidth=0.6,
            color=MISSION_COLORS[mid], alpha=0.5)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Mission GPS Tracks -- March 15, 2024", fontsize=12)
ax.legend(title="Mission", fontsize=8, title_fontsize=9)
ax.ticklabel_format(useOffset=False, style="plain")
plt.tight_layout()
fig.savefig(OUT_DIR / "05_gps_tracks.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 9. GPS + DEPTH SCATTER (colour = depth)
# ==============================================================
print("Generating depth map ...")
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(
    all_missions["Longitude"], all_missions["Latitude"],
    c=all_missions["Depth m"], cmap="viridis_r", s=20,
    edgecolors="none", alpha=0.85,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Depth (m)", fontsize=10)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Depth Along GPS Tracks -- March 15, 2024", fontsize=12)
ax.ticklabel_format(useOffset=False, style="plain")
plt.tight_layout()
fig.savefig(OUT_DIR / "06_depth_map.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 10. PAIR PLOT (key water quality features)
# ==============================================================
print("Generating pair plot ...")
pair_cols = ["ODO mg/L", "Sal psu", "Temp °C", "Turbidity FNU", "pH", "Depth m"]
pair_df = all_missions[pair_cols + ["Mission"]].dropna()
pair_df["Mission"] = pair_df["Mission"].astype(str)

g = sns.pairplot(pair_df, hue="Mission", diag_kind="kde",
                 palette={str(k): v for k, v in MISSION_COLORS.items()},
                 plot_kws={"s": 15, "alpha": 0.6},
                 diag_kws={"linewidth": 1})
g.figure.suptitle("Pair Plot -- Key Water Quality Features (March 15, 2024)",
                   y=1.01, fontsize=13)
g.savefig(OUT_DIR / "07_pairplot.png", bbox_inches="tight")
plt.close(g.figure)

# ==============================================================
# 11. DEPTH PROFILES (sensor vs depth per mission)
# ==============================================================
print("Generating depth profiles ...")
depth_features = ["DO Conc.", "Salinity", "Temperature", "Turbidity"]

fig, axes = plt.subplots(1, len(depth_features), figsize=(16, 6))
for i, feat in enumerate(depth_features):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    for mid in MISSION_IDS:
        mdf = all_missions[all_missions["Mission"] == mid].dropna(subset=["Depth m", col])
        if len(mdf) > 0:
            ax.scatter(mdf[col], mdf["Depth m"], s=14,
                       color=MISSION_COLORS[mid], label=str(mid), alpha=0.7)
    ax.invert_yaxis()
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_xlabel(label_str, fontsize=9)
    if i == 0:
        ax.set_ylabel("Depth (m)", fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8,
           title="Mission", title_fontsize=9)
fig.suptitle("Depth Profiles -- March 15, 2024", fontsize=13)
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
fig.savefig(OUT_DIR / "08_depth_profiles.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 12. DISSOLVED OXYGEN DETAIL ANALYSIS
# ==============================================================
print("Generating DO analysis ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 12a. DO mg/L vs DO % sat -- coloured by mission
ax = axes[0]
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    ax.scatter(mdf["ODO % sat"], mdf["ODO mg/L"], s=18,
               color=MISSION_COLORS[mid], label=str(mid), alpha=0.7)
ax.axhline(DO_STRESS_MGL, ls="--", color="orange", lw=1)
ax.axhline(DO_HYPOXIC_MGL, ls="--", color="red", lw=1)
ax.set_xlabel("DO Saturation (%)", fontsize=10)
ax.set_ylabel("DO Concentration (mg/L)", fontsize=10)
ax.set_title("DO Concentration vs Saturation", fontsize=11)
ax.legend(title="Mission", fontsize=7, title_fontsize=8)

# 12b. DO mg/L vs Temperature -- coloured by salinity
ax = axes[1]
sc = ax.scatter(
    all_missions["Temp °C"], all_missions["ODO mg/L"],
    c=all_missions["Sal psu"], cmap="YlGnBu", s=18, alpha=0.7,
)
ax.axhline(DO_STRESS_MGL, ls="--", color="orange", lw=1, label="Stress")
ax.axhline(DO_HYPOXIC_MGL, ls="--", color="red", lw=1, label="Hypoxic")
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Salinity (psu)", fontsize=9)
ax.set_xlabel("Temperature (deg C)", fontsize=10)
ax.set_ylabel("DO Concentration (mg/L)", fontsize=10)
ax.set_title("DO vs Temperature (coloured by Salinity)", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("Dissolved Oxygen Analysis -- March 15, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "09_do_analysis.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 13. TURBIDITY / TSS RELATIONSHIP
# ==============================================================
print("Generating turbidity-TSS plot ...")
fig, ax = plt.subplots(figsize=(7, 5))
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    ax.scatter(mdf["Turbidity FNU"], mdf["TSS mg/L"], s=18,
               color=MISSION_COLORS[mid], label=str(mid), alpha=0.7)
ax.set_xlabel("Turbidity (FNU)", fontsize=10)
ax.set_ylabel("TSS (mg/L)", fontsize=10)
ax.set_title("Turbidity vs Total Suspended Solids -- March 15, 2024", fontsize=12)
ax.legend(title="Mission", fontsize=8, title_fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "10_turbidity_tss.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 14a. SALINITY ANALYSIS
# ==============================================================
print("Generating salinity analysis ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Salinity over time per mission
ax = axes[0, 0]
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    ax.plot(mdf["Datetime"], mdf["Sal psu"], marker=".", markersize=3,
            linewidth=1, color=MISSION_COLORS[mid], label=str(mid), alpha=0.8)
ax.set_ylabel("Salinity (psu)", fontsize=10)
ax.set_xlabel("Time", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("Salinity Over Time", fontsize=11, fontweight="bold")
ax.legend(title="Mission", fontsize=7, title_fontsize=8)

# (b) Salinity box plot per mission
ax = axes[0, 1]
data_list, labels_list = [], []
for mid in MISSION_IDS:
    vals = all_missions.loc[all_missions["Mission"] == mid, "Sal psu"].dropna()
    data_list.append(vals.values)
    labels_list.append(str(mid))
bp = ax.boxplot(data_list, tick_labels=labels_list, patch_artist=True,
                medianprops=dict(color="black"))
for patch, mid in zip(bp["boxes"], MISSION_IDS):
    patch.set_facecolor(MISSION_COLORS[mid])
    patch.set_alpha(0.6)
ax.set_ylabel("Salinity (psu)", fontsize=10)
ax.set_xlabel("Mission", fontsize=10)
ax.set_title("Salinity Distribution by Mission", fontsize=11, fontweight="bold")

# (c) Salinity spatial map (GPS coloured by salinity)
ax = axes[1, 0]
sc = ax.scatter(
    all_missions["Longitude"], all_missions["Latitude"],
    c=all_missions["Sal psu"], cmap="YlGnBu", s=20,
    edgecolors="none", alpha=0.85,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Salinity (psu)", fontsize=9)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Salinity Along GPS Tracks", fontsize=11, fontweight="bold")
ax.ticklabel_format(useOffset=False, style="plain")

# (d) Salinity vs Conductivity
ax = axes[1, 1]
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    ax.scatter(mdf["SpCond µS/cm"], mdf["Sal psu"], s=18,
               color=MISSION_COLORS[mid], label=str(mid), alpha=0.7)
ax.set_xlabel("Specific Conductance (uS/cm)", fontsize=10)
ax.set_ylabel("Salinity (psu)", fontsize=10)
ax.set_title("Salinity vs Specific Conductance", fontsize=11, fontweight="bold")
ax.legend(title="Mission", fontsize=7, title_fontsize=8)

fig.suptitle("Salinity Analysis -- March 15, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "11_salinity_analysis.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 15. CHLOROPHYLL & PHYCOCYANIN
# ==============================================================
print("Generating pigment analysis ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Chlorophyll over time
ax = axes[0]
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    ax.plot(mdf["Datetime"], mdf["Chlorophyll RFU"], marker=".", markersize=3,
            linewidth=1, color=MISSION_COLORS[mid], label=str(mid), alpha=0.8)
ax.set_ylabel("Chlorophyll (RFU)", fontsize=10)
ax.set_xlabel("Time", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("Chlorophyll RFU Over Time", fontsize=11)
ax.legend(title="Mission", fontsize=7, title_fontsize=8)

# Chlorophyll vs TAL PC (phycocyanin proxy)
ax = axes[1]
for mid in MISSION_IDS:
    mdf = all_missions[all_missions["Mission"] == mid]
    ax.scatter(mdf["Chlorophyll RFU"], mdf["TAL PC RFU"], s=18,
               color=MISSION_COLORS[mid], label=str(mid), alpha=0.7)
ax.set_xlabel("Chlorophyll (RFU)", fontsize=10)
ax.set_ylabel("TAL PC / Phycocyanin (RFU)", fontsize=10)
ax.set_title("Chlorophyll vs Phycocyanin", fontsize=11)
ax.legend(title="Mission", fontsize=7, title_fontsize=8)

fig.suptitle("Algal Pigment Analysis -- March 15, 2024", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "12_pigments.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# SUMMARY PRINT
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
below_stress = all_missions[all_missions["ODO mg/L"] < DO_STRESS_MGL]
below_hypoxic = all_missions[all_missions["ODO mg/L"] < DO_HYPOXIC_MGL]
high_turb = all_missions[all_missions["Turbidity FNU"] > TURB_HIGH_FNU]
print(f"⚠  Rows with DO < {DO_STRESS_MGL} mg/L (stress):   {len(below_stress)} / {len(all_missions)}")
print(f"⚠  Rows with DO < {DO_HYPOXIC_MGL} mg/L (hypoxic):  {len(below_hypoxic)} / {len(all_missions)}")
print(f"⚠  Rows with Turbidity > {TURB_HIGH_FNU} FNU:        {len(high_turb)} / {len(all_missions)}")
