# ============================================================
# Exploratory Data Analysis -- Underwater Missions, March 18 2025
# ============================================================
# This script performs an EDA on 20 underwater mission datasets
# collected on March 18, 2025 in Biscayne Bay.
#
# Two sensor schemas are present:
#   * Sonde schema (first 4 files, 24 cols) -- full water-quality
#     sonde with conductivity, nLF cond, SpCond, TDS, etc.
#   * ASV schema (remaining 16 files, 17 cols) -- autonomous
#     surface vehicle with BGA-PE, Chlorophyll (ug/L), etc.
#
# Common columns are harmonised into a single DataFrame.
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
DATA_DIR = Path(__file__).parent / "data" / "March 18th 2025"
OUT_DIR  = Path(__file__).parent / "eda_plots_march_2025"
OUT_DIR.mkdir(exist_ok=True)

# ── style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 140

# ── column name mappings to common schema ────────────────────
# Sonde schema -> common
SONDE_RENAME = {
    "Date (MM/DD/YYYY)": "Date",
    "Time (HH:MM:SS)":   "Time",
    "Cond (uS/cm)":      "Cond_uS",
    "Depth m":            "Depth_m",
    "nLF Cond µS/cm":    "nLF_Cond_uS",
    "ODO % sat":          "ODO_pct",
    "ODO % local":        "ODO_pct_local",
    "ODO mg/L":           "ODO_mgL",
    "Pressure psi a":     "Pressure_psia",
    "Sal psu":            "Sal_psu",
    "SpCond µS/cm":       "SpCond_uS",
    "TDS mg/L":           "TDS_mgL",
    "Turb (FNU)":         "Turb_FNU",
    "TSS mg/L":           "TSS_mgL",
    "Temp (C)":           "Temp_C",
    "Vertical Position m":"VertPos_m",
    "latitude":           "Latitude",
    "longitude":          "Longitude",
    "Altitude m":         "Altitude_m",
    "Barometer mmHg":     "Barometer_mmHg",
    "Battery V":          "Battery_V",
    "Cable Pwr V":        "CablePwr_V",
}

# ASV schema -> common
ASV_RENAME = {
    "Cond (uS/cm)":      "Cond_uS",
    "Depth (m)":          "Depth_m",
    "ODO (%sat)":         "ODO_pct",
    "ODO (mg/l)":         "ODO_mgL",
    "Pressure (psi a)":   "Pressure_psia",
    "Sal (PPT)":          "Sal_psu",
    "TSS (mg/L)":         "TSS_mgL",
    "Temp (C)":           "Temp_C",
    "Turb (FNU)":         "Turb_FNU",
    "latitude":           "Latitude",
    "longitude":          "Longitude",
    "BGA-PE (ug/L)":      "BGA_PE_ugL",
    "Chl (ug/L)":         "Chl_ugL",
}

# Features used in plots (common name -> col, unit)
FEATURE_MAP = {
    "Conductivity":  ("Cond_uS",       "uS/cm"),
    "Depth":         ("Depth_m",        "m"),
    "DO Saturation": ("ODO_pct",        "%"),
    "Salinity":      ("Sal_psu",        "psu"),
    "Turbidity":     ("Turb_FNU",       "FNU"),
    "Temperature":   ("Temp_C",         "deg C"),
    "Pressure":      ("Pressure_psia",  "psia"),
}

# Ecological thresholds (using % sat -- ASV missions don't record mg/L)
DO_STRESS_PCT  = 50.0   # % sat -- approximate fish stress
DO_HYPOXIC_PCT = 25.0   # % sat -- approximate hypoxic zone
TURB_HIGH_FNU  = 25.0

# ==============================================================
# 1. LOAD & HARMONISE DATA
# ==============================================================
print("=" * 60)
print("Loading March 18, 2025 mission data ...")
print("=" * 60)

csv_files = sorted(DATA_DIR.glob("*.csv"))
frames = []
mission_meta = []

for fp in csv_files:
    raw = pd.read_csv(fp)
    # Derive a short mission label from the filename timestamp
    mission_label = fp.stem  # e.g. "2025-03-18-14-20-32"

    if len(raw.columns) == 24:
        schema = "sonde"
        tmp = raw.rename(columns=SONDE_RENAME)
        # Parse datetime -- sonde date format is M/D/YYYY
        tmp["Datetime"] = pd.to_datetime(
            tmp["Date"] + " " + tmp["Time"],
            format="%m/%d/%Y %H:%M:%S",
        )
    else:
        schema = "asv"
        tmp = raw.rename(columns=ASV_RENAME)
        # Parse datetime -- ASV date is YYYY-MM-DD, time is HH:MM:SS
        tmp["Datetime"] = pd.to_datetime(
            tmp["Date"] + " " + tmp["Time"],
            format="%Y-%m-%d %H:%M:%S",
        )

    tmp["Mission"] = mission_label
    tmp["Schema"] = schema
    frames.append(tmp)
    mission_meta.append({
        "Mission": mission_label,
        "Schema": schema,
        "Rows": len(tmp),
    })

all_data = pd.concat(frames, ignore_index=True)
all_data.sort_values("Datetime", inplace=True)
all_data.reset_index(drop=True, inplace=True)

# Elapsed time
t0 = all_data["Datetime"].min()
all_data["Elapsed_min"] = (all_data["Datetime"] - t0).dt.total_seconds() / 60.0

# Short mission ID for legends (last 8 chars = HH-MM-SS)
all_data["MissionShort"] = all_data["Mission"].str[-8:]

n_sonde = len(all_data[all_data["Schema"] == "sonde"])
n_asv   = len(all_data[all_data["Schema"] == "asv"])
print(f"Total rows: {len(all_data)}  (sonde: {n_sonde}, ASV: {n_asv})")
print(f"Missions loaded: {len(csv_files)}")
print(f"Time span: {all_data['Datetime'].min()} -> {all_data['Datetime'].max()}")
print()

# ==============================================================
# 2. PER-MISSION SUMMARY TABLE
# ==============================================================
print("=" * 60)
print("Per-Mission Summary")
print("=" * 60)

summary_rows = []
for label in all_data["Mission"].unique():
    mdf = all_data[all_data["Mission"] == label]
    t_start = mdf["Datetime"].min()
    t_end   = mdf["Datetime"].max()
    dur = (t_end - t_start).total_seconds()
    summary_rows.append({
        "Mission": label[-8:],
        "Schema": mdf["Schema"].iloc[0],
        "Rows": len(mdf),
        "Start": t_start.strftime("%H:%M:%S"),
        "End": t_end.strftime("%H:%M:%S"),
        "Dur(s)": int(dur),
        "Depth max": round(mdf["Depth_m"].max(), 2) if "Depth_m" in mdf and mdf["Depth_m"].notna().any() else np.nan,
        "Temp mean": round(mdf["Temp_C"].mean(), 2) if mdf["Temp_C"].notna().any() else np.nan,
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
print()

# ==============================================================
# 3. DESCRIPTIVE STATISTICS
# ==============================================================
print("=" * 60)
print("Descriptive Statistics (all missions combined)")
print("=" * 60)

stat_cols = [v[0] for v in FEATURE_MAP.values()]
valid_stat_cols = [c for c in stat_cols if c in all_data.columns]
desc = all_data[valid_stat_cols].describe().T
desc["missing"] = all_data[valid_stat_cols].isna().sum()
desc["missing%"] = (desc["missing"] / len(all_data) * 100).round(1)
print(desc.to_string())
print()

# ── colour palette for missions ──────────────────────────────
missions_ordered = list(all_data["MissionShort"].unique())
palette = sns.color_palette("tab20", n_colors=len(missions_ordered))
MISSION_COLORS = dict(zip(missions_ordered, [sns.color_palette("tab20").as_hex()[i] for i in range(len(missions_ordered))]))
SCHEMA_COLORS = {"sonde": "#1f77b4", "asv": "#ff7f0e"}

# ==============================================================
# 4. CORRELATION HEAT MAP
# ==============================================================
print("Generating correlation heatmap ...")
fig, ax = plt.subplots(figsize=(10, 8))
corr = all_data[valid_stat_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    annot_kws={"size": 8},
)
ax.set_title("Feature Correlation -- All Missions (March 18, 2025)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "01_correlation_heatmap.png")
plt.close(fig)

# ==============================================================
# 5. DISTRIBUTION PLOTS
# ==============================================================
print("Generating distribution plots ...")
features_to_plot = [
    "Conductivity", "Depth", "DO Saturation",
    "Salinity", "Turbidity", "Temperature", "Pressure",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for i, feat in enumerate(features_to_plot):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    for schema, color in SCHEMA_COLORS.items():
        subset = all_data.loc[all_data["Schema"] == schema, col].dropna()
        if len(subset) > 0:
            ax.hist(subset, bins=25, alpha=0.5, color=color,
                    label=schema, edgecolor="none")
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_xlabel(label_str, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
axes[-1].set_visible(False)  # 7 features -> hide unused 8th subplot

fig.suptitle("Feature Distributions by Schema -- March 18, 2025", fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(OUT_DIR / "02_distributions.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 6. BOX PLOTS BY SCHEMA
# ==============================================================
print("Generating box plots ...")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for i, feat in enumerate(features_to_plot):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    data_by = []
    labels_by = []
    for schema in ["sonde", "asv"]:
        vals = all_data.loc[all_data["Schema"] == schema, col].dropna()
        if len(vals) > 0:
            data_by.append(vals.values)
            labels_by.append(schema)
    if data_by:
        bp = ax.boxplot(data_by, tick_labels=labels_by, patch_artist=True,
                        medianprops=dict(color="black"))
        for patch, schema in zip(bp["boxes"], labels_by):
            patch.set_facecolor(SCHEMA_COLORS[schema])
            patch.set_alpha(0.6)
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_ylabel(label_str, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
axes[-1].set_visible(False)  # 7 features -> hide unused 8th subplot

fig.suptitle("Feature Box Plots by Schema -- March 18, 2025", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "03_boxplots_schema.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 7. BOX PLOTS BY MISSION
# ==============================================================
print("Generating per-mission box plots ...")
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.ravel()
for i, feat in enumerate(features_to_plot):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    data_list, label_list = [], []
    for ms in missions_ordered:
        vals = all_data.loc[all_data["MissionShort"] == ms, col].dropna()
        if len(vals) > 0:
            data_list.append(vals.values)
            label_list.append(ms)
    if data_list:
        bp = ax.boxplot(data_list, tick_labels=label_list, patch_artist=True,
                        medianprops=dict(color="black"))
        for patch, ms in zip(bp["boxes"], label_list):
            patch.set_facecolor(MISSION_COLORS[ms])
            patch.set_alpha(0.6)
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_ylabel(label_str, fontsize=8)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=8)
axes[-1].set_visible(False)  # 7 features -> hide unused 8th subplot

fig.suptitle("Feature Box Plots by Mission -- March 18, 2025", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "04_boxplots_mission.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 8. TIME-SERIES PLOTS
# ==============================================================
print("Generating time-series plots ...")
ts_features = ["DO Saturation", "Temperature", "Salinity", "Turbidity", "Depth", "Pressure"]

fig, axes = plt.subplots(len(ts_features), 1,
                         figsize=(14, 3.0 * len(ts_features)), sharex=True)
for i, feat in enumerate(ts_features):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    for ms in missions_ordered:
        mdf = all_data[all_data["MissionShort"] == ms]
        ax.plot(mdf["Datetime"], mdf[col], marker=".", markersize=2,
                linewidth=0.6, color=MISSION_COLORS[ms], alpha=0.8)
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_ylabel(label_str, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    if feat == "DO Saturation":
        ax.axhline(DO_STRESS_PCT, ls="--", color="orange", lw=1,
                    label=f"Stress ({DO_STRESS_PCT}%)")
        ax.axhline(DO_HYPOXIC_PCT, ls="--", color="red", lw=1,
                    label=f"Hypoxic ({DO_HYPOXIC_PCT}%)")
        ax.legend(fontsize=7)
    if feat == "Turbidity":
        ax.axhline(TURB_HIGH_FNU, ls="--", color="red", lw=1,
                    label=f"High ({TURB_HIGH_FNU} FNU)")
        ax.legend(fontsize=7)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[-1].set_xlabel("Time (HH:MM)", fontsize=9)
fig.suptitle("Sensor Time Series -- March 18, 2025", fontsize=13, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_DIR / "05_timeseries.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 9. GPS TRACK MAP (coloured by elapsed time)
# ==============================================================
print("Generating GPS track map ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) coloured by elapsed time
ax = axes[0]
sc = ax.scatter(all_data["Longitude"], all_data["Latitude"],
                c=all_data["Elapsed_min"], cmap="plasma", s=14,
                alpha=0.85, edgecolors="none")
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Elapsed Time (min)", fontsize=9)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Coloured by Time", fontsize=11, fontweight="bold")
ax.ticklabel_format(useOffset=False, style="plain")

# (b) coloured by schema
ax = axes[1]
for schema, color in SCHEMA_COLORS.items():
    subset = all_data[all_data["Schema"] == schema]
    ax.scatter(subset["Longitude"], subset["Latitude"], s=14,
               color=color, label=schema, alpha=0.7, edgecolors="none")
    ax.plot(subset["Longitude"], subset["Latitude"], linewidth=0.3,
            color=color, alpha=0.3)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Coloured by Schema", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.ticklabel_format(useOffset=False, style="plain")

fig.suptitle("Mission GPS Tracks -- March 18, 2025", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "06_gps_tracks.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 10. GPS + DEPTH MAP
# ==============================================================
print("Generating depth map ...")
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(
    all_data["Longitude"], all_data["Latitude"],
    c=all_data["Depth_m"], cmap="viridis_r", s=18,
    edgecolors="none", alpha=0.85,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Depth (m)", fontsize=10)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Depth Along GPS Tracks -- March 18, 2025", fontsize=12)
ax.ticklabel_format(useOffset=False, style="plain")
plt.tight_layout()
fig.savefig(OUT_DIR / "07_depth_map.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 11. PAIR PLOT
# ==============================================================
print("Generating pair plot ...")
pair_cols = ["ODO_pct", "Sal_psu", "Temp_C", "Turb_FNU", "Depth_m"]
pair_df = all_data[pair_cols + ["Schema"]].dropna()

g = sns.pairplot(pair_df, hue="Schema", diag_kind="kde",
                 palette=SCHEMA_COLORS,
                 plot_kws={"s": 12, "alpha": 0.5},
                 diag_kws={"linewidth": 1})
g.figure.suptitle("Pair Plot -- Key Water Quality Features (March 18, 2025)",
                   y=1.01, fontsize=13)
g.savefig(OUT_DIR / "08_pairplot.png", bbox_inches="tight")
plt.close(g.figure)

# ==============================================================
# 12. DEPTH PROFILES
# ==============================================================
print("Generating depth profiles ...")
depth_features = ["DO Saturation", "Salinity", "Temperature", "Turbidity"]

fig, axes = plt.subplots(1, len(depth_features), figsize=(16, 6))
for i, feat in enumerate(depth_features):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    for schema, color in SCHEMA_COLORS.items():
        subset = all_data[all_data["Schema"] == schema].dropna(subset=["Depth_m", col])
        if len(subset) > 0:
            ax.scatter(subset[col], subset["Depth_m"], s=14,
                       color=color, label=schema, alpha=0.6)
    ax.invert_yaxis()
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_xlabel(label_str, fontsize=9)
    if i == 0:
        ax.set_ylabel("Depth (m)", fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)

fig.suptitle("Depth Profiles -- March 18, 2025", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "09_depth_profiles.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 13. DISSOLVED OXYGEN ANALYSIS
# ==============================================================
print("Generating DO analysis ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DO % sat by schema over time
ax = axes[0]
for schema, color in SCHEMA_COLORS.items():
    subset = all_data[all_data["Schema"] == schema]
    ax.plot(subset["Datetime"], subset["ODO_pct"], marker=".", markersize=3,
            linewidth=0.6, color=color, label=schema, alpha=0.8)
ax.axhline(DO_STRESS_PCT, ls="--", color="orange", lw=1,
            label=f"Stress ({DO_STRESS_PCT}%)")
ax.axhline(DO_HYPOXIC_PCT, ls="--", color="red", lw=1,
            label=f"Hypoxic ({DO_HYPOXIC_PCT}%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Time", fontsize=10)
ax.set_ylabel("DO Saturation (%)", fontsize=10)
ax.set_title("DO Saturation Over Time by Schema", fontsize=11)
ax.legend(fontsize=8)

# DO % sat vs Temperature coloured by salinity
ax = axes[1]
valid = all_data.dropna(subset=["Temp_C", "ODO_pct", "Sal_psu"])
sc = ax.scatter(valid["Temp_C"], valid["ODO_pct"],
                c=valid["Sal_psu"], cmap="YlGnBu", s=18, alpha=0.7)
ax.axhline(DO_STRESS_PCT, ls="--", color="orange", lw=1, label="Stress")
ax.axhline(DO_HYPOXIC_PCT, ls="--", color="red", lw=1, label="Hypoxic")
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Salinity (psu)", fontsize=9)
ax.set_xlabel("Temperature (deg C)", fontsize=10)
ax.set_ylabel("DO Saturation (%)", fontsize=10)
ax.set_title("DO Saturation vs Temperature (coloured by Salinity)", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("Dissolved Oxygen Analysis -- March 18, 2025", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "10_do_analysis.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 14. SALINITY ANALYSIS
# ==============================================================
print("Generating salinity analysis ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Salinity over time
ax = axes[0]
for schema, color in SCHEMA_COLORS.items():
    subset = all_data[all_data["Schema"] == schema]
    ax.plot(subset["Datetime"], subset["Sal_psu"], marker=".", markersize=2,
            linewidth=0.6, color=color, label=schema, alpha=0.8)
ax.set_ylabel("Salinity (psu)", fontsize=10)
ax.set_xlabel("Time", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("Salinity Over Time", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# Salinity spatial map
ax = axes[1]
valid_sal = all_data.dropna(subset=["Sal_psu"])
sc = ax.scatter(valid_sal["Longitude"], valid_sal["Latitude"],
                c=valid_sal["Sal_psu"], cmap="YlGnBu", s=18,
                edgecolors="none", alpha=0.85)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Salinity (psu)", fontsize=9)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Salinity Along GPS Tracks", fontsize=11, fontweight="bold")
ax.ticklabel_format(useOffset=False, style="plain")

# Salinity vs Conductivity
ax = axes[2]
valid_sc = all_data.dropna(subset=["Cond_uS", "Sal_psu"])
for schema, color in SCHEMA_COLORS.items():
    subset = valid_sc[valid_sc["Schema"] == schema]
    ax.scatter(subset["Cond_uS"], subset["Sal_psu"], s=18,
               color=color, label=schema, alpha=0.6)
ax.set_xlabel("Conductivity (uS/cm)", fontsize=10)
ax.set_ylabel("Salinity (psu)", fontsize=10)
ax.set_title("Salinity vs Conductivity", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

fig.suptitle("Salinity Analysis -- March 18, 2025", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "11_salinity_analysis.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 15. TURBIDITY / TSS RELATIONSHIP
# ==============================================================
print("Generating turbidity-TSS plot ...")
fig, ax = plt.subplots(figsize=(7, 5))
for schema, color in SCHEMA_COLORS.items():
    subset = all_data[all_data["Schema"] == schema]
    ax.scatter(subset["Turb_FNU"], subset["TSS_mgL"], s=18,
               color=color, label=schema, alpha=0.6)
ax.set_xlabel("Turbidity (FNU)", fontsize=10)
ax.set_ylabel("TSS (mg/L)", fontsize=10)
ax.set_title("Turbidity vs Total Suspended Solids -- March 18, 2025", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "12_turbidity_tss.png", bbox_inches="tight")
plt.close(fig)

# ==============================================================
# 16. ASV-ONLY: BGA-PE & CHLOROPHYLL (ug/L)
# ==============================================================
print("Generating ASV pigment analysis ...")
asv_data = all_data[all_data["Schema"] == "asv"].copy()

if len(asv_data) > 0 and "BGA_PE_ugL" in asv_data.columns and "Chl_ugL" in asv_data.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # BGA-PE over time
    ax = axes[0]
    ax.plot(asv_data["Datetime"], asv_data["BGA_PE_ugL"], marker=".", markersize=2,
            linewidth=0.6, color="#2ca02c", alpha=0.8)
    ax.set_ylabel("BGA-PE (ug/L)", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title("Blue-Green Algae Over Time", fontsize=11, fontweight="bold")

    # Chlorophyll over time
    ax = axes[1]
    ax.plot(asv_data["Datetime"], asv_data["Chl_ugL"], marker=".", markersize=2,
            linewidth=0.6, color="#d62728", alpha=0.8)
    ax.set_ylabel("Chlorophyll (ug/L)", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title("Chlorophyll Over Time", fontsize=11, fontweight="bold")

    # BGA-PE vs Chlorophyll
    ax = axes[2]
    sc = ax.scatter(asv_data["Chl_ugL"], asv_data["BGA_PE_ugL"],
                    c=asv_data["Elapsed_min"], cmap="plasma", s=18, alpha=0.7)
    fig.colorbar(sc, ax=ax, shrink=0.85).set_label("Elapsed (min)", fontsize=9)
    ax.set_xlabel("Chlorophyll (ug/L)", fontsize=10)
    ax.set_ylabel("BGA-PE (ug/L)", fontsize=10)
    ax.set_title("BGA-PE vs Chlorophyll", fontsize=11, fontweight="bold")

    fig.suptitle("ASV Pigment Analysis -- March 18, 2025", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "13_asv_pigments.png", bbox_inches="tight")
    plt.close(fig)

# ==============================================================
# 17. SONDE-ONLY EXTRAS (nLF Cond, SpCond, TDS, Altitude, Barometer)
# ==============================================================
print("Generating sonde-only extras ...")
sonde_data = all_data[all_data["Schema"] == "sonde"].copy()

if len(sonde_data) > 0:
    sonde_extras = ["nLF_Cond_uS", "SpCond_uS", "TDS_mgL", "Altitude_m", "Barometer_mmHg"]
    sonde_extras = [c for c in sonde_extras if c in sonde_data.columns and sonde_data[c].notna().any()]
    if sonde_extras:
        fig, axes = plt.subplots(len(sonde_extras), 1,
                                 figsize=(14, 2.8 * len(sonde_extras)), sharex=True)
        if len(sonde_extras) == 1:
            axes = [axes]
        for i, col in enumerate(sonde_extras):
            ax = axes[i]
            ax.plot(sonde_data["Datetime"], sonde_data[col], marker=".",
                    markersize=3, linewidth=0.8, color="#1f77b4", alpha=0.8)
            ax.set_ylabel(col.replace("_", " "), fontsize=9)
            ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].set_xlabel("Time (HH:MM)", fontsize=9)
        fig.suptitle("Sonde-Only Sensors -- March 18, 2025", fontsize=13, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(OUT_DIR / "14_sonde_extras.png", bbox_inches="tight")
        plt.close(fig)

# ==============================================================
# 18. SCHEMA COMPARISON -- side-by-side overlapping features
# ==============================================================
print("Generating schema comparison ...")
overlap_feats = ["DO Saturation", "Salinity", "Temperature", "Turbidity"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for i, feat in enumerate(overlap_feats):
    col, unit = FEATURE_MAP[feat]
    ax = axes[i]
    for schema, color in SCHEMA_COLORS.items():
        subset = all_data.loc[all_data["Schema"] == schema, col].dropna()
        # KDE requires >1 point and non-zero variance
        if len(subset) > 1 and subset.std() > 0:
            try:
                subset.plot.kde(ax=ax, color=color, label=schema, linewidth=1.5)
            except Exception:
                ax.hist(subset, bins=20, alpha=0.4, color=color,
                        label=schema, density=True, edgecolor="none")
    label_str = f"{feat} ({unit})" if unit else feat
    ax.set_xlabel(label_str, fontsize=10)
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

fig.suptitle("Schema Comparison -- KDE of Shared Features (March 18, 2025)", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "15_schema_comparison.png", bbox_inches="tight")
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

# Quick flags
below_stress  = all_data[all_data["ODO_pct"] < DO_STRESS_PCT]
below_hypoxic = all_data[all_data["ODO_pct"] < DO_HYPOXIC_PCT]
high_turb     = all_data[all_data["Turb_FNU"] > TURB_HIGH_FNU]
print(f"⚠  Rows with DO < {DO_STRESS_PCT}% sat (stress):   {len(below_stress)} / {len(all_data)}")
print(f"⚠  Rows with DO < {DO_HYPOXIC_PCT}% sat (hypoxic):  {len(below_hypoxic)} / {len(all_data)}")
print(f"⚠  Rows with Turbidity > {TURB_HIGH_FNU} FNU:        {len(high_turb)} / {len(all_data)}")
