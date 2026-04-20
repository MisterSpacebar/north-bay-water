# ============================================================
# Comprehensive EDA -- Biscayne Bay Water Quality Data
# ============================================================
# This script performs an in-depth exploratory data analysis
# of the merged water-quality dataset across 8 stations:
#
#   L0 -- Biscayne Bay at FIU campus (open bay)
#   L1 -- Biscayne Canal (freshwater source)
#   L2 -- Biscayne Bay, canal-bay junction
#   L3 -- Little River station A (urban river)
#   L4 -- Little River station B (urban river)
#   L5 -- North Bay Village, north of JFK Causeway (lagoon)
#   L6 -- North Bay Village, south of JFK Causeway (lagoon)
#   L7 -- Miami River, south of Port Boulevard (NO DATA)
#
# Subsystems:
#   North of JFK Causeway: L0, L1 -> L2, L5
#   South of JFK Causeway: L3, L4 -> L6
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# ── output folder ────────────────────────────────────────────
OUT = Path("eda_plots")
OUT.mkdir(exist_ok=True)

# ── pretty settings ──────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 140

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
WATER_FEATURES = {
    "temperature_c":              ("Temperature", "deg C"),
    "specific_conductance_us_cm": ("Conductance", "uS/cm"),
    "salinity_ppt":               ("Salinity", "ppt"),
    "odo_sat":                    ("DO Saturation", "%"),
    "odo_mg_l":                   ("DO Concentration", "mg/L"),
    "turbidity_fnu":              ("Turbidity", "FNU"),
    "depth_m":                    ("Depth", "m"),
    "pressure_psia":              ("Pressure", "psia"),
}

# Stations that actually have data (L7 is all NaN)
STATIONS = ["l0", "l1", "l2", "l3", "l4", "l5", "l6"]

STATION_LABELS = {
    "l0": "L0 - FIU Bay",
    "l1": "L1 - Bisc. Canal",
    "l2": "L2 - Canal-Bay jct",
    "l3": "L3 - Little River A",
    "l4": "L4 - Little River B",
    "l5": "L5 - NBV North",
    "l6": "L6 - NBV South",
}

STATION_COLORS = {
    "l0": "#1f77b4",
    "l1": "#ff7f0e",
    "l2": "#2ca02c",
    "l3": "#d62728",
    "l4": "#9467bd",
    "l5": "#8c564b",
    "l6": "#e377c2",
}

# Ecological thresholds
DO_STRESS_MGL   = 4.0   # mg/L -- fish stress begins
DO_HYPOXIC_MGL  = 2.0   # mg/L -- hypoxic, lethal zone
DO_STRESS_PCT   = 50.0   # % sat rough equivalent
TURB_HIGH_FNU   = 25.0   # elevated turbidity
SAL_FRESH       = 0.5    # ppt -- essentially freshwater
SAL_BRACKISH_LO = 0.5
SAL_BRACKISH_HI = 30.0
SAL_MARINE      = 30.0

# ==================================================================
# STEP 1 -- LOAD & PARSE
# ==================================================================
print("=" * 65)
print("  STEP 1: Loading Data")
print("=" * 65)

raw = pd.read_csv("data/merged_keep.csv")
raw["datetime_5min"] = pd.to_datetime(raw["datetime_5min"], format="mixed")
raw.sort_values("datetime_5min", inplace=True)
raw.reset_index(drop=True, inplace=True)

print(f"  Rows:    {raw.shape[0]:,}")
print(f"  Columns: {raw.shape[1]}")
print(f"  Date range: {raw['datetime_5min'].min()} -> {raw['datetime_5min'].max()}")
total_days = (raw["datetime_5min"].max() - raw["datetime_5min"].min()).days
print(f"  Span: ~{total_days} days")
expected_5min = total_days * 24 * 12
print(f"  Expected 5-min rows for full coverage: ~{expected_5min:,}")
print(f"  Actual rows: {len(raw):,}  ({len(raw)/expected_5min:.0%} coverage)\n")


# ==================================================================
# STEP 2 -- MISSING DATA ANALYSIS
# ==================================================================
print("=" * 65)
print("  STEP 2: Missing Data Analysis")
print("=" * 65)

# 2a. Per-station availability summary
avail = {}
for s in STATIONS:
    # Use temperature as a proxy -- if temp is present, the station was on
    col = f"temperature_c_{s}"
    if col in raw.columns:
        n_valid = raw[col].notna().sum()
        pct = n_valid / len(raw) * 100
        avail[s] = {"valid_rows": n_valid, "pct_available": pct}
        print(f"  {STATION_LABELS[s]:25s}: {n_valid:>6,} / {len(raw):,}  ({pct:.1f}%)")

# L7 note
print(f"  {'L7 - Miami River':25s}:      0 / {len(raw):,}  (0.0%) -- NO DATA")

# 2b. Missing data heatmap
# Build a matrix: rows = timestamps (sampled to keep it readable),
# columns = station x feature
feat_cols = []
feat_labels = []
for s in STATIONS:
    for fk in ["temperature_c", "salinity_ppt", "odo_sat", "turbidity_fnu"]:
        col = f"{fk}_{s}"
        if col in raw.columns:
            feat_cols.append(col)
            feat_labels.append(f"{s.upper()} {WATER_FEATURES[fk][0][:4]}")

# Sample every 100th row for visibility
sample_idx = raw.index[::100]
missing_matrix = raw.loc[sample_idx, feat_cols].isnull().astype(int)

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(missing_matrix.T, cbar=False, cmap=["#4CAF50", "#F44336"],
            yticklabels=feat_labels, ax=ax)
ax.set_xlabel("Time index (every 100th row)")
ax.set_title("Missing Data Map (green = present, red = missing)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "01_missing_data_heatmap.png", bbox_inches="tight")
plt.close()
print(f"\n  -> saved {OUT}/01_missing_data_heatmap.png")

# 2c. Missing data by station (bar chart)
fig, ax = plt.subplots(figsize=(10, 5))
stations_sorted = sorted(avail, key=lambda s: avail[s]["pct_available"], reverse=True)
bars = ax.bar(
    [STATION_LABELS[s] for s in stations_sorted],
    [avail[s]["pct_available"] for s in stations_sorted],
    color=[STATION_COLORS[s] for s in stations_sorted],
    edgecolor="white",
)
ax.set_ylabel("% of rows with data")
ax.set_title("Data Availability by Station", fontweight="bold")
ax.set_ylim(0, 105)
for bar, s in zip(bars, stations_sorted):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{avail[s]['pct_available']:.1f}%", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "02_data_availability.png", bbox_inches="tight")
plt.close()
print(f"  -> saved {OUT}/02_data_availability.png\n")


# ==================================================================
# STEP 3 -- DESCRIPTIVE STATISTICS PER STATION
# ==================================================================
print("=" * 65)
print("  STEP 3: Descriptive Statistics")
print("=" * 65)

core_feats = ["temperature_c", "specific_conductance_us_cm",
              "salinity_ppt", "odo_sat", "odo_mg_l", "turbidity_fnu"]

for fk in core_feats:
    fname, unit = WATER_FEATURES[fk]
    print(f"\n  ── {fname} ({unit}) ──")
    rows = []
    for s in STATIONS:
        col = f"{fk}_{s}"
        if col not in raw.columns:
            continue
        d = raw[col].dropna()
        if len(d) == 0:
            continue
        rows.append({
            "Station": STATION_LABELS[s],
            "Count": len(d),
            "Mean": d.mean(),
            "Std": d.std(),
            "Min": d.min(),
            "25%": d.quantile(0.25),
            "Median": d.median(),
            "75%": d.quantile(0.75),
            "Max": d.max(),
            "Skew": d.skew(),
        })
    if rows:
        stats_df = pd.DataFrame(rows)
        print(stats_df.to_string(index=False, float_format="{:.2f}".format))


# ==================================================================
# STEP 4 -- DISTRIBUTION PLOTS  (histograms + KDE per station)
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 4: Distribution Plots")
print("=" * 65)

for fk in core_feats:
    fname, unit = WATER_FEATURES[fk]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 4a. Overlaid KDE
    for s in STATIONS:
        col = f"{fk}_{s}"
        if col not in raw.columns:
            continue
        d = raw[col].dropna()
        if len(d) < 50:
            continue
        # Clip extreme outliers for visualization (keep 0.5-99.5 percentile)
        lo, hi = d.quantile(0.005), d.quantile(0.995)
        d_clip = d[(d >= lo) & (d <= hi)]
        axes[0].hist(d_clip, bins=80, alpha=0.3, density=True,
                     color=STATION_COLORS[s], label=STATION_LABELS[s])
        d_clip.plot.kde(ax=axes[0], color=STATION_COLORS[s], linewidth=1.5)

    axes[0].set_xlabel(f"{fname} ({unit})")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{fname} -- Distribution by Station")
    axes[0].legend(fontsize=7)

    # 4b. Box plot
    box_data = []
    box_labels = []
    for s in STATIONS:
        col = f"{fk}_{s}"
        if col not in raw.columns:
            continue
        d = raw[col].dropna()
        if len(d) < 50:
            continue
        box_data.append(d.values)
        box_labels.append(s.upper())

    if box_data:
        bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True,
                             showfliers=False, medianprops=dict(color="black"))
        for patch, s in zip(bp["boxes"], STATIONS[:len(box_data)]):
            patch.set_facecolor(STATION_COLORS[s])
            patch.set_alpha(0.6)
        axes[1].set_ylabel(f"{fname} ({unit})")
        axes[1].set_title(f"{fname} -- Box Plot Comparison (outliers hidden)")

    plt.suptitle(fname, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT / f"03_dist_{fk}.png", bbox_inches="tight")
    plt.close()

print(f"  -> saved distribution plots to {OUT}/03_dist_*.png")


# ==================================================================
# STEP 5 -- TIME SERIES PLOTS
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 5: Time Series Plots")
print("=" * 65)

for fk in core_feats:
    fname, unit = WATER_FEATURES[fk]

    fig, ax = plt.subplots(figsize=(16, 5))
    for s in STATIONS:
        col = f"{fk}_{s}"
        if col not in raw.columns:
            continue
        d = raw[["datetime_5min", col]].dropna()
        if len(d) < 50:
            continue
        # Resample to hourly mean for smoother plots
        d = d.set_index("datetime_5min").resample("1h").mean().dropna()
        ax.plot(d.index, d[col], linewidth=0.6, alpha=0.7,
                color=STATION_COLORS[s], label=STATION_LABELS[s])

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{fname} ({unit})")
    ax.set_title(f"{fname} Over Time -- All Stations (hourly mean)", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / f"04_ts_{fk}.png", bbox_inches="tight")
    plt.close()

print(f"  -> saved time series plots to {OUT}/04_ts_*.png")


# ==================================================================
# STEP 6 -- SALINITY REGIME ANALYSIS
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 6: Salinity Regime Analysis")
print("=" * 65)

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# 6a. Raw salinity time series -- north subsystem
for s in ["l0", "l1", "l2", "l5"]:
    col = f"salinity_ppt_{s}"
    if col not in raw.columns:
        continue
    d = raw[["datetime_5min", col]].dropna().set_index("datetime_5min").resample("1h").mean().dropna()
    axes[0].plot(d.index, d[col], linewidth=0.6, alpha=0.8,
                 color=STATION_COLORS[s], label=STATION_LABELS[s])

axes[0].axhline(SAL_FRESH, color="gray", linestyle=":", linewidth=0.8, label=f"Freshwater < {SAL_FRESH} ppt")
axes[0].axhline(SAL_MARINE, color="navy", linestyle=":", linewidth=0.8, label=f"Marine > {SAL_MARINE} ppt")
axes[0].set_ylabel("Salinity (ppt)")
axes[0].set_title("North of JFK Causeway -- Salinity", fontweight="bold")
axes[0].legend(fontsize=7)

# 6b. South subsystem
for s in ["l3", "l4", "l6"]:
    col = f"salinity_ppt_{s}"
    if col not in raw.columns:
        continue
    d = raw[["datetime_5min", col]].dropna().set_index("datetime_5min").resample("1h").mean().dropna()
    axes[1].plot(d.index, d[col], linewidth=0.6, alpha=0.8,
                 color=STATION_COLORS[s], label=STATION_LABELS[s])

axes[1].axhline(SAL_FRESH, color="gray", linestyle=":", linewidth=0.8)
axes[1].axhline(SAL_MARINE, color="navy", linestyle=":", linewidth=0.8)
axes[1].set_ylabel("Salinity (ppt)")
axes[1].set_xlabel("Date")
axes[1].set_title("South of JFK Causeway -- Salinity", fontweight="bold")
axes[1].legend(fontsize=7)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout()
plt.savefig(OUT / "05_salinity_regime.png", bbox_inches="tight")
plt.close()

# 6c. Salinity classification pie charts
fig, axes = plt.subplots(1, len(STATIONS), figsize=(3.5 * len(STATIONS), 3.5))
for i, s in enumerate(STATIONS):
    col = f"salinity_ppt_{s}"
    if col not in raw.columns:
        axes[i].set_visible(False)
        continue
    d = raw[col].dropna()
    fresh    = (d < SAL_BRACKISH_LO).sum()
    brackish = ((d >= SAL_BRACKISH_LO) & (d < SAL_MARINE)).sum()
    marine   = (d >= SAL_MARINE).sum()
    total    = fresh + brackish + marine
    if total == 0:
        axes[i].set_visible(False)
        continue
    axes[i].pie(
        [fresh, brackish, marine],
        labels=["Fresh", "Brackish", "Marine"],
        colors=["#42A5F5", "#66BB6A", "#26A69A"],
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 7}
    )
    axes[i].set_title(STATION_LABELS[s], fontsize=9, fontweight="bold")

plt.suptitle("Salinity Classification -- % of Readings", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT / "06_salinity_classification.png", bbox_inches="tight")
plt.close()
print(f"  -> saved salinity regime plots to {OUT}/05_*, 06_*")


# ==================================================================
# STEP 7 -- DISSOLVED OXYGEN ECOLOGICAL ANALYSIS
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 7: Dissolved Oxygen -- Ecological Thresholds")
print("=" * 65)

# 7a. DO time series with thresholds
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

for ax_idx, (title, stations) in enumerate([
    ("North of JFK Causeway", ["l0", "l1", "l2", "l5"]),
    ("South of JFK Causeway", ["l3", "l4", "l6"]),
]):
    ax = axes[ax_idx]
    for s in stations:
        col = f"odo_mg_l_{s}"
        if col not in raw.columns:
            continue
        tmp = raw[["datetime_5min", col]].dropna().copy()
        # Filter out physically impossible values (sensor errors)
        tmp = tmp[(tmp[col] >= 0) & (tmp[col] <= 20)]
        d = tmp.set_index("datetime_5min").resample("1h").mean().dropna()
        ax.plot(d.index, d[col], linewidth=0.6, alpha=0.8,
                color=STATION_COLORS[s], label=STATION_LABELS[s])

    ax.axhline(DO_STRESS_MGL, color="orange", linestyle="--", linewidth=1.2,
               label=f"Fish stress ({DO_STRESS_MGL} mg/L)")
    ax.axhline(DO_HYPOXIC_MGL, color="red", linestyle="--", linewidth=1.2,
               label=f"Hypoxia ({DO_HYPOXIC_MGL} mg/L)")
    ax.fill_between(ax.get_xlim(), 0, DO_HYPOXIC_MGL, alpha=0.08, color="red")
    ax.set_ylabel("DO (mg/L)")
    ax.set_title(f"{title} -- Dissolved Oxygen", fontweight="bold")
    ax.legend(fontsize=7)

axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout()
plt.savefig(OUT / "07_do_thresholds.png", bbox_inches="tight")
plt.close()

# 7b. % of time below thresholds per station
print("\n  % of readings below ecological thresholds (DO mg/L):")
do_threshold_rows = []
for s in STATIONS:
    col = f"odo_mg_l_{s}"
    if col not in raw.columns:
        continue
    d = raw[col].dropna()
    if len(d) == 0:
        continue
    pct_stress  = (d < DO_STRESS_MGL).mean() * 100
    pct_hypoxic = (d < DO_HYPOXIC_MGL).mean() * 100
    do_threshold_rows.append({
        "Station": STATION_LABELS[s],
        "N": len(d),
        f"% < {DO_STRESS_MGL} mg/L (stress)": pct_stress,
        f"% < {DO_HYPOXIC_MGL} mg/L (hypoxia)": pct_hypoxic,
    })
    print(f"    {STATION_LABELS[s]:25s}: stress={pct_stress:5.1f}%  hypoxia={pct_hypoxic:5.1f}%")

# Bar chart of hypoxia %
fig, ax = plt.subplots(figsize=(10, 5))
do_df = pd.DataFrame(do_threshold_rows)
x = range(len(do_df))
width = 0.35
ax.bar([i - width/2 for i in x], do_df[f"% < {DO_STRESS_MGL} mg/L (stress)"],
       width, color="orange", alpha=0.8, label=f"< {DO_STRESS_MGL} mg/L (stress)")
ax.bar([i + width/2 for i in x], do_df[f"% < {DO_HYPOXIC_MGL} mg/L (hypoxia)"],
       width, color="red", alpha=0.8, label=f"< {DO_HYPOXIC_MGL} mg/L (hypoxia)")
ax.set_xticks(x)
ax.set_xticklabels(do_df["Station"], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("% of readings")
ax.set_title("Dissolved Oxygen -- % of Time Below Ecological Thresholds", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "08_do_threshold_bars.png", bbox_inches="tight")
plt.close()
print(f"  -> saved DO plots to {OUT}/07_*, 08_*")


# ==================================================================
# STEP 8 -- TURBIDITY SPIKE ANALYSIS
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 8: Turbidity Spike Analysis")
print("=" * 65)

fig, axes = plt.subplots(len(STATIONS), 1, figsize=(16, 3 * len(STATIONS)), sharex=True)
for i, s in enumerate(STATIONS):
    col = f"turbidity_fnu_{s}"
    if col not in raw.columns:
        axes[i].text(0.5, 0.5, "no data", ha="center", va="center",
                     transform=axes[i].transAxes)
        axes[i].set_ylabel(s.upper())
        continue
    d = raw[["datetime_5min", col]].dropna().set_index("datetime_5min")
    # Hourly mean and max
    hourly_mean = d.resample("1h").mean().dropna()
    hourly_max  = d.resample("1h").max().dropna()

    axes[i].fill_between(hourly_max.index, 0, hourly_max[col],
                         alpha=0.2, color=STATION_COLORS[s], label="Hourly max")
    axes[i].plot(hourly_mean.index, hourly_mean[col],
                 linewidth=0.5, color=STATION_COLORS[s], label="Hourly mean")
    axes[i].axhline(TURB_HIGH_FNU, color="red", linestyle=":", linewidth=0.8)
    axes[i].set_ylabel(f"{s.upper()}\n(FNU)", fontsize=8)

    pct_high = (raw[col].dropna() > TURB_HIGH_FNU).mean() * 100
    axes[i].text(0.01, 0.85, f"{STATION_LABELS[s]}\n>{TURB_HIGH_FNU} FNU: {pct_high:.1f}% of time",
                 transform=axes[i].transAxes, fontsize=7, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

axes[0].set_title("Turbidity Over Time -- Per Station (hourly)", fontweight="bold")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig(OUT / "09_turbidity_spikes.png", bbox_inches="tight")
plt.close()
print(f"  -> saved {OUT}/09_turbidity_spikes.png")


# ==================================================================
# STEP 9 -- DIURNAL (24-HOUR) PATTERNS
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 9: Diurnal (Daily) Patterns")
print("=" * 65)

raw["hour"] = raw["datetime_5min"].dt.hour + raw["datetime_5min"].dt.minute / 60

for fk in ["temperature_c", "odo_sat", "salinity_ppt", "turbidity_fnu"]:
    fname, unit = WATER_FEATURES[fk]
    fig, ax = plt.subplots(figsize=(10, 5))

    for s in STATIONS:
        col = f"{fk}_{s}"
        if col not in raw.columns:
            continue
        d = raw[["hour", col]].dropna()
        if len(d) < 200:
            continue
        # Bin by hour and compute mean +/- std
        hourly = d.groupby(d["hour"].round(0))[col].agg(["mean", "std"])
        ax.plot(hourly.index, hourly["mean"], linewidth=1.5,
                color=STATION_COLORS[s], label=STATION_LABELS[s])
        ax.fill_between(hourly.index,
                        hourly["mean"] - hourly["std"],
                        hourly["mean"] + hourly["std"],
                        alpha=0.1, color=STATION_COLORS[s])

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(f"{fname} ({unit})")
    ax.set_title(f"Diurnal Pattern -- {fname}\n(mean +/- 1 std by hour of day)", fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(OUT / f"10_diurnal_{fk}.png", bbox_inches="tight")
    plt.close()

print(f"  -> saved diurnal plots to {OUT}/10_diurnal_*.png")


# ==================================================================
# STEP 10 -- CORRELATION BETWEEN FEATURES (per station)
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 10: Intra-Station Feature Correlations")
print("=" * 65)

n_stations = len(STATIONS)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, s in enumerate(STATIONS):
    cols = {}
    for fk in core_feats:
        col = f"{fk}_{s}"
        if col in raw.columns:
            cols[col] = WATER_FEATURES[fk][0]
    if len(cols) < 2:
        axes[i].set_visible(False)
        continue
    corr = raw[list(cols.keys())].rename(columns=cols).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=axes[i], linewidths=0.5,
                cbar_kws={"shrink": 0.6})
    axes[i].set_title(STATION_LABELS[s], fontsize=10, fontweight="bold")
    axes[i].tick_params(labelsize=7)

# Hide extra subplot
for j in range(len(STATIONS), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Correlations Within Each Station", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "11_intra_station_corr.png", bbox_inches="tight")
plt.close()
print(f"  -> saved {OUT}/11_intra_station_corr.png")


# ==================================================================
# STEP 11 -- CROSS-STATION CORRELATION (per feature)
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 11: Cross-Station Correlations (per feature)")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, fk in enumerate(core_feats):
    fname, unit = WATER_FEATURES[fk]
    col_map = {}
    for s in STATIONS:
        col = f"{fk}_{s}"
        if col in raw.columns:
            col_map[col] = s.upper()
    if len(col_map) < 2:
        axes[idx].set_visible(False)
        continue

    corr = raw[list(col_map.keys())].rename(columns=col_map).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=axes[idx], linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.7})
    axes[idx].set_title(f"{fname} ({unit})", fontsize=10, fontweight="bold")

for j in range(len(core_feats), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Station-to-Station Correlation per Feature", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "12_cross_station_corr.png", bbox_inches="tight")
plt.close()
print(f"  -> saved {OUT}/12_cross_station_corr.png")


# ==================================================================
# STEP 12 -- SCATTER: TEMPERATURE vs. DO  (ecological relationship)
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 12: Temperature vs. DO Scatter (ecological link)")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# mg/L
for s in STATIONS:
    tc = f"temperature_c_{s}"
    do = f"odo_mg_l_{s}"
    if tc not in raw.columns or do not in raw.columns:
        continue
    d = raw[[tc, do]].dropna()
    if len(d) < 50:
        continue
    axes[0].scatter(d[tc], d[do], s=2, alpha=0.1,
                    color=STATION_COLORS[s], label=STATION_LABELS[s])

axes[0].axhline(DO_STRESS_MGL, color="orange", linestyle="--", linewidth=1)
axes[0].axhline(DO_HYPOXIC_MGL, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Temperature (deg C)")
axes[0].set_ylabel("DO (mg/L)")
axes[0].set_title("Temperature vs. Dissolved Oxygen (mg/L)")
axes[0].legend(fontsize=7, markerscale=5)

# % sat
for s in STATIONS:
    tc = f"temperature_c_{s}"
    do = f"odo_sat_{s}"
    if tc not in raw.columns or do not in raw.columns:
        continue
    d = raw[[tc, do]].dropna()
    if len(d) < 50:
        continue
    axes[1].scatter(d[tc], d[do], s=2, alpha=0.1,
                    color=STATION_COLORS[s], label=STATION_LABELS[s])

axes[1].set_xlabel("Temperature (deg C)")
axes[1].set_ylabel("DO Saturation (%)")
axes[1].set_title("Temperature vs. DO Saturation (%)")
axes[1].legend(fontsize=7, markerscale=5)

plt.suptitle("Temperature-DO Relationship (warmer water holds less O₂)",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT / "13_temp_vs_do.png", bbox_inches="tight")
plt.close()
print(f"  -> saved {OUT}/13_temp_vs_do.png")


# ==================================================================
# STEP 13 -- SCATTER: SALINITY vs. CONDUCTANCE (linearity check)
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 13: Salinity vs. Conductance Check")
print("=" * 65)

fig, ax = plt.subplots(figsize=(9, 6))
for s in STATIONS:
    sc = f"salinity_ppt_{s}"
    cc = f"specific_conductance_us_cm_{s}"
    if sc not in raw.columns or cc not in raw.columns:
        continue
    d = raw[[sc, cc]].dropna()
    if len(d) < 50:
        continue
    ax.scatter(d[cc], d[sc], s=2, alpha=0.1,
               color=STATION_COLORS[s], label=STATION_LABELS[s])

ax.set_xlabel("Specific Conductance (uS/cm)")
ax.set_ylabel("Salinity (ppt)")
ax.set_title("Conductance vs. Salinity -- All Stations\n(verifying the expected ~linear relationship)",
             fontweight="bold")
ax.legend(fontsize=7, markerscale=5)
plt.tight_layout()
plt.savefig(OUT / "14_conductance_vs_salinity.png", bbox_inches="tight")
plt.close()
print(f"  -> saved {OUT}/14_conductance_vs_salinity.png")


# ==================================================================
# STEP 14 -- SUBSYSTEM PAIR PLOTS
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 14: Subsystem Pair Plots")
print("=" * 65)

for name, stations in [("North", ["l0", "l1", "l2", "l5"]),
                        ("South", ["l3", "l4", "l6"])]:

    # Build merged columns: pick salinity + DO + turbidity for each station
    pair_feats = ["salinity_ppt", "odo_sat", "turbidity_fnu"]
    cols = {}
    for s in stations:
        for fk in pair_feats:
            col = f"{fk}_{s}"
            if col in raw.columns:
                short_label = f"{s.upper()} {WATER_FEATURES[fk][0][:5]}"
                cols[col] = short_label

    if len(cols) < 4:
        continue

    subset = raw[list(cols.keys())].rename(columns=cols).dropna()
    # Sample for speed
    if len(subset) > 3000:
        subset = subset.sample(3000, random_state=42)

    g = sns.pairplot(subset, corner=True, plot_kws={"s": 5, "alpha": 0.2},
                     diag_kws={"bins": 40})
    g.figure.suptitle(f"Pair Plot -- {name} Subsystem", y=1.01, fontweight="bold")
    g.savefig(OUT / f"15_pairplot_{name.lower()}.png", bbox_inches="tight")
    plt.close()

print(f"  -> saved {OUT}/15_pairplot_*.png")


# ==================================================================
# STEP 15 -- SUMMARY: KEY FINDINGS
# ==================================================================
print(f"\n{'=' * 65}")
print("  STEP 15: Automated Key Findings Summary")
print("=" * 65)

print("\n  ── Salinity Regimes ──")
for s in STATIONS:
    col = f"salinity_ppt_{s}"
    if col not in raw.columns:
        continue
    d = raw[col].dropna()
    if len(d) == 0:
        continue
    median_sal = d.median()
    if median_sal < SAL_BRACKISH_LO:
        regime = "FRESHWATER"
    elif median_sal < SAL_MARINE:
        regime = "BRACKISH"
    else:
        regime = "MARINE"
    print(f"    {STATION_LABELS[s]:25s}: median = {median_sal:6.1f} ppt  -> {regime}")

print("\n  ── DO Concern Stations ──")
for s in STATIONS:
    col = f"odo_mg_l_{s}"
    if col not in raw.columns:
        continue
    d = raw[col].dropna()
    if len(d) == 0:
        continue
    pct_stress = (d < DO_STRESS_MGL).mean() * 100
    if pct_stress > 5:
        print(f"    ⚠ {STATION_LABELS[s]:25s}: {pct_stress:.1f}% of readings below {DO_STRESS_MGL} mg/L")

print("\n  ── Turbidity Concern Stations ──")
for s in STATIONS:
    col = f"turbidity_fnu_{s}"
    if col not in raw.columns:
        continue
    d = raw[col].dropna()
    if len(d) == 0:
        continue
    pct_high = (d > TURB_HIGH_FNU).mean() * 100
    if pct_high > 10:
        print(f"    ⚠ {STATION_LABELS[s]:25s}: {pct_high:.1f}% of readings above {TURB_HIGH_FNU} FNU")

print("\n  ── Data Gaps ──")
for s in STATIONS:
    col = f"temperature_c_{s}"
    if col not in raw.columns:
        continue
    pct_missing = raw[col].isnull().mean() * 100
    if pct_missing > 20:
        print(f"    ⚠ {STATION_LABELS[s]:25s}: {pct_missing:.1f}% missing -- interpret with caution")

print(f"\n  ── JFK Causeway Effect ──")
north_sal = []
south_sal = []
for s in ["l2", "l5"]:
    col = f"salinity_ppt_{s}"
    if col in raw.columns:
        north_sal.extend(raw[col].dropna().tolist())
for s in ["l6"]:
    col = f"salinity_ppt_{s}"
    if col in raw.columns:
        south_sal.extend(raw[col].dropna().tolist())

if north_sal and south_sal:
    north_med = np.median(north_sal)
    south_med = np.median(south_sal)
    print(f"    North lagoon median salinity: {north_med:.1f} ppt")
    print(f"    South lagoon median salinity: {south_med:.1f} ppt")
    diff = south_med - north_med
    if abs(diff) > 2:
        direction = "saltier" if diff > 0 else "fresher"
        print(f"    -> South is {abs(diff):.1f} ppt {direction} than North -- causeway creates distinct zones")
    else:
        print(f"    -> Difference is small ({abs(diff):.1f} ppt) -- minimal causeway effect on salinity")


print(f"\n{'=' * 65}")
print(f"  All EDA plots saved to: {OUT.resolve()}")
print(f"  Total plots: {len(list(OUT.glob('*.png')))}")
print("=" * 65)
print("\n✓ EDA complete!")
