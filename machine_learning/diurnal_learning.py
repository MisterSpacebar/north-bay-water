"""
diurnal_learning.py
-------------------
Computes and visualises diurnal (hour-of-day) patterns for every water-quality
feature in merged_keep.csv, broken out by calendar month.

Outputs
-------
diurnal_output/
    plots/   - one PNG per feature, subplots for each station that has data
               Lines = months, shading = +/-1 std around the hourly mean
    stats/   - one CSV per feature+station with hourly summary statistics
               (mean, std, median, min, max, count) per month
    summary_report.txt  - plain-text overview of all computed diurnal stats
"""

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# -- paths ---------------------------------------------------------------------
HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(HERE, "data", "cleaned_merged.csv")
OUT_ROOT    = os.path.join(HERE, "diurnal_output")
PLOTS_DIR   = os.path.join(OUT_ROOT, "plots")
STATS_DIR   = os.path.join(OUT_ROOT, "stats")

for d in (PLOTS_DIR, STATS_DIR):
    os.makedirs(d, exist_ok=True)

# -- load data -----------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df["datetime_5min"] = pd.to_datetime(df["datetime_5min"], format="mixed")
df["hour"]  = df["datetime_5min"].dt.hour
df["month"] = df["datetime_5min"].dt.to_period("M").astype(str)   # e.g. "2025-03"

# -- feature / station inventory -----------------------------------------------
STATIONS = [f"l{i}" for i in range(8)]

# Physical measurement families (skip wiper, lat, lon, time_hhmmss)
FEATURE_BASES = [
    "temperature_c",
    "specific_conductance_us_cm",
    "salinity_ppt",
    "pressure_psia",
    "depth_m",
    "odo_sat",
    "odo_mg_l",
    "turbidity_fnu",
    "tss_mg_l",
]

FEATURE_LABELS = {
    "temperature_c":              "Temperature (degC)",
    "specific_conductance_us_cm": "Specific Conductance (uS/cm)",
    "salinity_ppt":               "Salinity (ppt)",
    "pressure_psia":              "Pressure (psia)",
    "depth_m":                    "Depth (m)",
    "odo_sat":                    "DO Saturation (%)",
    "odo_mg_l":                   "DO (mg/L)",
    "turbidity_fnu":              "Turbidity (FNU)",
    "tss_mg_l":                   "TSS (mg/L)",
}

# -- discover which (feature, station) cols actually exist and have data -------
def col_name(feature, station):
    return f"{feature}_{station}"

def usable_col(feature, station):
    c = col_name(feature, station)
    if c not in df.columns:
        return False
    pct_valid = df[c].notna().mean()
    return pct_valid >= 0.05          # require at least 5 % non-NA

months_sorted = sorted(df["month"].unique())
MONTH_LABELS  = {m: pd.Period(m, "M").strftime("%b %Y") for m in months_sorted}

# colour map: one colour per month
cmap   = cm.get_cmap("tab10", len(months_sorted))
COLORS = {m: cmap(i) for i, m in enumerate(months_sorted)}

HOURS = np.arange(24)

# -- helper: compute hourly diurnal stats per month ----------------------------
def diurnal_stats(series: pd.Series) -> pd.DataFrame:
    """
    Given a Series aligned with df (sharing index), return a DataFrame with
    MultiIndex (month, hour) and columns mean/std/median/min/max/count.
    """
    tmp = pd.DataFrame({
        "value": series,
        "hour":  df["hour"],
        "month": df["month"],
    })
    grouped = tmp.groupby(["month", "hour"])["value"]
    stats = grouped.agg(
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
        count="count",
    ).reset_index()
    stats["std"] = stats["std"].fillna(0)
    return stats


# -- main loop -----------------------------------------------------------------
summary_lines = []
summary_lines.append("=" * 72)
summary_lines.append("DIURNAL PATTERN SUMMARY  -  North Biscayne Bay Water Stations")
summary_lines.append(f"Data file : {os.path.basename(DATA_PATH)}")
summary_lines.append(f"Months    : {', '.join(MONTH_LABELS.values())}")
summary_lines.append(f"Stations  : l0 - l7")
summary_lines.append("=" * 72)

for feature in FEATURE_BASES:
    y_label    = FEATURE_LABELS[feature]
    active_sta = [s for s in STATIONS if usable_col(feature, s)]

    if not active_sta:
        print(f"  [skip] {feature} - no stations with sufficient data")
        continue

    print(f"Processing feature: {feature}  ({len(active_sta)} stations)")

    # -- figure setup ----------------------------------------------------------
    ncols  = min(3, len(active_sta))
    nrows  = int(np.ceil(len(active_sta) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        sharey=False, squeeze=False,
    )
    fig.suptitle(
        f"Diurnal Pattern - {y_label}\n(mean +/- 1 SD by month)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # flatten axes grid for easy indexing
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    summary_lines.append("")
    summary_lines.append(f"{'-'*72}")
    summary_lines.append(f"FEATURE: {feature}")
    summary_lines.append(f"{'-'*72}")

    for idx, station in enumerate(active_sta):
        c    = col_name(feature, station)
        ax   = ax_flat[idx]
        stats = diurnal_stats(df[c])

        # -- save CSV stats -------------------------------------------------
        csv_path = os.path.join(STATS_DIR, f"{feature}_{station}.csv")
        stats.to_csv(csv_path, index=False, float_format="%.4f")

        # -- plot -----------------------------------------------------------
        for month in months_sorted:
            sub   = stats[stats["month"] == month].set_index("hour").reindex(HOURS)
            mean  = sub["mean"].values
            std   = sub["std"].values
            color = COLORS[month]
            label = MONTH_LABELS[month]
            ax.plot(HOURS, mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(HOURS, mean - std, mean + std,
                            color=color, alpha=0.15)

        ax.set_title(f"Station {station.upper()}", fontsize=10)
        ax.set_xlabel("Hour of Day (local)", fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_xticks(range(0, 24, 3))
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

        # legends only on first subplot to reduce clutter
        if idx == 0:
            ax.legend(fontsize=7, loc="best", title="Month", title_fontsize=8)

        # -- text summary (peak hour, range) per station/month -------------
        summary_lines.append(f"\n  Station {station.upper()}")
        for month in months_sorted:
            sub = stats[stats["month"] == month].set_index("hour").reindex(HOURS)
            mean_vals = sub["mean"].dropna()
            if mean_vals.empty:
                continue
            peak_hour = int(mean_vals.idxmax())
            trough_hr = int(mean_vals.idxmin())
            amplitude = mean_vals.max() - mean_vals.min()
            overall_mean = mean_vals.mean()
            summary_lines.append(
                f"    {MONTH_LABELS[month]:>10s} | "
                f"overall mean={overall_mean:9.3f} | "
                f"peak hour={peak_hour:02d}:00 (max={mean_vals.max():.3f}) | "
                f"trough hour={trough_hr:02d}:00 (min={mean_vals.min():.3f}) | "
                f"diurnal amplitude={amplitude:.3f}"
            )

    # hide unused subplot panels
    for idx in range(len(active_sta), len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{feature}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> plot saved: {os.path.basename(plot_path)}")

# -- write summary report ------------------------------------------------------
report_path = os.path.join(OUT_ROOT, "summary_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(summary_lines) + "\n")

print(f"\nDone.\n  Plots  : {PLOTS_DIR}")
print(f"  Stats  : {STATS_DIR}")
print(f"  Report : {report_path}")
