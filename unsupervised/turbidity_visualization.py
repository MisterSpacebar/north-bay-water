"""
Turbidity Visualizations
=========================
Generates targeted plots for investigating turbidity patterns, sediment events,
and light attenuation effects on seagrass across platforms L0 (FIU BBC),
L2 & L5 (north of JFK causeway), L6 (south).

Outputs saved to unsupervised/unsupervised_output/turbidity_visualizations/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "unsupervised_output", "turbidity_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLATFORM_FILES = {
    "L0 - FIU BBC": "raw-data-platformL0_parameters.csv",
    "L2 - North of Causeway": "raw-data-platformL2_parameters.csv",
    "L5 - North of Causeway": "raw-data-platformL5_parameters.csv",
    "L6 - South of Causeway": "raw-data-platformL6_parameters.csv",
}

COLORS = {
    "L0 - FIU BBC": "#e63946",
    "L2 - North of Causeway": "#457b9d",
    "L5 - North of Causeway": "#2a9d8f",
    "L6 - South of Causeway": "#e9c46a",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_platforms() -> dict[str, pd.DataFrame]:
    """Load and clean the four platforms of interest."""
    numeric_cols = [
        "ODO (mg/L)", "ODO (%Sat)", "Turbidity (FNU)",
        "Temperature (C)", "Salinity (PPT)", "Specific Conductance (uS/cm)",
        "Pressure (psia)", "Depth (m)",
    ]
    data = {}
    for label, fname in PLATFORM_FILES.items():
        path = os.path.join(RAW_DATA_DIR, fname)
        df = pd.read_csv(path, low_memory=False)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        data[label] = df
        print(f"  Loaded {label}: {len(df):,} rows  ({df['datetime'].min().date()} to {df['datetime'].max().date()})")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TURBIDITY TIME SERIES - ALL PLATFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_timeseries(data: dict[str, pd.DataFrame]):
    """Daily mean turbidity across all four platforms."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        daily = df.set_index("datetime")["Turbidity (FNU)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, label=label, color=COLORS[label], linewidth=1.2)
    ax.axhline(25, ls=":", color="orange", alpha=0.6, label="Moderate turbidity (25 FNU)")
    ax.axhline(100, ls="--", color="red", alpha=0.5, label="High turbidity (100 FNU)")
    ax.set_ylabel("Turbidity (FNU)")
    ax.set_title("Daily Mean Turbidity by Platform")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_turbidity_timeseries_daily.png"), dpi=150)
    plt.close(fig)
    print("  Saved 01_turbidity_timeseries_daily.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TURBIDITY TIME SERIES (LOG SCALE)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_timeseries_log(data: dict[str, pd.DataFrame]):
    """Daily mean turbidity on log scale to see patterns across magnitudes."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        daily = df.set_index("datetime")["Turbidity (FNU)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, label=label, color=COLORS[label], linewidth=1.2)
    ax.set_yscale("log")
    ax.axhline(25, ls=":", color="orange", alpha=0.6, label="25 FNU")
    ax.axhline(100, ls="--", color="red", alpha=0.5, label="100 FNU")
    ax.axhline(1000, ls="--", color="darkred", alpha=0.5, label="1000 FNU")
    ax.set_ylabel("Turbidity (FNU) -- log scale")
    ax.set_title("Daily Mean Turbidity by Platform (Log Scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_turbidity_timeseries_log.png"), dpi=150)
    plt.close(fig)
    print("  Saved 02_turbidity_timeseries_log.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TURBIDITY RANGE (MIN/MAX) PER DAY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_range(data: dict[str, pd.DataFrame]):
    """Daily min/max turbidity showing spike events per platform."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        daily = df.set_index("datetime")["Turbidity (FNU)"].resample("D").agg(["min", "max", "mean"]).dropna()
        ax.fill_between(daily.index, daily["min"], daily["max"],
                        alpha=0.25, color=COLORS[label], label="Min-Max range")
        ax.plot(daily.index, daily["mean"], color=COLORS[label], linewidth=1.2, label="Daily mean")
        ax.set_ylabel("Turbidity (FNU)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.suptitle("Daily Turbidity Range (Min-Max) by Platform", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "03_turbidity_daily_range.png"), dpi=150)
    plt.close(fig)
    print("  Saved 03_turbidity_daily_range.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TURBIDITY DISTRIBUTION VIOLIN PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_violins(data: dict[str, pd.DataFrame]):
    """Violin plots of turbidity distribution per platform (clipped at 95th pctile)."""
    records = []
    for label, df in data.items():
        vals = df["Turbidity (FNU)"].dropna()
        clip = vals.quantile(0.95)
        for v in vals[vals <= clip]:
            records.append({"Platform": label.split(" - ")[0], "Turbidity (FNU)": v})
    vdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=vdf, x="Platform", y="Turbidity (FNU)", inner="quartile",
                   palette=[COLORS[k] for k in PLATFORM_FILES.keys()], ax=ax)
    ax.axhline(25, ls=":", color="orange", alpha=0.6, label="Moderate (25 FNU)")
    ax.axhline(100, ls="--", color="red", alpha=0.5, label="High (100 FNU)")
    ax.set_title("Turbidity Distribution by Platform (clipped at 95th pctile)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_turbidity_violin_plots.png"), dpi=150)
    plt.close(fig)
    print("  Saved 04_turbidity_violin_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TURBIDITY vs DO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_vs_do(data: dict[str, pd.DataFrame]):
    """Scatter: turbidity vs DO -- does high turbidity suppress oxygen?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=True)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Turbidity (FNU)", "ODO (mg/L)"]].dropna()
        clip = sub["Turbidity (FNU)"].quantile(0.99)
        sub_clip = sub[sub["Turbidity (FNU)"] <= clip]
        ax.scatter(sub_clip["Turbidity (FNU)"], sub_clip["ODO (mg/L)"],
                   s=1, alpha=0.15, color=COLORS[label])
        ax.axhline(2, ls="--", color="red", alpha=0.6, lw=0.8)
        ax.set_xlabel("Turbidity (FNU)")
        ax.set_ylabel("DO (mg/L)")
        ax.set_title(label, fontsize=10)
        r = sub["Turbidity (FNU)"].corr(sub["ODO (mg/L)"])
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Turbidity vs Dissolved Oxygen", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "05_turbidity_vs_do.png"), dpi=150)
    plt.close(fig)
    print("  Saved 05_turbidity_vs_do.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TURBIDITY vs SALINITY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_vs_salinity(data: dict[str, pd.DataFrame]):
    """Scatter: turbidity vs salinity -- does runoff bring sediment?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Turbidity (FNU)", "Salinity (PPT)"]].dropna()
        clip = sub["Turbidity (FNU)"].quantile(0.99)
        sub_clip = sub[sub["Turbidity (FNU)"] <= clip]
        ax.scatter(sub_clip["Salinity (PPT)"], sub_clip["Turbidity (FNU)"],
                   s=1, alpha=0.15, color=COLORS[label])
        ax.set_xlabel("Salinity (PPT)")
        ax.set_ylabel("Turbidity (FNU)")
        ax.set_title(label, fontsize=10)
        r = sub["Salinity (PPT)"].corr(sub["Turbidity (FNU)"])
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Salinity vs Turbidity -- Runoff Brings Sediment?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "06_turbidity_vs_salinity.png"), dpi=150)
    plt.close(fig)
    print("  Saved 06_turbidity_vs_salinity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TURBIDITY vs DEPTH
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_vs_depth(data: dict[str, pd.DataFrame]):
    """Scatter: turbidity vs depth -- is turbidity bottom-driven?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Turbidity (FNU)", "Depth (m)"]].dropna()
        clip = sub["Turbidity (FNU)"].quantile(0.99)
        sub_clip = sub[sub["Turbidity (FNU)"] <= clip]
        ax.scatter(sub_clip["Depth (m)"], sub_clip["Turbidity (FNU)"],
                   s=1, alpha=0.15, color=COLORS[label])
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Turbidity (FNU)")
        ax.set_title(label, fontsize=10)
        r = sub["Depth (m)"].corr(sub["Turbidity (FNU)"])
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Depth vs Turbidity", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "07_turbidity_vs_depth.png"), dpi=150)
    plt.close(fig)
    print("  Saved 07_turbidity_vs_depth.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DIURNAL TURBIDITY CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_diurnal_turbidity(data: dict[str, pd.DataFrame]):
    """Average turbidity by hour of day -- tidal/current patterns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in data.items():
        df_tmp = df[["datetime", "Turbidity (FNU)"]].dropna().copy()
        df_tmp["hour"] = df_tmp["datetime"].dt.hour
        hourly = df_tmp.groupby("hour")["Turbidity (FNU)"].mean()
        ax.plot(hourly.index, hourly.values, "o-", label=label,
                color=COLORS[label], linewidth=1.5, markersize=4)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Mean Turbidity (FNU)")
    ax.set_title("Diurnal Turbidity Cycle by Platform")
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_diurnal_turbidity_cycle.png"), dpi=150)
    plt.close(fig)
    print("  Saved 08_diurnal_turbidity_cycle.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TURBIDITY HEATMAP (date x hour) - L0 and L2
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_heatmaps(data: dict[str, pd.DataFrame]):
    """Heatmap of turbidity by date x hour for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        df_tmp = df[["datetime", "Turbidity (FNU)"]].dropna().copy()
        df_tmp["date"] = df_tmp["datetime"].dt.date
        df_tmp["hour"] = df_tmp["datetime"].dt.hour
        pivot = df_tmp.pivot_table(index="date", columns="hour",
                                   values="Turbidity (FNU)", aggfunc="mean")
        # clip for colour scale readability
        vmax = pivot.stack().quantile(0.95)
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.06)))
        sns.heatmap(pivot, cmap="YlOrRd", vmin=0, vmax=vmax, ax=ax,
                    cbar_kws={"label": "Turbidity (FNU)"})
        ax.set_title(f"Turbidity Heatmap (date x hour) -- {label}")
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Date")
        step = max(1, len(pivot) // 20)
        for j, lbl in enumerate(ax.get_yticklabels()):
            if j % step != 0:
                lbl.set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"09_turbidity_heatmap_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 09_turbidity_heatmap_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. HIGH-TURBIDITY EVENT TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_high_turbidity_events(data: dict[str, pd.DataFrame]):
    """Highlight periods where turbidity exceeds 100 FNU."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    threshold = 100
    for ax, (label, df) in zip(axes, data.items()):
        daily = df.set_index("datetime")["Turbidity (FNU)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, color=COLORS[label], linewidth=1.2)
        above = daily[daily > threshold]
        if len(above) > 0:
            ax.fill_between(daily.index, threshold, daily.values,
                            where=daily.values > threshold,
                            alpha=0.3, color="red", label=f"> {threshold} FNU")
        ax.axhline(threshold, ls="--", color="red", alpha=0.4, lw=0.8)
        ax.set_ylabel("Turbidity (FNU)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.suptitle(f"High-Turbidity Events (daily mean > {threshold} FNU)", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "10_high_turbidity_events.png"), dpi=150)
    plt.close(fig)
    print("  Saved 10_high_turbidity_events.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TURBIDITY + DO DUAL-AXIS (L0, L2)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_do_dual(data: dict[str, pd.DataFrame]):
    """Dual-axis turbidity and DO for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[["Turbidity (FNU)", "ODO (mg/L)"]].resample("D").mean().dropna()
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(daily.index, daily["Turbidity (FNU)"], color="#9b5de5", linewidth=1.3, label="Turbidity (FNU)")
        ax1.set_ylabel("Turbidity (FNU)", color="#9b5de5")
        ax1.tick_params(axis="y", labelcolor="#9b5de5")

        ax2 = ax1.twinx()
        ax2.plot(daily.index, daily["ODO (mg/L)"], color="#2a9d8f", linewidth=1.3, label="DO (mg/L)")
        ax2.axhline(2, ls="--", color="red", alpha=0.5, lw=0.8)
        ax2.set_ylabel("DO (mg/L)", color="#2a9d8f")
        ax2.tick_params(axis="y", labelcolor="#2a9d8f")

        ax1.set_title(f"Turbidity vs DO -- {label}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"11_turbidity_vs_do_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 11_turbidity_vs_do_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TURBIDITY + SALINITY DUAL-AXIS (L0, L2)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_salinity_dual(data: dict[str, pd.DataFrame]):
    """Dual-axis turbidity and salinity for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[["Turbidity (FNU)", "Salinity (PPT)"]].resample("D").mean().dropna()
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(daily.index, daily["Turbidity (FNU)"], color="#9b5de5", linewidth=1.3, label="Turbidity (FNU)")
        ax1.set_ylabel("Turbidity (FNU)", color="#9b5de5")
        ax1.tick_params(axis="y", labelcolor="#9b5de5")

        ax2 = ax1.twinx()
        ax2.plot(daily.index, daily["Salinity (PPT)"], color="#e76f51", linewidth=1.3, label="Salinity (PPT)")
        ax2.set_ylabel("Salinity (PPT)", color="#e76f51")
        ax2.tick_params(axis="y", labelcolor="#e76f51")

        ax1.set_title(f"Turbidity vs Salinity -- {label}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"12_turbidity_vs_salinity_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 12_turbidity_vs_salinity_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. CAUSEWAY COMPARISON - TURBIDITY BOXPLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_causeway_turbidity(data: dict[str, pd.DataFrame]):
    """Boxplot comparing turbidity north vs south of JFK causeway."""
    records = []
    for label, df in data.items():
        side = "South" if "South" in label else "North" if "North" in label else "BBC"
        vals = df["Turbidity (FNU)"].dropna()
        clip = vals.quantile(0.95)
        for v in vals[vals <= clip]:
            records.append({"Platform": label.split(" - ")[0], "Side": side, "Turbidity (FNU)": v})
    box_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=box_df, x="Platform", y="Turbidity (FNU)", hue="Side",
                palette={"BBC": "#e63946", "North": "#457b9d", "South": "#e9c46a"},
                fliersize=0.5, ax=ax)
    ax.axhline(100, ls="--", color="red", alpha=0.5, label="High (100 FNU)")
    ax.set_title("Turbidity: North vs South of JFK Causeway")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "13_causeway_turbidity_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  Saved 13_causeway_turbidity_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. WEEKLY HIGH-TURBIDITY COUNT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_weekly_high_turbidity(data: dict[str, pd.DataFrame]):
    """Count of readings with turbidity > 100 FNU per week, per platform."""
    fig, ax = plt.subplots(figsize=(14, 5))
    width = pd.Timedelta(days=1.5)
    offsets = {k: pd.Timedelta(days=d) for k, d in
               zip(PLATFORM_FILES.keys(), [-2, -0.7, 0.7, 2])}

    for label, df in data.items():
        df_tmp = df[["datetime", "Turbidity (FNU)"]].dropna().copy()
        df_tmp["week"] = df_tmp["datetime"].dt.to_period("W").apply(lambda p: p.start_time)
        df_tmp["high"] = df_tmp["Turbidity (FNU)"] > 100
        weekly = df_tmp.groupby("week")["high"].sum()
        if weekly.sum() > 0:
            ax.bar(weekly.index + offsets[label], weekly.values,
                   width=width, label=label, color=COLORS[label], alpha=0.8)

    ax.set_ylabel("Readings with Turbidity > 100 FNU")
    ax.set_title("Weekly High-Turbidity Readings by Platform")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "14_weekly_high_turbidity.png"), dpi=150)
    plt.close(fig)
    print("  Saved 14_weekly_high_turbidity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. MULTI-FACTOR PANEL (turbidity focus)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multifactor_panel(data: dict[str, pd.DataFrame]):
    """Stacked panels: Turbidity, DO, Salinity, Temperature for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[
            ["Turbidity (FNU)", "ODO (mg/L)", "Salinity (PPT)", "Temperature (C)"]
        ].resample("D").mean().dropna()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        params = [
            ("Turbidity (FNU)", "#9b5de5"),
            ("ODO (mg/L)", "#2a9d8f"),
            ("Salinity (PPT)", "#e76f51"),
            ("Temperature (C)", "#264653"),
        ]
        for ax, (col, color) in zip(axes, params):
            ax.plot(daily.index, daily[col], color=color, linewidth=1.2)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(alpha=0.2)
            if col == "ODO (mg/L)":
                ax.axhline(2, ls="--", color="red", alpha=0.5, lw=0.8)
            if col == "Turbidity (FNU)":
                ax.axhline(100, ls="--", color="red", alpha=0.3, lw=0.8)
        axes[0].set_title(f"Multi-Factor Panel (Turbidity Focus) -- {label}", fontsize=12)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"15_multifactor_turbidity_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 15_multifactor_turbidity_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  TURBIDITY VISUALIZATIONS")
    print("=" * 60)

    print("\nLoading platform data ...")
    data = load_platforms()

    print("\nGenerating plots ...")
    plot_turbidity_timeseries(data)
    plot_turbidity_timeseries_log(data)
    plot_turbidity_range(data)
    plot_turbidity_violins(data)
    plot_turbidity_vs_do(data)
    plot_turbidity_vs_salinity(data)
    plot_turbidity_vs_depth(data)
    plot_diurnal_turbidity(data)
    plot_turbidity_heatmaps(data)
    plot_high_turbidity_events(data)
    plot_turbidity_do_dual(data)
    plot_turbidity_salinity_dual(data)
    plot_causeway_turbidity(data)
    plot_weekly_high_turbidity(data)
    plot_multifactor_panel(data)

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
