"""
Dissolved Oxygen & Ecological Factor Visualizations
=====================================================
Generates targeted plots for investigating seagrass/fish die-off factors
across platforms L0 (FIU BBC), L2 & L5 (north of JFK causeway), L6 (south).

Outputs saved to unsupervised/unsupervised_output/do_visualizations/
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
OUTPUT_DIR = os.path.join(BASE_DIR, "unsupervised_output", "do_visualizations")
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
# 1. DO TIME SERIES - ALL PLATFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_do_timeseries(data: dict[str, pd.DataFrame]):
    """Daily mean DO (mg/L) across all four platforms."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        daily = df.set_index("datetime")["ODO (mg/L)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, label=label, color=COLORS[label], linewidth=1.2)
    ax.axhline(2, ls="--", color="red", alpha=0.7, label="Hypoxic threshold (2 mg/L)")
    ax.axhline(4, ls=":", color="orange", alpha=0.6, label="Stress threshold (4 mg/L)")
    ax.set_ylabel("Dissolved Oxygen (mg/L)")
    ax.set_title("Daily Mean Dissolved Oxygen by Platform")
    ax.legend(loc="lower left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_do_timeseries_daily.png"), dpi=150)
    plt.close(fig)
    print("  Saved 01_do_timeseries_daily.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DO % SATURATION TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_do_sat_timeseries(data: dict[str, pd.DataFrame]):
    """Daily mean DO percent saturation across platforms."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        daily = df.set_index("datetime")["ODO (%Sat)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, label=label, color=COLORS[label], linewidth=1.2)
    ax.axhline(30, ls="--", color="red", alpha=0.7, label="Severe hypoxia (~30%)")
    ax.set_ylabel("DO Saturation (%)")
    ax.set_title("Daily Mean Dissolved Oxygen Saturation by Platform")
    ax.legend(loc="lower left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_do_saturation_timeseries.png"), dpi=150)
    plt.close(fig)
    print("  Saved 02_do_saturation_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SALINITY vs DO SCATTER (per platform)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_vs_do(data: dict[str, pd.DataFrame]):
    """Scatter: salinity vs DO coloured by platform."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=True)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Salinity (PPT)", "ODO (mg/L)"]].dropna()
        ax.scatter(sub["Salinity (PPT)"], sub["ODO (mg/L)"],
                   s=1, alpha=0.15, color=COLORS[label])
        ax.axhline(2, ls="--", color="red", alpha=0.6, lw=0.8)
        ax.set_xlabel("Salinity (PPT)")
        ax.set_ylabel("DO (mg/L)")
        ax.set_title(label, fontsize=10)
        # add correlation text
        r = sub["Salinity (PPT)"].corr(sub["ODO (mg/L)"])
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Salinity vs Dissolved Oxygen", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "03_salinity_vs_do.png"), dpi=150)
    plt.close(fig)
    print("  Saved 03_salinity_vs_do.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TURBIDITY vs DO SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_vs_do(data: dict[str, pd.DataFrame]):
    """Scatter: turbidity vs DO coloured by platform."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=True)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Turbidity (FNU)", "ODO (mg/L)"]].dropna()
        # clip turbidity for vis (99th pctile)
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
    fig.savefig(os.path.join(OUTPUT_DIR, "04_turbidity_vs_do.png"), dpi=150)
    plt.close(fig)
    print("  Saved 04_turbidity_vs_do.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TEMPERATURE vs DO SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_temperature_vs_do(data: dict[str, pd.DataFrame]):
    """Scatter: temperature vs DO coloured by platform."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=True)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Temperature (C)", "ODO (mg/L)"]].dropna()
        ax.scatter(sub["Temperature (C)"], sub["ODO (mg/L)"],
                   s=1, alpha=0.15, color=COLORS[label])
        ax.axhline(2, ls="--", color="red", alpha=0.6, lw=0.8)
        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("DO (mg/L)")
        ax.set_title(label, fontsize=10)
        r = sub["Temperature (C)"].corr(sub["ODO (mg/L)"])
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Temperature vs Dissolved Oxygen", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "05_temperature_vs_do.png"), dpi=150)
    plt.close(fig)
    print("  Saved 05_temperature_vs_do.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DIURNAL DO CYCLE (hourly avg)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_diurnal_do(data: dict[str, pd.DataFrame]):
    """Average DO by hour of day to reveal photosynthesis/respiration cycles."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in data.items():
        df_tmp = df[["datetime", "ODO (mg/L)"]].dropna().copy()
        df_tmp["hour"] = df_tmp["datetime"].dt.hour
        hourly = df_tmp.groupby("hour")["ODO (mg/L)"].mean()
        ax.plot(hourly.index, hourly.values, "o-", label=label,
                color=COLORS[label], linewidth=1.5, markersize=4)
    ax.axhline(2, ls="--", color="red", alpha=0.5, label="Hypoxic (2 mg/L)")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Mean DO (mg/L)")
    ax.set_title("Diurnal Dissolved Oxygen Cycle by Platform")
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_diurnal_do_cycle.png"), dpi=150)
    plt.close(fig)
    print("  Saved 06_diurnal_do_cycle.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HYPOXIC HOURS HEATMAP (day vs hour)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hypoxic_heatmaps(data: dict[str, pd.DataFrame]):
    """Heatmap of DO (mg/L) by date x hour for L0 and L2 (worst sites)."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        df_tmp = df[["datetime", "ODO (mg/L)"]].dropna().copy()
        df_tmp["date"] = df_tmp["datetime"].dt.date
        df_tmp["hour"] = df_tmp["datetime"].dt.hour
        pivot = df_tmp.pivot_table(index="date", columns="hour",
                                   values="ODO (mg/L)", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.06)))
        sns.heatmap(pivot, cmap="RdYlGn", vmin=0, vmax=10, ax=ax,
                    cbar_kws={"label": "DO (mg/L)"})
        ax.set_title(f"DO Heatmap (date x hour) -- {label}")
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Date")
        # thin out y-tick labels
        yticks = ax.get_yticks()
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        step = max(1, len(ylabels) // 20)
        for j, lbl in enumerate(ax.get_yticklabels()):
            if j % step != 0:
                lbl.set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"07_hypoxic_heatmap_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 07_hypoxic_heatmap_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SALINITY TIME SERIES (shows intrusion events)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_timeseries(data: dict[str, pd.DataFrame]):
    """Daily mean salinity to identify saltwater intrusion events."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        daily = df.set_index("datetime")["Salinity (PPT)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, label=label, color=COLORS[label], linewidth=1.2)
    ax.set_ylabel("Salinity (PPT)")
    ax.set_title("Daily Mean Salinity by Platform")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_salinity_timeseries.png"), dpi=150)
    plt.close(fig)
    print("  Saved 08_salinity_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TURBIDITY TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turbidity_timeseries(data: dict[str, pd.DataFrame]):
    """Daily mean turbidity by platform."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        daily = df.set_index("datetime")["Turbidity (FNU)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, label=label, color=COLORS[label], linewidth=1.2)
    ax.set_ylabel("Turbidity (FNU)")
    ax.set_title("Daily Mean Turbidity by Platform")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_turbidity_timeseries.png"), dpi=150)
    plt.close(fig)
    print("  Saved 09_turbidity_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DO + SALINITY DUAL-AXIS (L0 focus)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_do_salinity_dual(data: dict[str, pd.DataFrame]):
    """Dual-axis plot of DO and Salinity for L0 and L2 to show inverse relationship."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[["ODO (mg/L)", "Salinity (PPT)"]].resample("D").mean().dropna()
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(daily.index, daily["ODO (mg/L)"], color="#2a9d8f", linewidth=1.3, label="DO (mg/L)")
        ax1.axhline(2, ls="--", color="red", alpha=0.5, lw=0.8)
        ax1.set_ylabel("DO (mg/L)", color="#2a9d8f")
        ax1.tick_params(axis="y", labelcolor="#2a9d8f")

        ax2 = ax1.twinx()
        ax2.plot(daily.index, daily["Salinity (PPT)"], color="#e76f51", linewidth=1.3, label="Salinity (PPT)")
        ax2.set_ylabel("Salinity (PPT)", color="#e76f51")
        ax2.tick_params(axis="y", labelcolor="#e76f51")

        ax1.set_title(f"DO vs Salinity -- {label}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"10_do_vs_salinity_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 10_do_vs_salinity_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. DO + TURBIDITY DUAL-AXIS (L0 focus)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_do_turbidity_dual(data: dict[str, pd.DataFrame]):
    """Dual-axis plot of DO and Turbidity for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[["ODO (mg/L)", "Turbidity (FNU)"]].resample("D").mean().dropna()
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(daily.index, daily["ODO (mg/L)"], color="#2a9d8f", linewidth=1.3, label="DO (mg/L)")
        ax1.axhline(2, ls="--", color="red", alpha=0.5, lw=0.8)
        ax1.set_ylabel("DO (mg/L)", color="#2a9d8f")
        ax1.tick_params(axis="y", labelcolor="#2a9d8f")

        ax2 = ax1.twinx()
        ax2.plot(daily.index, daily["Turbidity (FNU)"], color="#9b5de5", linewidth=1.3, label="Turbidity (FNU)")
        ax2.set_ylabel("Turbidity (FNU)", color="#9b5de5")
        ax2.tick_params(axis="y", labelcolor="#9b5de5")

        ax1.set_title(f"DO vs Turbidity -- {label}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"11_do_vs_turbidity_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 11_do_vs_turbidity_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. NORTH vs SOUTH OF CAUSEWAY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_causeway_comparison(data: dict[str, pd.DataFrame]):
    """Box plots comparing north (L2, L5) vs south (L6) of JFK causeway."""
    records = []
    for label, df in data.items():
        side = "South" if "South" in label else "North" if "North" in label else "BBC"
        for _, row in df[["ODO (mg/L)", "Salinity (PPT)", "Turbidity (FNU)"]].dropna().iterrows():
            records.append({
                "Platform": label.split(" - ")[0],
                "Side": side,
                "DO (mg/L)": row["ODO (mg/L)"],
                "Salinity (PPT)": row["Salinity (PPT)"],
                "Turbidity (FNU)": row["Turbidity (FNU)"],
            })
    box_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, ["DO (mg/L)", "Salinity (PPT)", "Turbidity (FNU)"]):
        # clip turbidity for readability
        plot_df = box_df.copy()
        if col == "Turbidity (FNU)":
            clip = plot_df[col].quantile(0.95)
            plot_df = plot_df[plot_df[col] <= clip]
        sns.boxplot(data=plot_df, x="Platform", y=col, hue="Side", ax=ax,
                    palette={"BBC": "#e63946", "North": "#457b9d", "South": "#e9c46a"},
                    fliersize=0.5)
        ax.set_title(col)
        if col == "DO (mg/L)":
            ax.axhline(2, ls="--", color="red", alpha=0.5, lw=0.8)

    fig.suptitle("Platform Comparison: North vs South of JFK Causeway", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, "12_causeway_comparison_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  Saved 12_causeway_comparison_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. WEEKLY HYPOXIC HOURS BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

def plot_weekly_hypoxic_hours(data: dict[str, pd.DataFrame]):
    """Count of readings with DO < 2 mg/L per week, per platform."""
    fig, ax = plt.subplots(figsize=(14, 5))
    width = pd.Timedelta(days=1.5)
    offsets = {k: pd.Timedelta(days=d) for k, d in
               zip(PLATFORM_FILES.keys(), [-2, -0.7, 0.7, 2])}

    for label, df in data.items():
        df_tmp = df[["datetime", "ODO (mg/L)"]].dropna().copy()
        df_tmp["week"] = df_tmp["datetime"].dt.to_period("W").apply(lambda p: p.start_time)
        df_tmp["hypoxic"] = df_tmp["ODO (mg/L)"] < 2
        weekly = df_tmp.groupby("week")["hypoxic"].sum()
        if weekly.sum() > 0:
            ax.bar(weekly.index + offsets[label], weekly.values,
                   width=width, label=label, color=COLORS[label], alpha=0.8)

    ax.set_ylabel("Hypoxic readings (DO < 2 mg/L)")
    ax.set_title("Weekly Hypoxic Readings by Platform")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "13_weekly_hypoxic_hours.png"), dpi=150)
    plt.close(fig)
    print("  Saved 13_weekly_hypoxic_hours.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. MULTI-FACTOR PANEL (L0 deep dive)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multifactor_panel(data: dict[str, pd.DataFrame]):
    """Stacked time-series panels for L0: DO, Salinity, Turbidity, Temperature."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[
            ["ODO (mg/L)", "Salinity (PPT)", "Turbidity (FNU)", "Temperature (C)"]
        ].resample("D").mean().dropna()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        params = [
            ("ODO (mg/L)", "#2a9d8f", 2),
            ("Salinity (PPT)", "#e76f51", None),
            ("Turbidity (FNU)", "#9b5de5", None),
            ("Temperature (C)", "#264653", None),
        ]
        for ax, (col, color, thresh) in zip(axes, params):
            ax.plot(daily.index, daily[col], color=color, linewidth=1.2)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(alpha=0.2)
            if thresh is not None:
                ax.axhline(thresh, ls="--", color="red", alpha=0.5, lw=0.8)
        axes[0].set_title(f"Multi-Factor Panel -- {label}", fontsize=12)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"14_multifactor_panel_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 14_multifactor_panel_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. DO DISTRIBUTION VIOLIN PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_do_violins(data: dict[str, pd.DataFrame]):
    """Violin plots of DO distribution per platform."""
    records = []
    for label, df in data.items():
        vals = df["ODO (mg/L)"].dropna()
        for v in vals:
            records.append({"Platform": label.split(" - ")[0], "DO (mg/L)": v})
    vdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=vdf, x="Platform", y="DO (mg/L)", inner="quartile",
                   palette=[COLORS[k] for k in PLATFORM_FILES.keys()], ax=ax)
    ax.axhline(2, ls="--", color="red", alpha=0.7, label="Hypoxic (2 mg/L)")
    ax.axhline(4, ls=":", color="orange", alpha=0.5, label="Stress (4 mg/L)")
    ax.set_title("Dissolved Oxygen Distribution by Platform")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "15_do_violin_plots.png"), dpi=150)
    plt.close(fig)
    print("  Saved 15_do_violin_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DO & ECOLOGICAL FACTOR VISUALIZATIONS")
    print("=" * 60)

    print("\nLoading platform data ...")
    data = load_platforms()

    print("\nGenerating plots ...")
    plot_do_timeseries(data)
    plot_do_sat_timeseries(data)
    plot_salinity_vs_do(data)
    plot_turbidity_vs_do(data)
    plot_temperature_vs_do(data)
    plot_diurnal_do(data)
    plot_hypoxic_heatmaps(data)
    plot_salinity_timeseries(data)
    plot_turbidity_timeseries(data)
    plot_do_salinity_dual(data)
    plot_do_turbidity_dual(data)
    plot_causeway_comparison(data)
    plot_weekly_hypoxic_hours(data)
    plot_multifactor_panel(data)
    plot_do_violins(data)

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
