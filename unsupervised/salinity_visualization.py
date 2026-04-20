"""
Salinity Visualizations
========================
Generates targeted plots for investigating salinity patterns and saltwater
intrusion events across platforms L0 (FIU BBC), L2 & L5 (north of JFK
causeway), L6 (south).

Outputs saved to unsupervised/unsupervised_output/salinity_visualizations/
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
OUTPUT_DIR = os.path.join(BASE_DIR, "unsupervised_output", "salinity_visualizations")
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
# 1. SALINITY TIME SERIES - ALL PLATFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_timeseries(data: dict[str, pd.DataFrame]):
    """Daily mean salinity across all four platforms."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        daily = df.set_index("datetime")["Salinity (PPT)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, label=label, color=COLORS[label], linewidth=1.2)
    ax.axhline(35, ls="--", color="grey", alpha=0.5, label="Typical seawater (~35 PPT)")
    ax.axhline(0.5, ls=":", color="blue", alpha=0.5, label="Freshwater threshold (0.5 PPT)")
    ax.set_ylabel("Salinity (PPT)")
    ax.set_title("Daily Mean Salinity by Platform")
    ax.legend(loc="lower left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_salinity_timeseries_daily.png"), dpi=150)
    plt.close(fig)
    print("  Saved 01_salinity_timeseries_daily.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SALINITY RANGE (MIN/MAX) PER DAY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_range(data: dict[str, pd.DataFrame]):
    """Daily min/max salinity range showing variability per platform."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        daily = df.set_index("datetime")["Salinity (PPT)"].resample("D").agg(["min", "max", "mean"]).dropna()
        ax.fill_between(daily.index, daily["min"], daily["max"],
                        alpha=0.25, color=COLORS[label], label="Min-Max range")
        ax.plot(daily.index, daily["mean"], color=COLORS[label], linewidth=1.2, label="Daily mean")
        ax.set_ylabel("Salinity (PPT)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.suptitle("Daily Salinity Range (Min-Max) by Platform", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "02_salinity_daily_range.png"), dpi=150)
    plt.close(fig)
    print("  Saved 02_salinity_daily_range.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SALINITY DISTRIBUTION VIOLIN PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_violins(data: dict[str, pd.DataFrame]):
    """Violin plots of salinity distribution per platform."""
    records = []
    for label, df in data.items():
        vals = df["Salinity (PPT)"].dropna()
        for v in vals:
            records.append({"Platform": label.split(" - ")[0], "Salinity (PPT)": v})
    vdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=vdf, x="Platform", y="Salinity (PPT)", inner="quartile",
                   palette=[COLORS[k] for k in PLATFORM_FILES.keys()], ax=ax)
    ax.axhline(35, ls="--", color="grey", alpha=0.5, label="Seawater (~35 PPT)")
    ax.set_title("Salinity Distribution by Platform")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_salinity_violin_plots.png"), dpi=150)
    plt.close(fig)
    print("  Saved 03_salinity_violin_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SALINITY vs CONDUCTANCE (validation)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_vs_conductance(data: dict[str, pd.DataFrame]):
    """Scatter: salinity vs specific conductance -- should be tightly correlated."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Salinity (PPT)", "Specific Conductance (uS/cm)"]].dropna()
        ax.scatter(sub["Specific Conductance (uS/cm)"], sub["Salinity (PPT)"],
                   s=1, alpha=0.15, color=COLORS[label])
        ax.set_xlabel("Specific Conductance (uS/cm)")
        ax.set_ylabel("Salinity (PPT)")
        ax.set_title(label, fontsize=10)
        r = sub["Specific Conductance (uS/cm)"].corr(sub["Salinity (PPT)"])
        ax.text(0.95, 0.05, f"r = {r:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Salinity vs Specific Conductance", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "04_salinity_vs_conductance.png"), dpi=150)
    plt.close(fig)
    print("  Saved 04_salinity_vs_conductance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SALINITY vs DO SCATTER (per platform)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_vs_do(data: dict[str, pd.DataFrame]):
    """Scatter: salinity vs DO coloured by platform -- key relationship."""
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
        r = sub["Salinity (PPT)"].corr(sub["ODO (mg/L)"])
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Salinity vs Dissolved Oxygen", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "05_salinity_vs_do.png"), dpi=150)
    plt.close(fig)
    print("  Saved 05_salinity_vs_do.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SALINITY vs TEMPERATURE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_vs_temperature(data: dict[str, pd.DataFrame]):
    """Scatter: salinity vs temperature -- shows water mass mixing."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.flatten()
    for i, (label, df) in enumerate(data.items()):
        ax = axes[i]
        sub = df[["Temperature (C)", "Salinity (PPT)"]].dropna()
        ax.scatter(sub["Temperature (C)"], sub["Salinity (PPT)"],
                   s=1, alpha=0.15, color=COLORS[label])
        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("Salinity (PPT)")
        ax.set_title(label, fontsize=10)
        r = sub["Temperature (C)"].corr(sub["Salinity (PPT)"])
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Temperature-Salinity Diagram (T-S Plot)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "06_temperature_salinity_diagram.png"), dpi=150)
    plt.close(fig)
    print("  Saved 06_temperature_salinity_diagram.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DIURNAL SALINITY CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_diurnal_salinity(data: dict[str, pd.DataFrame]):
    """Average salinity by hour of day -- reveals tidal influence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in data.items():
        df_tmp = df[["datetime", "Salinity (PPT)"]].dropna().copy()
        df_tmp["hour"] = df_tmp["datetime"].dt.hour
        hourly = df_tmp.groupby("hour")["Salinity (PPT)"].mean()
        ax.plot(hourly.index, hourly.values, "o-", label=label,
                color=COLORS[label], linewidth=1.5, markersize=4)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Mean Salinity (PPT)")
    ax.set_title("Diurnal Salinity Cycle by Platform")
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_diurnal_salinity_cycle.png"), dpi=150)
    plt.close(fig)
    print("  Saved 07_diurnal_salinity_cycle.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SALINITY HEATMAP (date x hour) - L0 and L2
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_heatmaps(data: dict[str, pd.DataFrame]):
    """Heatmap of salinity by date x hour for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        df_tmp = df[["datetime", "Salinity (PPT)"]].dropna().copy()
        df_tmp["date"] = df_tmp["datetime"].dt.date
        df_tmp["hour"] = df_tmp["datetime"].dt.hour
        pivot = df_tmp.pivot_table(index="date", columns="hour",
                                   values="Salinity (PPT)", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.06)))
        sns.heatmap(pivot, cmap="YlGnBu", vmin=0, vmax=40, ax=ax,
                    cbar_kws={"label": "Salinity (PPT)"})
        ax.set_title(f"Salinity Heatmap (date x hour) -- {label}")
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Date")
        step = max(1, len(pivot) // 20)
        for j, lbl in enumerate(ax.get_yticklabels()):
            if j % step != 0:
                lbl.set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"08_salinity_heatmap_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 08_salinity_heatmap_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FRESHWATER INTRUSION EVENTS (salinity drops)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_freshwater_events(data: dict[str, pd.DataFrame]):
    """Highlight periods where salinity drops below 5 PPT (freshwater pulse)."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    threshold = 5
    for ax, (label, df) in zip(axes, data.items()):
        daily = df.set_index("datetime")["Salinity (PPT)"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, color=COLORS[label], linewidth=1.2)
        # shade freshwater events
        below = daily[daily < threshold]
        if len(below) > 0:
            ax.fill_between(daily.index, 0, daily.values,
                            where=daily.values < threshold,
                            alpha=0.3, color="blue", label=f"< {threshold} PPT")
        ax.axhline(threshold, ls="--", color="blue", alpha=0.4, lw=0.8)
        ax.set_ylabel("Salinity (PPT)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.suptitle(f"Freshwater Intrusion Events (salinity < {threshold} PPT)", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTPUT_DIR, "09_freshwater_intrusion_events.png"), dpi=150)
    plt.close(fig)
    print("  Saved 09_freshwater_intrusion_events.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SALINITY + DO DUAL-AXIS (L0, L2)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_do_dual(data: dict[str, pd.DataFrame]):
    """Dual-axis salinity and DO for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[["Salinity (PPT)", "ODO (mg/L)"]].resample("D").mean().dropna()
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(daily.index, daily["Salinity (PPT)"], color="#e76f51", linewidth=1.3, label="Salinity (PPT)")
        ax1.set_ylabel("Salinity (PPT)", color="#e76f51")
        ax1.tick_params(axis="y", labelcolor="#e76f51")

        ax2 = ax1.twinx()
        ax2.plot(daily.index, daily["ODO (mg/L)"], color="#2a9d8f", linewidth=1.3, label="DO (mg/L)")
        ax2.axhline(2, ls="--", color="red", alpha=0.5, lw=0.8)
        ax2.set_ylabel("DO (mg/L)", color="#2a9d8f")
        ax2.tick_params(axis="y", labelcolor="#2a9d8f")

        ax1.set_title(f"Salinity vs DO -- {label}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"10_salinity_vs_do_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 10_salinity_vs_do_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SALINITY + TURBIDITY DUAL-AXIS (L0, L2)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_turbidity_dual(data: dict[str, pd.DataFrame]):
    """Dual-axis salinity and turbidity -- do runoff events bring sediment?"""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[["Salinity (PPT)", "Turbidity (FNU)"]].resample("D").mean().dropna()
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(daily.index, daily["Salinity (PPT)"], color="#e76f51", linewidth=1.3, label="Salinity (PPT)")
        ax1.set_ylabel("Salinity (PPT)", color="#e76f51")
        ax1.tick_params(axis="y", labelcolor="#e76f51")

        ax2 = ax1.twinx()
        ax2.plot(daily.index, daily["Turbidity (FNU)"], color="#9b5de5", linewidth=1.3, label="Turbidity (FNU)")
        ax2.set_ylabel("Turbidity (FNU)", color="#9b5de5")
        ax2.tick_params(axis="y", labelcolor="#9b5de5")

        ax1.set_title(f"Salinity vs Turbidity -- {label}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"11_salinity_vs_turbidity_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 11_salinity_vs_turbidity_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CAUSEWAY COMPARISON - SALINITY BOXPLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_causeway_salinity(data: dict[str, pd.DataFrame]):
    """Boxplot comparing salinity north vs south of JFK causeway."""
    records = []
    for label, df in data.items():
        side = "South" if "South" in label else "North" if "North" in label else "BBC"
        for v in df["Salinity (PPT)"].dropna():
            records.append({"Platform": label.split(" - ")[0], "Side": side, "Salinity (PPT)": v})
    box_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=box_df, x="Platform", y="Salinity (PPT)", hue="Side",
                palette={"BBC": "#e63946", "North": "#457b9d", "South": "#e9c46a"},
                fliersize=0.5, ax=ax)
    ax.axhline(35, ls="--", color="grey", alpha=0.5, label="Seawater")
    ax.set_title("Salinity: North vs South of JFK Causeway")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "12_causeway_salinity_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  Saved 12_causeway_salinity_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. WEEKLY SALINITY VARIABILITY (std dev)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_weekly_salinity_variability(data: dict[str, pd.DataFrame]):
    """Weekly standard deviation of salinity -- high variability = mixing events."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        df_tmp = df[["datetime", "Salinity (PPT)"]].dropna().copy()
        df_tmp["week"] = df_tmp["datetime"].dt.to_period("W").apply(lambda p: p.start_time)
        weekly_std = df_tmp.groupby("week")["Salinity (PPT)"].std().dropna()
        ax.plot(weekly_std.index, weekly_std.values, "o-", label=label,
                color=COLORS[label], linewidth=1.2, markersize=4)
    ax.set_ylabel("Salinity Std Dev (PPT)")
    ax.set_title("Weekly Salinity Variability (Std Dev) -- Higher = More Mixing Events")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "13_weekly_salinity_variability.png"), dpi=150)
    plt.close(fig)
    print("  Saved 13_weekly_salinity_variability.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. MULTI-FACTOR PANEL (salinity focus)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multifactor_panel(data: dict[str, pd.DataFrame]):
    """Stacked panels: Salinity, Conductance, DO, Temperature for L0 and L2."""
    for label in ["L0 - FIU BBC", "L2 - North of Causeway"]:
        df = data[label]
        daily = df.set_index("datetime")[
            ["Salinity (PPT)", "Specific Conductance (uS/cm)", "ODO (mg/L)", "Temperature (C)"]
        ].resample("D").mean().dropna()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        params = [
            ("Salinity (PPT)", "#e76f51"),
            ("Specific Conductance (uS/cm)", "#264653"),
            ("ODO (mg/L)", "#2a9d8f"),
            ("Temperature (C)", "#9b5de5"),
        ]
        for ax, (col, color) in zip(axes, params):
            ax.plot(daily.index, daily[col], color=color, linewidth=1.2)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(alpha=0.2)
            if col == "ODO (mg/L)":
                ax.axhline(2, ls="--", color="red", alpha=0.5, lw=0.8)
        axes[0].set_title(f"Multi-Factor Panel (Salinity Focus) -- {label}", fontsize=12)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = label.split(" - ")[0].lower()
        fig.savefig(os.path.join(OUTPUT_DIR, f"14_multifactor_salinity_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved 14_multifactor_salinity_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. SALINITY HISTOGRAM (all platforms overlaid)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_salinity_histogram(data: dict[str, pd.DataFrame]):
    """Overlaid histograms of salinity distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in data.items():
        vals = df["Salinity (PPT)"].dropna()
        ax.hist(vals, bins=80, density=True, alpha=0.4, color=COLORS[label], label=label)
    ax.set_xlabel("Salinity (PPT)")
    ax.set_ylabel("Density")
    ax.set_title("Salinity Distribution (all platforms overlaid)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "15_salinity_histogram.png"), dpi=150)
    plt.close(fig)
    print("  Saved 15_salinity_histogram.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SALINITY VISUALIZATIONS")
    print("=" * 60)

    print("\nLoading platform data ...")
    data = load_platforms()

    print("\nGenerating plots ...")
    plot_salinity_timeseries(data)
    plot_salinity_range(data)
    plot_salinity_violins(data)
    plot_salinity_vs_conductance(data)
    plot_salinity_vs_do(data)
    plot_salinity_vs_temperature(data)
    plot_diurnal_salinity(data)
    plot_salinity_heatmaps(data)
    plot_freshwater_events(data)
    plot_salinity_do_dual(data)
    plot_salinity_turbidity_dual(data)
    plot_causeway_salinity(data)
    plot_weekly_salinity_variability(data)
    plot_multifactor_panel(data)
    plot_salinity_histogram(data)

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
