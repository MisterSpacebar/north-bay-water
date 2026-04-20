"""diurnal_do_v2.py
------------------
V2 diurnal DO (mg/L) plot - 2 x 3 grid (6 panels).
L3 and L4 are merged into a single panel using solid vs dashed lines.

Layout
------
  L0 FIU Bay      |  L1 Bisc. Canal   |  L2 Bisc. Bay
  L3+L4 Little R. |  L5 NBV North     |  L6 NBV South

Lines  = months (tab10 colour, same scheme as original)
Shading = +/-1 SD around hourly mean  (lighter for L4)

Output: diurnal_output/plots/odo_mg_l_v2.png
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

warnings.filterwarnings("ignore")

# -- paths ---------------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "cleaned_merged.csv")
PLOTS_DIR = os.path.join(HERE, "diurnal_output", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE   = "odo_mg_l"
Y_LABEL   = "DO (mg/L)"

# -- load data -----------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df["datetime_5min"] = pd.to_datetime(df["datetime_5min"], format="mixed")
df["hour"]  = df["datetime_5min"].dt.hour
df["month"] = df["datetime_5min"].dt.to_period("M").astype(str)

months_sorted = sorted(df["month"].unique())
MONTH_LABELS  = {m: pd.Period(m, "M").strftime("%b %Y") for m in months_sorted}

cmap   = cm.get_cmap("tab10", len(months_sorted))
COLORS = {m: cmap(i) for i, m in enumerate(months_sorted)}
HOURS  = np.arange(24)

C_BG   = "#f8f9fa"
C_DARK = "#2c2c2c"


def diurnal_stats(col: str) -> pd.DataFrame:
    tmp = df[["hour", "month"]].copy()
    tmp["value"] = pd.to_numeric(df[col], errors="coerce")
    grp  = tmp.groupby(["month", "hour"])["value"]
    stat = grp.agg(mean="mean", std="std").reset_index()
    stat["std"] = stat["std"].fillna(0)
    return stat


def plot_station(ax, col, months_sorted, alpha_fill=0.15, ls="-", lw=1.8, label_suffix=""):
    if col not in df.columns or df[col].notna().mean() < 0.05:
        return False
    stats = diurnal_stats(col)
    for month in months_sorted:
        sub   = stats[stats["month"] == month].set_index("hour").reindex(HOURS)
        mean  = sub["mean"].values
        std   = sub["std"].values
        color = COLORS[month]
        lbl   = MONTH_LABELS[month] + label_suffix
        ax.plot(HOURS, mean, color=color, linewidth=lw, linestyle=ls, label=lbl)
        ax.fill_between(HOURS, mean - std, mean + std,
                        color=color, alpha=alpha_fill)
    return True


# -- figure ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(24, 11), sharey=True)
fig.patch.set_facecolor(C_BG)

PANELS = [
    # (ax row, ax col, station(s), title, merged?)
    (0, 0, ["l0"], "L0  -  FIU Bay",         False),
    (0, 1, ["l1"], "L1  -  Bisc. Canal",     False),
    (0, 2, ["l2"], "L2  -  Bisc. Bay",       False),
    (1, 0, ["l3", "l4"], "L3 & L4  -  Little River\n(solid = L3 . dashed = L4)", True),
    (1, 1, ["l5"], "L5  -  NBV North",       False),
    (1, 2, ["l6"], "L6  -  NBV South",       False),
]

for r, c, sts, title, merged in PANELS:
    ax = axes[r][c]
    ax.set_facecolor(C_BG)

    if not merged:
        col = f"{FEATURE}_{sts[0]}"
        plot_station(ax, col, months_sorted)
        # legend on first panel only
        if r == 0 and c == 0:
            handles = [
                mlines.Line2D([], [], color=COLORS[m], linewidth=1.8,
                              label=MONTH_LABELS[m])
                for m in months_sorted
            ]
            ax.legend(handles=handles, fontsize=7.5, title="Month",
                      title_fontsize=8, loc="best")
    else:
        # Merged L3+L4 panel - solid for L3, dashed for L4
        col3 = f"{FEATURE}_l3"
        col4 = f"{FEATURE}_l4"
        have3 = plot_station(ax, col3, months_sorted, alpha_fill=0.13, ls="-",  lw=2.0)
        have4 = plot_station(ax, col4, months_sorted, alpha_fill=0.08, ls="--", lw=1.5)

        # Build a two-section legend: month colours + line-style guide
        month_handles = [
            mlines.Line2D([], [], color=COLORS[m], linewidth=1.8,
                          label=MONTH_LABELS[m])
            for m in months_sorted
        ]
        style_handles = [
            mlines.Line2D([], [], color="black", linewidth=2.0, linestyle="-",
                          label="L3 (solid)"),
            mlines.Line2D([], [], color="black", linewidth=1.5, linestyle="--",
                          label="L4 (dashed)"),
        ]
        leg1 = ax.legend(handles=month_handles, fontsize=7,
                         title="Month", title_fontsize=7.5,
                         loc="upper left", framealpha=0.85)
        ax.add_artist(leg1)
        ax.legend(handles=style_handles, fontsize=7.5,
                  loc="lower right", framealpha=0.85)

    ax.set_title(title, fontsize=9.5, fontweight="bold", color=C_DARK, pad=5)
    ax.set_xlabel("Hour of Day (local)", fontsize=8)
    ax.set_ylabel(Y_LABEL if c == 0 else "", fontsize=8)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=7)
    ax.set_ylim(0, 10)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
    ax.axhline(2.0, color="#c62828", linewidth=0.9, linestyle=":",
               alpha=0.7, zorder=1, label="_nolegend_")  # hypoxia threshold
    # small in-axes label anchored to top-left of axes coords
    ax.text(0.01, 0.03, "-- 2 mg/L hypoxia",
            ha="left", va="bottom", fontsize=6,
            color="#c62828", style="italic",
            transform=ax.transAxes)

fig.suptitle(
    "Diurnal Pattern - DO (mg/L)\n(mean +/- 1 SD by month)  v2 - L3 & L4 merged",
    fontsize=13, fontweight="bold", color=C_DARK,
)
fig.tight_layout()

out = os.path.join(PLOTS_DIR, "odo_mg_l_v2.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
