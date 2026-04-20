# ============================================================
# Random Forest Regression -- Predicting Lagoon Conditions
#   from Source Station Readings
# ============================================================
# The idea: use the water-quality readings at the source stations
# (canals/rivers) to PREDICT conditions at the lagoon stations.
#
# If the model is accurate and feature importance is high for a
# source feature, that source is meaningfully influencing the lagoon.
#
# Two subsystems:
#   NORTH of JFK Causeway:
#     Predictors: L0 (FIU Bay), L1 (Biscayne Canal)
#     Targets:    L2 (canal-bay junction), L5 (NBV north)
#
#   SOUTH of JFK Causeway:
#     Predictors: L3 (Little River A), L4 (Little River B)
#     Targets:    L6 (NBV south)
#
# We also include time-based features (hour of day, day of week)
# to let the model capture tidal/diurnal cycles.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- avoids tkinter threading errors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Step 1: Load data and parse timestamps
# ------------------------------------------------------------------
raw = pd.read_csv("data/merged_keep.csv")
raw["datetime_5min"] = pd.to_datetime(raw["datetime_5min"], format="mixed")
raw.sort_values("datetime_5min", inplace=True)
raw.reset_index(drop=True, inplace=True)
print(f"Loaded {len(raw)} rows\n")

# ------------------------------------------------------------------
# Step 2: Engineer time features
# ------------------------------------------------------------------
# Hour-of-day captures diurnal patterns (sun heating, land-sea breeze).
# We encode it as sin/cos so the model knows 23:00 and 00:00 are close.
# Day-of-week can capture weekday vs weekend boat traffic differences.
raw["hour"] = raw["datetime_5min"].dt.hour + raw["datetime_5min"].dt.minute / 60
raw["hour_sin"] = np.sin(2 * np.pi * raw["hour"] / 24)
raw["hour_cos"] = np.cos(2 * np.pi * raw["hour"] / 24)
raw["dayofweek"] = raw["datetime_5min"].dt.dayofweek  # 0=Mon ... 6=Sun

TIME_FEATURES = ["hour_sin", "hour_cos", "dayofweek"]

# ------------------------------------------------------------------
# Step 3: Define the features at each station
# ------------------------------------------------------------------
WATER_FEATURES = ["temperature_c", "specific_conductance_us_cm",
                  "salinity_ppt", "odo_sat", "turbidity_fnu"]

FEATURE_LABELS = {
    "temperature_c":              "Temperature",
    "specific_conductance_us_cm": "Conductance",
    "salinity_ppt":               "Salinity",
    "odo_sat":                    "DO Sat",
    "turbidity_fnu":              "Turbidity",
}

STATION_LABELS = {
    "l0": "L0 - FIU Bay",
    "l1": "L1 - Bisc. Canal",
    "l2": "L2 - Canal-Bay jct",
    "l3": "L3 - Little River A",
    "l4": "L4 - Little River B",
    "l5": "L5 - NBV North",
    "l6": "L6 - NBV South",
}

# ------------------------------------------------------------------
# Step 4: Define the two subsystems
# ------------------------------------------------------------------
NORTH = {
    "name": "North of JFK Causeway",
    "sources": ["l0", "l1"],
    "targets": ["l2", "l5"],
}
SOUTH = {
    "name": "South of JFK Causeway",
    "sources": ["l3", "l4"],
    "targets": ["l6"],
}

# ------------------------------------------------------------------
# Step 5: Apply non-linear transforms
# ------------------------------------------------------------------
# As we discussed, turbidity and conductance are right-skewed.
# Log-transforming them compresses outlier spikes and helps the
# model focus on the meaningful range.
for station in ["l0", "l1", "l2", "l3", "l4", "l5", "l6"]:
    turb_col = f"turbidity_fnu_{station}"
    cond_col = f"specific_conductance_us_cm_{station}"
    if turb_col in raw.columns:
        # log1p = log(1 + x), handles zeros safely
        raw[f"log_turbidity_{station}"] = np.log1p(raw[turb_col].clip(lower=0))
    if cond_col in raw.columns:
        raw[f"log_conductance_{station}"] = np.log1p(raw[cond_col].clip(lower=0))


# ==================================================================
# Step 6: Run Random Forest for each (subsystem, target, feature)
# ==================================================================
# For each target station and each water-quality feature, we build a
# separate Random Forest model:
#
#   Inputs  = all source-station features + log transforms + time
#   Output  = one target feature (e.g. salinity at L2)
#
# This tells us: "Can we predict L2's salinity from L0 and L1's
# readings?"  If yes (high R^2) -> the source stations drive it.
# Feature importance shows WHICH source variable matters most.
# ==================================================================

all_results = []  # collect results for summary table

for subsystem in [NORTH, SOUTH]:
    print("=" * 65)
    print(f"  {subsystem['name']}")
    print(f"  Sources: {', '.join(STATION_LABELS[s] for s in subsystem['sources'])}")
    print(f"  Targets: {', '.join(STATION_LABELS[s] for s in subsystem['targets'])}")
    print("=" * 65)

    # Build the predictor column list:
    # original features + log transforms for each source station + time
    predictor_cols = []
    predictor_labels = []
    for src in subsystem["sources"]:
        for feat in WATER_FEATURES:
            col = f"{feat}_{src}"
            if col in raw.columns:
                predictor_cols.append(col)
                predictor_labels.append(
                    f"{STATION_LABELS[src].split('-')[0].strip()} {FEATURE_LABELS[feat]}")
        # add log-transformed columns
        for prefix in ["log_turbidity", "log_conductance"]:
            col = f"{prefix}_{src}"
            if col in raw.columns:
                predictor_cols.append(col)
                short = prefix.replace("log_", "log ")
                predictor_labels.append(
                    f"{STATION_LABELS[src].split('-')[0].strip()} {short}")

    predictor_cols += TIME_FEATURES
    predictor_labels += ["Hour (sin)", "Hour (cos)", "Day of week"]

    # --- Loop over each target station and target feature ---
    for tgt in subsystem["targets"]:
        # Set up a figure: one subplot per target feature
        # Row 0 = actual vs predicted scatter
        # Row 1 = residual distribution
        # Row 2 = feature importance
        fig, axes = plt.subplots(3, len(WATER_FEATURES),
                                 figsize=(4.5 * len(WATER_FEATURES), 12),
                                 gridspec_kw={"height_ratios": [1, 0.8, 1.3]})

        for col_idx, target_feat in enumerate(WATER_FEATURES):
            target_col = f"{target_feat}_{tgt}"
            if target_col not in raw.columns:
                axes[0, col_idx].text(0.5, 0.5, "no data", ha="center",
                                      va="center", transform=axes[0, col_idx].transAxes)
                axes[1, col_idx].set_visible(False)
                axes[2, col_idx].set_visible(False)
                continue

            # Assemble X, y and drop rows with any NaN
            cols_needed = predictor_cols + [target_col]
            subset = raw[cols_needed].dropna()

            if len(subset) < 200:
                axes[0, col_idx].text(0.5, 0.5, f"n={len(subset)}\ntoo few",
                                      ha="center", va="center",
                                      transform=axes[0, col_idx].transAxes)
                axes[1, col_idx].set_visible(False)
                axes[2, col_idx].set_visible(False)
                continue

            X = subset[predictor_cols].values
            y = subset[target_col].values

            # Train/test split (80/20, chronological would be ideal but
            # random is fine for a first look)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # --- Build the Random Forest ---
            # n_estimators=200: build 200 decision trees and average them.
            #   More trees = more stable predictions, diminishing returns past ~200.
            # max_depth=15: limit tree depth to prevent overfitting (memorising noise).
            # min_samples_leaf=10: each leaf must have >=10 samples -- another
            #   guard against overfitting on sparse regions.
            # n_jobs=-1: use all CPU cores for speed.
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            # --- Metrics ---
            residuals = y_test - y_pred

            mse    = mean_squared_error(y_test, y_pred)
            rmse   = mse ** 0.5
            mae    = mean_absolute_error(y_test, y_pred)
            med_ae = median_absolute_error(y_test, y_pred)
            r2     = r2_score(y_test, y_pred)
            ev     = explained_variance_score(y_test, y_pred)
            nrmse  = rmse / (y_test.max() - y_test.min()) if (y_test.max() - y_test.min()) > 0 else np.nan

            # MAPE -- mean absolute percentage error
            # Skipped if any actual value is zero (division by zero)
            if np.all(y_test != 0):
                mape = mean_absolute_percentage_error(y_test, y_pred)
            else:
                mape = np.nan

            # Max error -- the single worst prediction
            max_err = np.max(np.abs(residuals))

            print(f"\n  {STATION_LABELS[tgt]} -- {FEATURE_LABELS[target_feat]}")
            print(f"    Samples            : {len(subset)}  (train {len(X_train)}, test {len(X_test)})")
            print(f"    R^2                 : {r2:.3f}")
            print(f"    Explained Variance : {ev:.3f}")
            print(f"    MSE                : {mse:.3f}")
            print(f"    RMSE               : {rmse:.3f}")
            print(f"    NRMSE              : {nrmse:.2%}")
            print(f"    MAE                : {mae:.3f}")
            print(f"    Median AE          : {med_ae:.3f}")
            print(f"    MAPE               : {mape:.2%}" if not np.isnan(mape) else "    MAPE               : N/A (zeros in actuals)")
            print(f"    Max Error          : {max_err:.3f}")

            all_results.append({
                "subsystem": subsystem["name"],
                "target_station": STATION_LABELS[tgt],
                "target_feature": FEATURE_LABELS[target_feat],
                "r2": r2, "explained_var": ev,
                "rmse": rmse, "nrmse": nrmse,
                "mae": mae, "median_ae": med_ae,
                "mape": mape, "max_error": max_err,
                "n_samples": len(subset),
            })

            # --- Top subplot: Actual vs Predicted scatter ---
            ax_scatter = axes[0, col_idx]
            ax_scatter.scatter(y_test, y_pred, alpha=0.15, s=8, edgecolors="none")

            # Perfect-prediction line
            lo = min(y_test.min(), y_pred.min())
            hi = max(y_test.max(), y_pred.max())
            ax_scatter.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")

            # Best-fit line
            slope, intercept = np.polyfit(y_test, y_pred, 1)
            ax_scatter.plot([lo, hi],
                            [slope * lo + intercept, slope * hi + intercept],
                            "g-", linewidth=1, label=f"fit (slope={slope:.2f})")

            ax_scatter.set_xlabel("Actual")
            ax_scatter.set_ylabel("Predicted")
            ax_scatter.set_title(
                f"{FEATURE_LABELS[target_feat]}\n"
                f"R^2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}"
                + (f"  MAPE={mape:.1%}" if not np.isnan(mape) else ""),
                fontsize=9,
            )
            ax_scatter.legend(fontsize=7)

            # --- Middle subplot: Residual distribution ---
            # Residuals = actual − predicted.  A good model has residuals
            # centred on zero with a tight, symmetric bell shape.
            # Skew or long tails reveal systematic bias or outlier problems.
            ax_resid = axes[1, col_idx]
            ax_resid.hist(residuals, bins=50, color="slategray", edgecolor="white", alpha=0.8)
            ax_resid.axvline(0, color="red", linestyle="--", linewidth=1)
            ax_resid.axvline(np.mean(residuals), color="orange", linestyle="-", linewidth=1,
                             label=f"mean={np.mean(residuals):.2f}")
            ax_resid.set_xlabel("Residual (actual − predicted)", fontsize=8)
            ax_resid.set_ylabel("Count", fontsize=8)
            ax_resid.set_title(f"Residuals -- MedAE={med_ae:.2f}", fontsize=9)
            ax_resid.legend(fontsize=7)

            # --- Bottom subplot: Feature importance (top 10) ---
            importances = rf.feature_importances_
            indices = np.argsort(importances)[-10:]  # top 10

            ax_imp = axes[2, col_idx]
            ax_imp.barh(
                [predictor_labels[i] for i in indices],
                importances[indices],
                color="steelblue",
            )
            ax_imp.set_xlabel("Importance", fontsize=8)
            ax_imp.tick_params(axis="y", labelsize=7)

        fig.suptitle(
            f"Random Forest Regression -- {STATION_LABELS[tgt]}\n"
            f"Predicting from {', '.join(STATION_LABELS[s] for s in subsystem['sources'])}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"rf_{tgt}_{subsystem['name'].lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  -> saved {fname}")

# ==================================================================
# Step 7: Summary table
# ==================================================================
print("\n" + "=" * 65)
print("  SUMMARY -- Random Forest Regression Results")
print("=" * 65)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False, float_format="{:.3f}".format))

# Heatmaps for multiple metrics
metric_configs = [
    ("r2",          "R^2",    "YlGn",   0, 1,    ".3f"),
    ("rmse",        "RMSE",  "YlOrRd", None, None, ".3f"),
    ("nrmse",       "NRMSE", "YlOrRd", 0, 0.5,  ".2%"),
    ("mae",         "MAE",   "YlOrRd", None, None, ".3f"),
    ("mape",        "MAPE",  "YlOrRd", 0, 1,    ".1%"),
    ("explained_var", "Explained Variance", "YlGn", 0, 1, ".3f"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()

for idx, (col, title, cmap, vmin, vmax, fmt) in enumerate(metric_configs):
    pivot = results_df.pivot_table(index="target_station", columns="target_feature", values=col)
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=0.5, ax=axes[idx])
    axes[idx].set_title(title, fontsize=11, fontweight="bold")
    axes[idx].tick_params(axis="both", labelsize=8)

fig.suptitle("Summary -- All Regression Metrics by Station & Feature",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("rf_metrics_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> saved rf_metrics_heatmaps.png")


# ==================================================================
# Step 8: FOCUSED Random Forest -- Salinity & DO Only (L5 and L6)
# ==================================================================
# Same model setup as above but limited to two target features
# so the graphs are larger and easier to read.
# ==================================================================

FOCUS_TARGETS = ["salinity_ppt", "odo_sat"]
FOCUS_LABELS  = {"salinity_ppt": "Salinity (ppt)", "odo_sat": "DO Saturation (%)"}

FOCUS_SYSTEMS = [
    {"name": "North of JFK Causeway", "sources": ["l0", "l1"], "target": "l5"},
    {"name": "South of JFK Causeway", "sources": ["l3", "l4"], "target": "l6"},
]

print("\n" + "=" * 65)
print("  FOCUSED RF -- Salinity & DO Only")
print("=" * 65)

for system in FOCUS_SYSTEMS:
    tgt = system["target"]

    # Build predictor columns (same as main loop)
    predictor_cols_f = []
    predictor_labels_f = []
    for src in system["sources"]:
        for feat in WATER_FEATURES:
            col = f"{feat}_{src}"
            if col in raw.columns:
                predictor_cols_f.append(col)
                predictor_labels_f.append(
                    f"{STATION_LABELS[src].split('-')[0].strip()} {FEATURE_LABELS[feat]}")
        for prefix in ["log_turbidity", "log_conductance"]:
            col = f"{prefix}_{src}"
            if col in raw.columns:
                predictor_cols_f.append(col)
                short = prefix.replace("log_", "log ")
                predictor_labels_f.append(
                    f"{STATION_LABELS[src].split('-')[0].strip()} {short}")

    predictor_cols_f += TIME_FEATURES
    predictor_labels_f += ["Hour (sin)", "Hour (cos)", "Day of week"]

    n_focus = len(FOCUS_TARGETS)
    fig, axes = plt.subplots(3, n_focus, figsize=(7 * n_focus, 14),
                             gridspec_kw={"height_ratios": [1, 0.8, 1.3]})

    for col_idx, target_feat in enumerate(FOCUS_TARGETS):
        target_col = f"{target_feat}_{tgt}"
        if target_col not in raw.columns:
            for r in range(3):
                axes[r, col_idx].text(0.5, 0.5, "no data", ha="center",
                                      va="center", transform=axes[r, col_idx].transAxes)
            continue

        cols_needed = predictor_cols_f + [target_col]
        subset = raw[cols_needed].dropna()

        if len(subset) < 200:
            for r in range(3):
                axes[r, col_idx].text(0.5, 0.5, f"n={len(subset)}\ntoo few",
                                      ha="center", va="center",
                                      transform=axes[r, col_idx].transAxes)
            continue

        X = subset[predictor_cols_f].values
        y = subset[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        residuals = y_test - y_pred
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae  = mean_absolute_error(y_test, y_pred)
        med_ae = median_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) if np.all(y_test != 0) else np.nan

        print(f"\n  {STATION_LABELS[tgt]} -- {FOCUS_LABELS[target_feat]}")
        print(f"    R^2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}  MedAE={med_ae:.2f}"
              + (f"  MAPE={mape:.1%}" if not np.isnan(mape) else ""))

        # --- Scatter ---
        ax = axes[0, col_idx]
        ax.scatter(y_test, y_pred, alpha=0.2, s=12, edgecolors="none",
                   color="#1f77b4" if target_feat == "salinity_ppt" else "#d62728")
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")
        slope, intercept = np.polyfit(y_test, y_pred, 1)
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                "g-", linewidth=1.2, label=f"fit (slope={slope:.2f})")
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(
            f"{FOCUS_LABELS[target_feat]}\n"
            f"R^2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}"
            + (f"  MAPE={mape:.1%}" if not np.isnan(mape) else ""),
            fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        # --- Residuals ---
        ax = axes[1, col_idx]
        ax.hist(residuals, bins=50, edgecolor="white", alpha=0.8,
                color="#1f77b4" if target_feat == "salinity_ppt" else "#d62728")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.axvline(np.mean(residuals), color="orange", linewidth=1.2,
                   label=f"mean={np.mean(residuals):.2f}")
        ax.set_xlabel("Residual (actual − predicted)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"Residuals -- MedAE={med_ae:.2f}", fontsize=10)
        ax.legend(fontsize=8)

        # --- Feature importance (top 10) ---
        ax = axes[2, col_idx]
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-10:]
        ax.barh([predictor_labels_f[i] for i in indices], importances[indices],
                color="#1f77b4" if target_feat == "salinity_ppt" else "#d62728")
        ax.set_xlabel("Importance", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        f"Random Forest -- {STATION_LABELS[tgt]} (Salinity & DO Focus)\n"
        f"Predicting from {', '.join(STATION_LABELS[s] for s in system['sources'])}",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"rf_focus_{tgt}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  -> saved {fname}")

print("\n✓ Done!")
