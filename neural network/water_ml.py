"""
Neural Network Analysis of Water Quality in Biscayne Bay Lagoon System
=======================================================================
Predicts dissolved oxygen (DO) levels -- the primary indicator of lagoon health --
from other water quality parameters. Also classifies conditions as:
  - Healthy:  DO >= 4 mg/L
  - Stressed: 2 <= DO < 4 mg/L
  - Hypoxic:  DO < 2 mg/L

Platform locations:
  L0 -- FIU BBC
  L1 -- Biscayne Canal
  L2 -- Biscayne Bay (near canal)
  L3 -- Little River (upstream)
  L4 -- Little River (downstream)
  L5 -- JFK Causeway / North Bay Village (north end)
  L6 -- JFK Causeway / North Bay Village (south end)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLATFORM_LABELS = {
    "L0": "FIU BBC",
    "L1": "Biscayne Canal",
    "L2": "Biscayne Bay (near canal)",
    "L3": "Little River (upstream)",
    "L4": "Little River (downstream)",
    "L5": "JFK Causeway (north)",
    "L6": "JFK Causeway (south)",
}

# Features used to predict DO
FEATURES = [
    "Temperature (C)",
    "Specific Conductance (uS/cm)",
    "Salinity (PPT)",
    "Depth (m)",
    "Turbidity (FNU)",
]
TARGET = "ODO (mg/L)"

RANDOM_STATE = 42


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_all_platforms() -> pd.DataFrame:
    """Load and merge all platform CSVs, adding a platform label column."""
    frames = []
    for i in range(7):
        tag = f"L{i}"
        path = os.path.join(RAW_DIR, f"raw-data-platform{tag}_parameters.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["platform"] = tag
        df["location"] = PLATFORM_LABELS[tag]
        frames.append(df)
        print(f"  Loaded {tag} ({PLATFORM_LABELS[tag]}): {len(df):,} rows")
    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows (raw): {len(merged):,}")
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only needed columns, drop NaN/invalid rows, remove outliers."""
    cols_needed = FEATURES + [TARGET, "platform", "location"]
    df = df[cols_needed].copy()

    # Coerce to numeric
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    # Remove physically implausible values (sensor errors produce extreme outliers)
    df = df[df[TARGET].between(0, 20)]                       # DO: 0-20 mg/L
    df = df[df["Temperature (C)"].between(10, 45)]            # Temp: realistic range
    df = df[df["Turbidity (FNU)"].between(0, 1000)]           # Turbidity: cap sensor noise
    df = df[df["Salinity (PPT)"].between(0, 45)]              # Salinity: ocean max ~35
    df = df[df["Specific Conductance (uS/cm)"].between(0, 60000)]  # Conductance cap
    df = df[df["Depth (m)"].between(0, 15)]                   # No negative depths

    print(f"  Rows after cleaning: {len(df):,}")
    return df.reset_index(drop=True)


# ── Labelling ────────────────────────────────────────────────────────────────
def add_health_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Classify water health based on DO concentration."""
    conditions = [
        df[TARGET] < 2,
        df[TARGET].between(2, 4, inclusive="left"),
    ]
    labels = ["Hypoxic (<2 mg/L)", "Stressed (2-4 mg/L)"]
    df["health"] = np.select(conditions, labels, default="Healthy (>=4 mg/L)")
    print("\n  Health distribution:")
    for label, count in df["health"].value_counts().items():
        print(f"    {label}: {count:,} ({100*count/len(df):.1f}%)")
    return df


# ── DO Regression ────────────────────────────────────────────────────────────
def train_do_regressor(df: pd.DataFrame):
    """Train a neural network to predict DO from water quality features."""
    print("\n" + "=" * 60)
    print("REGRESSION: Predicting Dissolved Oxygen (mg/L)")
    print("=" * 60)

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  MAE:  {mae:.3f} mg/L")
    print(f"  RMSE: {rmse:.3f} mg/L")
    print(f"  R^2:   {r2:.3f}")

    # ── Feature importance via permutation ────────────────────────────────
    print("\n  Feature importance (permutation-based):")
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        model, X_test_s, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance_order = perm.importances_mean.argsort()[::-1]
    for idx in importance_order:
        print(f"    {FEATURES[idx]:>40s}: {perm.importances_mean[idx]:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Predicted vs actual
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.05, s=2)
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual DO (mg/L)")
    ax.set_ylabel("Predicted DO (mg/L)")
    ax.set_title(f"DO Regression (R^2={r2:.3f})")

    # Residuals
    ax = axes[1]
    residuals = y_test - y_pred
    ax.hist(residuals, bins=80, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")

    # Training loss curve
    ax = axes[2]
    ax.plot(model.loss_curve_, label="Training loss")
    if hasattr(model, "validation_scores_"):
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, color="orange", label="Validation R^2")
        ax2.set_ylabel("Validation R^2", color="orange")
        ax2.legend(loc="center right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "do_regression_results.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Saved: output/do_regression_results.png")

    # ── Feature importance bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = perm.importances_mean.argsort()
    ax.barh(
        [FEATURES[i] for i in sorted_idx],
        perm.importances_mean[sorted_idx],
        xerr=perm.importances_std[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Mean Importance (drop in R^2)")
    ax.set_title("Feature Importance for DO Prediction")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: output/feature_importance.png")

    return model, scaler


# ── Health Classification ────────────────────────────────────────────────────
def train_health_classifier(df: pd.DataFrame):
    """Classify water conditions as Healthy / Stressed / Hypoxic."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION: Lagoon Health Status")
    print("=" * 60)

    X = df[FEATURES].values
    y_raw = df["health"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Lagoon Health Classification -- Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "health_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: output/health_confusion_matrix.png")

    return model, scaler


# ── Per-platform analysis ────────────────────────────────────────────────────
def platform_summary(df: pd.DataFrame):
    """Show summary stats and health breakdown per platform location."""
    print("\n" + "=" * 60)
    print("PER-LOCATION SUMMARY")
    print("=" * 60)

    summary_rows = []
    for tag in sorted(df["platform"].unique()):
        sub = df[df["platform"] == tag]
        loc = PLATFORM_LABELS[tag]
        do_vals = sub[TARGET]
        turb = sub["Turbidity (FNU)"]
        sal = sub["Salinity (PPT)"]
        n_hypoxic = (sub["health"] == "Hypoxic (<2 mg/L)").sum()
        n_stressed = (sub["health"] == "Stressed (2-4 mg/L)").sum()
        pct_problem = 100 * (n_hypoxic + n_stressed) / len(sub)

        summary_rows.append({
            "Platform": tag,
            "Location": loc,
            "Rows": len(sub),
            "DO mean": round(do_vals.mean(), 2),
            "DO min": round(do_vals.min(), 2),
            "Turbidity mean": round(turb.mean(), 2),
            "Salinity mean": round(sal.mean(), 2),
            "Hypoxic": n_hypoxic,
            "Stressed": n_stressed,
            "% Problematic": round(pct_problem, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "platform_summary.csv"), index=False)
    print(f"\n  Saved: output/platform_summary.csv")

    # Bar chart of problematic readings per location
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["Hypoxic"], width, label="Hypoxic (<2 mg/L)", color="crimson")
    ax.bar(x + width / 2, summary_df["Stressed"], width, label="Stressed (2-4 mg/L)", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['Platform']}\n{r['Location']}" for _, r in summary_df.iterrows()],
        fontsize=8,
    )
    ax.set_ylabel("Number of Readings")
    ax.set_title("Problematic Water Quality Readings by Location")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "problematic_by_location.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: output/problematic_by_location.png")

    # DO distribution by platform
    fig, ax = plt.subplots(figsize=(10, 5))
    platforms = sorted(df["platform"].unique())
    data = [df[df["platform"] == p][TARGET].values for p in platforms]
    bp = ax.boxplot(data, labels=[f"{p}\n{PLATFORM_LABELS[p]}" for p in platforms], patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(platforms)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.axhline(y=4, color="orange", linestyle="--", alpha=0.7, label="Stressed threshold (4 mg/L)")
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.7, label="Hypoxic threshold (2 mg/L)")
    ax.set_ylabel("Dissolved Oxygen (mg/L)")
    ax.set_title("DO Distribution Across Lagoon System Locations")
    ax.legend()
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "do_boxplot_by_location.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: output/do_boxplot_by_location.png")

    return summary_df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading platform data...")
    df = load_all_platforms()

    print("\nCleaning data...")
    df = clean_data(df)

    df = add_health_labels(df)

    reg_model, reg_scaler = train_do_regressor(df)
    clf_model, clf_scaler = train_health_classifier(df)
    summary = platform_summary(df)

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS FOR LAGOON HEALTH")
    print("=" * 60)
    print("""
  Dissolved oxygen (DO) is the single most critical indicator of
  lagoon and estuary health. The factors that drive DO down include:

    1. HIGH TEMPERATURE -- warm water holds less oxygen
    2. HIGH SALINITY / CONDUCTANCE -- saltwater stratification traps
       low-oxygen water near the bottom
    3. HIGH TURBIDITY -- blocks light, kills seagrass that produces O₂,
       and indicates sediment/nutrient loading
    4. DEPTH -- deeper water is farther from atmospheric re-oxygenation

  Locations where the canal meets the bay (L1-L2) and the river
  stations (L3-L4) are particularly vulnerable because nutrient-rich
  freshwater mixes with saltwater, creating stratification and
  fueling algal blooms that later decompose and consume oxygen.
""")
    print("Done -- all results saved to neural network/output/\n")


if __name__ == "__main__":
    main()
