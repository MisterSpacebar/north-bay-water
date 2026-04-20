# ============================================================
# KNN Classification on Biscayne Bay Water Quality Data
# ============================================================
# Goal: Given a set of water-quality readings (temperature,
#        salinity, conductance, dissolved oxygen, turbidity),
#        predict WHICH STATION (L0-L7) the sample came from.
#
# Station locations:
#   L0 -- Biscayne Bay at FIU campus
#   L1 -- Biscayne Canal
#   L2 -- Biscayne Bay near canal-bay junction
#   L3 -- Little River (station A)
#   L4 -- Little River (station B)
#   L5 -- North Bay Village, north of JFK Causeway
#   L6 -- North Bay Village, south of JFK Causeway
#   L7 -- Miami River, south of Port Boulevard
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Step 1: Load the merged data
# ------------------------------------------------------------------
# Each row is a 5-minute reading across all stations.
# Columns are named like  temperature_c_l0, salinity_ppt_l1, etc.
raw = pd.read_csv("data/merged_keep.csv")
print(f"Raw data shape: {raw.shape}")

# ------------------------------------------------------------------
# Step 2: Reshape -- stack each station's readings into one long table
# ------------------------------------------------------------------
# Right now every station lives in its own set of columns.
# We want one row = one station-reading so KNN can learn per-station
# patterns.  We'll keep the 5 core water-quality features that every
# station shares:
#   temperature (deg C), specific conductance (uS/cm), salinity (ppt),
#   dissolved-oxygen saturation (%), turbidity (FNU)

FEATURES = ["temperature_c", "specific_conductance_us_cm",
            "salinity_ppt", "odo_sat", "turbidity_fnu"]

STATIONS = [f"l{i}" for i in range(8)]  # l0 ... l7

STATION_LABELS = {
    "l0": "L0 - FIU Bay",
    "l1": "L1 - Biscayne Canal",
    "l2": "L2 - Canal-Bay junction",
    "l3": "L3 - Little River A",
    "l4": "L4 - Little River B",
    "l5": "L5 - NBV north",
    "l6": "L6 - NBV south",
    "l7": "L7 - Miami River",
}

frames = []
for station in STATIONS:
    # Build the column names for this station, e.g. "temperature_c_l0"
    cols = {f"{feat}_{station}": feat for feat in FEATURES}

    # Check that the columns actually exist in the dataframe
    existing_cols = {k: v for k, v in cols.items() if k in raw.columns}
    if not existing_cols:
        print(f"  ⚠ skipping {station} -- no matching columns found")
        continue

    subset = raw[list(existing_cols.keys())].rename(columns=existing_cols).copy()
    subset["station"] = station
    frames.append(subset)

df = pd.concat(frames, ignore_index=True)
print(f"\nStacked data shape (before dropping NaN): {df.shape}")

# ------------------------------------------------------------------
# Step 3: Drop rows with missing values
# ------------------------------------------------------------------
# Many stations have gaps (sensors offline, etc.).
# KNN can't handle NaN, so we drop incomplete rows.
df.dropna(inplace=True)
print(f"Stacked data shape (after dropping NaN):  {df.shape}")
print(f"\nSamples per station:\n{df['station'].value_counts().sort_index()}\n")

# ------------------------------------------------------------------
# Step 4: Define features (X) and target (y)
# ------------------------------------------------------------------
# X = the 5 water-quality measurements
# y = the station label (what we're predicting)
X = df[FEATURES].values
y = df["station"].values

# ------------------------------------------------------------------
# Step 5: Scale the features
# ------------------------------------------------------------------
# KNN measures *distance* between points.  If one feature (e.g.
# conductance in the thousands) has a much larger range than another
# (e.g. temperature around 25-35), it would dominate the distance
# calculation.  StandardScaler centres each feature to mean=0 and
# std=1 so every feature contributes equally.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# Step 6: Train / test split (80 / 20)
# ------------------------------------------------------------------
# stratify=y ensures the same station proportions in train & test.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}\n")

# ------------------------------------------------------------------
# Step 7: Build & train the KNN classifier
# ------------------------------------------------------------------
# n_neighbors=7 -- we look at the 7 closest training points to vote
# on the station.  (With 8 classes an odd k avoids ties.)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# ------------------------------------------------------------------
# Step 8: Predict and evaluate
# ------------------------------------------------------------------
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {accuracy:.2%}\n")

# Per-class precision / recall / F1
# Precision = of all predicted as station X, how many truly were X?
# Recall    = of all actual station X samples, how many did we find?
# F1        = balanced mean of precision and recall
target_names = [STATION_LABELS[s] for s in sorted(df["station"].unique())]
print("=== Classification Report ===")
print(classification_report(y_test, y_pred,
                            labels=sorted(df["station"].unique()),
                            target_names=target_names))

# ------------------------------------------------------------------
# Step 9: Confusion matrix
# ------------------------------------------------------------------
# Rows = actual station, columns = predicted station.
# A perfect model would have numbers only on the diagonal.
cm = confusion_matrix(y_test, y_pred, labels=sorted(df["station"].unique()))
fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
ax.set_title("KNN Confusion Matrix -- Biscayne Bay Water Stations")
plt.tight_layout()
plt.savefig("water_knn_confusion.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# Step 10: Try different values of k and plot accuracy
# ------------------------------------------------------------------
# This helps you pick the best k.  Too small -> overfitting (noisy);
# too large -> underfitting (loses local patterns).
k_values = range(1, 26)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)

best_k = k_values[np.argmax(accuracies)]
print(f"\nBest k = {best_k}  (accuracy {max(accuracies):.2%})")

plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies, marker="o", markersize=4)
plt.axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.title("KNN Accuracy vs. k -- Biscayne Bay Station Classification")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("water_knn_k_accuracy.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# Step 11: Feature-pair scatter to visualise separation
# ------------------------------------------------------------------
# Plot salinity vs temperature, coloured by station.
# This gives a quick sense of how separable the stations are in just
# two dimensions (KNN uses all 5 features, so it sees even more).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

unique_stations = sorted(df["station"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_stations)))

for i, s in enumerate(unique_stations):
    mask = y_test == s
    axes[0].scatter(X_test[mask, 0], X_test[mask, 2],
                    c=[colors[i]], label=STATION_LABELS[s],
                    alpha=0.4, s=20, edgecolors="none")
    mask_pred = y_pred == s
    axes[1].scatter(X_test[mask_pred, 0], X_test[mask_pred, 2],
                    c=[colors[i]], label=STATION_LABELS[s],
                    alpha=0.4, s=20, edgecolors="none")

axes[0].set_title("Actual Station Labels")
axes[0].set_xlabel("Temperature (scaled)")
axes[0].set_ylabel("Salinity (scaled)")

axes[1].set_title("KNN Predicted Labels")
axes[1].set_xlabel("Temperature (scaled)")
axes[1].set_ylabel("Salinity (scaled)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("water_knn_scatter.png", dpi=150)
plt.show()

print("\nDone! Plots saved as water_knn_confusion.png, "
      "water_knn_k_accuracy.png, water_knn_scatter.png")
