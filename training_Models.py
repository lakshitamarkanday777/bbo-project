# ============================================================
# training_models.py — Final Training Script for BBO Streamlit App
# Trains Model A, Model B, Model C
# Saves: modelA_final.pkl, modelB_final.pkl, modelC_final.pkl
#        + feature lists
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================
# CREATE MODELS FOLDER
# ============================================================
os.makedirs("models", exist_ok=True)

print("\n=======================================================")
print("🚀 Starting Full Model Training Pipeline (A, B, C)")
print("=======================================================\n")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("data/detections_master.csv")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["dtStart"] = pd.to_datetime(df["dtStart"], errors="coerce")
df = df.dropna(subset=["datetime"])

##############################################################
# 🔵 MODEL A — MIGRATION PREDICTION (with SMOTE)
##############################################################

print("\n==============================")
print("MODEL A — Migration Prediction (SMOTE)")
print("==============================\n")

dfA = df.copy()

# Remove leakage features (same as before)
leak_cols = ["dir_change", "snr_spike", "night_move"]
dfA = dfA.drop(columns=leak_cols, errors="ignore")

features_A = [
    "snr", "sig", "sigsd", "noise",
    "freq", "freqsd", "freq_drift",
    "runLen", "motusFilter",
    "port", "angle", "bearing_deg",
    "hour", "day", "month", "weekday", "dayofyear",
    "sex", "age", "weight", "wing"
]

XA = dfA[features_A].copy()
yA = dfA["migration_activity"]

# Print distribution BEFORE fixing imbalance
print("\n🔍 Migration Activity Distribution BEFORE SMOTE:")
print(yA.value_counts())

# Encode categorical features exactly as before
XA["sex"] = XA["sex"].replace({"F": 0, "M": 1, "U": 0})
XA["age"] = XA["age"].replace({"ad": 1, "imm": 0, "unk": 0})

# Convert to numeric
XA = XA.apply(pd.to_numeric, errors="coerce").fillna(0)

# ==========================
# ⭐ Apply SMOTE Oversampling
# ==========================
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)

XA_resampled, yA_resampled = sm.fit_resample(XA, yA)
# ==============================================================
# 📊 PLOT — BEFORE vs AFTER SMOTE
# ==============================================================

import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- BEFORE SMOTE
sns.countplot(x=yA, ax=axes[0], palette="Blues")
axes[0].set_title("Before SMOTE — Migration Activity\n(Class Imbalance)")
axes[0].set_xlabel("Migration Activity")
axes[0].set_ylabel("Count")

# --- AFTER SMOTE
sns.countplot(x=yA_resampled, ax=axes[1], palette="Greens")
axes[1].set_title("After SMOTE — Balanced Classes")
axes[1].set_xlabel("Migration Activity")
axes[1].set_ylabel("Count")

plt.tight_layout()

# Save the figure
plt.savefig("plots/smote_comparison.png", dpi=300)
print("✔ Saved SMOTE comparison plot → plots/smote_comparison.png")

plt.close()

print("\n🔍 AFTER SMOTE (balanced):")
print(pd.Series(yA_resampled).value_counts())

# Train-test split
XA_train, XA_test, yA_train, yA_test = train_test_split(
    XA_resampled, yA_resampled,
    test_size=0.2,
    stratify=yA_resampled,
    random_state=42
)

# Define models for comparison
models_A = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

results_A = []

print("\nTraining Model A candidates...\n")

# Train model(s)
for name, model in models_A.items():
    model.fit(XA_train, yA_train)

    preds = model.predict(XA_test)
    proba = model.predict_proba(XA_test)[:, 1]

    acc = accuracy_score(yA_test, preds)
    f1 = f1_score(yA_test, preds)
    auc = roc_auc_score(yA_test, proba)

    print(f"{name}: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    results_A.append([name, acc, f1, auc])

# Pick best model
best_A_name = sorted(results_A, key=lambda x: x[3], reverse=True)[0][0]
best_A = models_A[best_A_name]

print(f"\n🔥 Best Model A: {best_A_name}")

# Retrain on FULL RESAMPLED DATA
best_A.fit(XA_resampled, yA_resampled)

# Save
joblib.dump(best_A, "models/modelA_final.pkl")
joblib.dump(features_A, "models/modelA_features.pkl")

print("✔ Model A saved!")
print("✔ A feature list saved!")



##############################################################
# 🟩 MODEL B — DETECTION DURATION (REGRESSION)
##############################################################

print("\n==============================")
print("MODEL B — Detection Duration")
print("==============================\n")

dfB = df.copy()
dfB["days_since_deployment"] = (dfB["datetime"] - dfB["dtStart"]).dt.days
dfB = dfB[dfB["days_since_deployment"] >= 0]

features_B = [
    "sex", "age", "weight", "wing", "speciesName",
    "manufacturer", "model", "lifespan", "nomFreq",
    "bearing_deg", "direction_str",
    "hour", "day", "month", "weekday", "dayofyear"
]

XB = dfB[features_B]
yB = dfB["days_since_deployment"]

# FIX: Ensure age is categorical
categorical_cols = [
    "sex", "age", "speciesName", "manufacturer",
    "model", "direction_str", "weekday"
]

numeric_cols = [
    "weight", "wing", "lifespan", "nomFreq",
    "bearing_deg", "hour", "day", "month", "dayofyear"
]

preprocess_B = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

XB_train, XB_test, yB_train, yB_test = train_test_split(
    XB, yB, test_size=0.2, random_state=42
)

models_B = {
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
}

results_B = []

print("Training Model B candidates...\n")

for name, model in models_B.items():
    pipe = Pipeline([("pre", preprocess_B), ("model", model)])
    pipe.fit(XB_train, yB_train)
    preds = pipe.predict(XB_test)

    mae = mean_absolute_error(yB_test, preds)
    rmse = np.sqrt(mean_squared_error(yB_test, preds))
    r2 = r2_score(yB_test, preds)

    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    results_B.append([name, mae, rmse, r2, pipe])

best_B_name, _, _, _, best_B_pipe = sorted(
    results_B, key=lambda x: x[3], reverse=True
)[0]

print(f"\n🔥 Best Model B: {best_B_name}")

best_B_pipe.fit(XB, yB)

joblib.dump(best_B_pipe, "models/modelB_final.pkl")
joblib.dump(features_B, "models/modelB_features.pkl")

print("✔ Model B saved!")
print("✔ B feature list saved!")
# ============================================================
# 📊 MODEL B — Top 10 Feature Importances
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

print("\nPlotting Model B Feature Importances (Top 10)...")

best_model = best_B_pipe.named_steps["model"]
preprocessor = best_B_pipe.named_steps["pre"]

# Get full feature names
cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
num_features = numeric_cols
all_features = np.concatenate([cat_features, num_features])

importances = best_model.feature_importances_

# Sort & select top 10
indices = np.argsort(importances)[::-1][:10]
top_features = all_features[indices]
top_importances = importances[indices]

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances, y=top_features, palette="viridis")
plt.title("Top 10 Feature Importances — Model B")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig("plots/modelB_feature_importance_top10.png", dpi=300)
plt.close()

print("✔ Saved → plots/modelB_feature_importance_top10.png")


# ============================================================
# 📊 MODEL B — Performance Comparison Plot
# ============================================================

results_df = pd.DataFrame(results_B, columns=["Model", "MAE", "RMSE", "R2", "Pipe"])
results_df = results_df.drop(columns=["Pipe"])

df_melt = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="magma")
plt.title("Model B — Performance Comparison")
plt.xlabel("Metric")
plt.ylabel("Score (lower better except R²)")
plt.tight_layout()

plt.savefig("plots/modelB_performance.png", dpi=300)
plt.close()

print("✔ Saved Model B performance plot → plots/modelB_performance.png")


##############################################################
# 🟧 MODEL C — SNR REGRESSION
##############################################################

print("\n==============================")
print("MODEL C — SNR Regression")
print("==============================\n")

dfC = df.copy()
dfC = dfC.dropna(subset=["snr"])

dfC["days_since_deployment"] = (dfC["datetime"] - dfC["dtStart"]).dt.days

features_C = [
    "days_since_deployment",
    "bearing_deg", "port",
    "hour", "day", "month", "weekday", "dayofyear",
    "freq", "freqsd", "freq_drift",
    "manufacturer", "model", "lifespan", "nomFreq",
    "age", "sex", "weight", "wing"
]

XC = dfC[features_C].copy()
yC = dfC["snr"].astype(float)

# encodings same as Model A
XC["sex"] = XC["sex"].replace({"F": 0, "M": 1, "U": 0})
XC["age"] = XC["age"].replace({"ad": 1, "imm": 0, "unk": 0})

XC["manufacturer"] = XC["manufacturer"].astype("category").cat.codes
XC["model"] = XC["model"].astype("category").cat.codes

XC = XC.apply(pd.to_numeric, errors="coerce").fillna(0)

XC_train, XC_test, yC_train, yC_test = train_test_split(
    XC, yC, test_size=0.2, random_state=42
)

models_C = {
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
}

results_C = []

print("Training Model C candidates...\n")

for name, model in models_C.items():
    model.fit(XC_train, yC_train)
    preds = model.predict(XC_test)

    mae = mean_absolute_error(yC_test, preds)
    rmse = np.sqrt(mean_squared_error(yC_test, preds))
    r2 = r2_score(yC_test, preds)

    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    results_C.append([name, mae, rmse, r2, model])

best_C_name, _, _, _, best_C = sorted(
    results_C, key=lambda x: x[3], reverse=True
)[0]

print(f"\n🔥 Best Model C: {best_C_name}")

best_C.fit(XC, yC)

joblib.dump(best_C, "models/modelC_final.pkl")
joblib.dump(features_C, "models/modelC_features.pkl")

print("✔ Model C saved!")
print("✔ C feature list saved!")
# ============================================================
# 📊 MODEL C — Top 10 Feature Importances
# ============================================================

print("\nPlotting Model C Feature Importances (Top 10)...")

feature_names_C = np.array(features_C)
importances_C = best_C.feature_importances_

indices_C = np.argsort(importances_C)[::-1][:10]

plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances_C[indices_C],
    y=feature_names_C[indices_C],
    palette="cool"
)
plt.title("Top 10 Feature Importances — Model C")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig("plots/modelC_feature_importance_top10.png", dpi=300)
plt.close()

print("✔ Saved → plots/modelC_feature_importance_top10.png")


# ============================================================
# 📊 MODEL C — Performance Comparison Plot
# ============================================================

results_df_C = pd.DataFrame(results_C, columns=["Model", "MAE", "RMSE", "R2", "Estimator"])
results_df_C = results_df_C.drop(columns=["Estimator"])

df_melt_C = results_df_C.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melt_C, x="Metric", y="Score", hue="Model", palette="crest")
plt.title("Model C — Performance Comparison (SNR Regression)")
plt.xlabel("Metric")
plt.ylabel("Score (lower better except R²)")
plt.tight_layout()

plt.savefig("plots/modelC_performance.png", dpi=300)
plt.close()

print("✔ Saved Model C performance plot → plots/modelC_performance.png")

print("\n=======================================================")
print("🎉 ALL MODELS TRAINED & SAVED SUCCESSFULLY (A, B, C)")
print("=======================================================\n")
