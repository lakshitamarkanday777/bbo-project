import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from theme import apply_theme

apply_theme()

st.set_page_config(page_title="🧠 XAI — Model Interpretability", layout="wide")
st.title("🧠 Explainable AI — SHAP Interpretation")


# -------------------------------------------------------
# Load shared data and models once
# -------------------------------------------------------
@st.cache_resource
def load_all():
    df = pd.read_csv("data/detections_master.csv")

    models = {
        "A": joblib.load("models/modelA_final.pkl"),
        "B": joblib.load("models/modelB_final.pkl"),
        "C": joblib.load("models/modelC_final.pkl"),
    }

    features = {
        "A": joblib.load("models/modelA_features.pkl"),
        "B": joblib.load("models/modelB_features.pkl"),
        "C": joblib.load("models/modelC_features.pkl"),
    }

    return df, models, features


df_raw, models, features = load_all()


# =======================================================
# PREPROCESS FUNCTIONS
# =======================================================
def preprocess_A(df, featuresA):
    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    if df["weekday"].dtype == "object":
        df["weekday"] = df["weekday"].map(weekday_map)

    df["sex"] = df["sex"].replace({"F": 0, "M": 1, "U": 0})
    df["age"] = df["age"].replace({"ad": 1, "imm": 0, "unk": 0})

    X = df[featuresA].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X


def preprocess_B(df, featuresB):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["dtStart"]  = pd.to_datetime(df["dtStart"], errors="coerce")
    df["days_since_deployment"] = (df["datetime"] - df["dtStart"]).dt.days
    df = df[df["days_since_deployment"] >= 0]
    return df[featuresB]


def preprocess_C(df, featuresC):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["dtStart"]  = pd.to_datetime(df["dtStart"], errors="coerce")
    df["days_since_deployment"] = (df["datetime"] - df["dtStart"]).dt.days

    df = df.dropna(subset=["snr"])
    df = df[df["days_since_deployment"] >= 0]

    X = df[featuresC].copy()

    X["sex"] = X["sex"].replace({"F": 0, "M": 1, "U": 0})
    X["age"] = X["age"].replace({"ad": 1, "imm": 0, "unk": 0})
    X["manufacturer"] = X["manufacturer"].astype("category").cat.codes
    X["model"] = X["model"].astype("category").cat.codes

    return X.apply(pd.to_numeric, errors="coerce").fillna(0)


# =======================================================
# UI TABS
# =======================================================
tabA, tabB, tabC = st.tabs([
    "🟦 Model A — Migration Activity",
    "🟩 Model B — Residency Duration",
    "🟧 Model C — SNR Regression"
])


# =======================================================
# TAB A
# =======================================================
with tabA:
    st.header("Model A — Migration Activity")

    modelA = models["A"]
    featA = features["A"]
    X = preprocess_A(df_raw.copy(), featA)

    sample = st.slider("Sample Size", 50, 400, 200, key="A")

    runA = st.button("Run SHAP for Model A")

    if runA:
        X_shap = X.sample(sample, random_state=42)

        with st.spinner("Computing SHAP..."):
            explainer = shap.Explainer(modelA.predict, X_shap)
            shap_values = explainer(X_shap)

        st.success("SHAP Computed!")

        col1, col2 = st.columns(2)

        with col1:
            fig = plt.figure()
            shap.summary_plot(shap_values.values, X_shap, plot_type="bar", show=False)
            st.pyplot(fig)

        with col2:
            fig = plt.figure()
            shap.summary_plot(shap_values.values, X_shap, show=False)
            st.pyplot(fig)

        top_feature = X_shap.columns[np.argmax(np.abs(shap_values.values).mean(axis=0))]
        st.subheader(f"Dependence Plot — {top_feature}")

        # Convert Explanation → numpy
        shap_array = shap_values.values if hasattr(shap_values, "values") else shap_values

        fig, ax = plt.subplots(figsize=(5, 4))

        shap.dependence_plot(
            top_feature,
            shap_array,   # <-- FIXED HERE
            X_shap,
            ax=ax,
            show=False
        )

        st.pyplot(fig,use_container_width=False)



# =======================================================
# TAB B
# =======================================================
with tabB:
    st.header("Model B — Residency Duration")

    modelB = models["B"]
    featB = features["B"]
    X = preprocess_B(df_raw.copy(), featB)

    sample = st.slider("Sample Size (slow for large values)", 50, 200, 100, key="B")

    runB = st.button("Run SHAP for Model B")

    if runB:
        X_shap = X.sample(sample, random_state=42)
        background = X.sample(40, random_state=0)

        with st.spinner("Running KernelExplainer (slow)..."):
            explainer = shap.KernelExplainer(
                lambda arr: modelB.predict(pd.DataFrame(arr, columns=featB)),
                background.values
            )
            shap_values = explainer.shap_values(X_shap.values)

        st.success("SHAP Computed!")

        col1, col2 = st.columns(2)

        with col1:
            fig = plt.figure()
            shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
            st.pyplot(fig)

        with col2:
            fig = plt.figure()
            shap.summary_plot(shap_values, X_shap, show=False)
            st.pyplot(fig)

        top_feature = X_shap.columns[np.argmax(np.abs(shap_values).mean(axis=0))]
        st.subheader(f"Dependence Plot — {top_feature}")

        # Convert Explanation → numpy
        shap_array = shap_values.values if hasattr(shap_values, "values") else shap_values

        fig, ax = plt.subplots(figsize=(5, 4))

        shap.dependence_plot(
            top_feature,
            shap_array,   # <-- FIXED HERE
            X_shap,
            ax=ax,
            show=False
        )

        st.pyplot(fig,use_container_width=False)



# =======================================================
# TAB C
# =======================================================
with tabC:
    st.header("Model C — SNR Regression")

    modelC = joblib.load("models/modelC_final.pkl")
    features_C = joblib.load("models/modelC_features.pkl")

    # --------------------------------------------
    # 2️⃣ Load data
    # --------------------------------------------
    df = pd.read_csv("data/detections_master.csv")

    # Fix datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="ignore")
    df["dtStart"] = pd.to_datetime(df["dtStart"], errors="ignore")

    # Compute days since deployment
    df["days_since_deployment"] = (df["datetime"] - df["dtStart"]).dt.days
    df = df[df["days_since_deployment"] >= 0]

    # Drop rows with missing SNR (like in training)
    df = df.dropna(subset=["snr"])

    # --------------------------------------------
    # 3️⃣ Build XC exactly like training
    # --------------------------------------------
    XC = df[features_C].copy()

    # Same categorical encodings
    XC["sex"] = XC["sex"].replace({"F": 0, "M": 1, "U": 0})
    XC["age"] = XC["age"].replace({"ad": 1, "imm": 0, "unk": 0})

    XC["manufacturer"] = XC["manufacturer"].astype("category").cat.codes
    XC["model"] = XC["model"].astype("category").cat.codes

    # Convert to numeric
    XC = XC.apply(pd.to_numeric, errors="coerce").fillna(0)

    # --------------------------------------------
    # 4️⃣ UI — Sample size
    # --------------------------------------------
    sample_size = st.slider("Sample Size (slow for large values)", 50, 200, 100, key="C")

    run_shap_btn = st.button("Run SHAP for Model C")

    if run_shap_btn:

        with st.spinner("Computing TreeExplainer…"):

            # Sample data
            X_shap = XC.sample(sample_size, random_state=42)

            # Tree SHAP - fast, stable
            explainer = shap.TreeExplainer(
                modelC,
                feature_perturbation="tree_path_dependent"
            )
            
            shap_values = explainer.shap_values(
                X_shap,
                approximate=True
            )

        st.success("SHAP computation complete!")

        # --------------------------------------------
        # 5️⃣ SHAP Summary Bar Plot
        # --------------------------------------------
        st.subheader("Summary Bar Plot")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        st.pyplot(fig1,use_container_width=False)

        # --------------------------------------------
        # 6️⃣ SHAP Beeswarm Plot
        # --------------------------------------------
        st.subheader("Beeswarm Plot")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_shap, show=False)
        st.pyplot(fig2)

        # --------------------------------------------
        # 7️⃣ Dependence Plot for Top Feature
        # --------------------------------------------
        st.subheader("Dependence Plot — Top Feature")
        shap_mean = np.abs(shap_values).mean(axis=0)

        # If all zeros → no dependency plot
        if np.all(shap_mean == 0):
            st.warning("Cannot compute dependence plot — all SHAP impacts are zero.")
            st.stop()

        top_feature = X_shap.columns[np.argmax(shap_mean)]

        st.subheader(f"Dependence Plot — {top_feature}")

        # convert top feature to numeric if needed
        if X_shap[top_feature].dtype == "object":
            X_shap[top_feature] = pd.to_numeric(X_shap[top_feature], errors="coerce")

        fig3, ax3 = plt.subplots(figsize=(6, 6))
        shap.dependence_plot(top_feature, shap_values, X_shap, show=False)
        st.pyplot(fig3,use_container_width=False)
            


