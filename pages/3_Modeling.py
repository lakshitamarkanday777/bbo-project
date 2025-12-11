import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px

from theme import apply_theme
apply_theme()

def plot_feature_importance(model, feature_list, title):
    if not hasattr(model, "feature_importances_"):
        st.warning("This model does not support feature importances.")
        return

    importance_df = pd.DataFrame({
        "Feature": feature_list,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def get_modelB_expanded_features(modelB):
    pre = modelB.named_steps["pre"]
    ohe = pre.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(['sex','age','speciesName','manufacturer','model','direction_str','weekday'])
    
    # numeric features
    num_features = [
        "weight", "wing", "lifespan", "nomFreq",
        "bearing_deg", "hour", "day", "month", "dayofyear"
    ]

    return list(cat_features) + num_features

st.set_page_config(page_title="Modeling & Predictions", layout="wide")

# -------------------------------------------------------
# LOAD MODELS & FEATURE LISTS
# -------------------------------------------------------
@st.cache_resource
def load_models():
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
    return models, features

models, features = load_models()
st.success("Models loaded successfully!")

# -------------------------------------------------------
# LOAD DATA FOR DROPDOWNS
# -------------------------------------------------------
df = pd.read_csv("data/detections_master.csv")

# Weekday mapping
WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_map = {name: i for i, name in enumerate(WEEKDAY_ORDER)}

# Categorical value lists
CATEGORIES = {
    "sex": sorted(df["sex"].dropna().unique().tolist()),
    "age": sorted(df["age"].dropna().unique().tolist()),
    "speciesName": sorted(df["speciesName"].dropna().unique().tolist()),
    "manufacturer": sorted(df["manufacturer"].dropna().unique().tolist()),
    "model": sorted(df["model"].dropna().unique().tolist()),
    "direction_str": sorted(df["direction_str"].dropna().unique().tolist()),
    "weekday": WEEKDAY_ORDER
}

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def create_input(feature, key_prefix):
    """Creates appropriate input widget for a feature"""
    if feature in CATEGORIES:
        return st.selectbox(
            feature,
            CATEGORIES[feature],
            key=f"{key_prefix}_{feature}"
        )

    # numeric feature → use median as default
    default_val = (
        float(df[feature].dropna().median())
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
        else 0.0
    )

    return st.number_input(
        feature,
        value=float(default_val),
        key=f"{key_prefix}_{feature}"
    )

import shap

# -----------------------------------------------------------
# Helper — SHAP-based explanation for Model A
# -----------------------------------------------------------
def explain_model_a(model, X_row):
    """Returns prediction, probability, and natural SHAP explanation."""

    # 1 — predict
    prob = float(model.predict_proba(X_row)[0][1])
    pred = int(prob >= 0.5)

    # 2 — compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_row)[0]

    # --- FIX: convert any array → scalar float ---
    def to_scalar(v):
        try:
            # If v is array-like, take first element
            return float(v[0]) if hasattr(v, "__len__") else float(v)
        except:
            return float(v)

    shap_dict = {f: to_scalar(v) for f, v in zip(X_row.columns, shap_vals)}

    # 3 — top 3 contributing features
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    explanation = []
    for feat, val in top:
        direction = "increases" if val > 0 else "decreases"
        explanation.append(
            f"- **{feat}** {direction} migration likelihood ({val:+.3f})"
        )

    return pred, prob, "\n".join(explanation)

# -------------------------------------------------------
# PAGE TABS
# -------------------------------------------------------
tabA, tabB, tabC = st.tabs([
    "Model A — Migration Prediction",
    "Model B — Detection Duration",
    "Model C — SNR Prediction"
])


# =======================================================
# ⭐ MODEL A — MIGRATION ACTIVITY
# =======================================================
with tabA:
    st.header("🟦 Model A — Migration Activity Prediction")

    # Short explanation
    st.markdown("""
    ### 📘 What the Model Does
    **Model A predicts whether an owl is actively migrating or simply moving locally.**  
    **This helps researchers identify true migratory events and understand seasonal movement patterns.**
    """)

    featA = features["A"]
    modelA = models["A"]
    
    cols = st.columns(3)
    inputsA = {}

    for i, feature in enumerate(featA):
        col = cols[i % 3]
        inputsA[feature] = create_input(feature, key_prefix=f"A_{feature}")

    if st.button("Predict Migration Activity", key="btn_A"):

        # Build input row
        X = pd.DataFrame([{f: inputsA[f] for f in featA}])

        # Encode categorical features (same as training)
        X["sex"] = X["sex"].replace({"F": 0, "M": 1, "U": 0})
        X["age"] = X["age"].replace({"ad": 1, "imm": 0, "unk": 0})

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Run explanation
        pred, prob, shap_text = explain_model_a(modelA, X)

        # ============================
        # RESULT UI BLOCK
        # ============================

        if pred == 1:
            st.markdown(f"""
            <div style="
                background-color:#FFE5CC;
                padding:18px;
                border-radius:12px;
                border-left:6px solid #FF8C42;
                font-size:18px;">
                🟠 <strong>Migration Likely</strong><br>
                Probability: <strong>{prob*100:.1f}%</strong><br>
                This detection shows characteristics consistent with active migration.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background-color:#E3F2FD;
                padding:18px;
                border-radius:12px;
                border-left:6px solid #1E88E5;
                font-size:18px;">
                🔵 <strong>No Migration Detected</strong><br>
                Migration probability: <strong>{prob*100:.1f}%</strong><br>
                This detection resembles local or short-distance movement.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔍 Why the model predicted this")
        st.markdown(shap_text)

        st.markdown("<br><small>Positive SHAP values push toward migration (1). Negative SHAP values push toward local activity (0).</small>",
                    unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔎 Feature Importance — Model A")
        plot_feature_importance(modelA, featA, "Model A Feature Importance")



# =======================================================
# ⭐ MODEL B — Duration
# =======================================================
with tabB:
    st.header("🟩 Model B — Detection Duration Prediction")
    
    featB = features["B"]
    modelB = models["B"]
    

    cols = st.columns(2)
    inputsB = {}

    for i, feature in enumerate(featB):
        col = cols[i % 2]
        inputsB[feature] = create_input(feature, key_prefix="B")

    if st.button("Predict Duration", key="btn_B"):
        X = pd.DataFrame([{f: inputsB[f] for f in featB}])

        # pipeline handles encoding — just pass raw
        duration = modelB.predict(X)[0]

        st.subheader("Predicted Detection Duration")
        st.success(f"Estimated Days Since Deployment: **{round(duration, 2)} days**")

        st.markdown("""
        <div style="
            background-color: #f7f3e9;
            padding: 18px 22px;
            border-left: 6px solid #5a8f77;
            border-radius: 6px;
            margin-top: 10px;
            font-size: 17px;
            line-height: 1.5;">
        <strong>📘 How to Interpret This Prediction:</strong><br><br>
        This estimate reflects how many days the owl continued to be detected at the station 
        after its tag was deployed. 

        <ul style="margin-top: 8px;">
        <li><strong>0–3 days</strong> → The owl likely made a brief stop or passed through quickly.</li>
        <li><strong>3–10 days</strong> → The owl remained in the area for a moderate duration (typical stopover).</li>
        <li><strong>10+ days</strong> → The owl stayed near the station for an extended time, 
        possibly showing temporary residency behaviour.</li>
        </ul>

        Higher values generally mean the owl stayed longer at the site, while lower values 
        indicate rapid movement away from the deployment location.
        </div>
        """, unsafe_allow_html=True)


        # ---------- FIXED FEATURE IMPORTANCE FOR MODEL B ----------
        st.subheader("🔎 Feature Importance — Model B")

        expanded_featB = get_modelB_expanded_features(modelB)  # <-- NEW

        rf = modelB.named_steps["model"]  # extract RandomForest from pipeline

        importance_df = (
            pd.DataFrame({
                "Feature": expanded_featB,
                "Importance": rf.feature_importances_
            })
            .sort_values("Importance", ascending=False)
            .head(10)
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Model B — Top Feature Importances"
        )

        st.plotly_chart(fig, use_container_width=True)


# =======================================================
# ⭐ MODEL C — SNR
# =======================================================
with tabC:
    st.header("🟧 Model C — SNR Prediction")

    featC = features["C"]
    modelC = models["C"]
    
    cols = st.columns(2)
    inputsC = {}

    for i, feature in enumerate(featC):
        col = cols[i % 2]
        inputsC[feature] = create_input(feature, key_prefix="C")

    if st.button("Predict SNR", key="btn_C"):
        X = pd.DataFrame([{f: inputsC[f] for f in featC}])

        X["sex"] = X["sex"].replace({"F": 0, "M": 1, "U": 0})
        X["age"] = X["age"].replace({"ad": 1, "imm": 0, "unk": 0})

        # manufacturer & model encoded as category codes during training
        X["manufacturer"] = X["manufacturer"].astype("category").cat.codes
        X["model"] = X["model"].astype("category").cat.codes

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        snr_pred = modelC.predict(X)[0]

        st.subheader("Predicted SNR")
        st.success(f"Estimated SNR: **{round(snr_pred, 3)}**")
        
        # ---- Dynamic Interpretation for Model C ----
        snr_value = float(snr_pred)

        if snr_value < 0:
            label = "🔴 Very Weak / No Detectable Signal"
            color = "#ffcccc"
            explanation = """
The signal is extremely weak or buried in noise.<br>
The tag was likely <strong>far from the station</strong>, obstructed, or not transmitting clearly.
"""
        elif snr_value < 5:
            label = "🟠 Weak but Detectable Signal"
            color = "#ffe5cc"
            explanation = """
The signal is weak but still detectable.<br>
This often occurs when the owl is <strong>near the edge of detection range</strong> or partially obstructed.
"""
        elif snr_value < 10:
            label = "🟡 Moderate Signal Strength"
            color = "#fff7cc"
            explanation = """
A moderate, reliable signal.<br>
The owl is likely <strong>within a reasonable distance</strong> from the station with minimal obstruction.
"""
        else:
            label = "🟢 Strong Signal"
            color = "#e5ffcc"
            explanation = """
A strong and clear signal.<br>
This usually indicates the owl is <strong>very close to the station</strong> with a clean line of sight.
"""

        # ⭐ FIXED HTML BLOCK — NO INDENTATION ⭐
        html_block = f"""
<div style="
    padding: 18px;
    border-radius: 10px;
    background-color: {color};
    border-left: 10px solid black;
    font-size: 16px;
    line-height: 1.6;
">

<strong style="font-size: 18px;">Signal Strength Interpretation:</strong><br>
<span style="font-size: 18px;">{label}</span>
<br><br>

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
{explanation}
</div>

</div>
"""

        st.markdown(html_block, unsafe_allow_html=True)

        st.subheader("🔎 Feature Importance — Model C")
        plot_feature_importance(modelC, featC, "Model C Feature Importance")


