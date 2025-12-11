# source venv310/bin/activate
# streamlit run app.py

#python training_Models.py

# ------------------------------
# streamlit run app.py
# ------------------------------

# import streamlit as st
# from PIL import Image
# import base64
# import pandas as pd

# if "detections_master" not in st.session_state:
#     st.session_state["detections_master"] = pd.read_csv("data/detections_master.csv")

# if "movement_summary" not in st.session_state:
#     st.session_state["movement_summary"] = pd.read_csv("data/movement_summary.csv")


# st.set_page_config(layout="wide")

# # -------------------------------------------------------
# # 🔧 FIX ALL EXTRA STREAMLIT SPACING (IMPORTANT!)
# # -------------------------------------------------------
# st.markdown("""
# <style>

# div.block-container {
#     padding-top: 0rem !important;
# }

# section.main > div {
#     padding-top: 0 !important;
#     margin-top: -25px !important;
# }

# [data-testid="stVerticalBlock"] {
#     padding-top: 0 !important;
#     margin-top: 0 !important;
# }

# </style>
# """, unsafe_allow_html=True)

# # -------------------------------------------------------
# # 🎨 GLOBAL STYLE
# # -------------------------------------------------------
# st.markdown("""
# <style>

# body { background-color: #f5f5f5; }



# .section-title {
#     font-size: 24px;
#     font-weight: 700;
#     margin-bottom: 12px;
# }

# </style>
# """, unsafe_allow_html=True)


# # -------------------------------------------------------
# # 🦉 HERO BANNER — WITH BASE64 IMAGE
# # -------------------------------------------------------
# def get_base64_image(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode()

# img_base64 = get_base64_image("images/owl_banner.jpg")

# st.markdown(f"""
# <style>
# .hero {{
#     width: 100%;
#     height: 360px;
#     border-radius: 16px;

#     background-image: url("data:image/jpg;base64,{img_base64}");
#     background-size: cover;
#     background-position: center -40px;
#     background-repeat: no-repeat;

#     display: flex;
#     align-items: flex-end; 
#     justify-content: flex-start;

#     padding-left: 55px;    /* Move horizontally */
#     padding-bottom: 35px;  /* Move vertically */
#     box-shadow: 0 4px 16px rgba(0,0,0,0.28);
# }}

# .hero-title {{
#     font-size: 40px;
#     font-weight: 800;
#     color: white;
#     text-shadow: 0 4px 14px rgba(0,0,0,0.65);
#     margin-bottom: 100px;   /* Lift heading upward */
# }}
# </style>

# <div class="hero">
#     <div class="hero-title">🦉 Northern Saw-Whet Owl Migration Dashboard</div>
# </div>
# """, unsafe_allow_html=True)


# # -------------------------------------------------------
# # 🔍 OVERVIEW
# # -------------------------------------------------------
# st.markdown('<div class="card">', unsafe_allow_html=True)
# st.markdown('<div class="section-title">🔍 Overview</div>', unsafe_allow_html=True)

# st.write("""
# This dashboard analyzes **Saw-Whet Owl detections** using real Motus wildlife-tracking data.

# It integrates **machine learning**, **data visualization**, and **explainable AI (XAI)** to help understand:

# - 🕊️ When owls show active migration behavior  
# - 🌿 How biological & environmental factors influence signal quality (SNR)  
# - 📡 Movement patterns during the fall migration season  
# """)

# st.markdown('</div>', unsafe_allow_html=True)


# # -------------------------------------------------------
# # 🎯 PROJECT GOALS
# # -------------------------------------------------------
# st.markdown('<div class="card">', unsafe_allow_html=True)
# st.markdown('<div class="section-title">🎯 Project Goals</div>', unsafe_allow_html=True)

# colA, colB = st.columns(2)

# with colA:
#     st.markdown("### 🐦 Problem A — Migration Classification")
#     st.write("""
#     **Goal:** Predict whether an owl detection indicates active migration.  
#     **Model Used:** Random Forest Classifier  

#     **Key Features:**
#     - SNR, signal strength, noise  
#     - Antenna direction (port)  
#     - Bearing changes  
#     - Hour-of-day & day-of-year  
#     - Burst/movement characteristics  
#     """)

# with colB:
#     st.markdown("### 📡 Problem B — SNR Regression (Nanotag Performance)")
#     st.write("""
#     **Goal:** Identify what factors influence signal-to-noise ratio (SNR).  
#     **Model Used:** Random Forest Regression  

#     **Key Features:**
#     - Days since deployment  
#     - Bearing rotation & antenna direction  
#     - Biological metadata (age, sex, wing, weight)  
#     - Tag specifications (model, lifespan, nominal frequency)  
#     - Temporal features (month, weekday, hour)  
#     """)

# st.markdown('</div>', unsafe_allow_html=True)


# # -------------------------------------------------------
# # 🧭 WHAT YOU CAN EXPLORE
# # -------------------------------------------------------
# st.markdown('<div class="card">', unsafe_allow_html=True)
# st.markdown('<div class="section-title">🧭 What You Can Explore</div>', unsafe_allow_html=True)

# st.write("""
# - 📊 **EDA** — Explore owl movement, timing, bursts, and signal patterns  
# - 🤖 **Modeling** — Migration classification & SNR prediction models  
# - ✨ **XAI** — SHAP plots, feature contributions, local explanations  
# """)

# st.markdown('</div>', unsafe_allow_html=True)
import streamlit as st
from PIL import Image
import base64
import pandas as pd

# -------------------------------------------------------
# LOAD DATA INTO SESSION STATE
# -------------------------------------------------------
if "detections_master" not in st.session_state:
    st.session_state["detections_master"] = pd.read_csv("data/detections_master.csv")

if "movement_summary" not in st.session_state:
    st.session_state["movement_summary"] = pd.read_csv("data/movement_summary.csv")

st.set_page_config(layout="wide")

# -------------------------------------------------------
# 🔧 GLOBAL CSS + SPACING FIXES
# -------------------------------------------------------

st.markdown("""
<style>

/* ------------------------- */
/* PAGE BACKGROUND THEME     */
/* ------------------------- */

html, body, [data-testid="stAppViewContainer"], main, .main, .block-container {
    background-color: #f2eee9 !important;
}

/* Remove header background */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}

/* Remove top decoration */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Fix top padding */
div.block-container {
    padding-top: 0 !important;
}

/* ------------------------- */
/* SIDEBAR STYLING           */
/* ------------------------- */

[data-testid="stSidebar"] {
    background-color: #efe8df !important;   /* soft beige */
    padding: 30px 20px 20px 20px !important;
    border-right: 1px solid #ddcfc3 !important;
}

[data-testid="stSidebar"] * {
    font-size: 18px !important;
    color: #4a3f35 !important;   /* soft dark brown text */
}

/* Active link */
[data-testid="stSidebarNavLink"].active {
    background-color: #d4c7bb !important;
    border-radius: 10px !important;
    border-left: 5px solid #8c6f57 !important;
    font-weight: 700 !important;
}

/* Hover */
[data-testid="stSidebarNavLink"]:hover {
    background-color: #ded4c8 !important;
    border-radius: 10px !important;
}

/* ------------------------- */
/* CARD STYLING              */
/* ------------------------- */

.card {
    background-color: #ffffff !important;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 36px;
    box-shadow: 0px 4px 14px rgba(0, 0, 0, 0.08);
    transition: all 0.25s ease-in-out;
    border: 1px solid rgba(230, 230, 230, 0.6);
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 12px 28px rgba(0, 0, 0, 0.18);
}

/* Optional premium accent bar */
/*
.card {
    border-left: 6px solid #b59a7e !important;
}
*/

/* ------------------------- */
/* TYPOGRAPHY                */
/* ------------------------- */

body, div, p, li {
    font-size: 19px !important;
    line-height: 1.55 !important;
    color: #333 !important;
}

.section-title {
    font-size: 30px !important;
    font-weight: 900 !important;
    margin-bottom: 14px;
    color: #2e2b29 !important;
}

.sub-header {
    font-size: 26px !important;
    font-weight: 800 !important;
    margin-top: 18px;
    margin-bottom: 12px;
    color: #3a3735 !important;
}

</style>
""", unsafe_allow_html=True)



# -------------------------------------------------------
# 🦉 HERO BANNER — LEFT ALIGNED TEXT
# -------------------------------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image("images/owl_banner.jpg")

st.markdown(f"""
<style>

.hero {{
    width: 100%;
    height: 380px;
    border-radius: 18px;

    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center -40px;
    background-repeat: no-repeat;

    display: flex;
    flex-direction: column;
    justify-content: flex-end;

    padding-left: 60px;
    padding-bottom: 60px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.30);
    position: relative;
}}

.hero::after {{
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,0.35);
    border-radius: 18px;
}}

.hero-title {{
    position: relative;
    z-index: 2;
    font-size: 66px !important;
    font-weight: 900 !important;
    color: white !important;
    text-shadow: 0 6px 18px rgba(0,0,0,1);
    margin: 0;
    padding: 0;
    text-align: left;
}}

.hero-subtitle {{
    position: relative;
    z-index: 2;
    font-size: 66px !important;
    font-weight: 900 !important;
    color: white !important;
    margin-top: -5px;            /* tighten spacing */
    padding-left: 95px;          /* align under text, not emoji */
    text-shadow: 0 6px 18px rgba(0,0,0,1);
    text-align: left;
}}

</style>

<div class="hero">
    <div class="hero-title">🦉 Northern Saw-Whet Owl Migration</div>
    <div class="hero-subtitle">Dashboard</div>
</div>
""", unsafe_allow_html=True)




# -------------------------------------------------------
# 🔍 OVERVIEW
# -------------------------------------------------------
st.markdown("""
<div class="card">
<div class="section-title">🔍 Overview</div>

Welcome to the <b>Saw-Whet Owl Migration Analytics Dashboard</b>, an interactive tool built using
real <b>Motus Wildlife Tracking System detections</b>.

<br><br>
This dashboard integrates <b>machine learning</b>, <b>signal analysis</b>, and
<b>explainable AI (XAI)</b> to help you explore:

<ul>
<li>🕊️ When owls are actively migrating</li>
<li>📡 How SNR and signal characteristics vary across antennas</li>
<li>🌍 How owls move across tracking towers during fall migration</li>
<li>🧠 Why models make certain predictions (SHAP explanations)</li>
</ul>

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# 🎯 PURPOSE OF THE APP
# -------------------------------------------------------
st.markdown("""
<div class="card">
<div class="section-title">🎯 Purpose of the App</div>

This dashboard supports:

<ul>
<li><b>Biologists</b> exploring owl movement</li>
<li><b>Researchers</b> evaluating nanotag performance</li>
<li><b>Students</b> learning ML & XAI</li>
<li><b>Data analysts</b> studying ecological patterns</li>
</ul>

The goal is to transform complex detection logs into 
<b>clear, interpretable insights.</b>

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# 🧠 WHAT THE APP HELPS YOU UNDERSTAND
# -------------------------------------------------------
st.markdown("""
<div class="card">
<div class="section-title">🧠 What This Dashboard Helps You Understand</div>

<ul>
<li>When detections likely indicate <b>active migration</b></li>
<li>Which biological & environmental factors influence <b>SNR</b></li>
<li>How antenna direction, bearing, tag metadata, and time affect detections</li>
<li>How machine learning models interpret signals & movements</li>
</ul>

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# ⚙️ MACHINE LEARNING MODELS
# -------------------------------------------------------
st.markdown("""
<div class="card">
<div class="section-title">⚙️ Machine Learning Models</div>

<div class="sub-header">🦉 Model A — Migration Classification</div>
<p>
<b>Goal:</b> Predict whether a detection represents <b>active migration</b>.<br>
<b>Model:</b> Random Forest Classifier<br><br>
Uses signal strength, SNR, burst timing, bearing change, and port direction.
</p>

<div class="sub-header">📡 Model B — SNR Regression (Nanotag Performance)</div>
<p>
<b>Goal:</b> Understand what influences <b>SNR</b>.<br>
<b>Model:</b> Random Forest Regressor<br><br>
Uses tag metadata, deployment age, and time variables.
</p>

<div class="sub-header">🛰️ Model C — SNR Regression (Movement & Environment)</div>
<p>
<b>Goal:</b> Predict SNR using movement-related and environmental indicators.<br>
<b>Model:</b> Random Forest Regressor<br><br>
Focuses on bearing rotation, distance change, bursts, and seasonality.
</p>

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# 🧭 USER GUIDE
# -------------------------------------------------------
st.markdown("""
<div class="card">
<div class="section-title">🧭 User Guide</div>

<h4>1️⃣ EDA</h4>
Explore owl activity, migration timing, bursts, antennas, and SNR patterns.

<h4>2️⃣ Modeling</h4>
View ML predictions and SHAP explanations for all three models.

<h4>3️⃣ Owl Explorer</h4>
Browse detections by tag, timestamp, antenna, or signal attributes.

<h4>4️⃣ XAI</h4>
Understand <b>why</b> the models make specific predictions using SHAP.

<h4>5️⃣ RAG Chatbot</h4>
Ask natural language questions like:<br>
<i>“Why was SNR low here?”</i><br>
<i>“What affects migration probability the most?”</i>

</div>
""", unsafe_allow_html=True)



# -------------------------------------------------------
# ENDING NOTE
# -------------------------------------------------------
st.write("""
Use the sidebar to explore migration patterns, SNR behavior, XAI explanations, and more.  
Enjoy discovering patterns in one of North America’s most fascinating migrating owls! 🦉✨
""")
