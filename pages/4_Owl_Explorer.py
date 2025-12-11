import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from theme import apply_theme
apply_theme()

st.set_page_config(page_title="🦉 Owl Explorer", layout="wide")

# ============================================
# COLOR SCHEME FOR PORTS (MATCHING YOUR CHOSEN COLORS)
# ============================================
port_colors = {
    "omni": "#FFF44F",      # Yellow
    215: "#E9965A",         # Orange (SE)
    135: "#A8D38B",         # Green (SW)
    0:   "#9CC7E8"          # Blue (North)
}

# FUNCTION — Map movement degrees to port color (Omni excluded from direction plots)
def map_direction_color(bearing):
    if bearing == -10:
        return port_colors["omni"]
    
    elif bearing >= 315 or bearing < 45:
        return port_colors[0]      # North
    elif 45 <= bearing < 135:
        return None                # Omni → EXCLUDE from directional plots
    elif 135 <= bearing < 215:
        return port_colors[135]    # SW
    elif 215 <= bearing < 315:
        return port_colors[215]    # SE


# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    det = pd.read_csv("data/detections_master.csv")
    mov = pd.read_csv("data/movement_summary.csv")

    for col in ["datetime", "DATE", "ts", "tsCorrected"]:
        if col in det.columns:
            det[col] = pd.to_datetime(det[col], errors="coerce")

    return det, mov

det, mov = load_data()

# ============================================
# PAGE TITLE
# ============================================
st.title("🦉 Owl Explorer — Deep Individual Analysis")

# ============================================
# SELECT OWL
# ============================================
tag_ids = det["tagID"].dropna().unique()
selected_tag = st.selectbox("Select an Owl (Tag ID)", sorted(tag_ids))

owl_data = det[det["tagID"] == selected_tag].copy()
owl_summary = mov[mov["tagID"] == selected_tag]

st.markdown("---")
st.subheader(f"📌 Information for Owl Tag **{selected_tag}**")

if len(owl_summary) > 0:
    st.write(owl_summary.T)
else:
    st.warning("No summary data found for this owl.")

st.markdown("---")

# =========================================================
# 🧭 MOVEMENT INTERPRETATION SUMMARY (NEW)
# =========================================================

st.subheader("🧭 Movement Interpretation Summary")

if len(owl_data) > 0:

    # --- compute last known direction ---
    if "direction_str" in owl_data.columns:
        last_dir = owl_data.sort_values("datetime")["direction_str"].dropna().iloc[-1]
    else:
        last_dir = "Unknown"

    # --- get movement summary from movement_summary.csv ---
    if len(owl_summary) > 0:
        dom = owl_summary["dominant_direction"].iloc[0]
        pct_se = owl_summary["pct_SE"].iloc[0]
        pct_sw = owl_summary["pct_SW"].iloc[0]
        pct_n = owl_summary["pct_N"].iloc[0]
        pct_omni = owl_summary["pct_omni"].iloc[0]
    else:
        dom = pct_se = pct_sw = pct_n = pct_omni = None

    # --- interpretation logic ---
    if pct_omni is not None and pct_omni > 40:
        movement_type = (
            "This owl spent much of its time **near the tower**, "
            "with limited directional movement (high omni detections)."
        )
    elif dom == "SE":
        movement_type = (
            "The owl predominantly moved **toward the Southeast**, "
            "consistent with possible *migration departure*."
        )
    elif dom == "SW":
        movement_type = (
            "The owl frequently moved **Southwest**, "
            "suggesting regional/local movement patterns."
        )
    elif dom == "N":
        movement_type = (
            "The owl showed **Northward movement**, "
            "which may indicate local foraging routes."
        )
    else:
        movement_type = (
            "This owl showed **mixed or low-confidence movement direction**, "
            "with no strong directional bias."
        )

    # --- summary box ---
    st.markdown(
        f"""
        <div style="padding:18px;background:#fdf2d0;border-left:8px solid #d4a017;
                    border-radius:8px;font-size:16px;line-height:1.6;">
                    
        <strong>📍 Dominant Direction:</strong> {dom}<br>
        <strong>🔄 Last Recorded Direction:</strong> {last_dir}<br>
        <strong>📊 Direction Breakdown:</strong><br>
        • SE: {pct_se:.1f}% &nbsp; • SW: {pct_sw:.1f}% &nbsp; • N: {pct_n:.1f}% &nbsp; • Omni: {pct_omni:.1f}%<br><br>

        <strong>🧭 Interpretation:</strong><br>
        {movement_type}
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# 🧭 MOVEMENT ROSE PLOT — USING PORT COLOR SCHEME
# =========================================================


col_left, col_right = st.columns([1, 1])

# ---------------------------------------
# LEFT COLUMN — ROSE PLOT + TITLE
# ---------------------------------------
with col_left:
    st.markdown("### 🧭 Movement Rose Plot (Directional Only)")
    
    if "bearing_deg" in owl_data.columns:
        owl_data["dir_color"] = owl_data["bearing_deg"].apply(map_direction_color)
        dir_df = owl_data[owl_data["dir_color"].notna()]

        if len(dir_df) > 0:
            radians = np.deg2rad(dir_df["bearing_deg"])

            fig = plt.figure(figsize=(4, 4))  # smaller to match right plot
            ax = fig.add_subplot(111, projection="polar")

            ax.scatter(
                radians,
                np.ones_like(radians),
                c=dir_df["dir_color"],
                s=45,
                alpha=0.9
            )

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title("")   # remove duplicate title spacing

            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No directional detections for this owl.")
    else:
        st.info("No bearing data available.")

# ---------------------------------------
# RIGHT COLUMN — HISTOGRAM + TITLE
# ---------------------------------------
with col_right:
    st.subheader("🧭 Bearing Histogram (Directional Only)")

    if len(dir_df) > 0:

        fig, ax = plt.subplots(figsize=(5,4.7))

        # Histogram
        counts, bins, patches = ax.hist(
            dir_df["bearing_deg"],
            bins=36,              # More bins = more accurate degree matching
            edgecolor="black",
            linewidth=0.8
        )

        # Exact degrees and colors — NOW INCLUDES -10° (YELLOW)
        exact_degrees = {
            -10: port_colors["omni"],  # Yellow for Omni cases
            0: port_colors[0],         # Blue for North
            135: port_colors[135],     # Green for SW
            215: port_colors[215],     # Orange for SE
        }

        TOL = 10   # tolerance for bin midpoint matching

        # Color each bin
        for patch, left_bin, right_bin in zip(patches, bins[:-1], bins[1:]):
            mid = (left_bin + right_bin) / 2  # midpoint of histogram bin

            # Find which special degree the midpoint is closest to
            closest = min(exact_degrees.keys(), key=lambda d: abs(mid - d))

            # If midpoint is within ±10° of that degree → color it
            if abs(mid - closest) <= TOL:
                patch.set_facecolor(exact_degrees[closest])
            else:
                patch.set_facecolor("#d8d8d8")   # Grey for all other bins

        # Plot labels
        ax.set_title("Distribution of Owl Movement Bearings")
        ax.set_xlabel("Bearing (°)")
        ax.set_ylabel("Detections")

        st.pyplot(fig)

    else:
        st.info("No directional antenna detections to plot.")



# ✔ SHOW OMNI COUNT AS TEXT
omni_ct = owl_data[owl_data["port"] == 1].shape[0]
st.info(f"ℹ️ Omni detections (no direction): **{omni_ct}**")

# LEGEND
st.markdown("### Port / Color Legend")
for name, col in {
    "Omni (No Direction)": port_colors["omni"],
    "SE – 215° (Port 2)": port_colors[215],
    "SW – 135° (Port 3)": port_colors[135],
    "North – 0° (Port 4)": port_colors[0]
}.items():
    st.markdown(
        f"<div style='display:flex;align-items:center;'>"
        f"<div style='width:18px;height:18px;background:{col};margin-right:8px;'></div>"
        f"{name}</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================================================
# ANTENNA BEARING USAGE (FIXED — NOW UPDATES CORRECTLY)
# =========================================================
# st.subheader("📡 Antenna Bearing Usage (Port Colors)")

# if "antBearing" in owl_data.columns:

#     counts = owl_data["antBearing"].value_counts().sort_index()
#     bar_colors = [port_colors.get(b, "#777777") for b in counts.index]

#     fig, ax = plt.subplots(figsize=(5,4))
#     counts.plot(kind="bar", color=bar_colors, ax=ax)
#     ax.set_title("Antenna Bearings Hit by This Owl")
#     ax.set_xlabel("Bearing")
#     ax.set_ylabel("Detections")

#     st.pyplot(fig)

# else:
#     st.info("No antenna bearing data available.")

# st.markdown("---")

# =========================================================
# DISTRIBUTION OF TRUE MOVEMENT BEARINGS
# =========================================================
# st.subheader("🧭 Bearing Histogram (Directional Only)")

# if len(dir_df) > 0:

#     fig, ax = plt.subplots(figsize=(5,4))
#     sns.histplot(
#         dir_df["bearing_deg"],
#         bins=24,
#         edgecolor="black",
#         color="#6aaf97"
#     )

#     ax.set_title("Distribution of Owl Movement Bearings")
#     ax.set_xlabel("Bearing (°)")
#     ax.set_ylabel("Detections")

#     st.pyplot(fig,use_container_width=False)

# else:
#     st.info("No directional antenna detections to plot.")

# st.markdown("---")

# =========================================================
# HOURLY ACTIVITY
# =========================================================
st.subheader("⏱ Hourly Activity Pattern")

if "datetime" in owl_data.columns:
    owl_data["hour"] = owl_data["datetime"].dt.hour

    fig, ax = plt.subplots(figsize=(7,3))
    owl_data["hour"].value_counts().sort_index().plot(kind="bar", ax=ax)

    ax.set_title("Detections Per Hour")

    st.pyplot(fig,use_container_width=False)

st.markdown("---")

# =========================================================
# SIGNAL QUALITY
# =========================================================
st.subheader("📶 Signal Quality Insights")

col1, col2 = st.columns(2)

with col1:
    if "snr" in owl_data.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(owl_data["snr"], bins=20, ax=ax)
        ax.set_title("SNR Distribution")
        st.pyplot(fig,use_container_width=False)

with col2:
    if "noise" in owl_data.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(owl_data["noise"], bins=20, color="purple", ax=ax)
        ax.set_title("Noise Distribution")
        st.pyplot(fig,use_container_width=False)

# Scatter: Freq vs SNR
if {"freq", "snr"}.issubset(owl_data.columns):
    fig, ax = plt.subplots(figsize=(7,3))
    sns.scatterplot(data=owl_data, x="freq", y="snr", alpha=0.5, ax=ax)
    ax.set_title("Frequency vs SNR")
    st.pyplot(fig,use_container_width=False)

st.markdown("---")

# =========================================================
# TIMELINE OF DETECTIONS
# =========================================================
st.subheader("📅 Detection Timeline")

if "datetime" in owl_data.columns:
    start = owl_data["datetime"].min()
    end = owl_data["datetime"].max()

    fig, ax = plt.subplots(figsize=(10,1))
    ax.plot([start, end], [1,1], linewidth=14, color="#3AAFA9")
    ax.set_yticks([])
    ax.set_title("Timeline of Owl Detections")

    st.pyplot(fig,use_container_width=False)

st.markdown("---")

# =========================================================
# COMPARISON TO ALL OWLS
# =========================================================
st.subheader("📊 Comparison: This Owl vs All Birds")

colA, colB = st.columns(2)

with colA:
    st.metric("This Owl's Avg SNR", round(owl_data["snr"].mean(), 3))
with colB:
    st.metric("Overall Avg SNR", round(det["snr"].mean(), 3))

with colA:
    st.metric("This Owl's Avg Noise", round(owl_data["noise"].mean(), 3))
with colB:
    st.metric("Overall Avg Noise", round(det["noise"].mean(), 3))

with colA:
    peak_hour = owl_data["hour"].value_counts().idxmax()
    st.metric("Peak Hour (This Owl)", int(peak_hour))

with colB:
    overall_peak = det["datetime"].dt.hour.value_counts().idxmax()
    st.metric("Peak Hour (All Owls)", int(overall_peak))
