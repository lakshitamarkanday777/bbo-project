import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="EDA", layout="wide")

from theme import apply_theme
apply_theme()

# ================================================================
# HELPERS
# ================================================================
def downsample(df, max_points=5000):
    """Downsample large datasets for scatter/histogram speed."""
    if len(df) > max_points:
        return df.sample(max_points, random_state=42)
    return df


# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data():
    det = pd.read_csv("data/detections_master.csv")
    mov = pd.read_csv("data/movement_summary.csv")

    det["datetime"] = pd.to_datetime(det["datetime"], errors="coerce")
    det["dtStart"] = pd.to_datetime(det["dtStart"], errors="coerce")

    return det, mov

detections_master, movement_summary = load_data()

# ================================================================
# PAGE NAVIGATION
# ================================================================
page = st.sidebar.radio(
    "📌 Select EDA Section",
    [
        "Dataset Overview",
        "Model A — Migration Activity EDA",
        "Model B — Duration Regression EDA",
        "Model C — SNR Modeling EDA",
    ]
)


# =================================================================
# 📍 PAGE 1 — Dataset Overview
# =================================================================
if page == "Dataset Overview":

    tab = st.tabs(["Dataset Overview"])[0]

    with tab:
        st.header("🔎 Dataset Overview")

        # ------------------------------------------------------
        # Detections Master Summary
        # ------------------------------------------------------
        st.subheader("📘 Detections Master — Summary")
        st.write(f"**Shape:** {detections_master.shape[0]} rows × {detections_master.shape[1]} columns")

        # Column Names
        with st.expander("📦 Show Column Names"):
            st.markdown(
                """
                <style>
                .column-badge {
                    display: inline-block;
                    background-color: #eef2ff;
                    padding: 6px 10px;
                    margin: 4px;
                    border-radius: 8px;
                    font-size: 14px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("".join([f"<span class='column-badge'>{c}</span>" for c in detections_master.columns]), unsafe_allow_html=True)

        # Data Types
        with st.expander("🧬 Show Column Data Types"):
            st.dataframe(detections_master.dtypes.to_frame("dtype"), use_container_width=True)

        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Total Detections", len(detections_master))

        with c2:
            st.metric("Unique Tags", detections_master["tagID"].nunique())

        with c3:
            st.metric("Unique Ports", detections_master["port"].nunique())

        st.metric(
            "Date Range",
            f"{detections_master['datetime'].min().date()} → {detections_master['datetime'].max().date()}",
        )

        # Preview
        st.subheader("📄 Preview — Detections Master")
        st.dataframe(detections_master.head(50), use_container_width=True)

        st.markdown("---")

        # Movement Summary
        st.subheader("📗 Movement Summary — Overview")
        st.write(f"**Shape:** {movement_summary.shape[0]} rows × {movement_summary.shape[1]} columns")

        with st.expander("📦 Show Movement Summary Columns"):
            st.markdown("".join([f"<span class='column-badge'>{c}</span>" for c in movement_summary.columns]), unsafe_allow_html=True)

        with st.expander("🧬 Show Movement Data Types"):
            st.dataframe(movement_summary.dtypes.to_frame("dtype"), use_container_width=True)

        st.markdown("### 📈 Movement Summary Stats")
        st.metric("Movement Rows", len(movement_summary))
        st.metric("Unique Tags", movement_summary["tagID"].nunique())
        st.metric("Avg Detection Days", round(movement_summary["detection_days"].mean(), 2))

        st.subheader("📄 Preview — Movement Summary")
        st.dataframe(movement_summary.head(50), use_container_width=True)



# =================================================================
# 📍 PAGE 2 — Model A EDA (Migration Activity)
# =================================================================
elif page == "Model A — Migration Activity EDA":

    st.header("🟦 Model A — Migration Activity EDA")

    tab_signal, tab_temporal, tab_direction, tab_corr = st.tabs(
        ["Signal Differences", "Temporal Patterns", "Directional Features", "Correlation Heatmap"]
    )


    # =======================================
    # TAB 1 — SIGNAL DIFFERENCES
    # =======================================
    with tab_signal:
        st.subheader("📡 1. Signal-Based Differences")

        c1, c2, c3 = st.columns(3)

        with c1:
            fig = px.box(detections_master, x="migration_activity", y="snr", color="migration_activity",
                         title="SNR vs Migration")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.box(detections_master, x="migration_activity", y="noise", color="migration_activity",
                         title="Noise vs Migration")
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            fig = px.box(detections_master, x="migration_activity", y="freq_drift", color="migration_activity",
                         title="Frequency Drift vs Migration")
            st.plotly_chart(fig, use_container_width=True)



    # =======================================
    # TAB 2 — TEMPORAL PATTERNS
    # =======================================
    with tab_temporal:
        st.subheader("⏱️ 2. Temporal Detection Patterns")

        # Hour plot (downsampled)
        fig = px.histogram(
            downsample(detections_master),
            x="hour",
            facet_col="migration_activity",
            nbins=24,
            color="migration_activity",
            title="Hourly Detection Pattern"
        )
        st.plotly_chart(fig, use_container_width=True)

        c4, c5 = st.columns(2)

        with c4:
            fig = px.histogram(detections_master, x="weekday", color="migration_activity",
                               barmode="group", title="Weekday Migration Trend")
            st.plotly_chart(fig, use_container_width=True)

        with c5:
            fig = px.histogram(detections_master, x="month", color="migration_activity",
                               barmode="group", title="Monthly Migration Pattern")
            st.plotly_chart(fig, use_container_width=True)



    # =======================================
    # TAB 3 — DIRECTIONAL FEATURES
    # =======================================
    with tab_direction:
        st.subheader("🧭 3. Directional Characteristics")

        c6, c7 = st.columns(2)

        with c6:
            fig = px.scatter(
                downsample(detections_master),
                x="bearing_deg", y="snr",
                color="migration_activity",
                opacity=0.4,
                title="Bearing vs SNR"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c7:
            fig = px.histogram(
                detections_master,
                x="port",
                color="migration_activity",
                barmode="group",
                title="Port Activity vs Migration"
            )
            st.plotly_chart(fig, use_container_width=True)



    # =======================================
    # TAB 4 — CORRELATION HEATMAP
    # =======================================
    with tab_corr:

        st.subheader("📈 4. Correlation Heatmap")

        num_cols = [
            "snr", "sig", "noise", "freq", "freqsd",
            "freq_drift", "bearing_deg",
            "hour", "day", "month", "dayofyear"
        ]

        corr_df = detections_master[num_cols].corr()

        fig = px.imshow(
            corr_df, text_auto=True, aspect="auto",
            title="Correlation Heatmap — Migration Activity Features"
        )
        st.plotly_chart(fig, use_container_width=True)



# =================================================================
# 📍 PAGE 3 — Model B EDA
# =================================================================
elif page == "Model B — Duration Regression EDA":

    st.header("🟩 Model B — Detection Duration EDA")

    # Tabs for Model B
    tab_bio, tab_device, tab_direction, tab_temp, tab_corr = st.tabs(
        ["Biological", "Tag/Device", "Directional", "Temporal", "Correlation"]
    )

    # Preprocess for B
    dfB = detections_master.copy()
    dfB["days_since_deployment"] = (dfB["datetime"] - dfB["dtStart"]).dt.days
    dfB = dfB[dfB["days_since_deployment"] >= 0]


    # ---------------------- TAB: Biological -----------------------
    with tab_bio:
        st.subheader("🧬 1. Biological Features")

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(px.box(dfB, x="sex", y="days_since_deployment",
                                   title="Duration by Sex"), use_container_width=True)

        with c2:
            st.plotly_chart(px.box(dfB, x="age", y="days_since_deployment",
                                   title="Duration by Age"), use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.plotly_chart(px.scatter(downsample(dfB), x="weight", y="days_since_deployment",
                                       color="sex", opacity=0.5, title="Weight vs Duration"), use_container_width=True)

        with c4:
            st.plotly_chart(px.scatter(downsample(dfB), x="wing", y="days_since_deployment",
                                       color="sex", opacity=0.5, title="Wing vs Duration"), use_container_width=True)

        st.plotly_chart(px.box(dfB, x="speciesName", y="days_since_deployment",
                               title="Duration by Species"), use_container_width=True)


    # ---------------------- TAB: Device -----------------------
    with tab_device:
        c5, c6 = st.columns(2)

        with c5:
            st.plotly_chart(px.box(dfB, x="manufacturer", y="days_since_deployment",
                                   title="Manufacturer vs Duration"), use_container_width=True)

        with c6:
            st.plotly_chart(px.box(dfB, x="model", y="days_since_deployment",
                                   title="Model vs Duration"), use_container_width=True)

        c7, c8 = st.columns(2)

        with c7:
            st.plotly_chart(px.scatter(downsample(dfB), x="lifespan", y="days_since_deployment",
                                       title="Tag Lifespan vs Duration"), use_container_width=True)

        with c8:
            st.plotly_chart(px.scatter(downsample(dfB), x="nomFreq", y="days_since_deployment",
                                       title="NomFreq vs Duration"), use_container_width=True)


    # ---------------------- TAB: Directional -----------------------
    with tab_direction:
        st.subheader("🧭 3. Directional Features")

        c9, c10 = st.columns(2)

        with c9:
            st.plotly_chart(px.scatter(downsample(dfB), x="bearing_deg", y="days_since_deployment",
                                       color="direction_str", title="Bearing vs Duration"),
                            use_container_width=True)

        with c10:
            st.plotly_chart(px.box(dfB, x="direction_str", y="days_since_deployment",
                                   title="Direction vs Duration"), use_container_width=True)


    # ---------------------- TAB: Temporal -----------------------
    with tab_temp:
        st.subheader("⏱️ 4. Temporal Patterns")

        c11, c12, c13 = st.columns(3)

        with c11:
            st.plotly_chart(px.scatter(downsample(dfB), x="hour", y="days_since_deployment",
                                       title="Hour vs Duration"), use_container_width=True)

        with c12:
            st.plotly_chart(px.box(dfB, x="weekday", y="days_since_deployment",
                                   title="Weekday vs Duration"), use_container_width=True)

        with c13:
            st.plotly_chart(px.box(dfB, x="month", y="days_since_deployment",
                                   title="Month vs Duration"), use_container_width=True)


    # ---------------------- TAB: Correlation -----------------------
    with tab_corr:
        st.subheader("📈 5. Duration Correlation Map")

        corr_cols = [
            "days_since_deployment", "bearing_deg", "weight", "wing",
            "lifespan", "nomFreq", "hour", "day", "month"
        ]

        corr_df = dfB[corr_cols].corr()

        st.plotly_chart(px.imshow(corr_df, text_auto=True, title="Correlation Heatmap — Duration"),
                        use_container_width=True)



# =================================================================
# 📍 PAGE 4 — Model C EDA
# =================================================================
elif page == "Model C — SNR Modeling EDA":

    st.header("🟧 Model C — SNR Regression EDA")

    # Tabs for Model C
    tab_signal, tab_temp, tab_direction, tab_bio, tab_corr = st.tabs(
        ["Signal", "Temporal", "Directional", "Biological", "Correlation"]
    )

    dfC = detections_master.dropna(subset=["snr"]).copy()
    dfC["days_since_deployment"] = (dfC["datetime"] - dfC["dtStart"]).dt.days


    # ---------------------- TAB: Signal -----------------------
    with tab_signal:
        st.subheader("📡 1. Signal Feature Relationships")

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(px.scatter(downsample(dfC), x="freq", y="snr",
                                       opacity=0.3, title="Frequency vs SNR"),
                            use_container_width=True)

        with c2:
            st.plotly_chart(px.scatter(downsample(dfC), x="freqsd", y="snr",
                                       opacity=0.3, title="Frequency SD vs SNR"),
                            use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.plotly_chart(px.scatter(downsample(dfC), x="freq_drift", y="snr",
                                       opacity=0.3, title="Frequency Drift vs SNR"),
                            use_container_width=True)

        with c4:
            st.plotly_chart(px.box(dfC, x="manufacturer", y="snr",
                                   title="Manufacturer vs SNR"),
                            use_container_width=True)

        st.plotly_chart(px.box(dfC, x="model", y="snr", title="Model vs SNR"),
                        use_container_width=True)


    # ---------------------- TAB: Temporal -----------------------
    with tab_temp:
        st.subheader("⏱️ 2. Temporal SNR Patterns")

        c5, c6, c7 = st.columns(3)

        with c5:
            st.plotly_chart(px.scatter(downsample(dfC), x="hour", y="snr",
                                       title="SNR by Hour"), use_container_width=True)

        with c6:
            st.plotly_chart(px.scatter(downsample(dfC), x="day", y="snr",
                                       title="SNR by Day"), use_container_width=True)

        with c7:
            st.plotly_chart(px.scatter(downsample(dfC), x="month", y="snr",
                                       title="SNR by Month"), use_container_width=True)


    # ---------------------- TAB: Direction -----------------------
    with tab_direction:
        st.subheader("🧭 3. Directional Features")

        c8, c9 = st.columns(2)

        with c8:
            st.plotly_chart(px.scatter(downsample(dfC), x="bearing_deg", y="snr",
                                       opacity=0.3, title="Bearing vs SNR"),
                            use_container_width=True)

        with c9:
            st.plotly_chart(px.box(dfC, x="port", y="snr",
                                   title="Port vs SNR"), use_container_width=True)


    # ---------------------- TAB: Biological -----------------------
    with tab_bio:
        st.subheader("🧬 4. Biological Influence on SNR")

        c10, c11 = st.columns(2)

        with c10:
            st.plotly_chart(px.box(dfC, x="sex", y="snr", title="Sex vs SNR"),
                            use_container_width=True)

        with c11:
            st.plotly_chart(px.box(dfC, x="age", y="snr", title="Age vs SNR"),
                            use_container_width=True)

        c12, c13 = st.columns(2)

        with c12:
            st.plotly_chart(px.scatter(downsample(dfC), x="weight", y="snr",
                                       opacity=0.3, title="Weight vs SNR"),
                            use_container_width=True)

        with c13:
            st.plotly_chart(px.scatter(downsample(dfC), x="wing", y="snr",
                                       opacity=0.3, title="Wing vs SNR"),
                            use_container_width=True)


    # ---------------------- TAB: Correlation -----------------------
    with tab_corr:
        st.subheader("📈 5. Correlation Heatmap")

        corr_cols = [
            "snr", "days_since_deployment", "freq", "freqsd",
            "freq_drift", "bearing_deg",
            "weight", "wing", "hour", "day", "month"
        ]

        corr_df = dfC[corr_cols].corr()

        st.plotly_chart(px.imshow(corr_df, text_auto=True,
                                  title="Correlation Heatmap — SNR Features"),
                        use_container_width=True)
