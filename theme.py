import streamlit as st

def apply_theme():
    st.markdown("""
                
    <style>

    html, body, [data-testid="stAppViewContainer"], main, .main, .block-container {
        background-color: #f2eee9 !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    div.block-container {
        padding-top: 0 !important;
    }

    /* ---------------------- SIDEBAR ---------------------- */
    [data-testid="stSidebar"] {
        background-color: #efe8df !important;
        padding: 30px 20px !important;
        border-right: 1px solid #ddcfc3 !important;
    }

    [data-testid="stSidebar"] * {
        font-size: 18px !important;
        color: #4a3f35 !important;
    }

    [data-testid="stSidebarNavLink"].active {
        background-color: #d4c7bb !important;
        border-radius: 10px !important;
        border-left: 5px solid #8c6f57 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebarNavLink"]:hover {
        background-color: #ded4c8 !important;
        border-radius: 10px !important;
    }

    /* ----------------------- CARDS ----------------------- */
    .card {
        background-color: #ffffff !important;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 36px;
        box-shadow: 0px 4px 14px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(230, 230, 230, 0.6);
        transition: all 0.2s ease-in-out;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 12px 28px rgba(0, 0, 0, 0.18);
    }

    /* -------------------- TYPOGRAPHY --------------------- */
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
