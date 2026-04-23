"""
Social Media Sentiment Analyzer — Main Entry Point
Run:  streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# ── Fix imports for running from project root ────────────────────
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from config import APP_TITLE, APP_ICON, APP_DESCRIPTION, COLORS, ASSETS_DIR
from components.sidebar import render_sidebar
from components.dashboard import render_dashboard
from components.analysis import render_analysis
from components.ml_page import render_ml_page
from components.predict_page import render_predict_page
from components.live_stream import render_live_stream
from utils.helpers import generate_sample_data


# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer — DSBDA",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load Custom CSS ──────────────────────────────────────────────
css_path = ASSETS_DIR / "style.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
filters = render_sidebar()


# ══════════════════════════════════════════════════════════════════
# ── Data Loading ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

# Handle reload
if "data_version" not in st.session_state:
    st.session_state.data_version = 0
if filters["regenerate"]:
    st.session_state.data_version += 1


@st.cache_data(show_spinner="🎲 Generating demo data…")
def _load_demo(sample_size: int, _version: int = 0) -> pd.DataFrame:
    return generate_sample_data(n=sample_size)


@st.cache_data(show_spinner="⚡ Running PySpark pipeline on Sentiment140…")
def _load_sentiment140(csv_path: str, limit: int, method: str, _version: int = 0) -> pd.DataFrame:
    """Load real Sentiment140 data through the PySpark pipeline."""
    from spark.session import get_spark
    from spark.loader import run_sentiment140_pipeline

    spark = get_spark()
    pdf = run_sentiment140_pipeline(
        spark=spark,
        csv_path=csv_path,
        limit=limit,
        method=method,
    )
    return pdf


# ── Choose data source ──────────────────────────────────────────
if filters["data_source"] == "sentiment140" and not filters["page"].startswith("Live"):
    csv_path = filters["csv_path"]
    if not csv_path or not Path(csv_path).exists():
        st.error(
            f"**CSV not found:** `{csv_path}`\n\n"
            "Please download the Sentiment140 dataset and place the CSV in `data/sample/`.\n\n"
            "**Download:** [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)"
        )
        st.info(
            "**Quick fix:** Switch to **🎲 Demo Data** in the sidebar to explore the app with synthetic data."
        )
        st.stop()

    try:
        df = _load_sentiment140(
            csv_path=csv_path,
            limit=filters["sample_size"],
            method=filters["method"],
            _version=st.session_state.data_version,
        )

        # Show pipeline success banner
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {COLORS['positive']}12, {COLORS['secondary']}08);
                border: 1px solid {COLORS['positive']}33;
                border-radius: 14px;
                padding: 0.8rem 1.2rem;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.8rem;
            ">
                <span style="font-size:1.4rem;">⚡</span>
                <div>
                    <span style="font-weight:700; color:{COLORS['positive']};">PySpark Pipeline Active</span>
                    <span style="color:{COLORS['text']}88; font-size:0.85rem; margin-left:0.6rem;">
                        {len(df):,} tweets loaded from Sentiment140 ·
                        Engine: {filters['method'].upper()} ·
                        {df['topic'].nunique()} topics detected
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"**PySpark pipeline failed:** {e}")
        st.markdown(
            f"""
            <div style="
                background: {COLORS['negative']}10;
                border: 1px solid {COLORS['negative']}33;
                border-radius: 12px;
                padding: 1rem 1.2rem;
                margin-top: 0.5rem;
            ">
                <p style="color:{COLORS['text']}; font-weight:600; margin:0 0 0.5rem;">Common fixes:</p>
                <ul style="color:{COLORS['text']}aa; margin:0; font-size:0.85rem;">
                    <li>Install Java 8, 11, or 17 and set <code>JAVA_HOME</code></li>
                    <li>Run <code>pip install pyspark</code></li>
                    <li>Ensure the CSV path is correct</li>
                    <li>Try reducing the <b>Row Limit</b> in the sidebar</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

else:
    # Demo mode
    df = _load_demo(filters["sample_size"], _version=st.session_state.data_version)


# ── Apply Sidebar Filters ───────────────────────────────────────
mask = df["sentiment_label"].str.lower().isin(filters["sentiment"])

# Platform filter only applies if the column has multiple values
if "platform" in df.columns and df["platform"].nunique() > 1:
    # For demo data, sidebar still works
    pass
mask = mask  # sentiment filter always applies

df_filtered = df[mask].copy()


# ══════════════════════════════════════════════════════════════════
# ── Route to Page ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
page = filters["page"]

if page.startswith("Dashboard"):
    render_dashboard(df_filtered)

elif page.startswith("Live"):
    render_live_stream(filters)

elif page.startswith("Battle"):
    from components.battle_mode import render_battle_mode
    render_battle_mode(filters)

elif page.startswith("Analysis"):
    render_analysis(df_filtered)

elif page.startswith("ML"):
    render_ml_page(
        df=df_filtered,
        data_source=filters["data_source"],
        csv_path=filters.get("csv_path", ""),
    )

elif page.startswith("Predict"):
    render_predict_page()

elif page.startswith("Settings"):
    # ── Settings page ────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin:0; font-size:2rem;">Settings & Info</h1>
            <p style="color:{COLORS['text']}77; font-size:0.95rem; margin:0.3rem 0 0;">
                Pipeline configuration and system information
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        source_label = "Sentiment140 CSV" if filters["data_source"] == "sentiment140" else "Demo (synthetic)"
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(26,26,46,0.7), rgba(22,33,62,0.5));
                border: 1px solid {COLORS['primary']}33;
                border-radius: 18px;
                padding: 2rem;
            ">
                <h3 style="margin-top:0;">🔧 Pipeline Configuration</h3>
                <table style="width:100%; color:{COLORS['text']}; font-size:0.9rem;">
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Data Source</td>
                        <td style="padding:0.5rem 0; font-weight:600;">{source_label}</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Sentiment Engine</td>
                        <td style="padding:0.5rem 0; font-weight:600;">{filters['method'].upper()}</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Rows Loaded</td>
                        <td style="padding:0.5rem 0; font-weight:600;">{len(df):,}</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">After Filters</td>
                        <td style="padding:0.5rem 0; font-weight:600;">{len(df_filtered):,}</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Pos. Threshold</td>
                        <td style="padding:0.5rem 0; font-weight:600;">0.05</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Neg. Threshold</td>
                        <td style="padding:0.5rem 0; font-weight:600;">-0.05</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(26,26,46,0.7), rgba(22,33,62,0.5));
                border: 1px solid {COLORS['secondary']}33;
                border-radius: 18px;
                padding: 2rem;
            ">
                <h3 style="margin-top:0;">📦 Tech Stack</h3>
                <table style="width:100%; color:{COLORS['text']}; font-size:0.9rem;">
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Frontend</td>
                        <td style="padding:0.5rem 0; font-weight:600;">Streamlit</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Processing</td>
                        <td style="padding:0.5rem 0; font-weight:600;">PySpark (local)</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">NLP</td>
                        <td style="padding:0.5rem 0; font-weight:600;">VADER / TextBlob</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Charts</td>
                        <td style="padding:0.5rem 0; font-weight:600;">Plotly</td></tr>
                    <tr><td style="padding:0.5rem 0; color:{COLORS['text']}88;">Data</td>
                        <td style="padding:0.5rem 0; font-weight:600;">Pandas + PySpark DF</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── PySpark Connection Test ──────────────────────────────
    st.markdown(f"### 🧪 Spark Connection Test")
    if st.button("Test PySpark Connection", use_container_width=False):
        with st.spinner("Initialising Spark…"):
            try:
                from spark.session import get_spark
                spark = get_spark()
                sdf = spark.createDataFrame(df_filtered.head(10))
                st.success(f"Spark is running! Created DataFrame with {sdf.count()} rows.")
                st.code(f"Spark Version: {spark.version}\nMaster: {spark.sparkContext.master}")
            except Exception as e:
                st.error(f"Spark connection failed: {e}")
                st.info("Make sure Java 8/11/17 is installed and JAVA_HOME is set.")

    # ── Dataset Info (when Sentiment140 is loaded) ───────────
    if filters["data_source"] == "sentiment140":
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(f"### 📄 Dataset Info")

        info_cols = st.columns(3, gap="medium")
        with info_cols[0]:
            st.metric("Total Rows", f"{len(df):,}")
        with info_cols[1]:
            st.metric("Unique Topics", df["topic"].nunique())
        with info_cols[2]:
            st.metric("Date Range", f"{df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}" if pd.notna(df['timestamp'].min()) else "N/A")

        with st.expander("Column Schema", expanded=False):
            schema_df = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(df[c].dtype) for c in df.columns],
                "Non-Null": [df[c].notna().sum() for c in df.columns],
                "Sample": [str(df[c].iloc[0])[:80] if len(df) > 0 else "" for c in df.columns],
            })
            st.dataframe(schema_df, use_container_width=True)
