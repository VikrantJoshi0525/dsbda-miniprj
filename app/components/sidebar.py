"""
Sidebar navigation and filters.
"""

import streamlit as st
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import APP_TITLE, VERSION, COLORS, SAMPLE_DATA_DIR


def render_sidebar() -> dict:
    """
    Render the sidebar with navigation, data source, and filter controls.
    Returns a dict of user-selected values.
    """
    with st.sidebar:
        # ── Brand ────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align:center; padding: 1.5rem 0 1rem;">
                <div style="font-size: 2.8rem;">📊</div>
                <h2 style="
                    margin: 0.4rem 0 0;
                    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 1.3rem;
                    font-weight: 800;
                    letter-spacing: -0.01em;
                ">Sentiment Analyzer</h2>
                <p style="color: {COLORS['text']}55; font-size:0.75rem; margin:0.2rem 0 0;">
                    v{VERSION} • PySpark + Streamlit
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Navigation ──────────────────────────────────────
        st.markdown(
            f"<p style='color:{COLORS['text']}; font-weight:600; font-size:0.8rem; "
            f"text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;'>"
            f"Navigation</p>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            options=["🏠 Dashboard", "🔍 Analysis", "🤖 ML Model", "🔮 Predict", "⚙️ Settings"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Data Source ─────────────────────────────────────
        st.markdown(
            f"<p style='color:{COLORS['text']}; font-weight:600; font-size:0.8rem; "
            f"text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;'>"
            f"Data Source</p>",
            unsafe_allow_html=True,
        )

        data_source = st.radio(
            "Source",
            options=["📁 Sentiment140 (CSV)", "🎲 Demo Data"],
            index=0,
            label_visibility="collapsed",
            help="Use real Sentiment140 tweets or synthetic demo data",
        )

        csv_path = ""
        if "Sentiment140" in data_source:
            # Auto-detect CSV in data/sample/
            auto_path = _find_sentiment140_csv()
            if auto_path:
                st.markdown(
                    f"""<div style="
                        background: {COLORS['positive']}15;
                        border: 1px solid {COLORS['positive']}33;
                        border-radius: 10px;
                        padding: 0.6rem 0.8rem;
                        font-size: 0.78rem;
                        color: {COLORS['positive']};
                        margin-bottom: 0.5rem;
                    ">✅ Auto-detected:<br/><b>{Path(auto_path).name}</b></div>""",
                    unsafe_allow_html=True,
                )
                csv_path = auto_path
            else:
                csv_path = st.text_input(
                    "CSV Path",
                    value=str(SAMPLE_DATA_DIR / "sentiment140.csv"),
                    help="Absolute path to the Sentiment140 CSV file",
                )
                st.markdown(
                    f"""<div style="
                        background: {COLORS['accent']}15;
                        border: 1px solid {COLORS['accent']}33;
                        border-radius: 10px;
                        padding: 0.6rem 0.8rem;
                        font-size: 0.75rem;
                        color: {COLORS['accent']};
                        margin-bottom: 0.5rem;
                    ">💡 Place the CSV in <b>data/sample/</b><br/>
                    or enter the full path above</div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Filters ─────────────────────────────────────────
        st.markdown(
            f"<p style='color:{COLORS['text']}; font-weight:600; font-size:0.8rem; "
            f"text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;'>"
            f"Filters</p>",
            unsafe_allow_html=True,
        )

        sentiment = st.multiselect(
            "Sentiment",
            options=["Positive", "Negative", "Neutral"],
            default=["Positive", "Negative", "Neutral"],
        )

        method = st.selectbox(
            "Sentiment Engine",
            options=["Dataset Labels", "VADER", "TextBlob"],
            index=0,
            help="'Dataset Labels' uses original Sentiment140 labels; VADER/TextBlob re-scores via PySpark UDFs",
        )

        st.markdown("---")

        # ── Data Controls ───────────────────────────────────
        st.markdown(
            f"<p style='color:{COLORS['text']}; font-weight:600; font-size:0.8rem; "
            f"text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;'>"
            f"Data</p>",
            unsafe_allow_html=True,
        )

        if "Sentiment140" in data_source:
            sample_size = st.slider(
                "Row Limit",
                min_value=1000,
                max_value=50000,
                value=5000,
                step=1000,
                help="Number of rows to load from the CSV (controls speed)",
            )
        else:
            sample_size = st.slider(
                "Sample Size",
                min_value=100,
                max_value=5000,
                value=500,
                step=100,
            )

        regenerate = st.button("🔄  Reload Data", use_container_width=True)

        st.markdown("---")

        # ── Footer ──────────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align:center; padding:1rem 0; color:{COLORS['text']}44; font-size:0.7rem;">
                DSBDA Lab — Mini Project<br/>
                Built with ❤️ using PySpark
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Map method name
    method_key = {
        "Dataset Labels": "dataset",
        "VADER": "vader",
        "TextBlob": "textblob",
    }.get(method, "dataset")

    return {
        "page": page,
        "data_source": "sentiment140" if "Sentiment140" in data_source else "demo",
        "csv_path": csv_path,
        "sentiment": [s.lower() for s in sentiment],
        "method": method_key,
        "sample_size": sample_size,
        "regenerate": regenerate,
    }


def _find_sentiment140_csv() -> str | None:
    """Auto-detect a Sentiment140 CSV in the data/sample directory."""
    search_dir = SAMPLE_DATA_DIR
    if not search_dir.exists():
        return None

    # Common file names for Sentiment140
    candidates = [
        "training.1600000.processed.noemoticon.csv",
        "sentiment140.csv",
        "Sentiment140.csv",
        "testdata.manual.2009.06.14.csv",
    ]

    for name in candidates:
        path = search_dir / name
        if path.exists():
            return str(path)

    # Fallback: any CSV in the directory
    csvs = list(search_dir.glob("*.csv"))
    if csvs:
        return str(csvs[0])

    return None
