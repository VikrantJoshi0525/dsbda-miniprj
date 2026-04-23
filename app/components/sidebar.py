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
            <div style="padding: 1rem 0 2rem;">
                <h1 style="
                    margin: 0;
                    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 1.8rem;
                    font-weight: 800;
                    letter-spacing: -0.02em;
                ">Sentiment Analytics</h1>
                <p style="color: {COLORS['text']}88; font-size:0.85rem; margin:0; font-weight: 500;">
                    Enterprise Edition v{VERSION}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Navigation ──────────────────────────────────────
        st.markdown("#### Navigation")
        page = st.radio(
            "Navigate",
            options=["Dashboard", "Live Stream", "Battle Mode", "Analysis", "ML Model", "Predict", "Settings"],
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Configuration ─────────────────────────────────────
        with st.expander("⚙️ Configuration", expanded=True):
            data_source = st.selectbox(
                "Data Source",
                options=["Demo Data", "Sentiment140 (CSV)"],
                index=0,
                help="Use real Sentiment140 tweets or synthetic demo data",
            )

            csv_path = ""
            if "Sentiment140" in data_source:
                auto_path = _find_sentiment140_csv()
                if auto_path:
                    st.success(f"Auto-detected: {Path(auto_path).name}", icon="✅")
                    csv_path = auto_path
                else:
                    csv_path = st.text_input(
                        "CSV Path",
                        value=str(SAMPLE_DATA_DIR / "sentiment140.csv"),
                    )

            sentiment = st.multiselect(
                "Sentiment Filter",
                options=["Positive", "Negative", "Neutral"],
                default=["Positive", "Negative", "Neutral"],
            )

            method = st.selectbox(
                "Analysis Engine",
                options=["Dataset Labels", "VADER", "TextBlob"],
                index=0,
            )

            live_subreddit = st.selectbox(
                "Live Subreddit",
                options=["all", "news", "technology", "worldnews", "politics", "movies", "gaming", "science"],
                index=2,
            )

            if "Sentiment140" in data_source:
                sample_size = st.slider("Row Limit", 1000, 50000, 5000, 1000)
            else:
                sample_size = st.slider("Sample Size", 100, 5000, 500, 100)

            regenerate = st.button("Apply & Reload", use_container_width=True)

        # ── Footer ──────────────────────────────────────────
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align:center; padding:1rem 0; color:{COLORS['text']}55; font-size:0.75rem;">
                Powered by Apache Spark & Streamlit
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
        "live_subreddit": live_subreddit,
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
