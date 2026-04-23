"""
Battle Mode - Compare two brands/topics in real-time.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLORS
from utils.reddit_live import search_live_reddit_posts
from components.visualizations import _LAYOUT_DEFAULTS

# Initialize VADER for scoring
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except ImportError:
    vader = None

def get_sentiment(text: str) -> float:
    if vader is None:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
    return vader.polarity_scores(text)["compound"]

def classify_sentiment(score: float) -> str:
    if score > 0.05: return "Positive"
    if score < -0.05: return "Negative"
    return "Neutral"

def render_battle_mode(filters: dict):
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin:0; font-size:2.2rem; color:{COLORS['primary']};">Brand Battle Mode</h1>
            <p style="color:{COLORS['text']}77; font-size:1rem; margin:0.3rem 0 0;">
                Live comparative sentiment analysis across Reddit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([2, 2, 1], gap="medium")
    with c1:
        entity_a = st.text_input("Brand / Topic A", value="AI", max_chars=30)
    with c2:
        entity_b = st.text_input("Brand / Topic B", value="Crypto", max_chars=30)
    with c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    if not entity_a or not entity_b:
        st.warning("Please enter two entities to compare.")
        return

    st.markdown("---")

    if auto_refresh:
        _render_battle_fragment(entity_a, entity_b)
    else:
        _render_battle_fragment(entity_a, entity_b, static=True)

@st.fragment(run_every="10s")
def _render_battle_fragment(entity_a: str, entity_b: str, static=False):
    # Fetch Data
    with st.spinner("Fetching live data..."):
        posts_a = search_live_reddit_posts(entity_a, limit=50)
        posts_b = search_live_reddit_posts(entity_b, limit=50)

    if not posts_a and not posts_b:
        st.error("Could not fetch data for either entity.")
        return

    df_a = pd.DataFrame(posts_a)
    df_b = pd.DataFrame(posts_b)

    if not df_a.empty:
        df_a["sentiment_score"] = df_a["text"].apply(get_sentiment)
        df_a["sentiment_label"] = df_a["sentiment_score"].apply(classify_sentiment)
        df_a["entity"] = entity_a
    
    if not df_b.empty:
        df_b["sentiment_score"] = df_b["text"].apply(get_sentiment)
        df_b["sentiment_label"] = df_b["sentiment_score"].apply(classify_sentiment)
        df_b["entity"] = entity_b

    # Combine
    dfs = []
    if not df_a.empty: dfs.append(df_a)
    if not df_b.empty: dfs.append(df_b)
    df = pd.concat(dfs, ignore_index=True)

    # Metrics calculation
    score_a = df_a["sentiment_score"].mean() if not df_a.empty else 0.0
    score_b = df_b["sentiment_score"].mean() if not df_b.empty else 0.0

    pos_a = len(df_a[df_a["sentiment_label"] == "Positive"]) if not df_a.empty else 0
    pos_b = len(df_b[df_b["sentiment_label"] == "Positive"]) if not df_b.empty else 0

    count_a = len(df_a)
    count_b = len(df_b)

    # Determine Winner
    if score_a > score_b + 0.05:
        winner = entity_a
        win_color = COLORS["positive"]
    elif score_b > score_a + 0.05:
        winner = entity_b
        win_color = COLORS["positive"]
    else:
        winner = "It's a Tie"
        win_color = COLORS["neutral"]

    st.markdown(
        f"""
        <div style="text-align:center; padding: 1rem; background: linear-gradient(135deg, rgba(26,26,46,0.7), rgba(22,33,62,0.5)); border-radius: 12px; border: 1px solid {win_color}55; margin-bottom: 2rem;">
            <div style="color:{COLORS['text']}88; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Current Winner</div>
            <div style="font-size:2rem; font-weight:900; color:{win_color};">{winner}</div>
            <div style="font-size:0.8rem; color:{COLORS['text']}55;">Based on average sentiment score of latest {count_a + count_b} posts</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Side-by-side KPIs
    colA, colB = st.columns(2)
    
    def _kpi_card(entity, score, pos, total, color):
        pct = (pos / total * 100) if total > 0 else 0
        return f"""
        <div style="background: {color}15; border-top: 4px solid {color}; border-radius: 8px; padding: 1.5rem; text-align: center;">
            <h2 style="margin:0 0 0.5rem; color:{COLORS['text']};">{entity}</h2>
            <div style="font-size: 2.5rem; font-weight: 800; color:{color};">{score:+.2f}</div>
            <div style="font-size: 0.8rem; color:{COLORS['text']}88; margin-top:0.5rem;">Average Sentiment</div>
            <hr style="border-color:{color}33; margin: 1rem 0;"/>
            <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
                <span style="color:{COLORS['text']}aa;">Positive Posts</span>
                <span style="font-weight:700; color:{color};">{pct:.1f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-top:0.3rem;">
                <span style="color:{COLORS['text']}aa;">Volume</span>
                <span style="font-weight:700; color:{COLORS['text']};">{total}</span>
            </div>
        </div>
        """

    with colA:
        c_color = COLORS["primary"] if winner == entity_a else COLORS["text"]
        st.markdown(_kpi_card(entity_a, score_a, pos_a, count_a, c_color), unsafe_allow_html=True)
    
    with colB:
        c_color = COLORS["secondary"] if winner == entity_b else COLORS["text"]
        st.markdown(_kpi_card(entity_b, score_b, pos_b, count_b, c_color), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Overlapping Distribution
    fig = px.histogram(
        df, 
        x="sentiment_score", 
        color="entity", 
        barmode="overlay",
        nbins=20,
        color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]]
    )
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_layout(
        title="Sentiment Distribution Comparison",
        xaxis_title="Sentiment Score",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(opacity=0.7)
    
    st.plotly_chart(fig, use_container_width=True)

    if static:
        st.info("Auto-refresh is disabled. Toggle the checkbox above to stream live data.")
