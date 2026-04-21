"""
Dashboard page — enhanced with tabs: Overview, Keywords & WordCloud,
Engagement & Topics, and Data Export.
"""

import streamlit as st
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLORS
from utils.helpers import format_number
from utils.constants import SENTIMENT_ICONS, PLATFORM_ICONS
from components.visualizations import (
    sentiment_pie,
    sentiment_over_time,
    platform_bar,
    score_histogram,
    topic_heatmap,
    engagement_scatter,
    top_keywords_bar,
    top_keywords_by_sentiment,
    generate_wordcloud_image,
    topic_treemap,
    hourly_heatmap,
)


def _kpi_card_html(icon: str, label: str, value: str, delta: str = "", color: str = "") -> str:
    """Generate HTML for a single glassmorphism KPI card."""
    delta_html = ""
    if delta:
        delta_color = COLORS["positive"] if "+" in delta or "↑" in delta else COLORS["negative"]
        delta_html = f'<span style="font-size:0.85rem; color:{delta_color}; font-weight:600;">{delta}</span>'

    accent = color or COLORS["primary"]
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(26,26,46,0.7), rgba(22,33,62,0.5));
        backdrop-filter: blur(16px);
        border: 1px solid {accent}33;
        border-radius: 18px;
        padding: 1.5rem 1.4rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    ">
        <div style="font-size:2rem; margin-bottom:0.3rem;">{icon}</div>
        <div style="
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {COLORS['text']}88;
            font-weight: 600;
            margin-bottom: 0.4rem;
        ">{label}</div>
        <div style="
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, {accent}, {COLORS['secondary']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
        ">{value}</div>
        {delta_html}
    </div>
    """


def render_dashboard(df: pd.DataFrame):
    """Render the full enhanced dashboard page."""

    # ── Hero Header ──────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="margin:0; font-size:2.2rem;">Social Media Sentiment</h1>
            <p style="color:{COLORS['text']}77; font-size:1rem; margin:0.3rem 0 0;">
                Real-time insights from {format_number(len(df))} posts across multiple platforms
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards ────────────────────────────────────────────
    total = len(df)
    pos_count = len(df[df["sentiment_label"].str.lower() == "positive"])
    neg_count = len(df[df["sentiment_label"].str.lower() == "negative"])
    neu_count = len(df[df["sentiment_label"].str.lower() == "neutral"])
    avg_score = df["sentiment_score"].mean() if total > 0 else 0.0
    platforms_count = df["platform"].nunique() if "platform" in df.columns else 0
    topics_count = df["topic"].nunique() if "topic" in df.columns else 0

    if platforms_count > 1:
        fifth_card = ("📊", "Avg Score", f"{avg_score:+.3f}", f"{platforms_count} platforms", COLORS["secondary"])
    else:
        fifth_card = ("📌", "Topics", str(topics_count), f"Avg: {avg_score:+.3f}", COLORS["secondary"])

    cols = st.columns(5, gap="medium")
    cards = [
        ("📝", "Total Posts",   format_number(total),     "",                                       COLORS["primary"]),
        ("😊", "Positive",      format_number(pos_count), f"↑ {pos_count*100//max(total,1)}%",      COLORS["positive"]),
        ("😠", "Negative",      format_number(neg_count), f"↓ {neg_count*100//max(total,1)}%",      COLORS["negative"]),
        ("😐", "Neutral",       format_number(neu_count), f"  {neu_count*100//max(total,1)}%",      COLORS["neutral"]),
        fifth_card,
    ]
    for col, (icon, label, value, delta, color) in zip(cols, cards):
        col.markdown(_kpi_card_html(icon, label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ── Tabbed Dashboard ─────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    tab_overview, tab_keywords, tab_engage, tab_data = st.tabs([
        "📊 Overview & Trends",
        "🔤 Keywords & WordCloud",
        "📈 Engagement & Topics",
        "📋 Data Export",
    ])

    # ── TAB 1: Overview & Trends ─────────────────────────────
    with tab_overview:
        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            st.plotly_chart(sentiment_pie(df), use_container_width=True)
        with c2:
            st.plotly_chart(sentiment_over_time(df), use_container_width=True)

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            st.plotly_chart(score_histogram(df), use_container_width=True)
        with c4:
            if platforms_count > 1:
                st.plotly_chart(platform_bar(df), use_container_width=True)
            else:
                st.plotly_chart(hourly_heatmap(df), use_container_width=True)

    # ── TAB 2: Keywords & WordCloud ──────────────────────────
    with tab_keywords:
        st.markdown(
            f"""
            <div style="margin-bottom:1rem;">
                <h3 style="margin:0; color:{COLORS['text']};">🔤 Keyword Analysis</h3>
                <p style="color:{COLORS['text']}77; font-size:0.85rem; margin:0.2rem 0 0;">
                    Most frequent terms extracted from post text (stop-words removed)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top keywords overall
        kw1, kw2 = st.columns(2, gap="medium")
        with kw1:
            text_col = "clean_text" if "clean_text" in df.columns else "text"
            st.plotly_chart(
                top_keywords_bar(df, text_col=text_col, top_n=20, title="Top 20 Keywords (All Posts)"),
                use_container_width=True,
            )
        with kw2:
            st.plotly_chart(
                top_keywords_by_sentiment(df, text_col=text_col, top_n=12),
                use_container_width=True,
            )

        # Word Clouds
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="margin-bottom:1rem;">
                <h3 style="margin:0; color:{COLORS['text']};">☁️ Word Clouds</h3>
                <p style="color:{COLORS['text']}77; font-size:0.85rem; margin:0.2rem 0 0;">
                    Visual representation of word frequency — larger words appear more often
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        wc_all, wc_pos, wc_neg = st.columns(3, gap="medium")

        for col, (sentiment_filter, label, border_clr) in zip(
            [wc_all, wc_pos, wc_neg],
            [(None, "All Posts", COLORS["primary"]),
             ("Positive", "Positive Posts", COLORS["positive"]),
             ("Negative", "Negative Posts", COLORS["negative"])],
        ):
            with col:
                st.markdown(
                    f'<p style="text-align:center; font-weight:600; color:{border_clr}; '
                    f'font-size:0.85rem; margin-bottom:0.4rem;">{label}</p>',
                    unsafe_allow_html=True,
                )
                wc_bytes = generate_wordcloud_image(
                    df, text_col=text_col, max_words=120,
                    sentiment_filter=sentiment_filter,
                )
                if wc_bytes:
                    st.image(wc_bytes, use_container_width=True)
                else:
                    st.info("Install `wordcloud` package for word cloud visualisation.")

    # ── TAB 3: Engagement & Topics ───────────────────────────
    with tab_engage:
        e1, e2 = st.columns(2, gap="medium")
        with e1:
            st.plotly_chart(topic_heatmap(df), use_container_width=True)
        with e2:
            st.plotly_chart(topic_treemap(df), use_container_width=True)

        e3, e4 = st.columns(2, gap="medium")
        with e3:
            if all(c in df.columns for c in ["likes", "shares", "user_followers"]):
                st.plotly_chart(engagement_scatter(df), use_container_width=True)
            else:
                st.info("Engagement data (likes, shares) not available in this dataset.")
        with e4:
            st.plotly_chart(hourly_heatmap(df), use_container_width=True)

    # ── TAB 4: Data Export ───────────────────────────────────
    with tab_data:
        st.markdown(
            f"""
            <div style="margin-bottom:1rem;">
                <h3 style="margin:0; color:{COLORS['text']};">📋 Data Preview & Export</h3>
                <p style="color:{COLORS['text']}77; font-size:0.85rem; margin:0.2rem 0 0;">
                    Browse the processed dataset and download as CSV
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_cols = [
            "timestamp", "platform", "topic", "text", "sentiment_label",
            "sentiment_score", "likes", "shares",
        ]
        available_cols = [c for c in display_cols if c in df.columns]

        # Summary stats
        stat1, stat2, stat3, stat4 = st.columns(4, gap="medium")
        stat1.metric("Rows", f"{len(df):,}")
        stat2.metric("Columns", len(available_cols))
        stat3.metric("Pos/Neg Ratio", f"{pos_count / max(neg_count, 1):.2f}")
        stat4.metric("Avg Score", f"{avg_score:+.3f}")

        st.dataframe(
            df[available_cols].head(200),
            use_container_width=True,
            height=450,
        )

        csv = df[available_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Full CSV",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
