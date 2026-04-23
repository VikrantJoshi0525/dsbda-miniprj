"""
Analysis page — deep-dive filtering, keyword search, and detailed breakdowns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLORS
from utils.constants import SENTIMENT_COLORS


def render_analysis(df: pd.DataFrame):
    """Render the deep-dive analysis page."""

    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin:0; font-size:2rem;">Deep-Dive Analysis</h1>
            <p style="color:{COLORS['text']}77; font-size:0.95rem; margin:0.3rem 0 0;">
                Filter, search, and explore sentiment patterns
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Filters Row ──────────────────────────────────────────
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        keyword = st.text_input("Keyword Search", placeholder="e.g. AI, climate…")
    with f2:
        topic_filter = st.multiselect(
            "Topic",
            options=sorted(df["topic"].unique()),
            default=[],
        )
    with f3:
        sort_by = st.selectbox(
            "Sort By",
            options=["Most Recent", "Highest Score", "Lowest Score", "Most Liked", "Most Shared"],
        )

    # ── Apply Filters ────────────────────────────────────────
    filtered = df.copy()

    if keyword:
        filtered = filtered[
            filtered["text"].str.contains(keyword, case=False, na=False)
        ]

    if topic_filter:
        filtered = filtered[filtered["topic"].isin(topic_filter)]

    # Sort
    sort_map = {
        "Most Recent":   ("timestamp", False),
        "Highest Score":  ("sentiment_score", False),
        "Lowest Score":   ("sentiment_score", True),
        "Most Liked":     ("likes", False),
        "Most Shared":    ("shares", False),
    }
    sort_col, ascending = sort_map.get(sort_by, ("timestamp", False))
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    # ── Results Summary bar ──────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['primary']}15, {COLORS['secondary']}10);
            border: 1px solid {COLORS['primary']}33;
            border-radius: 14px;
            padding: 1rem 1.5rem;
            margin: 1rem 0 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span style="font-weight:600; color:{COLORS['text']};">
                📄 {len(filtered):,} results found
            </span>
            <span style="color:{COLORS['text']}88; font-size:0.85rem;">
                {len(filtered[filtered['sentiment_label'].str.lower()=='positive'])} positive ·
                {len(filtered[filtered['sentiment_label'].str.lower()=='negative'])} negative ·
                {len(filtered[filtered['sentiment_label'].str.lower()=='neutral'])} neutral
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────
    tab_cards, tab_table, tab_topics, tab_graph = st.tabs(["Cards", "Table", "Topic Breakdown", "Knowledge Graph"])

    # ── Cards Tab ────────────────────────────────────────────
    with tab_cards:
        if filtered.empty:
            st.info("No results match your filters.")
        else:
            for _, row in filtered.head(20).iterrows():
                label = row["sentiment_label"]
                clr = SENTIMENT_COLORS.get(label, COLORS["neutral"])
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(26,26,46,0.6), rgba(22,33,62,0.4));
                        border-left: 4px solid {clr};
                        border-radius: 12px;
                        padding: 1rem 1.3rem;
                        margin-bottom: 0.7rem;
                        backdrop-filter: blur(10px);
                    ">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <span style="
                                background: {clr}22;
                                color: {clr};
                                padding: 0.2rem 0.6rem;
                                border-radius: 20px;
                                font-size: 0.75rem;
                                font-weight: 700;
                                text-transform: uppercase;
                            ">{label} ({row['sentiment_score']:+.3f})</span>
                            <span style="color:{COLORS['text']}55; font-size:0.75rem;">
                                {row.get('platform', '')} · {row.get('topic', '')}
                            </span>
                        </div>
                        <p style="color:{COLORS['text']}; margin:0; font-size:0.92rem; line-height:1.5;">
                            {row['text']}
                        </p>
                        <div style="margin-top:0.5rem; color:{COLORS['text']}55; font-size:0.75rem;">
                            {row.get('likes', 0):,} · {row.get('shares', 0):,} · {row.get('user_followers', 0):,} followers
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Table Tab ────────────────────────────────────────────
    with tab_table:
        display_cols = ["timestamp", "platform", "topic", "text", "sentiment_label", "sentiment_score", "likes", "shares"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols], use_container_width=True, height=500)

    # ── Topic Breakdown Tab ──────────────────────────────────
    with tab_topics:
        if filtered.empty:
            st.info("No data available for topic breakdown.")
        else:
            topic_stats = (
                filtered.groupby("topic")
                .agg(
                    count=("text", "size"),
                    avg_score=("sentiment_score", "mean"),
                    total_likes=("likes", "sum"),
                )
                .reset_index()
                .sort_values("count", ascending=False)
            )

            fig = px.bar(
                topic_stats,
                x="topic",
                y="count",
                color="avg_score",
                color_continuous_scale=[COLORS["negative"], COLORS["neutral"], COLORS["positive"]],
                text="count",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=COLORS["text"]),
                margin=dict(l=40, r=20, t=50, b=40),
                title="Posts per Topic (coloured by avg sentiment)",
                xaxis_title="",
                yaxis_title="Post Count",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                topic_stats.rename(columns={
                    "topic": "Topic",
                    "count": "Posts",
                    "avg_score": "Avg Score",
                    "total_likes": "Total Likes",
                }),
                use_container_width=True,
            )
            
    # ── Knowledge Graph Tab ──────────────────────────────────
    with tab_graph:
        if filtered.empty:
            st.info("No data available for knowledge graph.")
        else:
            from components.visualizations import semantic_network_graph
            with st.spinner("Generating Semantic Network..."):
                fig_graph = semantic_network_graph(filtered, text_col="text", top_n=40)
                st.plotly_chart(fig_graph, use_container_width=True)
                
            st.markdown(
                f"""
                <div style="background:{COLORS['primary']}10; border:1px solid {COLORS['primary']}25; padding:1rem; border-radius:12px; margin-top:1rem; font-size:0.85rem; color:{COLORS['text']}aa;">
                    <b>How to read this graph:</b> Nodes represent frequently used words. Edges (lines) connect words that frequently appear together in the same post. Larger nodes indicate higher frequency. This helps visualize the underlying narrative structure of the dataset.
                </div>
                """, unsafe_allow_html=True
            )
