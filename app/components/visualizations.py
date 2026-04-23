"""
Reusable Plotly / Matplotlib chart components.
Includes: pie, trends, bars, histogram, heatmap, scatter, keywords, wordcloud.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
import re
import io

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLORS


# ── Plotly defaults ─────────────────────────────────────────────
_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
)

_COLOR_MAP = {
    "Positive": COLORS["positive"],
    "Negative": COLORS["negative"],
    "Neutral":  COLORS["neutral"],
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral":  COLORS["neutral"],
}

# Common English stop words for keyword extraction
_STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year",
    "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most",
    "us", "im", "dont", "ive", "cant", "got", "was", "am", "are", "is",
    "has", "had", "been", "were", "being", "did", "does", "did", "doing",
    "really", "very", "much", "still", "going", "too", "more", "here",
    "right", "oh", "yeah", "yes", "lol", "haha", "gonna", "wanna", "gotta",
}


# ═══════════════════════════════════════════════════════════════
#  Existing charts
# ═══════════════════════════════════════════════════════════════

def sentiment_pie(df: pd.DataFrame, label_col: str = "sentiment_label"):
    """Donut chart of sentiment distribution."""
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        hole=0.55,
        color="Sentiment",
        color_discrete_map=_COLOR_MAP,
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Sentiment Distribution",
        showlegend=True,
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=13,
        marker=dict(line=dict(color=COLORS["background"], width=2)),
    )
    return fig


def sentiment_over_time(df: pd.DataFrame, date_col: str = "timestamp", label_col: str = "sentiment_label"):
    """Area chart: sentiment counts by day."""
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.date
    daily = (
        df.groupby(["date", label_col])
        .size()
        .reset_index(name="count")
    )
    daily.columns = ["Date", "Sentiment", "Count"]
    fig = px.area(
        daily,
        x="Date",
        y="Count",
        color="Sentiment",
        color_discrete_map=_COLOR_MAP,
        line_shape="spline",
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Sentiment Trend Over Time",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def platform_bar(df: pd.DataFrame, platform_col: str = "platform", label_col: str = "sentiment_label"):
    """Grouped bar chart: sentiment breakdown by platform."""
    grouped = (
        df.groupby([platform_col, label_col])
        .size()
        .reset_index(name="count")
    )
    grouped.columns = ["Platform", "Sentiment", "Count"]
    fig = px.bar(
        grouped,
        x="Platform",
        y="Count",
        color="Sentiment",
        barmode="group",
        color_discrete_map=_COLOR_MAP,
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Sentiment by Platform",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    fig.update_traces(
        marker_line_width=0,
        opacity=0.9,
    )
    return fig


def score_histogram(df: pd.DataFrame, score_col: str = "sentiment_score"):
    """Histogram of sentiment scores with gradient fill."""
    fig = px.histogram(
        df,
        x=score_col,
        nbins=40,
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Sentiment Score Distribution",
        xaxis_title="Polarity Score",
        yaxis_title="Frequency",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    fig.update_traces(
        marker_line_width=0,
        opacity=0.85,
    )
    return fig


def topic_heatmap(df: pd.DataFrame, topic_col: str = "topic", label_col: str = "sentiment_label"):
    """Heatmap: sentiment counts per topic."""
    pivot = pd.crosstab(df[topic_col], df[label_col])
    for col in ["positive", "negative", "neutral", "Positive", "Negative", "Neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0, COLORS["negative"]],
            [0.5, COLORS["neutral"]],
            [1, COLORS["positive"]],
        ],
        hovertemplate="Topic: %{y}<br>Sentiment: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Topic × Sentiment Heatmap",
        xaxis_title="Sentiment",
        yaxis_title="Topic",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def engagement_scatter(df: pd.DataFrame):
    """Scatter: likes vs shares coloured by sentiment."""
    fig = px.scatter(
        df,
        x="likes",
        y="shares",
        color="sentiment_label",
        color_discrete_map=_COLOR_MAP,
        size="user_followers",
        size_max=18,
        opacity=0.7,
        hover_data=["platform", "topic"],
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Engagement vs Sentiment",
        xaxis_title="Likes",
        yaxis_title="Shares",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  NEW: Keyword & WordCloud charts
# ═══════════════════════════════════════════════════════════════

def _extract_words(df: pd.DataFrame, text_col: str = "text") -> list[str]:
    """Extract all words from the text column, lowercased and filtered."""
    all_text = " ".join(df[text_col].dropna().astype(str))
    words = re.findall(r"[a-zA-Z]{3,}", all_text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def top_keywords_bar(
    df: pd.DataFrame,
    text_col: str = "text",
    top_n: int = 20,
    title: str = "Top Keywords",
):
    """Horizontal bar chart of most frequent keywords."""
    words = _extract_words(df, text_col)
    counts = Counter(words).most_common(top_n)

    if not counts:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title=title)
        return fig

    kw_df = pd.DataFrame(counts, columns=["Keyword", "Count"])
    kw_df = kw_df.sort_values("Count", ascending=True)

    fig = px.bar(
        kw_df,
        x="Count",
        y="Keyword",
        orientation="h",
        color="Count",
        color_continuous_scale=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]],
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=title,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False,
        height=500,
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    return fig


def top_keywords_by_sentiment(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "sentiment_label",
    top_n: int = 10,
):
    """Side-by-side keyword bars for Positive vs Negative."""
    results = []
    for label in ["Positive", "Negative", "positive", "negative"]:
        subset = df[df[label_col].str.lower() == label.lower()]
        if subset.empty:
            continue
        words = _extract_words(subset, text_col)
        for word, count in Counter(words).most_common(top_n):
            results.append({"Keyword": word, "Count": count, "Sentiment": label.capitalize()})

    if not results:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="Keywords by Sentiment")
        return fig

    kw_df = pd.DataFrame(results)

    fig = px.bar(
        kw_df,
        x="Count",
        y="Keyword",
        color="Sentiment",
        orientation="h",
        barmode="group",
        color_discrete_map=_COLOR_MAP,
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Top Keywords: Positive vs Negative",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=450,
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    return fig


def generate_wordcloud_image(
    df: pd.DataFrame,
    text_col: str = "text",
    max_words: int = 150,
    sentiment_filter: str | None = None,
    label_col: str = "sentiment_label",
) -> bytes | None:
    """
    Generate a word cloud PNG image and return as bytes.
    Returns None if wordcloud library is not available.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None

    subset = df
    if sentiment_filter:
        subset = df[df[label_col].str.lower() == sentiment_filter.lower()]

    words = _extract_words(subset, text_col)
    if not words:
        return None

    freq = Counter(words)

    # Choose colour based on sentiment filter
    if sentiment_filter and sentiment_filter.lower() == "positive":
        colormap = "YlGn"
    elif sentiment_filter and sentiment_filter.lower() == "negative":
        colormap = "OrRd"
    else:
        colormap = "cool"

    wc = WordCloud(
        width=900,
        height=450,
        max_words=max_words,
        background_color=COLORS["background"],
        colormap=colormap,
        contour_width=0,
        prefer_horizontal=0.75,
        min_font_size=8,
        max_font_size=80,
    ).generate_from_frequencies(freq)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def topic_treemap(df: pd.DataFrame, topic_col: str = "topic", label_col: str = "sentiment_label"):
    """Treemap of topics sized by count, coloured by average sentiment score."""
    if topic_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="Topic Treemap")
        return fig

    topic_stats = (
        df.groupby(topic_col)
        .agg(
            count=("sentiment_score", "size"),
            avg_score=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    fig = px.treemap(
        topic_stats,
        path=[topic_col],
        values="count",
        color="avg_score",
        color_continuous_scale=[COLORS["negative"], COLORS["neutral"], COLORS["positive"]],
        color_continuous_midpoint=0,
    )
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_layout(
        title="Topic Distribution (colour = avg sentiment)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def hourly_heatmap(df: pd.DataFrame, date_col: str = "timestamp", label_col: str = "sentiment_label"):
    """Heatmap: post volume by day-of-week × hour."""
    df = df.copy()
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df["hour"] = df["_dt"].dt.hour
    df["day"] = df["_dt"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.groupby(["day", "hour"]).size().reset_index(name="count")
    pivot_wide = pivot.pivot(index="day", columns="hour", values="count").fillna(0)
    pivot_wide = pivot_wide.reindex(day_order)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_wide.values,
        x=[f"{h}:00" for h in range(24)],
        y=pivot_wide.index.tolist(),
        colorscale=[
            [0.0, COLORS["background"]],
            [0.4, COLORS["primary"]],
            [0.7, COLORS["secondary"]],
            [1.0, COLORS["accent"]],
        ],
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Posts: %{z}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_layout(
        title="Posting Activity Heatmap",
        xaxis_title="Hour of Day",
        yaxis=dict(autorange="reversed"),
        height=350,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  NEW: Semantic Network Graph
# ═══════════════════════════════════════════════════════════════

def semantic_network_graph(df: pd.DataFrame, text_col: str = "text", top_n: int = 30):
    """Renders a Plotly Network Graph showing keyword co-occurrence."""
    try:
        import networkx as nx
    except ImportError:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="NetworkX not installed")
        return fig
        
    from collections import defaultdict
    import itertools
    
    # 1. Extract words per document
    doc_words = []
    for text in df[text_col].dropna().astype(str):
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        filtered = [w for w in set(words) if w not in _STOP_WORDS]
        if filtered:
            doc_words.append(filtered)
            
    # 2. Count frequencies and co-occurrences
    word_freq = Counter()
    co_occur = defaultdict(int)
    
    for words in doc_words:
        for w in words:
            word_freq[w] += 1
        for w1, w2 in itertools.combinations(sorted(words), 2):
            co_occur[(w1, w2)] += 1
            
    if not word_freq:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="Semantic Network Graph")
        return fig
        
    # 3. Get top nodes
    top_words = [w for w, c in word_freq.most_common(top_n)]
    
    # 4. Build graph
    G = nx.Graph()
    for w in top_words:
        G.add_node(w, size=word_freq[w])
        
    for (w1, w2), weight in co_occur.items():
        if w1 in top_words and w2 in top_words and weight > 1:
            G.add_edge(w1, w2, weight=weight)
            
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="Semantic Network Graph")
        return fig
        
    # 5. Position nodes using spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # 6. Create Plotly traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        sz = G.nodes[node]['size']
        node_size.append(min(max(sz * 2, 10), 50))
        node_text.append(f"{node} ({sz})")
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="bottom center",
        textfont=dict(color=COLORS["text"], size=10),
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=node_size,
            size=node_size,
            line_width=2,
            line_color=COLORS["background"]
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_layout(
        title="Semantic Knowledge Graph (Co-occurrence)",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig
