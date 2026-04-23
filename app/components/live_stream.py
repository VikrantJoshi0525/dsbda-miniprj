import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import COLORS, SENTIMENT_POS_THRESHOLD, SENTIMENT_NEG_THRESHOLD
from utils.reddit_live import fetch_live_reddit_posts
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter

# Initialize VADER globally for performance
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    comp = scores['compound']
    if comp >= SENTIMENT_POS_THRESHOLD:
        return 'Positive', comp
    elif comp <= SENTIMENT_NEG_THRESHOLD:
        return 'Negative', comp
    else:
        return 'Neutral', comp

@st.fragment(run_every="5s")
def render_live_dashboard(subreddit):
    """
    Renders the live auto-refreshing dashboard.
    """
    # Fetch live data
    posts = fetch_live_reddit_posts(subreddit=subreddit, limit=30)
    
    if not posts:
        st.warning(f"No live data available for r/{subreddit} right now.")
        return

    # Process sentiment
    processed_data = []
    for p in posts:
        label, score = get_sentiment(p["text"])
        processed_data.append({
            "id": p["id"],
            "timestamp": p["timestamp"],
            "text": p["text"],
            "author": p["author"],
            "subreddit": p["subreddit"],
            "sentiment_label": label,
            "sentiment_score": score
        })
        
        # Unique Feature: Alert System
        if score <= -0.8:
            st.toast(f"**High Negative Alert** in r/{subreddit}:\n{p['text'][:50]}...")
        elif score >= 0.8:
            st.toast(f"**High Positive Alert** in r/{subreddit}:\n{p['text'][:50]}...")

    df = pd.DataFrame(processed_data)
    
    # KPIs
    st.markdown("### Live Pulse")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    avg_score = df["sentiment_score"].mean()
    pos_count = len(df[df["sentiment_label"] == "Positive"])
    neg_count = len(df[df["sentiment_label"] == "Negative"])
    neu_count = len(df[df["sentiment_label"] == "Neutral"])
    
    # Use delta to show general trend dynamically
    kpi_col1.metric("Live Avg Sentiment", f"{avg_score:.2f}")
    kpi_col2.metric("Positive Posts", pos_count)
    kpi_col3.metric("Negative Posts", neg_count)
    kpi_col4.metric("Neutral Posts", neu_count)
    
    # Layout for charts
    col_chart, col_trends = st.columns([2, 1], gap="large")
    
    with col_chart:
        # Live Trend Chart
        st.markdown("### Live Sentiment Trend (Anomaly Detection)")
        # Reverse to have oldest to newest for the line chart
        chart_df = df.sort_values("timestamp").copy()
        
        # Calculate Rolling Statistics for Anomaly Detection
        window_size = min(10, len(chart_df))
        if window_size > 2:
            chart_df['rolling_mean'] = chart_df['sentiment_score'].rolling(window=window_size, min_periods=1).mean()
            chart_df['rolling_std'] = chart_df['sentiment_score'].rolling(window=window_size, min_periods=1).std().fillna(0)
            chart_df['upper_band'] = chart_df['rolling_mean'] + (2 * chart_df['rolling_std'])
            chart_df['lower_band'] = chart_df['rolling_mean'] - (2 * chart_df['rolling_std'])
            
            # Detect Anomalies
            chart_df['is_anomaly'] = (chart_df['sentiment_score'] > chart_df['upper_band']) | (chart_df['sentiment_score'] < chart_df['lower_band'])
        else:
            chart_df['rolling_mean'] = chart_df['sentiment_score']
            chart_df['upper_band'] = chart_df['sentiment_score']
            chart_df['lower_band'] = chart_df['sentiment_score']
            chart_df['is_anomaly'] = False
        
        fig = go.Figure()

        # Add Confidence Bands
        fig.add_trace(go.Scatter(
            x=chart_df["timestamp"], y=chart_df["upper_band"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=chart_df["timestamp"], y=chart_df["lower_band"],
            line=dict(width=0), fill="tonexty", fillcolor="rgba(108, 92, 231, 0.15)",
            showlegend=False, hoverinfo="skip", name="Normal Range"
        ))
        
        # Add Actual Sentiment Line
        fig.add_trace(go.Scatter(
            x=chart_df["timestamp"], y=chart_df["sentiment_score"],
            mode="lines", line=dict(color=COLORS["accent"], width=3),
            name="Sentiment Score"
        ))
        
        # Add Anomaly Markers
        anomalies = chart_df[chart_df["is_anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["timestamp"], y=anomalies["sentiment_score"],
                mode="markers", marker=dict(color="red", size=10, symbol="x"),
                name="Anomaly Detected"
            ))
        
        # Add coloring based on sentiment
        fig.add_hrect(y0=SENTIMENT_POS_THRESHOLD, y1=1, line_width=0, fillcolor=COLORS["positive"], opacity=0.05)
        fig.add_hrect(y0=-1, y1=SENTIMENT_NEG_THRESHOLD, line_width=0, fillcolor=COLORS["negative"], opacity=0.05)
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text"],
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="Compound Score",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col_trends:
        st.markdown("### 🔥 Trending Entities")
        # Extract noun phrases using TextBlob
        all_text = " ".join(df["text"].tolist())
        blob = TextBlob(all_text)
        # Simple filter: length > 3 and alphabetic, avoiding basic stop words if they slip through
        nouns = [n.lower() for n in blob.noun_phrases if len(n) > 3 and not n.isnumeric()]
        top_nouns = Counter(nouns).most_common(6)
        
        if top_nouns:
            for noun, count in top_nouns:
                st.markdown(
                    f"""
                    <div style="background:{COLORS['surface']}; padding:0.8rem; margin-bottom:0.5rem; border-radius:8px; border-left: 4px solid {COLORS['secondary']};">
                        <span style="color:{COLORS['text']}; font-weight:bold;">#{noun.replace(' ', '')}</span>
                        <span style="color:{COLORS['neutral']}; float:right;">{count} mentions</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No significant trends detected right now.")
            
    # Live Feed Table
    st.markdown("### Live Feed")
    
    # Custom display for posts
    for _, row in df.head(10).iterrows():
        bg_color = COLORS["positive"] + "22" if row["sentiment_label"] == "Positive" else (
            COLORS["negative"] + "22" if row["sentiment_label"] == "Negative" else COLORS["surface"]
        )
        border_color = COLORS["positive"] if row["sentiment_label"] == "Positive" else (
            COLORS["negative"] if row["sentiment_label"] == "Negative" else COLORS["neutral"]
        )
        
        st.markdown(
            f"""
            <div style="background:{bg_color}; padding:1rem; margin-bottom:1rem; border-radius:12px; border-left: 5px solid {border_color};">
                <div style="color:{COLORS['text']}88; font-size:0.85rem; margin-bottom:0.5rem;">
                    <b>u/{row['author']}</b> • {row['timestamp'].strftime('%H:%M:%S')} • 
                    <span style="color:{border_color}; font-weight:bold;">{row['sentiment_label']}</span> ({row['sentiment_score']:.2f})
                </div>
                <div style="color:{COLORS['text']}; font-size:1rem; line-height:1.4;">
                    {row['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_live_stream(filters):
    """
    Main entry for the Live Stream page.
    """
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin:0; font-size:2.5rem; background: -webkit-linear-gradient(45deg, {COLORS['primary']}, {COLORS['accent']}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Live Social Media Stream
            </h1>
            <p style="color:{COLORS['text']}aa; font-size:1.1rem; margin-top:0.5rem;">
                Real-time sentiment tracking, trend alerts, and entity extraction from a live firehose.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    subreddit = filters.get("live_subreddit", "technology")
    
    st.info(f"🟢 **Connected to Live Firehose:** Polling `r/{subreddit}` for live updates...", icon="🟢")
    
    # Call the auto-refreshing fragment
    render_live_dashboard(subreddit)
