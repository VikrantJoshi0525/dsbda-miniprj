"""
🔮 Predict page — custom tweet / text sentiment prediction.

Scores user-entered text using VADER, TextBlob, and (if trained) the Spark MLlib model.
Shows animated result cards and a sentiment gauge.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLORS


def _clean_input(text: str) -> str:
    """Basic text cleaning for prediction input."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z0-9\s!?.,']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _sentiment_gauge(score: float, title: str = "Sentiment Score") -> go.Figure:
    """Create a sleek gauge chart for sentiment polarity."""
    if score > 0.05:
        bar_color = COLORS["positive"]
    elif score < -0.05:
        bar_color = COLORS["negative"]
    else:
        bar_color = COLORS["neutral"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            font=dict(size=40, color=COLORS["text"], family="Inter"),
            valueformat="+.4f",
        ),
        gauge=dict(
            axis=dict(
                range=[-1, 1],
                tickwidth=2,
                tickcolor=COLORS["text"],
                tickfont=dict(size=11, color=f"{COLORS['text']}88"),
            ),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[-1, -0.05], color=f"{COLORS['negative']}20"),
                dict(range=[-0.05, 0.05], color=f"{COLORS['neutral']}20"),
                dict(range=[0.05, 1], color=f"{COLORS['positive']}20"),
            ],
            threshold=dict(
                line=dict(color=COLORS["accent"], width=3),
                thickness=0.8,
                value=score,
            ),
        ),
        title=dict(
            text=title,
            font=dict(size=14, color=f"{COLORS['text']}aa"),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=30, r=30, t=60, b=20),
        height=250,
    )
    return fig


def _result_card(label: str, score: float, engine: str, emoji: str) -> str:
    """Generate a glassmorphism result card."""
    if label.lower() == "positive":
        clr = COLORS["positive"]
    elif label.lower() == "negative":
        clr = COLORS["negative"]
    else:
        clr = COLORS["neutral"]

    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(26,26,46,0.7), rgba(22,33,62,0.5));
        backdrop-filter: blur(16px);
        border: 1px solid {clr}44;
        border-left: 4px solid {clr};
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    ">
        <div style="font-size:2.5rem; margin-bottom:0.3rem;">{emoji}</div>
        <div style="
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {COLORS['text']}88;
            font-weight: 600;
            margin-bottom: 0.3rem;
        ">{engine}</div>
        <div style="
            font-size: 1.4rem;
            font-weight: 800;
            color: {clr};
            margin-bottom: 0.2rem;
        ">{label}</div>
        <div style="
            font-size: 1.1rem;
            font-weight: 600;
            background: linear-gradient(135deg, {clr}, {COLORS['secondary']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">{score:+.4f}</div>
    </div>
    """


def _classify(score: float) -> tuple[str, str]:
    """Return (label, emoji) from score."""
    if score > 0.05:
        return "Positive", "😊"
    elif score < -0.05:
        return "Negative", "😠"
    return "Neutral", "😐"


def render_predict_page():
    """Render the custom tweet prediction page."""

    # ── Header ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin:0; font-size:2.2rem;">Custom Tweet Predictor</h1>
            <p style="color:{COLORS['text']}77; font-size:1rem; margin:0.3rem 0 0;">
                Enter any text to analyse its sentiment using multiple engines
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input Section ────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(26,26,46,0.6), rgba(22,33,62,0.4));
            border: 1px solid {COLORS['primary']}33;
            border-radius: 18px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
        ">
            <h3 style="margin:0 0 0.5rem; color:{COLORS['text']};">✍️ Enter Your Text</h3>
            <p style="color:{COLORS['text']}66; font-size:0.82rem; margin:0;">
                Type a tweet, review, or any social media text below
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Tweet / Text",
        height=120,
        placeholder="e.g. I absolutely love this new product! It's amazing and works perfectly 🚀",
        label_visibility="collapsed",
    )

    # Example tweets
    st.markdown(
        f"<p style='color:{COLORS['text']}66; font-size:0.78rem; margin:0.3rem 0 0.8rem;'>"
        f"💡 Try these examples:</p>",
        unsafe_allow_html=True,
    )
    ex1, ex2, ex3 = st.columns(3, gap="small")
    with ex1:
        if st.button("😊 Positive example", use_container_width=True, key="ex_pos"):
            st.session_state["predict_text"] = "This is absolutely wonderful! Best experience I've ever had. Highly recommend to everyone! 🌟"
            st.rerun()
    with ex2:
        if st.button("😠 Negative example", use_container_width=True, key="ex_neg"):
            st.session_state["predict_text"] = "Terrible customer service, waited 3 hours and nobody helped. Completely disappointed and frustrated."
            st.rerun()
    with ex3:
        if st.button("😐 Neutral example", use_container_width=True, key="ex_neu"):
            st.session_state["predict_text"] = "Just read an article about the latest developments in AI technology. Interesting perspective."
            st.rerun()

    # Use stored text from example buttons
    if "predict_text" in st.session_state and not user_text:
        user_text = st.session_state.pop("predict_text")

    predict_clicked = st.button(
        "🔮  Analyse Sentiment",
        use_container_width=True,
        type="primary",
        disabled=not user_text.strip(),
    )

    # ── Results ──────────────────────────────────────────────
    if predict_clicked and user_text.strip():
        _run_prediction(user_text.strip())
    elif "predict_results" in st.session_state:
        _display_prediction_results(st.session_state["predict_results"])


def _run_prediction(raw_text: str):
    """Run sentiment prediction using VADER, TextBlob, and optionally the ML model."""
    cleaned = _clean_input(raw_text)

    results = {
        "raw_text": raw_text,
        "clean_text": cleaned,
        "engines": [],
    }

    # ── VADER ────────────────────────────────────────────────
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        vader_scores = vader.polarity_scores(cleaned)
        compound = vader_scores["compound"]
        label, emoji = _classify(compound)
        results["engines"].append({
            "name": "VADER",
            "score": compound,
            "label": label,
            "emoji": emoji,
            "details": vader_scores,
        })
    except ImportError:
        pass

    # ── TextBlob ─────────────────────────────────────────────
    try:
        from textblob import TextBlob
        blob = TextBlob(cleaned)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        label, emoji = _classify(polarity)
        results["engines"].append({
            "name": "TextBlob",
            "score": polarity,
            "label": label,
            "emoji": emoji,
            "details": {"polarity": polarity, "subjectivity": subjectivity},
        })
    except ImportError:
        pass

    # ── Spark MLlib (if model trained) ───────────────────────
    if "ml_result" in st.session_state and st.session_state.ml_result.model is not None:
        try:
            from spark.session import get_spark
            spark = get_spark()
            pred_df = spark.createDataFrame(
                [{"clean_text": cleaned, "sentiment_label": "Neutral"}]
            )
            model = st.session_state.ml_result.model
            prediction = model.transform(pred_df)

            # Get label mapping
            indexer_model = model.stages[4]
            label_order = list(indexer_model.labels)

            row = prediction.collect()[0]
            pred_idx = int(row["prediction"])
            pred_label = label_order[pred_idx] if pred_idx < len(label_order) else "Neutral"

            # Get probability
            prob_vector = row["probability"]
            pred_prob = float(prob_vector[pred_idx]) if pred_idx < len(prob_vector) else 0.5

            # Map to score-like value
            if pred_label.lower() == "positive":
                synth_score = pred_prob * 0.8
            elif pred_label.lower() == "negative":
                synth_score = -pred_prob * 0.8
            else:
                synth_score = 0.0

            _, emoji = _classify(synth_score)

            results["engines"].append({
                "name": "Spark MLlib (LR)",
                "score": synth_score,
                "label": pred_label,
                "emoji": emoji,
                "details": {
                    "predicted_class": pred_label,
                    "confidence": f"{pred_prob:.1%}",
                    "probabilities": {label_order[i]: f"{float(prob_vector[i]):.4f}" for i in range(len(label_order))},
                },
            })
        except Exception as e:
            results["engines"].append({
                "name": "Spark MLlib (LR)",
                "score": 0.0,
                "label": "Error",
                "emoji": "⚠️",
                "details": {"error": str(e)},
            })

    st.session_state["predict_results"] = results
    _display_prediction_results(results)


def _display_prediction_results(results: dict):
    """Render the prediction results."""
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Cleaned Text Preview ─────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: {COLORS['primary']}10;
            border: 1px solid {COLORS['primary']}25;
            border-radius: 14px;
            padding: 1rem 1.3rem;
            margin-bottom: 1.5rem;
        ">
            <div style="color:{COLORS['text']}88; font-size:0.75rem; text-transform:uppercase;
                        letter-spacing:0.08em; font-weight:600; margin-bottom:0.3rem;">Input Text</div>
            <div style="color:{COLORS['text']}; font-size:0.95rem; line-height:1.5;">
                "{results['raw_text']}"
            </div>
            <div style="color:{COLORS['text']}55; font-size:0.75rem; margin-top:0.4rem;">
                Cleaned → <i>{results['clean_text'][:120]}{'…' if len(results['clean_text']) > 120 else ''}</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    engines = results.get("engines", [])

    if not engines:
        st.warning("No sentiment engines available. Install `vaderSentiment` and `textblob`.")
        return

    # ── Result Cards Row ─────────────────────────────────────
    cols = st.columns(len(engines), gap="medium")
    for col, eng in zip(cols, engines):
        col.markdown(
            _result_card(eng["label"], eng["score"], eng["name"], eng["emoji"]),
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Gauge Charts ─────────────────────────────────────────
    gauge_cols = st.columns(len(engines), gap="medium")
    for col, eng in zip(gauge_cols, engines):
        with col:
            st.plotly_chart(
                _sentiment_gauge(eng["score"], title=eng["name"]),
                use_container_width=True,
            )

    # ── Detailed Breakdown ───────────────────────────────────
    with st.expander("📊 Detailed Engine Output", expanded=False):
        for eng in engines:
            st.markdown(f"**{eng['name']}**")
            details = eng.get("details", {})
            if isinstance(details, dict):
                for k, v in details.items():
                    if isinstance(v, dict):
                        st.json(v)
                    else:
                        st.markdown(f"- **{k}**: `{v}`")
            st.markdown("---")

    # ── Consensus ────────────────────────────────────────────
    scores = [e["score"] for e in engines if e["label"] != "Error"]
    if scores:
        avg = sum(scores) / len(scores)
        consensus_label, consensus_emoji = _classify(avg)

        if consensus_label.lower() == "positive":
            consensus_clr = COLORS["positive"]
        elif consensus_label.lower() == "negative":
            consensus_clr = COLORS["negative"]
        else:
            consensus_clr = COLORS["neutral"]

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {consensus_clr}15, {COLORS['primary']}08);
                border: 2px solid {consensus_clr}44;
                border-radius: 18px;
                padding: 1.5rem;
                text-align: center;
                margin-top: 0.5rem;
            ">
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em;
                            color:{COLORS['text']}77; font-weight:600; margin-bottom:0.4rem;">
                    Consensus ({len(scores)} engines)
                </div>
                <div style="font-size:3rem; margin-bottom:0.3rem;">{consensus_emoji}</div>
                <div style="
                    font-size: 1.8rem;
                    font-weight: 800;
                    color: {consensus_clr};
                    margin-bottom: 0.2rem;
                ">{consensus_label}</div>
                <div style="
                    font-size: 1rem;
                    font-weight: 600;
                    color: {COLORS['text']}aa;
                ">Average Score: {avg:+.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
