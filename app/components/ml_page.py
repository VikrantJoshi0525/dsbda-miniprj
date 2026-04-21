"""
ML Model page — Train, evaluate, and inspect a Spark MLlib LogisticRegression
sentiment classifier with a premium glassmorphism UI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLORS


def _glass_card(content_html: str, border_color: str = "", extra_style: str = "") -> str:
    """Wrap HTML in a glassmorphism card."""
    bc = border_color or f"{COLORS['primary']}33"
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(26,26,46,0.7), rgba(22,33,62,0.5));
        backdrop-filter: blur(16px);
        border: 1px solid {bc};
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        {extra_style}
    ">
        {content_html}
    </div>
    """


def _metric_pill(label: str, value: str, color: str) -> str:
    """Inline metric pill for header row."""
    return f"""
    <div style="
        background: {color}12;
        border: 1px solid {color}44;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        min-width: 130px;
    ">
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
                    color:{COLORS['text']}88; font-weight:600; margin-bottom:0.3rem;">{label}</div>
        <div style="font-size:1.6rem; font-weight:800;
                    background: linear-gradient(135deg, {color}, {COLORS['secondary']});
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    line-height:1.1;">{value}</div>
    </div>
    """


def render_ml_page(df: pd.DataFrame, data_source: str, csv_path: str):
    """Render the ML Model training & evaluation page."""

    # ── Header ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin:0; font-size:2.2rem;">ML Sentiment Classifier</h1>
            <p style="color:{COLORS['text']}77; font-size:1rem; margin:0.3rem 0 0;">
                Train a Spark MLlib Logistic Regression model on your data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Hyperparameter Controls ──────────────────────────────
    st.markdown(
        _glass_card(f"""
            <h3 style="margin:0 0 1rem; color:{COLORS['text']};">🎛️ Model Hyperparameters</h3>
            <p style="color:{COLORS['text']}77; font-size:0.85rem; margin:0 0 0.8rem;">
                Configure the Logistic Regression pipeline before training
            </p>
        """),
        unsafe_allow_html=True,
    )

    hp1, hp2, hp3, hp4 = st.columns(4, gap="medium")
    with hp1:
        test_ratio = st.slider(
            "Test Split %",
            min_value=10, max_value=40, value=20, step=5,
            help="Percentage of data held out for testing",
        ) / 100.0
    with hp2:
        num_features = st.select_slider(
            "HashingTF Features",
            options=[1000, 5000, 10000, 20000, 50000],
            value=10000,
            help="Vocabulary size for the bag-of-words representation",
        )
    with hp3:
        reg_param = st.select_slider(
            "Regularisation (λ)",
            options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            value=0.01,
            help="L2 regularisation strength (lower = less regularisation)",
        )
    with hp4:
        max_iter = st.slider(
            "Max Iterations",
            min_value=20, max_value=300, value=100, step=20,
            help="Maximum iterations for the optimiser",
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Training trigger ─────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        train_clicked = st.button(
            "🚀  Train Model",
            use_container_width=True,
            type="primary",
        )
    with col_info:
        if data_source == "demo":
            st.markdown(
                f"""<div style="
                    background: {COLORS['accent']}12;
                    border: 1px solid {COLORS['accent']}33;
                    border-radius: 12px;
                    padding: 0.7rem 1rem;
                    color: {COLORS['accent']};
                    font-size: 0.85rem;
                ">⚠️ Demo mode: training on synthetic data. Use Sentiment140 for meaningful results.</div>""",
                unsafe_allow_html=True,
            )
        else:
            row_count = len(df)
            st.markdown(
                f"""<div style="
                    background: {COLORS['positive']}12;
                    border: 1px solid {COLORS['positive']}33;
                    border-radius: 12px;
                    padding: 0.7rem 1rem;
                    color: {COLORS['positive']};
                    font-size: 0.85rem;
                ">✅ Training on <b>{row_count:,}</b> tweets · {int(test_ratio*100)}% test split
                · {num_features:,} features · λ={reg_param}</div>""",
                unsafe_allow_html=True,
            )

    # ── Train & Display Results ──────────────────────────────
    if train_clicked:
        _run_training(df, data_source, csv_path, test_ratio, num_features, reg_param, max_iter)

    # ── Show previous results if stored ──────────────────────
    elif "ml_result" in st.session_state:
        _display_results(st.session_state.ml_result)
    else:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            _glass_card(f"""
                <div style="text-align:center; padding: 2rem 0;">
                    <div style="font-size:3rem; margin-bottom:0.5rem;">🤖</div>
                    <h3 style="color:{COLORS['text']}; margin:0 0 0.5rem;">No Model Trained Yet</h3>
                    <p style="color:{COLORS['text']}77; font-size:0.9rem; max-width:400px; margin:0 auto;">
                        Configure hyperparameters above and click <b>Train Model</b>
                        to build a Spark MLlib Logistic Regression classifier.
                    </p>
                </div>
            """, border_color=f"{COLORS['neutral']}33"),
            unsafe_allow_html=True,
        )


def _run_training(df, data_source, csv_path, test_ratio, num_features, reg_param, max_iter):
    """Execute the Spark MLlib training pipeline and display results."""
    with st.spinner("⚙️ Initialising Spark and building ML pipeline…"):
        try:
            from spark.session import get_spark
            from spark.ml_pipeline import train_sentiment_model
            from spark.loader import load_sentiment140_spark, SENTIMENT140_SCHEMA

            spark = get_spark()

            # Build Spark DataFrame
            if data_source == "sentiment140":
                from spark.loader import load_sentiment140_spark, preprocess_sentiment140
                sdf = load_sentiment140_spark(spark, csv_path, limit=len(df))
                sdf = preprocess_sentiment140(sdf)
            else:
                # Demo: use pandas df → spark
                # Ensure clean_text exists
                if "clean_text" not in df.columns:
                    df["clean_text"] = df["text"].str.lower()
                sdf = spark.createDataFrame(
                    df[["clean_text", "sentiment_label"]].dropna()
                )

            progress = st.progress(0, text="🔧 Training Logistic Regression model…")
            progress.progress(10, text="🔧 Building pipeline stages…")

            result = train_sentiment_model(
                spark=spark,
                df=sdf,
                test_ratio=test_ratio,
                num_features=num_features,
                reg_param=reg_param,
                max_iter=max_iter,
            )

            progress.progress(100, text="✅ Training complete!")

            # Store in session
            st.session_state.ml_result = result

            _display_results(result)

        except Exception as e:
            st.error(f"❌ **ML Pipeline failed:** {e}")
            import traceback
            with st.expander("Stack Trace"):
                st.code(traceback.format_exc())


def _display_results(result):
    """Render the full results dashboard for a trained model."""

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Success Banner ───────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['positive']}12, {COLORS['primary']}08);
            border: 1px solid {COLORS['positive']}33;
            border-radius: 14px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 1.5rem;
            display: flex; align-items: center; gap: 0.8rem;
        ">
            <span style="font-size:1.4rem;">✅</span>
            <div>
                <span style="font-weight:700; color:{COLORS['positive']};">Model Trained Successfully</span>
                <span style="color:{COLORS['text']}88; font-size:0.85rem; margin-left:0.6rem;">
                    {result.train_count:,} train · {result.test_count:,} test ·
                    {result.train_time_sec}s · {result.num_features:,} features
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Metrics Row ──────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    metrics_data = [
        ("Accuracy",  f"{result.accuracy:.1%}",           COLORS["positive"]),
        ("Precision", f"{result.weighted_precision:.1%}",  COLORS["primary"]),
        ("Recall",    f"{result.weighted_recall:.1%}",     COLORS["secondary"]),
        ("F1 Score",  f"{result.weighted_f1:.1%}",         COLORS["accent"]),
    ]
    for col, (label, value, color) in zip([m1, m2, m3, m4], metrics_data):
        col.markdown(_metric_pill(label, value, color), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab_cm, tab_class, tab_feat, tab_config = st.tabs([
        "📊 Confusion Matrix",
        "📈 Per-Class Metrics",
        "🔬 Feature Importance",
        "⚙️ Configuration",
    ])

    # ── Confusion Matrix Tab ─────────────────────────────────
    with tab_cm:
        _render_confusion_matrix(result)

    # ── Per-Class Metrics Tab ────────────────────────────────
    with tab_class:
        _render_class_metrics(result)

    # ── Feature Importance Tab ───────────────────────────────
    with tab_feat:
        _render_feature_importance(result)

    # ── Configuration Tab ────────────────────────────────────
    with tab_config:
        _render_config(result)


def _render_confusion_matrix(result):
    """Render an interactive confusion matrix heatmap."""
    labels = result.label_order
    n = len(labels)

    # Build matrix
    matrix = []
    for row_dict in result.confusion_matrix:
        row_vals = [row_dict.get(f"Pred_{lbl}", 0) for lbl in labels]
        matrix.append(row_vals)

    matrix_np = np.array(matrix)

    # Normalised version (row-wise = recall per class)
    row_sums = matrix_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = (matrix_np / row_sums * 100).round(1)

    # Annotations: show both count and percentage
    annotations = []
    for i in range(n):
        for j in range(n):
            annotations.append(
                f"{matrix_np[i][j]}<br><span style='font-size:0.75em'>({matrix_pct[i][j]}%)</span>"
            )

    fig = go.Figure(data=go.Heatmap(
        z=matrix_np,
        x=[f"Predicted<br>{l}" for l in labels],
        y=[f"Actual<br>{l}" for l in labels],
        text=np.array(annotations).reshape(n, n),
        texttemplate="%{text}",
        colorscale=[
            [0.0, COLORS["background"]],
            [0.3, COLORS["primary"]],
            [0.7, COLORS["secondary"]],
            [1.0, COLORS["positive"]],
        ],
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Count"),
    ))

    fig.update_layout(
        title="Confusion Matrix",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Also show raw table
    with st.expander("📋 Raw Confusion Matrix Table"):
        cm_df = pd.DataFrame(result.confusion_matrix)
        st.dataframe(cm_df, use_container_width=True)


def _render_class_metrics(result):
    """Render per-class precision/recall/F1 as a grouped bar chart + table."""
    if not result.class_metrics:
        st.info("No per-class metrics available.")
        return

    class_df = pd.DataFrame([
        {
            "Class": cm.label,
            "Precision": cm.precision,
            "Recall": cm.recall,
            "F1 Score": cm.f1,
            "Support": cm.support,
        }
        for cm in result.class_metrics
    ])

    # Grouped bar chart
    melted = class_df.melt(
        id_vars=["Class", "Support"],
        value_vars=["Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )

    color_map = {
        "Precision": COLORS["primary"],
        "Recall": COLORS["secondary"],
        "F1 Score": COLORS["accent"],
    }

    fig = px.bar(
        melted,
        x="Class",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_map=color_map,
        text_auto=".3f",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        margin=dict(l=40, r=20, t=50, b=40),
        title="Per-Class Classification Metrics",
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=400,
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        opacity=0.9,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### 📋 Detailed Metrics")
    display_df = class_df.copy()
    display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x:.4f}")
    display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x:.4f}")
    display_df["F1 Score"] = display_df["F1 Score"].apply(lambda x: f"{x:.4f}")
    display_df["Support"] = display_df["Support"].apply(lambda x: f"{x:,}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_feature_importance(result):
    """Render top feature weights as a horizontal bar chart."""
    if not result.feature_importance_top:
        st.info("Feature importance data not available.")
        return

    feat_df = pd.DataFrame(
        result.feature_importance_top,
        columns=["Feature Index", "Coefficient Sum"],
    )
    feat_df["Feature"] = feat_df["Feature Index"].apply(lambda x: f"Feature #{x}")
    feat_df = feat_df.sort_values("Coefficient Sum", ascending=True)

    fig = px.bar(
        feat_df,
        x="Coefficient Sum",
        y="Feature",
        orientation="h",
        color="Coefficient Sum",
        color_continuous_scale=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        margin=dict(l=100, r=20, t=50, b=40),
        title="Top 20 Feature Weights (Sum of Absolute Coefficients)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False,
        height=500,
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="
            background: {COLORS['primary']}10;
            border: 1px solid {COLORS['primary']}25;
            border-radius: 12px;
            padding: 1rem;
            color: {COLORS['text']}aa;
            font-size: 0.82rem;
        ">
            💡 Feature indices correspond to <b>HashingTF</b> bucket positions.
            Higher coefficient sums indicate features (word hashes) that the model
            relies on most heavily for classification. The mapping is hashed, so
            indices don't directly correspond to specific words.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_config(result):
    """Render the model configuration summary."""
    st.markdown(
        _glass_card(f"""
            <h3 style="margin:0 0 1rem; color:{COLORS['text']};">🔧 Model Configuration</h3>
            <table style="width:100%; color:{COLORS['text']}; font-size:0.9rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88; width:50%;">Algorithm</td>
                    <td style="padding:0.6rem 0; font-weight:600;">Logistic Regression (Multinomial)</td>
                </tr>
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Feature Extraction</td>
                    <td style="padding:0.6rem 0; font-weight:600;">HashingTF + IDF</td>
                </tr>
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Vocabulary Size</td>
                    <td style="padding:0.6rem 0; font-weight:600;">{result.num_features:,}</td>
                </tr>
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Regularisation (λ)</td>
                    <td style="padding:0.6rem 0; font-weight:600;">{result.reg_param}</td>
                </tr>
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Max Iterations</td>
                    <td style="padding:0.6rem 0; font-weight:600;">{result.max_iter}</td>
                </tr>
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Training Samples</td>
                    <td style="padding:0.6rem 0; font-weight:600;">{result.train_count:,}</td>
                </tr>
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Test Samples</td>
                    <td style="padding:0.6rem 0; font-weight:600;">{result.test_count:,}</td>
                </tr>
                <tr style="border-bottom:1px solid {COLORS['primary']}15;">
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Training Time</td>
                    <td style="padding:0.6rem 0; font-weight:600;">{result.train_time_sec}s</td>
                </tr>
                <tr>
                    <td style="padding:0.6rem 0; color:{COLORS['text']}88;">Classes</td>
                    <td style="padding:0.6rem 0; font-weight:600;">{', '.join(result.label_order)}</td>
                </tr>
            </table>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown(
        _glass_card(f"""
            <h3 style="margin:0 0 1rem; color:{COLORS['text']};">🔄 Pipeline Stages</h3>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
                {"".join([
                    f'''<div style="
                        background: {COLORS['primary']}18;
                        border: 1px solid {COLORS['primary']}44;
                        border-radius: 10px;
                        padding: 0.5rem 0.9rem;
                        font-size: 0.82rem;
                        font-weight: 600;
                        color: {COLORS['text']};
                    ">{stage}</div>
                    <span style="color:{COLORS['text']}44; font-size:1.2rem;">→</span>'''
                    for stage in [
                        "📝 Tokenizer",
                        "🚫 StopWords",
                        "📊 HashingTF",
                        "📐 IDF",
                        "🏷️ StringIndexer",
                        "🤖 LogisticRegression",
                    ]
                ])}
            </div>
        """, border_color=f"{COLORS['secondary']}33"),
        unsafe_allow_html=True,
    )
