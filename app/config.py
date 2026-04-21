"""
Centralised configuration for the Sentiment Analysis app.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# ── Spark ────────────────────────────────────────────────────────────────
SPARK_APP_NAME = "SocialMedia_SentimentAnalysis"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_LOG_LEVEL = "WARN"

# ── App Metadata ─────────────────────────────────────────────────────────
APP_TITLE = "📊 Social Media Sentiment Analyzer"
APP_ICON = "📊"
APP_DESCRIPTION = "Big-data sentiment analysis powered by PySpark"
VERSION = "2.0.0"

# ── Theme Colours (HSL based for consistency) ────────────────────────────
COLORS = {
    "primary":     "#6C5CE7",   # Purple
    "secondary":   "#00CEC9",   # Teal
    "accent":      "#FD79A8",   # Pink
    "positive":    "#00B894",   # Green
    "negative":    "#D63031",   # Red
    "neutral":     "#636E72",   # Grey
    "background":  "#0F0F1A",   # Dark
    "surface":     "#1A1A2E",   # Card bg
    "text":        "#DFE6E9",   # Light text
}

# ── Sentiment Thresholds ─────────────────────────────────────────────────
SENTIMENT_POS_THRESHOLD = 0.05
SENTIMENT_NEG_THRESHOLD = -0.05
