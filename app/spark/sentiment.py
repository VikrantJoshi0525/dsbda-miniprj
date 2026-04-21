"""
Sentiment analysis pipeline using PySpark + TextBlob / VADER.
Scores each row and classifies into Positive / Negative / Neutral.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SENTIMENT_POS_THRESHOLD, SENTIMENT_NEG_THRESHOLD


# ── TextBlob UDF ─────────────────────────────────────────────────
def _textblob_polarity(text: str) -> float:
    if not text:
        return 0.0
    return float(TextBlob(text).sentiment.polarity)

textblob_udf = F.udf(_textblob_polarity, FloatType())


# ── VADER UDF ────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()

def _vader_compound(text: str) -> float:
    if not text:
        return 0.0
    return float(_vader.polarity_scores(text)["compound"])

vader_udf = F.udf(_vader_compound, FloatType())


# ── Classification UDF ──────────────────────────────────────────
def _classify(score: float) -> str:
    if score is None:
        return "Neutral"
    if score > SENTIMENT_POS_THRESHOLD:
        return "Positive"
    elif score < SENTIMENT_NEG_THRESHOLD:
        return "Negative"
    return "Neutral"

classify_udf = F.udf(_classify, StringType())


# ── Public API ───────────────────────────────────────────────────
def analyse_sentiment(
    df: DataFrame,
    text_col: str = "clean_text",
    method: str = "vader",
) -> DataFrame:
    """
    Add sentiment_score and sentiment_label columns.

    Args:
        df:       Preprocessed Spark DataFrame (must contain `text_col`).
        text_col: Column with cleaned text.
        method:   'vader' or 'textblob'.

    Returns:
        DataFrame with 'sentiment_score' and 'sentiment_label' columns.
    """
    if method == "textblob":
        df = df.withColumn("sentiment_score", textblob_udf(F.col(text_col)))
    else:
        df = df.withColumn("sentiment_score", vader_udf(F.col(text_col)))

    df = df.withColumn("sentiment_label", classify_udf(F.col("sentiment_score")))
    return df
