"""
Sentiment140 CSV Loader — PySpark pipeline.

Loads the Sentiment140 dataset (1.6M tweets), cleans and preprocesses it,
runs sentiment analysis, and returns a Pandas DataFrame matching the app schema.

Sentiment140 CSV format (no header):
    Column 0: target   — 0 = negative, 2 = neutral, 4 = positive
    Column 1: ids      — tweet id
    Column 2: date     — date string  (e.g. "Mon May 11 03:17:40 UTC 2009")
    Column 3: flag     — query flag (NO_QUERY if none)
    Column 4: user     — username
    Column 5: text     — tweet text
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    LongType,
    StringType,
    FloatType,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SAMPLE_DATA_DIR

# ── Sentiment140 schema ──────────────────────────────────────────
SENTIMENT140_SCHEMA = StructType([
    StructField("target",  IntegerType(), True),
    StructField("ids",     LongType(),    True),
    StructField("date",    StringType(),  True),
    StructField("flag",    StringType(),  True),
    StructField("user",    StringType(),  True),
    StructField("text",    StringType(),  True),
])

# Map Sentiment140 target codes to labels
_TARGET_MAP = {0: "Negative", 2: "Neutral", 4: "Positive"}


def _map_target_label(target: int) -> str:
    """Map Sentiment140 integer target to Positive/Negative/Neutral."""
    if target is None:
        return "Neutral"
    return _TARGET_MAP.get(target, "Neutral")


def _map_target_score(target: int) -> float:
    """Map Sentiment140 integer target to approximate polarity score."""
    if target == 4:
        return 0.6
    elif target == 0:
        return -0.6
    return 0.0


def load_sentiment140_spark(
    spark: SparkSession,
    csv_path: str,
    limit: int | None = None,
) -> DataFrame:
    """
    Load the Sentiment140 CSV into a Spark DataFrame.

    Args:
        spark:    Active SparkSession.
        csv_path: Absolute path to the CSV file.
        limit:    Optional row limit (for faster dev iteration).

    Returns:
        Raw Spark DataFrame with original columns.
    """
    sdf = (
        spark.read
        .option("header", "false")
        .option("encoding", "latin1")          # Sentiment140 uses ISO-8859-1
        .option("quote", '"')
        .option("escape", '"')
        .schema(SENTIMENT140_SCHEMA)
        .csv(csv_path)
    )

    if limit:
        sdf = sdf.limit(limit)

    return sdf


def preprocess_sentiment140(sdf: DataFrame) -> DataFrame:
    """
    Full PySpark preprocessing pipeline for Sentiment140.

    Steps:
        1. Drop nulls / empty text
        2. Map target → sentiment_label  (original dataset labels)
        3. Map target → sentiment_score  (approximate polarity)
        4. Parse timestamp from the date string
        5. Clean text (lowercase, remove URLs / @mentions / special chars)
        6. Tokenise
        7. Add derived columns: platform, topic, likes, shares, user_followers
           (Sentiment140 is Twitter only, so we derive reasonable columns)

    Returns:
        Cleaned Spark DataFrame ready for .toPandas().
    """
    from spark.preprocessing import clean_text_udf

    # ── Register UDFs ────────────────────────────────────────
    map_label_udf = F.udf(_map_target_label, StringType())
    map_score_udf = F.udf(_map_target_score, FloatType())

    # 1. Drop nulls
    sdf = sdf.filter(F.col("text").isNotNull() & (F.trim(F.col("text")) != ""))

    # 2–3. Sentiment from original target
    sdf = sdf.withColumn("sentiment_label", map_label_udf(F.col("target")))
    sdf = sdf.withColumn("sentiment_score", map_score_udf(F.col("target")))

    # 4. Parse timestamp — Sentiment140 format: "Mon May 11 03:17:40 UTC 2009"
    #    We remove 'UTC' because to_timestamp doesn't handle named TZ in Spark < 4
    sdf = sdf.withColumn(
        "date_clean",
        F.regexp_replace(F.col("date"), r"\s*(PDT|UTC|PST|EST|CST)\s*", " ")
    )
    sdf = sdf.withColumn(
        "timestamp",
        F.to_timestamp(F.col("date_clean"), "EEE MMM dd HH:mm:ss yyyy")
    )
    # Fallback: if parsing fails, use a fixed date
    sdf = sdf.withColumn(
        "timestamp",
        F.coalesce(F.col("timestamp"), F.lit("2009-06-01 00:00:00").cast("timestamp"))
    )

    # 5. Clean text
    sdf = sdf.withColumn("clean_text", clean_text_udf(F.col("text")))

    # 6. Tokenise
    sdf = sdf.withColumn("tokens", F.split(F.col("clean_text"), r"\s+"))
    sdf = sdf.withColumn(
        "tokens",
        F.expr("filter(tokens, t -> length(t) >= 2)")
    )
    sdf = sdf.withColumn("word_count", F.size(F.col("tokens")))

    # 7. Derived columns for compatibility with dashboard
    sdf = sdf.withColumn("platform", F.lit("Twitter"))

    # Derive topic from keywords in the text using a simple rule-based mapping
    sdf = sdf.withColumn(
        "topic",
        F.when(F.col("clean_text").rlike("(?i)tech|ai|computer|software|code|app"), "Technology")
         .when(F.col("clean_text").rlike("(?i)music|song|album|concert|band"), "Music")
         .when(F.col("clean_text").rlike("(?i)movie|film|show|watch|tv|netflix"), "Entertainment")
         .when(F.col("clean_text").rlike("(?i)sport|game|play|team|win|score"), "Sports")
         .when(F.col("clean_text").rlike("(?i)food|eat|cook|restaurant|recipe"), "Food")
         .when(F.col("clean_text").rlike("(?i)work|job|office|boss|career|hire"), "Work")
         .when(F.col("clean_text").rlike("(?i)school|class|study|learn|exam|university"), "Education")
         .when(F.col("clean_text").rlike("(?i)love|heart|miss|feel|happy|sad"), "Emotions")
         .when(F.col("clean_text").rlike("(?i)weather|rain|sun|cold|hot|snow"), "Weather")
         .when(F.col("clean_text").rlike("(?i)health|sick|doctor|hospital|pain|sleep"), "Health")
         .otherwise("General")
    )

    # Synthetic engagement (Sentiment140 doesn't have these, generate realistic ones)
    sdf = sdf.withColumn("likes",          (F.rand() * 5000).cast("int"))
    sdf = sdf.withColumn("shares",         (F.rand() * 2000).cast("int"))
    sdf = sdf.withColumn("user_followers", (F.rand() * 500000).cast("int"))

    # Select final columns
    sdf = sdf.select(
        "timestamp",
        "platform",
        "topic",
        "text",
        "clean_text",
        "tokens",
        "word_count",
        "sentiment_label",
        "sentiment_score",
        "likes",
        "shares",
        "user_followers",
        "user",
    )

    return sdf


def run_sentiment140_pipeline(
    spark: SparkSession,
    csv_path: str,
    limit: int = 5000,
    method: str = "dataset",
) -> pd.DataFrame:
    """
    End-to-end pipeline: load → clean → (optionally re-score) → Pandas.

    Args:
        spark:    Active SparkSession.
        csv_path: Path to Sentiment140 CSV.
        limit:    Row limit for performance.
        method:   'dataset' uses original labels, 'vader' or 'textblob' re-scores.

    Returns:
        Pandas DataFrame ready for the dashboard.
    """
    # Load
    sdf = load_sentiment140_spark(spark, csv_path, limit=limit)

    # Preprocess
    sdf = preprocess_sentiment140(sdf)

    # Optionally re-score with VADER/TextBlob
    if method in ("vader", "textblob"):
        from spark.sentiment import analyse_sentiment
        sdf = sdf.drop("sentiment_score", "sentiment_label")
        sdf = analyse_sentiment(sdf, text_col="clean_text", method=method)

    # Convert to Pandas
    pdf = sdf.toPandas()

    # Ensure proper types
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], errors="coerce")
    pdf["sentiment_score"] = pdf["sentiment_score"].astype(float)

    return pdf
