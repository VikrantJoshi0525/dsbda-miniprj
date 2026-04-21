"""
PySpark text preprocessing pipeline.
Cleans, tokenises, and prepares social media text for sentiment analysis.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import re


def _clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtag symbols, and extra whitespace."""
    if not text:
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                     # @mentions
    text = re.sub(r"#", "", text)                         # hashtag symbol only
    text = re.sub(r"[^A-Za-z0-9\s!?.,']", "", text)     # special chars
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text.lower()


# Register as UDF
clean_text_udf = F.udf(_clean_text, StringType())


def preprocess(df: DataFrame, text_col: str = "text") -> DataFrame:
    """
    Apply full preprocessing pipeline to a Spark DataFrame.

    Steps:
        1. Drop nulls in text column
        2. Clean text (lowercase, remove URLs/mentions/special chars)
        3. Simple whitespace tokenisation
        4. Remove very short tokens (< 2 chars)

    Returns:
        DataFrame with added 'clean_text' and 'tokens' columns.
    """
    df = df.dropna(subset=[text_col])

    # Clean
    df = df.withColumn("clean_text", clean_text_udf(F.col(text_col)))

    # Tokenise (simple split)
    df = df.withColumn("tokens", F.split(F.col("clean_text"), r"\s+"))

    # Filter short tokens
    df = df.withColumn(
        "tokens",
        F.expr("filter(tokens, t -> length(t) >= 2)")
    )

    return df
