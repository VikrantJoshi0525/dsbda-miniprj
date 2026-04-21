"""
Basic tests for the sentiment analysis pipeline.
"""

import pytest
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def test_generate_sample_data():
    from utils.helpers import generate_sample_data
    df = generate_sample_data(n=50)
    assert len(df) == 50
    assert "text" in df.columns
    assert "sentiment_label" in df.columns
    assert "sentiment_score" in df.columns
    assert "platform" in df.columns


def test_classify_sentiment():
    from utils.helpers import classify_sentiment
    assert classify_sentiment(0.5) == "Positive"
    assert classify_sentiment(-0.5) == "Negative"
    assert classify_sentiment(0.0) == "Neutral"


def test_format_number():
    from utils.helpers import format_number
    assert format_number(999) == "999"
    assert format_number(1500) == "1.5K"
    assert format_number(2_500_000) == "2.5M"


def test_constants_exist():
    from utils.constants import PLATFORM_ICONS, SENTIMENT_COLORS, NAV_PAGES
    assert "Twitter" in PLATFORM_ICONS
    assert "Positive" in SENTIMENT_COLORS
    assert len(NAV_PAGES) > 0
