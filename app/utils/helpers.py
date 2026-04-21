"""
Utility helper functions used across the application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(n: int = 500) -> pd.DataFrame:
    """
    Generate synthetic social media sentiment data for demonstration.
    Returns a DataFrame with realistic-looking tweets/posts.
    """
    platforms = ["Twitter", "Reddit", "Instagram", "Facebook", "YouTube"]
    topics = [
        "climate change", "AI technology", "cryptocurrency", "remote work",
        "electric vehicles", "mental health", "space exploration",
        "online education", "social media regulation", "metaverse"
    ]

    positive_templates = [
        "Really excited about the progress in {topic}! 🚀",
        "Amazing developments in {topic} this week. The future looks bright!",
        "Love seeing the innovation around {topic}. Keep it going! 💪",
        "Great news about {topic} today! This is what we need.",
        "{topic} is changing lives for the better. Incredible!",
        "So proud of what the community is doing for {topic} 🌟",
        "The advancements in {topic} are truly remarkable!",
    ]
    negative_templates = [
        "Concerned about the direction of {topic}. We need to do better.",
        "This approach to {topic} is completely wrong. Disappointing.",
        "Can we talk about how {topic} is being mishandled? 😤",
        "Not happy with the latest {topic} news. Very frustrating.",
        "{topic} continues to disappoint. When will things change?",
        "The problems with {topic} are getting worse, not better.",
    ]
    neutral_templates = [
        "Just read an interesting article about {topic}.",
        "What do you think about the latest {topic} developments?",
        "Here's a summary of {topic} trends for this month.",
        "Looking at {topic} from a different perspective today.",
        "New report on {topic} just dropped. Thoughts?",
    ]

    rows = []
    base_date = datetime.now() - timedelta(days=30)
    for _ in range(n):
        topic = random.choice(topics)
        sentiment_type = random.choices(
            ["positive", "negative", "neutral"],
            weights=[0.4, 0.3, 0.3]
        )[0]

        if sentiment_type == "positive":
            text = random.choice(positive_templates).format(topic=topic)
            score = round(random.uniform(0.1, 1.0), 3)
        elif sentiment_type == "negative":
            text = random.choice(negative_templates).format(topic=topic)
            score = round(random.uniform(-1.0, -0.1), 3)
        else:
            text = random.choice(neutral_templates).format(topic=topic)
            score = round(random.uniform(-0.1, 0.1), 3)

        rows.append({
            "timestamp": base_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            ),
            "platform": random.choice(platforms),
            "topic": topic,
            "text": text,
            "sentiment_label": sentiment_type,
            "sentiment_score": score,
            "likes": random.randint(0, 5000),
            "shares": random.randint(0, 2000),
            "user_followers": random.randint(50, 500_000),
        })

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def classify_sentiment(score: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> str:
    """Classify a polarity score into Positive / Negative / Neutral."""
    if score > pos_thresh:
        return "Positive"
    elif score < neg_thresh:
        return "Negative"
    return "Neutral"


def format_number(n: int | float) -> str:
    """Human-friendly number formatting (e.g. 1.2K, 3.4M)."""
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))
