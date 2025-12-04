"""
User Review Features Pipeline
This module defines the feature pipeline for building user review features
from the Reviews DataFrame.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def build_user_review_features(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user review features from the Reviews DataFrame.
    Args:
        reviews_df (pd.DataFrame): DataFrame containing user reviews.
    Returns:
        pd.DataFrame: DataFrame with user review features.
    """
    # -----------------------------
    # 1. Calculate Review Features
    # -----------------------------
    total_reviews = reviews_df.groupby('user_id')['rating'].count().rename('total_reviews')
    avg_rating = reviews_df.groupby('user_id')['rating'].mean().rename('avg_rating')
    avg_text_length = reviews_df.groupby('user_id')['text_length'].mean().rename('avg_text_length')

    # -----------------------------
    # 2. Sentiment Analysis Feature
    # -----------------------------
    # Ensure review_text is string
    reviews_df['review_text'] = reviews_df['review_text'].fillna('').astype(str)
    
    # Compute sentiment scores
    reviews_df['sentiment_score'] = reviews_df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'] if x.strip() else 0)
    sentiment_score = reviews_df.groupby('user_id')['sentiment_score'].mean().rename('avg_sentiment_score')

    # -----------------------------
    # 3. Review Embedding Feature
    # -----------------------------
    reviews_df['embedding'] = reviews_df['review_text'].apply(lambda x: model.encode(x))
    topic_vector = (
        reviews_df.groupby("user_id")["embedding"]
        .apply(lambda x: np.mean(np.stack(x), axis=0))
        .rename("topic_vector")
    )

    # -----------------------------
    # 4. Time Since Last Review Feature
    # -----------------------------
    reviews_df["review_time"] = pd.to_datetime(reviews_df["review_time"], format='ISO8601')
    max_time = reviews_df["review_time"].max()
    last_review_days = (max_time - reviews_df.groupby("user_id")["review_time"].max()).dt.days.rename("last_review_days")

    # -----------------------------
    # 5. Combine All Features
    # -----------------------------
    review_features = (
        pd.concat([
            avg_rating,
            total_reviews,
            avg_text_length,
            sentiment_score,
            topic_vector,
            last_review_days
        ], axis=1)
        .fillna(0)
        .reset_index()
    )

    # -----------------------------
    # 6. Save Features
    # -----------------------------
    review_features.to_parquet("offline_store/user_review_features.parquet", index=False)
    print("User review features generated successfully!")

    return review_features

if __name__ == "__main__":
    data = pd.read_csv("data/processed/review_table.csv")
    build_user_review_features(data)