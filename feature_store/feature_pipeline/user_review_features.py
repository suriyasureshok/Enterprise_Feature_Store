"""
User Review Features Pipeline
This module defines the feature pipeline for building user review features
from the Reviews DataFrame.
"""

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

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
    reviews_df['sentiment_score'] = reviews_df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    sentiment_score = reviews_df.groupby('user_id')['sentiment_score'].mean().rename('avg_sentiment_score')

    # -----------------------------
    # 3. Review Embedding Feature
    # -----------------------------
    reviews_df['embedding'] = reviews_df['review_text'].apply(lambda x: model.encode(x))
    topic_vector = reviews_df.groupby("user_id")["embedding"].apply(lambda x: x.iloc[0]).rename("topic_vector")

    # -----------------------------
    # 4. Time Since Last Review Feature
    # -----------------------------
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