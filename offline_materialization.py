"""
Offline Materialization Script
This script loads all feature groups from the offline store,
merges them into a single wide training dataset,
and stores the output in materialized_store/.
"""

import pandas as pd
import os

OFFLINE_STORE = "offline_store/"
MATERIALIZED_STORE = "materialized_store/"

def load_feature(path):
    full_path = os.path.join(OFFLINE_STORE, path)
    return pd.read_parquet(full_path)

def materialize_offline_dataset():
    # --------------------------------------------------
    # 1. Load Feature Groups
    # --------------------------------------------------
    print("Loading feature groups...")
    user_profile = load_feature("user_profile_features.parquet")
    user_activity = load_feature("user_activity_features.parquet")
    user_review = load_feature("user_review_features.parquet")
    churn_labels = load_feature("churn_labels.parquet")

    # --------------------------------------------------
    # 2. Merge all features using user_id as key
    # --------------------------------------------------
    print("Merging feature groups...")

    df = user_profile.merge(user_activity, on="user_id", how="left")
    df = df.merge(user_review, on="user_id", how="left")
    df = df.merge(churn_labels, on="user_id", how="left")

    # --------------------------------------------------
    # 3. Handle missing values (important!)
    # --------------------------------------------------
    df = df.fillna({
        "total_events": 0,
        "purchase_count": 0,
        "add_to_cart_count": 0,
        "session_count": 0,
        "avg_session_length": 0.0,
        "avg_rating": 0.0,
        "total_reviews": 0,
        "avg_helpful_ratio": 0.0,
        "avg_text_length": 0.0,
        "avg_sentiment_score": 0.0,
        "last_active_days": df["last_active_days"].max(),
        "last_review_days": df["last_review_days"].max()
    })

    # --------------------------------------------------
    # 4. Save final materialized dataset
    # --------------------------------------------------
    os.makedirs(MATERIALIZED_STORE, exist_ok=True)
    output_path = os.path.join(MATERIALIZED_STORE, "training_dataset.parquet")
    df.to_parquet(output_path, index=False)

    print(f"Materialized training dataset created at: {output_path}")
    print(df.head())

if __name__ == "__main__":
    materialize_offline_dataset()