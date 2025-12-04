"""
User Activity Feature Pipeline
This module defines the feature pipeline for building user activity features
from the Event Logs DataFrame.
"""

import numpy as np
import pandas as pd

def build_user_activity_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user activity features from the Event Logs DataFrame.
    Args:
        events_df (pd.DataFrame): DataFrame containing event logs
    Returns:
        pd.DataFrame: DataFrame with user activity features.
    """
    # Ensure timestamp is datetime
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format='ISO8601')

    # -----------------------------
    # 1. Basic Counts
    # -----------------------------
    total_events = events_df.groupby("user_id")["event_type"].count().rename("total_events")
    purchase_count = (events_df["event_type"] == "purchase").groupby(events_df["user_id"]).sum().rename("purchase_count")
    add_to_cart_count = (events_df["event_type"] == "add_to_cart").groupby(events_df["user_id"]).sum().rename("add_to_cart_count")

    # -----------------------------
    # 2. Session Calculation
    # -----------------------------
    events_df = events_df.sort_values(["user_id", "timestamp"])
    events_df["prev_time"] = events_df.groupby("user_id")["timestamp"].shift(1)
    events_df["time_diff"] = (events_df["timestamp"] - events_df["prev_time"]).dt.seconds.div(60).fillna(0)

    # New session if more than 30 mins passed
    events_df["new_session"] = (events_df["time_diff"] > 30).astype(int)

    # Session ID
    events_df["session_id"] = events_df.groupby("user_id")["new_session"].cumsum()

    # Compute session durations
    session_lengths = events_df.groupby(["user_id", "session_id"])["time_diff"].sum()
    avg_session_length = session_lengths.groupby("user_id").mean().rename("avg_session_length")

    session_count = events_df.groupby("user_id")["session_id"].nunique().rename("session_count")

    # -----------------------------
    # 3. Last Active Days
    # -----------------------------
    max_time = events_df["timestamp"].max()
    last_active_days = (max_time - events_df.groupby("user_id")["timestamp"].max()).dt.days.rename("last_active_days")

    # -----------------------------
    # 4. Categorical Event Vector (One-Hot Frequency)
    # -----------------------------
    event_vector = pd.get_dummies(events_df[["user_id", "event_type"]], columns=["event_type"])
    event_vector = event_vector.groupby("user_id").sum()

    # Rename columns to mark as vector components
    event_vector.columns = [f"event_vector_{c.replace('event_type_', '')}" for c in event_vector.columns]

    # -----------------------------
    # 5. Combine All Features
    # -----------------------------
    activity_features = (
        pd.concat([
            total_events,
            purchase_count,
            add_to_cart_count,
            session_count,
            avg_session_length,
            last_active_days,
            event_vector
        ], axis=1)
        .fillna(0)
        .reset_index()
    )

    # -----------------------------
    # 6. Save Features
    # -----------------------------
    activity_features.to_parquet("offline_store/user_activity_features.parquet", index=False)
    print("User activity features generated successfully!")

    return activity_features