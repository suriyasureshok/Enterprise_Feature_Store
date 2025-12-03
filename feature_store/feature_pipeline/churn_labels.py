"""
Churn Labels Feature Pipeline
This module defines the feature pipeline for building churn labels
from the Target DataFrame.
"""

import pandas as pd

def build_churn_labels(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build churn labels from the Target DataFrame.
    Args:
        target_df (pd.DataFrame): DataFrame containing target data.
    Returns:
        pd.DataFrame: DataFrame with churn labels.
    """
    churn_features = target_df[["user_id", "churned"]].copy()

    # -----------------------------
    # 1. Save Features
    # -----------------------------
    churn_features.to_parquet("offline_store/churn_labels.parquet", index=False)
    print("Churn labels generated successfully!")

    return churn_features