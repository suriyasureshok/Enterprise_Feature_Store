"""
User Profile Feature Pipeline
This module defines the feature pipeline for building user profile features
from the User DataFrame.
"""

import pandas as pd

def build_user_profile_features(user_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user profile features from the User DataFrame.

    Args:
        user_df (pd.DataFrame): DataFrame containing user data.

    Returns:
        pd.DataFrame: DataFrame with user profile features.
    """
    # Select relevant user profile features
    profile_features = user_df[[
        'user_id',
        'age',
        'gender',
        'location',
        'income',
        'education',
        'loyalty_status',
        'account_age_days',
        'premium',
        'device'    
    ]].copy()

    # -----------------------------
    # 1. Feature Engineering
    # -----------------------------

    # Handle missing values
    profile_features['age'].fillna(profile_features['age'].median(), inplace=True)
    profile_features['income'].fillna(profile_features['income'].median(), inplace=True)

    for col in ['gender', 'location', 'education', 'loyalty_status', 'device']:
        profile_features[col].fillna('Unknown', inplace=True)

    profile_features['account_age_days'].fillna(0, inplace=True)
    profile_features['premium'].fillna(False, inplace=True)

    # -----------------------------
    # 2. Save Features
    # -----------------------------
    profile_features.to_parquet("offline_store/user_profile_features.parquet", index=False)
    print("User profile features generated successfully!")
    
    return profile_features