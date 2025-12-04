"""
Master Orchestrator for Feature Store Offline Pipelines
Runs all feature group builders and generates parquet files in offline_store/.
"""

import os
import pandas as pd

# Import feature builders
from .user_profile_features import build_user_profile_features
from .user_activity_features import build_user_activity_features
from .user_review_features import build_user_review_features
from .churn_labels import build_churn_labels

# -------------------------------------------------------------------------
# 1. Ensure offline_store directory exists
# -------------------------------------------------------------------------
os.makedirs("offline_store", exist_ok=True)
print("[INFO] offline_store/ directory is ready")

# -------------------------------------------------------------------------
# 2. Load RAW tables (you already created these earlier)
# -------------------------------------------------------------------------
print("[INFO] Loading raw data tables...")

user_df = pd.read_csv("data/processed/user_table.csv")
event_df = pd.read_csv("data/processed/event_logs.csv")
review_df = pd.read_csv("data/processed/review_table.csv")
target_df = pd.read_csv("data/processed/target_table.csv")

print("[INFO] Successfully loaded raw tables")

# -------------------------------------------------------------------------
# 3. Run Feature Builders (each returns a DataFrame)
# -------------------------------------------------------------------------

print("\n==== Building User Profile Features ====")
if 'user_profile_features.parquet' not in os.listdir('offline_store'):
    profile_features = build_user_profile_features(user_df)

print("\n==== Building User Activity Features ====")
if 'user_activity_features.parquet' not in os.listdir('offline_store'):
    activity_features = build_user_activity_features(event_df)

print("\n==== Building User Review (NLP) Features ====")
if 'user_review_features.parquet' not in os.listdir('offline_store'):
    review_features = build_user_review_features(review_df)

print("\n==== Building Churn Labels (Training Only) ====")
if 'churn_labels.parquet' not in os.listdir('offline_store'):
    label_features = build_churn_labels(target_df)

# -------------------------------------------------------------------------
# 4. Final Confirmation
# -------------------------------------------------------------------------

print("\nAll feature groups built successfully!")
print("Files saved under offline_store/:")
for file in os.listdir("offline_store"):
    print("  -", file)
