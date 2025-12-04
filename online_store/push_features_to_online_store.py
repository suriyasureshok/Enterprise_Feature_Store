"""
Push Features to Online Store (Redis)
This script takes the materialized training dataset and pushes
per-user OLTP features to Redis for real-time inference.
"""

import pandas as pd
import redis
import json

MATERIALIZED_DATA = "materialized_store/training_dataset.parquet"

def connect_redis():
    return redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True  # auto-decode UTF-8 strings
    )

def serialize_value(v):
    """
    Redis cannot store Python lists or numpy arrays directly.
    Convert them to JSON strings.
    """
    if v is None:
        return ""

    if isinstance(v, (list, tuple)):
        return json.dumps(v)

    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return json.dumps(v.tolist())
        if isinstance(v, np.bool_):
            return str(v)
    except ImportError:
        pass

    if isinstance(v, bool):
        return str(v)

    # primitive values can stay as-is
    return v

def push_to_redis():
    print("Loading materialized dataset...")
    df = pd.read_parquet(MATERIALIZED_DATA)

    print("Connecting to Redis...")
    r = connect_redis()

    print("Pushing features to online store...")

    for _, row in df.iterrows():
        user_id = row["user_id"]
        redis_key = f"user:{user_id}"

        feature_map = {}

        for col in df.columns:
            if col == "user_id":
                continue

            value = row[col]
            value = serialize_value(value)

            # Redis HSET needs str values
            feature_map[col] = value

        # Push the entire user feature map at once
        r.hset(redis_key, mapping=feature_map)

    print("Feature push to Redis completed successfully!")
    print(f"Total users pushed: {len(df)}")

if __name__ == "__main__":
    push_to_redis()
