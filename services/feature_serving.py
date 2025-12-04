"""
Feature Serving API
This service exposes real-time feature lookup from Redis Online Store.

It allows ML inference systems to fetch:
- individual feature values
- entire feature groups
- full feature vectors for a user
"""

import json
import redis
from fastapi import FastAPI, HTTPException

# ----------------------------------------------
# Redis Connection (Same config used earlier)
# ----------------------------------------------
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True  # returns normal strings instead of bytes
)

app = FastAPI(
    title="Online Feature Store API",
    description="Serves user features from Redis for real-time ML inference",
    version="1.0.0"
)

# ----------------------------------------------
# 1. Fetch all features for a user_id
# ----------------------------------------------
@app.get("/features/{user_id}")
def get_user_features(user_id: str):
    try:
        raw_data = redis_client.hgetall(f"user:{user_id}")
        if not raw_data:
            raise HTTPException(status_code=404, detail="User features not found")

        # Convert JSON-serialized values (e.g., vectors) back to Python types
        parsed = {}
        for k, v in raw_data.items():
            try:
                parsed[k] = json.loads(v)
            except:
                parsed[k] = v

        return {
            "user_id": user_id,
            "features": parsed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------
# 2. Fetch a specific feature for a user
# ----------------------------------------------
@app.get("/features/{user_id}/{feature_name}")
def get_single_feature(user_id: str, feature_name: str):
    try:
        value = redis_client.hget(f"user:{user_id}", feature_name)
        if value is None:
            raise HTTPException(status_code=404, detail="Feature not found")

        try:
            value = json.loads(value)
        except:
            pass

        return {
            "user_id": user_id,
            "feature_name": feature_name,
            "value": value,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------
# 3. Health Check Endpoint
# ----------------------------------------------
@app.get("/health")
def health_check():
    try:
        redis_client.ping()
        return {"status": "OK", "redis": "connected"}
    except:
        raise HTTPException(status_code=500, detail="Redis connection failed")
