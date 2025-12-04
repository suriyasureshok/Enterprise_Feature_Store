"""
Online Store Setup
Creates and returns a Redis client for real-time feature serving.
"""

import redis
import os

def get_redis_client():
    return redis.StrictRedis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=False  # because we will store embeddings as bytes
    )
