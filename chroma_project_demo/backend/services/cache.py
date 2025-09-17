"""
Redis-based caching helpers for frequently asked questions.
Optional: if REDIS_URL is not set or redis package not available, functions are no-ops.
"""
from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Optional

try:
    import redis  # type: ignore
    _HAS_REDIS = True
except Exception:
    redis = None  # type: ignore
    _HAS_REDIS = False

_REDIS_URL = os.getenv("REDIS_URL")
_QA_CACHE_TTL = int(os.getenv("QA_CACHE_TTL", "3600"))  # seconds
_client: Optional["redis.Redis"] = None


def _get_client() -> Optional["redis.Redis"]:
    global _client
    if not (_HAS_REDIS and _REDIS_URL):
        return None
    if _client is None:
        try:
            _client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        except Exception:
            return None
    return _client


def _hash_str(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def make_qa_key(message: str, context: str) -> str:
    # Use short hashes to keep keys compact
    return f"qa:{_hash_str(message)}:{_hash_str(context)}"


def qa_cache_get(message: str, context: str) -> Optional[dict]:
    cli = _get_client()
    if not cli:
        return None
    try:
        raw = cli.get(make_qa_key(message, context))
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def qa_cache_set(message: str, context: str, data: dict) -> None:
    cli = _get_client()
    if not cli:
        return
    try:
        key = make_qa_key(message, context)
        cli.setex(key, _QA_CACHE_TTL, json.dumps(data, ensure_ascii=False))
    except Exception:
        pass

