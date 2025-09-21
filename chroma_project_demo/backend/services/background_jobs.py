import os

# Try to use Redis + RQ if available; otherwise, fall back to inline processing
try:
    from redis import Redis  # type: ignore
    from rq import Queue  # type: ignore
    _bg_mode = "rq"
    redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    q = Queue(connection=redis_conn)
except Exception as _e:
    print(f"[Background] RQ/Redis not available: {_e}. Using inline processing fallback.")
    _bg_mode = "inline"
    q = None  # type: ignore


def enqueue_file_processing(file_path: str, document_id: str):
    """Add a file processing job to the queue (or run inline if RQ unavailable)."""
    from services.file_processing import process_file_and_embed
    if _bg_mode == "rq" and q is not None:
        try:
            q.enqueue(process_file_and_embed, file_path, document_id)
            return
        except Exception as e:
            print(f"[Background] Failed to enqueue job: {e}. Running inline.")
    # Inline fallback (dev-friendly). Consider removing in production.
    try:
        process_file_and_embed(file_path, document_id)
    except Exception as e:
        print(f"[Background] Inline processing failed: {e}")

