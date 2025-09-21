import os
import time
from typing import Optional, Tuple

# Optional S3 support via boto3 (works with AWS or MinIO-compatible endpoints)
S3_ENABLED = os.getenv("S3_ENABLED", "0") == "1"

_s3 = None
_bucket = None
_region = None
_endpoint = None

if S3_ENABLED:
    try:
        import boto3  # type: ignore
        _region = os.getenv("S3_REGION", "us-east-1")
        _bucket = os.getenv("S3_BUCKET")
        _endpoint = os.getenv("S3_ENDPOINT_URL")  # e.g., http://localhost:9000 for MinIO
        if not _bucket:
            raise RuntimeError("S3_BUCKET not set")
        session = boto3.session.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=_region,
        )
        _s3 = session.client("s3", endpoint_url=_endpoint)
        print("[Storage] S3 enabled; bucket=", _bucket)
    except Exception as e:
        print(f"[Storage] Failed to init S3: {e}. Falling back to local storage.")
        S3_ENABLED = False

LOCAL_DIR = os.path.join("static", "chat_uploads")
os.makedirs(LOCAL_DIR, exist_ok=True)


def save_bytes(path_prefix: str, filename: str, data: bytes) -> Tuple[str, str]:
    """Save bytes either to S3 (if enabled) or to local static directory.
    Returns (public_url, storage_key).
    - For S3: storage_key is the object key; public_url is a presigned URL with short expiry.
    - For local: storage_key is the relative static path; public_url is "/static/..." URL.
    """
    safe_name = filename.replace("..", "_").replace("/", "_")
    key = f"{path_prefix.strip('/').rstrip('/')}/{int(time.time())}_{safe_name}"
    if S3_ENABLED and _s3 is not None and _bucket:
        _s3.put_object(Bucket=_bucket, Key=key, Body=data, ContentType=_guess_mime(filename))
        url = generate_presigned_url(key, expires_seconds=int(os.getenv("S3_URL_EXPIRES", "3600")))
        return url, key
    # Local fallback
    disk_path = os.path.join(LOCAL_DIR, key.replace("/", "_"))
    os.makedirs(os.path.dirname(disk_path), exist_ok=True)
    with open(disk_path, "wb") as f:
        f.write(data)
    public_url = f"/static/chat_uploads/{os.path.basename(disk_path)}"
    return public_url, disk_path


def generate_presigned_url(key: str, expires_seconds: int = 3600) -> str:
    if S3_ENABLED and _s3 is not None and _bucket:
        try:
            return _s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": _bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )
        except Exception as e:
            print(f"[Storage] presign failed: {e}")
    # Fallback for local
    return f"/static/chat_uploads/{key.replace('/', '_')}"


def _guess_mime(filename: str) -> str:
    import mimetypes
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"

