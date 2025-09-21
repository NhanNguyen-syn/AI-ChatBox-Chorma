import os
import re
from typing import Tuple

# Simple, fast validations. For production, integrate antivirus like ClamAV via clamd.

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ALLOWED_EXTS = {
    "pdf", "txt", "doc", "docx", "png", "jpg", "jpeg", "xlsx", "xls", "csv",
}
ALLOWED_MIME_PREFIXES = (
    "application/pdf",
    "text/plain",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/png",
    "image/jpeg",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
)


def antivirus_scan_bytes(_data: bytes) -> Tuple[bool, str]:
    """Stub antivirus scan.
    Return (is_clean, details). Replace with clamd when available.
    """
    # Hook: if CLAMAV_REQUIRED=1, always reject (force integration) to avoid false sense of security
    if os.getenv("CLAMAV_REQUIRED", "0") == "1":
        return False, "Antivirus not integrated"
    return True, "clean"


def validate_meta(filename: str, content_type: str | None, total_bytes: int) -> Tuple[bool, str]:
    if not filename:
        return False, "Missing filename"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        return False, f"Unsupported file extension: .{ext}"
    if total_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        return False, f"File too large. Max {MAX_UPLOAD_MB}MB"
    if content_type:
        ok = any(content_type.startswith(pref) for pref in ALLOWED_MIME_PREFIXES)
        if not ok:
            return False, f"Unsupported content type: {content_type}"
    return True, "ok"

