import os
import re
import threading
import time
import unicodedata
from collections import defaultdict
from functools import wraps
from typing import Callable

from flask import request, jsonify

API_TOKEN = os.environ.get("API_TOKEN", "")

if not API_TOKEN:
    import logging
    logging.getLogger(__name__).warning(
        "API_TOKEN 未设置！上传和删除等受保护接口将在无认证状态下运行。"
        "如需启用认证，请在 .env 中设置 API_TOKEN。"
    )

_rate_limits: dict[str, tuple[int, int]] = {
    "upload": (10, 3600),
    "chat": (30, 60),
    "cancel": (60, 60),
}

_requests_store: dict[str, list[float]] = defaultdict(list)
_requests_lock = threading.Lock()
_store_check_count: int = 0


def sanitize_filename(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename)
    filename = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename)
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    if len(filename) > 200:
        name_part, ext_part = os.path.splitext(filename)
        filename = name_part[:195] + ext_part
    return filename.strip()


def compose_safe_name(file_id: str, filename: str) -> str:
    """Build a storage-safe filename: {file_id}_{sanitized_original_name}."""
    return f"{file_id}_{sanitize_filename(filename)}"


def check_rate_limit(endpoint: str) -> bool:
    global _store_check_count
    if endpoint not in _rate_limits:
        return True
    max_req, window = _rate_limits[endpoint]
    now = time.time()
    key = f"{request.remote_addr}:{endpoint}"

    with _requests_lock:
        # Prune current key
        _requests_store[key] = [t for t in _requests_store[key] if now - t < window]
        if not _requests_store[key]:
            del _requests_store[key]

        # Periodic full sweep: evict stale keys every ~1000 checks
        _store_check_count = (_store_check_count + 1) % 1000
        if _store_check_count == 0:
            stale = [k for k, ts in _requests_store.items()
                     if not ts or all(now - t >= window for t in ts)]
            for k in stale:
                del _requests_store[k]

        if len(_requests_store.get(key, [])) >= max_req:
            return False
        _requests_store[key].append(now)
        return True


def require_auth(f: "Callable") -> "Callable":
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_TOKEN:
            return f(*args, **kwargs)
        token = request.headers.get("Authorization", "")
        if token != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),
}


def register_security_headers(app: "Flask") -> None:
    @app.after_request
    def add_security_headers(response):
        for name, value in _SECURITY_HEADERS.items():
            response.headers[name] = value
        return response
