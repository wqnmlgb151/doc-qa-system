import os
import re
import time
import unicodedata
from collections import defaultdict
from functools import wraps

from flask import request, jsonify

API_TOKEN = os.environ.get("API_TOKEN", "")

_rate_limits: dict[str, tuple[int, int]] = {
    "upload": (10, 3600),
    "chat": (30, 60),
}

_requests_store: dict[str, list[float]] = defaultdict(list)
_store_check_count: int = 0


def sanitize_filename(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename)
    filename = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename)
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    if len(filename) > 200:
        name_part, ext_part = os.path.splitext(filename)
        filename = name_part[:195] + ext_part
    return filename.strip()


def check_rate_limit(endpoint: str) -> bool:
    global _store_check_count
    if endpoint not in _rate_limits:
        return True
    max_req, window = _rate_limits[endpoint]
    now = time.time()
    key = f"{request.remote_addr}:{endpoint}"

    # Prune current key
    _requests_store[key] = [t for t in _requests_store[key] if now - t < window]
    if not _requests_store[key]:
        del _requests_store[key]

    # Periodic full sweep: evict all stale keys every ~1000 checks
    _store_check_count += 1
    if _store_check_count % 1000 == 0:
        stale = [k for k, ts in _requests_store.items() if not ts]
        for k in stale:
            del _requests_store[k]

    if len(_requests_store.get(key, [])) >= max_req:
        return False
    _requests_store[key].append(now)
    return True


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_TOKEN:
            return f(*args, **kwargs)
        token = request.headers.get("Authorization", "")
        if token != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def register_security_headers(app):
    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        return response
