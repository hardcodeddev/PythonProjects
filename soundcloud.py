"""
soundcloud.py

Optional SoundCloud lookup for the streaming tracks in a Rekordbox library.

Rekordbox stores a SoundCloud track as a Location like 'soundcloud:tracks:<ID>',
where <ID> is the SoundCloud track ID. With a SoundCloud developer app
(client_id + client_secret) we can get an app-only OAuth token via the
client_credentials grant, then GET /tracks/{id} to read whether the track is a
free download, a paid purchase, or neither — answering "can I grab this one?".

Credentials are read from the request or the SOUNDCLOUD_CLIENT_ID /
SOUNDCLOUD_CLIENT_SECRET environment variables. They are never written to disk
or logged. Endpoints/auth scheme are overridable via env in case SoundCloud
adjusts them.

No third-party deps — uses urllib from the stdlib.
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

# Overridable in case SoundCloud changes these.
TOKEN_URL = os.environ.get("SOUNDCLOUD_TOKEN_URL", "https://secure.soundcloud.com/oauth/token")
API_BASE = os.environ.get("SOUNDCLOUD_API_BASE", "https://api.soundcloud.com")
AUTH_SCHEME = os.environ.get("SOUNDCLOUD_AUTH_SCHEME", "OAuth")  # SoundCloud uses "OAuth <token>"


def _ssl_context() -> ssl.SSLContext:
    """A verifying SSL context that actually has CA roots.

    Fixes the common macOS "unable to get local issuer certificate" — the stock
    python.org build ships no system trust store, so we point urllib at certifi's
    bundle (or a CA file from SOUNDCLOUD_CA_BUNDLE / SSL_CERT_FILE).
    """
    ca = os.environ.get("SOUNDCLOUD_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if ca and os.path.exists(ca):
        return ssl.create_default_context(cafile=ca)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to system defaults
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()

_TRACK_ID_RE = re.compile(r"soundcloud[:/]+tracks[:/]+(\d+)", re.I)
_URL_RE = re.compile(r"https?://(?:www\.|m\.)?soundcloud\.com/\S+", re.I)
_token_cache: dict = {"token": None, "exp": 0.0}


class SoundCloudError(Exception):
    """Raised for auth / configuration problems worth surfacing to the user."""


def extract_track_id(location: str | None) -> str | None:
    """'soundcloud:tracks:979092301' -> '979092301' (None if not an id location)."""
    if not location:
        return None
    m = _TRACK_ID_RE.search(str(location))
    return m.group(1) if m else None


def extract_track_ref(location: str | None):
    """Return ('id', '12345') or ('url', 'https://soundcloud.com/...') or None.

    Handles both how Rekordbox stores streaming tracks — a numeric id
    ('soundcloud:tracks:<id>') and, as a fallback, a public permalink URL
    (resolved via the API).
    """
    if not location:
        return None
    tid = extract_track_id(location)
    if tid:
        return ("id", tid)
    m = _URL_RE.search(str(location))
    if m:
        return ("url", m.group(0))
    return None


def is_soundcloud(location: str | None) -> bool:
    return extract_track_ref(location) is not None


def _http(method: str, url: str, data: bytes | None = None, headers: dict | None = None,
          timeout: int = 15) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        body = resp.read().decode("utf-8") or "{}"
        return resp.status, json.loads(body)


def _request_token(client_id: str, client_secret: str, use_basic: bool) -> tuple[str, float]:
    """One client_credentials token request. Credentials via HTTP Basic auth
    (OAuth2 standard) when use_basic, otherwise as body params."""
    params = {"grant_type": "client_credentials"}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json; charset=utf-8",
    }
    if use_basic:
        cred = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers["Authorization"] = f"Basic {cred}"
    else:
        params["client_id"] = client_id
        params["client_secret"] = client_secret
    _status, data = _http("POST", TOKEN_URL, urllib.parse.urlencode(params).encode(), headers)
    token = data.get("access_token")
    if not token:
        raise SoundCloudError("SoundCloud did not return an access_token.")
    return token, float(data.get("expires_in", 3600))


def get_token(client_id: str, client_secret: str, force: bool = False) -> str:
    """App-only OAuth token via client_credentials, cached until near expiry.

    Tries HTTP Basic auth first (what OAuth2 / SoundCloud expect), then falls
    back to sending the credentials as body params, since SoundCloud has
    documented both over time.
    """
    now = time.time()
    if not force and _token_cache["token"] and _token_cache["exp"] > now + 30:
        return _token_cache["token"]
    if not client_id or not client_secret:
        raise SoundCloudError("Missing SoundCloud client_id / client_secret.")

    last_code = None
    for use_basic in (True, False):
        try:
            token, expires_in = _request_token(client_id, client_secret, use_basic)
            _token_cache.update(token=token, exp=now + expires_in)
            return token
        except urllib.error.HTTPError as exc:
            last_code = exc.code
            if exc.code in (400, 401, 403):  # credential-format issue — try the other style
                continue
            raise SoundCloudError(f"Token request failed (HTTP {exc.code}).") from exc
        except urllib.error.URLError as exc:
            raise SoundCloudError(f"Could not reach SoundCloud: {exc.reason}") from exc

    raise SoundCloudError(
        f"Token request failed (HTTP {last_code}) with both Basic-auth and body credentials. "
        "Confirm the Client ID/Secret are from a SoundCloud app with API access enabled."
    )


def _track_fields(data: dict) -> dict:
    return {
        "title": data.get("title"),
        "user": (data.get("user") or {}).get("username"),
        "downloadable": bool(data.get("downloadable")),
        "download_url": data.get("download_url"),
        "purchase_url": data.get("purchase_url"),
        "purchase_title": data.get("purchase_title"),
        "permalink_url": data.get("permalink_url"),
        "streamable": bool(data.get("streamable")),
    }


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"{AUTH_SCHEME} {token}", "Accept": "application/json"}


def lookup_track(track_id: str, token: str) -> dict:
    """Download/purchase status for one SoundCloud track by numeric id."""
    _status, data = _http("GET", f"{API_BASE}/tracks/{track_id}", None, _auth_headers(token))
    return _track_fields(data)


def resolve_track(url: str, token: str) -> dict:
    """Resolve a SoundCloud permalink URL to its track and read the same fields."""
    resolve = f"{API_BASE}/resolve?url={urllib.parse.quote(url, safe='')}"
    _status, data = _http("GET", resolve, None, _auth_headers(token))
    return _track_fields(data)


def lookup_ref(ref, token: str) -> dict:
    kind, value = ref
    return lookup_track(value, token) if kind == "id" else resolve_track(value, token)


def classify(info: dict) -> str:
    """Coarse label for the UI: 'free', 'buy', 'stream', or 'error'."""
    if info.get("error"):
        return "error"
    if info.get("downloadable"):
        return "free"
    if info.get("purchase_url"):
        return "buy"
    return "stream"


def enrich_library(tracks: list[dict], client_id: str, client_secret: str,
                   limit: int | None = None, sleep: float = 0.12) -> dict:
    """Look up every SoundCloud track in `tracks`, attaching an `sc` dict to each.

    Returns {index: sc_info}. Refreshes the token once on 401; stops early and
    marks the current track on 429 (rate limit).
    """
    token = get_token(client_id, client_secret)
    results: dict[int, dict] = {}
    checked = 0
    for i, t in enumerate(tracks):
        ref = extract_track_ref(t.get("location") or t.get("path"))
        if not ref:
            continue
        if limit and checked >= limit:
            break
        checked += 1
        try:
            info = lookup_ref(ref, token)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:  # token expired mid-run — refresh once and retry
                token = get_token(client_id, client_secret, force=True)
                try:
                    info = lookup_ref(ref, token)
                except Exception as exc2:  # noqa: BLE001
                    info = {"error": f"HTTP {getattr(exc2, 'code', '?')}"}
            elif exc.code == 429:
                t["sc"] = {"error": "rate limited — try again later"}
                results[i] = t["sc"]
                break
            elif exc.code == 404:
                info = {"error": "not found on SoundCloud"}
            else:
                info = {"error": f"HTTP {exc.code}"}
        except urllib.error.URLError as exc:
            info = {"error": str(exc.reason)}
        info["kind"] = classify(info)
        t["sc"] = info
        results[i] = info
        time.sleep(sleep)
    return results
