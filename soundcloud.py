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

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

# Overridable in case SoundCloud changes these.
TOKEN_URL = os.environ.get("SOUNDCLOUD_TOKEN_URL", "https://secure.soundcloud.com/oauth/token")
API_BASE = os.environ.get("SOUNDCLOUD_API_BASE", "https://api.soundcloud.com")
AUTH_SCHEME = os.environ.get("SOUNDCLOUD_AUTH_SCHEME", "OAuth")  # SoundCloud uses "OAuth <token>"

_TRACK_ID_RE = re.compile(r"soundcloud:tracks:(\d+)")
_token_cache: dict = {"token": None, "exp": 0.0}


class SoundCloudError(Exception):
    """Raised for auth / configuration problems worth surfacing to the user."""


def extract_track_id(location: str | None) -> str | None:
    """'soundcloud:tracks:979092301' -> '979092301' (None if not a SC location)."""
    if not location:
        return None
    m = _TRACK_ID_RE.search(str(location))
    return m.group(1) if m else None


def _http(method: str, url: str, data: bytes | None = None, headers: dict | None = None,
          timeout: int = 15) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8") or "{}"
        return resp.status, json.loads(body)


def get_token(client_id: str, client_secret: str, force: bool = False) -> str:
    """App-only OAuth token via client_credentials, cached until near expiry."""
    now = time.time()
    if not force and _token_cache["token"] and _token_cache["exp"] > now + 30:
        return _token_cache["token"]
    if not client_id or not client_secret:
        raise SoundCloudError("Missing SoundCloud client_id / client_secret.")

    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    try:
        _status, data = _http("POST", TOKEN_URL, body, headers)
    except urllib.error.HTTPError as exc:
        raise SoundCloudError(
            f"Token request failed (HTTP {exc.code}). Check your client id/secret."
        ) from exc
    except urllib.error.URLError as exc:
        raise SoundCloudError(f"Could not reach SoundCloud: {exc.reason}") from exc

    token = data.get("access_token")
    if not token:
        raise SoundCloudError("SoundCloud did not return an access_token.")
    _token_cache.update(token=token, exp=now + float(data.get("expires_in", 3600)))
    return token


def lookup_track(track_id: str, token: str) -> dict:
    """Fetch the download/purchase status for one SoundCloud track."""
    url = f"{API_BASE}/tracks/{track_id}"
    headers = {"Authorization": f"{AUTH_SCHEME} {token}", "Accept": "application/json"}
    _status, data = _http("GET", url, None, headers)
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
        if not t.get("streaming"):
            continue
        tid = extract_track_id(t.get("location") or t.get("path"))
        if not tid:
            continue
        if limit and checked >= limit:
            break
        checked += 1
        try:
            info = lookup_track(tid, token)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:  # token expired mid-run — refresh once and retry
                token = get_token(client_id, client_secret, force=True)
                try:
                    info = lookup_track(tid, token)
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
