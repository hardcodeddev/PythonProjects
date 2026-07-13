"""
app.py — local Flask front end for the doubling-compatibility engine.

Run:
    pip install -r requirements.txt
    python app.py           # then open http://127.0.0.1:5000

Upload a rekordbox library XML export (or a JSON track list), tune the
sub-score weights, and browse each track's top doubling partners. The full
audio "pocket" analysis (spectral + rhythm) can be run across the whole
library, or on demand for a single pair — both go through doubling_score's
DoublingEngine so the scoring is identical to the CLI.

This is a single-user local tool: the loaded library lives in process memory.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from dataclasses import asdict
from urllib.parse import unquote, urlparse

from flask import Flask, abort, jsonify, render_template, request, send_file

import doubling_score as ds
import library_cues as lc
import soundcloud as sc

app = Flask(__name__)

# In-memory library for the current session (single-user, local use).
STATE: dict = {"tracks": [], "source": None}

_TRUE = {"1", "true", "on", "yes", "True"}
_SUMMARY_FIELDS = ("title", "artist", "bpm", "camelot_key", "path", "streaming", "location")


def _library_response(tracks: list[dict], **extra) -> dict:
    """The library summary payload the front end expects (upload/scan/autocue)."""
    payload = dict(
        source=STATE.get("source"),
        count=len(tracks),
        with_key=sum(1 for t in tracks if t.get("camelot_key")),
        with_bpm=sum(1 for t in tracks if t.get("bpm")),
        with_path=sum(1 for t in tracks if t.get("path")),
        streaming=sum(1 for t in tracks if t.get("streaming")),
        total_cues=sum(len(t.get("cues", [])) for t in tracks),
        tracks=[
            {**{k: t.get(k) for k in _SUMMARY_FIELDS}, "cues": t.get("cues", []), "sc": t.get("sc")}
            for t in tracks
        ],
    )
    payload.update(extra)
    return payload


def _engine_from(payload: dict) -> ds.DoublingEngine:
    """Build an engine from the request's weight/tolerance config."""
    w = payload.get("weights") or {}
    return ds.DoublingEngine(
        weight_key=float(w.get("key", ds.WEIGHT_KEY)),
        weight_bpm=float(w.get("bpm", ds.WEIGHT_BPM)),
        weight_spectral=float(w.get("spectral", ds.WEIGHT_SPECTRAL_POCKET)),
        weight_rhythm=float(w.get("rhythm", ds.WEIGHT_RHYTHM_POCKET)),
        bpm_tolerance_pct=float(payload.get("tolerance_pct", 2.0)),
    )


def _location_formats(tracks: list[dict]) -> str:
    """Histogram of Location URI schemes in the library, for diagnostics."""
    from collections import Counter
    counts: Counter = Counter()
    for t in tracks:
        loc = t.get("location") or t.get("path") or ""
        if not loc:
            counts["(no location)"] += 1
            continue
        m = re.match(r"^([a-zA-Z][\w+.\-]*):", loc)
        counts[m.group(1) if m else "(bare path)"] += 1
    return ", ".join(f"{k}: {v}" for k, v in counts.most_common())


def _normalize_input_path(raw: str) -> str:
    """Make a user-supplied folder path usable: strip quotes, expand ~, decode a
    file:// URL, and URL-decode (%20 -> space) when that resolves to a real dir.
    """
    if not raw:
        return ""
    p = raw.strip().strip('"').strip("'")
    if p.startswith("file://"):
        p = unquote(urlparse(p).path)
    p = os.path.expanduser(p)
    if os.path.isdir(p):
        return p
    decoded = unquote(p)  # handles pasted paths with %20 etc.
    return decoded if os.path.isdir(decoded) else p


def _remap_from(payload: dict):
    r = payload.get("remap") or {}
    frm = (r.get("from") or "").strip()
    to = (r.get("to") or "").strip()
    return (frm, to) if frm else None


@app.get("/")
def index():
    return render_template("index.html", defaults={
        "key": ds.WEIGHT_KEY,
        "bpm": ds.WEIGHT_BPM,
        "spectral": ds.WEIGHT_SPECTRAL_POCKET,
        "rhythm": ds.WEIGHT_RHYTHM_POCKET,
    })


@app.post("/api/upload")
def upload():
    """Parse an uploaded rekordbox XML / JSON track list into the library."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file uploaded."), 400

    fmt = request.form.get("format", "auto") or "auto"
    skip_missing = request.form.get("skip_missing", "") in _TRUE

    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        tracks = ds.load_tracks(tmp.name, fmt=fmt, skip_missing=skip_missing)
    except Exception as exc:  # noqa: BLE001 - surface parse errors to the UI
        return jsonify(error=f"Failed to parse {file.filename}: {exc}"), 400
    finally:
        os.unlink(tmp.name)

    STATE["tracks"] = tracks
    STATE["source"] = file.filename
    return jsonify(_library_response(tracks))


@app.post("/api/rank")
def rank():
    """Rank the whole library's top-N doubling partners per track."""
    tracks = STATE["tracks"]
    if not tracks:
        return jsonify(error="No library loaded — upload a file first."), 400

    payload = request.get_json(force=True, silent=True) or {}
    engine = _engine_from(payload)
    sr = payload.get("sr")
    try:
        records = engine.rank(
            tracks,
            top_n=int(payload.get("top_n", 5)),
            use_audio=bool(payload.get("use_audio", False)),
            sr=int(sr) if sr else None,
            remap=_remap_from(payload),
            audio_only=bool(payload.get("audio_only", False)),
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Ranking failed: {exc}"), 500
    return jsonify(records=records, count=len(tracks), compared=len(records))


@app.post("/api/pair")
def pair():
    """Deep-analyze a single pair on demand (loads audio for just these two)."""
    tracks = STATE["tracks"]
    payload = request.get_json(force=True, silent=True) or {}
    try:
        i, j = int(payload["i"]), int(payload["j"])
        meta_a, meta_b = tracks[i], tracks[j]
    except (KeyError, IndexError, ValueError, TypeError):
        return jsonify(error="Invalid track indices."), 400

    engine = _engine_from(payload)
    remap = _remap_from(payload)
    feat_a = engine.load_features(meta_a, remap=remap)
    feat_b = engine.load_features(meta_b, remap=remap)
    result = engine.score_features(meta_a, meta_b, feat_a, feat_b)

    missing = [
        m.get("title", "?")
        for m, f in ((meta_a, feat_a), (meta_b, feat_b))
        if f is None
    ]
    return jsonify(
        result=asdict(result),
        audio_ok=not result.needs_audio_analysis,
        missing_audio=missing,
    )


@app.post("/api/cues")
def cues():
    """Suggest mixing cue points for doubling two tracks (needs local audio)."""
    tracks = STATE["tracks"]
    payload = request.get_json(force=True, silent=True) or {}
    try:
        i, j = int(payload["i"]), int(payload["j"])
        meta_a, meta_b = tracks[i], tracks[j]
    except (KeyError, IndexError, ValueError, TypeError):
        return jsonify(ok=False, reason="Invalid track indices."), 400

    out = ds.DoublingEngine().suggest_cues(meta_a, meta_b, remap=_remap_from(payload))
    return jsonify(out)


@app.get("/api/browse")
def browse():
    """List sub-folders of a directory for the folder picker (local machine)."""
    path = _normalize_input_path(request.args.get("path", "")) or os.path.expanduser("~")
    if not os.path.isdir(path):
        path = os.path.expanduser("~")

    entries = []
    try:
        for name in sorted(os.listdir(path), key=str.lower):
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append({"name": name, "path": full})
    except OSError as exc:
        return jsonify(error=f"Can't open {path}: {exc}"), 400

    try:
        here_audio = sum(1 for f in os.listdir(path) if f.lower().endswith(lc.AUDIO_EXTS))
    except OSError:
        here_audio = 0

    # Quick links: home, root, and any mounted volumes (USB / external drives).
    quick = [{"name": "🏠 Home", "path": os.path.expanduser("~")}, {"name": "/ Root", "path": "/"}]
    for mount in ("/Volumes", "/media", "/mnt"):
        if os.path.isdir(mount):
            try:
                for d in sorted(os.listdir(mount)):
                    p = os.path.join(mount, d)
                    if os.path.isdir(p) and not d.startswith("."):
                        quick.append({"name": f"💾 {d}", "path": p})
            except OSError:
                pass

    parent = os.path.dirname(path.rstrip("/")) or "/"
    return jsonify(path=path, parent=parent, entries=entries, quick=quick, here_audio=here_audio)


@app.post("/api/scan")
def scan():
    """Scan folder(s) of audio files, detect BPM/key/cues, load as the library."""
    payload = request.get_json(force=True, silent=True) or {}
    folders = payload.get("folders") or []
    if isinstance(folders, str):
        folders = [ln for ln in folders.splitlines() if ln.strip()]
    folders = [_normalize_input_path(f) for f in folders if f and str(f).strip()]
    folders = [f for f in folders if f]
    if not folders:
        return jsonify(error="Pick at least one folder."), 400

    bad = [f for f in folders if not os.path.isdir(f)]
    if bad:
        return jsonify(error=f"Not a folder on this machine: {', '.join(bad)}"), 400

    recursive = bool(payload.get("recursive", True))
    max_files = payload.get("max_files")
    try:
        tracks = lc.analyze_folders(
            folders, recursive=recursive, max_files=int(max_files) if max_files else None
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Scan failed: {exc}"), 500

    STATE["tracks"] = tracks
    STATE["source"] = f"{len(folders)} folder(s)"
    failed = [t["title"] for t in tracks if t.get("error")]
    return jsonify(_library_response(tracks, failed=failed))


@app.post("/api/autocue")
def autocue():
    """One-click: detect + attach cues to every local track in the loaded library.

    Keeps each track's existing BPM/key (e.g. Rekordbox's own analysis). Streaming
    and non-local tracks are left untouched.
    """
    tracks = STATE["tracks"]
    if not tracks:
        return jsonify(error="Load a library first (upload a Rekordbox XML)."), 400
    remap = _remap_from(request.get_json(force=True, silent=True) or {})

    updated, failed = [], []
    for t in tracks:
        path = ds._apply_path_remap(t.get("path"), remap)
        if path and not t.get("streaming") and ds._is_local_path(path) and os.path.isfile(path):
            try:
                updated.append(lc.add_cues({**t, "path": path}))
                continue
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{t.get('title', '?')}: {exc}")
        updated.append({**t, "cues": t.get("cues", [])})

    STATE["tracks"] = updated
    return jsonify(_library_response(updated, failed=failed))


def _local_track_or_404(index: int) -> dict:
    tracks = STATE["tracks"]
    if not (0 <= index < len(tracks)):
        abort(404)
    meta = tracks[index]
    path = meta.get("path")
    if not path or not ds._is_local_path(path) or not os.path.isfile(path):
        abort(404)
    return meta


@app.get("/api/audio/<int:index>")
def audio(index):
    """Stream a track's local audio file (Range-enabled) for the preview player."""
    meta = _local_track_or_404(index)
    return send_file(meta["path"], conditional=True)


@app.get("/api/waveform/<int:index>")
def waveform(index):
    """Downsampled waveform peaks + current cues for the cue editor."""
    meta = _local_track_or_404(index)
    try:
        peaks, duration = lc.waveform_peaks(meta["path"])
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"waveform failed: {exc}"), 500
    return jsonify(
        peaks=peaks,
        duration=duration,
        bpm=meta.get("bpm"),
        first_beat_sec=meta.get("first_beat_sec", 0.0),
        cues=meta.get("cues", []),
    )


@app.post("/api/track/<int:index>/cues")
def save_cues(index):
    """Persist manually-edited cues back onto a track (used by export)."""
    tracks = STATE["tracks"]
    if not (0 <= index < len(tracks)):
        return jsonify(error="Invalid track index."), 400
    payload = request.get_json(force=True, silent=True) or {}
    cleaned = []
    for c in payload.get("cues", []):
        try:
            start = float(c["start_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        cleaned.append({
            "name": str(c.get("name", "Cue"))[:40],
            "start_sec": round(max(0.0, start), 3),
            "kind": c.get("kind") if c.get("kind") in ("drop", "mix_in", "mix_out") else "cue",
        })
    cleaned.sort(key=lambda c: c["start_sec"])
    tracks[index]["cues"] = cleaned
    return jsonify(ok=True, cues=cleaned)


@app.post("/api/soundcloud")
def soundcloud_lookup():
    """Look up free-download / purchase status for the library's SoundCloud tracks."""
    tracks = STATE["tracks"]
    if not tracks:
        return jsonify(error="Load a library first."), 400
    payload = request.get_json(force=True, silent=True) or {}
    client_id = (payload.get("client_id") or os.environ.get("SOUNDCLOUD_CLIENT_ID") or "").strip()
    client_secret = (payload.get("client_secret") or os.environ.get("SOUNDCLOUD_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        return jsonify(error="Enter your SoundCloud client_id and client_secret "
                             "(or set SOUNDCLOUD_CLIENT_ID / SOUNDCLOUD_CLIENT_SECRET)."), 400

    sc_tracks = sum(1 for t in tracks if sc.is_soundcloud(t.get("location") or t.get("path")))
    if not sc_tracks:
        return jsonify(error=(
            "No SoundCloud tracks detected in this library. "
            f"Location formats found: {_location_formats(tracks)}. "
            "If your SoundCloud tracks use a different format, tell me one Location value "
            "and I'll match it."
        )), 400

    try:
        results = sc.enrich_library(tracks, client_id, client_secret, limit=payload.get("limit"))
    except sc.SoundCloudError as exc:
        return jsonify(error=str(exc)), 400

    vals = list(results.values())
    return jsonify(_library_response(
        tracks,
        sc_streaming=sc_tracks,
        sc_checked=len(results),
        sc_free=sum(1 for v in vals if v.get("kind") == "free"),
        sc_buy=sum(1 for v in vals if v.get("kind") == "buy"),
        sc_errors=sum(1 for v in vals if v.get("kind") == "error"),
    ))


@app.get("/api/export.xml")
def export_xml():
    """Download the scanned library as a Rekordbox XML with beatgrid + hot cues."""
    tracks = [t for t in STATE["tracks"] if t.get("cues")]
    if not tracks:
        return jsonify(error="No analyzed tracks with cues — scan a folder first."), 400
    tree = lc.build_rekordbox_xml(tracks)
    import xml.etree.ElementTree as ET
    ET.indent(tree, space="  ")
    buf = io.BytesIO()
    tree.write(buf, encoding="UTF-8", xml_declaration=True)
    buf.seek(0)
    return send_file(
        buf, mimetype="application/xml", as_attachment=True, download_name="rekordbox_auto_cues.xml"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
