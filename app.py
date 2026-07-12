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

import os
import tempfile
from dataclasses import asdict

from flask import Flask, jsonify, render_template, request

import doubling_score as ds

app = Flask(__name__)

# In-memory library for the current session (single-user, local use).
STATE: dict = {"tracks": [], "source": None}

_TRUE = {"1", "true", "on", "yes", "True"}
_SUMMARY_FIELDS = ("title", "artist", "bpm", "camelot_key", "path", "streaming")


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
    return jsonify(
        source=file.filename,
        count=len(tracks),
        with_key=sum(1 for t in tracks if t.get("camelot_key")),
        with_bpm=sum(1 for t in tracks if t.get("bpm")),
        with_path=sum(1 for t in tracks if t.get("path")),
        streaming=sum(1 for t in tracks if t.get("streaming")),
        tracks=[{k: t.get(k) for k in _SUMMARY_FIELDS} for t in tracks],
    )


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
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Ranking failed: {exc}"), 500
    return jsonify(records=records, count=len(tracks))


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
