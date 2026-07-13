"""
library_cues.py

Point at folder(s) of audio files (.mp3/.wav/.aiff/.aif/.flac), auto-detect each
track's tempo, key and structure, and place hot cues at the musically useful mix
points — the drops and the mix-in / mix-out phrases. The result is written as a
Rekordbox library XML (COLLECTION + per-track TEMPO beatgrid + POSITION_MARK hot
cues), which Rekordbox imports directly and which the wider ecosystem (Lexicon,
etc.) converts on to Serato / Traktor / Engine.

Nothing here mutates the source audio files — the cues live only in the exported
XML. Detected tracks are also plain doubling-engine metadata dicts, so a scanned
folder can be ranked for doubling by doubling_score.DoublingEngine straight away.

Cue detection is heuristic (energy + beat grid), meant as solid starting cues to
nudge by ear — not a claim of frame-perfect structural analysis.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote

import numpy as np

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:  # optional: nicer artist/title than the filename
    import mutagen
except ImportError:  # pragma: no cover
    mutagen = None

# Reuse the Camelot maps + key parsing from the scorer (single source of truth).
from doubling_score import _MAJOR_CAMELOT, _MINOR_CAMELOT, _BEATS_PER_PHRASE

AUDIO_EXTS = (".mp3", ".wav", ".aiff", ".aif", ".flac", ".m4a", ".ogg")

# Analysis sample rate — mono at 22.05k is plenty for beat/key/energy and keeps
# a full-library scan fast.
_ANALYSIS_SR = 22050
_RMS_HOP = 512

# Krumhansl-Schmuckler key profiles (major / minor), indexed from C.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Hot-cue colours (R,G,B) by kind, for a readable Rekordbox import.
_CUE_COLORS = {
    "mix_in": (40, 200, 120),   # green
    "drop": (224, 87, 91),      # red
    "mix_out": (75, 156, 245),  # blue
}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def scan_folders(folders, recursive: bool = True) -> list[str]:
    """Return audio file paths under the given folder(s), sorted, de-duplicated."""
    if isinstance(folders, (str, bytes)):
        folders = [folders]
    found: list[str] = []
    seen: set[str] = set()
    for folder in folders:
        folder = os.path.expanduser(str(folder).strip())
        if not folder or not os.path.isdir(folder):
            continue
        walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
        for root, _dirs, files in walker:
            for name in files:
                if name.lower().endswith(AUDIO_EXTS) and not name.startswith("."):
                    path = os.path.abspath(os.path.join(root, name))
                    if path not in seen:
                        seen.add(path)
                        found.append(path)
    return sorted(found)


# ---------------------------------------------------------------------------
# Tempo / key
# ---------------------------------------------------------------------------
def _title_artist_from(path: str) -> tuple[str, str]:
    """Best-effort artist/title from tags (if mutagen present) or the filename."""
    if mutagen is not None:
        try:
            f = mutagen.File(path, easy=True)
            if f:
                title = (f.get("title") or [None])[0]
                artist = (f.get("artist") or [None])[0]
                if title or artist:
                    return title or os.path.splitext(os.path.basename(path))[0], artist or ""
        except Exception:  # noqa: BLE001
            pass
    stem = os.path.splitext(os.path.basename(path))[0]
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return title.strip(), artist.strip()
    return stem, ""


def _normalize_bpm(bpm: float) -> float:
    """Fold obvious octave errors into a sane DJ range."""
    if bpm <= 0:
        return 0.0
    while bpm < 70:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return bpm


def detect_key(y: np.ndarray, sr: int) -> Optional[str]:
    """Estimate the Camelot key via Krumhansl-Schmuckler correlation on chroma."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if not np.any(chroma):
        return None

    def best(profile):
        scores = [np.corrcoef(np.roll(profile, k), chroma)[0, 1] for k in range(12)]
        k = int(np.nanargmax(scores))
        return k, scores[k]

    maj_pc, maj_score = best(_KS_MAJOR)
    min_pc, min_score = best(_KS_MINOR)
    if maj_score >= min_score:
        return _MAJOR_CAMELOT[maj_pc]
    return _MINOR_CAMELOT[min_pc]


# ---------------------------------------------------------------------------
# Cue detection
# ---------------------------------------------------------------------------
def _downbeats(beat_times: np.ndarray) -> np.ndarray:
    """Assume 4/4 and take every 4th beat as a downbeat (anchored at beat 0)."""
    return beat_times[::4] if len(beat_times) else beat_times


def _bar_energy(rms: np.ndarray, times: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """Mean RMS energy in the bar following each downbeat."""
    out = np.zeros(len(downbeats))
    for k, db in enumerate(downbeats):
        end = downbeats[k + 1] if k + 1 < len(downbeats) else (times[-1] if len(times) else db)
        mask = (times >= db) & (times < end)
        out[k] = rms[mask].mean() if mask.any() else 0.0
    return out


def detect_cue_points(y: np.ndarray, sr: int, beat_times: np.ndarray) -> list[dict]:
    """Place Drop and Mix-in / Mix-out cues on the track's energy + beat grid.

    Returns cue dicts {name, start_sec, kind} sorted by time. All cues snap to a
    downbeat; mix-in / mix-out snap to a 16-beat phrase boundary.
    """
    downbeats = _downbeats(beat_times)
    if len(downbeats) < 4:
        return []

    rms = librosa.feature.rms(y=y, hop_length=_RMS_HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=_RMS_HOP)
    energy = _bar_energy(rms, times, downbeats)
    peak = np.percentile(energy, 95) or float(energy.max() or 1.0)

    bars_per_phrase = _BEATS_PER_PHRASE // 4  # 16 beats -> 4 bars

    # --- Drops: bars whose energy jumps well above the preceding baseline. ---
    drops: list[tuple[int, float]] = []
    for k in range(2, len(energy)):
        base = np.median(energy[max(0, k - 8):k])
        if base > 0 and energy[k] > 1.4 * base and energy[k] > 0.55 * peak:
            drops.append((k, energy[k] / base))
    drops.sort(key=lambda t: t[1], reverse=True)
    chosen: list[int] = []
    for idx, _ratio in drops:
        if all(abs(idx - c) >= bars_per_phrase for c in chosen):  # dedupe within a phrase
            chosen.append(idx)
        if len(chosen) >= 3:
            break
    chosen.sort()

    # --- Mix-in: first phrase boundary where the track leaves the quiet intro. ---
    loud = 0.45 * peak
    first_loud = next((k for k, e in enumerate(energy) if e >= loud), 0)
    mix_in_bar = (first_loud // bars_per_phrase) * bars_per_phrase

    # --- Mix-out: phrase boundary at the start of the trailing quiet outro. ---
    last_loud = next((k for k in range(len(energy) - 1, -1, -1) if energy[k] >= loud), len(energy) - 1)
    mix_out_bar = min(len(downbeats) - 1, ((last_loud // bars_per_phrase) + 1) * bars_per_phrase)

    cues: list[dict] = []
    cues.append({"name": "Mix In", "start_sec": float(downbeats[mix_in_bar]), "kind": "mix_in"})
    for n, bar in enumerate(chosen):
        label = "Drop" if n == 0 else f"Drop {n + 1}"
        cues.append({"name": label, "start_sec": float(downbeats[bar]), "kind": "drop"})
    if mix_out_bar > mix_in_bar:
        cues.append({"name": "Mix Out", "start_sec": float(downbeats[mix_out_bar]), "kind": "mix_out"})

    # De-dupe cues on the same downbeat, preferring a Drop label over mix in/out.
    priority = {"drop": 0, "mix_in": 1, "mix_out": 2}
    seen: set[float] = set()
    unique = []
    for c in sorted(cues, key=lambda c: (c["start_sec"], priority.get(c["kind"], 9))):
        key = round(c["start_sec"], 2)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return sorted(unique, key=lambda c: c["start_sec"])


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------
def analyze_file(path: str, sr: int = _ANALYSIS_SR) -> dict:
    """Analyze one audio file into a track dict with detected cues.

    Keys match the doubling-engine metadata (title/artist/path/bpm/camelot_key)
    plus: duration_sec, first_beat_sec, cues[]. On failure returns a dict with
    an "error" key so a batch scan can carry on.
    """
    if librosa is None:
        raise ImportError("librosa is required for library cue analysis.")

    title, artist = _title_artist_from(path)
    base = {
        "title": f"{artist} - {title}".strip(" -") if artist else title,
        "artist": artist,
        "name": title,
        "path": path,
        "streaming": False,
    }
    try:
        y, used_sr = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:  # noqa: BLE001
        return {**base, "bpm": None, "camelot_key": None, "cues": [], "error": str(exc)}

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=used_sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=used_sr)
    bpm = round(_normalize_bpm(float(np.atleast_1d(tempo)[0])), 2)

    try:
        camelot = detect_key(y, used_sr)
    except Exception:  # noqa: BLE001
        camelot = None

    cues = detect_cue_points(y, used_sr, beat_times)

    return {
        **base,
        "bpm": bpm,
        "camelot_key": camelot,
        "duration_sec": round(len(y) / used_sr, 2),
        "first_beat_sec": round(float(beat_times[0]), 3) if len(beat_times) else 0.0,
        "cues": cues,
    }


def analyze_folders(folders, recursive: bool = True, max_files: Optional[int] = None,
                    progress=None) -> list[dict]:
    """Scan folder(s) and analyze every audio file found."""
    paths = scan_folders(folders, recursive=recursive)
    if max_files:
        paths = paths[:max_files]
    tracks = []
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), p)
        tracks.append(analyze_file(p))
    return tracks


# ---------------------------------------------------------------------------
# Rekordbox XML export
# ---------------------------------------------------------------------------
def _location_uri(path: str) -> str:
    """Local path -> rekordbox 'file://localhost/...' URL-encoded Location."""
    abs_path = os.path.abspath(path)
    # quote but keep the path separators; ensure a leading slash.
    return "file://localhost" + quote(abs_path)


def build_rekordbox_xml(tracks: list[dict]) -> ET.ElementTree:
    """Build a Rekordbox library XML tree with beatgrid + hot cues per track.

    Only tracks that have a local path are included. Cues come from each track's
    'cues' list; a TEMPO anchor is written from first_beat_sec + bpm.
    """
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="rekordbox", Version="doubling-engine", Company="AlphaTheta")

    usable = [t for t in tracks if t.get("path") and not t.get("streaming")]
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(usable)))
    playlist_keys: list[str] = []

    for i, t in enumerate(usable, start=1):
        track_id = str(t.get("track_id") or i)
        attrs = {
            "TrackID": track_id,
            "Name": t.get("name") or t.get("title") or "",
            "Artist": t.get("artist") or "",
            "Location": _location_uri(t["path"]),
            "Kind": os.path.splitext(t["path"])[1].lstrip(".").upper() + " File",
        }
        if t.get("bpm"):
            attrs["AverageBpm"] = f"{float(t['bpm']):.2f}"
        if t.get("camelot_key"):
            attrs["Tonality"] = t["camelot_key"]
        if t.get("duration_sec"):
            attrs["TotalTime"] = str(int(round(t["duration_sec"])))
        track_el = ET.SubElement(collection, "TRACK", attrs)

        # Beatgrid anchor.
        if t.get("bpm"):
            ET.SubElement(
                track_el, "TEMPO",
                Inizio=f"{float(t.get('first_beat_sec', 0.0)):.3f}",
                Bpm=f"{float(t['bpm']):.2f}", Metro="4/4", Battito="1",
            )

        # Hot cues.
        for num, cue in enumerate(t.get("cues", [])):
            r, g, b = _CUE_COLORS.get(cue.get("kind"), (200, 200, 200))
            mark = {
                "Name": cue.get("name", ""),
                "Type": "0",
                "Start": f"{float(cue['start_sec']):.3f}",
                "Num": str(num if num < 8 else -1),  # 0-7 hot cues, rest memory cues
            }
            if num < 8:
                mark.update(Red=str(r), Green=str(g), Blue=str(b))
            ET.SubElement(track_el, "POSITION_MARK", mark)

        playlist_keys.append(track_id)

    # A single playlist holding everything, so the import lands in one crate.
    playlists = ET.SubElement(root, "PLAYLISTS")
    root_node = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    pl = ET.SubElement(
        root_node, "NODE", Name="Auto Cues", Type="1", KeyType="0", Entries=str(len(playlist_keys))
    )
    for key in playlist_keys:
        ET.SubElement(pl, "TRACK", Key=key)

    return ET.ElementTree(root)


def write_rekordbox_xml(tracks: list[dict], out_path: str) -> int:
    """Write the Rekordbox XML to disk; returns the number of tracks exported."""
    tree = build_rekordbox_xml(tracks)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    return sum(1 for t in tracks if t.get("path") and not t.get("streaming"))
