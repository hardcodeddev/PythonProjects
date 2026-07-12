#!/usr/bin/env python3
"""
classify_set_role.py

Analyzes each audio file's energy and brightness, then ranks tracks
relative to the rest of the folder to suggest a set-building role:
    Start  = lowest energy third  (groove/opener material)
    Build  = middle energy third  (rising energy)
    End    = highest energy third (peak/climax material)

IMPORTANT: Rekordbox's My Tag checkboxes live in Rekordbox's own database,
not in the audio file, so this script can't tick those boxes directly.
Instead it writes "Set Role: Start/Build/End" into the file's Comment
tag, which Rekordbox reads natively as a sortable/filterable column --
you can glance at that and manually flip the matching My Tag checkbox
much faster than guessing from scratch, or just filter/search by Comment
text directly.

This is a heuristic, not a definitive answer -- energy alone doesn't
capture the full "narrative role" of a track (an anthemic opener can be
loud too). Tracks near a tier boundary get flagged in review.csv so you
can eyeball and adjust those by hand.

SETUP
-----
    pip install librosa soundfile mutagen numpy --break-system-packages

USAGE
-----
    python classify_set_role.py "/path/to/music/folder" --dry-run

    Drop --dry-run once the review.csv rankings look right to actually
    write the Comment tags.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import librosa
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, COMM, ID3NoHeaderError
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.wave import WAVE
from mutagen.aiff import AIFF

AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".wav", ".aiff", ".aif"}

# How close to a tier boundary (in percentile) counts as "flag for review"
BOUNDARY_MARGIN = 0.07


def analyze_track(filepath: Path):
    """Returns (rms_energy, brightness) or None if the file can't be read."""
    try:
        y, sr = librosa.load(filepath, sr=22050, mono=True)
    except Exception as e:
        print(f"  ! Couldn't decode {filepath.name}: {e}")
        return None

    if y.size == 0:
        return None

    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return rms, centroid


def combined_score(rms, centroid, rms_range, centroid_range):
    """Normalize both metrics to 0-1 and blend them into one energy score."""
    rms_min, rms_max = rms_range
    c_min, c_max = centroid_range
    rms_norm = (rms - rms_min) / (rms_max - rms_min) if rms_max > rms_min else 0.5
    c_norm = (centroid - c_min) / (c_max - c_min) if c_max > c_min else 0.5
    return (rms_norm * 0.7) + (c_norm * 0.3)  # loudness weighted higher than brightness


def write_comment(filepath: Path, label: str) -> bool:
    ext = filepath.suffix.lower()
    comment_text = f"Set Role: {label}"
    try:
        if ext == ".mp3":
            try:
                tags = ID3(filepath)
            except ID3NoHeaderError:
                tags = ID3()
            tags.delall("COMM")
            tags.add(COMM(encoding=3, lang="eng", desc="", text=comment_text))
            tags.save(filepath)

        elif ext == ".flac":
            audio = FLAC(filepath)
            audio["comment"] = comment_text
            audio.save()

        elif ext == ".m4a":
            audio = MP4(filepath)
            audio["\xa9cmt"] = [comment_text]
            audio.save()

        elif ext == ".wav":
            audio = WAVE(filepath)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.delall("COMM")
            audio.tags.add(COMM(encoding=3, lang="eng", desc="", text=comment_text))
            audio.save()

        elif ext in (".aiff", ".aif"):
            audio = AIFF(filepath)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.delall("COMM")
            audio.tags.add(COMM(encoding=3, lang="eng", desc="", text=comment_text))
            audio.save()

        else:
            audio = MutagenFile(filepath, easy=True)
            if audio is None:
                return False
            audio["comment"] = comment_text
            audio.save()
        return True
    except Exception as e:
        print(f"  ! Failed to write comment for {filepath.name}: [{type(e).__name__}] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Suggest Start/Build/End set roles based on audio energy.")
    parser.add_argument("folder", help="Folder containing audio files (searched recursively)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing Comment tags")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    files = sorted(
        f for f in folder.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS and not f.name.startswith("._")
    )
    print(f"Found {len(files)} audio files. Analyzing (this can take a while for big libraries)...\n")

    results = []
    for f in files:
        print(f"Analyzing {f.name}...")
        metrics = analyze_track(f)
        if metrics is None:
            continue
        rms, centroid = metrics
        results.append({"path": f, "rms": rms, "centroid": centroid})

    if not results:
        print("No files could be analyzed.")
        sys.exit(1)

    rms_vals = [r["rms"] for r in results]
    centroid_vals = [r["centroid"] for r in results]
    rms_range = (min(rms_vals), max(rms_vals))
    centroid_range = (min(centroid_vals), max(centroid_vals))

    for r in results:
        r["score"] = combined_score(r["rms"], r["centroid"], rms_range, centroid_range)

    results.sort(key=lambda r: r["score"])
    n = len(results)
    for i, r in enumerate(results):
        percentile = i / (n - 1) if n > 1 else 0.5
        r["percentile"] = percentile
        if percentile < 1 / 3:
            r["label"] = "Start"
        elif percentile < 2 / 3:
            r["label"] = "Build"
        else:
            r["label"] = "End"
        # Flag anything close to a tier boundary for manual review
        r["near_boundary"] = (
            abs(percentile - 1 / 3) < BOUNDARY_MARGIN or
            abs(percentile - 2 / 3) < BOUNDARY_MARGIN
        )

    review_rows = []
    written = 0
    print("\n--- Results (sorted low to high energy) ---")
    for r in results:
        flag = "  <-- close to tier boundary, worth double-checking" if r["near_boundary"] else ""
        print(f"[{r['label']:>5}] {r['path'].name} (energy percentile: {r['percentile']:.2f}){flag}")
        review_rows.append([r["path"].name, r["label"], f"{r['percentile']:.2f}", r["near_boundary"]])

        if not args.dry_run:
            if write_comment(r["path"], r["label"]):
                written += 1
        else:
            written += 1

    review_path = folder / "set_role_review.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "energy_percentile", "near_boundary"])
        writer.writerows(review_rows)

    print(f"\nFull ranking saved to {review_path}")
    print(f"Done. {written}/{len(results)} tagged {'(dry run, nothing written)' if args.dry_run else ''}.")
    print("\nReminder: after writing, use Rekordbox's 'Reload Tags from File' on these tracks")
    print("to pull the new Comment text in without touching existing cues/My Tags.")


if __name__ == "__main__":
    main()