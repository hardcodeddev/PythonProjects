#!/usr/bin/env python3
"""
apply_review.py

Takes the needs_review.csv produced by tag_from_filename.py (optionally
hand-edited first) and writes the artist/title columns into each file's
tags. Use this once you're happy with what's in the review file.

USAGE
-----
    python apply_review.py "/path/to/music/folder/needs_review.csv" --dry-run

    Drop --dry-run to actually write the tags.

NOTE: This looks for each file by name inside the SAME folder the CSV
lives in (searched recursively), so keep the CSV where it was generated.
"""

import argparse
import csv
import sys
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.id3 import ID3, TIT2, TPE1, ID3NoHeaderError
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.wave import WAVE


def write_tags(filepath: Path, artist: str, title: str) -> bool:
    ext = filepath.suffix.lower()
    try:
        if ext == ".mp3":
            try:
                tags = ID3(filepath)
            except ID3NoHeaderError:
                tags = ID3()
            tags["TIT2"] = TIT2(encoding=3, text=title)
            tags["TPE1"] = TPE1(encoding=3, text=artist)
            tags.save(filepath)

        elif ext == ".flac":
            audio = FLAC(filepath)
            audio["title"] = title
            audio["artist"] = artist
            audio.save()

        elif ext == ".m4a":
            audio = MP4(filepath)
            audio["\xa9nam"] = [title]
            audio["\xa9ART"] = [artist]
            audio.save()

        elif ext == ".wav":
            audio = WAVE(filepath)
            if audio.tags is None:
                audio.add_tags()
            audio.tags["TIT2"] = TIT2(encoding=3, text=title)
            audio.tags["TPE1"] = TPE1(encoding=3, text=artist)
            audio.save()

        else:
            audio = MutagenFile(filepath, easy=True)
            if audio is None:
                return False
            audio["title"] = title
            audio["artist"] = artist
            audio.save()
        return True
    except Exception as e:
        print(f"  ! Failed to write tags for {filepath.name}: {e}")
        return False


def find_file(root: Path, filename: str):
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Apply a reviewed needs_review.csv to file tags.")
    parser.add_argument("csv_path", help="Path to needs_review.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing tags")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"Not a file: {csv_path}")
        sys.exit(1)

    root = csv_path.parent
    applied = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} rows in {csv_path.name}.\n")

    for row in rows:
        filename = row.get("filename", "").strip()
        artist = row.get("guessed_artist", "").strip()
        title = row.get("guessed_title", "").strip()

        if not filename or not title:
            print(f"[skip] Missing filename or title in row: {row}")
            skipped += 1
            continue

        target = find_file(root, filename)
        if target is None:
            print(f"[skip] Couldn't find file: {filename}")
            skipped += 1
            continue

        print(f"[{filename}]\n  -> Artist: {artist or '(blank)'}\n  -> Title:  {title}")
        if not args.dry_run:
            if write_tags(target, artist, title):
                applied += 1
        else:
            applied += 1

    print(f"\nDone. {applied} applied, {skipped} skipped {'(dry run, nothing written)' if args.dry_run else ''}.")


if __name__ == "__main__":
    main()