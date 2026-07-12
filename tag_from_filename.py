#!/usr/bin/env python3
"""
tag_from_filename.py

Fills in artist/title tags for a folder of audio files by parsing the
filename itself -- no external API or internet lookup needed. Built for
libraries full of EDM edits/VIPs/bootlegs that won't show up in any
metadata database anyway.

Anything that doesn't cleanly match a known filename pattern gets logged
to needs_review.csv instead of a bad guess getting written.

SETUP
-----
    pip install mutagen --break-system-packages

USAGE
-----
    python tag_from_filename.py "/path/to/music/folder" --dry-run

    Drop --dry-run once the preview looks right to actually write tags.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.id3 import ID3, TIT2, TPE1, ID3NoHeaderError
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.wave import WAVE

AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".wav", ".aiff", ".aif"}

# Junk tokens commonly stuck in EDM edit filenames -- stripped before parsing
JUNK_PATTERNS = [
    r"\bfree ?download\b", r"\bofficial\b", r"\blyric(s)?\b", r"\baudio\b",
    r"\bhq\b", r"\bhd\b", r"\bexplicit\b", r"\bclean\b", r"\bout now\b",
    r"\bunreleased\b", r"\bpremiere\b", r"\bmastered?\b", r"\bfinal\b",
    r"\bv\d+\b", r"\b\d{3,4}kbps\b", r"\bwav\b", r"\bmp3\b",
]

# Remix/edit/VIP credit keywords -- used to pull an artist name out of a
# trailing "(Artist Remix)" style tag when there's no dash to split on.
CREDIT_KEYWORDS = r"(?:remix|edit|vip|flip|bootleg|mashup|rework)"


def clean_junk(name: str) -> str:
    cleaned = name
    for pat in JUNK_PATTERNS:
        cleaned = re.sub(rf"[\[\(\-_\s]*{pat}[\]\)\-_\s]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -_")
    return cleaned


def parse_filename(filename: str):
    """
    Returns (artist, title, ok) where ok=False means no confident
    artist/title split was found and this should go to the review file.
    """
    stem = Path(filename).stem
    name = clean_junk(stem)

    # 1) Artist - Title  (most explicit, check first)
    m = re.match(r"^(?P<artist>.+?)\s+-\s+(?P<title>.+)$", name)
    if m:
        artist, title = m.group("artist").strip(), m.group("title").strip()
        if artist and title:
            return artist, title, True

    # 2) Artist_Title (underscore separator survived cleaning)
    m = re.match(r"^(?P<artist>.+?)\s+_\s+(?P<title>.+)$", name)
    if m:
        artist, title = m.group("artist").strip(), m.group("title").strip()
        if artist and title:
            return artist, title, True

    # 3) Title (Artist Remix/Edit/VIP/etc.) -- no dash, but a remix credit
    #    in parens. Pull the artist out of the credit, keep the full
    #    "Title (Artist Remix)" string as the title.
    m = re.match(rf"^(?P<title_part>.+?)\s*\((?P<credit_artist>[^()]+?)\s+{CREDIT_KEYWORDS}\)$",
                 name, flags=re.IGNORECASE)
    if m:
        artist = m.group("credit_artist").strip()
        title = name.strip()  # keep the full "Title (Artist Remix)" as title
        if artist and title:
            return artist, title, True

    # 4) Title (Artist) -- generic parens with no credit keyword, lower confidence
    #    but still usable, e.g. "Some Song (GRiZ)"
    m = re.match(r"^(?P<title>.+?)\s*\((?P<artist>[^()]+)\)$", name)
    if m:
        artist, title = m.group("artist").strip(), m.group("title").strip()
        if artist and title:
            return artist, title, True

    # No separator found at all -- can't confidently split artist from title
    return "", name.strip(), False


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


def main():
    parser = argparse.ArgumentParser(description="Tag audio files by parsing artist/title from the filename.")
    parser.add_argument("folder", help="Folder containing audio files (searched recursively)")
    parser.add_argument("--dry-run", action="store_true", help="Preview parsed tags without writing them")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
    print(f"Found {len(files)} audio files.\n")

    review_rows = []
    tagged = 0

    for f in files:
        artist, title, ok = parse_filename(f.name)

        if ok:
            print(f"[{f.name}]\n  -> Artist: {artist}\n  -> Title:  {title}")
            if not args.dry_run:
                if write_tags(f, artist, title):
                    tagged += 1
            else:
                tagged += 1
        else:
            print(f"[{f.name}]\n  ? No clear artist/title split -- flagged for review (guessed title: '{title}')")
            review_rows.append([f.name, artist, title])

    if review_rows:
        review_path = folder / "needs_review.csv"
        with open(review_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "guessed_artist", "guessed_title"])
            writer.writerows(review_rows)
        print(f"\n{len(review_rows)} files need manual review -> {review_path}")

    print(f"\nDone. {tagged}/{len(files)} tagged {'(dry run, nothing written)' if args.dry_run else ''}.")


if __name__ == "__main__":
    main()