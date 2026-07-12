"""
doubling_score.py

Adds a "doubling compatibility" score to the existing track chemistry system.
Doubling = layering two drops/basslines simultaneously (common in riddim/bass
music DJ sets), as distinct from a standard sequential transition.

Originally sketched as a spec/handoff; now implemented end to end. The four
sub-scores (key / bpm / spectral pocket / rhythm pocket) are each normalized
to [0.0, 1.0] and blended with the tunable weights at the top of the file.

Sub-scores:
  - camelot_distance()       harmonic fit, strict (clashes are loud when stacked)
  - bpm_doubling_score()     near-identical or clean 2x/0.5x tempo relationship
  - spectral_pocket_score()  do the two drops leave each other frequency headroom
  - rhythm_pocket_score()    do the transients interleave or pile up

When audio is available the module analyzes a ~16-bar drop excerpt (either the
loudest window it finds, or an explicit section boundary passed in) rather than
whole tracks, to keep batch scoring fast across a large library.

CLI:
  python doubling_score.py tracks.json --top 3 --audio
See _build_arg_parser() / main() for the batch-runner contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:  # librosa is only needed for the audio sub-scores / batch audio loading.
    import librosa
except ImportError:  # pragma: no cover - keeps key/bpm scoring usable standalone
    librosa = None

# ---------------------------------------------------------------------------
# Tunable weights — doubling leans harder on key + spectral pocket than a
# normal harmonic-mixing transition score would.
# ---------------------------------------------------------------------------
WEIGHT_KEY = 0.35
WEIGHT_BPM = 0.15
WEIGHT_SPECTRAL_POCKET = 0.30
WEIGHT_RHYTHM_POCKET = 0.20

# Neutral value used for the spectral / rhythm sub-scores when audio is not
# supplied. Kept as a constant so the key/bpm-only estimate is honest about
# what it is (see needs_audio_analysis on DoublingResult).
AUDIO_UNAVAILABLE_DEFAULT = 0.5

# BPM matching: how tight "near-identical" is, and where the decay floors out.
BPM_FLOOR_PCT = 8.0

# STFT config for the spectral pocket analysis. These operate on short drop
# excerpts, not full tracks, so the defaults stay cheap.
_STFT_N_FFT = 2048
_STFT_HOP = 512

# Frequency band edges (Hz). Sub-bass and bass are the clash zone where doubled
# riddim/dubstep basslines fight for the same sonic space.
_BANDS = {
    "sub": (0.0, 80.0),
    "bass": (80.0, 300.0),
    "low_mid": (300.0, 800.0),
    "mid_high": (800.0, 20000.0),
}
# Bass-heavy weighting: anti-correlation in the bass band matters most.
_BAND_WEIGHTS = {
    "sub": 0.25,
    "bass": 0.45,
    "low_mid": 0.20,
    "mid_high": 0.10,
}

# Onset density (onsets/sec) mapped to a "crowding" range. Below LOW the drop is
# sparse enough to leave pockets; above HIGH it is busy/cluttered.
_DENSITY_LOW = 2.0
_DENSITY_HIGH = 8.0

# Length (seconds) of the drop excerpt to analyze when we have to pick one.
_DROP_EXCERPT_SEC = 8.0


@dataclass
class DoublingResult:
    track_a: str
    track_b: str
    score: float
    key_score: float
    bpm_score: float
    spectral_pocket_score: float
    rhythm_pocket_score: float
    verdict: str  # human-readable recommendation
    # True when the spectral/rhythm sub-scores fell back to a neutral default
    # because no audio was supplied — the score is a key/bpm-only estimate.
    needs_audio_analysis: bool = False


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------
def _parse_camelot(key: str) -> Optional[tuple[int, str]]:
    """Parse a Camelot key like '8A' / '12B' into (number 1-12, letter 'A'/'B').

    Returns None if the string cannot be parsed as a Camelot key.
    """
    if not key:
        return None
    token = str(key).strip().upper()
    if len(token) < 2:
        return None
    letter = token[-1]
    if letter not in ("A", "B"):
        return None
    try:
        number = int(token[:-1])
    except ValueError:
        return None
    if not 1 <= number <= 12:
        return None
    return number, letter


def _wheel_step(a: int, b: int) -> int:
    """Shortest distance between two positions on the 12-hour Camelot wheel."""
    diff = abs(a - b) % 12
    return min(diff, 12 - diff)


def camelot_distance(key_a: str, key_b: str) -> float:
    """Score harmonic compatibility for DOUBLING (stricter than sequential mixing).

    Same key or relative major/minor scores highest; anything beyond one step on
    the wheel scores low, since a bad harmonic clash is much more audible when two
    tracks play at once vs. sequentially.

      same key                              -> 1.0
      relative major/minor (same number)    -> 0.9
      adjacent number, same letter (+/-1)   -> 0.7
      energy-boost (+7 same letter)         -> 0.5
      anything else                         -> 0.1
    """
    a = _parse_camelot(key_a)
    b = _parse_camelot(key_b)
    if a is None or b is None:
        # Unknown/unparseable key — treat as a likely clash rather than crashing
        # a batch run. Same conservative floor as an unrelated key.
        return 0.1

    num_a, let_a = a
    num_b, let_b = b

    if num_a == num_b and let_a == let_b:
        return 1.0
    if num_a == num_b and let_a != let_b:
        return 0.9  # relative major/minor
    if let_a == let_b and _wheel_step(num_a, num_b) == 1:
        return 0.7  # adjacent on the wheel, same mode
    if let_a == let_b and _wheel_step(num_a, num_b) == 5:
        # +/-7 semitones == 5 steps on the wheel of fifths ("energy boost")
        return 0.5
    return 0.1


# ---------------------------------------------------------------------------
# BPM
# ---------------------------------------------------------------------------
def _pct_diff(a: float, b: float) -> float:
    """Percentage difference of b from reference a."""
    if a <= 0:
        return float("inf")
    return abs(a - b) / a * 100.0


def _decay(pct: float, peak: float, tol: float, floor_pct: float = BPM_FLOOR_PCT) -> float:
    """Linear ramp: `peak` at/under tolerance, 0 at/beyond floor_pct."""
    if pct <= tol:
        return peak
    if pct >= floor_pct:
        return 0.0
    frac = (pct - tol) / (floor_pct - tol)
    return peak * (1.0 - frac)


def bpm_doubling_score(bpm_a: float, bpm_b: float, tolerance_pct: float = 2.0) -> float:
    """Acceptable BPM relationships for doubling:

      - near-identical BPM (within tolerance_pct)  -> up to 1.0
      - clean half-time / double-time (a ~= 2*b)   -> up to 0.85

    Most of the reference catalog sits near-identical; the 2x path covers VIPs /
    edits that report true tempo at 70-75 vs a 140-150 half-time counterpart.
    Scores decay linearly and floor at 0.0 beyond ~8% drift with no 2x fit.
    """
    if not bpm_a or not bpm_b or bpm_a <= 0 or bpm_b <= 0:
        return 0.0

    # Near-identical relationship.
    identical = _decay(_pct_diff(bpm_a, bpm_b), peak=1.0, tol=tolerance_pct)

    # Best 2x / 0.5x fit: fold b up or down an octave and compare.
    double_pct = min(_pct_diff(bpm_a, bpm_b * 2.0), _pct_diff(bpm_a, bpm_b * 0.5))
    doubled = _decay(double_pct, peak=0.85, tol=tolerance_pct)

    return float(np.clip(max(identical, doubled), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _require_librosa() -> None:
    if librosa is None:
        raise ImportError(
            "librosa is required for spectral/rhythm scoring and audio loading. "
            "Install it with `pip install librosa`."
        )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with guards for length mismatch and zero variance.

    Returns 0.0 (treated as neutral / uncorrelated) when either envelope is
    constant or too short to correlate.
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def _band_energy_envelopes(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Per-band RMS energy envelope over time via a single STFT."""
    stft = np.abs(librosa.stft(y, n_fft=_STFT_N_FFT, hop_length=_STFT_HOP))
    power = stft ** 2  # [freq_bins, frames]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_STFT_N_FFT)

    envelopes: dict[str, np.ndarray] = {}
    for name, (lo, hi) in _BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            envelopes[name] = np.zeros(power.shape[1])
            continue
        # RMS-like energy per frame within the band.
        envelopes[name] = np.sqrt(power[mask, :].mean(axis=0))
    return envelopes


def load_drop_section(
    path: str,
    sr: Optional[int] = None,
    offset_sec: Optional[float] = None,
    duration_sec: float = _DROP_EXCERPT_SEC,
) -> tuple[np.ndarray, int]:
    """Load a short drop excerpt for analysis.

    If `offset_sec` is given (e.g. a drop boundary from classify_set_role.py),
    load `duration_sec` starting there. Otherwise load the full track and slice
    out the loudest `duration_sec` window as a fallback drop estimate.
    """
    _require_librosa()

    if offset_sec is not None:
        y, sr = librosa.load(path, sr=sr, offset=offset_sec, duration=duration_sec, mono=True)
        return y, sr

    y, sr = librosa.load(path, sr=sr, mono=True)
    win = int(duration_sec * sr)
    if win <= 0 or len(y) <= win:
        return y, sr

    # Find the loudest window via an RMS envelope, then map back to samples.
    hop = _STFT_HOP
    rms = librosa.feature.rms(y=y, frame_length=_STFT_N_FFT, hop_length=hop)[0]
    frames_per_win = max(1, win // hop)
    if len(rms) <= frames_per_win:
        return y[:win], sr
    # Rolling sum of energy over a window-worth of frames.
    csum = np.concatenate([[0.0], np.cumsum(rms)])
    window_energy = csum[frames_per_win:] - csum[:-frames_per_win]
    start_frame = int(np.argmax(window_energy))
    start = start_frame * hop
    return y[start:start + win], sr


# ---------------------------------------------------------------------------
# Spectral pocket
# ---------------------------------------------------------------------------
def spectral_pocket_score(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> float:
    """Do the two drops leave each other frequency-band headroom, or do they both
    saturate the same range (usually the sub/bass where basslines live)?

    For each band we correlate the two energy envelopes over time. HIGH
    correlation means both tracks pump in the same band at the same moments —
    they fight for that space when stacked. LOW / anti-correlation means one
    ducks while the other pushes: a complementary pocket. The bass band is
    weighted most heavily since that is where doubled tracks typically clash.

    Returns 0-1, where 1.0 = strong complementary pocket fit.
    """
    _require_librosa()
    if len(y_a) == 0 or len(y_b) == 0:
        return 0.0

    env_a = _band_energy_envelopes(y_a, sr)
    env_b = _band_energy_envelopes(y_b, sr)

    total = 0.0
    for name, weight in _BAND_WEIGHTS.items():
        corr = _safe_corr(env_a[name], env_b[name])
        # Map correlation [-1, 1] -> pocket goodness [1, 0]:
        # anti-correlated (-1) is the best complementary fit.
        pocket = (1.0 - corr) / 2.0
        total += weight * pocket

    return float(np.clip(total, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Rhythm pocket
# ---------------------------------------------------------------------------
def _onset_density(y: np.ndarray, sr: int) -> float:
    """Discrete onsets per second."""
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    duration = len(y) / sr if sr else 0.0
    if duration <= 0:
        return 0.0
    return len(onsets) / duration


def rhythm_pocket_score(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> float:
    """Rhythmic sparseness / complementarity via onset analysis.

    Sparse, repetitive riddim-style drops leave room for a second track's
    transients to sit in the pockets; two dense/busy drops stacked together
    sound cluttered. And when hits are offset/interleaved they read as a "duet"
    groove, whereas coincident hits from both tracks read as chaotic.

    Two components, evenly blended:
      1. sparseness — driven by the *quieter* of the two onset densities, since
         a clash needs BOTH tracks to be busy; one sparse track leaves room.
      2. interleave — 1 minus the correlation of the two onset-strength
         envelopes; low/anti correlation = interleaved hits = good.

    Returns 0-1.
    """
    _require_librosa()
    if len(y_a) == 0 or len(y_b) == 0:
        return 0.0

    # (1) Sparseness. Both must be dense to clash -> use the smaller density.
    dens_a = _onset_density(y_a, sr)
    dens_b = _onset_density(y_b, sr)
    crowd = min(dens_a, dens_b)
    span = _DENSITY_HIGH - _DENSITY_LOW
    sparseness = 1.0 - (crowd - _DENSITY_LOW) / span
    sparseness = float(np.clip(sparseness, 0.0, 1.0))

    # (2) Interleave. Continuous onset-strength envelope is less noisy than
    # discrete onsets for correlation.
    str_a = librosa.onset.onset_strength(y=y_a, sr=sr)
    str_b = librosa.onset.onset_strength(y=y_b, sr=sr)
    corr = _safe_corr(str_a, str_b)
    interleave = (1.0 - corr) / 2.0

    return float(np.clip(0.5 * sparseness + 0.5 * interleave, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------
def _verdict_for(score: float) -> str:
    if score >= 0.80:
        return "strong double candidate"
    if score >= 0.60:
        return "worth testing live"
    if score >= 0.40:
        return "risky — needs EQ carving"
    return "not recommended"


def score_doubling_pair(
    track_a_meta: dict,
    track_b_meta: dict,
    y_a: Optional[np.ndarray] = None,
    y_b: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> DoublingResult:
    """Main entry point. track_*_meta expected to contain at minimum:
      { "path": str, "bpm": float, "camelot_key": str, "title": str }

    If y_a/y_b/sr are provided the spectral + rhythm pockets are analyzed;
    otherwise the result is a key/bpm-only estimate with needs_audio_analysis
    set True so callers know the pockets were defaulted, not measured.
    """
    key_score = camelot_distance(track_a_meta["camelot_key"], track_b_meta["camelot_key"])
    bpm_score = bpm_doubling_score(track_a_meta["bpm"], track_b_meta["bpm"])

    have_audio = y_a is not None and y_b is not None and sr is not None
    if have_audio:
        spec_score = spectral_pocket_score(y_a, y_b, sr)
        rhythm_score = rhythm_pocket_score(y_a, y_b, sr)
    else:
        spec_score = AUDIO_UNAVAILABLE_DEFAULT
        rhythm_score = AUDIO_UNAVAILABLE_DEFAULT

    total = (
        WEIGHT_KEY * key_score
        + WEIGHT_BPM * bpm_score
        + WEIGHT_SPECTRAL_POCKET * spec_score
        + WEIGHT_RHYTHM_POCKET * rhythm_score
    )

    return DoublingResult(
        track_a=track_a_meta.get("title", track_a_meta.get("path", "?")),
        track_b=track_b_meta.get("title", track_b_meta.get("path", "?")),
        score=round(total, 3),
        key_score=round(key_score, 3),
        bpm_score=round(bpm_score, 3),
        spectral_pocket_score=round(spec_score, 3),
        rhythm_pocket_score=round(rhythm_score, 3),
        verdict=_verdict_for(total),
        needs_audio_analysis=not have_audio,
    )


# ---------------------------------------------------------------------------
# Batch runner (mirrors the CLI pattern used in apply_review.py)
# ---------------------------------------------------------------------------
def _load_audio_for(meta: dict, sr: Optional[int]) -> Optional[tuple[np.ndarray, int]]:
    """Load a drop excerpt for a track, tolerating missing files/librosa."""
    path = meta.get("path")
    if not path or librosa is None:
        return None
    try:
        return load_drop_section(path, sr=sr, offset_sec=meta.get("drop_offset_sec"))
    except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill a batch
        print(f"  ! audio load failed for {path}: {exc}", file=sys.stderr)
        return None


def batch_top_candidates(
    tracks: list[dict],
    top_n: int = 5,
    use_audio: bool = False,
    sr: Optional[int] = None,
) -> dict[str, list[DoublingResult]]:
    """For each track, return its top-N doubling partners sorted by score.

    tracks: list of metadata dicts (see score_doubling_pair for required keys).
    use_audio: when True, load a drop excerpt per track once and reuse it across
    all pairings so scoring stays cheap over a large library.
    """
    loaded: list[Optional[tuple[np.ndarray, int]]] = [None] * len(tracks)
    if use_audio:
        for i, meta in enumerate(tracks):
            loaded[i] = _load_audio_for(meta, sr)

    results: dict[str, list[DoublingResult]] = {}
    for i, meta_a in enumerate(tracks):
        pair_results: list[DoublingResult] = []
        for j, meta_b in enumerate(tracks):
            if i == j:
                continue
            ya = yb = s = None
            if loaded[i] is not None and loaded[j] is not None:
                ya, sa = loaded[i]
                yb, sb = loaded[j]
                s = sa if sa == sb else None  # sr must match to compare
                if s is None:
                    ya = yb = None
            pair_results.append(score_doubling_pair(meta_a, meta_b, ya, yb, s))
        pair_results.sort(key=lambda r: r.score, reverse=True)
        key = meta_a.get("title", meta_a.get("path", f"track_{i}"))
        results[key] = pair_results[:top_n]
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank doubling (simultaneous-drop) candidates for a track library."
    )
    parser.add_argument(
        "tracks_json",
        help="Path to a JSON file: a list of track metadata dicts "
        '({"title","path","bpm","camelot_key"[, "drop_offset_sec"]}).',
    )
    parser.add_argument("--top", type=int, default=5, help="Top-N partners per track.")
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Load drop excerpts and compute spectral/rhythm pockets (needs librosa).",
    )
    parser.add_argument("--sr", type=int, default=None, help="Resample rate for audio loading.")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit results as JSON instead of a human-readable table.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    with open(args.tracks_json) as fh:
        tracks = json.load(fh)
    if not isinstance(tracks, list):
        print("tracks_json must contain a JSON list of track metadata.", file=sys.stderr)
        return 2

    results = batch_top_candidates(tracks, top_n=args.top, use_audio=args.audio, sr=args.sr)

    if args.as_json:
        print(json.dumps({k: [asdict(r) for r in v] for k, v in results.items()}, indent=2))
        return 0

    for track, candidates in results.items():
        print(f"\n{track}")
        if not candidates:
            print("  (no candidates)")
            continue
        for r in candidates:
            flag = " [key/bpm only]" if r.needs_audio_analysis else ""
            print(f"  {r.score:>5.3f}  {r.verdict:<26} -> {r.track_b}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
