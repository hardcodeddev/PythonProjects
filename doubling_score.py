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

CLI (input can be a JSON list of track dicts, or a rekordbox library XML export):
  python doubling_score.py tracks.json --top 3 --audio
  python doubling_score.py rekordbox.xml --top 3 --skip-missing
See _build_arg_parser() / main() for the batch-runner contract.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import unquote

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


# Pitch class (0-11) for every note spelling we expect to see. Flats are written
# as "<letter>B" because musical_key_to_camelot() rewrites the '♭' symbol to 'B'.
_PITCH_CLASS = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}
# Camelot code per pitch class, derived from the wheel of fifths (each fifth up
# advances the Camelot number by one). Major = "B" ring, minor = "A" ring.
_MAJOR_CAMELOT = {
    0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B",
    6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B",
}
_MINOR_CAMELOT = {
    9: "8A", 4: "9A", 11: "10A", 6: "11A", 1: "12A", 8: "1A",
    3: "2A", 10: "3A", 5: "4A", 0: "5A", 7: "6A", 2: "7A",
}


def musical_key_to_camelot(key: Optional[str]) -> Optional[str]:
    """Normalize a key string to a Camelot code ('8A' / '12B').

    Accepts Camelot codes directly ('8A', '12B') and the musical notations
    rekordbox exports in its Tonality field: 'Am', 'C', 'F#m', 'Dbm', 'Ab',
    'A minor', 'C major', etc. (case-insensitive, '♯'/'♭' symbols accepted).
    Returns None if the key can't be recognized.
    """
    if not key:
        return None
    s = str(key).strip().upper()
    if not s:
        return None

    # Already a Camelot code?
    parsed = _parse_camelot(s)
    if parsed is not None:
        num, letter = parsed
        return f"{num}{letter}"

    # Normalize accidental symbols to their ASCII spellings ('D♭' -> 'DB').
    s = s.replace("♯", "#").replace("♭", "B")

    # Determine mode: default major; strip an explicit minor/major suffix.
    is_minor = False
    for suf in ("MINOR", "MIN", "M"):
        if s.endswith(suf):
            is_minor = True
            s = s[: -len(suf)]
            break
    else:
        for suf in ("MAJOR", "MAJ"):
            if s.endswith(suf):
                s = s[: -len(suf)]
                break

    pitch_class = _PITCH_CLASS.get(s.strip())
    if pitch_class is None:
        return None
    return _MINOR_CAMELOT[pitch_class] if is_minor else _MAJOR_CAMELOT[pitch_class]


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
    # Accept both Camelot codes and musical notation ('Am', 'C', 'F#m', ...).
    code_a = musical_key_to_camelot(key_a)
    code_b = musical_key_to_camelot(key_b)
    a = _parse_camelot(code_a) if code_a else None
    b = _parse_camelot(code_b) if code_b else None
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
    return _spectral_from_envelopes(_band_energy_envelopes(y_a, sr), _band_energy_envelopes(y_b, sr))


def _spectral_from_envelopes(env_a: dict, env_b: dict) -> float:
    """Weighted complementary-pocket score from two tracks' band envelopes."""
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
    return _rhythm_from_onsets(
        _onset_density(y_a, sr),
        _onset_density(y_b, sr),
        librosa.onset.onset_strength(y=y_a, sr=sr),
        librosa.onset.onset_strength(y=y_b, sr=sr),
    )


def _rhythm_from_onsets(
    dens_a: float, dens_b: float, str_a: np.ndarray, str_b: np.ndarray
) -> float:
    """Rhythm-pocket score from two tracks' onset densities + strength envelopes."""
    # (1) Sparseness. Both must be dense to clash -> use the smaller density.
    crowd = min(dens_a, dens_b)
    span = _DENSITY_HIGH - _DENSITY_LOW
    sparseness = float(np.clip(1.0 - (crowd - _DENSITY_LOW) / span, 0.0, 1.0))

    # (2) Interleave. Continuous onset-strength envelope is less noisy than
    # discrete onsets for correlation.
    interleave = (1.0 - _safe_corr(str_a, str_b)) / 2.0

    return float(np.clip(0.5 * sparseness + 0.5 * interleave, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-track audio features (extracted once, reused across every pairing so a
# whole-library audio scan stays O(n) STFTs instead of O(n^2)).
# ---------------------------------------------------------------------------
@dataclass
class TrackFeatures:
    band_env: dict          # per-band RMS energy envelope over time
    onset_strength: np.ndarray
    onset_density: float    # onsets/sec


def extract_features(y: np.ndarray, sr: int) -> TrackFeatures:
    """Compute the audio features the pocket scores need, once per track."""
    _require_librosa()
    return TrackFeatures(
        band_env=_band_energy_envelopes(y, sr),
        onset_strength=librosa.onset.onset_strength(y=y, sr=sr),
        onset_density=_onset_density(y, sr),
    )


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
def _verdict_for(score: float) -> str:
    if score >= 0.80:
        return "strong double candidate"
    if score >= 0.60:
        return "worth testing live"
    if score >= 0.40:
        return "risky — needs EQ carving"
    return "not recommended"


def _apply_path_remap(path: Optional[str], remap: Optional[tuple[str, str]]) -> Optional[str]:
    """Rewrite a track path prefix (e.g. a library recorded on another machine).

    remap is a (from_prefix, to_prefix) pair; paths that don't start with
    from_prefix are returned unchanged.
    """
    if not path or not remap:
        return path
    frm, to = remap
    if frm and path.startswith(frm):
        return to + path[len(frm):]
    return path


@dataclass
class DoublingEngine:
    """Configurable doubling-compatibility scorer.

    The four sub-score weights are instance attributes so a caller (e.g. the web
    UI) can tune them live without touching module globals. Weights are
    normalized by their sum at score time, so they need not add up to 1.0.
    """

    weight_key: float = WEIGHT_KEY
    weight_bpm: float = WEIGHT_BPM
    weight_spectral: float = WEIGHT_SPECTRAL_POCKET
    weight_rhythm: float = WEIGHT_RHYTHM_POCKET
    bpm_tolerance_pct: float = 2.0

    def _weights(self) -> tuple[float, float, float, float]:
        raw = (self.weight_key, self.weight_bpm, self.weight_spectral, self.weight_rhythm)
        total = sum(raw)
        if total <= 0:
            # Degenerate config — fall back to the module defaults.
            return WEIGHT_KEY, WEIGHT_BPM, WEIGHT_SPECTRAL_POCKET, WEIGHT_RHYTHM_POCKET
        return tuple(w / total for w in raw)  # type: ignore[return-value]

    def score_features(
        self,
        track_a_meta: dict,
        track_b_meta: dict,
        feat_a: Optional[TrackFeatures] = None,
        feat_b: Optional[TrackFeatures] = None,
    ) -> DoublingResult:
        """Score one pair from metadata + optional cached audio features."""
        key_score = camelot_distance(
            track_a_meta.get("camelot_key"), track_b_meta.get("camelot_key")
        )
        bpm_score = bpm_doubling_score(
            track_a_meta.get("bpm"), track_b_meta.get("bpm"), tolerance_pct=self.bpm_tolerance_pct
        )

        have_audio = feat_a is not None and feat_b is not None
        if have_audio:
            spec_score = _spectral_from_envelopes(feat_a.band_env, feat_b.band_env)
            rhythm_score = _rhythm_from_onsets(
                feat_a.onset_density, feat_b.onset_density,
                feat_a.onset_strength, feat_b.onset_strength,
            )
        else:
            spec_score = AUDIO_UNAVAILABLE_DEFAULT
            rhythm_score = AUDIO_UNAVAILABLE_DEFAULT

        w_key, w_bpm, w_spec, w_rhythm = self._weights()
        total = (
            w_key * key_score
            + w_bpm * bpm_score
            + w_spec * spec_score
            + w_rhythm * rhythm_score
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

    def score_pair(
        self,
        track_a_meta: dict,
        track_b_meta: dict,
        y_a: Optional[np.ndarray] = None,
        y_b: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> DoublingResult:
        """Score one pair. track_*_meta needs at least bpm / camelot_key / title.

        If y_a/y_b/sr are provided the spectral + rhythm pockets are measured;
        otherwise they fall back to a neutral default and needs_audio_analysis
        is set True so callers know the estimate is key/bpm-only.
        """
        feat_a = feat_b = None
        if y_a is not None and y_b is not None and sr is not None:
            feat_a = extract_features(y_a, sr)
            feat_b = extract_features(y_b, sr)
        return self.score_features(track_a_meta, track_b_meta, feat_a, feat_b)

    def load_features(
        self,
        meta: dict,
        sr: Optional[int] = None,
        remap: Optional[tuple[str, str]] = None,
    ) -> Optional[TrackFeatures]:
        """Load a drop excerpt and extract its features, tolerating bad files."""
        path = _apply_path_remap(meta.get("path"), remap)
        if not path or librosa is None:
            return None
        try:
            y, used_sr = load_drop_section(path, sr=sr, offset_sec=meta.get("drop_offset_sec"))
            return extract_features(y, used_sr)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill a batch
            print(f"  ! audio analysis failed for {path}: {exc}", file=sys.stderr)
            return None

    def rank(
        self,
        tracks: list[dict],
        top_n: int = 5,
        use_audio: bool = False,
        sr: Optional[int] = None,
        remap: Optional[tuple[str, str]] = None,
    ) -> list[dict]:
        """Rank every track's top-N doubling partners as JSON-ready records.

        Each record carries the source track's metadata plus a `candidates`
        list where every entry merges the partner's metadata with the pair's
        sub-scores — everything the web UI needs in one payload. When audio is
        enabled, each track's features are extracted once (O(n)) and reused
        across all pairings so the whole-library scan stays fast.
        """
        feats: list[Optional[TrackFeatures]] = [None] * len(tracks)
        if use_audio:
            for i, meta in enumerate(tracks):
                feats[i] = self.load_features(meta, sr=sr, remap=remap)

        records: list[dict] = []
        for i, meta_a in enumerate(tracks):
            candidates: list[dict] = []
            for j, meta_b in enumerate(tracks):
                if i == j:
                    continue
                res = self.score_features(meta_a, meta_b, feats[i], feats[j])
                candidates.append(
                    {
                        "partner_index": j,
                        "title": meta_b.get("title", f"track_{j}"),
                        "artist": meta_b.get("artist"),
                        "bpm": meta_b.get("bpm"),
                        "camelot_key": meta_b.get("camelot_key"),
                        "score": res.score,
                        "key_score": res.key_score,
                        "bpm_score": res.bpm_score,
                        "spectral_pocket_score": res.spectral_pocket_score,
                        "rhythm_pocket_score": res.rhythm_pocket_score,
                        "verdict": res.verdict,
                        "needs_audio_analysis": res.needs_audio_analysis,
                    }
                )
            candidates.sort(key=lambda c: c["score"], reverse=True)
            records.append(
                {
                    "index": i,
                    "title": meta_a.get("title", f"track_{i}"),
                    "artist": meta_a.get("artist"),
                    "bpm": meta_a.get("bpm"),
                    "camelot_key": meta_a.get("camelot_key"),
                    "candidates": candidates[:top_n],
                }
            )
        return records


# Default engine (module-level weights). The standalone functions below delegate
# to it so existing callers keep working unchanged.
_DEFAULT_ENGINE = DoublingEngine()


def score_doubling_pair(
    track_a_meta: dict,
    track_b_meta: dict,
    y_a: Optional[np.ndarray] = None,
    y_b: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> DoublingResult:
    """Score a single pair with the default engine (see DoublingEngine.score_pair)."""
    return _DEFAULT_ENGINE.score_pair(track_a_meta, track_b_meta, y_a, y_b, sr)


# ---------------------------------------------------------------------------
# Batch runner (mirrors the CLI pattern used in apply_review.py)
# ---------------------------------------------------------------------------
def _load_audio_for(meta: dict, sr: Optional[int]) -> Optional[TrackFeatures]:
    """Load + extract a track's audio features, tolerating missing files/librosa."""
    return _DEFAULT_ENGINE.load_features(meta, sr=sr)


def batch_top_candidates(
    tracks: list[dict],
    top_n: int = 5,
    use_audio: bool = False,
    sr: Optional[int] = None,
) -> dict[str, list[DoublingResult]]:
    """For each track, return its top-N doubling partners sorted by score.

    Back-compatible view over DoublingEngine.rank(), keyed by track title and
    returning DoublingResult objects (used by the CLI).
    """
    results: dict[str, list[DoublingResult]] = {}
    for rec in _DEFAULT_ENGINE.rank(tracks, top_n=top_n, use_audio=use_audio, sr=sr):
        src = tracks[rec["index"]]
        results[rec["title"]] = [
            DoublingResult(
                track_a=rec["title"],
                track_b=c["title"],
                score=c["score"],
                key_score=c["key_score"],
                bpm_score=c["bpm_score"],
                spectral_pocket_score=c["spectral_pocket_score"],
                rhythm_pocket_score=c["rhythm_pocket_score"],
                verdict=c["verdict"],
                needs_audio_analysis=c["needs_audio_analysis"],
            )
            for c in rec["candidates"]
        ]
    return results


# ---------------------------------------------------------------------------
# Input loading (JSON track list or rekordbox library XML)
# ---------------------------------------------------------------------------
def _rekordbox_location_to_path(location: Optional[str]) -> Optional[str]:
    """Decode a rekordbox Location attribute into a filesystem path.

    Locations look like 'file://localhost/Users/dj/Music/track.mp3' and are
    URL-encoded; on Windows they carry a drive letter ('/C:/...').
    """
    if not location:
        return None
    p = unquote(location)
    if p.startswith("file://"):
        p = p[len("file://"):]
        if p.startswith("localhost"):
            p = p[len("localhost"):]
    if re.match(r"^/[A-Za-z]:/", p):  # '/C:/Music' -> 'C:/Music'
        p = p[1:]
    return p


def load_rekordbox_xml(
    path: str,
    *,
    skip_missing_bpm: bool = False,
    skip_missing_key: bool = False,
) -> list[dict]:
    """Parse a rekordbox exported library XML into track metadata dicts
    compatible with score_doubling_pair() / batch_top_candidates().

    rekordbox stores tempo in the TRACK AverageBpm attribute and key in
    Tonality (musical notation, converted here to Camelot). File paths come
    from the URL-encoded Location attribute. Set skip_missing_* to drop tracks
    that lack a usable BPM or key rather than scoring them on partial data.
    """
    collection = ET.parse(path).getroot().find("COLLECTION")
    if collection is None:
        raise ValueError(f"{path}: no <COLLECTION> element (not a rekordbox library XML?)")

    tracks: list[dict] = []
    for tr in collection.findall("TRACK"):
        name = tr.get("Name") or ""
        artist = tr.get("Artist") or ""
        title = f"{artist} - {name}".strip(" -") if artist else name

        bpm_raw = tr.get("AverageBpm")
        try:
            bpm = float(bpm_raw) if bpm_raw else None
        except ValueError:
            bpm = None

        camelot = musical_key_to_camelot(tr.get("Tonality"))

        if skip_missing_bpm and not bpm:
            continue
        if skip_missing_key and not camelot:
            continue

        tracks.append(
            {
                "title": title or tr.get("TrackID") or "?",
                "artist": artist,
                "name": name,
                "path": _rekordbox_location_to_path(tr.get("Location")),
                "bpm": bpm,
                "camelot_key": camelot,
                "track_id": tr.get("TrackID"),
            }
        )
    return tracks


def load_tracks(path: str, fmt: str = "auto", skip_missing: bool = False) -> list[dict]:
    """Load track metadata from a JSON list or a rekordbox XML export.

    fmt: 'json', 'rekordbox', or 'auto' (detect by extension, then by sniffing
    whether the file starts with '<').
    """
    if fmt == "auto":
        lower = path.lower()
        if lower.endswith(".xml"):
            fmt = "rekordbox"
        elif lower.endswith(".json"):
            fmt = "json"
        else:
            with open(path) as fh:
                fmt = "rekordbox" if fh.read(256).lstrip().startswith("<") else "json"

    if fmt == "rekordbox":
        return load_rekordbox_xml(path, skip_missing_bpm=skip_missing, skip_missing_key=skip_missing)

    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("JSON track file must contain a list of metadata dicts.")
    if skip_missing:
        data = [t for t in data if t.get("bpm") and t.get("camelot_key")]
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank doubling (simultaneous-drop) candidates for a track library."
    )
    parser.add_argument(
        "tracks_file",
        help="A JSON file (list of track metadata dicts with "
        '"title"/"path"/"bpm"/"camelot_key"[/"drop_offset_sec"]) '
        "or a rekordbox library XML export.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json", "rekordbox"),
        default="auto",
        dest="fmt",
        help="Input format (default: auto-detect by extension/content).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop tracks that lack a usable BPM or key instead of scoring them.",
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

    try:
        tracks = load_tracks(args.tracks_file, fmt=args.fmt, skip_missing=args.skip_missing)
    except (OSError, ValueError, json.JSONDecodeError, ET.ParseError) as exc:
        print(f"Failed to load tracks from {args.tracks_file}: {exc}", file=sys.stderr)
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
