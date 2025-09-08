# finalize_lineup_agent.py
import os, sys, time, re, json
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from unidecode import unidecode

# --- Optional: point pytesseract to the tesseract binary explicitly (Windows) ---
# Set an env var once and keep code portable:
#   setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
_TESS = os.environ.get("TESSERACT_CMD")
if _TESS:
    pytesseract.pytesseract.tesseract_cmd = _TESS

# --- Optional RapidFuzz import; fallback to stdlib difflib if not available ---
try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    import difflib

    class _FuzzShim:
        @staticmethod
        def token_sort_ratio(a: str, b: str) -> int:
            ta = " ".join(sorted(a.lower().split()))
            tb = " ".join(sorted(b.lower().split()))
            return int(difflib.SequenceMatcher(None, ta, tb).ratio() * 100)

        @staticmethod
        def token_set_ratio(a: str, b: str) -> int:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            ja, jb = " ".join(sorted(sa)), " ".join(sorted(sb))
            return int(difflib.SequenceMatcher(None, ja, jb).ratio() * 100)

    class _ProcessShim:
        @staticmethod
        def extractOne(query, choices, scorer):
            best, best_score = None, -1
            for c in choices:
                s = scorer(query, c)
                if s > best_score:
                    best, best_score = c, s
            return (best, best_score, None) if best is not None else (None, 0, None)

    fuzz = _FuzzShim()
    process = _ProcessShim()

# --- Ollama (agent) ---
import ollama  # pip install ollama

# -----------------------------
# Utility / small example tools
# -----------------------------
def add_two_numbers(a: int, b: int) -> int:
    return a + b

def fetch_url(method: str, url: str) -> str:
    r = requests.request(method=method, url=url, timeout=15)
    return r.text[:1000]

# -----------------------------------
# Heuristics & OCR for lineup posters
# -----------------------------------
LOCAL_GENRE_MAP: Dict[str, List[str]] = {
    "ganja white night": ["dubstep", "melodic bass"],
    "boogie t": ["dubstep", "funk bass"],
    "liquid stranger": ["bass", "dubstep"],
    "excision": ["dubstep", "riddim"],
    "subtronics": ["dubstep", "riddim"],
    "svdden death": ["dubstep", "riddim"],
    "rezz": ["midtempo", "bass"],
    "griz": ["funk", "electro-soul", "bass"],
    "zeds dead": ["bass", "dubstep"],
    # Extend with your common artists for instant/accurate genres
}

def looks_like_lineup_filename(s: str) -> bool:
    s = s.lower()
    return any(k in s for k in ["lineup", "poster", "announce", "schedule", "phase"])

def guess_image_priority(img_tag) -> float:
    score = 0.0
    src = img_tag.get("src") or img_tag.get("data-src") or ""
    if looks_like_lineup_filename(src):
        score += 3.0
    try:
        w = int(img_tag.get("width") or 0)
        h = int(img_tag.get("height") or 0)
        area = w * h
        if area > 500_000:
            score += 2.0
        elif area > 200_000:
            score += 1.0
    except Exception:
        pass
    alt = (img_tag.get("alt") or "").lower()
    if any(k in alt for k in ["lineup", "poster", "phase"]):
        score += 2.0
    return score

def find_poster_image_url(page_url: str) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(page_url, timeout=25, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Consider common OG image too
    og = soup.find("meta", property="og:image")
    og_url = og.get("content") if og and og.get("content") else None
    best, best_score = (og_url, 2.5) if og_url else (None, -1.0)

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        full = src if src.startswith("http") else urljoin(page_url, src)
        s = guess_image_priority(img)
        if s > best_score:
            best_score, best = s, full

    return best

def download_image(url: str, folder: str = "lineup_images") -> Optional[str]:
    try:
        os.makedirs(folder, exist_ok=True)
        resp = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        name = os.path.basename(urlparse(url).path) or "lineup.jpg"
        if not os.path.splitext(name)[1]:
            name += ".jpg"
        path = os.path.join(folder, name)
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception as e:
        print(f"[download_image] {e}", flush=True)
        return None

# --- OCR helpers ---
def _prep_for_ocr(img: Image.Image) -> Image.Image:
    # Simple, effective preprocessing for busy posters
    g = ImageOps.grayscale(img)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    # Light threshold (tune 170–200 depending on posters)
    g = g.point(lambda p: 255 if p > 185 else 0)
    return g

def normalize_name(name: str) -> str:
    name = unidecode(name)
    name = re.sub(r"[^A-Za-z0-9\-’'&\.\+\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def is_plausible_artist_token(t: str) -> bool:
    if not t:
        return False
    words = t.split()
    if len(words) > 6:
        return False
    if sum(ch.isdigit() for ch in t) > 2:
        return False
    bad = {"and","with","b2b","live","stage","hosted","special","guest",
           "presents","takeover","vip","back-to-back","camp","village"}
    if set(w.lower() for w in words).issubset(bad):
        return False
    return True

def dedupe_artists(names: List[str]) -> List[str]:
    cleaned = [normalize_name(n) for n in names if is_plausible_artist_token(normalize_name(n))]
    out: List[str] = []
    for n in cleaned:
        if not out:
            out.append(n); continue
        match, score, _ = process.extractOne(n, out, scorer=fuzz.token_sort_ratio) or (None, 0, None)
        if score < 88:
            out.append(n)
    return out

def ocr_artists_from_image(img_path: str) -> List[str]:
    img = Image.open(img_path)
    img = _prep_for_ocr(img)
    raw = pytesseract.image_to_string(img)

    # Split into candidate tokens
    lines: List[str] = []
    for ln in raw.splitlines():
        ln = (ln or "").strip()
        if not ln:
            continue
        parts = re.split(r"[•·|/•]+", ln)
        lines.extend(p.strip() for p in parts if p and p.strip())

    artists = [ln for ln in lines if is_plausible_artist_token(ln)]
    return dedupe_artists(artists)

# --- Genres ---
def enrich_from_local_map(artist: str) -> Optional[List[str]]:
    key = artist.lower()
    if key in LOCAL_GENRE_MAP:
        return LOCAL_GENRE_MAP[key]
    match, score, _ = process.extractOne(key, LOCAL_GENRE_MAP.keys(), scorer=fuzz.token_set_ratio) or (None, 0, None)
    if match and score >= 92:
        return LOCAL_GENRE_MAP[match]
    return None

def musicbrainz_search_artist(name: str, pause: float = 0.4) -> Optional[str]:
    try:
        time.sleep(pause)  # polite
        r = requests.get(
            "https://musicbrainz.org/ws/2/artist/",
            params={"query": name, "fmt": "json", "limit": 1},
            headers={"User-Agent": "FestivalLineupAgent/1.0 (contact: you@example.com)"},
            timeout=20,
        )
        r.raise_for_status()
        js = r.json()
        arts = js.get("artists") or []
        if arts:
            return arts[0].get("id")
    except Exception:
        return None
    return None

def musicbrainz_genres_for_artist(mbid: str, pause: float = 0.4) -> List[str]:
    try:
        time.sleep(pause)
        r = requests.get(
            f"https://musicbrainz.org/ws/2/artist/{mbid}",
            params={"inc": "tags", "fmt": "json"},
            headers={"User-Agent": "FestivalLineupAgent/1.0 (contact: you@example.com)"},
            timeout=20,
        )
        r.raise_for_status()
        js = r.json()
        tags = js.get("tags") or []
        genres = [t.get("name","").lower() for t in tags if t.get("count",0) >= 1]
        return sorted(list(dict.fromkeys([g for g in genres if g])))[:6]
    except Exception:
        return []

def enrich_genres(artists: List[str], use_musicbrainz: bool, mb_max_lookups: int) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    # First pass: local only
    unknowns: List[str] = []
    for a in artists:
        local = enrich_from_local_map(a)
        if local:
            result[a] = local
        else:
            result[a] = []
            unknowns.append(a)

    if not use_musicbrainz or not unknowns:
        return result

    # Second pass: limited MusicBrainz lookups for unknowns
    for a in unknowns[:mb_max_lookups]:
        mbid = musicbrainz_search_artist(a)
        if mbid:
            genres = musicbrainz_genres_for_artist(mbid)
            if genres:
                result[a] = genres
    return result

# --- End-to-end ---
def scrape_lineup(
    page_url: str,
    use_musicbrainz: bool = False,    # default OFF for speed
    max_artists: int = 60,            # cap OCR results to keep runtime sane
    mb_max_lookups: int = 25          # if MB enabled, only look up first N unknowns
) -> Dict:
    try:
        print("[scrape] Finding poster...", flush=True)
        img_url = find_poster_image_url(page_url)
        if not img_url:
            return {"ok": False, "error": "No lineup-like image found on page."}

        print(f"[scrape] Downloading {img_url}", flush=True)
        saved = download_image(img_url, folder="lineup_images")
        if not saved:
            return {"ok": False, "error": f"Couldn't download image: {img_url}"}

        print("[scrape] Running OCR...", flush=True)
        artists = ocr_artists_from_image(saved)
        if not artists:
            return {"ok": False, "image": saved, "error": "OCR returned no artist candidates."}

        artists = artists[:max_artists]
        print(f"[scrape] OCR candidates: {len(artists)} (capped to {max_artists})", flush=True)

        print("[scrape] Enriching genres (local + optional MB)...", flush=True)
        genres_map = enrich_genres(artists, use_musicbrainz=use_musicbrainz, mb_max_lookups=mb_max_lookups)
        enriched = [{"name": a, "genres": genres_map.get(a, [])} for a in artists]

        return {"ok": True, "image": saved, "count": len(enriched), "artists": enriched}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Expose as a tool for the agent
def get_lineup_from_page(url: str, use_musicbrainz: bool = False, max_artists: int = 60, mb_max_lookups: int = 25) -> str:
    res = scrape_lineup(
        page_url=url,
        use_musicbrainz=use_musicbrainz,
        max_artists=max_artists,
        mb_max_lookups=mb_max_lookups
    )
    return json.dumps(res, ensure_ascii=False)

# -----------------
# Agent run script
# -----------------
if __name__ == "__main__":
    print("Write your query:", flush=True)
    userMessage = input().strip()

    # Keep tool list minimal to bias the model towards the one we want
    available = {
        "add_two_numbers": add_two_numbers,
        "fetch_url": fetch_url,
        "get_lineup_from_page": get_lineup_from_page,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agent that MUST call one of the provided tools whenever possible. "
                "If the user asks to scrape a lineup or a festival page, ALWAYS call get_lineup_from_page "
                "with the given URL. Prefer not to explain Selenium; prefer calling the tool."
            ),
        },
        {"role": "user", "content": userMessage},
    ]

    # Register only the tools we actually want used
    resp = ollama.chat(
        model="llama3.1",
        messages=messages,
        tools=[add_two_numbers, fetch_url, get_lineup_from_page],
    )

    messages.append(resp.message)

    # Execute tool calls (single pass)
    for call in (resp.message.tool_calls or []):
        fn = available.get(call.function.name)
        if not fn:
            continue
        out = fn(**call.function.arguments)
        messages.append({
            "role": "tool",
            "content": str(out),
            "name": call.function.name,
        })

    # Final summarization turn: disable tools to avoid loops
    messages.append({
        "role": "system",
        "content": (
            "Now produce a short JSON summary strictly from the latest tool output. "
            "Do NOT call any tools. If the tool returned JSON, return that JSON verbatim."
        ),
    })

    final = ollama.chat(
        model="llama3.1",
        messages=messages,
        tools=[]  # important: no tools in final turn
    )

    print(final.message.content, flush=True)
    sys.exit(0)
