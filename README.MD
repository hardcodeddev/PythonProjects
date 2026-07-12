# Doubling Compatibility Engine

`doubling_score.py` scores how well two tracks **double** — i.e. play their drops
layered simultaneously (common in riddim/bass DJ sets) rather than mixed
sequentially. It blends four normalized sub-scores:

- **Key** (`camelot_distance`) — strict harmonic fit; clashes are loud when stacked.
- **BPM** (`bpm_doubling_score`) — near-identical or a clean 2×/½× half-time fit.
- **Spectral pocket** (`spectral_pocket_score`) — do the two drops leave each other
  frequency-band headroom, or fight for the sub/bass? (bass-band weighted heaviest)
- **Rhythm pocket** (`rhythm_pocket_score`) — do their transients interleave, or pile up?

The reusable core is `DoublingEngine` (tunable weights + `rank()`). Input can be a
JSON track list or a **rekordbox library XML export** (`AverageBpm` → BPM,
`Tonality` → Camelot key via `musical_key_to_camelot`, `Location` → file path).

## Web app

A local Flask front end (`app.py` + `templates/index.html`) makes it usable
without the CLI: upload a rekordbox XML (or JSON), tune the sub-score weights
live, and browse each track's top doubling partners. Run the full audio
"pocket" analysis across the whole library, or deep-analyze a single pair on
demand.

```bash
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:5000
```

## CLI

```bash
python doubling_score.py rekordbox.xml --top 3 --skip-missing
python doubling_score.py tracks.json --top 3 --audio --json
```

---

# Festival Lineup Scraper

This project is a Python-based tool designed to scrape festival lineup posters from web pages, extract artist names using Optical Character Recognition (OCR), and enrich the data with genre information. The tool is modular and can be customized for various use cases.

## How It Works

### 1. **Image Fetching**

The tool identifies the most relevant image on a webpage (e.g., a lineup poster) by analyzing image tags and metadata. It uses heuristics to prioritize images based on their filenames, dimensions, and alt text.

- **Key Function:** `fetch_and_process_image`
- **Libraries Used:** `requests`, `BeautifulSoup`
- **Customization:** You can adjust the heuristics in `guess_image_priority` to better suit your target websites.

### 2. **OCR (Optical Character Recognition)**

The fetched image is preprocessed to enhance text readability, and OCR is performed to extract artist names.

- **Key Function:** `process_image_for_artists`
- **Libraries Used:** `Pillow`, `pytesseract`
- **Customization:** Modify `_prep_for_ocr` to fine-tune preprocessing for different poster styles.

### 3. **Genre Enrichment**

The extracted artist names are enriched with genre information using a local map and optional MusicBrainz lookups.

- **Key Function:** `enrich_genres`
- **Libraries Used:** `requests`, `rapidfuzz`
- **Customization:** Extend the `LOCAL_GENRE_MAP` dictionary to include more artists and genres.

### 4. **End-to-End Workflow**

The `scrape_lineup` function orchestrates the entire process, returning a JSON object with the results.

## Example Usage

```python
from tools import scrape_lineup

url = "https://example.com/festival-lineup"
result = scrape_lineup(url, use_musicbrainz=True, max_artists=50)
print(result)
```

## Improvements and Customization

- **Performance:**

  - Use a faster OCR library like `EasyOCR` for better performance.
  - Cache results to avoid redundant API calls.

- **Error Handling:**

  - Improve error messages for better debugging.
  - Add retries for network requests.

- **Scalability:**
  - Use a queue system like RabbitMQ for processing multiple URLs.
  - Deploy the tool as a web service using Flask or FastAPI.

## Helpful Documentation

- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
- [pytesseract Documentation](https://pypi.org/project/pytesseract/)
- [MusicBrainz API](https://musicbrainz.org/doc/Development/XML_Web_Service/Version_2)
- [RapidFuzz Documentation](https://maxbachmann.github.io/RapidFuzz/)

## Requirements

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## License

This project is licensed under the MIT License.
