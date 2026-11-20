# Comic OCR

Turns comic and manga pages into transcripts you can use for text analysis.

**Use it here:** https://yann-ryan.github.io/comic_ocr/

1. Open the link (Chrome, Edge or Firefox on a laptop or desktop).
2. Drop in some pages (JPG, PNG or WebP).
3. Choose the reading direction: left to right for Western comics, right to
   left for manga.
4. Click **Transcribe**, then download the transcript as CSV, JSON or plain text.

Each line of text comes out in reading order and tagged as **dialogue**,
**caption**, **sfx** (sound effect) or **other** (page numbers, signs). By
default the downloads contain dialogue and captions only.

## Your pages stay on your computer

Everything runs inside your browser tab. Your pages are not uploaded
anywhere: the site only downloads its own code and models (about 25 MB, once;
the browser caches them).

## How well does it work?

Tested on 32 hand-transcribed pages from six US comics and nine English
translated manga (12 of them held back and only used for the final check):

| | held-out pages |
|---|---|
| words correctly recovered (F1) | 0.97 |
| speech bubbles / captions read correctly | 96% |
| stray words per page | 2 |

Speed is roughly 5-15 seconds per page on an ordinary laptop.

Known weak spots: panels with slanted borders or double-page spreads can come
out in the wrong order; captions without a box (floating narration) are
tagged as dialogue; most drawn sound effects are skipped on purpose.

## How it works

1. A trained detector finds text blocks and speech bubbles on the page.
2. PaddleOCR models find the lines in each block and read them.
3. A dictionary pass fixes typical comic-lettering misreadings
   ("YOLI" -> "YOU") and splits run-together words.
4. The page is cut into panels at the gutters, and text is put in reading
   order.
5. Text in rectangular boxes is tagged as caption, the rest as dialogue.

All models run with ONNX Runtime Web (WebAssembly). See
THIRD_PARTY_NOTICES.md for model and library credits and licences.

## Running it yourself

The site is plain static files. Serve the folder with any web server, for
example `python3 -m http.server` and open http://localhost:8000/. Opening
index.html directly as a file will not work (browsers block the model
downloads).
