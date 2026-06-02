# Comic OCR

A browser-based OCR pipeline for comic books and manga. No server required — everything runs locally in your browser using WebAssembly.

🔗 **Live demo:** https://yann-ryan.github.io/comic_ocr/

## Features

- **Automatic text detection** — finds speech bubbles, captions, and floating text using OpenCV (via Pyodide/WASM)
- **Panel detection & editing** — detects page panels automatically; add, move, resize or delete panels in the editor
- **Interactive bounding box editor** — select, move, resize, draw new, and delete regions before running OCR
- **Region types** — classify regions as Bubble, Caption, Floating, or Panel
- **Multi-page support** — load a PDF or multiple images as a document; detect and OCR all pages, navigate between them to edit
- **Tesseract OCR** — runs entirely in the browser via Tesseract.js
- **Tunable detection** — sliders for brightness threshold, panel area, NMS, stroke filters, and more
- **Download results** — export transcript as JSON, CSV, or plain text; download annotated images

## Usage

1. Open the app in a browser (Chrome or Edge recommended)
2. Drop a comic page image (JPG/PNG/TIFF) or PDF onto the upload zone
3. Click **🔍 DETECT REGIONS** — OpenCV finds text blocks
4. Edit the detected boxes if needed (select, move, resize, draw new)
5. Click **⚡ RUN OCR** — Tesseract reads the text
6. Download the transcript

### Multi-page documents

- Select multiple images or a PDF to load as a document
- Use **🔍 DETECT ALL PAGES** to process every page
- Navigate between pages to review/edit regions
- Use **⚡ OCR ALL PAGES** to transcribe everything at once

### Tuning parameters

| Slider | Effect |
|--------|--------|
| Brightness | Threshold for bubble background brightness |
| Min panel size | Minimum panel area as fraction of page |
| NMS threshold | Controls overlap merging of detected regions |
| Min letter strokes | Minimum ink components per text block |
| Min stroke width | Filters out very thin lines (hatching) |
| Min ink density | Minimum ink fill ratio in a text block |
| Line gap % | Vertical merging distance — raise for large bold manga text |

## Technical details

- **Detection:** Python + OpenCV running in [Pyodide](https://pyodide.org/) (WebAssembly)
- **OCR:** [Tesseract.js](https://tesseract.projectnaptha.com/)
- **PDF rendering:** [PDF.js](https://mozilla.github.io/pdf.js/)
- No build step — single self-contained HTML file
- All processing happens client-side; no data is sent to any server

## Browser compatibility

Works best in **Chrome** or **Edge**. Firefox works but may be slower. Safari has limited WebAssembly support.

> **Note:** First load downloads ~30MB of WebAssembly libraries (OpenCV, Tesseract language data). Subsequent loads use the browser cache.

## Limitations

- OCR quality depends on image resolution and font style
- Works best on scanned comics at 150+ DPI
- Manga with highly stylised fonts may need PSM tuning (try PSM 6 or PSM 11)

## License

MIT
