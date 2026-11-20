# Third-party files

| File(s) | What | Source | Licence |
|---|---|---|---|
| `models/detector-v4-s_int8.onnx` | Comic text and speech-bubble detector (RT-DETR-v2) | [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) | Apache-2.0 |
| `models/ch_PP-OCRv4_det_infer.onnx` | Text line detector (PP-OCRv4) | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), ONNX conversion from [RapidOCR](https://github.com/RapidAI/RapidOCR) | Apache-2.0 |
| `models/en_PP-OCRv3_rec_infer.onnx`, `models/en_rec_chars.txt` | English text recogniser (PP-OCRv3) and its character list | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), ONNX conversion from [SWHL/RapidOCR](https://huggingface.co/SWHL/RapidOCR) | Apache-2.0 |
| `models/en_50k.txt` | English word frequencies (50,000 words), unmodified | Hermit Dave, [FrequencyWords](https://github.com/hermitdave/FrequencyWords) (OpenSubtitles 2018) | [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `vendor/ort.wasm.min.js`, `vendor/ort-wasm-simd-threaded.*` | ONNX Runtime Web 1.19.2 | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT, © Microsoft Corporation |
| `coi-serviceworker.js` | coi-serviceworker 0.1.7 (enables multi-threading on GitHub Pages) | [gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) | MIT, © Guido Zuidhof and contributors |
