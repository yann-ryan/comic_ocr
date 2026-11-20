// comic-ocr engine: a 1:1 port of src/pipeline_v2.py (--engine simple --fix,
// panel order, text types). Everything runs in the browser; no network calls
// except loading the model files that sit next to this script.
/* global ort */

// ───────────────────────── image helpers ─────────────────────────
// Images are {w, h, c, data}: interleaved Uint8Array/Float32Array. Page images
// are kept in BGR order (as OpenCV reads them in the Python reference).

export function imageDataToBGR(id) {
  const { width: w, height: h, data } = id;
  const out = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++, j += 3) {
    out[j] = data[i * 4 + 2]; out[j + 1] = data[i * 4 + 1]; out[j + 2] = data[i * 4];
  }
  return { w, h, c: 3, data: out };
}

// cv2.resize INTER_LINEAR for 8-bit images, reproduced bit-exactly: float32
// source coordinates, 11-bit fixed-point weights, OpenCV's vertical rounding.
const f32 = Math.fround;
function rint(v) { const r = Math.round(v); return (r - v === 0.5 && r % 2 !== 0) ? r - 1 : r; }
function linearTaps(dsize, ssize) {
  const scale = ssize / dsize, idx = new Int32Array(dsize), w0 = new Int32Array(dsize), w1 = new Int32Array(dsize);
  for (let d = 0; d < dsize; d++) {
    let f = f32(f32((d + 0.5) * scale) - 0.5);
    let s = Math.floor(f);
    f = f32(f - s);
    if (s < 0) { f = 0; s = 0; }
    if (s >= ssize - 1) { f = 0; s = ssize - 1; }
    idx[d] = s; w0[d] = rint(f32(1 - f) * 2048); w1[d] = rint(f * 2048);
  }
  return { idx, w0, w1 };
}
export function resize(img, dw, dh) {
  const { w: sw, h: sh, c, data } = img;
  const out = new Uint8Array(dw * dh * c);
  const X = linearTaps(dw, sw), Y = linearTaps(dh, sh);
  const rowCache = new Map();
  const hrow = sy => { // horizontal pass for one source row (ints scaled by 2048)
    let r = rowCache.get(sy);
    if (r) return r;
    r = new Int32Array(dw * c);
    const base = sy * sw;
    for (let x = 0; x < dw; x++) {
      const s0 = (base + X.idx[x]) * c, s1 = (base + Math.min(X.idx[x] + 1, sw - 1)) * c;
      for (let k = 0; k < c; k++) r[x * c + k] = data[s0 + k] * X.w0[x] + data[s1 + k] * X.w1[x];
    }
    if (rowCache.size > 4) rowCache.delete(rowCache.keys().next().value);
    rowCache.set(sy, r);
    return r;
  };
  for (let y = 0; y < dh; y++) {
    const r0 = hrow(Y.idx[y]), r1 = hrow(Math.min(Y.idx[y] + 1, sh - 1)), b0 = Y.w0[y], b1 = Y.w1[y];
    const o = y * dw * c;
    for (let i = 0; i < dw * c; i++) {
      const v = (((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16) + 2) >> 2;
      out[o + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return { w: dw, h: dh, c, data: out };
}

export function crop(img, x0, y0, x1, y1) {
  x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(img.w, x1); y1 = Math.min(img.h, y1);
  const w = Math.max(0, x1 - x0), h = Math.max(0, y1 - y0), c = img.c;
  const out = new Uint8Array(w * h * c);
  for (let y = 0; y < h; y++) out.set(img.data.subarray(((y0 + y) * img.w + x0) * c, ((y0 + y) * img.w + x0 + w) * c), y * w * c);
  return { w, h, c, data: out };
}

function gray(img) { // cv2 BGR2GRAY
  const n = img.w * img.h, g = new Uint8Array(n), d = img.data;
  for (let i = 0; i < n; i++) g[i] = Math.round(0.114 * d[i * 3] + 0.587 * d[i * 3 + 1] + 0.299 * d[i * 3 + 2]);
  return g;
}

// ───────────────────────── models ─────────────────────────
export class Models {
  static async load(base = 'models/', onProgress = () => {}) {
    const m = new Models();
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    onProgress('text detector'); m.detector = await ort.InferenceSession.create(base + 'detector-v4-s_int8.onnx', opt);
    onProgress('line finder'); m.lineDet = await ort.InferenceSession.create(base + 'ch_PP-OCRv4_det_infer.onnx', opt);
    onProgress('text reader'); m.rec = await ort.InferenceSession.create(base + 'en_PP-OCRv3_rec_infer.onnx', opt);
    onProgress('dictionary');
    const chars = (await (await fetch(base + 'en_rec_chars.txt')).text()).split('\n');
    if (chars[chars.length - 1] === '') chars.pop();
    m.chars = ['<blank>', ...chars, ' '];
    initDictionary(await (await fetch(base + 'en_50k.txt')).text());
    return m;
  }
}

// ───────────────────────── 1. text/bubble detector ─────────────────────────
const LABELS = ['bubble', 'text_bubble', 'text_free'];

async function detect(models, img, thr = 0.4) {
  const r = resize(img, 640, 640), n = 640 * 640, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { // BGR -> RGB, /255, CHW
    x[i] = r.data[i * 3 + 2] / 255; x[n + i] = r.data[i * 3 + 1] / 255; x[2 * n + i] = r.data[i * 3] / 255;
  }
  const out = await models.detector.run({
    images: new ort.Tensor('float32', x, [1, 3, 640, 640]),
    orig_target_sizes: new ort.Tensor('int64', BigInt64Array.from([BigInt(img.w), BigInt(img.h)]), [1, 2]),
  });
  const labels = out.labels.data, boxes = out.boxes.data, scores = out.scores.data, res = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] < thr) continue;
    res.push({
      label: LABELS[Number(labels[i])] ?? String(labels[i]), score: scores[i],
      box: [Math.max(0, boxes[i * 4]), Math.max(0, boxes[i * 4 + 1]), Math.min(img.w, boxes[i * 4 + 2]), Math.min(img.h, boxes[i * 4 + 3])],
    });
  }
  return res;
}

function overlap(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return ix * iy / Math.max(Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-6);
}

function nms(dets, thr = 0.6) {
  const keep = [];
  for (const d of [...dets].sort((a, b) => b.score - a.score)) if (keep.every(k => overlap(d.box, k.box) < thr)) keep.push(d);
  return keep;
}

// ───────────────────────── 2. line finder + reader (PP-OCR) ─────────────────────────
const DET_MEAN = [0.485, 0.456, 0.406], DET_STD = [0.229, 0.224, 0.225];

function letterbox(img, minH = 30, ratio = 8) {
  const { w, h } = img;
  if (h <= minH || w / h > ratio) {
    const newH = Math.max(Math.trunc(w / ratio), minH) * 2, pad = Math.trunc(Math.abs(newH - h) / 2);
    const out = new Uint8Array(w * (h + 2 * pad) * 3);
    out.set(img.data, pad * w * 3);
    return { w, h: h + 2 * pad, c: 3, data: out };
  }
  return img;
}

function pyRound(v) { // Python round(): half to even
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

async function detectLines(models, img) {
  const { w, h } = img;
  const r = Math.min(h, w) < 736 ? 736 / Math.min(h, w) : 1.0;
  const rh = Math.max(32, Math.trunc(pyRound(h * r / 32) * 32)), rw = Math.max(32, Math.trunc(pyRound(w * r / 32) * 32));
  const im = resize(img, rw, rh), n = rw * rh, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) x[k * n + i] = (im.data[i * 3 + k] / 255 - DET_MEAN[k]) / DET_STD[k];
  const out = await models.lineDet.run({ x: new ort.Tensor('float32', x, [1, 3, rh, rw]) });
  const prob = out[models.lineDet.outputNames[0]].data;
  // threshold + dilate 2x2 (anchor at the kernel centre -> offsets -1..0)
  const bm0 = new Uint8Array(n);
  for (let i = 0; i < n; i++) bm0[i] = prob[i] > 0.3 ? 1 : 0;
  const bm = new Uint8Array(n);
  for (let y = 0; y < rh; y++) for (let xx = 0; xx < rw; xx++) {
    const i = y * rw + xx;
    bm[i] = bm0[i] | (xx > 0 ? bm0[i - 1] : 0) | (y > 0 ? bm0[i - rw] : 0) | (xx > 0 && y > 0 ? bm0[i - rw - 1] : 0);
  }
  // connected components (4-conn), raster order labels
  const lab = new Int32Array(n).fill(-1), stack = new Int32Array(n), boxes = [];
  for (let s = 0; s < n; s++) {
    if (!bm[s] || lab[s] !== -1) continue;
    let sp = 0, x0 = rw, y0 = rh, x1 = 0, y1 = 0, sum = 0, cnt = 0;
    stack[sp++] = s; lab[s] = boxes.length;
    while (sp) {
      const i = stack[--sp], yy = (i / rw) | 0, xx = i - yy * rw;
      if (xx < x0) x0 = xx; if (xx > x1) x1 = xx; if (yy < y0) y0 = yy; if (yy > y1) y1 = yy;
      sum += prob[i]; cnt++;
      if (xx > 0 && bm[i - 1] && lab[i - 1] === -1) { lab[i - 1] = lab[s]; stack[sp++] = i - 1; }
      if (xx < rw - 1 && bm[i + 1] && lab[i + 1] === -1) { lab[i + 1] = lab[s]; stack[sp++] = i + 1; }
      if (yy > 0 && bm[i - rw] && lab[i - rw] === -1) { lab[i - rw] = lab[s]; stack[sp++] = i - rw; }
      if (yy < rh - 1 && bm[i + rw] && lab[i + rw] === -1) { lab[i + rw] = lab[s]; stack[sp++] = i + rw; }
    }
    boxes.push({ b: [x0, y0, x1 + 1, y1 + 1], score: sum / cnt });
  }
  const kept = boxes.filter(o => Math.min(o.b[2] - o.b[0], o.b[3] - o.b[1]) >= 3 && o.score >= 0.5).map(o => o.b);
  kept.sort((a, b) => a[1] - b[1]);
  const lines = [];
  for (const b of kept) {
    const L = lines.find(L => Math.min(L[3], b[3]) - Math.max(L[1], b[1]) > 0.5 * Math.min(L[3] - L[1], b[3] - b[1]));
    if (L) { L[0] = Math.min(L[0], b[0]); L[1] = Math.min(L[1], b[1]); L[2] = Math.max(L[2], b[2]); L[3] = Math.max(L[3], b[3]); }
    else lines.push([...b]);
  }
  const sx = w / rw, sy = h / rh;
  const res = lines.map(([x0, y0, x1, y1]) => {
    const bw = x1 - x0, bh = y1 - y0, d = bw * bh * 1.6 / (2 * (bw + bh));
    return [Math.max(0, (x0 - d) * sx), Math.max(0, (y0 - d) * sy), Math.min(w, (x1 + d) * sx), Math.min(h, (y1 + d) * sy)];
  });
  res.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  return res;
}

async function recognize(models, img) {
  if (img.h < 2 || img.w < 2) return ['', 0];
  const rw = Math.max(1, Math.ceil(48 * img.w / img.h));
  const im = resize(img, rw, 48), n = rw * 48, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) x[k * n + i] = (im.data[i * 3 + k] / 255 - 0.5) / 0.5;
  const out = await models.rec.run({ x: new ort.Tensor('float32', x, [1, 3, 48, rw]) });
  const t = out[models.rec.outputNames[0]], [, T, C] = t.dims, p = t.data;
  let text = '', ssum = 0, scnt = 0, prev = -1;
  for (let s = 0; s < T; s++) {
    let bi = 0, bv = -Infinity;
    for (let k = 0; k < C; k++) if (p[s * C + k] > bv) { bv = p[s * C + k]; bi = k; }
    if (bi !== prev && bi !== 0) { text += models.chars[bi]; ssum += bv; scnt++; }
    prev = bi;
  }
  return [text, scnt ? ssum / scnt : 0];
}

async function readBlock(models, img) {
  img = letterbox(img);
  const parts = [];
  for (const [x0, y0, x1, y1] of await detectLines(models, img)) {
    const [text, score] = await recognize(models, crop(img, Math.trunc(x0), Math.trunc(y0), Math.ceil(x1), Math.ceil(y1)));
    if (text.trim() && score >= 0.5) parts.push(text);
  }
  return parts.join(' ');
}

export function clean(text) {
  return text.split(/\s+/).filter(Boolean).join(' ').replace(/(\w)- (\w)/g, '$1$2');
}

// ───────────────────────── 3. dictionary clean-up (src/textfix.py) ─────────────────────────
let FREQ = new Map(), TOTAL = 0, MAXLEN = 0;
const SHORT_OK = new Set('a i o am an as at be by do go he if in is it me my no of oh ok on or so to up us we ah eh hm uh um ya yo ye'.split(' '));
const SUFFIX = new Set(['t', 's', 'm', 'd', 'll', 're', 've']);
const MIN_SEG = 300, UNKNOWN_PER_CHAR = 3.5, SUB_PENALTY = 4.0;
const SWAPS = [['LI', 'U'], ['LL', 'U'], ['L', 'U'], ['0', 'O'], ['1', 'I'], ['5', 'S'], ['|', 'I']];
const fixCache = new Map();

function initDictionary(txt) {
  FREQ = new Map(); TOTAL = 0; MAXLEN = 0;
  for (const line of txt.split('\n')) {
    const k = line.lastIndexOf(' ');
    if (k < 0) continue;
    const w = line.slice(0, k), c = parseInt(line.slice(k + 1), 10);
    if (/^[a-z]+$/.test(w)) { FREQ.set(w, c); TOTAL += c; if (w.length > MAXLEN) MAXLEN = w.length; }
  }
  fixCache.clear();
}

function cost(w, seg = false) {
  if (w.includes("'")) {
    const k = w.indexOf("'"), base = w.slice(0, k), suf = w.slice(k + 1);
    if (!SUFFIX.has(suf) || !base) return null;
    const c = cost(base, seg);
    return c === null ? null : c + 0.5;
  }
  const c = FREQ.get(w);
  if (c === undefined || (w.length <= 2 && !SHORT_OK.has(w))) return null;
  if (seg && c < MIN_SEG) return null;
  return -Math.log(c / TOTAL);
}

function known(word) {
  const parts = word.toLowerCase().split('-').filter(Boolean);
  return parts.length > 0 && parts.every(p => cost(p) !== null || (FREQ.has(p) && p.length > 2));
}

function segment(s) {
  const n = s.length, best = [[0, -1]];
  for (let i = 1; i <= n; i++) best.push([Infinity, -1]);
  for (let i = 1; i <= n; i++) for (let j = Math.max(0, i - MAXLEN - 3); j < i; j++) {
    const c = cost(s.slice(j, i), true);
    if (c !== null && best[j][0] + c + 3.0 < best[i][0]) best[i] = [best[j][0] + c + 3.0, j];
  }
  if (best[n][0] === Infinity) return null;
  const out = [];
  for (let i = n; i > 0;) { const j = best[i][1]; out.push(s.slice(j, i)); i = j; }
  return [best[n][0], out.reverse()];
}

function variants(word) {
  const seen = new Map([[word, 0]]);
  let frontier = [word];
  for (const k of [1, 2]) {
    const nxt = [];
    for (const w of frontier) for (const [a, b] of SWAPS) {
      for (let i = w.indexOf(a); i !== -1; i = w.indexOf(a, i + a.length)) {
        const v = w.slice(0, i) + b + w.slice(i + a.length);
        if (!seen.has(v)) { seen.set(v, k); nxt.push(v); }
      }
    }
    frontier = nxt;
  }
  return seen;
}

const isUpperPy = s => /[A-Za-z]/.test(s) && s === s.toUpperCase();
const isAlphaPy = s => /^[A-Za-z]+$/.test(s);

function keepCost(lw) {
  const c = cost(lw);
  return c !== null ? c : UNKNOWN_PER_CHAR * lw.replace(/'/g, '').length;
}

function fixWord(core) {
  if (fixCache.has(core)) return fixCache.get(core);
  const r = fixWordUncached(core);
  fixCache.set(core, r);
  return r;
}

function fixWordUncached(core) {
  if (!/[A-Za-z]/.test(core)) return core;
  if ((FREQ.get(core.toLowerCase()) || 0) >= 5000) return core;
  if (core.includes('-')) {
    const parts = core.split('-');
    if (parts.length > 1 && parts.every(Boolean)) return parts.map(fixWord).join('-');
  }
  let best = [/^[A-Za-z']+$/.test(core) ? keepCost(core.toLowerCase()) : Infinity, core];
  for (const [v, k] of variants(core.toUpperCase())) {
    const lw = v.toLowerCase();
    if (!/^[a-z']+$/.test(lw)) continue;
    const c = cost(lw);
    if (c !== null && c + SUB_PENALTY * k < best[0]) best = [c + SUB_PENALTY * k, v];
    if (lw.length >= 4) {
      const sg = segment(lw);
      if (sg && sg[1].length > 1) {
        const cand = [sg[0] - 2.0 * sg[1].length + SUB_PENALTY * k, sg[1].map(p => p.toUpperCase()).join(' ')];
        if (cand[0] < best[0]) best = cand;
      }
    }
  }
  const out = best[1];
  return isUpperPy(core) || !isAlphaPy(core) ? out : (out !== core.toUpperCase() ? out : core);
}

const TOKEN = /^([^\w']*)([\w'\-]*?)([^\w']*)$/;

export function fixText(text) {
  text = text.replace(/(?<=[A-Za-z])([.,!?;:]+)(?=[A-Za-z])/g, '$1 ');
  return text.split(/\s+/).filter(Boolean).map(tok => {
    const m = TOKEN.exec(tok);
    if (!m || !m[2]) return tok;
    return m[1] + fixWord(m[2]) + m[3];
  }).join(' ');
}

export function englishScore(text) {
  let tot = 0, good = 0;
  for (const tok of text.match(/[A-Za-z][A-Za-z'\-]*/g) || []) { tot += tok.length; if (known(tok)) good += tok.length; }
  return tot ? good / tot : 0;
}

// ───────────────────────── 4. reading order (src/order.py) ─────────────────────────
function gutterMaps(img, boxes) {
  const g = gray(img), n = img.w * img.h;
  const white = new Uint8Array(n), black = new Uint8Array(n), trans = new Uint8Array(n);
  for (let i = 0; i < n; i++) { white[i] = g[i] >= 215; black[i] = g[i] <= 45; }
  for (const [x0, y0, x1, y1] of boxes) {
    const xa = Math.trunc(x0), ya = Math.trunc(y0), xb = Math.min(img.w, Math.trunc(x1)), yb = Math.min(img.h, Math.trunc(y1));
    for (let y = ya; y < yb; y++) trans.fill(1, y * img.w + xa, y * img.w + xb);
  }
  return { white, black, trans, W: img.w };
}

function profile(maps, axis, x0, y0, x1, y1) {
  // axis=1: one value per row; axis=0: one value per column
  const { white, black, trans, W } = maps;
  const len = axis === 1 ? y1 - y0 : x1 - x0, n = axis === 1 ? x1 - x0 : y1 - y0;
  const vis = new Float64Array(len), hw = new Float64Array(len), hb = new Float64Array(len);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = y * W + x, k = axis === 1 ? y - y0 : x - x0;
    if (!trans[i]) { vis[k]++; if (white[i]) hw[k]++; if (black[i]) hb[k]++; }
  }
  const lenient = new Float64Array(len), strict = new Float64Array(len);
  for (let k = 0; k < len; k++) {
    for (const hit of [hw[k], hb[k]]) {
      const l = vis[k] > 0.3 * n ? hit / Math.max(vis[k], 1) : 0;
      if (l > lenient[k]) lenient[k] = l;
      if (hit / n > strict[k]) strict[k] = hit / n;
    }
  }
  return [lenient, strict];
}

function cuts(lenient, strict, frac, minGap, minPart) {
  const n = lenient.length, cs = [];
  for (let i = 0; i < n;) {
    if (!(lenient[i] >= frac)) { i++; continue; }
    let j = i;
    while (j < n && lenient[j] >= frac) j++;
    if (j - i >= minGap && i > 0 && j < n) {
      let mx = -1;
      for (let k = i; k < j; k++) mx = Math.max(mx, strict[k]);
      let s = 0, c = 0;
      for (let k = i; k < j; k++) if (strict[k] === mx) { s += k - i; c++; }
      cs.push(i + Math.trunc(s / c));
    }
    i = j;
  }
  const out = [];
  let last = 0;
  for (const c of cs) if (c - last >= minPart && n - c >= minPart) { out.push(c); last = c; }
  return out;
}

function xyCut(maps, rtl, x0, y0, x1, y1, minPart, depth = 0, frac = 0.92, minGap = 3) {
  if (depth > 12 || x1 - x0 < 2 * minPart || y1 - y0 < 2 * minPart) return [[x0, y0, x1, y1]];
  const hc = cuts(...profile(maps, 1, x0, y0, x1, y1), frac, minGap, minPart);
  if (hc.length) {
    const e = [0, ...hc, y1 - y0], out = [];
    for (let i = 0; i + 1 < e.length; i++) out.push(...xyCut(maps, rtl, x0, y0 + e[i], x1, y0 + e[i + 1], minPart, depth + 1, frac, minGap));
    return out;
  }
  const vc = cuts(...profile(maps, 0, x0, y0, x1, y1), frac, minGap, minPart);
  if (vc.length) {
    const e = [0, ...vc, x1 - x0];
    let parts = [];
    for (let i = 0; i + 1 < e.length; i++) parts.push([e[i], e[i + 1]]);
    if (rtl) parts = parts.reverse();
    const out = [];
    for (const [a, b] of parts) out.push(...xyCut(maps, rtl, x0 + a, y0, x0 + b, y1, minPart, depth + 1, frac, minGap));
    return out;
  }
  return [[x0, y0, x1, y1]];
}

function rowsOrder(dets, rtl) {
  let rest = [...dets].sort((a, b) => a.box[1] - b.box[1]);
  const ordered = [];
  while (rest.length) {
    const top = rest[0], y0 = top.box[1], y1 = top.box[3];
    const row = rest.filter(d => d === top || (y0 <= (d.box[1] + d.box[3]) / 2 && (d.box[1] + d.box[3]) / 2 <= y1));
    rest = rest.filter(d => !row.includes(d));
    row.sort(rtl ? (a, b) => b.box[2] - a.box[2] : (a, b) => a.box[0] - b.box[0]);
    ordered.push(...row);
  }
  return ordered;
}

function panelOrder(img, dets, bubbles, rtl) {
  const { w, h } = img;
  const maps = gutterMaps(img, [...dets.map(d => d.box), ...bubbles.map(b => b.box)]);
  const panels = xyCut(maps, rtl, 0, 0, w, h, Math.trunc(0.06 * Math.min(h, w)));
  const buckets = panels.map(() => []);
  for (const d of dets) {
    const cx = (d.box[0] + d.box[2]) / 2, cy = (d.box[1] + d.box[3]) / 2;
    let idx = panels.findIndex(([a, b, c, e]) => a <= cx && cx < c && b <= cy && cy < e);
    if (idx < 0) {
      let bd = Infinity;
      panels.forEach(([a, b, c, e], i) => {
        const dd = Math.max(a - cx, 0, cx - c) ** 2 + Math.max(b - cy, 0, cy - e) ** 2;
        if (dd < bd) { bd = dd; idx = i; }
      });
    }
    buckets[idx].push(d);
  }
  return { ordered: buckets.flatMap(b => rowsOrder(b, rtl)), panels };
}

// ───────────────────────── 5. text types (src/textkind.py) ─────────────────────────
function paperColour(img) {
  // dominant colour of a crop: 2-means, return the centre of the larger cluster
  const n = img.w * img.h, d = img.data;
  if (n < 10) { const m = [0, 0, 0]; for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) m[k] += d[i * 3 + k] / n; return m; }
  let lo = 0, hi = 0, lv = Infinity, hv = -Infinity;
  for (let i = 0; i < n; i++) { const v = d[i * 3] + d[i * 3 + 1] + d[i * 3 + 2]; if (v < lv) { lv = v; lo = i; } if (v > hv) { hv = v; hi = i; } }
  let c = [[d[lo * 3], d[lo * 3 + 1], d[lo * 3 + 2]], [d[hi * 3], d[hi * 3 + 1], d[hi * 3 + 2]]];
  const lab = new Uint8Array(n);
  for (let it = 0; it < 10; it++) {
    const s = [[0, 0, 0], [0, 0, 0]], cnt = [0, 0];
    for (let i = 0; i < n; i++) {
      let d0 = 0, d1 = 0;
      for (let k = 0; k < 3; k++) { d0 += (d[i * 3 + k] - c[0][k]) ** 2; d1 += (d[i * 3 + k] - c[1][k]) ** 2; }
      const l = d1 < d0 ? 1 : 0; lab[i] = l; cnt[l]++;
      for (let k = 0; k < 3; k++) s[l][k] += d[i * 3 + k];
    }
    c = c.map((cc, l) => cnt[l] ? s[l].map(v => v / cnt[l]) : cc);
    if (it === 9) return cnt[0] >= cnt[1] ? c[0] : c[1];
  }
}

function boxShape(img, box) {
  const { w: W, h: H } = img;
  const [x0, y0, x1, y1] = box.map(v => Math.trunc(v));
  if (x1 - x0 < 12 || y1 - y0 < 12) return 0;
  const paper = paperColour(crop(img, x0, y0, x1, y1));
  const near = (x, y) => {
    const i = (y * W + x) * 3, d = img.data;
    return Math.max(Math.abs(d[i] - paper[0]), Math.abs(d[i + 1] - paper[1]), Math.abs(d[i + 2] - paper[2])) < 40;
  };
  const lim = Math.max(30, Math.trunc(0.6 * Math.min(x1 - x0, y1 - y0)));
  const cx0 = x0 + Math.trunc(0.05 * (x1 - x0)), cx1 = x1 - Math.trunc(0.05 * (x1 - x0));
  const cy0 = y0 + Math.trunc(0.05 * (y1 - y0)), cy1 = y1 - Math.trunc(0.05 * (y1 - y0));
  // each side: list of lines, each line = the pixel sequence walking outward (starting 3 px inside)
  const sides = [];
  if (y0 - lim >= 0) sides.push({ n: cx1 - cx0, at: (k, i) => near(cx0 + k, y0 + 2 - i), len: lim + 3 });
  if (y1 + lim <= H) sides.push({ n: cx1 - cx0, at: (k, i) => near(cx0 + k, y1 - 3 + i), len: lim + 3 });
  if (x0 - lim >= 0) sides.push({ n: cy1 - cy0, at: (k, i) => near(x0 + 2 - i, cy0 + k), len: lim + 3 });
  if (x1 + lim <= W) sides.push({ n: cy1 - cy0, at: (k, i) => near(x1 - 3 + i, cy0 + k), len: lim + 3 });
  let straight = 0, judged = 0;
  for (const s of sides) {
    if (s.n < 15) continue;
    judged++;
    const found = [];
    for (let k = 0; k < s.n; k++) {
      let seen = 0;
      for (let i = 0; i < s.len; i++) {
        if (s.at(k, i)) seen++;
        else if (seen >= 2) { found.push(i); break; }
      }
    }
    if (found.length >= 0.85 * s.n) {
      const so = [...found].sort((a, b) => a - b), m = so.length;
      const med = m % 2 ? so[(m - 1) / 2] : (so[m / 2 - 1] + so[m / 2]) / 2;
      if (found.filter(v => Math.abs(v - med) <= 1.5).length / m >= 0.85) straight++;
    }
  }
  return judged < 2 ? 0 : straight / judged;
}

function classify(img, det, text) {
  const t = text.trim();
  if (t && /^[\d\s.,:/\-]+$/.test(t)) return 'other';
  if (boxShape(img, det.box) >= 0.75) return 'caption';
  if (det.label === 'text_free' && englishScore(t) < 0.5) return 'sfx';
  return 'dialogue';
}

// ───────────────────────── page pipeline ─────────────────────────
export async function processPage(models, img, { rtl = false, onRegion = () => {} } = {}) {
  const raw = await detect(models, img);
  const dets = nms(raw.filter(d => d.label !== 'bubble'));
  const bubbles = raw.filter(d => d.label === 'bubble');
  const { ordered, panels } = panelOrder(img, dets, bubbles, rtl);
  const regions = [];
  for (let i = 0; i < ordered.length; i++) {
    const d = ordered[i], [x0, y0, x1, y1] = d.box, pad = 6;
    const c = crop(img, Math.trunc(x0) - pad, Math.trunc(y0) - pad, Math.trunc(x1) + pad, Math.trunc(y1) + pad);
    onRegion(i, ordered.length);
    if (c.w === 0 || c.h === 0) continue;
    const text = clean(fixText(clean(await readBlock(models, c))));
    if (text) regions.push({ type: classify(img, d, text), text, box: d.box.map(v => Math.round(v)) });
  }
  return { regions, panels };
}
