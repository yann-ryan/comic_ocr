// Background worker: loads the models once and processes pages, so the page
// itself stays responsive. Messages in:  {type:'page', id, blob, rtl}
// Messages out: {type:'status', text} | {type:'ready'} | {type:'progress', id, i, n}
//               {type:'result', id, result} | {type:'error', id, message}
importScripts('vendor/ort.wasm.min.js');
/* global ort */
ort.env.wasm.wasmPaths = new URL('vendor/', self.location.href).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let engine = null, models = null;
const queue = [];
let busy = false;

async function init() {
  engine = await import('./comicocr.js');
  models = await engine.Models.load('models/', what => postMessage({ type: 'status', text: `Loading ${what}…` }));
  postMessage({ type: 'ready', threads: ort.env.wasm.numThreads });
  pump();
}

async function decode(blob) {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const cv = new OffscreenCanvas(bmp.width, bmp.height), ctx = cv.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return engine.imageDataToBGR(ctx.getImageData(0, 0, cv.width, cv.height));
}

async function pump() {
  if (busy || !models) return;
  busy = true;
  while (queue.length) {
    const { id, blob, rtl } = queue.shift();
    try {
      const img = await decode(blob);
      const result = await engine.processPage(models, img, {
        rtl, onRegion: (i, n) => postMessage({ type: 'progress', id, i, n }),
      });
      postMessage({ type: 'result', id, result });
    } catch (e) {
      postMessage({ type: 'error', id, message: String(e && e.message || e) });
    }
  }
  busy = false;
}

onmessage = e => {
  if (e.data.type === 'page') { queue.push(e.data); pump(); }
};

init().catch(e => postMessage({ type: 'error', id: null, message: String(e && e.message || e) }));
