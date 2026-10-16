// Train page: quick-train a dataset with the pipeline's recipe (train/train_yolo.py --quick, then scored on real
// images), or set up a full training (yolo-training repo); follow both kinds of runs and their real-image scores.
import { h, api, btn, icon, iconBtn, toast, confirmDialog, modal, notice, segmented, switchEl, fmtInt, fmtNum, fmtDuration, fmtBytes, timeAgo, lineChart, clone } from './ui.js';
import { G } from './store.js';

let root = null;
let timer = null;
let T = null;            // {opts, datasets, form, info, runs, sel, detail}
const el = {};

export async function mount(node, params) {
  root = node;
  root.replaceChildren(h('div', { class: 'page' }, h('div', { class: 'empty' }, h('div', { class: 'spinner-lg' }))));
  const [opts, datasets, runs] = await Promise.all([api('/api/train/options'), api('/api/datasets'), api('/api/train/runs')]);
  if (!root) return;
  const keep = T?.form;
  T = { opts, datasets, runs, info: null, sel: T?.sel || null, detail: null, form: keep || newForm(opts) };
  const want = params.get('dataset');
  if (want) {
    T.form.dataset = want;
    T.form.nameTouched = false;
    if (!datasets.some((d) => d.path === want)) {
      try { T.datasets.unshift(await api('/api/dataset?path=' + encodeURIComponent(want))); } catch { /* shown below */ }
    }
  } else if (!T.form.dataset && datasets.length) {
    const zip = datasets.find((d) => d.kind === 'zip');
    T.form.dataset = (zip || datasets[0]).path;
  }
  if (T.sel && !T.sel.includes(':')) T.sel = null;           // (older app version kept a bare name)
  if (!T.sel && runs.length) T.sel = (runs.find((r) => r.status === 'running') || runs[0]).id;
  build();
  loadInfo();
  if (T.sel) loadDetail();
  timer = setInterval(refreshRuns, 5000);
}

export function unmount() { clearInterval(timer); root = null; }

function newForm(opts) {
  const d = opts.defaults;
  const p = opts.presets[d.preset];
  return { dataset: '', version: d.version, size: d.size, base: '', epochs: d.epochs, imgsz: d.imgsz, batch: d.batch,
    patience: d.patience, device: d.device, workers: d.workers, cache: d.cache, cos_lr: d.cos_lr, close_mosaic: d.close_mosaic,
    time: '', freeze: d.freeze, preset: d.preset, aug: clone(p.values), albumentations: p.albumentations,
    model_name: '', nameTouched: false };
}

// ------------------------------------------------------------------ layout
function build() {
  const o = T.opts;
  const warn = !o.repo_ok ? notice('warn', 'alert', h('b', {}, 'yolo-training repo not found - full trainings need it'),
    h('span', {}, `Looked in "${o.repo || '(not set)'}". Set its folder in Settings (bottom left). Quick train works without it.`))
    : !o.python_ok ? notice('warn', 'alert', h('b', {}, 'Python for full trainings not found'), h('span', {}, 'Set it in Settings.')) : null;
  el.left = h('div', { class: 'stack', style: { gap: '16px' } },
    datasetCard(), quickCard(), modelCard(), trainingCard(), augCard(), startCard());
  el.runs = h('div', { class: 'runs' });
  el.detail = h('div', {});
  const right = h('div', { class: 'stack', style: { gap: '16px' } },
    h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('div', { class: 'titles' }, h('h2', {}, 'Training runs'),
      h('p', {}, (o.run_roots || []).map((r) => r.dir).join('  ·  '))), iconBtn('refresh', 'Refresh', refreshRuns)), el.runs),
    el.detail);
  root.replaceChildren(h('div', { class: 'page' },
    h('div', { class: 'page-head' }, h('div', { class: 'titles' }, h('h1', {}, 'Train'),
      h('p', {}, 'Quick-train a dataset with the pipeline recipe, or set up a full training - training runs in its own window'))),
    warn, h('div', { class: 'train-layout' }, el.left, right)));
  drawRuns();
  updateName();
}

function card(num, title, sub, body, actions) {
  return h('section', { class: 'card' },
    h('div', { class: 'card-head' }, num ? h('span', { class: 'section-num' }, num) : null,
      h('div', { class: 'titles' }, h('h2', {}, title), sub ? h('p', {}, sub) : null), actions || null),
    h('div', { class: 'card-body' }, body));
}

// ------------------------------------------------------------------ 1 dataset
function dsLabel(d) {
  if (d.error) return `${d.name} (unreadable)`;
  const kind = d.kind === 'zip' ? 'zip' : d.task === 'pose' ? 'pose' : 'folder';
  return `${d.name}  ·  ${fmtInt(d.images)} images  ·  ${d.task}  ·  ${kind}`;
}

function datasetCard() {
  const sel = h('select', { class: 'wide' });
  const fill = () => {
    const known = T.datasets.some((d) => d.path === T.form.dataset);
    sel.replaceChildren(
      ...T.datasets.map((d) => h('option', { value: d.path, title: d.path }, dsLabel(d))),
      !known && T.form.dataset ? h('option', { value: T.form.dataset }, T.form.dataset) : null);
    sel.value = T.form.dataset;
  };
  fill();
  sel.addEventListener('change', () => { T.form.dataset = sel.value; path.value = sel.value; loadInfo(); });
  const path = h('input', { type: 'text', value: T.form.dataset, spellcheck: false, placeholder: 'dataset folder or .zip' });
  path.addEventListener('change', () => { T.form.dataset = path.value.trim(); fill(); loadInfo(); });
  const browse = btn('Browse', async () => {
    const r = await api('/api/browse', { kind: 'file', initial: T.form.dataset });
    if (!r.path) return;
    let p = r.path;
    if (!p.toLowerCase().endsWith('.zip')) {       // a file inside a dataset folder: walk up to the dataset root
      const parts = p.split('/');
      const i = parts.lastIndexOf('images');
      p = parts.slice(0, i > 0 ? i : parts.length - 1).join('/');
    }
    T.form.dataset = p; path.value = p; fill(); loadInfo();
  }, { icon: 'folder', busy: 'Choosing' });
  el.dsInfo = h('div', {});
  el.dsFill = fill;
  return card(1, 'Dataset', 'An Unreal dataset folder or its shareable zip, or any YOLO dataset',
    h('div', { class: 'stack' }, sel, h('div', { class: 'path-input' }, path, browse), el.dsInfo));
}

async function loadInfo() {
  const p = T.form.dataset;
  el.dsInfo.replaceChildren(p ? h('div', { class: 'muted small' }, 'Reading the dataset…') : null);
  updateName();
  if (!p) return;
  try {
    const info = await api('/api/dataset?path=' + encodeURIComponent(p));
    if (T.form.dataset !== p) return;
    T.info = info;
    drawInfo();
    updateName();
    drawAugHints();
  } catch (e) {
    T.info = null;
    el.dsInfo.replaceChildren(notice('bad', 'alert', e.message));
  }
}

function drawInfo() {
  const i = T.info;
  const sp = i.splits || {};
  const counts = i.class_counts || {};
  const max = Math.max(1, ...Object.values(counts).map(Number));
  const warn = [];
  if (!sp.val) warn.push('No validation split - training needs images/val (or valid/images).');
  if (!i.names?.length) warn.push('No class names found (data.yaml).');
  el.dsInfo.replaceChildren(h('div', { class: 'stack' },
    h('div', { class: 'ds-info' },
      kpi(fmtInt(sp.train), 'train images'), kpi(fmtInt(sp.val), 'val images'), sp.test ? kpi(fmtInt(sp.test), 'test images') : null,
      kpi(i.task, 'task'), i.kind === 'zip' ? kpi(fmtBytes(i.size), 'zip size') : null),
    i.names?.length ? h('div', { class: 'class-bars' }, i.names.map((n, id) => h('div', { class: 'cb' },
      h('span', {}, h('span', { class: 'faint' }, `${id} `), n),
      h('div', { class: 'progress' }, h('i', { style: { width: `${counts[n] ? (100 * counts[n]) / max : 0}%` } })),
      h('span', { class: 'num' }, counts[n] != null ? fmtInt(counts[n]) : '')))) : null,
    i.names?.length ? h('p', { class: 'muted small' }, 'Class ids (the numbers) follow this order. Keep it the same as the real data you will evaluate on.') : null,
    warn.map((w) => notice('warn', 'alert', w))));
}

function kpi(v, label) { return h('div', { class: 'kpi' }, h('b', {}, v ?? '—'), h('span', {}, label)); }

// ------------------------------------------------------------------ quick train
function quickCard() {
  const o = T.opts, q = o.quick || {};
  const sets = o.real_sets || [];
  el.quickCmd = h('div', { class: 'cmd-preview hidden' });
  const oldServer = !o.quick;         // a server started before Quick train existed (the app banner explains)
  const go = btn('Quick train', quickTrain, { kind: 'primary', icon: 'zap', busy: 'Starting', disabled: oldServer || !o.sdg_python_ok,
    title: 'train/train_yolo.py --quick on the dataset above, then score it on the real datasets' });
  return h('section', { class: 'card quick-card' },
    h('div', { class: 'card-head' }, h('span', { class: 'section-num' }, icon('zap')),
      h('div', { class: 'titles' }, h('h2', {}, 'Quick train'),
        h('p', {}, `The dataset above${q.epochs ? `: ${q.epochs} epochs` : ''} with the fast pipeline recipe, then scored on real images`)), go),
    h('div', { class: 'card-body stack' },
      h('div', { class: 'kpis' }, kpi(q.epochs, 'epochs'), kpi(q.patience, 'patience'), kpi(q.workers, 'loader workers'),
        kpi(sets.length || '—', 'real datasets')),
      oldServer ? notice('warn', 'alert', 'The app server was started before this update: double-click SDG App.bat to restart it.') : null,
      oldServer ? null : sets.length
        ? h('p', { class: 'muted small' }, 'Scored on ', h('b', {}, sets.map((x) => x.label).join(', ')),
          o.real_unstretch ? ` (square images unstretched to ${o.real_unstretch})` : '', '. Change them in Settings.')
        : notice('warn', 'alert', 'No real datasets set: the model trains but is not scored on real images. Add them in Settings (bottom left).'),
      h('p', { class: 'muted small' }, 'Strong augmentation without RandomFog (the renders have their own haze), model version and size from "Model" below. For the model you ship, run a full training.'),
      oldServer || o.sdg_python_ok ? null : notice('bad', 'alert', 'No Python with ultralytics found for the pipeline (SegPipeline/.venv, ~/seg-train/.venv or SDG_PYTHON).'),
      el.quickCmd));
}

async function quickTrain() {
  const f = T.form;
  if (!f.dataset) { toast('Choose a dataset first', 'bad'); return; }
  let r;
  try {
    r = await api('/api/train/quick', { dataset: f.dataset, version: f.version, size: f.size, base: f.base });
  } catch (e) { toast(e.message, 'bad', 9000); return; }
  el.quickCmd.textContent = r.command;
  el.quickCmd.classList.remove('hidden');
  toast(`Quick train started in a new window: ${r.name}` + (r.real_sets ? `, then scored on ${r.real_sets} real dataset(s)` : ''), 'ok', 7000);
  T.sel = r.id;
  setTimeout(refreshRuns, 2500);
}

// ------------------------------------------------------------------ 2 model
function modelCard() {
  const o = T.opts, f = T.form;
  const vDoc = h('div', { class: 'opt-doc' });
  const sDoc = h('div', { class: 'opt-doc' });
  const setDocs = () => {
    vDoc.textContent = o.versions.find((v) => v.id === f.version)?.doc || '';
    sDoc.textContent = o.sizes.find((s) => s.id === f.size)?.doc || '';
  };
  setDocs();
  const base = h('input', { type: 'text', value: f.base, placeholder: 'empty = start from the pretrained (COCO) weights', spellcheck: false });
  base.addEventListener('change', () => { f.base = base.value.trim(); });
  const pick = btn('Browse', async () => {
    const r = await api('/api/browse', { kind: 'model', initial: o.repo });
    if (r.path) { f.base = r.path; base.value = r.path; }
  }, { icon: 'folder', busy: 'Choosing' });
  return card(2, 'Model', 'Family and size of the network',
    h('div', { class: 'stack', style: { gap: '14px' } },
      h('div', { class: 'lbl-block' }, 'Version', segmented(o.versions, f.version, (v) => { f.version = v; setDocs(); updateName(); }), vDoc),
      h('div', { class: 'lbl-block' }, 'Size', segmented(o.sizes.map((s) => ({ ...s, label: s.id.toUpperCase() })), f.size, (v) => { f.size = v; setDocs(); updateName(); }), sDoc),
      h('div', { class: 'lbl-block' }, 'Start from (optional)', h('div', { class: 'path-input' }, base, pick),
        h('div', { class: 'opt-doc' }, 'Continue from one of your earlier models (a .pt file), e.g. to fine-tune on new data. Then the version and size come from that file.'))));
}

// ------------------------------------------------------------------ 3 training
function numField(label, key, attrs = {}, doc = '') {
  const inp = h('input', { type: 'number', value: T.form[key] ?? '', ...attrs });
  inp.addEventListener('change', () => {
    const t = inp.value.trim();
    T.form[key] = t === '' ? '' : Number(t);
    inp.classList.toggle('invalid', t !== '' && !Number.isFinite(Number(t)));
  });
  return h('label', { title: doc }, label, inp);
}

function selField(label, key, options, doc = '') {
  const sel = h('select', { value: String(T.form[key]) }, options.map(([v, l]) => h('option', { value: v }, l)));
  sel.addEventListener('change', () => { T.form[key] = /^\d+$/.test(sel.value) ? Number(sel.value) : sel.value; });
  return h('label', { title: doc }, label, sel);
}

function trainingCard() {
  const f = T.form;
  return card(3, 'Training', 'How long and how big', h('div', { class: 'stack' },
    h('div', { class: 'form-grid' },
      numField('Epochs', 'epochs', { min: 1 }, 'Passes over the dataset. Training stops earlier if it stops improving (patience).'),
      selField('Image size', 'imgsz', [[320, '320'], [480, '480'], [640, '640 (default)'], [800, '800'], [960, '960'], [1280, '1280']], 'Images are resized to this. Bigger helps small / far targets but is slower.'),
      numField('Batch', 'batch', { min: -1 }, 'Images per step. -1 = pick automatically for your GPU memory. Lower it on "CUDA out of memory".'),
      numField('Patience', 'patience', { min: 0 }, 'Stop after this many epochs without improvement.'),
      numField('Time limit (hours)', 'time', { min: 0, step: 'any', placeholder: 'none' }, 'Train for this long instead of a number of epochs.'),
      selField('Device', 'device', [['auto', 'auto (GPU if any)'], ['0', 'GPU 0'], ['cpu', 'CPU (slow)']]),
      numField('Workers', 'workers', { min: 0 }, 'CPU processes loading images.'),
      selField('Cache', 'cache', [['disk', 'disk (recommended)'], ['ram', 'RAM'], ['off', 'off']], 'Caches resized images. RAM is copied per worker on Windows - disk is safer.'),
      numField('Freeze layers', 'freeze', { min: 0 }, 'Fine-tuning on a few real images: freeze the first ~10 layers.'),
      numField('Close mosaic', 'close_mosaic', { min: 0 }, 'Turn mosaic off for the last N epochs.')),
    h('div', { class: 'row' }, switchEl(f.cos_lr, (c) => { f.cos_lr = c; }, 'Cosine learning-rate schedule'))));
}

// ------------------------------------------------------------------ 4 augmentation
function augCard() {
  const o = T.opts, f = T.form;
  const presets = Object.entries(o.presets).map(([id, p]) => ({ id, label: p.label, doc: p.doc }));
  el.presetDoc = h('div', { class: 'opt-doc' });
  el.presetSeg = segmented([...presets, { id: 'custom', label: 'Custom', doc: 'your own values below' }], f.preset, (id) => {
    if (id === 'custom') return;
    f.preset = id;
    f.aug = clone(o.presets[id].values);
    f.albumentations = o.presets[id].albumentations;
    drawSliders();
    el.albSwitch.querySelector('input').checked = f.albumentations === 'underwater';
    el.presetDoc.textContent = o.presets[id].doc;
  }, 'wide');
  el.presetDoc.textContent = o.presets[f.preset]?.doc || 'your own values';
  el.albSwitch = switchEl(f.albumentations === 'underwater', (c) => { f.albumentations = c ? 'underwater' : 'none'; markCustom(); }, 'Underwater camera effects');
  el.sliders = h('div', { class: 'aug-grid' });
  el.augHint = h('div', {});
  drawSliders();
  return card(4, 'Augmentation', 'Random changes to the training images so the model generalizes to real water',
    h('div', { class: 'stack' }, el.presetSeg, el.presetDoc,
      h('div', { class: 'stack', style: { gap: '4px' } }, el.albSwitch,
        h('div', { class: 'opt-doc', style: { marginTop: 0 } }, 'Blur and defocus, sensor noise, JPEG artefacts, low resolution, colour shifts, fog, grayscale: what a real underwater camera adds and a render lacks.')),
      el.augHint,
      h('details', {}, h('summary', { class: 'small', style: { cursor: 'pointer', color: 'var(--accent)' } }, 'Fine-tune each setting'), el.sliders)));
}

function markCustom() {
  const o = T.opts, f = T.form;
  const match = Object.entries(o.presets).find(([, p]) => p.albumentations === f.albumentations
    && Object.entries(p.values).every(([k, v]) => Math.abs((f.aug[k] ?? 0) - v) < 1e-9));
  f.preset = match ? match[0] : 'custom';
  el.presetSeg.set(f.preset);
  el.presetDoc.textContent = match ? o.presets[f.preset].doc : 'your own values';
}

function drawSliders() {
  const f = T.form;
  el.sliders.replaceChildren(...T.opts.aug.map((a) => {
    const out = h('output', {}, fmtNum(f.aug[a.key], 4));
    const r = h('input', { type: 'range', min: a.min, max: a.max, step: a.step, value: f.aug[a.key] });
    r.addEventListener('input', () => { f.aug[a.key] = Number(r.value); out.textContent = fmtNum(Number(r.value), 4); markCustom(); drawAugHints(); });
    return h('div', { class: 'aug' }, h('div', { class: 'top' }, h('span', {}, a.label), out), r, h('small', {}, a.doc));
  }));
  drawAugHints();
}

function drawAugHints() {
  if (!el.augHint) return;
  const f = T.form, task = T.info?.task;
  const hints = [];
  if (task === 'pose' && f.aug.copy_paste > 0) hints.push('Copy-paste only applies to segmentation; it is ignored for pose.');
  if (task === 'pose' && f.aug.fliplr > 0) hints.push('Pose: mirroring is only used when the dataset has flip_idx (the Unreal presets do); otherwise it is switched off.');
  if (f.aug.mixup > 0.2 && task === 'segment') hints.push('Mixup above 0.2 usually hurts segmentation.');
  if (f.aug.flipud > 0) hints.push('Upside-down flips: only if your objects can really appear upside down.');
  el.augHint.replaceChildren(...hints.map((t) => notice('info', 'info', t)));
}

// ------------------------------------------------------------------ 5 start
function suggestName() {
  const f = T.form;
  let base = (T.info?.name || f.dataset.split('/').pop() || 'model').replace(/_yolo_seg$/, '').replace(/\.zip$/i, '')
    .replace(/ \(pose\)$/, '_pose').replace(/[^A-Za-z0-9_-]+/g, '_');
  base = `${base}_${f.version}${f.size}`;
  const taken = new Set(T.runs.map((r) => r.name));
  let n = base, i = 2;
  while (taken.has(n)) n = `${base}_${i++}`;
  return n;
}

function updateName() {
  if (!el.nameIn || !T) return;
  if (!T.form.nameTouched) { T.form.model_name = suggestName(); el.nameIn.value = T.form.model_name; }
  el.namePath.textContent = `Saved as ${T.opts.repo}/${T.form.model_name || '<name>'}.pt, run folder runs/${T.form.model_name || '<name>'}`;
}

function startCard() {
  el.nameIn = h('input', { type: 'text', spellcheck: false, style: { width: '100%' } });
  el.nameIn.addEventListener('input', () => { T.form.model_name = el.nameIn.value.trim(); T.form.nameTouched = true; updateName(); });
  el.namePath = h('div', { class: 'opt-doc' });
  el.cmd = h('div', { class: 'cmd-preview hidden' });
  const start = btn('Start training', startTraining, { kind: 'primary big', icon: 'play', busy: 'Starting', disabled: !T.opts.repo_ok || !T.opts.python_ok });
  return card(5, 'Start full training', 'With the settings above (yolo-training repo), in its own window; you can close this app meanwhile',
    h('div', { class: 'stack' }, h('label', { class: 'lbl-block' }, 'Model name', el.nameIn), el.namePath,
      h('div', { class: 'row' }, h('span', { class: 'grow' }), start), el.cmd));
}

function requestBody(overwrite) {
  const f = T.form;
  return { dataset: f.dataset, model_name: f.model_name, version: f.version, size: f.size, base: f.base,
    epochs: f.epochs, imgsz: f.imgsz, batch: f.batch, patience: f.patience, device: f.device, workers: f.workers,
    cache: f.cache, cos_lr: f.cos_lr, close_mosaic: f.close_mosaic, time: f.time === '' ? null : f.time, freeze: f.freeze || 0,
    aug: f.aug, albumentations: f.albumentations, overwrite };
}

async function startTraining() {
  if (!T.info) { toast('Choose a dataset first', 'bad'); return; }
  if (!T.info.splits?.val) {
    if (!(await confirmDialog('No validation split', 'This dataset has no val images; training will stop with an error. Start anyway?', 'Start anyway'))) return;
  }
  let r;
  try {
    r = await api('/api/train/start', requestBody(false));
    if (r.exists) {
      if (!(await confirmDialog('Replace the model?', `${r.model} exists already. Replace it when this training finishes?`, 'Replace', 'danger'))) return;
      r = await api('/api/train/start', requestBody(true));
    }
  } catch (e) { toast(e.message, 'bad', 9000); return; }
  if (!r.ok) return;
  el.cmd.textContent = r.command;
  el.cmd.classList.remove('hidden');
  toast(`Training started in a new window: ${T.form.model_name}`, 'ok', 6000);
  T.sel = r.id || `app:${T.form.model_name}`;
  T.form.nameTouched = false;
  setTimeout(refreshRuns, 2500);
}

// ------------------------------------------------------------------ runs
const STATUS = {
  running: ['busy', 'running'], finished: ['ok', 'finished'], ended: ['warn', 'ended early'],
  failed: ['bad', 'failed'], stopped: ['warn', 'stopped'], unknown: ['', 'no results'],
};

async function refreshRuns() {
  if (!root) return;
  try {
    T.runs = await api('/api/train/runs');
    if (!root) return;
    drawRuns();
    const sel = T.runs.find((r) => r.id === T.sel) || T.runs.find((r) => r.id.startsWith(T.sel || '\0'));
    if (sel && (sel.status === 'running' || !T.detail || T.detail.id !== sel.id || T.detail.status !== sel.status
        || T.detail.mtime !== sel.mtime)) { T.sel = sel.id; loadDetail(); }
  } catch { /* next time */ }
}

function drawRuns() {
  if (!el.runs) return;
  if (!T.runs.length) { el.runs.replaceChildren(h('div', { class: 'empty small' }, 'No training runs yet.')); return; }
  el.runs.replaceChildren(...T.runs.slice(0, 30).map((r) => {
    const [cls, label] = STATUS[r.status] || ['', r.status];
    const best = r.best ? `synthetic ${r.best.kind} mAP50-95 ${r.best.value.toFixed(3)}` : '';
    const real = r.real?.length ? `real mAP50 ${r.real.map((x) => `${x.name} ${x.mAP50.toFixed(3)}`).join(', ')}` : '';
    const row = h('div', { class: 'run' + (r.id === T.sel ? ' sel' : ''), title: r.dir },
      h('span', { class: 'dot ' + cls }), h('span', { class: 'n' }, r.name, r.quick ? h('span', { class: 'tag' }, 'quick') : null,
        h('span', { class: 'tag faint' }, r.source)),
      h('span', { class: 'v', title: 'best synthetic-val mAP50-95' }, r.best ? r.best.value.toFixed(3) : ''),
      h('span', { class: 'm' }, [label, r.epochs ? `epoch ${r.epoch} / ${r.epochs}` : null, real, best, timeAgo(r.mtime)].filter(Boolean).join('  ·  ')));
    row.addEventListener('click', () => { T.sel = r.id; drawRuns(); loadDetail(); });
    return row;
  }));
}

async function loadDetail() {
  if (!T.sel) return;
  try {
    const d = await api('/api/train/run?id=' + encodeURIComponent(T.sel));
    if (!root || d.id !== T.sel) return;
    T.detail = d;
    drawDetail();
  } catch (e) { el.detail.replaceChildren(notice('bad', 'alert', e.message)); }
}

function series(d, cols, color, name, dash) {
  const ep = d.columns.indexOf('epoch');
  const idx = cols.map((c) => d.columns.indexOf(c)).filter((i) => i >= 0);
  if (!idx.length) return null;
  const points = d.rows.map((r) => [r[ep], idx.reduce((a, i) => a + r[i], 0)]);
  return { name, color, dash, points };
}

function drawDetail() {
  const d = T.detail;
  const [cls, label] = STATUS[d.status] || ['', d.status];
  const cols = d.columns || [];
  const metrics = [
    series(d, ['metrics/mAP50-95(M)'], 'var(--chart-1)', 'mask mAP50-95'),
    series(d, ['metrics/mAP50(M)'], 'var(--chart-1)', 'mask mAP50', '5 4'),
    series(d, ['metrics/mAP50-95(P)'], 'var(--chart-3)', 'pose mAP50-95'),
    series(d, ['metrics/mAP50(P)'], 'var(--chart-3)', 'pose mAP50', '5 4'),
    series(d, ['metrics/mAP50-95(B)'], 'var(--chart-2)', 'box mAP50-95'),
  ].filter(Boolean);
  const trainL = cols.filter((c) => c.startsWith('train/') && c.endsWith('_loss') && d.rows.some((r) => r[cols.indexOf(c)]));
  const valL = cols.filter((c) => c.startsWith('val/') && c.endsWith('_loss') && d.rows.some((r) => r[cols.indexOf(c)]));
  const losses = [series(d, trainL, 'var(--chart-2)', 'train loss (total)'), series(d, valL, 'var(--chart-1)', 'val loss (total)')].filter(Boolean);
  const perEpoch = d.elapsed && d.epoch ? d.elapsed / d.epoch : null;
  const eta = d.status === 'running' && perEpoch && d.epochs ? (d.epochs - d.epoch) * perEpoch : null;
  const real = d.real_eval || [];
  const realMean = real.length ? real.reduce((a, r) => a + r.all.mAP50, 0) / real.length : null;
  el.detail.replaceChildren(h('section', { class: 'card' },
    h('div', { class: 'card-head' }, h('div', { class: 'titles' }, h('h2', {}, d.name), h('p', {}, d.dir)),
      h('span', { class: 'chip ' + cls }, cls === 'busy' ? h('span', { class: 'spin' }) : null, label)),
    h('div', { class: 'card-body stack', style: { gap: '14px' } },
      h('div', { class: 'kpis' },
        kpi(d.epochs ? `${d.epoch} / ${d.epochs}` : String(d.epoch || 0), 'epochs'),
        kpi(realMean != null ? realMean.toFixed(3) : '—', real.length > 1 ? 'real mAP50 (mean)' : 'real mAP50'),
        kpi(d.best ? d.best.value.toFixed(3) : '—', d.best ? `synthetic ${d.best.kind} mAP50-95` : 'mAP50-95'),
        kpi(perEpoch ? fmtDuration(perEpoch) : '—', 'per epoch'),
        kpi(eta ? fmtDuration(eta) : d.elapsed ? fmtDuration(d.elapsed) : '—', eta ? 'left (at most)' : 'training time')),
      realSection(d),
      h('div', {}, h('h3', { style: { marginBottom: '6px' } }, 'Validation mAP (synthetic val split - optimistic)'), lineChart(metrics, { yMin: 0, yMax: 1, height: 190 })),
      h('div', {}, h('h3', { style: { marginBottom: '6px' } }, 'Loss'), lineChart(losses, { height: 170 })),
      d.images?.length ? h('div', { class: 'stack', style: { gap: '6px' } }, h('h3', {}, 'Training images and plots'),
        thumbs(d, d.images)) : null,
      h('div', { class: 'row' },
        btn('Open run folder', () => api('/api/open', { path: d.dir }).catch((e) => toast(e.message, 'bad')), { kind: 'small', icon: 'folder' }),
        btn('Analyse', () => tool('analyse'), { kind: 'small', icon: 'chart', busy: 'Starting', disabled: !T.opts.repo_ok,
          title: 'Curves, split scores and an overfitting check (yolo-training analyse_model.py)' })),
      evaluateSection(d))));
}

const fileUrl = (d, f) => `/files/run?id=${encodeURIComponent(d.id)}&f=${encodeURIComponent(f)}`;

function thumbs(d, files) {
  return h('div', { class: 'thumbs' }, files.map((f) => {
    const img = h('img', { src: fileUrl(d, f), loading: 'lazy', alt: f, title: f.split('/').pop() });
    img.addEventListener('click', () => modal({ title: f.split('/').pop(), wide: true,
      body: [h('img', { src: fileUrl(d, f), class: 'full-img', alt: f })], actions: [{ label: 'Close', value: true }] }));
    return img;
  }));
}

function realSection(d) {
  const real = d.real_eval || [];
  const head = h('h3', {}, 'Real images (box metrics, human labels)');
  if (!real.length) {
    return h('div', { class: 'stack', style: { gap: '6px' } }, head, h('p', { class: 'muted small' },
      d.status === 'running' && d.quick ? 'This quick run is scored on the real datasets when training ends.'
        : 'Not scored on real images yet - use Evaluate below.'));
  }
  const cols = [['Real set', ''], ['img', 'images scored'], ['inst', 'labelled drawings'], ['P', 'precision'], ['R', 'recall'],
    ['mAP50', 'box mAP at IoU 0.5'], ['50-95', 'box mAP50-95'], ['false', 'detections at confidence 0.5 or higher that match no label']];
  const table = h('div', { class: 'table-scroll' }, h('table', { class: 'stage-table' },
    h('thead', {}, h('tr', {}, cols.map(([c, t], i) => h('th', { class: i ? 'num' : '', title: t }, c)))),
    h('tbody', {}, real.map((r) => h('tr', {},
      h('td', { title: r.dir }, r.name), h('td', { class: 'num' }, fmtInt(r.images)), h('td', { class: 'num' }, fmtInt(r.instances)),
      h('td', { class: 'num' }, r.all.P.toFixed(3)), h('td', { class: 'num' }, r.all.R.toFixed(3)),
      h('td', { class: 'num' }, h('b', {}, r.all.mAP50.toFixed(3))), h('td', { class: 'num' }, r.all['mAP50-95'].toFixed(3)),
      h('td', { class: 'num', title: 'detections at 0.5 confidence or higher that match no label' }, r.false50 ?? '—'))))));
  const perClass = real.map((r) => h('details', {},
    h('summary', { class: 'small', style: { cursor: 'pointer' } }, `${r.name}: per class, prediction images (${r.overlays.length})`),
    h('table', { class: 'stage-table' },
      h('thead', {}, h('tr', {}, ['class', 'inst', 'P', 'R', 'AP50', 'AP50-95'].map((c, i) => h('th', { class: i ? 'num' : '' }, c)))),
      h('tbody', {}, Object.entries(r.per_class).map(([n, m]) => h('tr', {}, h('td', {}, n), h('td', { class: 'num' }, fmtInt(m.instances)),
        h('td', { class: 'num' }, m.P.toFixed(3)), h('td', { class: 'num' }, m.R.toFixed(3)),
        h('td', { class: 'num' }, m.AP50.toFixed(3)), h('td', { class: 'num' }, m['AP50-95'].toFixed(3)))))),
    r.overlays.length ? h('p', { class: 'muted small' }, 'Coloured boxes / masks: the model (confidence ≥ 0.35). Thin white boxes: the human labels.') : null,
    r.overlays.length ? thumbs(d, r.overlays) : null,
    h('div', { class: 'row' }, btn('Open folder', () => api('/api/open', { path: r.dir }).catch((e) => toast(e.message, 'bad')), { kind: 'small', icon: 'folder' }))));
  return h('div', { class: 'stack', style: { gap: '8px' } }, head, table, ...perClass);
}

function evaluateSection(d) {
  const o = T.opts;
  const dataIn = h('textarea', { rows: 2, spellcheck: false, class: 'wide mono', placeholder: 'real datasets, one per line: …/data.yaml or a Roboflow .zip' });
  dataIn.value = (o.real_sets || []).map((x) => x.path).join('\n');
  const add = btn('Add', async () => {
    const r = await api('/api/browse', { kind: 'real', initial: dataIn.value.split('\n').pop() || '' });
    if (r.path) dataIn.value = (dataIn.value.trim() ? dataIn.value.trim() + '\n' : '') + r.path;
  }, { kind: 'small', icon: 'folder', busy: '…' });
  const splitsIn = h('input', { type: 'text', value: 'valid test', style: { width: '110px' }, title: 'Splits of the real datasets to score' });
  const unIn = h('input', { type: 'text', value: o.real_unstretch ?? '1280 720', style: { width: '110px' },
    title: 'Square (Roboflow-stretched) images are resized to W H before scoring; empty = off' });
  return h('div', { class: 'stack', style: { gap: '6px' } },
    h('h3', {}, 'Score on real images'),
    h('p', { class: 'muted small' }, 'mAP on real labelled images (classes matched by name). Results go to real_eval/ in the run folder and show up above.'),
    h('div', { class: 'path-input' }, dataIn, add),
    h('div', { class: 'row' }, h('span', { class: 'muted small' }, 'splits'), splitsIn, h('span', { class: 'muted small' }, 'unstretch'), unIn,
      h('span', { class: 'grow' }),
      btn('Evaluate', () => {
        const data = dataIn.value.split('\n').map((x) => x.trim()).filter(Boolean);
        o.real_sets = data.map((p) => ({ path: p, label: p.split('/').slice(-2)[0] }));
        o.real_unstretch = unIn.value.trim();
        return tool('evaluate', { data, splits: splitsIn.value, unstretch: unIn.value.trim() });
      }, { kind: 'small', icon: 'target', busy: 'Starting', disabled: !d.best_pt })));
}

async function tool(name, extra = {}) {
  try {
    const r = await api('/api/train/tool', { tool: name, run: T.sel, ...extra });
    toast(`Started in a new window:\n${r.command}`, 'ok', 7000);
  } catch (e) { toast(e.message, 'bad', 8000); }
}
