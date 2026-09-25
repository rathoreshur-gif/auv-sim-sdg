// Render page: setup, quick test and full render in the open Unreal Editor, with live progress.
import { h, api, btn, icon, iconBtn, toast, confirmDialog, copyLine, notice, fmtInt, fmtBytes, fmtDuration, switchEl } from './ui.js';
import { G, setConfig, on, unreal } from './store.js';

let root = null;
let timer = null;
let off = null;
let R = { run: null, logOffset: -1, lines: [], busy: null, quickN: 20, saveAfterSetup: true, lightbox: null };
const el = {};

export async function mount(node) {
  root = node;
  const configs = G.state?.configs || [];
  if (!configs.includes(G.config)) setConfig(configs.includes('torpedo') ? 'torpedo' : configs[0] || null);
  R.run = null;
  R.lines = [];
  R.logOffset = -1;
  build();
  off = on((ev) => { if (ev === 'state') { updateUnreal(); updateButtons(); } });
  await poll();
  timer = setInterval(poll, 1500);
}

export function unmount() {
  clearInterval(timer);
  off?.();
  root = null;
}

function build() {
  const configs = G.state?.configs || [];
  const sel = h('select', { value: G.config || '' }, configs.map((c) => h('option', { value: c }, c)));
  sel.addEventListener('change', async () => { setConfig(sel.value); R.run = null; await poll(); });
  el.unrealBox = h('div', {});
  el.steps = h('div', { class: 'grid-3' }, stepSetup(), stepQuick(), stepFull());
  el.progress = h('section', { class: 'card hidden' });
  el.result = h('section', { class: 'card hidden' });
  el.gallery = h('div', { class: 'gallery' });
  el.galleryCard = h('section', { class: 'card hidden' },
    h('div', { class: 'card-head' }, h('div', { class: 'titles' }, h('h2', {}, 'Previews'),
      el.prevSub = h('p', {}, 'Images drawn with their labels - check them before training')),
      btn('Open folder', () => openPath(R.run?.dataset + '/previews'), { kind: 'small', icon: 'folder' })),
    h('div', { class: 'card-body' }, el.gallery));
  el.log = h('pre', { class: 'log' });
  const logCard = h('section', { class: 'card' },
    h('div', { class: 'card-head' }, h('div', { class: 'titles' }, h('h2', {}, 'Unreal log'),
      h('p', {}, 'Pipeline messages and render errors from the editor (Saved/Logs)'))), el.log);
  root.replaceChildren(h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('div', { class: 'titles' }, h('h1', {}, 'Render'), h('p', {}, 'Prepare the level, test, then render the full dataset')),
      h('div', { class: 'actions' }, configs.length ? sel : null)),
    el.unrealBox, el.steps, el.progress, el.result, el.galleryCard, logCard));
  updateUnreal();
}

// ------------------------------------------------------------------ steps
function stepCard(num, title, text, controls) {
  return h('section', { class: 'card step' },
    h('div', { class: 'step-top' }, h('span', { class: 'section-num' }, num), h('h2', {}, title)),
    h('p', {}, text), h('div', { class: 'grow-space' }), controls);
}

function stepSetup() {
  el.setupBtn = btn('Run setup', runSetup, { icon: 'tool', busy: 'Running setup' });
  return stepCard(1, 'Prepare the level',
    'Tags the labelled props and adds the SDG camera and post-process volume. Needed once, and again after changing props or classes.',
    h('div', { class: 'stack' }, switchEl(R.saveAfterSetup, (c) => { R.saveAfterSetup = c; }, 'Save the level afterwards'),
      h('div', { class: 'row' }, el.setupBtn)));
}

function stepQuick() {
  const n = h('input', { type: 'number', min: 1, max: 500, value: R.quickN, style: { width: '76px' } });
  n.addEventListener('change', () => { R.quickN = Math.max(1, Math.min(500, parseInt(n.value, 10) || 20)); n.value = R.quickN; });
  el.quickBtn = btn('Run quick test', () => startRender(R.quickN), { icon: 'eye', busy: 'Starting' });
  return stepCard(2, 'Quick test',
    'Renders a few images per stage so you can check the previews (labels, look, camera) before the long run.',
    h('div', { class: 'row' }, n, h('span', { class: 'muted small' }, 'images per stage'), h('span', { class: 'grow' }), el.quickBtn));
}

function stepFull() {
  el.fullInfo = h('p', {}, 'Renders every stage into one dataset.');
  el.fullBtn = btn('Start full render', () => startRender(0), { kind: 'primary', icon: 'play', busy: 'Starting' });
  el.cancelBtn = btn('Stop render', cancelRender, { kind: 'danger', icon: 'stop', busy: 'Stopping' });
  return h('section', { class: 'card step' },
    h('div', { class: 'step-top' }, h('span', { class: 'section-num' }, 3), h('h2', {}, 'Full render')),
    el.fullInfo, h('div', { class: 'grow-space' }), h('div', { class: 'row' }, h('span', { class: 'grow' }), el.cancelBtn, el.fullBtn));
}

function connected() { return !!G.state?.unreal?.connected; }
function rendering() { return !!G.state?.unreal?.rendering; }

function updateButtons() {
  if (!el.setupBtn) return;
  const ok = connected() && !!G.config;
  const busy = rendering() || !!R.busy;
  for (const b of [el.setupBtn, el.quickBtn, el.fullBtn]) {
    if (!b.querySelector('.spin')) b.disabled = !ok || busy;
  }
  el.cancelBtn.classList.toggle('hidden', !rendering());
  const tip = !connected() ? 'Unreal is not connected' : busy ? 'Wait for the current job to finish' : '';
  for (const b of [el.setupBtn, el.quickBtn, el.fullBtn]) b.title = tip;
}

function updateUnreal() {
  if (!el.unrealBox) return;
  const u = G.state?.unreal || {};
  const map = R.run?.map;
  el.unrealBox.replaceChildren();
  if (!u.connected) {
    el.unrealBox.append(notice('warn', 'alert',
      h('b', {}, u.never ? 'Unreal is not connected yet' : 'Unreal is not responding'),
      h('span', {}, 'Open the project in Unreal. The connection starts by itself when the editor opens. In an editor that is already open, paste this once into the Output Log (Cmd):'),
      copyLine(`py "${G.state?.listener || ''}"`)));
  } else if (map && u.level && u.level !== map) {
    el.unrealBox.append(notice('warn', 'alert', h('b', {}, `Open ${map} in Unreal`),
      h('span', {}, `The editor has ${u.level} open, but this recipe renders ${map}.`)));
  }
  updateButtons();
}

async function runSetup() {
  if (!G.config) return;
  R.busy = 'setup'; updateButtons();
  try {
    const r = await unreal('setup', [G.config]);
    if (!r.ok) { toast('Setup failed - see the Unreal log below\n' + firstLine(r.error), 'bad', 10000); return; }
    if (R.saveAfterSetup) {
      const s = await unreal('save_level', [], { timeoutS: 120 });
      toast(s.ok ? 'Setup done and the level is saved' : 'Setup done, but saving failed: ' + firstLine(s.error), s.ok ? 'ok' : 'bad', 7000);
    } else {
      toast('Setup done - save the level in Unreal (Ctrl+S)', 'ok', 7000);
    }
  } catch (e) { toast(e.message, 'bad'); } finally { R.busy = null; updateButtons(); }
}

async function startRender(quick) {
  if (!G.config) return;
  if (!quick) {
    const planned = R.run?.planned?.images;
    const ok = await confirmDialog('Start the full render?',
      `This renders ${fmtInt(planned)} images${estimate() ? ` (about ${fmtDuration(estimate())})` : ''}. The dataset folder of "${R.run?.name}" is replaced. You can keep working in other programs meanwhile.`,
      'Start', 'primary');
    if (!ok) return;
  }
  R.busy = 'render'; updateButtons();
  try {
    const args = [G.config].concat(quick ? ['--quick', String(quick)] : []);
    const r = await unreal('render', args, { timeoutS: 600 });
    if (!r.ok) { toast('Could not start the render - see the Unreal log\n' + firstLine(r.error), 'bad', 10000); return; }
    toast(quick ? `Quick test started: ${quick} images per stage` : 'Full render started', 'ok');
    R.lines.push('--- ' + (quick ? 'quick test' : 'full render') + ' started from the SDG app');
  } catch (e) { toast(e.message, 'bad'); } finally { R.busy = null; updateButtons(); poll(); }
}

async function cancelRender() {
  if (!(await confirmDialog('Stop the render?', 'Movie Render Queue stops; the images rendered so far stay in the dataset.', 'Stop', 'danger'))) return;
  const r = await unreal('cancel', [], { timeoutS: 60 });
  toast(r.ok ? 'Render stopped' : firstLine(r.error), r.ok ? 'ok' : 'bad');
}

function firstLine(s) { return (s || '').trim().split('\n').slice(-1)[0]; }

function estimate() {
  const run = R.run;
  if (!run?.last_fps) return null;
  const frames = run.planned.stages.reduce((a, s) => a + (s.frames || 0) * (s.hold || 1), 0);
  return frames / run.last_fps + 30 * run.planned.stages.length;
}

async function openPath(p) {
  try { await api('/api/open', { path: p }); } catch (e) { toast(e.message, 'bad'); }
}

// ------------------------------------------------------------------ polling
async function poll() {
  if (!root || !G.config) return;
  try {
    const [run, log] = await Promise.all([
      api('/api/run?config=' + encodeURIComponent(G.config)),
      api('/api/log?since=' + R.logOffset),
    ]);
    if (!root) return;
    R.run = run;
    R.logOffset = log.offset;
    if (log.lines.length) { R.lines.push(...log.lines); R.lines = R.lines.slice(-400); drawLog(); }
    else if (!el.log.textContent) drawLog(log.missing);
    drawRun();
    updateUnreal();
  } catch (e) {
    if (root) el.progress.replaceChildren(h('div', { class: 'card-body' }, notice('bad', 'alert', e.message)));
  }
}

function drawLog(missing) {
  const atBottom = el.log.scrollHeight - el.log.scrollTop - el.log.clientHeight < 30;
  el.log.replaceChildren(...(R.lines.length ? R.lines : [missing ? 'Editor log not found - set it in Settings.' : 'No pipeline messages yet.'])
    .map((ln) => h('div', { class: /error|failed|Traceback/i.test(ln) ? 'err' : /finished|done|saved/i.test(ln) ? 'ok' : '' }, ln)));
  if (atBottom) el.log.scrollTop = el.log.scrollHeight;
}

function phase(run) {
  if (!run?.run) return 'none';
  const p = run.progress;
  if (p?.state === 'done') return p.failed ? 'stopped' : 'done';
  if (!p && run.stats) return run.failed_flag ? 'stopped' : 'done';     // finished before progress.json existed
  if (p?.state === 'zipping') return 'zipping';
  if (rendering()) return 'rendering';
  if (run.failed_flag) return 'stopped';
  if (run.done_flag) return 'converting';
  if (Date.now() / 1000 - run.run.created_t < 120 || p) return connected() ? 'rendering' : 'lost';
  return 'lost';
}

const PHASE = {
  rendering: ['busy', 'Rendering'], converting: ['busy', 'Render finished - converting the last frames'],
  zipping: ['busy', 'Packing the shareable zip'], done: ['ok', 'Dataset ready'],
  stopped: ['warn', 'Stopped - the dataset holds what was rendered'], lost: ['warn', 'No progress - is the render still running?'],
};

function drawRun() {
  const run = R.run;
  if (!run || !run.ok) {
    el.fullInfo.textContent = run?.error || '';
    el.progress.classList.add('hidden');
    return;
  }
  const est = estimate();
  el.fullInfo.textContent = `${fmtInt(run.planned.images)} images in ${run.planned.stages.length} stage${run.planned.stages.length === 1 ? '' : 's'}`
    + (est ? `, about ${fmtDuration(est)} (${run.last_fps.toFixed(1)} frames/s last time)` : '') + ` → ${run.dataset}`;
  const ph = phase(run);
  if (ph === 'none') {
    el.progress.classList.add('hidden'); el.result.classList.add('hidden'); el.galleryCard.classList.add('hidden');
    return;
  }
  const p = run.progress || {};
  const legacyDone = !run.progress && ph === 'done';
  const stages = run.run.stages.map((s, i) => {
    const conv = legacyDone ? s.expected : p.stages?.[i]?.rendered || 0;
    const rendered = Math.min(s.expected, conv + (s.pending || 0));
    return { ...s, conv, rendered };
  });
  const expected = stages.reduce((a, s) => a + s.expected, 0);
  const rendered = stages.reduce((a, s) => a + s.rendered, 0);
  const conv = stages.reduce((a, s) => a + s.conv, 0);
  const done = ph === 'done';
  const frac = done ? 1 : expected ? rendered / expected : 0;
  const flagSecs = Number((run.done_flag || '').match(/in ([\d.]+) s/)?.[1]);
  const elapsed = p.elapsed || (legacyDone && flagSecs ? flagSecs : Date.now() / 1000 - run.run.created_t);
  const rate = conv && elapsed && (p.elapsed || legacyDone) ? conv / elapsed : null;
  const eta = !done && rate && expected > rendered ? (expected - rendered) / rate : null;
  const [kind, label] = PHASE[ph] || ['busy', ph];
  const c = p.counts || run.stats?.counts || {};
  el.progress.classList.remove('hidden');
  el.progress.replaceChildren(
    h('div', { class: 'card-head' },
      h('div', { class: 'titles' }, h('h2', {}, run.run.quick ? `Quick test (${run.run.quick} per stage)` : 'Full render'),
        h('p', {}, `started ${run.run.created || ''}`)),
      h('span', { class: 'chip ' + kind }, kind === 'busy' ? h('span', { class: 'spin' }) : icon(kind === 'ok' ? 'check' : 'alert'), label)),
    h('div', { class: 'card-body stack' },
      h('div', { class: 'progress big' + (done ? ' ok' : '') }, h('i', { style: { width: `${(frac * 100).toFixed(1)}%` } })),
      h('div', { class: 'kpis' },
        kpi(fmtInt(c.frames ?? 0), 'images written'),
        kpi(`${fmtInt(rendered)} / ${fmtInt(expected)}`, 'frames rendered'),
        kpi(rate ? `${rate.toFixed(1)}/s` : '—', 'frames per second'),
        kpi(done ? fmtDuration(elapsed) : eta ? fmtDuration(eta) : '—', done ? 'took' : 'left (estimate)'),
        kpi(fmtInt(c.empty_kept ?? 0), 'background images'),
        c.errors ? kpi(fmtInt(c.errors), 'errors') : null),
      h('table', { class: 'stage-table' },
        h('tr', {}, h('th', {}, 'Stage'), h('th', {}, 'Progress'), h('th', { class: 'num' }, 'Frames'), h('th', { class: 'num' }, 'Images')),
        stages.map((s) => h('tr', {}, h('td', {}, s.name),
          h('td', {}, h('div', { class: 'progress' + (s.conv >= s.expected ? ' ok' : '') }, h('i', { style: { width: `${s.expected ? (100 * s.rendered / s.expected).toFixed(1) : 0}%` } }))),
          h('td', { class: 'num' }, `${fmtInt(s.rendered)} / ${fmtInt(s.expected)}`),
          h('td', { class: 'num' }, fmtInt(s.images))))),
      h('div', { class: 'row' }, h('span', { class: 'muted small grow' }, 'A console window ("SDG dataset") converts frames while Unreal renders.'),
        btn('Open dataset folder', () => openPath(run.dataset), { kind: 'small', icon: 'folder' }))));
  drawResult(run, ph);
  drawGallery(run);
}

function kpi(v, label) { return h('div', { class: 'kpi' }, h('b', {}, v), h('span', {}, label)); }

function drawResult(run, ph) {
  const st = run.stats;
  if (ph !== 'done' || !st) { el.result.classList.add('hidden'); return; }
  const seg = st.seg || {};
  const max = Math.max(1, ...Object.values(seg));
  const zip = run.zip && !run.run.quick && run.zip.mtime >= run.run.created_t ? run.zip : null;
  const target = zip ? zip.path : run.dataset;
  el.result.classList.remove('hidden');
  el.result.replaceChildren(
    h('div', { class: 'card-head' }, h('div', { class: 'titles' }, h('h2', {}, run.run.quick ? 'Quick test result' : 'Dataset'),
      h('p', {}, run.run.quick ? 'Look at the previews below. If they look right, start the full render.' : `${fmtInt(st.counts?.frames)} images in ${run.dataset}`))),
    h('div', { class: 'card-body grid-2' },
      h('div', { class: 'stack' }, h('h3', {}, 'Labels per class'),
        h('div', { class: 'class-bars' }, Object.entries(seg).map(([k, v]) => h('div', { class: 'cb' }, h('span', {}, k),
          h('div', { class: 'progress' }, h('i', { style: { width: `${100 * v / max}%` } })), h('span', { class: 'num' }, fmtInt(v)))))),
      h('div', { class: 'stack' },
        zip ? h('div', {}, h('h3', {}, 'Shareable zip'), h('p', { class: 'muted small' }, `${zip.path} (${fmtBytes(zip.size)})`)) : null,
        run.run.quick ? null : h('div', { class: 'row' },
          btn('Show zip', () => openPath(zip ? zip.path : run.dataset), { icon: 'folder' }),
          btn('Train on this dataset', () => { location.hash = '#train?dataset=' + encodeURIComponent(target); }, { kind: 'primary', icon: 'arrow' })))));
}

function drawGallery(run) {
  const list = run.previews || [];
  el.galleryCard.classList.toggle('hidden', !list.length);
  if (!list.length) return;
  el.prevSub.textContent = `${run.previews_total} preview${run.previews_total === 1 ? '' : 's'} - images drawn with their labels. Check them before training.`;
  const shown = list.slice(0, 18);
  const key = shown.join('|');
  if (el.gallery.dataset.key === key) return;
  el.gallery.dataset.key = key;
  el.gallery.replaceChildren(...shown.map((f, i) => h('button', { type: 'button', title: f, onClick: () => lightbox(list, i) },
    h('img', { src: previewUrl(f), loading: 'lazy', alt: f }))));
}

function previewUrl(f) {
  return `/files/preview?config=${encodeURIComponent(G.config)}&f=${encodeURIComponent(f)}&v=${R.run?.run?.created_t || 0}`;
}

function lightbox(list, i) {
  const img = h('img', { alt: '' });
  const cap = h('span', {});
  const show = (j) => { i = (j + list.length) % list.length; img.src = previewUrl(list[i]); cap.textContent = `${list[i]}  (${i + 1} / ${list.length})`; };
  const box = h('div', { class: 'lightbox', onClick: (e) => { if (e.target === box) close(); } }, img,
    h('div', { class: 'bar' }, btn('', () => show(i - 1), { icon: 'left', kind: 'small' }), cap,
      btn('', () => show(i + 1), { icon: 'right', kind: 'small' }), btn('Close', () => close(), { kind: 'small' })));
  const onKey = (e) => {
    if (e.key === 'ArrowLeft') show(i - 1);
    else if (e.key === 'ArrowRight') show(i + 1);
    else if (e.key === 'Escape') close();
  };
  const close = () => { box.remove(); document.removeEventListener('keydown', onKey); };
  document.addEventListener('keydown', onKey);
  document.body.append(box);
  show(i);
}
