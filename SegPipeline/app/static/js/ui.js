// Small DOM / API helpers shared by the pages.

export function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  let selectValue;
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'value' && tag === 'select') selectValue = v;
    else if (k === 'value' || k === 'checked' || k === 'disabled' || k === 'selected') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  append(el, kids);
  if (selectValue !== undefined) el.value = selectValue;
  return el;
}

export function append(el, kids) {
  for (const c of [kids].flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  let j = null;
  try { j = await r.json(); } catch { /* not json */ }
  if (!r.ok) throw new Error((j && j.error) || `${r.status} ${r.statusText}`);
  return j;
}

// ------------------------------------------------------------------ icons (feather-style paths)
const ICONS = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  up: '<path d="M18 15l-6-6-6 6"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  chev: '<path d="M6 9l6 6 6-6"/>',
  x: '<path d="M18 6L6 18M6 6l12 12"/>',
  reset: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  folder: '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
  play: '<path d="M6 4l14 8-14 8z"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  check: '<path d="M20 6L9 17l-5-5"/>',
  alert: '<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  save: '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/>',
  code: '<path d="M16 18l6-6-6-6M8 6l-6 6 6 6"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
  zap: '<path d="M13 2L3 14h9l-1 8 10-12h-9z"/>',
  tool: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9a2.1 2.1 0 0 1-3-3l6.9-6.9a6 6 0 0 1 7.9-7.9z"/>',
  film: '<rect x="2" y="2" width="20" height="20" rx="2"/><path d="M7 2v20M17 2v20M2 12h20M2 7h5M2 17h5M17 17h5M17 7h5"/>',
  arrow: '<path d="M5 12h14M12 5l7 7-7 7"/>',
  left: '<path d="M15 18l-6-6 6-6"/>',
  right: '<path d="M9 18l6-6-6-6"/>',
  refresh: '<path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.5 9a9 9 0 0 1 14.8-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/>',
  chart: '<path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/>',
  eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
  box: '<path d="M21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4a2 2 0 0 0 1-1.7z"/>',
};
export function icon(name) {
  const t = document.createElement('template');
  t.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ''}</svg>`;
  return t.content.firstChild;
}

export function btn(label, onClick, opts = {}) {
  const b = h('button', { class: 'btn ' + (opts.kind || ''), title: opts.title, disabled: opts.disabled, type: 'button' },
    opts.icon && icon(opts.icon), label);
  b.addEventListener('click', async (e) => {
    if (b.disabled) return;
    if (opts.busy) {
      const old = [...b.childNodes];
      b.disabled = true;
      b.replaceChildren(h('span', { class: 'spin' }), opts.busy);
      try { await onClick(e); } finally { b.disabled = false; b.replaceChildren(...old); }
    } else onClick(e);
  });
  return b;
}

export function iconBtn(name, title, onClick, cls = '') {
  const b = h('button', { class: 'icon-btn ' + cls, title, 'aria-label': title, type: 'button' }, icon(name));
  b.addEventListener('click', (e) => { e.stopPropagation(); onClick(e); });
  return b;
}

export function switchEl(checked, onChange, label) {
  const inp = h('input', { type: 'checkbox', checked: !!checked });
  inp.addEventListener('change', () => onChange(inp.checked));
  const lab = h('label', { class: 'switch', onClick: (e) => e.stopPropagation() }, inp, h('span', { class: 'track' }),
    label && h('span', { class: 'lbl' }, label));
  return lab;
}

export function segmented(options, value, onChange, cls = '') {
  const wrap = h('div', { class: 'seg ' + cls, role: 'group' });
  const draw = (v) => {
    wrap.replaceChildren(...options.map((o) => {
      const b = h('button', { type: 'button', class: o.id === v ? 'on' : '', title: o.doc || '' }, o.label);
      b.addEventListener('click', () => { draw(o.id); onChange(o.id); });
      return b;
    }));
  };
  draw(value);
  wrap.set = draw;
  return wrap;
}

// ------------------------------------------------------------------ feedback
export function toast(msg, kind = 'info', ms = 4500) {
  const box = document.getElementById('toasts');
  const t = h('div', { class: 'toast ' + kind }, icon(kind === 'ok' ? 'check' : kind === 'bad' ? 'alert' : 'info'),
    h('div', { class: 'grow' }, msg), iconBtn('x', 'Close', () => t.remove()));
  box.append(t);
  if (ms) setTimeout(() => t.remove(), ms);
  return t;
}

export function modal({ title, body, actions, wide }) {
  return new Promise((resolve) => {
    const back = h('div', { class: 'modal-back' });
    const close = (v) => { back.remove(); document.removeEventListener('keydown', onKey); resolve(v); };
    const onKey = (e) => { if (e.key === 'Escape') close(null); };
    const foot = h('footer', {}, (actions || [{ label: 'Close', value: null }]).map((a) =>
      h('button', { class: 'btn ' + (a.kind || ''), type: 'button', onClick: async () => {
        if (a.check && !(await a.check())) return;
        close(typeof a.value === 'function' ? a.value() : a.value);
      } }, a.label)));
    const m = h('div', { class: 'modal' + (wide ? ' wide' : ''), role: 'dialog', 'aria-modal': 'true' },
      h('header', {}, h('h2', {}, title)), h('div', { class: 'm-body' }, typeof body === 'function' ? body(close) : body), foot);
    back.append(m);
    back.addEventListener('mousedown', (e) => { if (e.target === back) close(null); });
    document.addEventListener('keydown', onKey);
    document.body.append(back);
    const first = m.querySelector('input, select, textarea');
    (first || foot.lastChild).focus();
  });
}

export function confirmDialog(title, text, okLabel = 'OK', kind = 'primary') {
  return modal({ title, body: h('p', {}, text), actions: [{ label: 'Cancel', value: false }, { label: okLabel, value: true, kind }] });
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied', 'ok', 1500); } catch { toast('Copy failed - select the text and press Ctrl+C', 'bad'); }
}

export function copyLine(text) {
  return h('div', { class: 'copy-line' }, h('code', {}, text), iconBtn('copy', 'Copy', () => copyText(text)));
}

export function notice(kind, iconName, ...body) {
  return h('div', { class: 'notice ' + kind }, icon(iconName), h('div', { class: 'body' }, body));
}

// ------------------------------------------------------------------ formatting
export const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
export const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));
export function fmtInt(n) { return n == null ? '—' : Math.round(n).toLocaleString('en-US'); }
export function fmtNum(n, d = 3) {
  if (n == null || !Number.isFinite(n)) return '—';
  if (Number.isInteger(n)) return String(n);
  return String(+n.toFixed(d));
}
export function fmtRange(r) { return Array.isArray(r) ? `${fmtNum(r[0])} – ${fmtNum(r[1])}` : fmtNum(r); }
export function fmtBytes(b) {
  if (b == null) return '—';
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(i ? 1 : 0)} ${u[i]}`;
}
export function fmtDuration(s) {
  if (s == null || !Number.isFinite(s)) return '—';
  s = Math.max(0, Math.round(s));
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  if (hh) return `${hh} h ${mm} min`;
  if (mm) return `${mm} min${mm < 10 && ss ? ` ${ss} s` : ''}`;
  return `${ss} s`;
}
export function timeAgo(t) {
  if (!t) return '';
  const s = Date.now() / 1000 - t;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
}
export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// TOML inline form of a value (what a config file would say)
export function tomlInline(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(tomlInline).join(', ') + ']';
  if (typeof v === 'object') {
    if ('$toml' in v) return v.$toml;
    const items = Object.entries(v).filter(([, x]) => x != null)
      .map(([k, x]) => `${/^[A-Za-z0-9_-]+$/.test(k) ? k : JSON.stringify(k)} = ${tomlInline(x)}`);
    return items.length ? `{ ${items.join(', ')} }` : '{}';
  }
  return String(v);
}

// ------------------------------------------------------------------ line chart (SVG)
const SVGNS = 'http://www.w3.org/2000/svg';
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

export function lineChart(series, { height = 200, yMin, yMax, xLabel = 'epoch', fmt = (v) => fmtNum(v, 2) } = {}) {
  const W = 620, H = height, L = 44, R = 12, T = 10, B = 26;
  const pts = series.flatMap((se) => se.points);
  const wrap = h('div', {});
  if (!pts.length) { wrap.append(h('div', { class: 'empty small' }, 'No epochs yet')); return wrap; }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]).filter(Number.isFinite);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = yMin ?? Math.min(...ys), y1 = yMax ?? Math.max(...ys);
  if (y1 - y0 < 1e-9) { y1 = y0 + 1; }
  const pad = (y1 - y0) * 0.05; if (yMin == null) y0 -= pad; if (yMax == null) y1 += pad;
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img' });
  const grid = s('g', { class: 'grid' });
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + ((y1 - y0) * i) / 4;
    grid.append(s('line', { x1: L, x2: W - R, y1: Y(yv), y2: Y(yv) }));
    svg.append(s('text', { x: L - 6, y: Y(yv) + 4, 'text-anchor': 'end' }, fmt(yv)));
  }
  svg.prepend(grid);
  const nx = Math.min(6, Math.max(1, Math.round(x1 - x0)));
  for (let i = 0; i <= nx; i++) {
    const xv = Math.round(x0 + ((x1 - x0) * i) / nx);
    svg.append(s('text', { x: X(xv), y: H - 8, 'text-anchor': 'middle' }, xv));
  }
  svg.append(s('line', { class: 'axis', x1: L, x2: W - R, y1: H - B, y2: H - B }));
  for (const se of series) {
    const p = se.points.filter((q) => Number.isFinite(q[1]));
    if (!p.length) continue;
    svg.append(s('path', { d: p.map((q, i) => `${i ? 'L' : 'M'}${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(''),
      fill: 'none', style: `stroke:${se.color}`, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-dasharray': se.dash || '' }));
    const last = p[p.length - 1];
    svg.append(s('circle', { cx: X(last[0]), cy: Y(last[1]), r: 3, style: `fill:${se.color}` }));
  }
  wrap.append(svg, h('div', { class: 'legend' }, series.map((se) =>
    h('span', {}, h('i', { style: { background: se.color } }), se.name))));
  return wrap;
}
