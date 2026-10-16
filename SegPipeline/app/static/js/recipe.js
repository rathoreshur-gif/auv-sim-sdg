// Recipe page: create and edit dataset configs (sdg/configs/*.toml) with forms.
import {
  h, api, btn, iconBtn, icon, switchEl, toast, modal, confirmDialog, copyText, clone, isNum,
  fmtInt, fmtRange, debounce, tomlInline,
} from './ui.js';
import { G, setConfig, emit, unreal } from './store.js';

const VALUE_SPEC = ['uniform', 'normal', 'loguniform', 'choice', 'randint', 'hsv', 'rgb', 'kelvin'];
const MODE_LABEL = { random_path: 'smooth flight', teleport: 'random views', sequence: 'planned path' };
const LABELS = {
  fov: 'Field of view', aim_jitter: 'Aim jitter', min_area: 'Min label area', empty_keep: 'Keep empty frames',
  val_fraction: 'Validation share', max_label_depth: 'Max label distance', jpg_quality: 'JPEG quality',
  keep_exr: 'Keep EXR files', dataset_dir: 'Dataset folder', raw_dir: 'Raw frames folder', warmup_frames: 'Warm-up frames',
  anti_aliasing: 'Anti-aliasing', texture_streaming: 'Texture streaming', exr_compression: 'EXR compression',
  cvars: 'Console variables', train_args: 'Training arguments', open_trainer: 'Open Train page when done',
  min_visibility: 'Min visibility', switch_target: 'Switch target', sample_fps: 'Sample rate', jitter_pos: 'Position jitter',
  jitter_rot: 'Rotation jitter', fov_every: 'FOV change every', actor_label: 'Actor label', actor_tag: 'Actor tag',
  static_mesh: 'Static mesh', body_class: 'Body class', pose_class: 'Pose class', per_target: 'Per actor',
  material_slot: 'Material slot', postprocess: 'Post-process setting', mpc: 'Parameter collection', param: 'Parameter',
  seg: 'Segmentation classes', pose: 'Pose classes', map: 'Level', seed: 'Random seed', frames: 'Images',
  override: 'Overrides', disable: 'Switched off', python: 'Python (dataset builder)', workers: 'Workers',
  train: 'After a full render', resolution: 'Resolution', executor: 'Executor', every: 'Every N images',
};
const labelFor = (k) => LABELS[k] || (k.charAt(0).toUpperCase() + k.slice(1)).replace(/_/g, ' ');

let S = null;          // editor state: {stem, raw, dirty, isNew, check, showToml, adv, collapsed, openObjs}
let root = null;
let refreshers = [];
let statusUpdate = null;
let saveBtn = null;

export const isDirty = () => !!(S && S.dirty);

export async function mount(el) {
  root = el;
  const configs = G.state?.configs || [];
  if (!G.config || (!configs.includes(G.config) && !(S && S.isNew && S.stem === G.config))) {
    setConfig(configs.includes('torpedo') ? 'torpedo' : (configs[0] || null));
  }
  if (G.config && (!S || S.stem !== G.config)) {
    try { await load(G.config); } catch (e) { toast(e.message, 'bad'); S = null; }
  }
  draw();
}
export function unmount() { root = null; }

export async function saveShortcut() { if (S && S.dirty) await save(); }

async function load(stem) {
  const c = await api('/api/config?name=' + encodeURIComponent(stem));
  S = { stem, raw: c.raw, generated: c.generated, dirty: false, isNew: false, check: null, showToml: false,
    adv: {}, collapsed: { output: true, render: true, post: true }, openObjs: new WeakSet() };
  runCheck();
}

// ------------------------------------------------------------------ checking / saving
const runCheck = debounce(async () => {
  if (!S) return;
  const stem = S.stem;
  try {
    const r = await api('/api/config/check', { raw: S.raw });
    if (!S || S.stem !== stem) return;
    S.check = r;
  } catch (e) {
    if (S) S.check = { ok: false, errors: [{ path: '', message: e.message }] };
  }
  statusUpdate?.();
  updateSaveBtn();
}, 450);

function changed() {
  S.dirty = true;
  if (S.check) S.check.stale = true;
  for (const r of refreshers) r();
  statusUpdate?.();
  updateSaveBtn();
  runCheck();
}

function updateSaveBtn() {
  if (!saveBtn || !S) return;
  saveBtn.disabled = !S.dirty;
}

async function save() {
  const r = await api('/api/config/save', { name: S.stem, raw: S.raw });
  if (!r.ok) {
    S.check = { ok: false, errors: r.errors, text: S.check?.text };
    statusUpdate?.();
    toast('Not saved - ' + r.errors[0].message, 'bad', 9000);
    return;
  }
  const wasNew = S.isNew;
  S.dirty = false; S.isNew = false; S.generated = true;
  S.check = { ok: true, errors: [], text: r.text, summary: r.summary };
  toast(`Saved sdg/configs/${S.stem}.toml` + (r.backup ? `\nYour hand-written original (with its comments) is kept as ${r.backup.split('/').pop()}` : ''), 'ok', 7000);
  if (wasNew) emit('refresh-state');
  draw();
}

// ------------------------------------------------------------------ page
function draw() {
  if (!root) return;
  refreshers = [];
  const page = h('div', { class: 'page' }, header());
  if (!S) {
    page.append(h('section', { class: 'card' }, h('div', { class: 'empty' }, 'No recipe yet - press New to create one.')));
  } else {
    page.append(statusCard(), basicsCard(), classesCard(), propsCard(), cameraCard(), randomizeCard(), fogCard(),
      stagesCard(), sectionCard(8, 'output', 'Output', 'Dataset folder, validation split and what gets a label'),
      sectionCard(9, 'render', 'Rendering', 'Image size and Movie Render Queue settings'),
      sectionCard(10, 'post', 'After the render', 'Shareable zip, or open the Train page automatically'));
  }
  root.replaceChildren(page);
}

function header() {
  const names = [...(G.state?.configs || [])];
  if (S?.isNew && !names.includes(S.stem)) names.push(S.stem);
  const sel = h('select', { value: G.config || '', title: 'Recipe (sdg/configs/<name>.toml)' },
    names.map((n) => h('option', { value: n }, n + (S?.isNew && n === S.stem ? ' (new, unsaved)' : ''))));
  sel.addEventListener('change', async () => {
    if (S?.dirty && !(await confirmDialog('Discard changes?', `Your unsaved changes to "${S.stem}" will be lost.`, 'Discard', 'danger'))) {
      sel.value = S.stem;
      return;
    }
    setConfig(sel.value);
    await load(sel.value);
    draw();
  });
  saveBtn = btn('Save', save, { kind: 'primary', icon: 'save', busy: 'Saving', title: 'Save the recipe (Ctrl+S)' });
  updateSaveBtn();
  return h('div', { class: 'page-head' },
    h('div', { class: 'titles' }, h('h1', {}, 'Recipe'),
      h('p', {}, S ? `sdg/configs/${S.stem}.toml${S.isNew ? ' - not saved yet' : ''}` : 'Describe a dataset: props, camera, randomization, stages')),
    h('div', { class: 'actions' }, names.length ? sel : null, btn('New', newConfig, { icon: 'plus' }), S ? saveBtn : null));
}

async function newConfig() {
  const nameIn = h('input', { type: 'text', value: 'new_dataset' });
  const from = h('select', {}, h('option', { value: '' }, 'An empty recipe'),
    (G.state?.configs || []).map((c) => h('option', { value: c, selected: c === G.config }, `A copy of "${c}"`)));
  const ok = await modal({
    title: 'New recipe',
    body: [
      h('label', { class: 'lbl-block' }, 'Name (file sdg/configs/<name>.toml and the dataset folder)', nameIn),
      h('label', { class: 'lbl-block' }, 'Start from', from),
      h('p', { class: 'muted small' }, 'A copy keeps the props, camera and randomization of the original: usually the quickest start.'),
    ],
    actions: [{ label: 'Cancel', value: false }, {
      label: 'Create', value: true, kind: 'primary', check: () => {
        const n = nameIn.value.trim();
        if (!/^[A-Za-z0-9_-]+$/.test(n)) { toast('Use letters, digits, _ and - only', 'bad'); return false; }
        if ((G.state?.configs || []).includes(n)) { toast(`"${n}" exists already`, 'bad'); return false; }
        return true;
      },
    }],
  });
  if (!ok) return;
  if (S?.dirty && !(await confirmDialog('Discard changes?', `Your unsaved changes to "${S.stem}" will be lost.`, 'Discard', 'danger'))) return;
  const name = nameIn.value.trim();
  const raw = from.value ? (await api('/api/config?name=' + encodeURIComponent(from.value))).raw : clone(G.schema.new_config);
  raw.name = name;
  S = { stem: name, raw, generated: true, dirty: true, isNew: true, check: null, showToml: false, adv: {},
    collapsed: { output: true, render: true, post: true }, openObjs: new WeakSet() };
  setConfig(name);
  draw();
  runCheck();
}

function statusCard() {
  const chip = h('span', { class: 'chip' });
  const info = h('span', { class: 'muted small' });
  const probs = h('ul', { class: 'problems' });
  const pre = h('pre', { class: 'toml hidden' });
  const tomlBtn = btn('Show TOML', () => { S.showToml = !S.showToml; update(); }, { icon: 'code', kind: 'small' });
  const copyB = iconBtn('copy', 'Copy the TOML text', () => copyText(S.check?.text || ''));
  const card = h('section', { class: 'card' },
    h('div', { class: 'status-strip' }, chip, info, h('span', { class: 'grow' }), copyB, tomlBtn), probs, pre);
  function update() {
    const c = S.check;
    if (!c || c.stale) { chip.className = 'chip busy'; chip.replaceChildren(h('span', { class: 'spin' }), 'Checking'); }
    else if (c.ok) { chip.className = 'chip ok'; chip.replaceChildren(icon('check'), 'Valid recipe'); }
    else { chip.className = 'chip bad'; chip.replaceChildren(icon('alert'), 'Problem'); }
    const sm = c?.summary || {};
    const parts = [];
    if (S.dirty) parts.push('unsaved changes');
    if (sm.images != null) parts.push(`${fmtInt(sm.images)} images in ${sm.stages.length} stage${sm.stages.length === 1 ? '' : 's'}`);
    if (sm.dataset) parts.push(`dataset → ${sm.dataset}`);
    info.textContent = parts.join(' · ');
    probs.replaceChildren(...(c && !c.stale ? c.errors || [] : []).map((e) =>
      h('li', {}, icon('alert'), h('span', {}, (e.path ? e.path + ': ' : '') + e.message))));
    pre.classList.toggle('hidden', !S.showToml);
    pre.textContent = c?.text || '';
    tomlBtn.lastChild.textContent = S.showToml ? 'Hide TOML' : 'Show TOML';
  }
  statusUpdate = update;
  update();
  return card;
}

// ------------------------------------------------------------------ hints and value kinds
function hintFor(path) {
  const H = G.schema.hints;
  const p = path.replace(/\[\d+\]/g, '[]');
  if (H[p]) return H[p];
  let m = p.match(/^stages\[\]\.camera\.(.+)$/);
  if (m && H['camera.' + m[1]]) return H['camera.' + m[1]];
  if (/^props\[\]\.slots\.[^.]+$/.test(p)) return { enum_from: 'classes.seg', doc: 'Class of the slots that match this pattern.' };
  if (/^props\[\]\.regions\.[^.]+$/.test(p)) return { doc: 'Mask PNG for this class (relative to SegPipeline/).' };
  m = p.match(/^stages\[\]\.override\.([^.]+)$/);
  if (m) return { doc: `Range of "${m[1]}" in this stage: [min, max], or { value = [..], every = N }.` };
  m = p.match(/^randomize\[\]\.(location|rotation|scale)\.[^.]+$/);
  if (m) return { doc: m[1] === 'rotation' ? 'Degrees.' : m[1] === 'location' ? 'cm.' : 'Factor.' };
  return {};
}

function enumValues(hint) {
  if (hint.enum) return hint.enum;
  const c = S.raw.classes || {};
  if (hint.enum_from === 'classes.seg') return c.seg || [];
  if (hint.enum_from === 'classes.pose') return c.pose || [];
  if (hint.enum_from === 'props.names') return (S.raw.props || []).filter((p) => p.type === 'texture_regions').map((p) => p.name).filter(Boolean);
  if (hint.enum_from === 'randomize.names') {
    const names = (S.raw.randomize || []).map((r) => r.name).filter(Boolean);
    if (S.raw.water_fog?.enabled) names.unshift('water_fog_density', 'water_fog_color');
    return names;
  }
  return null;
}

const isValueSpec = (v) => v && typeof v === 'object' && !Array.isArray(v) && VALUE_SPEC.some((k) => k in v);

function kindOf(v, hint) {
  if (hint.widget === 'volume' && (v == null || (v && typeof v === 'object' && 'min' in v))) return v == null ? 'unset' : 'volume';
  if (v && typeof v === 'object' && !Array.isArray(v) && '$toml' in v) return 'toml';
  if (hint.enum_from === 'randomize.names' && Array.isArray(v)) return 'multi';
  if ((hint.enum || hint.enum_from) && (typeof v === 'string')) return 'enum';
  if (v === null || v === undefined) return 'unset';
  if (typeof v === 'boolean') return 'bool';
  if (typeof v === 'number') return 'number';
  if (typeof v === 'string') return 'string';
  if (Array.isArray(v)) {
    if (v.length === 2 && v.every(isNum)) return 'range';
    if (v.length === 3 && v.every(isNum)) return 'vec3';
    if (v.every((x) => typeof x === 'string')) return 'strlist';
    return 'toml';
  }
  if (isValueSpec(v)) return 'toml';
  return 'object';
}

function numInput(v) {
  return h('input', { type: 'number', step: 'any', value: v ?? '' });
}
function numOf(inp) {
  const t = inp.value.trim();
  const n = Number(t);
  const ok = t !== '' && Number.isFinite(n);
  inp.classList.toggle('invalid', !ok);
  return ok ? (/[.eE]/.test(t) ? n : parseInt(t, 10)) : null;
}

// One editable setting: label, control, help text, reset/remove button.
function fieldRow(o) {
  const row = h('div', { class: 'field' });
  const drawRow = () => {
    const v = o.get();
    const kind = kindOf(v, o.hint);
    const err = h('div', { class: 'field-err hidden' });
    const ctl = control(kind, v, o.hint, o.path, (nv, redraw) => {
      o.set(nv);
      row.classList.remove('is-default');
      if (redraw) drawRow();
    }, err);
    const def = o.isDefault ? o.isDefault() : false;
    row.classList.toggle('is-default', def);
    let act = null;
    if (o.reset && !def) {
      act = iconBtn(o.isDefault ? 'reset' : 'x', o.isDefault ? 'Back to the default value' : 'Remove this setting', () => {
        const removed = o.reset();
        if (removed) row.remove(); else drawRow();
      });
    }
    row.replaceChildren(
      h('div', { class: 'field-label' }, o.label, def ? h('span', { class: 'default-tag' }, 'default') : null,
        o.key && o.key !== o.label ? h('span', { class: 'key' }, o.key) : null),
      h('div', { class: 'field-ctl' }, ctl, err, o.hint.doc ? h('div', { class: 'field-help' }, o.hint.doc) : null),
      h('div', { class: 'field-act' }, act));
  };
  drawRow();
  return row;
}

function control(kind, v, hint, path, onChange, err) {
  switch (kind) {
    case 'enum': {
      const opts = enumValues(hint) || [];
      const list = opts.includes(v) || v === '' ? [...opts] : [...opts, v];
      if (hint.enum_from && !list.includes('')) list.unshift('');
      const sel = h('select', { value: v }, list.map((o) => h('option', { value: o }, o === '' ? blankLabel(path) : o)));
      sel.addEventListener('change', () => onChange(sel.value));
      return sel;
    }
    case 'bool':
      return switchEl(v, (c) => onChange(c));
    case 'number': {
      const inp = numInput(v);
      inp.addEventListener('change', () => { const n = numOf(inp); if (n != null) onChange(n); });
      return inp;
    }
    case 'string': {
      const inp = h('input', { type: 'text', class: 'wide', value: v, spellcheck: false });
      inp.addEventListener('input', () => onChange(inp.value));
      return inp;
    }
    case 'range': {
      const a = numInput(v[0]), b = numInput(v[1]);
      const upd = () => {
        const x = numOf(a), y = numOf(b);
        if (x == null || y == null) return;
        const bad = x > y;
        a.classList.toggle('invalid', bad); b.classList.toggle('invalid', bad);
        err.textContent = bad ? 'The first number must not be larger than the second.' : '';
        err.classList.toggle('hidden', !bad);
        onChange([x, y]);
      };
      a.addEventListener('change', upd); b.addEventListener('change', upd);
      return h('div', { class: 'range-inputs' }, a, h('span', {}, 'to'), b);
    }
    case 'vec3': {
      const ins = v.map((x) => numInput(x));
      ins.forEach((inp, i) => inp.addEventListener('change', () => { const n = numOf(inp); if (n != null) { v[i] = n; onChange(v); } }));
      return h('div', { class: 'vec-inputs' }, ['x', 'y', 'z'].map((a, i) => h('label', {}, a, ins[i])));
    }
    case 'strlist': {
      const inp = h('input', { type: 'text', class: 'wide', value: v.join(', '), placeholder: 'names separated by commas', spellcheck: false });
      inp.addEventListener('input', () => onChange(inp.value.split(',').map((x) => x.trim()).filter(Boolean)));
      return inp;
    }
    case 'multi': {
      const opts = enumValues(hint) || [];
      if (!opts.length) return h('span', { class: 'faint small' }, 'no randomizers yet');
      return h('div', { class: 'multi' }, [...new Set([...opts, ...v])].map((o) => {
        const cb = h('input', { type: 'checkbox', checked: v.includes(o) });
        cb.addEventListener('change', () => {
          const set = new Set(v);
          if (cb.checked) set.add(o); else set.delete(o);
          v.splice(0, v.length, ...opts.filter((x) => set.has(x)));
          onChange(v);
        });
        return h('label', {}, cb, o);
      }));
    }
    case 'object':
      return objectEditor(v, path, () => onChange(v));
    case 'toml': {
      const text = tomlInline(v);
      const ta = h('textarea', { rows: Math.min(4, Math.max(1, Math.ceil(text.length / 64))), spellcheck: false });
      ta.value = text;
      ta.addEventListener('change', async () => {
        const r = await api('/api/toml/parse', { text: ta.value });
        if (r.ok) {
          ta.classList.remove('invalid'); err.classList.add('hidden');
          onChange(r.value, kindOf(r.value, hint) !== 'toml');
        } else {
          ta.classList.add('invalid'); err.textContent = r.error; err.classList.remove('hidden');
          onChange({ $toml: ta.value });
        }
      });
      return h('div', { class: 'stack', style: { gap: '4px' } }, ta,
        h('div', { class: 'field-help' }, 'Written like in a config file, e.g. [0.5, 1.5] or { normal = [1, 0.3], clip = [0.2, 2] }'));
    }
    case 'unset':
      return h('div', { class: 'unset' }, 'not set',
        btn('Set', () => onChange(clone(hint.example ?? ''), true), { kind: 'small', icon: 'plus' }));
    case 'volume':
      return volumeEditor(v, onChange);
    default:
      return h('span', {}, String(v));
  }
}

function blankLabel(path) {
  if (/^post\.train$/.test(path)) return 'nothing - make a shareable zip';
  return '— choose —';
}

// camera.volume: two corners, each can be taken from the Unreal viewport
function volumeEditor(v, onChange) {
  const rows = ['min', 'max'].map((k, ci) => {
    if (!Array.isArray(v[k])) v[k] = [0, 0, 0];
    const ins = v[k].map((x) => numInput(x));
    ins.forEach((inp, i) => inp.addEventListener('change', () => { const n = numOf(inp); if (n != null) { v[k][i] = n; onChange(v); } }));
    const connected = !!G.state?.unreal?.connected;
    const use = btn('Use viewport', async () => {
      const r = await unreal('where', [], { timeoutS: 30 });
      if (!r.ok) { toast(r.error || 'Unreal could not read the viewport', 'bad'); return; }
      v[k] = r.location;
      ins.forEach((inp, i) => { inp.value = r.location[i]; inp.classList.remove('invalid'); });
      onChange(v);
      toast(`Corner ${ci + 1} = ${r.location.join(', ')} (from the Unreal viewport)`, 'ok');
    }, { kind: 'small', icon: 'target', busy: 'Asking Unreal', disabled: !connected,
      title: connected ? 'Take the position of the Unreal viewport camera' : 'Unreal is not connected (see the bottom left)' });
    return h('div', { class: 'row' }, h('b', { style: { width: '70px', fontWeight: 500 } }, `Corner ${ci + 1}`),
      h('div', { class: 'vec-inputs' }, ['x', 'y', 'z'].map((a, i) => h('label', {}, a, ins[i]))), use);
  });
  return h('div', { class: 'stack', style: { gap: '6px' } }, rows,
    h('div', { class: 'field-help' }, 'In Unreal, fly the viewport to one corner of the space the camera may use (inside the pool walls, under the surface) and press Use viewport; then do the same at the opposite corner.'));
}

// free-form tables: stage camera / overrides, transform ranges, regions, slots, cvars
function objectSuggestions(path) {
  const p = path.replace(/\[\d+\]/g, '[]');
  const d = G.schema.defaults;
  if (p === 'stages[].camera') return Object.keys(d.camera);
  if (p === 'stages[].override') return enumValues({ enum_from: 'randomize.names' });
  if (p === 'props[].regions') return S.raw.classes?.seg || [];
  return G.schema.object_keys[p] || [];
}

function newObjectValue(path, key) {
  const p = path.replace(/\[\d+\]/g, '[]');
  const pi = path.match(/^props\[(\d+)\]/);
  const ctx = pi ? S.raw.props?.[+pi[1]] : null;
  if (p === 'stages[].camera') {
    const v = S.raw.camera?.[key] ?? G.schema.defaults.camera[key];
    return clone(v ?? G.schema.hints['camera.' + key]?.example ?? '');
  }
  if (p === 'stages[].override') {
    if (key === 'water_fog_density') return clone(S.raw.water_fog?.density ?? [0.0002, 0.002]);
    const rz = (S.raw.randomize || []).find((r) => r.name === key);
    return clone(rz && rz.value !== undefined ? rz.value : [0, 1]);
  }
  if (p === 'props[].regions') return `props/${ctx?.name || 'prop'}/T_Mask_${key}.png`;
  if (p === 'props[].slots') return (S.raw.classes?.seg || [''])[0] || '';
  if (/\.rotation$/.test(p)) return [-10, 10];
  if (/\.scale$/.test(p)) return [0.9, 1.1];
  if (/\.location$/.test(p)) return [-20, 20];
  if (p === 'render.cvars') return 0;
  return '';
}

function objectEditor(obj, path, commit) {
  const box = h('div', { class: 'nested' });
  const drawObj = () => {
    box.replaceChildren();
    for (const k of Object.keys(obj)) {
      const p = `${path}.${k}`;
      box.append(fieldRow({
        label: k, path: p, hint: hintFor(p), get: () => obj[k],
        set: (nv) => { obj[k] = nv; commit(); },
        reset: () => { delete obj[k]; commit(); drawObj(); return true; },
      }));
    }
    const sugg = objectSuggestions(path).filter((k) => !(k in obj));
    const listId = 'dl-' + Math.random().toString(36).slice(2);
    const keyIn = h('input', { type: 'text', placeholder: sugg.length ? 'choose or type a key' : 'key', list: listId, spellcheck: false });
    const add = () => {
      const k = keyIn.value.trim();
      if (!k) return;
      if (k in obj) { toast(`"${k}" is there already`, 'bad'); return; }
      obj[k] = newObjectValue(path, k);
      commit();
      drawObj();
    };
    keyIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') add(); });
    box.append(h('div', { class: 'add-row' }, keyIn, h('datalist', { id: listId }, sugg.map((s) => h('option', { value: s }))),
      btn('Add', add, { kind: 'small', icon: 'plus' })));
  };
  drawObj();
  return box;
}

// ------------------------------------------------------------------ cards
function sectionShell(num, key, title, sub, body, actions) {
  const collapsed = S.collapsed[key] ?? false;
  const card = h('section', { class: 'card' + (collapsed ? ' collapsed' : '') });
  const chev = icon('chev'); chev.classList.add('chev');
  const subEl = h('p', {}, sub);
  const head = h('div', { class: 'card-head clickable' }, h('span', { class: 'section-num' }, num),
    h('div', { class: 'titles' }, h('h2', {}, title), subEl), actions, chev);
  head.addEventListener('click', () => {
    card.classList.toggle('collapsed');
    S.collapsed[key] = card.classList.contains('collapsed');
  });
  card.append(head, h('div', { class: 'card-body' }, body));
  card.setSub = (t) => { subEl.textContent = t; };
  return card;
}

function sectionFields(sec) {
  const box = h('div', { class: 'fields' });
  const drawFields = () => {
    const defs = G.schema.defaults[sec] || {};
    const raw = S.raw[sec] || {};
    const keys = [...new Set([...Object.keys(defs), ...Object.keys(raw)])];
    const all = !!S.adv[sec];
    box.replaceChildren();
    for (const k of keys) {
      const p = `${sec}.${k}`;
      const hint = hintFor(p);
      const inRaw = k in raw;
      if (!inRaw && !all) {
        if (hint.advanced) continue;
        if (hint.show_if) {
          const [fk, vals] = Object.entries(hint.show_if)[0];
          if (!vals.includes(raw[fk] ?? defs[fk])) continue;
        }
      }
      box.append(fieldRow({
        label: labelFor(k), key: k, path: p, hint,
        get: () => (k in (S.raw[sec] || {}) ? S.raw[sec][k] : clone(defs[k])),
        set: (v) => { (S.raw[sec] ??= {})[k] = v; changed(); if (sec === 'camera' && k === 'mode') drawFields(); },
        isDefault: () => !(k in (S.raw[sec] || {})),
        reset: () => {
          delete S.raw[sec][k];
          if (!Object.keys(S.raw[sec]).length) delete S.raw[sec];
          changed();
          if (sec === 'camera' && k === 'mode') { drawFields(); return true; }
          return false;
        },
      }));
    }
  };
  drawFields();
  box.redraw = drawFields;
  return box;
}

function allToggle(sec, fields) {
  return switchEl(!!S.adv[sec], (c) => { S.adv[sec] = c; fields.redraw(); }, 'All settings');
}

function sectionCard(num, sec, title, sub) {
  const f = sectionFields(sec);
  return sectionShell(num, sec, title, sub, f, allToggle(sec, f));
}

function topRow(key) {
  const hint = hintFor(key);
  const def = G.schema.defaults[key];
  return fieldRow({
    label: labelFor(key), key, path: key, hint,
    get: () => S.raw[key] ?? def ?? '',
    set: (v) => { S.raw[key] = v; changed(); },
    isDefault: def !== undefined ? () => !(key in S.raw) : null,
    reset: def !== undefined ? () => { delete S.raw[key]; changed(); return false; } : null,
  });
}

function basicsCard() {
  return sectionShell(1, 'basics', 'Basics', 'Dataset name, the level to render and the random seed',
    h('div', { class: 'fields' }, topRow('name'), topRow('map'), topRow('seed')));
}

function classesCard() {
  return sectionShell(2, 'classes', 'Classes', 'What the model learns to find. The order sets the YOLO class ids.', sectionFields('classes'));
}

function cameraCard() {
  const f = sectionFields('camera');
  return sectionShell(4, 'camera', 'Camera', 'Where the camera may be and how it frames the props', f, allToggle('camera', f));
}

function fogCard() {
  const f = sectionFields('water_fog');
  return sectionShell(6, 'water_fog', 'Underwater haze', 'A randomized haze over the image (clear to murky water); labels are dropped where the haze hides an object', f, allToggle('water_fog', f));
}

// ------------------------------------------------------------------ lists (props, randomize, stages)
function propSummary(p) {
  if (p.type === 'texture_regions') {
    const n = (p.images || []).length;
    return `drawings on a texture: ${Object.keys(p.regions || {}).join(', ') || 'no regions yet'}${n ? `  ·  ${n + 1} images` : ''}`;
  }
  if (p.type === 'material_slots') return `material slots: ${Object.entries(p.slots || {}).map(([k, v]) => `${k} → ${v || '?'}`).join(', ')}`;
  if (p.type === 'actor') return `whole actor → ${p.class || '?'}${p.actor_label ? `  (${p.actor_label})` : ''}`;
  return p.type || 'choose a type';
}

function rzType(rz) {
  return rz.type || (rz.mpc ? 'mpc' : rz.postprocess ? 'postprocess' : rz.material_slot != null ? 'material_param' : 'float');
}

function valueSummary(v, mode) {
  if (Array.isArray(v) && v.length === 2 && v.every(isNum)) return (mode === 'multiply' ? '× ' : mode === 'add' ? '+ ' : '') + fmtRange(v);
  if (v && typeof v === 'object' && !Array.isArray(v)) {
    if ('$toml' in v) return 'custom';
    const k = Object.keys(v)[0];
    return { hsv: 'random colour (hsv)', rgb: 'random colour (rgb)', kelvin: 'colour temperature', normal: 'normal distribution', choice: 'one of a list', randint: 'random integer', loguniform: 'log-uniform', uniform: 'uniform' }[k] || k;
  }
  return v === undefined ? '' : tomlInline(v);
}

function rzSummary(rz) {
  const t = rzType(rz);
  const target = rz.postprocess ? `post-process ${rz.postprocess}` : t === 'mpc' ? `material parameter ${rz.param || '?'}`
    : t === 'image' ? `image on ${rz.prop || '?'}`
    : [rz.actor || (rz.tag && `tag ${rz.tag}`), rz.component, rz.property || rz.param].filter(Boolean).join(' › ');
  let val;
  if (t === 'transform') {
    val = ['location', 'rotation', 'scale'].filter((k) => rz[k] && typeof rz[k] === 'object')
      .map((k) => `${k} ${Object.entries(rz[k]).map(([a, r]) => `${a} ${fmtRange(r)}`).join(', ')}`).join('; ');
  } else if (t === 'visible') {
    val = `shown ${Math.round((rz.probability ?? 0.5) * 100)}% of the time`;
  } else if (t === 'image' && rz.value === undefined) {
    val = 'any of its images';
  } else {
    val = valueSummary(rz.value, rz.mode);
  }
  const extra = rz.every > 1 ? ` · new value every ${rz.every}` : '';
  return `${target || '?'}  ·  ${val}${extra}`;
}

function stageSummary(st) {
  const cam = st.camera || {};
  const mode = cam.mode || S.raw.camera?.mode || 'random_path';
  const parts = [st.frames != null ? `${fmtInt(st.frames)} images` : 'length of the path', MODE_LABEL[mode] || mode];
  if (Array.isArray(cam.distance)) parts.push(`${fmtRange(cam.distance)} cm`);
  if (cam.hold > 1) parts.push(`each view ×${cam.hold}`);
  const ov = Object.keys(st.override || {});
  if (ov.length) parts.push(`changed: ${ov.join(', ')}`);
  if (st.disable?.length) parts.push(`off: ${st.disable.join(', ')}`);
  return parts.join('  ·  ');
}

function uniqueName(list, base) {
  const names = new Set((S.raw[list] || []).map((x) => x.name));
  if (!names.has(base)) return base;
  let i = 2;
  while (names.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}

function listCard(num, list, title, sub, opts) {
  const body = h('div', {});
  let card;
  const drawList = () => {
    const items = S.raw[list] || [];
    const wrap = h('div', { class: 'items' });
    items.forEach((item, i) => wrap.append(itemCard(list, item, i, drawList, opts)));
    if (!items.length) wrap.append(h('div', { class: 'empty small' }, opts.empty));
    body.replaceChildren(wrap, h('div', { class: 'list-foot' },
      btn(opts.addLabel, () => addFromTemplate(list, drawList), { icon: 'plus' }), opts.foot || null));
    if (card && opts.sub) card.setSub(opts.sub());
  };
  drawList();
  card = sectionShell(num, list, title, opts.sub ? opts.sub() : sub, body);
  if (opts.sub) refreshers.push(() => card.setSub(opts.sub()));
  return card;
}

function itemCard(list, item, i, redrawList, opts) {
  const open = S.openObjs.has(item);
  const el = h('div', { class: 'item' + (open ? '' : ' collapsed') });
  const title = h('b', {});
  const sub = h('small', {});
  const refresh = () => {
    title.textContent = item.name || '(no name)';
    sub.textContent = opts.summary(item);
    el.classList.toggle('off', item.enabled === false);
  };
  refreshers.push(refresh);
  const arr = S.raw[list];
  const move = (d) => {
    const j = i + d;
    if (j < 0 || j >= arr.length) return;
    [arr[i], arr[j]] = [arr[j], arr[i]];
    changed(); redrawList();
  };
  const dup = () => {
    const c = clone(item);
    c.name = uniqueName(list, item.name || 'copy');
    arr.splice(i + 1, 0, c);
    S.openObjs.add(c);
    changed(); redrawList();
  };
  const del = async () => {
    if (!(await confirmDialog('Delete?', `Delete "${item.name || 'this entry'}"?`, 'Delete', 'danger'))) return;
    arr.splice(i, 1);
    if (!arr.length) delete S.raw[list];
    changed(); redrawList();
  };
  const tools = h('div', { class: 'item-tools' },
    opts.toggle ? switchEl(item.enabled !== false, (c) => { if (c) delete item.enabled; else item.enabled = false; changed(); }) : null,
    iconBtn('up', 'Move up', () => move(-1)), iconBtn('down', 'Move down', () => move(1)),
    iconBtn('copy', 'Duplicate', dup), iconBtn('trash', 'Delete', del, 'danger'));
  tools.addEventListener('click', (e) => e.stopPropagation());
  const chev = icon('chev'); chev.classList.add('chev');
  const head = h('div', { class: 'item-head' }, chev, h('div', { class: 't' }, title, sub), tools);
  const body = h('div', { class: 'item-body' });
  let drawn = false;
  const drawBody = () => {
    drawn = true;
    const f = h('div', { class: 'fields' });
    for (const k of Object.keys(item)) {
      if (list === 'randomize' && k === 'enabled') continue;
      const p = `${list}[${i}].${k}`;
      f.append(fieldRow({ label: labelFor(k), key: k, path: p, hint: hintFor(p), get: () => item[k],
        set: (v) => { item[k] = v; changed(); },
        reset: k === 'name' ? null : () => { delete item[k]; changed(); drawBody(); return true; } }));
    }
    const missing = G.schema.item_fields[list].filter((k) => !(k in item) && !(list === 'randomize' && k === 'enabled'));
    const add = missing.length ? h('select', {}, h('option', { value: '' }, '+ Add a setting…'),
      missing.map((k) => h('option', { value: k, title: hintFor(`${list}[].${k}`).doc || '' }, `${labelFor(k)}  (${k})`))) : null;
    add?.addEventListener('change', () => {
      const k = add.value;
      if (!k) return;
      const hint = hintFor(`${list}[].${k}`);
      item[k] = clone(hint.example ?? (hint.enum ? hint.enum[0] : ''));
      changed();
      drawBody();
    });
    body.replaceChildren(f, add ? h('div', { class: 'add-row' }, add) : null);
  };
  head.addEventListener('click', () => {
    el.classList.toggle('collapsed');
    if (el.classList.contains('collapsed')) S.openObjs.delete(item);
    else { S.openObjs.add(item); if (!drawn) drawBody(); }
  });
  if (open) drawBody();
  refresh();
  el.append(head, body);
  return el;
}

async function addFromTemplate(list, redrawList) {
  const titles = { props: 'Add a labelled prop', randomize: 'Add a randomizer', stages: 'Add a stage' };
  const tpls = G.schema.templates[list];
  const idx = await modal({
    title: titles[list],
    body: (close) => h('div', { class: 'template-list' }, tpls.map((t, i) =>
      h('button', { type: 'button', onClick: () => close(i) }, h('b', {}, t.label), h('small', {}, t.doc)))),
    actions: [{ label: 'Cancel', value: null }],
  });
  if (idx == null) return;
  const item = clone(tpls[idx].item);
  item.name = uniqueName(list, item.name);
  const seg = S.raw.classes?.seg || [];
  if (list === 'props') {
    if ('class' in item) item.class = seg[0] || '';
    if (item.slots) for (const k of Object.keys(item.slots)) item.slots[k] = seg[0] || '';
  }
  (S.raw[list] ??= []).push(item);
  S.openObjs.add(item);
  changed();
  redrawList();
}

function propsCard() {
  return listCard(3, 'props', 'Labelled props', '', {
    summary: propSummary, addLabel: 'Add a prop', empty: 'No labelled props yet.',
    sub: () => 'Which objects get labels. After changing props or classes, run Setup on the Render page.',
  });
}

function randomizeCard() {
  const n = () => (S.raw.randomize || []).length;
  const off = () => (S.raw.randomize || []).filter((r) => r.enabled === false).length;
  return listCard(5, 'randomize', 'Randomization', '', {
    summary: rzSummary, addLabel: 'Add a randomizer', toggle: true, empty: 'Nothing is randomized yet.',
    sub: () => `${n()} randomizer${n() === 1 ? '' : 's'}${off() ? `, ${off()} switched off` : ''} - lights, post-process, materials, props. The haze has its own section.`,
  });
}

function stagesCard() {
  const total = () => (S.raw.stages || []).reduce((a, st) => a + (Number(st.frames) || 0), 0);
  return listCard(7, 'stages', 'Stages', '', {
    summary: stageSummary, addLabel: 'Add a stage', empty: 'Add at least one stage.',
    sub: () => `${fmtInt(total())} images in ${(S.raw.stages || []).length} stage(s), rendered back to back into one dataset`,
  });
}
