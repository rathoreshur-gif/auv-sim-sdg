// App shell: routing between the pages, Unreal status, settings.
import { h, api, btn, toast, modal, copyLine, notice } from './ui.js';
import { G, on, emit } from './store.js';
import * as recipe from './recipe.js';
import * as render from './render.js';
import * as train from './train.js';

const PAGES = { recipe, render, train };
const main = document.getElementById('page');
let current = null;

async function refreshState() {
  try {
    G.state = await api('/api/state');
    drawStatus();
    emit('state');
  } catch {
    drawStatus(true);
  }
}

function drawStatus(serverDown) {
  const b = document.getElementById('unreal-status');
  const u = G.state?.unreal || {};
  let dot = 'bad', title = 'Unreal', sub = 'not connected';
  if (serverDown) { title = 'App server'; sub = 'not running - restart SDG App'; }
  else if (u.rendering) { dot = 'busy'; sub = 'rendering…'; }
  else if (u.connected) { dot = 'ok'; sub = (u.level || '').split('/').pop() || 'connected'; }
  else if (!u.never) { dot = 'warn'; sub = 'not responding'; }
  b.replaceChildren(h('span', { class: 'dot ' + dot }), h('div', {}, h('b', {}, title), h('small', {}, sub)));
}

function parseHash() {
  const [page, query] = (location.hash.slice(1) || 'recipe').split('?');
  return { page: PAGES[page] ? page : 'recipe', params: new URLSearchParams(query || '') };
}

async function route() {
  const { page, params } = parseHash();
  for (const a of document.querySelectorAll('#nav a')) a.classList.toggle('active', a.dataset.page === page);
  if (current) PAGES[current].unmount?.();
  current = page;
  main.replaceChildren();
  try {
    await PAGES[page].mount(main, params);
  } catch (e) {
    main.replaceChildren(h('div', { class: 'page' }, notice('bad', 'alert', h('b', {}, 'This page could not load'), e.message)));
  }
}

async function unrealHelp() {
  const u = G.state?.unreal || {};
  await modal({
    title: 'Unreal connection',
    body: [
      u.connected
        ? notice('ok', 'check', h('b', {}, 'Connected'), h('span', {}, `Level open: ${u.level || '?'}${u.rendering ? ' - rendering' : ''}`))
        : notice('warn', 'alert', h('b', {}, 'Not connected'), h('span', {}, 'The app sends setup and render commands to the open Unreal Editor.')),
      h('p', {}, 'The connection starts by itself when the editor opens (Content/Python/init_unreal.py). For an editor that is already open, paste this once into the Output Log with Cmd selected:'),
      copyLine(`py "${G.state?.listener || ''}"`),
    ],
  });
}

async function settingsDialog() {
  const s = G.state?.settings || {};
  const fields = [
    ['trainer_repo', 'yolo-training folder', 'folder'],
    ['trainer_python', 'Python used for training (the repo venv)', 'python'],
    ['real_data', 'Real labelled dataset for evaluation (data.yaml)', 'yaml'],
    ['editor_log', 'Unreal editor log (Saved/Logs/<project>.log)', 'file'],
  ];
  const inputs = {};
  const body = fields.map(([k, label, kind]) => {
    const inp = h('input', { type: 'text', value: s[k] || '', spellcheck: false });
    inputs[k] = inp;
    return h('label', { class: 'lbl-block' }, label, h('div', { class: 'path-input' }, inp,
      btn('Browse', async () => { const r = await api('/api/browse', { kind, initial: inp.value }); if (r.path) inp.value = r.path; }, { kind: 'small', busy: '…' })));
  });
  const ok = await modal({ title: 'Settings', wide: true, body, actions: [{ label: 'Cancel', value: false }, { label: 'Save', value: true, kind: 'primary' }] });
  if (!ok) return;
  await api('/api/settings', Object.fromEntries(Object.entries(inputs).map(([k, i]) => [k, i.value.trim()])));
  toast('Settings saved', 'ok');
  await refreshState();
  route();
}

async function start() {
  try {
    [G.schema] = await Promise.all([api('/api/schema'), refreshState()]);
  } catch (e) {
    main.replaceChildren(h('div', { class: 'page' }, notice('bad', 'alert', h('b', {}, 'The app server is not reachable'), e.message)));
    return;
  }
  document.getElementById('unreal-status').addEventListener('click', unrealHelp);
  document.getElementById('open-settings').addEventListener('click', settingsDialog);
  window.addEventListener('hashchange', route);
  window.addEventListener('beforeunload', (e) => { if (recipe.isDirty()) { e.preventDefault(); e.returnValue = ''; } });
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's' && current === 'recipe') { e.preventDefault(); recipe.saveShortcut(); }
  });
  on((ev) => { if (ev === 'refresh-state') refreshState(); });
  setInterval(refreshState, 2000);
  route();
}

start();
