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
    checkServer();
    emit('state');
  } catch {
    drawStatus(true);
  }
}

// ------------------------------------------------------------------ app server version
// The pages are read from disk on every request, the server keeps the code it started with. After an update
// the server says so (server.stale) and can restart itself; when a different server answers (a restart, or
// SDG App.bat started again), these pages reload so they match it.
let bootVersion;                 // server version these pages loaded with ('' = a server from before versions)
let bannerKind = null;
const banner = h('div', { id: 'app-banner', class: 'hidden' });
document.body.append(banner);

function setBanner(kind, make) {
  if (kind === bannerKind) return;
  bannerKind = kind;
  banner.replaceChildren(...(kind ? [make()] : []));
  banner.classList.toggle('hidden', !kind);
  fitBanner();
}

function fitBanner() {           // keep the page below the banner
  main.style.paddingTop = bannerKind ? `${banner.offsetHeight + 24}px` : '';
}
window.addEventListener('resize', fitBanner);

function withButton(n, b) { n.append(b); return n; }

function reloadPages() {
  if (recipe.isDirty()) {
    setBanner('reload', () => withButton(notice('info', 'refresh', h('b', {}, 'The app server was restarted'),
      h('span', {}, 'Save your recipe, then reload to get the pages that match it.')),
      btn('Reload', () => location.reload(), { kind: 'primary', icon: 'refresh' })));
    return;
  }
  location.reload();
}

function checkServer() {
  const s = G.state?.server;
  const version = s?.version || '';
  if (bootVersion === undefined) bootVersion = version;
  else if (version !== bootVersion) return reloadPages();
  if (!s) {
    setBanner('old', () => notice('warn', 'alert', h('b', {}, 'The app server is running code from before the last update'),
      h('span', {}, 'Quick train, real-image results and the pipeline runs need the new server. Double-click ',
        h('b', {}, 'SDG App.bat'), ' again: it replaces the running server, and this window reloads by itself.')));
  } else if (s.stale) {
    setBanner('stale', () => withButton(notice('info', 'refresh', h('b', {}, 'The app was updated'),
      h('span', {}, 'Restart the app server to use the new version. Renders and trainings running in their own windows keep going.')),
      btn('Restart app', restartServer, { kind: 'primary', icon: 'refresh', busy: 'Restarting' })));
  } else if (bannerKind !== 'reload') {
    setBanner(null);
  }
}

async function restartServer() {
  const before = G.state?.server?.pid;
  await api('/api/restart', {});
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const p = await api('/api/ping');
      if (p.pid !== before) { reloadPages(); return; }
    } catch { /* switching over */ }
  }
  toast('The app server did not restart: close the "SDG App" window and double-click SDG App.bat', 'bad', 12000);
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
    ['trainer_repo', 'yolo-training folder (full trainings)', 'folder'],
    ['trainer_python', 'Python used for training (the repo venv)', 'python'],
    ['editor_log', 'Unreal editor log (Saved/Logs/<project>.log)', 'file'],
  ];
  const inputs = {};
  const body = fields.map(([k, label, kind]) => {
    const inp = h('input', { type: 'text', value: s[k] || '', spellcheck: false });
    inputs[k] = inp;
    return h('label', { class: 'lbl-block' }, label, h('div', { class: 'path-input' }, inp,
      btn('Browse', async () => { const r = await api('/api/browse', { kind, initial: inp.value }); if (r.path) inp.value = r.path; }, { kind: 'small', busy: '…' })));
  });
  // real labelled datasets: every quick train (and Evaluate) scores the model on all of them
  const sets = h('textarea', { rows: 3, spellcheck: false, class: 'wide mono', placeholder: 'one per line: …/data.yaml or a Roboflow export .zip' });
  sets.value = (s.real_sets || []).join('\n');
  const unstretch = h('input', { type: 'text', value: s.real_unstretch ?? '1280 720', spellcheck: false, style: { width: '120px' } });
  body.splice(2, 0,
    h('label', { class: 'lbl-block' }, 'Real labelled datasets for scoring models (one per line)',
      h('div', { class: 'path-input' }, sets, btn('Add', async () => {
        const r = await api('/api/browse', { kind: 'real', initial: sets.value.split('\n').pop() || '' });
        if (r.path) sets.value = (sets.value.trim() ? sets.value.trim() + '\n' : '') + r.path;
      }, { kind: 'small', busy: '…' }))),
    h('label', { class: 'lbl-block' }, 'Unstretch real images to (W H, empty = off)', unstretch,
      h('div', { class: 'opt-doc' }, 'Roboflow exports stretch 1280x720 camera frames to 640x640; square images are resized back before scoring.')));
  const ok = await modal({ title: 'Settings', wide: true, body, actions: [{ label: 'Cancel', value: false }, { label: 'Save', value: true, kind: 'primary' }] });
  if (!ok) return;
  await api('/api/settings', {
    ...Object.fromEntries(Object.entries(inputs).map(([k, i]) => [k, i.value.trim()])),
    real_sets: sets.value.split('\n').map((x) => x.trim()).filter(Boolean),
    real_unstretch: unstretch.value.trim(),
  });
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
