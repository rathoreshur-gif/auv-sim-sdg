// App-wide state shared by the pages, plus a tiny event bus.
export const G = {
  schema: null,          // /api/schema: defaults, hints, templates
  state: null,           // /api/state: unreal connection, configs, settings (polled)
  config: null,          // config (file stem) selected on the Recipe / Render pages
};

let saved = null;
try { saved = localStorage.getItem('sdg.config'); } catch { /* storage blocked */ }
G.config = saved;

export function setConfig(stem) {
  G.config = stem;
  try { localStorage.setItem('sdg.config', stem || ''); } catch { /* ignore */ }
  emit('config', stem);
}

const listeners = new Set();
export function on(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export function emit(ev, data) { for (const fn of [...listeners]) fn(ev, data); }

// Unreal bridge: send a command and wait for the editor to run it.
import { api } from './ui.js';
export async function unreal(command, args = [], { timeoutS = 900 } = {}) {
  const { id } = await api('/api/unreal', { command, args });
  const t0 = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 700));
    const res = await api('/api/unreal/result?id=' + encodeURIComponent(id));
    if (res.status === 'done') return res;
    if ((Date.now() - t0) / 1000 > timeoutS) throw new Error('Unreal did not answer - is the editor open and the bridge running?');
  }
}
