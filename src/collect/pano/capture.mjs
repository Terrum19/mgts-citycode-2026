// Съёмка панорам Яндекса по точкам из OSM через официальный JS API (ключ YANDEX_MAPS_API_KEY_PANORAMS в .env).
// Инструкция для команды: docs/collect_panoramas.md
//
//   node src/collect/pano/capture.mjs --shard 2/3                 # обычный запуск участника команды
//   node src/collect/pano/capture.mjs --quota manhole=5 --debug   # проба
//
// Бережём лимит (100 запросов/сутки на бесплатном ключе): API грузится один раз на сессию,
// пропускаются точки из configs/panoramas_done.txt (снятые всей командой) и из своего meta.jsonl.
// Нужны только Node >= 22 (встроенный WebSocket) и Chrome/Chromium.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFileSync, writeFileSync, appendFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../../..');
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) =>
  a.startsWith('--') ? [...acc, [a.slice(2), arr[i + 1]]] : acc, []));
const POINTS = resolve(ROOT, args.points ?? 'configs/pano_targets.json');
const DONE_SHARED = resolve(ROOT, args.done ?? 'configs/panoramas_done.txt');
const OUT = resolve(ROOT, args.out ?? 'data/raw/panoramas');
const MAX = Number(args.max ?? 5);                 // сколько НОВЫХ точек обработать за запуск
const TYPES = (args.types ?? 'manhole,street_cabinet,street_lamp').split(',');
const MAX_DIST = {manhole: 25, street_cabinet: 25, street_lamp: 35};
const SPAN = [80, 50];                             // поле зрения, градусы (примерно как у регистратора)
const RENDER_WAIT_MS = Number(args.wait ?? 5000);
// домен, под которым Chrome открывает локальную страницу (должен быть в Referer-ограничениях ключа)
const HOST = args.host ?? 'citycode.test';

const key = (readFileSync(join(ROOT, '.env'), 'utf8').match(/^YANDEX_MAPS_API_KEY_PANORAMS=(.*)$/m) || [])[1]
  ?.trim().replace(/^['"]|['"]$/g, '');
if (!key) { console.error('YANDEX_MAPS_API_KEY_PANORAMS не найден в .env'); process.exit(1); }

mkdirSync(OUT, { recursive: true });
const metaPath = join(OUT, 'meta.jsonl');
const done = new Set(existsSync(metaPath)
  ? readFileSync(metaPath, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l).id) : []);
// точки, которые уже обработал кто-то из команды (обновляется через merge.py и git pull)
if (existsSync(DONE_SHARED))
  for (const l of readFileSync(DONE_SHARED, 'utf8').split('\n')) if (l.trim() && !l.startsWith('#')) done.add(l.trim());

// --shard k/N: детерминированно делим точки между N людьми, k = 1..N
const [SHARD_K, SHARD_N] = (args.shard ?? '1/1').split('/').map(Number);
if (!(SHARD_N >= 1 && SHARD_K >= 1 && SHARD_K <= SHARD_N)) { console.error('--shard должен быть вида 2/3'); process.exit(1); }
const hash = s => { let h = 2166136261; for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; } return h; };

// --quota manhole=40,street_lamp=25,street_cabinet=15 — сколько новых точек каждого типа за запуск.
// По умолчанию 90 точек: укладываемся в суточный лимит 100 с запасом на перезапуски.
const QUOTA = Object.fromEntries((args.quota ?? (args.max ? '' : 'manhole=40,street_lamp=30,street_cabinet=20'))
  .split(',').filter(Boolean).map(kv => { const [k, v] = kv.split('='); return [k, Number(v)]; }));
const todo = JSON.parse(readFileSync(POINTS, 'utf8'))
  .filter(p => TYPES.includes(p.type) && !done.has(p.id) && hash(p.id) % SHARD_N === SHARD_K - 1)
  .reduce((acc, p) => { (acc[p.type] ??= []).push(p); return acc; }, {});
const queue = [];
if (Object.keys(QUOTA).length) {
  // чередуем типы, чтобы при обрыве лимита были представлены все
  const lists = Object.entries(QUOTA).map(([t, n]) => (todo[t] ?? []).slice(0, n));
  for (let i = 0; lists.some(l => l.length > i); i++) for (const l of lists) if (l[i]) queue.push(l[i]);
} else {
  for (let i = 0; queue.length < MAX && Object.values(todo).some(l => l.length > i); i++)
    for (const t of TYPES) if (todo[t]?.[i] && queue.length < MAX) queue.push(todo[t][i]);
}
console.log(`шард ${SHARD_K}/${SHARD_N}; уже снято командой/мной: ${done.size}; в этом запуске: ${queue.length}`);
if (!queue.length) process.exit(0);

// локальный сервер только для page.html (JS API не грузится с file://)
const page = readFileSync(join(HERE, 'page.html'));
const server = createServer((req, res) => { res.writeHead(200, {'Content-Type': 'text/html'}); res.end(page); })
  .listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
const pageUrl = `http://${HOST}:${server.address().port}/page.html?key=${encodeURIComponent(key)}`;

const port = 9300 + Math.floor(Math.random() * 500);
// отдельный браузер для съёмки, чтобы его одного можно было пустить мимо VPN (per-app правило)
const CHROME = args.chrome ?? process.env.PANO_CHROME ?? [
  `${process.env.HOME}/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome`,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  `${process.env.LOCALAPPDATA}\\Google\\Chrome\\Application\\chrome.exe`,
].find(p => p && existsSync(p));
if (!CHROME) { console.error('Не нашёл Chrome/Chromium — укажи путь через --chrome'); process.exit(1); }
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${port}`, '--window-size=1280,800',
  `--host-resolver-rules=MAP ${HOST} 127.0.0.1`,
  '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
  `--user-data-dir=${join(OUT, '.chrome-profile')}`, 'about:blank',
], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));

let wsUrl;
for (let i = 0; i < 50 && !wsUrl; i++) {
  try {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    wsUrl = targets.find(t => t.type === 'page')?.webSocketDebuggerUrl;
  } catch { await sleep(200); }
}
if (!wsUrl) { console.error('Chrome не поднялся'); chrome.kill(); process.exit(1); }

const ws = new WebSocket(wsUrl);
await new Promise(r => ws.addEventListener('open', r, { once: true }));
let seq = 0; const pending = new Map();
const DEBUG = 'debug' in args;
const reqUrls = new Map();
const hideKey = u => u.replace(/apikey=[^&]+/i, 'apikey=***');
ws.addEventListener('message', ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (!DEBUG) return;
  if (m.method === 'Network.requestWillBeSent') reqUrls.set(m.params.requestId, m.params.request.url);
  if (m.method === 'Network.responseReceived' && m.params.response.status >= 400)
    console.error(`  [net] ${m.params.response.status} ${hideKey(m.params.response.url).slice(0, 160)}`);
  if (m.method === 'Network.loadingFailed')
    console.error(`  [net] FAIL ${m.params.errorText} ${hideKey(reqUrls.get(m.params.requestId) || '?').slice(0, 160)}`);
  if (m.method === 'Runtime.consoleAPICalled')
    console.error(`  [console] ${m.params.args.map(a => a.value ?? a.description).join(' ').slice(0, 200)}`);
});
const cdp = (method, params = {}) => new Promise(r => { const id = ++seq; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async expr => {
  const r = await cdp('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || 'JS error');
  return r.result?.result?.value;
};

let exitCode = 0;
try {
  await cdp('Page.enable');
  if (DEBUG) { await cdp('Network.enable'); await cdp('Runtime.enable'); }
  await cdp('Page.navigate', { url: pageUrl });
  let state = 'loading';
  for (let i = 0; i < 100 && state === 'loading'; i++) { await sleep(200); state = await evaluate('window.apiState').catch(() => 'loading'); }
  if (state !== 'ready') throw new Error(`JS API не загрузился: ${state}`);

  let ok = 0, calls = 0;
  for (const p of queue) {
    calls++;
    const r = await evaluate(`window.shoot(${p.lat}, ${p.lon}, ${JSON.stringify(p.type)}, ${MAX_DIST[p.type]}, ${JSON.stringify(SPAN)})`);
    const rec = { id: p.id, type: p.type, lat: p.lat, lon: p.lon, location_id: p.id, ...r };
    if (r.ok) {
      await sleep(RENDER_WAIT_MS);
      const shot = await cdp('Page.captureScreenshot', { format: 'jpeg', quality: 90 });
      const file = `${p.id}.jpg`;
      writeFileSync(join(OUT, file), Buffer.from(shot.result.data, 'base64'));
      rec.file = file; ok++;
    }
    console.log(`${p.id} ${p.type}: ${r.ok ? `ok dist=${r.dist.toFixed(1)}м` : r.reason}${r.detail ? ' ' + r.detail : ''}`);
    // ошибку API в кэш не пишем: точку надо будет повторить
    if (r.reason === 'locate_error') { console.error('ошибка API — останавливаюсь, чтобы не жечь лимит'); exitCode = 2; break; }
    appendFileSync(metaPath, JSON.stringify(rec) + '\n');
  }
  console.log(`\nснято ${ok} из ${calls}; запросов поиска панорамы: ${calls} (+1 загрузка API)`);
  console.log('Упаковать и отправить: node src/collect/pano/pack.mjs --name <твоё_имя>');
} catch (e) {
  console.error(e.message); exitCode = 1;
} finally {
  ws.close(); chrome.kill(); server.close();
}
process.exit(exitCode);
