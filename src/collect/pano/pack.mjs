// Упаковывает свои снятые панорамы в архив, чтобы отправить координатору.
//
//   node src/collect/pano/pack.mjs --name petya
//
// Результат: panoramas_<name>_<дата>.tar.gz в корне проекта (кадры + meta.jsonl).
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) =>
  a.startsWith('--') ? [...acc, [a.slice(2), arr[i + 1]]] : acc, []));
const name = (args.name ?? '').replace(/[^\w-]/g, '');
if (!name) { console.error('Укажи имя: --name petya'); process.exit(1); }

const dir = 'data/raw/panoramas';
const meta = join(ROOT, dir, 'meta.jsonl');
if (!existsSync(meta)) { console.error('Нечего паковать: нет data/raw/panoramas/meta.jsonl'); process.exit(1); }
const files = readFileSync(meta, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).filter(r => r.file);
const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const out = `panoramas_${name}_${stamp}.tar.gz`;
execFileSync('tar', ['-czf', out, '-C', join(ROOT, dir), 'meta.jsonl', ...files.map(r => r.file)], { cwd: ROOT, stdio: 'inherit' });
console.log(`${files.length} кадров -> ${out}\nОтправь этот файл координатору (Telegram/диск).`);
