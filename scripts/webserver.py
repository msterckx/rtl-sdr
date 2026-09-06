#!/usr/bin/env python3
"""Local web UI to remotely start/stop the SDR recorder and browse/play
the WAV files it writes.

Runs entirely on the standard library (no Flask/etc. needed, since the
system Python here is externally-managed and has SoapySDR only via
apt/dist-packages). Serves a single-page control panel plus a small JSON
API used by that page.

Usage:
    python3 scripts/webserver.py --host 0.0.0.0 --port 8080

The server binds to 0.0.0.0 by default so it can be reached from other
devices on the LAN ("remote control"). There is no authentication, so
only run it on a network you trust.
"""

import argparse
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from presets import PRESETS, GROUPS, CATEGORY_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORD_SCRIPT = PROJECT_ROOT / "scripts" / "record.py"
SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "scan.py"
OUTPUT_DIR = PROJECT_ROOT / "output"

LOG_MAXLEN = 1000
STOP_GRACE_SECONDS = 20


class Recorder:
    """Owns the single running record.py/scan.py subprocess, if any."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._args = None
        self._started_at = None
        self._returncode = None
        self._logs = deque(maxlen=LOG_MAXLEN)

    def status(self):
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            meta = self._args or {}
            return {
                "running": running,
                "pid": self._proc.pid if running else None,
                "kind": meta.get("kind"),
                "freq_hz": meta.get("freq"),
                "mode": meta.get("mode"),
                "prefix": meta.get("prefix"),
                "group": meta.get("group"),
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "returncode": self._returncode,
            }

    def logs(self):
        with self._lock:
            return list(self._logs)

    def _log(self, line: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._logs.append(f"[{stamp}] {line}")

    def _reader(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self._log(line.rstrip("\n"))
        proc.wait()
        with self._lock:
            self._returncode = proc.returncode
            self._log(f"process exited with code {proc.returncode}")

    def _start_process(self, cmd: list[str], gain: float | None, transcribe: bool, meta: dict):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("recorder is already running")

            if gain is not None:
                cmd = cmd + ["--gain", str(gain)]
            if not transcribe:
                cmd = cmd + ["--no-transcribe"]

            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self._proc = proc
            self._args = meta
            self._started_at = datetime.now()
            self._returncode = None
            self._logs.clear()
            self._log(f"started: {' '.join(cmd)}")
            threading.Thread(target=self._reader, args=(proc,), daemon=True).start()

    def start(self, freq_hz: float, mode: str, prefix: str, gain: float | None, transcribe: bool,
              category: str | None = None):
        cmd = [
            sys.executable, str(RECORD_SCRIPT),
            "--freq", str(freq_hz),
            "--mode", mode,
            "--prefix", prefix,
            "--out-dir", str(OUTPUT_DIR),
        ]
        if category:
            cmd = cmd + ["--category", category]
        self._start_process(cmd, gain, transcribe,
                             {"kind": "record", "freq": freq_hz, "mode": mode, "prefix": prefix,
                              "category": category})

    def start_scan(self, group: str, gain: float | None, transcribe: bool):
        cmd = [
            sys.executable, str(SCAN_SCRIPT),
            "--group", group,
            "--out-dir", str(OUTPUT_DIR),
        ]
        self._start_process(cmd, gain, transcribe, {"kind": "scan", "group": group})

    def stop(self):
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise RuntimeError("recorder is not running")
            proc.send_signal(signal.SIGINT)
            self._log("sent SIGINT, waiting for graceful shutdown...")

        def _force_kill_if_stuck():
            try:
                proc.wait(timeout=STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                with self._lock:
                    self._log("did not stop in time, killed")

        threading.Thread(target=_force_kill_if_stuck, daemon=True).start()


recorder = Recorder()


def list_audio_files():
    files = []
    for wav_path in sorted(OUTPUT_DIR.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            stat = wav_path.stat()
            duration = None
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    duration = wf.getnframes() / wf.getframerate()
            except (wave.Error, EOFError):
                pass
            txt_path = wav_path.with_suffix(".txt")
            transcript = txt_path.read_text().strip() if txt_path.exists() else None
            files.append({
                "path": str(wav_path.relative_to(OUTPUT_DIR)),
                "name": wav_path.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "duration": duration,
                "transcript": transcript,
                "archived": wav_path.parent != OUTPUT_DIR,
            })
        except OSError:
            continue
    return files


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDR Recorder</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; }
  section { border: 1px solid #8884; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
  h2 { margin-top: 0; font-size: 1.05rem; }
  label { display: block; margin: 0.5rem 0 0.2rem; font-size: 0.9rem; }
  select, input[type=text], input[type=number] { padding: 0.35rem; font-size: 0.95rem; }
  button { padding: 0.5rem 1.1rem; font-size: 0.95rem; cursor: pointer; border-radius: 6px; border: 1px solid #8886; }
  button.primary { background: #2563eb; color: white; border: none; }
  button.danger { background: #dc2626; color: white; border: none; }
  button:disabled { opacity: 0.5; cursor: default; }
  .status-pill { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
  .status-pill.running { background: #16a34a33; color: #16a34a; }
  .status-pill.idle { background: #8884; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid #8883; vertical-align: top; }
  .transcript { color: #888; font-style: italic; max-width: 320px; }
  pre#log { background: #0002; padding: 0.7rem; border-radius: 6px; max-height: 240px; overflow-y: auto; font-size: 0.8rem; white-space: pre-wrap; }
  .row { display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap; }
  #now-playing { position: sticky; bottom: 0; background: canvas; padding: 0.5rem 0; }
  audio { width: 100%; }
  .hint { color: #888; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>RTL-SDR / SDRplay Recorder</h1>

<section>
  <h2>Status: <span id="status-pill" class="status-pill idle">idle</span></h2>
  <p id="status-detail" class="hint"></p>
  <div class="row">
    <div>
      <label for="category">Band</label>
      <select id="category">
        __CATEGORY_OPTIONS__
        <option value="allscan">Scan ALL bands (__TOTAL_CHANNELS__ ch)</option>
        <option value="custom">Custom...</option>
      </select>
    </div>
    <div id="channel-field">
      <label for="channel">Channel</label>
      <select id="channel"></select>
    </div>
    <div id="custom-fields" style="display:none">
      <label for="freq">Freq (Hz)</label>
      <input type="number" id="freq" step="1" placeholder="118255000">
      <label for="custom-mode">Mode</label>
      <select id="custom-mode">
        <option value="am">AM</option>
        <option value="fm">FM</option>
      </select>
      <label for="prefix">Prefix</label>
      <input type="text" id="prefix" placeholder="MYFREQ">
    </div>
    <div>
      <label for="gain">Gain dB (blank = default)</label>
      <input type="number" id="gain" step="1" style="width:6rem">
    </div>
    <div>
      <label><input type="checkbox" id="transcribe" checked> transcribe</label>
    </div>
    <div>
      <button class="primary" id="start-btn" onclick="startRecording()">Start</button>
      <button class="danger" id="stop-btn" onclick="stopRecording()">Stop</button>
    </div>
  </div>
  <h3>Log</h3>
  <pre id="log"></pre>
</section>

<section>
  <h2>Recordings <label class="hint"><input type="checkbox" id="show-archived"> show archived</label></h2>
  <table>
    <thead><tr><th></th><th>File</th><th>Time</th><th>Duration</th><th>Transcript</th></tr></thead>
    <tbody id="files-body"></tbody>
  </table>
</section>

<div id="now-playing" style="display:none">
  <div id="now-playing-name" class="hint"></div>
  <audio id="player" controls></audio>
</div>

<script>
const PRESETS = __PRESETS_JSON__;
const CATEGORIES = __CATEGORIES_JSON__;

const categorySelect = document.getElementById('category');
const channelField = document.getElementById('channel-field');
const channelSelect = document.getElementById('channel');
const customFields = document.getElementById('custom-fields');

function populateChannels() {
  const cat = categorySelect.value;
  if (cat === 'custom' || cat === 'allscan') {
    channelField.style.display = 'none';
    customFields.style.display = cat === 'custom' ? 'block' : 'none';
    return;
  }
  channelField.style.display = 'block';
  customFields.style.display = 'none';
  const info = CATEGORIES[cat];
  channelSelect.innerHTML = '';
  const scanOpt = document.createElement('option');
  scanOpt.value = 'scan:' + cat;
  scanOpt.textContent = `Scan all (${info.channels.length} ch)`;
  channelSelect.appendChild(scanOpt);
  for (const ch of info.channels) {
    const opt = document.createElement('option');
    opt.value = 'record:' + ch.key;
    opt.textContent = ch.label;
    channelSelect.appendChild(opt);
  }
}
categorySelect.addEventListener('change', populateChannels);
populateChannels();

async function refreshStatus() {
  const res = await fetch('/api/status');
  const s = await res.json();
  const pill = document.getElementById('status-pill');
  const detail = document.getElementById('status-detail');
  pill.textContent = s.running ? 'running' : 'idle';
  pill.className = 'status-pill ' + (s.running ? 'running' : 'idle');
  document.getElementById('start-btn').disabled = s.running;
  document.getElementById('stop-btn').disabled = !s.running;
  if (s.running) {
    if (s.kind === 'scan') {
      detail.textContent = `pid ${s.pid} · scanning "${s.group}" · started ${new Date(s.started_at).toLocaleTimeString()}`;
    } else {
      detail.textContent = `pid ${s.pid} · ${(s.freq_hz/1e6).toFixed(3)} MHz (${(s.mode || 'am').toUpperCase()}) · prefix ${s.prefix} · started ${new Date(s.started_at).toLocaleTimeString()}`;
    }
  } else {
    detail.textContent = s.returncode !== null && s.returncode !== undefined ? `last exit code: ${s.returncode}` : 'not running';
  }
}

async function refreshLogs() {
  const res = await fetch('/api/logs');
  const lines = await res.json();
  const pre = document.getElementById('log');
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 10;
  pre.textContent = lines.join('\\n');
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

function fmtDuration(s) {
  if (s === null || s === undefined) return '-';
  return s.toFixed(1) + 's';
}

async function refreshFiles() {
  const res = await fetch('/api/files');
  const files = await res.json();
  const showArchived = document.getElementById('show-archived').checked;
  const body = document.getElementById('files-body');
  body.innerHTML = '';
  for (const f of files) {
    if (f.archived && !showArchived) continue;
    const tr = document.createElement('tr');
    const playTd = document.createElement('td');
    const playBtn = document.createElement('button');
    playBtn.textContent = '▶';
    playBtn.onclick = () => play(f.path, f.name);
    playTd.appendChild(playBtn);
    tr.appendChild(playTd);
    const nameTd = document.createElement('td');
    nameTd.textContent = f.name;
    tr.appendChild(nameTd);
    const timeTd = document.createElement('td');
    timeTd.textContent = new Date(f.mtime).toLocaleString();
    tr.appendChild(timeTd);
    const durTd = document.createElement('td');
    durTd.textContent = fmtDuration(f.duration);
    tr.appendChild(durTd);
    const trTd = document.createElement('td');
    trTd.className = 'transcript';
    trTd.textContent = f.transcript || '';
    tr.appendChild(trTd);
    body.appendChild(tr);
  }
}

function play(path, name) {
  const box = document.getElementById('now-playing');
  const player = document.getElementById('player');
  document.getElementById('now-playing-name').textContent = name;
  player.src = '/audio/' + path.split('/').map(encodeURIComponent).join('/');
  box.style.display = 'block';
  player.play();
}

async function startRecording() {
  const gainVal = document.getElementById('gain').value;
  const gain = gainVal === '' ? null : parseFloat(gainVal);
  const transcribe = document.getElementById('transcribe').checked;
  let body;
  if (categorySelect.value === 'custom') {
    body = {
      kind: 'record',
      freq_hz: parseFloat(document.getElementById('freq').value),
      mode: document.getElementById('custom-mode').value,
      prefix: document.getElementById('prefix').value || 'REC',
      gain, transcribe,
    };
  } else if (categorySelect.value === 'allscan') {
    body = { kind: 'scan', group: 'all', gain, transcribe };
  } else if (channelSelect.value.startsWith('scan:')) {
    body = { kind: 'scan', group: channelSelect.value.slice('scan:'.length), gain, transcribe };
  } else {
    const key = channelSelect.value.slice('record:'.length);
    const p = PRESETS[key];
    body = { kind: 'record', key, freq_hz: p.freq, mode: p.mode, prefix: p.prefix, gain, transcribe };
  }
  const res = await fetch('/api/start', { method: 'POST', body: JSON.stringify(body) });
  if (!res.ok) alert((await res.json()).error);
  refreshStatus();
}

async function stopRecording() {
  const res = await fetch('/api/stop', { method: 'POST' });
  if (!res.ok) alert((await res.json()).error);
  refreshStatus();
}

refreshStatus(); refreshLogs(); refreshFiles();
setInterval(refreshStatus, 2000);
setInterval(refreshLogs, 2000);
setInterval(refreshFiles, 5000);
</script>
</body>
</html>
"""


def render_index() -> bytes:
    category_options = "\n".join(
        f'<option value="{cat}">{label} ({len(GROUPS[cat])} ch)</option>'
        for cat, label in CATEGORY_LABELS.items()
    )
    presets_json = json.dumps({
        k: {"freq": v["freq"], "prefix": v["prefix"], "mode": v["mode"]}
        for k, v in PRESETS.items()
    })
    categories_json = json.dumps({
        cat: {
            "label": label,
            "channels": [
                {"key": k, "label": PRESETS[k]["label"]}
                for k in GROUPS[cat]
            ],
        }
        for cat, label in CATEGORY_LABELS.items()
    })
    html = (INDEX_HTML
            .replace("__CATEGORY_OPTIONS__", category_options)
            .replace("__TOTAL_CHANNELS__", str(len(PRESETS)))
            .replace("__PRESETS_JSON__", presets_json)
            .replace("__CATEGORIES_JSON__", categories_json))
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "SDRRecorderHTTP/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def _serve_audio(self, rel_path: str):
        rel_path = unquote(rel_path)
        target = (OUTPUT_DIR / rel_path).resolve()
        try:
            target.relative_to(OUTPUT_DIR.resolve())
        except ValueError:
            self._send_json({"error": "invalid path"}, 403)
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, 404)
            return

        file_size = target.stat().st_size
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            status = 206
            spec = range_header.split("=", 1)[1]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            end = min(end, file_size - 1)

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with open(target, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = render_index()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/status":
            self._send_json(recorder.status())
        elif parsed.path == "/api/logs":
            self._send_json(recorder.logs())
        elif parsed.path == "/api/files":
            self._send_json(list_audio_files())
        elif parsed.path.startswith("/audio/"):
            self._serve_audio(parsed.path[len("/audio/"):])
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            try:
                body = self._read_json_body()
                kind = str(body.get("kind") or "record")
                gain = body.get("gain")
                gain = float(gain) if gain is not None else None
                transcribe = bool(body.get("transcribe", True))
                if kind == "scan":
                    group = str(body["group"])
                    if group not in GROUPS and group != "all":
                        raise ValueError(f"invalid group: {group}")
                    recorder.start_scan(group, gain, transcribe)
                elif kind == "record":
                    freq_hz = float(body["freq_hz"])
                    mode = str(body.get("mode") or "am")
                    if mode not in ("am", "fm"):
                        raise ValueError(f"invalid mode: {mode}")
                    prefix = str(body.get("prefix") or "REC")
                    key = body.get("key")
                    category = PRESETS[key]["category"] if key in PRESETS else None
                    recorder.start(freq_hz, mode, prefix, gain, transcribe, category)
                else:
                    raise ValueError(f"invalid kind: {kind}")
                self._send_json({"ok": True})
            except (KeyError, ValueError) as exc:
                self._send_json({"error": f"bad request: {exc}"}, 400)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, 409)
        elif parsed.path == "/api/stop":
            try:
                recorder.stop()
                self._send_json({"ok": True})
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, 409)
        else:
            self._send_json({"error": "not found"}, 404)


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="port (default: 8080)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    httpd = Server((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port} (Ctrl+C to stop)", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            recorder.stop()
        except RuntimeError:
            pass
        httpd.shutdown()


if __name__ == "__main__":
    main()
