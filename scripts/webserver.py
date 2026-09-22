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
import urllib.error
import wave
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import brandmeister
import flight_lookup
import pass_predict
import shortwave_id
from presets import PRESETS, GROUPS, CATEGORY_LABELS, DMR_REPEATERS, SPECTRUM_RANGES
from satellites import SATELLITES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORD_SCRIPT = PROJECT_ROOT / "scripts" / "record.py"
SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "scan.py"
SATELLITE_SCRIPT = PROJECT_ROOT / "scripts" / "satellite.py"
DMR_SCRIPT = PROJECT_ROOT / "scripts" / "dmr.py"
SPECTRUM_SCRIPT = PROJECT_ROOT / "scripts" / "spectrum_scan.py"
OUTPUT_DIR = PROJECT_ROOT / "output"
SATELLITE_DIR = OUTPUT_DIR / "satellite"
SATELLITE_SCHEDULE_PATH = SATELLITE_DIR / "schedule.json"
BRANDMEISTER_DIR = OUTPUT_DIR / "brandmeister"
BRANDMEISTER_LOG_PATH = BRANDMEISTER_DIR / "calls.json"
SPECTRUM_LOG_PATH = OUTPUT_DIR / "spectrum_survey.jsonl"
SPECTRUM_WATERFALL_PATH = OUTPUT_DIR / "spectrum_waterfall.json"
SPECTRUM_HITS_LIMIT = 200

LOG_MAXLEN = 1000
STOP_GRACE_SECONDS = 20
SATELLITE_POLL_SECONDS = 30
SATELLITE_PASS_MARGIN_S = 30  # extra capture time past predicted LOS


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
                "start_mhz": meta.get("start_mhz"),
                "end_mhz": meta.get("end_mhz"),
                "record": meta.get("record"),
                "voice_check": meta.get("voice_check"),
                "antenna": meta.get("antenna"),
                "listen_audio": meta.get("listen_audio"),
                "listen_device": meta.get("listen_device"),
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

    def _start_process(self, cmd: list[str], gain: float | None, transcribe: bool, meta: dict,
                        antenna: str | None = None):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("recorder is already running")

            if gain is not None:
                cmd = cmd + ["--gain", str(gain)]
            if antenna:
                cmd = cmd + ["--antenna", antenna]
            if not transcribe:
                cmd = cmd + ["--no-transcribe"]

            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=_pipewire_env(),
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
              category: str | None = None, antenna: str | None = None,
              listen_audio: bool = False, listen_device: str | None = None):
        cmd = [
            sys.executable, str(RECORD_SCRIPT),
            "--freq", str(freq_hz),
            "--mode", mode,
            "--prefix", prefix,
            "--out-dir", str(OUTPUT_DIR),
        ]
        if category:
            cmd = cmd + ["--category", category]
        if listen_audio:
            cmd = cmd + ["--listen"]
            if listen_device:
                cmd = cmd + ["--listen-device", listen_device]
        self._start_process(cmd, gain, transcribe,
                             {"kind": "record", "freq": freq_hz, "mode": mode, "prefix": prefix,
                              "category": category, "antenna": antenna,
                              "listen_audio": listen_audio, "listen_device": listen_device}, antenna)

    def start_scan(self, group: str, gain: float | None, transcribe: bool, antenna: str | None = None):
        cmd = [
            sys.executable, str(SCAN_SCRIPT),
            "--group", group,
            "--out-dir", str(OUTPUT_DIR),
        ]
        self._start_process(cmd, gain, transcribe, {"kind": "scan", "group": group, "antenna": antenna},
                             antenna)

    def start_dmr(self, freq_hz: float, gain: float | None, transcribe: bool, antenna: str | None = None):
        """DMR uses a different decode pipeline (dsd-fme, via scripts/dmr.py)
        than record.py/scan.py's analog AM/FM discriminator chain -- see
        dmr.py's module docstring -- but shares the same single-SDR-owner
        slot as record/scan/satellite."""
        cmd = [
            sys.executable, str(DMR_SCRIPT),
            "--freq", str(freq_hz),
            "--out-dir", str(OUTPUT_DIR),
        ]
        self._start_process(cmd, gain, transcribe,
                             {"kind": "dmr", "freq": freq_hz, "mode": "dmr", "antenna": antenna}, antenna)

    def start_satellite(self, freq_hz: float, duration_s: float, prefix: str,
                         gain: float | None, pipeline: str, antenna: str | None = None):
        """Same single-subprocess-owner path as start()/start_scan() -- there is
        one physical SDR, so a satellite capture and ATC recording/scanning can
        never run concurrently regardless of how separate they look in the UI.
        `transcribe=True` is passed to _start_process purely so it never adds
        a --no-transcribe flag satellite.py doesn't have."""
        cmd = [
            sys.executable, str(SATELLITE_SCRIPT),
            "--freq", str(freq_hz),
            "--duration", str(duration_s),
            "--prefix", prefix,
            "--pipeline", pipeline,
            "--out-dir", str(SATELLITE_DIR),
        ]
        self._start_process(cmd, gain, True,
                             {"kind": "satellite", "freq": freq_hz, "prefix": prefix, "antenna": antenna},
                             antenna)

    def start_spectrum(self, start_mhz: float | None, end_mhz: float | None, gain: float | None,
                        record: bool = False, voice_check: bool = True, antenna: str | None = None):
        """Wide FFT-based survey (scripts/spectrum_scan.py) instead of the
        per-channel squelch record/scan use -- same single-SDR-owner slot.
        start_mhz/end_mhz are left off the command line when not given, so
        spectrum_scan.py's own defaults apply rather than duplicating them here.
        record=True additionally demodulates+writes audio for every detected
        channel (multi-channel, from the same wideband capture) instead of
        just logging where activity was found. voice_check=False disables
        spectrum_scan.py's post-recording noise-vs-speech check so every clip
        that passes the RF squelch gets written, for manually reviewing what
        the check would otherwise have discarded."""
        cmd = [sys.executable, str(SPECTRUM_SCRIPT), "--out", str(SPECTRUM_LOG_PATH),
               "--waterfall-out", str(SPECTRUM_WATERFALL_PATH)]
        if start_mhz is not None:
            cmd = cmd + ["--start-mhz", str(start_mhz)]
        if end_mhz is not None:
            cmd = cmd + ["--end-mhz", str(end_mhz)]
        if record:
            cmd = cmd + ["--record", "--out-dir", str(OUTPUT_DIR)]
            if not voice_check:
                cmd = cmd + ["--no-voice-check"]
        self._start_process(cmd, gain, True,
                             {"kind": "spectrum", "start_mhz": start_mhz, "end_mhz": end_mhz,
                              "record": record, "voice_check": voice_check, "antenna": antenna},
                             antenna)

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


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class SatelliteScheduler:
    """Background thread that fires scheduled satellite passes at their AOS
    time via recorder.start_satellite(). Schedule entries persist to
    SATELLITE_SCHEDULE_PATH so a server restart doesn't lose a scheduled pass;
    on startup, any pending/running entry whose LOS already passed is marked
    "missed" (the server was presumably down over the pass).

    Does not preempt an in-progress manual ATC recording/scan -- there's one
    physical SDR, so a pass that comes due while the recorder is busy is
    marked "skipped (recorder busy)" rather than killing whatever the user
    started by hand.
    """

    def __init__(self, recorder: Recorder):
        self._recorder = recorder
        self._lock = threading.Lock()
        self._entries = self._load()
        self._mark_stale_on_startup()

    def _load(self) -> list[dict]:
        if not SATELLITE_SCHEDULE_PATH.is_file():
            return []
        try:
            return json.loads(SATELLITE_SCHEDULE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def _save_locked(self) -> None:
        SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            SATELLITE_SCHEDULE_PATH.write_text(json.dumps(self._entries, indent=2))
        except OSError:
            pass

    def _mark_stale_on_startup(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            for e in self._entries:
                if e["status"] in ("pending", "running") and _parse_iso(e["los"]) < now:
                    e["status"] = "missed"
            self._save_locked()

    def list(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def schedule(self, key: str, label: str, freq_hz: float, aos: str, los: str,
                 max_elevation_deg: float) -> dict:
        with self._lock:
            for e in self._entries:
                if e["key"] == key and e["aos"] == aos:
                    return e
            entry = {"key": key, "label": label, "freq_hz": freq_hz, "aos": aos,
                      "los": los, "max_elevation_deg": max_elevation_deg, "status": "pending"}
            self._entries.append(entry)
            self._save_locked()
            return entry

    def unschedule(self, key: str, aos: str) -> bool:
        with self._lock:
            kept = [e for e in self._entries
                    if not (e["key"] == key and e["aos"] == aos and e["status"] == "pending")]
            changed = len(kept) != len(self._entries)
            self._entries = kept
            if changed:
                self._save_locked()
            return changed

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        recorder_running = self._recorder.status()["running"]

        with self._lock:
            for e in self._entries:
                if e["status"] == "running" and not recorder_running:
                    e["status"] = "done"
            self._save_locked()
            due = [e for e in self._entries if e["status"] == "pending" and _parse_iso(e["aos"]) <= now]

        for e in due:
            los_dt = _parse_iso(e["los"])
            if los_dt <= now:
                with self._lock:
                    e["status"] = "missed"
                    self._save_locked()
                continue
            duration_s = max(1.0, (los_dt - datetime.now(timezone.utc)).total_seconds()
                              + SATELLITE_PASS_MARGIN_S)
            sat = SATELLITES.get(e["key"], {})
            try:
                self._recorder.start_satellite(e["freq_hz"], duration_s,
                                                e["label"].replace(" ", "_"), None,
                                                sat.get("pipeline", "meteor_m2_lrpt"))
                status = "running"
            except RuntimeError:
                status = "skipped (recorder busy)"
            with self._lock:
                e["status"] = status
                self._save_locked()

    def run_forever(self) -> None:
        while True:
            try:
                self._tick()
            except Exception as exc:  # background thread must never die silently
                print(f"satellite scheduler error: {exc}", file=sys.stderr)
            time.sleep(SATELLITE_POLL_SECONDS)


satellite_scheduler = SatelliteScheduler(recorder)
brandmeister_monitor = brandmeister.BrandmeisterMonitor(BRANDMEISTER_LOG_PATH)


def list_audio_files():
    files = []
    for wav_path in sorted(OUTPUT_DIR.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        if wav_path.name.startswith("TEMP_"):
            continue  # dsd-fme's in-progress DMR call, not renamed to its final name yet
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
            flight_path = wav_path.with_suffix(".flight.json")
            flight = None
            if flight_path.exists():
                try:
                    flight = json.loads(flight_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            files.append({
                "path": str(wav_path.relative_to(OUTPUT_DIR)),
                "name": wav_path.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "duration": duration,
                "transcript": transcript,
                "flight": flight,
                "archived": wav_path.parent != OUTPUT_DIR,
            })
        except OSError:
            continue
    return files


SATELLITE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_satellite_images():
    items = []
    if not SATELLITE_DIR.is_dir():
        return items
    for raw_path in sorted(SATELLITE_DIR.glob("*.s16"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            stat = raw_path.stat()
        except OSError:
            continue
        decoded_dir = SATELLITE_DIR / f"{raw_path.stem}_decoded"
        images = []
        if decoded_dir.is_dir():
            for img_path in sorted(decoded_dir.rglob("*")):
                if img_path.suffix.lower() in SATELLITE_IMAGE_EXTENSIONS:
                    images.append(str(img_path.relative_to(SATELLITE_DIR)))
        items.append({
            "wav": raw_path.name,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size": stat.st_size,
            "images": images,
        })
    return items


def list_spectrum_hits(limit: int = SPECTRUM_HITS_LIMIT):
    if not SPECTRUM_LOG_PATH.is_file():
        return []
    try:
        lines = SPECTRUM_LOG_PATH.read_text().splitlines()
    except OSError:
        return []
    hits = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            hits.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    hits.reverse()  # newest first
    return hits


def _pipewire_env() -> dict:
    """os.environ with XDG_RUNTIME_DIR filled in if missing, so pactl/paplay
    subprocesses can find this user's PipeWire/PulseAudio session socket
    (/run/user/<uid>/pulse/native). webserver.py is launched by startup.sh
    as a plain background process, not from a login/desktop session, so it
    doesn't inherit XDG_RUNTIME_DIR the way an interactive shell does --
    without this, pactl/paplay fail with "Connection refused" and silently
    produce no sinks / no sound, both here and in any record.py child this
    process spawns with --listen."""
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return env


def list_audio_sinks():
    """Local playback sinks (`pactl list sinks short`), for the Radio tab's
    "Listen live" device picker -- lets the UI offer real sink names (a
    plugged-in USB/Bluetooth headset, an HDMI output, ...) instead of the
    user having to find and type one by hand. Excludes wsjtx_in: it's a
    null sink for feeding WSJT-X (see wsjtx_audio_setup.sh), picking it here
    would silently produce no audible sound, which is the exact footgun
    --listen-device exists to let the user avoid."""
    try:
        result = subprocess.run(["pactl", "list", "sinks", "short"],
                                 capture_output=True, text=True, timeout=5, env=_pipewire_env())
    except (OSError, subprocess.TimeoutExpired):
        return []
    sinks = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1] != "wsjtx_in":
            sinks.append({"name": fields[1], "state": fields[4] if len(fields) > 4 else None})
    return sinks


def read_spectrum_waterfall():
    if not SPECTRUM_WATERFALL_PATH.is_file():
        return None
    try:
        return json.loads(SPECTRUM_WATERFALL_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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
  .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  .tabs button { border-radius: 6px 6px 0 0; }
  .tabs button.active { background: #2563eb; color: white; border: none; }
  .status-pill.missed, .status-pill.skipped { background: #dc262633; color: #dc2626; }
  .status-pill.done { background: #8884; }
  .gallery { display: flex; flex-wrap: wrap; gap: 1rem; }
  .gallery figure { margin: 0; width: 200px; }
  .gallery img { width: 100%; border-radius: 6px; display: block; }
  .gallery figcaption { font-size: 0.8rem; color: #888; margin-top: 0.3rem; }
</style>
</head>
<body>
<h1>RTL-SDR / SDRplay Recorder</h1>

<div class="tabs">
  <button id="tab-btn-radio" class="active" onclick="showTab('radio')">Radio</button>
  <button id="tab-btn-satellite" onclick="showTab('satellite')">Satellites</button>
  <button id="tab-btn-dmr" onclick="showTab('dmr')">DMR</button>
  <button id="tab-btn-spectrum" onclick="showTab('spectrum')">Spectrum</button>
</div>

<div id="tab-radio">

<section>
  <h2>Status: <span id="status-pill" class="status-pill idle">idle</span></h2>
  <p id="status-detail" class="hint"></p>
  <div id="shortwave-id-box" style="display:none">
    <button id="shortwave-id-btn" onclick="identifyShortwave('shortwave-id-results')" style="font-size:0.8rem">Identify shortwave station</button>
    <div id="shortwave-id-results" class="hint"></div>
  </div>
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
        <option value="dmr">DMR</option>
      </select>
      <label for="prefix" id="prefix-label">Prefix</label>
      <input type="text" id="prefix" placeholder="MYFREQ">
      <label for="dmr-repeater" id="dmr-repeater-label" style="display:none">Known local repeater</label>
      <select id="dmr-repeater" style="display:none">
        <option value="">Custom frequency...</option>
        __DMR_REPEATER_OPTIONS__
      </select>
    </div>
    <div>
      <label for="gain">Gain dB (blank = default)</label>
      <input type="number" id="gain" step="1" style="width:6rem">
    </div>
    <div>
      <label for="antenna">Antenna</label>
      <select id="antenna">
        <option value="">Default (Antenna A)</option>
        <option value="A">Antenna A</option>
        <option value="B">Antenna B</option>
        <option value="C">Antenna C</option>
      </select>
    </div>
    <div>
      <label><input type="checkbox" id="transcribe" checked> transcribe</label>
    </div>
    <div>
      <label><input type="checkbox" id="listen" onchange="document.getElementById('listen-device-field').style.display = this.checked ? '' : 'none'"> listen live</label>
    </div>
    <div id="listen-device-field" style="display:none">
      <label for="listen-device">Play through</label>
      <select id="listen-device"></select>
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
    <thead><tr><th></th><th>File</th><th>Time</th><th>Duration</th><th>Transcript</th><th>Flight</th></tr></thead>
    <tbody id="files-body"></tbody>
  </table>
</section>

<div id="now-playing" style="display:none">
  <div id="now-playing-name" class="hint"></div>
  <audio id="player" controls></audio>
</div>

</div>

<div id="tab-satellite" style="display:none">

<section>
  <h2>Upcoming Meteor-M2 LRPT passes</h2>
  <p class="hint">Predicted from Antwerp. AOS/LOS in your local time.</p>
  <table>
    <thead><tr><th>AOS</th><th>Satellite</th><th>Max el.</th><th>Duration</th><th>Freq</th><th></th></tr></thead>
    <tbody id="passes-body"></tbody>
  </table>
</section>

<section>
  <h2>Scheduled</h2>
  <table>
    <thead><tr><th>AOS</th><th>Satellite</th><th>Status</th><th></th></tr></thead>
    <tbody id="schedule-body"></tbody>
  </table>
</section>

<section>
  <h2>Decoded images</h2>
  <div class="gallery" id="satellite-gallery"></div>
</section>

</div>

<div id="tab-dmr" style="display:none">

<section>
  <h2>Brandmeister (Belgium) <span id="bm-status-pill" class="status-pill idle">starting...</span></h2>
  <p class="hint">Belgian talkgroup activity on the Brandmeister DMR network (internet-linked repeaters), polled from
    <a href="https://www.ham-dmr.be/" target="_blank" rel="noopener">ham-dmr.be</a>'s community BM206 dashboard.
    This is internet DMR traffic, unrelated to whatever's on the local RF DMR channel (Radio tab).</p>
  <p id="bm-status" class="hint"></p>

  <div class="row">
    <div>
      <label for="bm-hours">Window</label>
      <select id="bm-hours">
        <option value="1" selected>Last hour</option>
        <option value="6">Last 6 hours</option>
        <option value="24">Last 24 hours</option>
      </select>
    </div>
    <div>
      <label for="bm-talkgroup">Talkgroup</label>
      <select id="bm-talkgroup">
        <option value="">All Belgian talkgroups</option>
      </select>
    </div>
  </div>
  <table>
    <thead><tr><th>Time</th><th>Callsign</th><th>Name</th><th>Talkgroup</th><th>Slot</th><th>Duration</th><th>Via</th></tr></thead>
    <tbody id="bm-calls-body"></tbody>
  </table>
</section>

</div>

<div id="tab-spectrum" style="display:none">

<section>
  <h2>Spectrum Survey: <span id="spectrum-status-pill" class="status-pill idle">idle</span></h2>
  <p class="hint">Sweeps a wide slice of spectrum at a time (FFT power per capture) and lists whatever
    frequencies show activity above the local noise floor -- a discovery tool for finding traffic that
    isn't in a preset yet. With "record audio" it also demodulates every simultaneously-active signal
    within each wide capture and writes each one its own WAV (shown in the Radio tab's Recordings list).
    Uses the same SDR as the Radio/DMR/Satellite tabs, so only one of them can run at once.</p>
  <p id="spectrum-status-detail" class="hint"></p>
  <div class="row">
    <div>
      <label for="spectrum-preset">Band</label>
      <select id="spectrum-preset">
        <option value="">Custom range...</option>
        __SPECTRUM_RANGE_OPTIONS__
      </select>
    </div>
    <div>
      <label for="spectrum-start-mhz">Start MHz</label>
      <input type="number" id="spectrum-start-mhz" step="0.1" placeholder="26">
    </div>
    <div>
      <label for="spectrum-end-mhz">End MHz</label>
      <input type="number" id="spectrum-end-mhz" step="0.1" placeholder="470">
    </div>
    <div>
      <label for="spectrum-gain">Gain dB (blank = default)</label>
      <input type="number" id="spectrum-gain" step="1" style="width:6rem">
    </div>
    <div>
      <label for="spectrum-antenna">Antenna</label>
      <select id="spectrum-antenna">
        <option value="">Default (Antenna A)</option>
        <option value="A">Antenna A</option>
        <option value="B">Antenna B</option>
        <option value="C">Antenna C</option>
      </select>
    </div>
    <div>
      <label><input type="checkbox" id="spectrum-record"> record audio</label>
    </div>
    <div>
      <label title="Uncheck to write every clip that passes the RF squelch, skipping the noise-vs-speech check -- for manually reviewing what would otherwise get discarded">
        <input type="checkbox" id="spectrum-voice-check" checked> voice check</label>
    </div>
    <div>
      <button class="primary" id="spectrum-start-btn" onclick="startSpectrum()">Start</button>
      <button class="danger" id="spectrum-stop-btn" onclick="stopRecording()">Stop</button>
    </div>
  </div>
  <h3>Log</h3>
  <pre id="spectrum-log"></pre>
</section>

<section>
  <h2>Waterfall <button id="waterfall-toggle-btn" onclick="toggleWaterfall()" style="font-size:0.8rem">Disable</button></h2>
  <p class="hint">Live FFT power across the whole swept range -- brighter/warmer means stronger signal.
    New rows scroll in from the top; the highlighted band marks whatever slice the sweep is
    currently dwelling on. Hover the waterfall to read off a frequency, click to tune the SDR there
    and (optionally) listen live -- this stops the sweep, since there's only one SDR; use "Resume
    sweep" below to go back to surveying the same range.</p>
  <div id="waterfall-freq-label" class="hint"></div>
  <div class="row">
    <div>
      <label><input type="checkbox" id="spectrum-listen" checked> listen live on click</label>
    </div>
    <div>
      <label for="spectrum-listen-device">Play through</label>
      <select id="spectrum-listen-device"></select>
    </div>
    <div>
      <label for="spectrum-tune-mode">Tune mode</label>
      <select id="spectrum-tune-mode">
        <option value="am">AM</option>
        <option value="fm" selected>FM</option>
      </select>
    </div>
  </div>
  <div id="spectrum-tuned-info" class="hint" style="display:none">
    Sweep paused while tuned in -- waterfall is showing the last frame from before the click.
    <button id="spectrum-resume-btn" onclick="resumeSpectrumSweep()">Resume sweep</button>
  </div>
  <div id="spectrum-shortwave-id-box" style="display:none">
    <button id="spectrum-shortwave-id-btn" onclick="identifyShortwave('spectrum-shortwave-id-results')" style="font-size:0.8rem">Identify shortwave station</button>
    <div id="spectrum-shortwave-id-results" class="hint"></div>
  </div>
  <div id="wf-wrap" style="position:relative">
    <canvas id="spectrum-waterfall" width="900" height="260"
      style="width:100%;height:260px;background:#000;border-radius:4px;image-rendering:pixelated;display:block"></canvas>
    <div id="wf-active-marker"
      style="position:absolute;top:0;height:100%;background:rgba(255,255,255,0.18);pointer-events:none;display:none"></div>
    <div id="wf-hover-line"
      style="position:absolute;top:0;height:100%;width:1px;background:rgba(255,255,255,0.6);pointer-events:none;display:none"></div>
    <div id="wf-hover-freq"
      style="position:absolute;top:2px;transform:translateX(-50%);background:rgba(0,0,0,0.7);color:#fff;
      padding:1px 4px;border-radius:3px;font-size:0.72rem;pointer-events:none;display:none;white-space:nowrap"></div>
  </div>
  <div id="wf-axis" style="position:relative;height:1.1rem;margin-top:2px;font-size:0.72rem"
    class="hint"></div>
</section>

<section>
  <h2>Recent activity</h2>
  <table>
    <thead><tr><th>Time</th><th>Freq</th><th>Bandwidth</th><th>SNR</th></tr></thead>
    <tbody id="spectrum-hits-body"></tbody>
  </table>
</section>

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

const customModeSelect = document.getElementById('custom-mode');
const dmrRepeaterSelect = document.getElementById('dmr-repeater');
function updateCustomModeFields() {
  // DMR has no --prefix concept: dsd-fme names each call's WAV itself from
  // the decoded talkgroup/radio IDs.
  const isDmr = customModeSelect.value === 'dmr';
  document.getElementById('prefix-label').style.display = isDmr ? 'none' : 'block';
  document.getElementById('prefix').style.display = isDmr ? 'none' : 'block';
  document.getElementById('dmr-repeater-label').style.display = isDmr ? 'block' : 'none';
  dmrRepeaterSelect.style.display = isDmr ? 'block' : 'none';
}
customModeSelect.addEventListener('change', updateCustomModeFields);
dmrRepeaterSelect.addEventListener('change', () => {
  if (dmrRepeaterSelect.value) document.getElementById('freq').value = dmrRepeaterSelect.value;
});
updateCustomModeFields();

async function refreshStatus() {
  const res = await fetch('/api/status');
  const s = await res.json();
  const pill = document.getElementById('status-pill');
  const detail = document.getElementById('status-detail');
  const isSpectrum = s.running && s.kind === 'spectrum';
  const isShortwaveAm = s.running && s.kind === 'record' && s.mode === 'am' && s.freq_hz != null && s.freq_hz < 30e6;
  pill.textContent = s.running ? 'running' : 'idle';
  pill.className = 'status-pill ' + (s.running ? 'running' : 'idle');
  document.getElementById('start-btn').disabled = s.running;
  document.getElementById('stop-btn').disabled = !s.running;
  if (s.running) {
    const antSuffix = s.antenna ? ` · antenna ${s.antenna}` : '';
    const listenSuffix = s.listen_audio ? ` · listening live -> ${s.listen_device || 'default sink'}` : '';
    if (s.kind === 'scan') {
      detail.textContent = `pid ${s.pid} · scanning "${s.group}"${antSuffix} · started ${new Date(s.started_at).toLocaleTimeString()}`;
    } else if (s.kind === 'dmr') {
      detail.textContent = `pid ${s.pid} · ${(s.freq_hz/1e6).toFixed(4)} MHz (DMR, via dsd-fme)${antSuffix} · started ${new Date(s.started_at).toLocaleTimeString()}`;
    } else if (s.kind === 'spectrum') {
      const range = (s.start_mhz != null ? s.start_mhz : '26') + '-' + (s.end_mhz != null ? s.end_mhz : '470');
      let mode = s.record ? 'survey+record' : 'survey';
      if (s.record && s.voice_check === false) mode += ', voice check off';
      detail.textContent = `pid ${s.pid} · sweeping ${range} MHz (${mode})${antSuffix} · started ${new Date(s.started_at).toLocaleTimeString()}`;
    } else {
      detail.textContent = `pid ${s.pid} · ${(s.freq_hz/1e6).toFixed(3)} MHz (${(s.mode || 'am').toUpperCase()}) · prefix ${s.prefix}${antSuffix}${listenSuffix} · started ${new Date(s.started_at).toLocaleTimeString()}`;
    }
  } else {
    detail.textContent = s.returncode !== null && s.returncode !== undefined ? `last exit code: ${s.returncode}` : 'not running';
  }
  document.getElementById('shortwave-id-box').style.display = isShortwaveAm ? 'block' : 'none';

  const isTuned = s.running && s.kind === 'record' && s.prefix === 'SPEC_TUNE';
  const spPill = document.getElementById('spectrum-status-pill');
  const spDetail = document.getElementById('spectrum-status-detail');
  spPill.textContent = isSpectrum ? 'running' : (isTuned ? 'listening' : (s.running ? 'busy (other tab)' : 'idle'));
  spPill.className = 'status-pill ' + ((isSpectrum || isTuned) ? 'running' : 'idle');
  document.getElementById('spectrum-start-btn').disabled = s.running;
  document.getElementById('spectrum-stop-btn').disabled = !(isSpectrum || isTuned);
  spDetail.textContent = (isSpectrum || isTuned) ? detail.textContent
    : (s.running ? `SDR is busy with "${s.kind}" -- stop it from the ${s.kind === 'dmr' ? 'Radio' : 'other'} tab first` : '');
  document.getElementById('spectrum-tuned-info').style.display = (isTuned && lastSweepParams) ? 'block' : 'none';
  document.getElementById('spectrum-shortwave-id-box').style.display = (isTuned && isShortwaveAm) ? 'block' : 'none';
}

async function identifyShortwave(targetId) {
  const target = document.getElementById(targetId);
  const status = await (await fetch('/api/status')).json();
  if (!status.running || status.mode !== 'am' || status.freq_hz == null) return;
  target.textContent = 'Looking up (EiBi shortwave schedule)...';
  const res = await fetch('/api/shortwave-id?freq_hz=' + status.freq_hz);
  const matches = await res.json();
  target.innerHTML = '';
  if (!res.ok) { target.textContent = matches.error || 'lookup failed'; return; }
  if (!Array.isArray(matches) || matches.length === 0) {
    target.textContent = 'No scheduled broadcast found for this frequency/time in the EiBi schedule.';
    return;
  }
  const note = document.createElement('div');
  note.className = 'hint';
  note.textContent = 'Candidates (schedule match, not a decode -- confirm by ear):';
  target.appendChild(note);
  for (const m of matches) {
    const row = document.createElement('div');
    const name = document.createElement('span');
    name.textContent = m.station;
    const meta = document.createElement('span');
    meta.className = 'hint';
    meta.textContent = ` (${m.time} UTC ${m.days || 'daily'} · lang=${m.language} `
      + `· target=${m.target} · ${m.country}${m.site ? ' · site=' + m.site : ''})`;
    row.appendChild(name);
    row.appendChild(meta);
    target.appendChild(row);
  }
}

async function refreshLogs() {
  const res = await fetch('/api/logs');
  const lines = await res.json();
  const text = lines.join('\\n');
  for (const id of ['log', 'spectrum-log']) {
    const pre = document.getElementById(id);
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 10;
    pre.textContent = text;
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }
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
    tr.appendChild(flightCell(f));
    body.appendChild(tr);
  }
}

function flightCell(f) {
  const td = document.createElement('td');
  if (!f.transcript) {
    td.textContent = '-';
    return td;
  }
  if (f.flight) {
    renderFlightResult(td, f.flight, f.path);
    return td;
  }
  const btn = document.createElement('button');
  btn.textContent = 'Locate';
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = 'Locating...';
    try {
      const res = await fetch('/api/locate?path=' + encodeURIComponent(f.path));
      const result = await res.json();
      renderFlightResult(td, result, f.path);
    } catch (e) {
      td.textContent = 'lookup failed';
    }
  };
  td.appendChild(btn);
  return td;
}

function renderFlightResult(td, result, path) {
  td.innerHTML = '';
  if (result.error) {
    td.className = 'hint';
    td.textContent = result.error;
    return;
  }
  const candidates = result.candidates || [];
  if (candidates.length === 0) {
    td.className = 'hint';
    td.textContent = `no ${result.prefix} flights found nearby`;
    return;
  }
  const top = candidates[0];
  const summary = document.createElement('div');
  const rangeText = top.range_km != null ? `, ${top.range_km} km` : '';
  summary.textContent = `${(top.callsign || '').trim()} (${top.icao24}${rangeText})`;
  td.appendChild(summary);
  if (result.warning) {
    const warn = document.createElement('div');
    warn.className = 'hint';
    warn.textContent = result.warning;
    td.appendChild(warn);
  }
  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = 'Refresh';
  refreshBtn.onclick = async () => {
    refreshBtn.disabled = true;
    const res = await fetch('/api/locate?path=' + encodeURIComponent(path) + '&refresh=1');
    renderFlightResult(td, await res.json(), path);
  };
  td.appendChild(refreshBtn);
}

function play(path, name) {
  const box = document.getElementById('now-playing');
  const player = document.getElementById('player');
  document.getElementById('now-playing-name').textContent = name;
  player.src = '/audio/' + path.split('/').map(encodeURIComponent).join('/');
  box.style.display = 'block';
  player.play();
}

async function refreshAudioSinks() {
  const res = await fetch('/api/audio-sinks');
  const sinks = await res.json();
  for (const id of ['listen-device', 'spectrum-listen-device']) {
    const select = document.getElementById(id);
    const prev = select.value;
    select.innerHTML = '<option value="">System default</option>';
    for (const s of sinks) {
      const opt = document.createElement('option');
      opt.value = s.name;
      opt.textContent = s.name;
      select.appendChild(opt);
    }
    if ([...select.options].some(o => o.value === prev)) select.value = prev;
  }
}

async function startRecording() {
  const gainVal = document.getElementById('gain').value;
  const gain = gainVal === '' ? null : parseFloat(gainVal);
  const transcribe = document.getElementById('transcribe').checked;
  const antenna = document.getElementById('antenna').value || null;
  const listenAudio = document.getElementById('listen').checked;
  const listenDevice = document.getElementById('listen-device').value || null;
  let body;
  if (categorySelect.value === 'custom') {
    const mode = document.getElementById('custom-mode').value;
    if (mode === 'dmr') {
      body = { kind: 'dmr', freq_hz: parseFloat(document.getElementById('freq').value), gain, transcribe, antenna };
    } else {
      body = {
        kind: 'record',
        freq_hz: parseFloat(document.getElementById('freq').value),
        mode,
        prefix: document.getElementById('prefix').value || 'REC',
        gain, transcribe, antenna, listen_audio: listenAudio, listen_device: listenDevice,
      };
    }
  } else if (categorySelect.value === 'allscan') {
    body = { kind: 'scan', group: 'all', gain, transcribe, antenna };
  } else if (channelSelect.value.startsWith('scan:')) {
    body = { kind: 'scan', group: channelSelect.value.slice('scan:'.length), gain, transcribe, antenna };
  } else {
    const key = channelSelect.value.slice('record:'.length);
    const p = PRESETS[key];
    body = {
      kind: 'record', key, freq_hz: p.freq, mode: p.mode, prefix: p.prefix,
      gain, transcribe, antenna, listen_audio: listenAudio, listen_device: listenDevice,
    };
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

function showTab(name) {
  document.getElementById('tab-radio').style.display = name === 'radio' ? 'block' : 'none';
  document.getElementById('tab-satellite').style.display = name === 'satellite' ? 'block' : 'none';
  document.getElementById('tab-dmr').style.display = name === 'dmr' ? 'block' : 'none';
  document.getElementById('tab-spectrum').style.display = name === 'spectrum' ? 'block' : 'none';
  document.getElementById('tab-btn-radio').className = name === 'radio' ? 'active' : '';
  document.getElementById('tab-btn-satellite').className = name === 'satellite' ? 'active' : '';
  document.getElementById('tab-btn-dmr').className = name === 'dmr' ? 'active' : '';
  document.getElementById('tab-btn-spectrum').className = name === 'spectrum' ? 'active' : '';
  if (name === 'satellite') { refreshPasses(); refreshSchedule(); refreshSatelliteGallery(); }
  if (name === 'dmr') { refreshBrandmeisterStatus(); refreshBrandmeisterCalls(); }
  if (name === 'spectrum') { refreshSpectrumHits(); refreshWaterfall(); }
}

const SPECTRUM_RANGES = __SPECTRUM_RANGES_JSON__;
document.getElementById('spectrum-preset').addEventListener('change', () => {
  const key = document.getElementById('spectrum-preset').value;
  if (!key) return;
  const r = SPECTRUM_RANGES[key];
  document.getElementById('spectrum-start-mhz').value = r.start_mhz;
  document.getElementById('spectrum-end-mhz').value = r.end_mhz;
});

async function startSpectrum() {
  const startVal = document.getElementById('spectrum-start-mhz').value;
  const endVal = document.getElementById('spectrum-end-mhz').value;
  const gainVal = document.getElementById('spectrum-gain').value;
  const body = {
    kind: 'spectrum',
    start_mhz: startVal === '' ? null : parseFloat(startVal),
    end_mhz: endVal === '' ? null : parseFloat(endVal),
    gain: gainVal === '' ? null : parseFloat(gainVal),
    record: document.getElementById('spectrum-record').checked,
    voice_check: document.getElementById('spectrum-voice-check').checked,
    antenna: document.getElementById('spectrum-antenna').value || null,
  };
  lastSweepParams = body;  // so a later click-to-tune's "Resume sweep" can restart this same sweep
  const res = await fetch('/api/start', { method: 'POST', body: JSON.stringify(body) });
  if (!res.ok) alert((await res.json()).error);
  wfRangeKey = null;  // next frame's range may differ -- force the waterfall to clear
  refreshStatus();
}

let lastSweepParams = null;

// Stops whatever's running (sweep or a previous tuned-listen) and waits for the
// SDR to actually release -- record.py/spectrum_scan.py handle SIGINT quickly,
// but starting a new process while the old one is still mid-teardown just 409s,
// so this polls briefly rather than guessing a fixed delay.
async function stopAndWait() {
  const status = await (await fetch('/api/status')).json();
  if (!status.running) return status;
  if (status.kind === 'spectrum' && !lastSweepParams) {
    lastSweepParams = {
      kind: 'spectrum', start_mhz: status.start_mhz, end_mhz: status.end_mhz,
      gain: null, record: status.record, voice_check: status.voice_check, antenna: status.antenna,
    };
  }
  await fetch('/api/stop', { method: 'POST' });
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 250));
    const s = await (await fetch('/api/status')).json();
    if (!s.running) return s;
  }
  return await (await fetch('/api/status')).json();
}

async function tuneListenAt(hz) {
  await stopAndWait();
  const gainVal = document.getElementById('spectrum-gain').value;
  const body = {
    kind: 'record',
    freq_hz: hz,
    mode: document.getElementById('spectrum-tune-mode').value,
    prefix: 'SPEC_TUNE',
    gain: gainVal === '' ? null : parseFloat(gainVal),
    transcribe: false,
    antenna: document.getElementById('spectrum-antenna').value || null,
    listen_audio: document.getElementById('spectrum-listen').checked,
    listen_device: document.getElementById('spectrum-listen-device').value || null,
  };
  const res = await fetch('/api/start', { method: 'POST', body: JSON.stringify(body) });
  if (!res.ok) alert((await res.json()).error);
  refreshStatus();
}

async function resumeSpectrumSweep() {
  if (!lastSweepParams) return;
  await stopAndWait();
  const params = lastSweepParams;
  lastSweepParams = null;
  const res = await fetch('/api/start', { method: 'POST', body: JSON.stringify(params) });
  if (!res.ok) alert((await res.json()).error);
  wfRangeKey = null;
  refreshStatus();
}

const wfCanvas = document.getElementById('spectrum-waterfall');
const wfCtx = wfCanvas.getContext('2d');
const wfRowCanvas = document.createElement('canvas');
wfRowCanvas.height = 1;
let wfLastTs = null;
let wfRangeKey = null;
let wfEnabled = true;
let wfStartHz = null;
let wfEndHz = null;

function toggleWaterfall() {
  wfEnabled = !wfEnabled;
  document.getElementById('waterfall-toggle-btn').textContent = wfEnabled ? 'Disable' : 'Enable';
  document.getElementById('wf-wrap').style.display = wfEnabled ? 'block' : 'none';
  document.getElementById('wf-axis').style.display = wfEnabled ? 'block' : 'none';
  document.getElementById('waterfall-freq-label').style.display = wfEnabled ? 'block' : 'none';
  if (wfEnabled) { wfLastTs = null; refreshWaterfall(); }
}

function updateWaterfallAxis(startHz, endHz) {
  const axis = document.getElementById('wf-axis');
  axis.innerHTML = '';
  const span = endHz - startHz;
  if (!(span > 0)) return;
  const numTicks = 6;
  const decimals = span < 20e6 ? 3 : (span < 200e6 ? 2 : 1);
  for (let i = 0; i < numTicks; i++) {
    const frac = i / (numTicks - 1);
    const label = document.createElement('span');
    label.textContent = ((startHz + span * frac) / 1e6).toFixed(decimals);
    label.style.position = 'absolute';
    label.style.left = (frac * 100) + '%';
    label.style.transform = frac === 0 ? 'translateX(0)' : (frac === 1 ? 'translateX(-100%)' : 'translateX(-50%)');
    axis.appendChild(label);
  }
}

wfCanvas.addEventListener('mousemove', (e) => {
  if (wfStartHz === null || wfEndHz === null) return;
  const rect = wfCanvas.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const hz = wfStartHz + (wfEndHz - wfStartHz) * frac;
  const line = document.getElementById('wf-hover-line');
  const label = document.getElementById('wf-hover-freq');
  line.style.left = (frac * 100) + '%';
  line.style.display = 'block';
  label.style.left = (frac * 100) + '%';
  label.textContent = (hz / 1e6).toFixed(4) + ' MHz';
  label.style.display = 'block';
});
wfCanvas.style.cursor = 'crosshair';
wfCanvas.addEventListener('click', (e) => {
  if (wfStartHz === null || wfEndHz === null) return;
  const rect = wfCanvas.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const hz = wfStartHz + (wfEndHz - wfStartHz) * frac;
  tuneListenAt(hz);
});
wfCanvas.addEventListener('mouseleave', () => {
  document.getElementById('wf-hover-line').style.display = 'none';
  document.getElementById('wf-hover-freq').style.display = 'none';
});

function wfColor(v, lo, hi) {
  if (v === null || v === undefined || Number.isNaN(v)) return [0, 0, 0];
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const stops = [
    [0.00, [0, 0, 40]],
    [0.25, [0, 0, 200]],
    [0.50, [0, 200, 200]],
    [0.75, [230, 230, 0]],
    [1.00, [230, 30, 30]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
      return [0, 1, 2].map(k => Math.round(c0[k] + (c1[k] - c0[k]) * f));
    }
  }
  return stops[stops.length - 1][1];
}

function clearWaterfall() {
  wfCtx.fillStyle = '#000';
  wfCtx.fillRect(0, 0, wfCanvas.width, wfCanvas.height);
  wfLastTs = null;
}

async function refreshWaterfall() {
  if (!wfEnabled) return;
  const res = await fetch('/api/spectrum/waterfall');
  const frame = res.ok ? await res.json() : null;
  const marker = document.getElementById('wf-active-marker');
  if (!frame || !Array.isArray(frame.power_db) || frame.power_db.length === 0) {
    marker.style.display = 'none';
    return;
  }
  const rangeKey = frame.start_hz + '-' + frame.end_hz;
  if (rangeKey !== wfRangeKey) {
    wfRangeKey = rangeKey;
    clearWaterfall();
    wfStartHz = frame.start_hz;
    wfEndHz = frame.end_hz;
    updateWaterfallAxis(frame.start_hz, frame.end_hz);
  }
  if (frame.ts === wfLastTs) return;
  wfLastTs = frame.ts;

  const values = frame.power_db.filter(v => v !== null && v !== undefined);
  if (values.length > 0) {
    const sorted = [...values].sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.05)];
    let hi = sorted[Math.floor(sorted.length * 0.995)];
    if (!(hi > lo)) hi = lo + 1;

    const n = frame.power_db.length;
    wfRowCanvas.width = n;
    const rowCtx = wfRowCanvas.getContext('2d');
    const imgData = rowCtx.createImageData(n, 1);
    for (let i = 0; i < n; i++) {
      const [r, g, b] = wfColor(frame.power_db[i], lo, hi);
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    rowCtx.putImageData(imgData, 0, 0);

    const W = wfCanvas.width, H = wfCanvas.height;
    wfCtx.drawImage(wfCanvas, 0, 0, W, H - 1, 0, 1, W, H - 1);
    wfCtx.imageSmoothingEnabled = false;
    wfCtx.drawImage(wfRowCanvas, 0, 0, n, 1, 0, 0, W, 1);
  }

  document.getElementById('waterfall-freq-label').textContent =
    `${(frame.start_hz / 1e6).toFixed(1)}–${(frame.end_hz / 1e6).toFixed(1)} MHz`;

  if (frame.active_start_hz != null && frame.end_hz > frame.start_hz) {
    const span = frame.end_hz - frame.start_hz;
    const leftPct = 100 * (frame.active_start_hz - frame.start_hz) / span;
    const widthPct = Math.max(0.5, 100 * (frame.active_end_hz - frame.active_start_hz) / span);
    marker.style.left = leftPct + '%';
    marker.style.width = widthPct + '%';
    marker.style.display = 'block';
  } else {
    marker.style.display = 'none';
  }
}

function fmtHitTime(iso) {
  return new Date(iso).toLocaleTimeString();
}

async function refreshSpectrumHits() {
  const res = await fetch('/api/spectrum/hits');
  const hits = await res.json();
  const body = document.getElementById('spectrum-hits-body');
  if (!Array.isArray(hits) || hits.length === 0) {
    body.innerHTML = '<tr><td colspan="4" class="hint">No activity logged yet -- start a survey above.</td></tr>';
    return;
  }
  body.innerHTML = '';
  for (const h of hits) {
    const tr = document.createElement('tr');
    const cells = [
      fmtHitTime(h.ts),
      (h.freq_hz / 1e6).toFixed(4) + ' MHz',
      (h.bandwidth_hz / 1e3).toFixed(1) + ' kHz',
      h.snr_db.toFixed(1) + ' dB',
    ];
    for (const text of cells) {
      const td = document.createElement('td');
      td.textContent = text;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function fmtDurationMin(s) {
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return `${m}m${sec.toString().padStart(2, '0')}s`;
}

let scheduledSet = new Set();

async function refreshPasses() {
  const res = await fetch('/api/satellite/passes?hours=48');
  const passes = await res.json();
  const body = document.getElementById('passes-body');
  if (!Array.isArray(passes)) {
    body.innerHTML = `<tr><td colspan="6" class="hint">${passes.error || 'prediction failed'}</td></tr>`;
    return;
  }
  body.innerHTML = '';
  for (const p of passes) {
    const tr = document.createElement('tr');
    const aosTd = document.createElement('td');
    aosTd.textContent = new Date(p.aos).toLocaleString();
    tr.appendChild(aosTd);
    const satTd = document.createElement('td');
    satTd.textContent = p.label;
    tr.appendChild(satTd);
    const elTd = document.createElement('td');
    elTd.textContent = p.max_elevation_deg.toFixed(0) + '°';
    tr.appendChild(elTd);
    const durTd = document.createElement('td');
    durTd.textContent = fmtDurationMin(p.duration_s);
    tr.appendChild(durTd);
    const freqTd = document.createElement('td');
    freqTd.textContent = (p.freq_hz / 1e6).toFixed(3) + ' MHz';
    tr.appendChild(freqTd);
    const actionTd = document.createElement('td');
    const id = p.key + '|' + p.aos;
    if (scheduledSet.has(id)) {
      actionTd.textContent = 'scheduled';
      actionTd.className = 'hint';
    } else {
      const btn = document.createElement('button');
      btn.textContent = 'Schedule';
      btn.onclick = async () => {
        btn.disabled = true;
        await fetch('/api/satellite/schedule', { method: 'POST', body: JSON.stringify(p) });
        refreshSchedule();
      };
      actionTd.appendChild(btn);
    }
    tr.appendChild(actionTd);
    body.appendChild(tr);
  }
}

async function refreshSchedule() {
  const res = await fetch('/api/satellite/schedule');
  const entries = await res.json();
  scheduledSet = new Set(entries.map(e => e.key + '|' + e.aos));
  const body = document.getElementById('schedule-body');
  body.innerHTML = '';
  for (const e of entries) {
    const tr = document.createElement('tr');
    const aosTd = document.createElement('td');
    aosTd.textContent = new Date(e.aos).toLocaleString();
    tr.appendChild(aosTd);
    const satTd = document.createElement('td');
    satTd.textContent = e.label;
    tr.appendChild(satTd);
    const statusTd = document.createElement('td');
    const pill = document.createElement('span');
    pill.className = 'status-pill ' + e.status.split(' ')[0];
    pill.textContent = e.status;
    statusTd.appendChild(pill);
    tr.appendChild(statusTd);
    const actionTd = document.createElement('td');
    if (e.status === 'pending') {
      const btn = document.createElement('button');
      btn.className = 'danger';
      btn.textContent = 'Cancel';
      btn.onclick = async () => {
        btn.disabled = true;
        await fetch('/api/satellite/unschedule', {
          method: 'POST', body: JSON.stringify({ key: e.key, aos: e.aos }),
        });
        refreshSchedule(); refreshPasses();
      };
      actionTd.appendChild(btn);
    }
    tr.appendChild(actionTd);
    body.appendChild(tr);
  }
}

async function refreshSatelliteGallery() {
  const res = await fetch('/api/satellite/images');
  const items = await res.json();
  const gallery = document.getElementById('satellite-gallery');
  gallery.innerHTML = '';
  for (const item of items) {
    for (const img of item.images) {
      const fig = document.createElement('figure');
      const image = document.createElement('img');
      image.src = '/satellite/' + img.split('/').map(encodeURIComponent).join('/');
      fig.appendChild(image);
      const caption = document.createElement('figcaption');
      caption.textContent = new Date(item.mtime).toLocaleString();
      fig.appendChild(caption);
      gallery.appendChild(fig);
    }
  }
  if (!gallery.children.length) {
    gallery.innerHTML = '<p class="hint">No decoded images yet.</p>';
  }
}

let bmTalkgroupsLoaded = false;
async function loadBrandmeisterTalkgroups() {
  if (bmTalkgroupsLoaded) return;
  const res = await fetch('/api/brandmeister/talkgroups');
  const tgs = await res.json();
  const select = document.getElementById('bm-talkgroup');
  for (const [id, label] of Object.entries(tgs)) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = `${label} (${id})`;
    select.appendChild(opt);
  }
  bmTalkgroupsLoaded = true;
}

async function refreshBrandmeisterStatus() {
  const res = await fetch('/api/brandmeister/status');
  const s = await res.json();
  const pill = document.getElementById('bm-status-pill');
  const status = document.getElementById('bm-status');
  if (s.last_poll_ok === false) {
    pill.textContent = 'poll failed';
    pill.className = 'status-pill missed';
    status.textContent = s.error || 'last poll to ham-dmr.be failed.';
  } else if (s.last_poll_ok === true) {
    pill.textContent = 'polling';
    pill.className = 'status-pill running';
    status.textContent = `${s.call_count} call(s) stored -- last checked ${new Date(s.last_poll_at * 1000).toLocaleTimeString()}.`;
  } else {
    pill.textContent = 'starting...';
    pill.className = 'status-pill idle';
    status.textContent = '';
  }
}

function fmtBmTime(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString();
}

async function refreshBrandmeisterCalls() {
  await loadBrandmeisterTalkgroups();
  const hours = document.getElementById('bm-hours').value;
  const talkgroup = document.getElementById('bm-talkgroup').value;
  let url = `/api/brandmeister/calls?hours=${hours}`;
  if (talkgroup) url += `&talkgroup=${talkgroup}`;
  const res = await fetch(url);
  const calls = await res.json();
  const body = document.getElementById('bm-calls-body');
  body.innerHTML = '';
  if (!Array.isArray(calls) || calls.length === 0) {
    body.innerHTML = '<tr><td colspan="7" class="hint">No Belgian calls in this window yet.</td></tr>';
    return;
  }
  for (const c of calls) {
    const tr = document.createElement('tr');
    const cells = [
      fmtBmTime(c.stop),
      c.source_call || '-',
      c.source_name || '-',
      `${c.destination_name} (${c.destination_id})`,
      c.slot != null ? c.slot : '-',
      fmtDuration(c.duration),
      c.via || '-',
    ];
    for (const text of cells) {
      const td = document.createElement('td');
      td.textContent = text;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}
document.getElementById('bm-hours').addEventListener('change', refreshBrandmeisterCalls);
document.getElementById('bm-talkgroup').addEventListener('change', refreshBrandmeisterCalls);

refreshStatus(); refreshLogs(); refreshFiles(); refreshAudioSinks();
setInterval(refreshStatus, 2000);
setInterval(refreshLogs, 2000);
setInterval(refreshFiles, 5000);
setInterval(() => { if (document.getElementById('tab-satellite').style.display !== 'none') { refreshSchedule(); refreshSatelliteGallery(); } }, 10000);
setInterval(() => { if (document.getElementById('tab-dmr').style.display !== 'none') { refreshBrandmeisterStatus(); refreshBrandmeisterCalls(); } }, 15000);
setInterval(() => { if (document.getElementById('tab-spectrum').style.display !== 'none') { refreshSpectrumHits(); } }, 5000);
setInterval(() => { if (document.getElementById('tab-spectrum').style.display !== 'none') { refreshWaterfall(); } }, 600);
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
    dmr_repeater_options = "\n".join(
        f'<option value="{v["freq"]}">{v["label"]}</option>'
        for v in DMR_REPEATERS.values()
    )
    spectrum_range_options = "\n".join(
        f'<option value="{key}">{r["label"]}</option>'
        for key, r in SPECTRUM_RANGES.items()
    )
    spectrum_ranges_json = json.dumps({
        key: {"start_mhz": r["start_mhz"], "end_mhz": r["end_mhz"]}
        for key, r in SPECTRUM_RANGES.items()
    })
    html = (INDEX_HTML
            .replace("__CATEGORY_OPTIONS__", category_options)
            .replace("__TOTAL_CHANNELS__", str(len(PRESETS)))
            .replace("__PRESETS_JSON__", presets_json)
            .replace("__CATEGORIES_JSON__", categories_json)
            .replace("__DMR_REPEATER_OPTIONS__", dmr_repeater_options)
            .replace("__SPECTRUM_RANGE_OPTIONS__", spectrum_range_options)
            .replace("__SPECTRUM_RANGES_JSON__", spectrum_ranges_json))
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

    def _serve_satellite_file(self, rel_path: str):
        rel_path = unquote(rel_path)
        target = (SATELLITE_DIR / rel_path).resolve()
        try:
            target.relative_to(SATELLITE_DIR.resolve())
        except ValueError:
            self._send_json({"error": "invalid path"}, 403)
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        elif parsed.path == "/api/audio-sinks":
            self._send_json(list_audio_sinks())
        elif parsed.path == "/api/locate":
            self._handle_locate(parse_qs(parsed.query))
        elif parsed.path.startswith("/audio/"):
            self._serve_audio(parsed.path[len("/audio/"):])
        elif parsed.path == "/api/shortwave-id":
            self._handle_shortwave_id(parse_qs(parsed.query))
        elif parsed.path == "/api/satellite/passes":
            self._handle_satellite_passes(parse_qs(parsed.query))
        elif parsed.path == "/api/satellite/schedule":
            self._send_json(satellite_scheduler.list())
        elif parsed.path == "/api/satellite/images":
            self._send_json(list_satellite_images())
        elif parsed.path.startswith("/satellite/"):
            self._serve_satellite_file(parsed.path[len("/satellite/"):])
        elif parsed.path == "/api/brandmeister/status":
            self._send_json(brandmeister_monitor.status())
        elif parsed.path == "/api/brandmeister/talkgroups":
            self._send_json(brandmeister_monitor.talkgroups())
        elif parsed.path == "/api/brandmeister/calls":
            self._handle_brandmeister_calls(parse_qs(parsed.query))
        elif parsed.path == "/api/spectrum/hits":
            self._send_json(list_spectrum_hits())
        elif parsed.path == "/api/spectrum/waterfall":
            self._send_json(read_spectrum_waterfall())
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_brandmeister_calls(self, query: dict):
        try:
            hours = float((query.get("hours") or ["1"])[0])
        except ValueError:
            self._send_json({"error": "invalid hours"}, 400)
            return
        talkgroup_raw = (query.get("talkgroup") or [""])[0]
        talkgroup = None
        if talkgroup_raw:
            try:
                talkgroup = int(talkgroup_raw)
            except ValueError:
                self._send_json({"error": "invalid talkgroup"}, 400)
                return
        self._send_json(brandmeister_monitor.calls_since(hours, talkgroup))

    def _handle_satellite_passes(self, query: dict):
        hours = float((query.get("hours") or ["48"])[0])
        min_elevation = float((query.get("min_elevation") or ["5"])[0])
        try:
            passes = pass_predict.upcoming_passes(hours, min_elevation)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            self._send_json({"error": f"pass prediction failed: {exc}"}, 502)
            return
        self._send_json(passes)

    def _handle_shortwave_id(self, query: dict):
        """Candidate broadcasters for a shortwave AM frequency right now,
        via EiBi's published schedule (see shortwave_id.py) -- AM carries no
        station ID in the signal itself, unlike FM's RDS, so this is a
        schedule lookup rather than a decode."""
        freq_vals = query.get("freq_hz")
        if not freq_vals:
            self._send_json({"error": "freq_hz is required"}, 400)
            return
        try:
            freq_hz = float(freq_vals[0])
        except ValueError:
            self._send_json({"error": "invalid freq_hz"}, 400)
            return
        try:
            schedule = shortwave_id.parse_schedule(shortwave_id.fetch_schedule())
            matches = shortwave_id.lookup(schedule, freq_hz)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            self._send_json({"error": f"shortwave lookup failed: {exc}"}, 502)
            return
        self._send_json(matches)

    def _handle_locate(self, query: dict):
        """Look up the real flight behind an ATC transmission (see
        flight_lookup.py). Results are cached in a `.flight.json` sidecar
        next to the WAV -- the recording's timestamp and transcript never
        change, so repeat page loads shouldn't re-spend OpenSky API credits.
        Only successful lookups are cached; transient errors (network,
        rate limit) get retried on the next request."""
        rel_path = unquote((query.get("path") or [""])[0])
        if not rel_path:
            self._send_json({"error": "missing path"}, 400)
            return
        wav_path = (OUTPUT_DIR / rel_path).resolve()
        try:
            wav_path.relative_to(OUTPUT_DIR.resolve())
        except ValueError:
            self._send_json({"error": "invalid path"}, 403)
            return
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.is_file():
            self._send_json({"error": "no transcript for this recording"}, 404)
            return

        cache_path = wav_path.with_suffix(".flight.json")
        refresh = (query.get("refresh") or ["0"])[0] == "1"
        if cache_path.is_file() and not refresh:
            self._send_json(json.loads(cache_path.read_text()))
            return

        result = flight_lookup.locate_flight(txt_path)
        if result.get("error") is None:
            try:
                cache_path.write_text(json.dumps(result))
            except OSError:
                pass
        self._send_json(result)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            try:
                body = self._read_json_body()
                kind = str(body.get("kind") or "record")
                gain = body.get("gain")
                gain = float(gain) if gain is not None else None
                transcribe = bool(body.get("transcribe", True))
                antenna = body.get("antenna") or None
                if antenna is not None and antenna not in ("A", "B", "C"):
                    raise ValueError(f"invalid antenna: {antenna}")
                if kind == "scan":
                    group = str(body["group"])
                    if group not in GROUPS and group != "all":
                        raise ValueError(f"invalid group: {group}")
                    recorder.start_scan(group, gain, transcribe, antenna)
                elif kind == "record":
                    freq_hz = float(body["freq_hz"])
                    mode = str(body.get("mode") or "am")
                    if mode not in ("am", "fm"):
                        raise ValueError(f"invalid mode: {mode}")
                    prefix = str(body.get("prefix") or "REC")
                    key = body.get("key")
                    category = PRESETS[key]["category"] if key in PRESETS else None
                    listen_audio = bool(body.get("listen_audio", False))
                    listen_device = body.get("listen_device") or None
                    recorder.start(freq_hz, mode, prefix, gain, transcribe, category, antenna,
                                    listen_audio, listen_device)
                elif kind == "dmr":
                    freq_hz = float(body["freq_hz"])
                    recorder.start_dmr(freq_hz, gain, transcribe, antenna)
                elif kind == "spectrum":
                    start_mhz = body.get("start_mhz")
                    end_mhz = body.get("end_mhz")
                    recorder.start_spectrum(
                        float(start_mhz) if start_mhz not in (None, "") else None,
                        float(end_mhz) if end_mhz not in (None, "") else None,
                        gain, record=bool(body.get("record", False)),
                        voice_check=bool(body.get("voice_check", True)), antenna=antenna)
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
        elif parsed.path == "/api/satellite/schedule":
            try:
                body = self._read_json_body()
                key = str(body["key"])
                if key not in SATELLITES:
                    raise ValueError(f"unknown satellite: {key}")
                entry = satellite_scheduler.schedule(
                    key, str(body["label"]), float(body["freq_hz"]),
                    str(body["aos"]), str(body["los"]), float(body["max_elevation_deg"]))
                self._send_json(entry)
            except (KeyError, ValueError) as exc:
                self._send_json({"error": f"bad request: {exc}"}, 400)
        elif parsed.path == "/api/satellite/unschedule":
            try:
                body = self._read_json_body()
                ok = satellite_scheduler.unschedule(str(body["key"]), str(body["aos"]))
                self._send_json({"ok": ok})
            except KeyError as exc:
                self._send_json({"error": f"bad request: {exc}"}, 400)
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
    SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
    BRANDMEISTER_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=satellite_scheduler.run_forever, daemon=True).start()
    threading.Thread(target=brandmeister_monitor.run_forever, daemon=True).start()
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
