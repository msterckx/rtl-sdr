#!/usr/bin/env python3
"""Listen to a radio frequency on an SDRplay device and record each
transmission to its own WAV file, using power-based voice-activated squelch.

Supports AM (airband, broadcast AM) and narrowband FM (PMR446, ham, other
NFM voice services) via --mode.
"""

import argparse
import base64
import io
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
from scipy.signal import butter, lfilter

SAMPLE_RATE = 2_000_000.0
AUDIO_RATE = 8_000
DECIMATION = int(SAMPLE_RATE // AUDIO_RATE)  # 250
CHUNK_SECONDS = 0.1
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

DEFAULT_FREQ_HZ = 135.205e6  # EBAW Tower


# Fixed path rather than Path.home(): the recorder is often started as root
# (needed for raw SDR device access), where Path.home() resolves to /root
# instead of the actual user's home where whisper.cpp is installed.
WHISPER_DIR = Path("/home/michel/src/whisper.cpp")
DEFAULT_WHISPER_BIN = WHISPER_DIR / "build" / "bin" / "whisper-cli"
DEFAULT_WHISPER_MODEL = WHISPER_DIR / "models" / "ggml-base.en.bin"

# Hosted ATC-tuned Whisper model (RunPod serverless), used only for clips
# whose category starts with "atc" instead of the generic local whisper.cpp model.
# The key is never hardcoded here since it grants billable API access --
# read it from the environment (or --runpod-api-key) at call time.
DEFAULT_RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/sgb8a8v6ielxna/runsync"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"
RUNPOD_BATCH_SIZE = 3

_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


class DcBlocker:
    """One-pole DC-removal filter, state kept across chunks."""

    def __init__(self, r: float = 0.995):
        self.r = r
        self.x_prev = 0.0
        self.y_prev = 0.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        x_prev, y_prev, r = self.x_prev, self.y_prev, self.r
        for i, xi in enumerate(x):
            yi = xi - x_prev + r * y_prev
            y[i] = yi
            x_prev, y_prev = xi, yi
        self.x_prev, self.y_prev = x_prev, y_prev
        return y


class FmDiscriminator:
    """Quadrature (phase-difference) FM discriminator, carrying the last IQ
    sample across chunks so the phase reference stays continuous."""

    def __init__(self):
        self._prev = None

    def apply(self, iq: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = iq[0]
        extended = np.concatenate(([self._prev], iq))
        self._prev = iq[-1]
        prod = extended[1:] * np.conj(extended[:-1])
        return np.angle(prod).astype(np.float32)


class LivePlayback:
    """Streams every demodulated audio chunk to a local audio sink via paplay,
    for hands-free live listening alongside (not gated by) the squelch-based
    recording -- so you can just tune to a frequency and hear it, the way
    gqrx does live, without running gqrx's GUI at all. Useful on this
    project's Pi 5: the GUI's FFT/waterfall repaint is real RAM/CPU pressure
    on a machine already running the recorder + webserver (see wsjtx_audio_
    setup.sh's sibling problem -- that one routes audio INTO a virtual sink
    for WSJT-X; this one gets audio OUT to a physical/BT/USB output).

    Uses a fixed floor + slow-release AGC rather than write_wav's per-clip
    peak normalization, since this is a live unbounded stream with no known
    peak to normalize against in advance -- fast attack (jumps to a loud
    chunk's peak immediately) avoids clipping, slow release (~1-2s decay)
    avoids the volume visibly pumping between chunks. Same non-blocking
    queue + writer-thread pattern as dmr.py's dsd-fme stdin feed: readStream()
    must keep being serviced every ~0.1s regardless of whether audio playback
    is keeping up, so a chunk is dropped rather than risking a block here.
    """

    AGC_FLOOR = 1e-3   # matches this project's documented raw-demod amplitude range (1e-4..1e-1)
    AGC_RELEASE = 0.95  # per-chunk decay, ~1.3s time constant at CHUNK_SECONDS=0.1s
    HEADROOM = 0.85

    def __init__(self, audio_rate: int, device: str | None = None):
        # No trailing filename/"-" argument: unlike most CLI tools, paplay
        # doesn't treat "-" as a stdin placeholder -- it tries to literally
        # open() a file named "-" and fails with ENOENT. Passing no filename
        # argument at all is what makes it read from stdin.
        cmd = ["paplay", "--raw", "--format=float32le", f"--rate={audio_rate}",
               "--channels=1", "--latency-msec=100"]
        if device:
            cmd += [f"--device={device}"]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._queue: "queue.Queue[bytes | None]" = queue.Queue(maxsize=50)  # ~5s headroom
        self._peak = self.AGC_FLOOR
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Must be drained continuously or a chatty paplay can fill the pipe
        # buffer and block it -- same rationale as dmr.py's dsd-fme forwarder.
        threading.Thread(target=self._forward_stderr, daemon=True).start()

    def _forward_stderr(self) -> None:
        for raw_line in self._proc.stderr:
            line = raw_line.decode("utf-8", "replace").rstrip("\n")
            if line:
                print(f"  [paplay] {line}", file=sys.stderr)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._proc.stdin.write(item)
            except (BrokenPipeError, OSError):
                return

    def write(self, chunk: np.ndarray) -> None:
        chunk_peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        self._peak = max(chunk_peak, self._peak * self.AGC_RELEASE, self.AGC_FLOOR)
        scaled = np.clip(chunk / self._peak * self.HEADROOM, -1.0, 1.0).astype(np.float32)
        try:
            self._queue.put_nowait(scaled.tobytes())
        except queue.Full:
            pass  # playback can't keep up -- drop this chunk rather than block the SDR read loop

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.terminate()


def open_sdr(gain_db, sample_rate=SAMPLE_RATE, bandwidth_hz=200_000, antenna=None):
    candidates = [r for r in SoapySDR.Device.enumerate() if r["driver"] == "sdrplay"]
    if not candidates:
        raise RuntimeError("no sdrplay device found")
    sdr = SoapySDR.Device(candidates[0])
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setBandwidth(SOAPY_SDR_RX, 0, bandwidth_hz)
    if antenna:
        # SoapySDR's sdrplay driver wants the full name ("Antenna A"), not
        # just the letter our --antenna flag takes.
        sdr.setAntenna(SOAPY_SDR_RX, 0, f"Antenna {antenna}")
    if gain_db is None:
        sdr.setGainMode(SOAPY_SDR_RX, 0, True)  # AGC
    else:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, gain_db)
    return sdr


def write_wav(out_dir: Path, prefix: str, start_time: datetime, audio: np.ndarray) -> Path:
    peak = np.max(np.abs(audio)) or 1.0
    pcm = np.clip(audio / peak * 0.95 * 32767, -32768, 32767).astype(np.int16)
    ts = start_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_path = out_dir / f"{prefix}_{ts}.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(pcm.tobytes())
    return out_path


def transcribe_file(wav_path: Path, whisper_bin: Path, whisper_model: Path,
                     threads: int, skip_ms: float = 0.0) -> None:
    """Run whisper.cpp on a recorded clip and write a .txt transcript next to it.

    Runs in a worker thread submitted from the main SDR read loop -- must never
    be called inline there, since a multi-second whisper call would stall
    readStream() servicing and cause dropped samples.

    `skip_ms` is how much leading pre-roll audio (captured before the squelch
    confirmed a signal was present) to skip via whisper's own --offset-t,
    without touching the WAV file on disk. On a weak-signal AM/FM squelch the
    carrier-open transient is often louder than the speech that follows it,
    and whisper's autoregressive decoder anchors on that noise burst and
    hallucinates a whole confident (wrong) sentence for the clip; skipping it
    reliably turns those into correct "no speech" results instead. Capped
    below at 70% of the clip's duration so a short transmission with little
    or no pre-roll still gets transcribed.
    """
    txt_path = wav_path.with_suffix(".txt")
    cmd = [str(whisper_bin), "-m", str(whisper_model), "-f", str(wav_path),
           "-l", "en", "-t", str(threads), "-np", "-nt"]
    if skip_ms > 0:
        try:
            with wave.open(str(wav_path), "rb") as wf:
                duration_ms = 1000 * wf.getnframes() / wf.getframerate()
        except (wave.Error, EOFError):
            duration_ms = skip_ms
        offset_ms = int(min(skip_ms, 0.7 * duration_ms))
        if offset_ms >= 50:
            cmd += ["-ot", str(offset_ms)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  transcription failed for {wav_path.name}: {exc}", file=sys.stderr)
        return
    if result.returncode != 0:
        print(f"  transcription failed for {wav_path.name}: "
              f"{result.stderr.strip()[-300:]}", file=sys.stderr)
        return
    transcript = result.stdout.strip()
    txt_path.write_text(transcript + "\n")
    preview = transcript if len(transcript) <= 100 else transcript[:97] + "..."
    print(f"  transcript ({wav_path.name}): {preview or '(empty)'}", file=sys.stderr)


def _wav_base64(wav_path: Path, skip_ms: float = 0.0) -> str:
    """Read a WAV file, optionally dropping skip_ms of leading audio (the
    same noise-burst-hallucination workaround used for the local whisper
    --offset-t, since a squelch-open transient anchors a wrong confident
    transcript on the hosted model too), and return it base64-encoded
    without touching the file on disk. Capped at 70% of the clip's duration
    so a short transmission still gets transcribed."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels, sampwidth, framerate, n_frames = (
            wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
        skip_frames = 0
        if skip_ms > 0 and framerate > 0:
            duration_ms = 1000 * n_frames / framerate
            offset_ms = min(skip_ms, 0.7 * duration_ms)
            skip_frames = min(n_frames, int(offset_ms / 1000 * framerate))
        wf.readframes(skip_frames)
        frames = wf.readframes(n_frames - skip_frames)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def transcribe_runpod_batch(batch: list[tuple[Path, float]], api_key: str,
                             endpoint: str = DEFAULT_RUNPOD_ENDPOINT) -> None:
    """Transcribe a batch of ATC clips (normally RUNPOD_BATCH_SIZE of them)
    in a single call to the hosted ATC-tuned Whisper endpoint, writing a
    .txt transcript next to each WAV.

    Runs in a worker thread submitted from the main SDR read loop, same as
    transcribe_file -- must never be called inline there. Bundling clips
    into one request (rather than one call per clip) amortizes the
    endpoint's cold-start/queue delay across several transmissions.
    """
    audios = []
    by_name = {}
    for path, skip_ms in batch:
        name = path.stem
        by_name[name] = path
        try:
            data = _wav_base64(path, skip_ms)
        except (wave.Error, EOFError, OSError) as exc:
            print(f"  runpod: failed to read {path.name}: {exc}", file=sys.stderr)
            continue
        audios.append({"data": data, "name": name})
    if not audios:
        return

    payload = json.dumps({"input": {"audios_base64": audios}}).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=payload, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    names = ", ".join(sorted(by_name))
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"  runpod batch transcription failed ({names}): {exc}", file=sys.stderr)
        return

    output = result.get("output", {})
    for item in output.get("results", []):
        path = by_name.get(item.get("source"))
        if path is None:
            continue
        transcript = (item.get("text") or "").strip()
        path.with_suffix(".txt").write_text(transcript + "\n")
        preview = transcript if len(transcript) <= 100 else transcript[:97] + "..."
        print(f"  transcript ({path.name}): {preview or '(empty)'}", file=sys.stderr)
    for failure in output.get("failed", []):
        print(f"  runpod: transcription failed for {failure}", file=sys.stderr)


class RunpodAtcBatcher:
    """Accumulates ATC clips and flushes them to the RunPod endpoint in
    groups of `batch_size`, so the batch endpoint is used efficiently
    instead of one call per clip. Only ever touched from the main SDR read
    loop (never concurrently), so it needs no locking of its own."""

    def __init__(self, executor: ThreadPoolExecutor, api_key: str,
                 endpoint: str = DEFAULT_RUNPOD_ENDPOINT, batch_size: int = RUNPOD_BATCH_SIZE):
        self._executor = executor
        self._api_key = api_key
        self._endpoint = endpoint
        self._batch_size = max(1, batch_size)
        self._pending: list[tuple[Path, float]] = []

    def add(self, path: Path, skip_ms: float = 0.0) -> None:
        self._pending.append((path, skip_ms))
        if len(self._pending) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        self._executor.submit(transcribe_runpod_batch, batch, self._api_key, self._endpoint)


class QualityLogger:
    """Appends one JSON line per squelch event -- an idle noise-floor sample,
    or a completed transmission's peak/mean signal level -- to a shared log
    file, keyed by channel (preset key/prefix) so per-channel idle sampling
    is throttled independently in scan.py's multi-channel case.

    This is the raw data for tracking receiver+antenna quality over time
    (SNR trend on a given channel, noise-floor drift across antenna/gain
    changes, day/night variation) without needing a transmission to compare
    against -- the complementary check is flight_lookup.py's ADS-B distance
    lookup, which ties a specific recording to a verified slant range.
    """

    def __init__(self, path: Path, idle_interval: float = 60.0):
        self.path = path
        self.idle_interval = idle_interval
        self._last_idle: dict[str, float] = {}

    def _write(self, record: dict) -> None:
        record["ts"] = datetime.now().astimezone().isoformat()
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"quality log write failed: {exc}", file=sys.stderr)

    def log_tx(self, key: str, prefix: str, freq_hz: float, mode: str,
               noise_floor: float, peak_power: float, mean_power: float,
               duration_s: float, wav_name: str) -> None:
        snr_db = (10 * math.log10(peak_power / noise_floor)
                  if noise_floor > 0 and peak_power > 0 else None)
        self._write({
            "event": "tx", "prefix": prefix, "freq_hz": freq_hz, "mode": mode,
            "noise_floor": noise_floor, "peak_power": peak_power,
            "mean_power": mean_power, "snr_db": snr_db,
            "duration_s": duration_s, "wav": wav_name,
        })

    def maybe_log_idle(self, key: str, prefix: str, freq_hz: float, mode: str,
                        noise_floor: float) -> None:
        now = time.monotonic()
        if now - self._last_idle.get(key, -self.idle_interval) < self.idle_interval:
            return
        self._last_idle[key] = now
        self._write({
            "event": "idle", "prefix": prefix, "freq_hz": freq_hz, "mode": mode,
            "noise_floor": noise_floor,
        })


def listen(freq_hz, mode, out_dir, prefix, gain_db, open_ratio, close_ratio,
           hang_time, min_duration, pre_roll, duration,
           transcribe, whisper_bin, whisper_model, whisper_threads,
           category=None, runpod_api_key=None, runpod_endpoint=DEFAULT_RUNPOD_ENDPOINT,
           runpod_batch_size=RUNPOD_BATCH_SIZE,
           quality_log=True, quality_log_path=None, quality_log_interval=60.0, antenna=None,
           listen_audio=False, listen_device=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    live = LivePlayback(AUDIO_RATE, listen_device) if listen_audio else None

    quality_logger = None
    if quality_log:
        qpath = Path(quality_log_path) if quality_log_path else out_dir / "rf_quality.jsonl"
        quality_logger = QualityLogger(qpath, quality_log_interval)

    executor = None
    runpod_batcher = None
    whisper_available = Path(whisper_bin).exists() and Path(whisper_model).exists()
    if transcribe:
        executor = ThreadPoolExecutor(max_workers=2)
        if category is not None and category.startswith("atc") and runpod_api_key:
            runpod_batcher = RunpodAtcBatcher(executor, runpod_api_key, runpod_endpoint,
                                               runpod_batch_size)
        if runpod_batcher is None and not whisper_available:
            print(f"Transcription disabled: whisper-cli or model not found "
                  f"({whisper_bin}, {whisper_model})", file=sys.stderr)

    def maybe_transcribe(path: Path, skip_ms: float = 0.0) -> None:
        if runpod_batcher is not None:
            runpod_batcher.add(path, skip_ms)
        elif executor is not None and whisper_available:
            executor.submit(transcribe_file, path, whisper_bin, whisper_model,
                             whisper_threads, skip_ms)

    def maybe_log_tx(path: Path, floor: float, peak: float, power_sum: float,
                      power_n: int, duration_s: float) -> None:
        if quality_logger is not None:
            quality_logger.log_tx(prefix, prefix, freq_hz, mode, floor, peak,
                                   power_sum / power_n if power_n else peak,
                                   duration_s, path.name)

    sdr = open_sdr(gain_db, antenna=antenna)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    lp_b, lp_a = butter(4, 4_000 / (SAMPLE_RATE / 2), btype="low")
    lp_zi_shape = max(len(lp_a), len(lp_b)) - 1
    lp_zi = np.zeros(lp_zi_shape, dtype=np.float64)
    dc_blocker = DcBlocker()
    fm_disc = FmDiscriminator()

    hang_chunks = max(1, int(round(hang_time / CHUNK_SECONDS)))
    pre_roll_chunks = max(0, int(round(pre_roll / CHUNK_SECONDS)))
    min_samples = int(min_duration * AUDIO_RATE)
    open_chunks_needed = 2  # ~200ms debounce before we commit to "signal present"

    open_counter = 0
    close_counter = 0
    recording = False
    tx_buffer = []
    tx_start_time = None
    tx_preroll_ms = 0.0
    tx_peak_power = 0.0
    tx_power_sum = 0.0
    tx_power_n = 0
    pre_roll_buf = deque(maxlen=pre_roll_chunks)

    buff = np.empty(CHUNK_SAMPLES, np.complex64)
    n_written = 0
    from scipy.signal import lfilter as _lfilter

    global _running
    signal.signal(signal.SIGINT, _handle_sigint)

    # Warm up on a few chunks and seed the noise floor from the *minimum*
    # observed power, so starting mid-transmission doesn't poison the floor
    # with an inflated value (a single first-chunk sample would be fragile).
    calib_chunks = max(1, int(round(1.0 / CHUNK_SECONDS)))
    calib_powers = []
    while len(calib_powers) < calib_chunks and _running:
        sr = sdr.readStream(rx, [buff], CHUNK_SAMPLES, timeoutUs=1_000_000)
        if sr.ret > 0:
            calib_powers.append(float(np.mean(np.abs(buff[: sr.ret]))))
        elif sr.ret < 0:
            print(f"readStream error during calibration: {sr.ret}", file=sys.stderr)
    noise_floor = min(calib_powers) if calib_powers else 1e-6

    listen_note = f", live audio -> {listen_device or 'default sink'}" if live is not None else ""
    print(f"Listening on {freq_hz / 1e6:.3f} MHz ({mode.upper()}), writing transmissions "
          f"to {out_dir}/{listen_note} -- Ctrl+C to stop", file=sys.stderr)

    elapsed = 0.0
    try:
        while _running and (duration is None or elapsed < duration):
            sr = sdr.readStream(rx, [buff], CHUNK_SAMPLES, timeoutUs=1_000_000)
            if sr.ret == 0:
                continue
            if sr.ret < 0:
                print(f"readStream error: {sr.ret}", file=sys.stderr)
                continue
            iq = buff[: sr.ret]
            elapsed += sr.ret / SAMPLE_RATE

            envelope = np.abs(iq).astype(np.float64)
            chunk_power = float(np.mean(envelope))

            if mode == "am":
                filtered, lp_zi = _lfilter(lp_b, lp_a, envelope, zi=lp_zi)
                audio_chunk = dc_blocker.apply(filtered[::DECIMATION].astype(np.float32))
            else:  # fm
                demod = fm_disc.apply(iq).astype(np.float64)
                filtered, lp_zi = _lfilter(lp_b, lp_a, demod, zi=lp_zi)
                audio_chunk = filtered[::DECIMATION].astype(np.float32)

            if live is not None:
                live.write(audio_chunk)

            if not recording:
                noise_floor = 0.98 * noise_floor + 0.02 * chunk_power
                if quality_logger is not None:
                    quality_logger.maybe_log_idle(prefix, prefix, freq_hz, mode, noise_floor)

            if not recording:
                pre_roll_buf.append(audio_chunk)
                if chunk_power > noise_floor * open_ratio:
                    open_counter += 1
                else:
                    open_counter = 0
                if open_counter >= open_chunks_needed:
                    recording = True
                    tx_start_time = datetime.now()
                    tx_buffer = list(pre_roll_buf)
                    tx_preroll_ms = len(tx_buffer) * CHUNK_SECONDS * 1000
                    pre_roll_buf.clear()
                    open_counter = 0
                    close_counter = 0
                    tx_peak_power = chunk_power
                    tx_power_sum = 0.0
                    tx_power_n = 0
            else:
                tx_buffer.append(audio_chunk)
                tx_peak_power = max(tx_peak_power, chunk_power)
                tx_power_sum += chunk_power
                tx_power_n += 1
                if chunk_power < noise_floor * close_ratio:
                    close_counter += 1
                else:
                    close_counter = 0
                if close_counter >= hang_chunks:
                    audio = np.concatenate(tx_buffer)
                    if len(audio) >= min_samples:
                        path = write_wav(out_dir, prefix, tx_start_time, audio)
                        n_written += 1
                        print(f"[{tx_start_time:%H:%M:%S}] wrote {path.name} "
                              f"({len(audio) / AUDIO_RATE:.1f}s)", file=sys.stderr)
                        maybe_transcribe(path, tx_preroll_ms)
                        maybe_log_tx(path, noise_floor, tx_peak_power, tx_power_sum,
                                     tx_power_n, len(audio) / AUDIO_RATE)
                    recording = False
                    tx_buffer = []
                    close_counter = 0
    finally:
        if recording and tx_buffer:
            audio = np.concatenate(tx_buffer)
            if len(audio) >= min_samples:
                path = write_wav(out_dir, prefix, tx_start_time, audio)
                n_written += 1
                print(f"[{tx_start_time:%H:%M:%S}] wrote {path.name} "
                      f"({len(audio) / AUDIO_RATE:.1f}s)", file=sys.stderr)
                maybe_transcribe(path, tx_preroll_ms)
                maybe_log_tx(path, noise_floor, tx_peak_power, tx_power_sum,
                             tx_power_n, len(audio) / AUDIO_RATE)
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        if live is not None:
            live.close()
        if runpod_batcher is not None:
            runpod_batcher.flush()
        if executor is not None:
            print("Waiting for pending transcriptions to finish...", file=sys.stderr)
            executor.shutdown(wait=True)
        print(f"Stopped. {n_written} transmission(s) written to {out_dir}/", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freq", type=float, default=DEFAULT_FREQ_HZ,
                         help=f"frequency in Hz (default: {DEFAULT_FREQ_HZ / 1e6:.3f} MHz, "
                              "EBAW Tower)")
    parser.add_argument("--mode", choices=["am", "fm"], default="am",
                         help="demodulation mode: am (airband/broadcast AM) or "
                              "fm (narrowband FM, e.g. PMR446) (default: am)")
    parser.add_argument("--out-dir", default="output", help="output directory for WAV files")
    parser.add_argument("--prefix", default="REC", help="filename prefix")
    parser.add_argument("--gain", type=float, default=40.0,
                         help="manual RF gain in dB, 0-66 (default: 40). AGC is NOT used by "
                              "default because it renormalizes power and defeats the "
                              "amplitude-based squelch; pass --gain=-1 to force AGC instead.")
    parser.add_argument("--antenna", choices=["A", "B", "C"], default=None,
                         help="RSPdx antenna input to use (default: device default, Antenna A)")
    parser.add_argument("--open-ratio", type=float, default=4.0,
                         help="signal/noise power ratio to open squelch")
    parser.add_argument("--close-ratio", type=float, default=2.0,
                         help="signal/noise power ratio to close squelch (hysteresis)")
    parser.add_argument("--hang-time", type=float, default=1.0,
                         help="seconds below threshold before closing squelch")
    parser.add_argument("--min-duration", type=float, default=0.4,
                         help="discard transmissions shorter than this (seconds)")
    parser.add_argument("--pre-roll", type=float, default=0.3,
                         help="seconds of audio to keep before squelch opens")
    parser.add_argument("--duration", type=float, default=None,
                         help="total seconds to listen (default: run until Ctrl+C)")
    parser.add_argument("--transcribe", dest="transcribe", action="store_true", default=True,
                         help="transcribe each clip with local whisper.cpp (default: on)")
    parser.add_argument("--no-transcribe", dest="transcribe", action="store_false",
                         help="disable transcription")
    parser.add_argument("--whisper-bin", default=str(DEFAULT_WHISPER_BIN),
                         help=f"path to whisper-cli binary (default: {DEFAULT_WHISPER_BIN})")
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER_MODEL),
                         help=f"path to whisper.cpp ggml model (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--whisper-threads", type=int, default=4,
                         help="threads for whisper.cpp to use per transcription")
    parser.add_argument("--category", default=None,
                         help="preset category of this channel (e.g. 'atc'); when 'atc' and "
                              "--runpod-api-key is set, clips are transcribed via the hosted "
                              "ATC Whisper model in batches of --runpod-batch-size instead of "
                              "local whisper.cpp")
    parser.add_argument("--runpod-api-key", default=os.environ.get(RUNPOD_API_KEY_ENV),
                         help=f"RunPod API key for ATC transcription (default: ${RUNPOD_API_KEY_ENV})")
    parser.add_argument("--runpod-endpoint", default=DEFAULT_RUNPOD_ENDPOINT,
                         help="RunPod ATC transcription endpoint URL")
    parser.add_argument("--runpod-batch-size", type=int, default=RUNPOD_BATCH_SIZE,
                         help=f"ATC clips to bundle per RunPod request (default: {RUNPOD_BATCH_SIZE})")
    parser.add_argument("--quality-log", dest="quality_log", action="store_true", default=True,
                         help="log noise-floor/SNR samples to <out-dir>/rf_quality.jsonl, for "
                              "tracking receiver/antenna quality over time (default: on)")
    parser.add_argument("--no-quality-log", dest="quality_log", action="store_false",
                         help="disable quality logging")
    parser.add_argument("--quality-log-path", default=None,
                         help="override the quality log file path "
                              "(default: <out-dir>/rf_quality.jsonl)")
    parser.add_argument("--quality-log-interval", type=float, default=60.0,
                         help="minimum seconds between idle noise-floor samples (default: 60)")
    parser.add_argument("--listen", dest="listen_audio", action="store_true", default=False,
                         help="also play the demodulated audio live to a local sink via paplay, "
                              "so you can listen by ear -- independent of the squelch-gated "
                              "recording, so you'll hear the noise floor between transmissions "
                              "too (default: off)")
    parser.add_argument("--listen-device", default=None,
                         help="paplay --device sink name to play through (default: system "
                              "default sink -- run `pactl get-default-sink` first if unsure, "
                              "e.g. it may currently be set to wsjtx_in for the WSJT-X setup, "
                              "which would silently swallow this)")
    args = parser.parse_args()

    gain = None if args.gain is not None and args.gain < 0 else args.gain
    listen(args.freq, args.mode, Path(args.out_dir), args.prefix, gain,
           args.open_ratio, args.close_ratio, args.hang_time,
           args.min_duration, args.pre_roll, args.duration,
           args.transcribe, args.whisper_bin, args.whisper_model, args.whisper_threads,
           args.category, args.runpod_api_key, args.runpod_endpoint, args.runpod_batch_size,
           args.quality_log, args.quality_log_path, args.quality_log_interval, args.antenna,
           args.listen_audio, args.listen_device)


if __name__ == "__main__":
    main()
