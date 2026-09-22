#!/usr/bin/env python3
"""Sweep a wide slice of spectrum at a time and report frequencies that show
RF activity, instead of dwelling on one predefined channel at a time.

scan.py can only ever find activity on a frequency that's already listed in
presets.py (or passed via --freqs) -- anything else on the dial is invisible
to it, and checking N narrow channels one at a time costs N dwells. This
script takes the opposite approach: it captures a wide (multi-MHz) IQ block,
takes its FFT, and flags whatever frequency bins sit well above the
surrounding noise floor -- checking every channel within that whole window in
a single dwell -- then slides the window to the next slice and repeats across
the requested range.

By default this is a discovery tool, not a recorder: when it finds activity
it logs the frequency/bandwidth/SNR to --out (default
output/spectrum_survey.jsonl) and prints it to stderr, so you can decide what
to add to presets.py or listen to directly with record.py. SNR/noise-floor
figures are relative (ratios against this receiver's own per-bin baseline),
not calibrated dBm.

Pass --record to also demodulate and write audio: since the entire wide
window is already captured as one IQ block, every simultaneously-active
signal within it (aviation *and* marine *and* whatever else happens to be on
at once, as long as they're all within the same step's bandwidth) can be
mixed down, filtered and decimated out of that same block in parallel --
one physical SDR, but no limit on how many channels within view get their own
WAV, other than CPU. Each channel is tracked independently (open on
detection, closed after --hang-time of no further detection) the same way
scan.py's single-channel squelch works, just multiplexed across every hit
found in the wide capture.

Frequency range defaults to the span everything else in this project already
cares about: CB (27 MHz) through amateur 70cm (433-438 MHz).
"""

import argparse
import json
import queue
import signal as signal_module
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

from record import AUDIO_RATE, CHUNK_SECONDS, DcBlocker, FmDiscriminator, open_sdr, write_wav

DEFAULT_START_MHZ = 26.0
DEFAULT_END_MHZ = 470.0
DEFAULT_STEP_MHZ = 8.0    # widest IF filter bandwidth SDRplay RSP devices support
DEFAULT_FFT_SIZE = 4096
DEFAULT_OVERLAP = 0.85     # advance each step by 85% of its width, 15% overlap at the edges
DC_GUARD_BINS = 4          # bins either side of center to ignore (LO/DC leakage)

# Mode-guessing heuristic for --record, matching presets.py's own convention:
# the civil VHF airband is AM, everything else this project cares about is FM.
AVIATION_BAND_HZ = (108e6, 138e6)

_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


class _StopSurvey(Exception):
    """Raised out of read_n() once Ctrl+C has been requested."""


class StreamReader:
    """Services readStream() continuously on its own thread, decoupled from
    however long this process spends per chunk (FFT, hit detection, and --
    with --record -- demodulating every tracked channel). At step_hz=8MHz
    the driver's internal buffer only holds a few chunks' worth of slack; a
    synchronous read-then-process-then-read loop stalls readStream() during
    processing and the buffer overflows (SOAPY_SDR_OVERFLOW / -4) almost
    every pass. Same rationale as dmr.py's dsd-fme stdin writer thread --
    the SDR side has to keep being serviced on a fixed cadence regardless of
    what the consumer is doing, so reading is pulled out onto its own thread
    and consumed from a queue instead of driving readStream() inline."""

    def __init__(self, sdr, rx, chunk_samples: int, timeout_us: int = 1_000_000,
                 queue_depth: int = 256):
        self.sdr = sdr
        self.rx = rx
        self.chunk_samples = chunk_samples
        self.timeout_us = timeout_us
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_depth)
        self._leftover: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        buff = np.empty(self.chunk_samples, np.complex64)
        while not self._stop.is_set():
            sr = self.sdr.readStream(self.rx, [buff], self.chunk_samples, timeoutUs=self.timeout_us)
            if sr.ret > 0:
                try:
                    self._queue.put_nowait(buff[:sr.ret].copy())
                except queue.Full:
                    # Consumer has fallen behind -- drop this chunk rather than block here,
                    # since blocking would stop readStream() from being serviced and bring
                    # back the exact driver-side overflow this thread exists to avoid.
                    pass
            elif sr.ret < 0 and not self._stop.is_set():
                print(f"readStream error: {sr.ret}", file=sys.stderr)

    def flush(self) -> None:
        """Discard queued and leftover samples -- call right after retuning
        so stale pre-tune data doesn't leak into the next step's capture."""
        self._leftover = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def read_n(self, n: int) -> np.ndarray:
        parts = []
        got = 0
        if self._leftover is not None:
            take = min(len(self._leftover), n)
            parts.append(self._leftover[:take])
            got += take
            self._leftover = self._leftover[take:] if take < len(self._leftover) else None
        while got < n:
            if not _running:
                raise _StopSurvey()
            try:
                chunk = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            take = min(len(chunk), n - got)
            parts.append(chunk[:take])
            got += take
            if take < len(chunk):
                self._leftover = chunk[take:]
        return np.concatenate(parts)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class SurveyLogger:
    """Appends one JSON line per detected hit (and, with --record, one per
    finished recording) to a shared log file. Hit events are throttled per
    frequency so a continuously-busy signal (a nearby broadcast, a beacon)
    doesn't flood the log every sweep pass; recording events aren't throttled
    since one is only written when a channel's clip actually closes."""

    def __init__(self, path: Path, relog_interval: float = 10.0):
        self.path = path
        self.relog_interval = relog_interval
        self._last_seen: dict[int, float] = {}

    def _write(self, record: dict) -> None:
        record["ts"] = datetime.now().astimezone().isoformat()
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"survey log write failed: {exc}", file=sys.stderr)

    def maybe_log(self, freq_hz: float, bandwidth_hz: float, snr_db: float,
                  noise_floor_db: float) -> bool:
        bucket = int(round(freq_hz / 1000))  # 1 kHz buckets so near-identical hits collapse
        now = time.monotonic()
        if now - self._last_seen.get(bucket, -self.relog_interval) < self.relog_interval:
            return False
        self._last_seen[bucket] = now
        self._write({
            "event": "hit", "freq_hz": freq_hz, "bandwidth_hz": bandwidth_hz,
            "snr_db": snr_db, "noise_floor_db": noise_floor_db,
        })
        return True

    def log_recording(self, freq_hz: float, mode: str, duration_s: float, wav_name: str) -> None:
        self._write({
            "event": "recording", "freq_hz": freq_hz, "mode": mode,
            "duration_s": duration_s, "wav": wav_name,
        })


DEFAULT_WATERFALL_BINS = 480
DEFAULT_WATERFALL_INTERVAL = 0.5  # min seconds between waterfall frame writes


class WaterfallWriter:
    """Maintains one composite power spectrum spanning the whole start_hz..end_hz
    sweep range, at a fixed display resolution, and periodically dumps it to a
    JSON file the webserver can serve to the browser for a live waterfall.

    Each step only ever refreshes its own slice of the composite array, so the
    file always shows the full range with whatever's most recently been swept
    -- freshest where the sweep currently is, stale (but not blank) everywhere
    else, the same tradeoff a real spectrum analyzer's max-hold/live trace has
    when a sweep takes longer than one screen refresh."""

    def __init__(self, path: Path, start_hz: float, end_hz: float,
                 n_bins: int = DEFAULT_WATERFALL_BINS,
                 min_interval: float = DEFAULT_WATERFALL_INTERVAL):
        self.path = path
        self.start_hz = start_hz
        self.end_hz = end_hz
        self.n_bins = n_bins
        self.min_interval = min_interval
        self.bin_edges = np.linspace(start_hz, end_hz, n_bins + 1)
        self.power_db = np.full(n_bins, np.nan, dtype=np.float64)
        self._last_write = 0.0

    def update(self, freqs: np.ndarray, power: np.ndarray) -> None:
        power_db = 10 * np.log10(np.maximum(power, 1e-12))
        idx = np.clip(np.searchsorted(self.bin_edges, freqs, side="right") - 1,
                       0, self.n_bins - 1)
        sums = np.bincount(idx, weights=power_db, minlength=self.n_bins)
        counts = np.bincount(idx, minlength=self.n_bins)
        touched = counts > 0
        self.power_db[touched] = sums[touched] / counts[touched]

    def maybe_write(self, active_start_hz: float, active_end_hz: float) -> None:
        now = time.monotonic()
        if now - self._last_write < self.min_interval:
            return
        self._last_write = now
        frame = {
            "ts": datetime.now().astimezone().isoformat(),
            "start_hz": self.start_hz,
            "end_hz": self.end_hz,
            "bin_hz": (self.end_hz - self.start_hz) / self.n_bins,
            "power_db": [None if np.isnan(v) else round(float(v), 1) for v in self.power_db],
            "active_start_hz": active_start_hz,
            "active_end_hz": active_end_hz,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(frame, f)
            tmp.replace(self.path)  # atomic, so the webserver never reads a half-written frame
        except OSError as exc:
            print(f"waterfall write failed: {exc}", file=sys.stderr)


def build_steps(start_hz: float, end_hz: float, step_hz: float, overlap: float) -> list[float]:
    advance = step_hz * overlap
    steps = []
    center = start_hz + step_hz / 2
    while center - step_hz / 2 < end_hz:
        steps.append(center)
        center += advance
    return steps


def power_spectrum(iq: np.ndarray, fft_size: int) -> np.ndarray | None:
    """Average periodogram of iq (linear power, Hann-windowed, FFT-shifted so
    index 0 is the most negative frequency), one estimate per fft_size block."""
    n = len(iq) - (len(iq) % fft_size)
    if n <= 0:
        return None
    blocks = iq[:n].reshape(-1, fft_size)
    window = np.hanning(fft_size).astype(np.float32)
    win_power = np.mean(window ** 2)
    spectrum = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    return np.mean(np.abs(spectrum) ** 2, axis=0) / (fft_size * win_power)


def contiguous_runs(mask: np.ndarray) -> np.ndarray:
    """(start, stop) index pairs (stop exclusive) of contiguous True runs in a 1-D bool array."""
    d = np.diff(mask.astype(np.int8))
    idx = np.flatnonzero(d) + 1
    if mask[0]:
        idx = np.r_[0, idx]
    if mask[-1]:
        idx = np.r_[idx, mask.size]
    return idx.reshape(-1, 2)


def detect_hits(power, noise_floor_bins, freqs, bin_hz, open_ratio, min_bins, max_bins, dc_guard):
    """Compare one power spectrum against its per-bin noise floor and return
    (hits, active_mask, power_with_dc_guard_applied). Each hit is a
    (centroid_hz, bandwidth_hz, peak_power, local_noise_floor) tuple. Pure
    detection -- no logging -- so both the log-only and --record code paths
    share one definition of "what counts as activity".

    min_bins/max_bins bound the plausible width of a real narrowband voice
    channel: too few bins is usually a single noisy FFT spike, too many is
    usually a broadband noise burst (a nearby switching supply, WiFi, an
    overloaded front end) rather than one modulated carrier."""
    power = power.copy()
    power[dc_guard] = noise_floor_bins[dc_guard]
    active = power > noise_floor_bins * open_ratio
    hits = []
    for lo, hi in contiguous_runs(active):
        if not (min_bins <= hi - lo <= max_bins):
            continue
        seg_power = power[lo:hi]
        seg_freqs = freqs[lo:hi]
        peak = float(np.max(seg_power))
        centroid = float(np.sum(seg_freqs * seg_power) / np.sum(seg_power))
        bandwidth = float((hi - lo) * bin_hz)
        local_floor = float(np.mean(noise_floor_bins[lo:hi]))
        hits.append((centroid, bandwidth, peak, local_floor))
    return hits, active, power


def guess_mode(freq_hz: float) -> str:
    lo, hi = AVIATION_BAND_HZ
    return "am" if lo <= freq_hz < hi else "fm"


VOICE_BAND_HZ = (300.0, 3000.0)


def spectral_flatness(audio: np.ndarray, audio_rate: int, band=VOICE_BAND_HZ) -> float:
    """Wiener entropy (geometric mean / arithmetic mean of the power spectrum)
    of a clip's voice-band content, in [0, 1]. Pure noise has a flat spectrum
    (flatness near 1); voice's formant structure concentrates energy at a few
    frequencies (flatness much lower) -- this is what tells an FM squelch
    triggering on RF power alone apart from an actual demodulated static
    burst, since a real modulated carrier and receiver noise both cross an
    RF-power threshold, but only one of them sounds like anything once
    demodulated."""
    if len(audio) < 64:
        return 1.0  # too short to judge -- treat as not-voice
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64) * np.hanning(len(audio)))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1 / audio_rate)
    band_power = spectrum[(freqs >= band[0]) & (freqs <= band[1])]
    band_power = band_power[band_power > 0]
    if len(band_power) < 8:
        return 1.0
    geo_mean = np.exp(np.mean(np.log(band_power)))
    arith_mean = np.mean(band_power)
    return float(geo_mean / arith_mean) if arith_mean > 0 else 1.0


def envelope_modulation_db(audio: np.ndarray, audio_rate: int, win_s: float = 0.1) -> float:
    """Dynamic range, in dB, of the clip's short-term RMS envelope, after
    trimming the loudest ~5% of windows so a squelch-open/close pop at the
    clip's edges doesn't get counted as part of the signal. Speech's
    syllable/pause structure swings this over a wide range (typically
    20-35 dB on real ATC traffic); a continuous carrier or tone holds a
    near-constant level, closer to 10-15 dB.

    This catches what spectral_flatness() structurally can't: a steady
    single-frequency tone/hum (an idling wireless-mic pilot carrier, a
    telemetry beacon) has energy *even more* concentrated than voice's
    formants, so it passes any flatness threshold as convincingly as real
    speech does -- flatness alone can't tell a held note apart from talking.
    Envelope variability can, since a held tone doesn't have pauses."""
    win = max(1, int(audio_rate * win_s))
    n_windows = len(audio) // win
    if n_windows < 4:
        return 0.0  # too short to judge modulation -- treat as not-voice
    windows = audio[: n_windows * win].reshape(n_windows, win).astype(np.float64)
    env = np.sqrt(np.mean(windows ** 2, axis=1))
    env_sorted = np.sort(env)
    trim = max(1, int(len(env_sorted) * 0.05))
    core = env_sorted[:-trim] if trim < len(env_sorted) else env_sorted
    # +1 here would silently assume audio amplitude is near 1 -- but this runs on the
    # raw demodulated signal (straight off the SDR's IQ magnitude), which for a real
    # received signal sits in the 1e-4..1e-1 range, not near 1. Adding 1 to values
    # that small swamps the actual max/min difference and crushes every ratio toward
    # 1 (0 dB) regardless of the clip's real dynamics -- a fixed floor epsilon avoids
    # log(0) without doing that.
    eps = 1e-9
    return float(20 * np.log10((core.max() + eps) / (core.min() + eps)))


class ChannelDemod:
    """Extracts one narrowband channel's audio from successive wideband IQ
    chunks: mixes it to baseband with a phase-continuous NCO (so there's no
    click at chunk boundaries), low-pass filters + decimates to record.py's
    AUDIO_RATE, then demodulates with the mode-appropriate discriminator --
    the same chain record.py runs per-SDR-retune, just fed from a shared
    wideband capture so several channels can be pulled from one capture."""

    LOWPASS_HZ = 4_000.0

    def __init__(self, offset_hz: float, mode: str, sample_rate: float, audio_rate: int):
        self.offset_hz = offset_hz
        self.mode = mode
        self.sample_rate = sample_rate
        self.decimation = max(1, int(sample_rate // audio_rate))
        self.lp_b, self.lp_a = butter(4, self.LOWPASS_HZ / (sample_rate / 2), btype="low")
        self.lp_zi = np.zeros(max(len(self.lp_a), len(self.lp_b)) - 1, dtype=np.float64)
        self.dc_blocker = DcBlocker() if mode == "am" else None
        self.fm_disc = FmDiscriminator() if mode == "fm" else None
        self._phase = 0.0

    def process(self, iq: np.ndarray) -> np.ndarray:
        n = len(iq)
        t = np.arange(n) / self.sample_rate
        mixer = np.exp(-1j * (2 * np.pi * self.offset_hz * t + self._phase)).astype(np.complex64)
        self._phase = (self._phase + 2 * np.pi * self.offset_hz * n / self.sample_rate) % (2 * np.pi)
        baseband = iq * mixer
        if self.mode == "am":
            envelope = np.abs(baseband).astype(np.float64)
            filtered, self.lp_zi = lfilter(self.lp_b, self.lp_a, envelope, zi=self.lp_zi)
            return self.dc_blocker.apply(filtered[::self.decimation].astype(np.float32))
        demod = self.fm_disc.apply(baseband).astype(np.float64)
        filtered, self.lp_zi = lfilter(self.lp_b, self.lp_a, demod, zi=self.lp_zi)
        return filtered[::self.decimation].astype(np.float32)


class PendingHit:
    """A hit seen in one chunk but not yet confirmed persistent enough to be
    worth opening a recording for -- see visit_step_record's open_chunks."""
    __slots__ = ("freq_hz", "count")

    def __init__(self, freq_hz: float):
        self.freq_hz = freq_hz
        self.count = 1


class TrackedChannel:
    __slots__ = ("freq_hz", "mode", "demod", "buffer", "close_counter", "opened_at")

    def __init__(self, freq_hz: float, mode: str, demod: ChannelDemod, opened_at: datetime):
        self.freq_hz = freq_hz
        self.mode = mode
        self.demod = demod
        self.buffer: list[np.ndarray] = []
        self.close_counter = 0
        self.opened_at = opened_at


def survey(start_hz, end_hz, step_hz, fft_size, overlap, dwell, gain_db,
           open_ratio, relog_interval, out_path, min_bandwidth_hz, max_bandwidth_hz,
           record_audio=False, out_dir=None, hang_time=1.0, min_duration=0.4, pre_roll=0.3,
           voice_check=True, flatness_threshold=0.3, min_modulation_db=15.0, open_chunks=2,
           waterfall_out=None, antenna=None):
    steps = build_steps(start_hz, end_hz, step_hz, overlap)
    if not steps:
        raise ValueError("empty sweep range")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = SurveyLogger(out_path, relog_interval)
    waterfall_path = Path(waterfall_out) if waterfall_out else out_path.parent / "spectrum_waterfall.json"
    waterfall_path.parent.mkdir(parents=True, exist_ok=True)
    waterfall = WaterfallWriter(waterfall_path, start_hz, end_hz)
    if record_audio:
        out_dir = Path(out_dir) if out_dir else out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

    sdr = open_sdr(gain_db, sample_rate=step_hz, bandwidth_hz=step_hz, antenna=antenna)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    dwell_samples = max(fft_size, int(step_hz * dwell))
    chunk_samples = max(fft_size, int(step_hz * CHUNK_SECONDS))
    bin_hz = step_hz / fft_size
    dc_guard = slice(fft_size // 2 - DC_GUARD_BINS, fft_size // 2 + DC_GUARD_BINS + 1)
    noise_floor = [np.full(fft_size, 1e-9, dtype=np.float64) for _ in steps]
    hang_chunks = max(1, int(round(hang_time / CHUNK_SECONDS)))
    preroll_chunks = max(0, int(round(pre_roll / CHUNK_SECONDS)))
    open_chunks = max(1, open_chunks)
    min_samples = int(min_duration * AUDIO_RATE)
    match_tolerance_hz = max(bin_hz * 4, 2_000.0)
    min_bins = max(1, round(min_bandwidth_hz / bin_hz))
    max_bins = max(min_bins, round(max_bandwidth_hz / bin_hz))

    reader = StreamReader(sdr, rx, chunk_samples)
    read_n = reader.read_n

    global _running
    signal_module.signal(signal_module.SIGINT, _handle_sigint)

    mode_word = "survey+record" if record_audio else "survey"
    print(f"Sweeping {start_hz / 1e6:.3f}-{end_hz / 1e6:.3f} MHz in {len(steps)} step(s) of "
          f"{step_hz / 1e6:.2f} MHz ({mode_word}) -- Ctrl+C to stop", file=sys.stderr)
    print(f"Logging hits to {out_path}", file=sys.stderr)
    if record_audio:
        print(f"Writing recordings to {out_dir}/", file=sys.stderr)
        if voice_check:
            print(f"Voice check on: keeps clips with flatness <= {flatness_threshold:.2f} "
                  f"and envelope modulation range >= {min_modulation_db:.1f} dB "
                  f"(min duration {min_duration:.2f}s) -- see per-clip debug lines below "
                  "for the actual measured values", file=sys.stderr)
        else:
            print("Voice check off: every clip that passes the RF squelch gets written",
                  file=sys.stderr)

    n_hits = 0
    n_recordings = 0
    n_discarded = 0

    def log_new_hit(centroid, bandwidth, peak, local_floor) -> None:
        nonlocal n_hits
        snr_db = 10 * np.log10(peak / local_floor) if local_floor > 0 else float("inf")
        floor_db = 10 * np.log10(local_floor) if local_floor > 0 else float("-inf")
        if logger.maybe_log(centroid, bandwidth, snr_db, floor_db):
            n_hits += 1
            print(f"[{datetime.now():%H:%M:%S}] activity at {centroid / 1e6:.4f} MHz "
                  f"(~{bandwidth / 1e3:.1f} kHz wide, {snr_db:.1f} dB over noise)", file=sys.stderr)

    def finalize_channel(ch: TrackedChannel) -> None:
        """Always logs both voice-check metrics against their thresholds --
        even the one that didn't cause a rejection -- so the algorithm can be
        eyeballed against real clips instead of only seeing whichever check
        happened to fire first."""
        nonlocal n_recordings, n_discarded
        audio = np.concatenate(ch.buffer) if ch.buffer else np.empty(0, np.float32)
        dur_s = len(audio) / AUDIO_RATE
        if len(audio) < min_samples:
            print(f"[{datetime.now():%H:%M:%S}] discarded {ch.freq_hz / 1e6:.4f} MHz clip "
                  f"({dur_s:.2f}s < {min_duration:.2f}s min-duration -- too short to judge)",
                  file=sys.stderr)
            return

        # Computed unconditionally -- even with voice_check off -- so every
        # written clip still carries what the algorithm *would* have decided,
        # for comparing against your own judgement once you've listened to it.
        flatness = spectral_flatness(audio, AUDIO_RATE)
        mod_range_db = envelope_modulation_db(audio, AUDIO_RATE)
        flat_ok = flatness <= flatness_threshold
        mod_ok = mod_range_db >= min_modulation_db
        would_reject = not (flat_ok and mod_ok)
        metrics = (f"flatness={flatness:.3f} ({'pass' if flat_ok else 'FAIL'}, "
                   f"threshold <= {flatness_threshold:.2f}) "
                   f"mod_range={mod_range_db:.1f}dB ({'pass' if mod_ok else 'FAIL'}, "
                   f"threshold >= {min_modulation_db:.1f})"
                   + ("" if voice_check else " [voice-check off, not enforced]"))
        reject_reason = None
        if voice_check and would_reject:
            reject_reason = ("flat spectrum -- sounds like noise, not voice" if not flat_ok
                              else "steady envelope -- sounds like a tone/carrier, not speech")

        if reject_reason:
            n_discarded += 1
            print(f"[{datetime.now():%H:%M:%S}] discarded {ch.freq_hz / 1e6:.4f} MHz clip "
                  f"({dur_s:.1f}s) -- {metrics} -- {reject_reason}", file=sys.stderr)
            return

        prefix = f"SPEC_{ch.freq_hz / 1e6:.4f}MHZ".replace(".", "_")
        path = write_wav(out_dir, prefix, ch.opened_at, audio)
        n_recordings += 1
        flag = " *** would have been discarded ***" if not voice_check and would_reject else ""
        print(f"[{datetime.now():%H:%M:%S}] wrote {path.name} "
              f"({dur_s:.1f}s, {ch.freq_hz / 1e6:.4f} MHz {ch.mode.upper()}) -- {metrics}{flag}",
              file=sys.stderr)
        logger.log_recording(ch.freq_hz, ch.mode, dur_s, path.name)

    def seed_step(i: int, center: float) -> None:
        sdr.setFrequency(SOAPY_SDR_RX, 0, center)
        reader.flush()  # drop anything the background thread queued pre-retune
        read_n(fft_size)  # let filters/PLL settle after retuning
        power = power_spectrum(read_n(dwell_samples), fft_size)
        if power is not None:
            noise_floor[i] = power
            freqs = center - step_hz / 2 + bin_hz * np.arange(fft_size)
            waterfall.update(freqs, power)
            waterfall.maybe_write(center - step_hz / 2, center + step_hz / 2)

    def visit_step_detect_only(i: int, center: float) -> None:
        sdr.setFrequency(SOAPY_SDR_RX, 0, center)
        reader.flush()
        read_n(fft_size)
        power = power_spectrum(read_n(dwell_samples), fft_size)
        if power is None:
            return
        freqs = center - step_hz / 2 + bin_hz * np.arange(fft_size)
        hits, active, power = detect_hits(power, noise_floor[i], freqs, bin_hz,
                                           open_ratio, min_bins, max_bins, dc_guard)
        for centroid, bandwidth, peak, local_floor in hits:
            log_new_hit(centroid, bandwidth, peak, local_floor)
        idle = ~active
        noise_floor[i][idle] = 0.98 * noise_floor[i][idle] + 0.02 * power[idle]
        waterfall.update(freqs, power)
        waterfall.maybe_write(center - step_hz / 2, center + step_hz / 2)

    def visit_step_record(i: int, center: float) -> None:
        """Keep dwelling on this step -- not moving to the next one -- for as
        long as anything within it is being tracked, demodulating every
        tracked channel from each chunk of the *same* wideband capture. Only
        once every channel has hung out (gone undetected for --hang-time) does
        this return control to the sweep so it can advance.

        A hit isn't promoted to a recording the instant it's seen -- it has to
        show up in --open-chunks consecutive chunks first (as PendingHit).
        Without this, a single noisy FFT spike or a one-shot digital burst
        (an AIS/DSC blip, a spur) opens a full recording exactly like a real
        sustained transmission would, and since it then immediately stops
        matching, what gets written is ~hang-time of near-silence with maybe
        one chunk of anything in it -- a clip that looks like it plays for a
        second and "stops immediately" with nothing in it."""
        sdr.setFrequency(SOAPY_SDR_RX, 0, center)
        reader.flush()
        read_n(fft_size)
        freqs = center - step_hz / 2 + bin_hz * np.arange(fft_size)
        tracked: dict[int, TrackedChannel] = {}
        pending: dict[int, PendingHit] = {}
        history: deque[np.ndarray] = deque(maxlen=preroll_chunks + open_chunks - 1)
        next_id = 0
        next_pending_id = 0

        while True:
            iq = read_n(chunk_samples)
            power = power_spectrum(iq, fft_size)
            hits, active, power = detect_hits(power, noise_floor[i], freqs, bin_hz,
                                               open_ratio, min_bins, max_bins, dc_guard)

            matched_ids = set()
            matched_pending = set()
            for centroid, bandwidth, peak, local_floor in hits:
                match_id = next((cid for cid, ch in tracked.items()
                                  if abs(ch.freq_hz - centroid) <= match_tolerance_hz), None)
                if match_id is not None:
                    matched_ids.add(match_id)
                    tracked[match_id].close_counter = 0
                    continue

                pending_id = next((pid for pid, p in pending.items()
                                    if abs(p.freq_hz - centroid) <= match_tolerance_hz), None)
                if pending_id is None:
                    pending[next_pending_id] = PendingHit(centroid)
                    matched_pending.add(next_pending_id)
                    next_pending_id += 1
                    continue
                matched_pending.add(pending_id)
                pending[pending_id].count += 1
                if pending[pending_id].count < open_chunks:
                    continue

                # Confirmed persistent -- promote to an actual recording,
                # backfilling from every chunk history still has (pre-roll
                # plus the chunks spent confirming this candidate).
                del pending[pending_id]
                mode = guess_mode(centroid)
                demod = ChannelDemod(centroid - center, mode, step_hz, AUDIO_RATE)
                ch = TrackedChannel(centroid, mode, demod, datetime.now())
                for hist_chunk in history:
                    ch.buffer.append(demod.process(hist_chunk))
                match_id = next_id
                tracked[match_id] = ch
                next_id += 1
                matched_ids.add(match_id)
                log_new_hit(centroid, bandwidth, peak, local_floor)

            # An unmatched pending candidate didn't repeat -- drop it rather
            # than granting it hang-time; it was never confirmed as real.
            for pending_id in list(pending.keys()):
                if pending_id not in matched_pending:
                    del pending[pending_id]

            for cid in list(tracked.keys()):
                ch = tracked[cid]
                ch.buffer.append(ch.demod.process(iq))
                if cid not in matched_ids:
                    ch.close_counter += 1
                    if ch.close_counter >= hang_chunks:
                        finalize_channel(ch)
                        del tracked[cid]

            history.append(iq)
            idle = ~active
            noise_floor[i][idle] = 0.98 * noise_floor[i][idle] + 0.02 * power[idle]
            waterfall.update(freqs, power)
            waterfall.maybe_write(center - step_hz / 2, center + step_hz / 2)

            if not tracked and not pending:
                break

    try:
        # Seed each step's per-bin noise floor before detecting anything, so
        # the very first pass doesn't flag its own startup transient as a hit.
        for i, center in enumerate(steps):
            seed_step(i, center)

        visit_step = visit_step_record if record_audio else visit_step_detect_only
        while True:
            for i, center in enumerate(steps):
                visit_step(i, center)
    except _StopSurvey:
        pass
    finally:
        reader.close()
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        summary = f"{n_hits} hit(s) logged"
        if record_audio:
            summary += (f", {n_recordings} recording(s) written to {out_dir}/, "
                        f"{n_discarded} discarded as noise-like")
        print(f"Stopped. {summary} to {out_path}.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-mhz", type=float, default=DEFAULT_START_MHZ,
                         help=f"low edge of the sweep range in MHz (default: {DEFAULT_START_MHZ})")
    parser.add_argument("--end-mhz", type=float, default=DEFAULT_END_MHZ,
                         help=f"high edge of the sweep range in MHz (default: {DEFAULT_END_MHZ})")
    parser.add_argument("--step-mhz", type=float, default=DEFAULT_STEP_MHZ,
                         help="width of each wide capture in MHz, i.e. how much spectrum is "
                              f"checked at once per retune (default: {DEFAULT_STEP_MHZ}); "
                              "SDRplay snaps this to its nearest supported IF filter bandwidth "
                              "(200k/300k/600k/1536k/5000k/6000k/7000k/8000k)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                         help="fraction of each step to advance by, <1 leaves overlap between "
                              f"steps to cover filter roll-off at the edges (default: {DEFAULT_OVERLAP})")
    parser.add_argument("--fft-size", type=int, default=DEFAULT_FFT_SIZE,
                         help="FFT size, sets frequency resolution to step-mhz/fft-size "
                              f"(default: {DEFAULT_FFT_SIZE})")
    parser.add_argument("--dwell", type=float, default=0.2,
                         help="seconds of IQ captured per step per sweep pass when not recording "
                              "(default: 0.2)")
    parser.add_argument("--gain", type=float, default=40.0,
                         help="manual RF gain in dB, 0-66 (default: 40); pass --gain=-1 for AGC")
    parser.add_argument("--antenna", choices=["A", "B", "C"], default=None,
                         help="RSPdx antenna input to use (default: device default, Antenna A)")
    parser.add_argument("--open-ratio", type=float, default=8.0,
                         help="power/noise-floor ratio for a bin to count as active (default: 8.0); "
                              "raise this if a noisy RF environment is triggering on noise variance")
    parser.add_argument("--min-bandwidth-hz", type=float, default=6_000.0,
                         help="minimum width for a hit to count as a channel rather than a single "
                              "noisy FFT spike (default: 6000)")
    parser.add_argument("--max-bandwidth-hz", type=float, default=30_000.0,
                         help="maximum width for a hit to still count as one narrowband voice "
                              "channel rather than a broadband noise burst (default: 30000)")
    parser.add_argument("--relog-interval", type=float, default=10.0,
                         help="minimum seconds before the same ~1 kHz bucket is logged again "
                              "(default: 10)")
    parser.add_argument("--out", default="output/spectrum_survey.jsonl",
                         help="JSONL log path for detected hits (default: output/spectrum_survey.jsonl)")
    parser.add_argument("--waterfall-out", default=None,
                         help="JSON path for the live waterfall frame, overwritten roughly every "
                              f"{DEFAULT_WATERFALL_INTERVAL}s (default: alongside --out, "
                              "spectrum_waterfall.json)")
    parser.add_argument("--record", action="store_true", default=False,
                         help="also demodulate and record audio for every detected channel, "
                              "extracting each simultaneously-active signal from the same wide "
                              "capture (multi-channel); default is log-only")
    parser.add_argument("--out-dir", default="output",
                         help="output directory for WAV files when --record is set (default: output)")
    parser.add_argument("--hang-time", type=float, default=1.0,
                         help="seconds a channel must go undetected before its clip is closed "
                              "and written (default: 1.0, only used with --record)")
    parser.add_argument("--open-chunks", type=int, default=2,
                         help="consecutive 0.1s chunks a hit must be detected in before it's "
                              "promoted to an actual recording (default: 2, i.e. 0.2s); filters "
                              "single-instant spikes/spurs/short digital bursts that would "
                              "otherwise open a full hang-time recording containing nothing, "
                              "only used with --record")
    parser.add_argument("--min-duration", type=float, default=0.4,
                         help="discard recorded clips shorter than this (default: 0.4, only used "
                              "with --record)")
    parser.add_argument("--pre-roll", type=float, default=0.3,
                         help="seconds of audio to backfill before a channel's detection, from "
                              "the same wideband capture history (default: 0.3, only used with "
                              "--record)")
    parser.add_argument("--voice-check", dest="voice_check", action="store_true", default=True,
                         help="before writing a clip, check that its demodulated audio has "
                              "voice-like (non-flat) spectral structure and discard it otherwise "
                              "-- an RF-power squelch alone can't tell a real weak carrier apart "
                              "from a noise burst that happens to cross the threshold (default: on, "
                              "only used with --record)")
    parser.add_argument("--no-voice-check", dest="voice_check", action="store_false",
                         help="write every clip that passes the RF squelch, skipping the "
                              "audio-domain noise check")
    parser.add_argument("--flatness-threshold", type=float, default=0.3,
                         help="spectral flatness above which a clip is discarded as noise-like, "
                              "0-1 (default: 0.3); lower = stricter (rejects more marginal audio), "
                              "higher = more permissive")
    parser.add_argument("--min-modulation-db", type=float, default=15.0,
                         help="minimum dynamic range, in dB, of a clip's short-term amplitude "
                              "envelope for it to count as speech-like (default: 15.0); a steady "
                              "carrier or tone holds a near-constant level and falls well under "
                              "this, unlike spoken syllables/pauses -- catches steady interferers "
                              "that pass the flatness check since a held tone's spectrum is even "
                              "less flat than voice's, not more, only used with --record")
    args = parser.parse_args()

    gain = None if args.gain is not None and args.gain < 0 else args.gain
    survey(args.start_mhz * 1e6, args.end_mhz * 1e6, args.step_mhz * 1e6,
           args.fft_size, args.overlap, args.dwell, gain, args.open_ratio,
           args.relog_interval, Path(args.out), args.min_bandwidth_hz, args.max_bandwidth_hz,
           record_audio=args.record, out_dir=args.out_dir, hang_time=args.hang_time,
           min_duration=args.min_duration, pre_roll=args.pre_roll,
           voice_check=args.voice_check, flatness_threshold=args.flatness_threshold,
           min_modulation_db=args.min_modulation_db,
           open_chunks=args.open_chunks, waterfall_out=args.waterfall_out, antenna=args.antenna)


if __name__ == "__main__":
    main()
