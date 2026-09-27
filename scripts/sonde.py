#!/usr/bin/env python3
"""Track weather balloons (radiosondes) in the 400-406 MHz meteorological
band on an SDRplay device and log their decoded telemetry -- position,
altitude, climb rate, temperature, humidity, pressure, battery, ...

The actual sonde demodulation/decoding is done by rs1729's decoders
(rs41mod, dfm09mod, m10m20mod, imet4iq) and dft_detect (sonde type
detection), as bundled/built by Project Horus' radiosonde_auto_rx -- see
Install below. auto_rx itself can't drive an SDRplay (it only speaks
rtl_fm / SpyServer / KA9Q), so this script does auto_rx's SDR-side job in
Python instead, the same split dmr.py uses with dsd-fme: tune, find
signals, mix each one down to its own 50 kHz complex baseband channel, and
stream that as raw 16-bit IQ into the external decoder's stdin
(`--IQ 0.0 --lpIQ --dc - 50000 16`).

Since the SDR captures a 2 MHz-wide block, every sonde within that window
gets its own channel (mixer + decimator + decoder process) from the same
capture -- the same multi-channel trick spectrum_scan.py's --record uses.

Each channel goes through:
  detect  -- DETECT_SECONDS of IQ into dft_detect, which reports the sonde
             type and a frequency-offset estimate (skipped with --type).
             No sonde found -> the frequency is ignored for a while (in scan
             mode), since it's most likely a birdie or some other carrier.
  decode  -- IQ into the matching decoder; every JSON telemetry frame it
             prints is enriched (rx time, frequency, SNR, and distance /
             azimuth / elevation from the station) and appended to
             <out-dir>/flights/<serial>.jsonl.
  closed  -- no frame for --lost-timeout seconds (landed / out of range).

With no --freq, the whole band is swept (a few 2 MHz FFT dwells) to find
candidate signals, the SDR is parked on the window containing the strongest
one(s), and new signals appearing inside that window are picked up while
decoding. When every channel has closed, it goes back to sweeping.
<out-dir>/status.json is rewritten about once a second with the current
state, for webserver.py's Sondes tab.

Install (decoders only -- no need to run or configure auto_rx itself):
    cd ~/src
    git clone --depth 1 https://github.com/projecthorus/radiosonde_auto_rx.git
    cd radiosonde_auto_rx/auto_rx && ./build.sh

Usage:
    scripts/sonde.py                             # sweep 400-406 MHz, decode whatever's up
    scripts/sonde.py --freq 402.7e6              # one known frequency
    scripts/sonde.py --freq 402.7e6 --type RS41  # skip type detection

    # offline self-test against a recorded sonde (auto_rx's test samples:
    # http://rfhead.net/sondes/sonde_samples.tar.gz), no SDR needed:
    scripts/sonde.py --replay rs41_96k_float.bin --replay-freq 403.5e6 --out-dir /tmp/sonde
"""

import argparse
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import firwin, upfirdn

import spectrum_scan
from flight_lookup import EBAW_LAT, EBAW_LON, EBAW_ELEV_M
from record import SAMPLE_RATE, CHUNK_SAMPLES, open_sdr
from spectrum_scan import StreamReader, _StopSurvey, contiguous_runs, power_spectrum, spectral_floor

DEFAULT_DECODER_DIR = Path("/home/michel/src/radiosonde_auto_rx/auto_rx")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "sonde"

BAND_START_HZ = 400.0e6
BAND_END_HZ = 406.0e6

# 2,000,000 / (8 * 5) = 50,000 exactly. The decoders take any sample rate
# on the command line; ~10 samples/symbol for the RS41's 4800 baud.
DEC1, DEC2 = 8, 5
CHANNEL_RATE = int(SAMPLE_RATE // (DEC1 * DEC2))

# Only the middle of each 2 MHz capture is used -- the SDRplay's IF filter
# (set to 1.536 MHz below) rolls off towards the edges.
USABLE_HALF_SPAN_HZ = 700e3
DC_GUARD_HZ = 25e3          # keep channels clear of LO leakage at the capture's center
SCAN_FFT_SIZE = 8192         # 244 Hz bins at 2 MSPS
SCAN_DWELL_S = 1.0
SCAN_MIN_WIDTH_HZ = 700.0    # narrower than this is a CW birdie, not a sonde
SCAN_MAX_WIDTH_HZ = 30e3
SCAN_FLOOR_WINDOW_HZ = 150e3
SAME_SIGNAL_HZ = 15e3        # two detections this close are the same transmitter

DETECT_SECONDS = 10.0
FIRST_FRAME_TIMEOUT_S = 45.0  # detected, but decoder produced nothing -> give up on it
DEFAULT_LOST_TIMEOUT_S = 180.0
DEFAULT_IGNORE_S = 900.0      # how long a non-sonde signal is skipped by the sweep
IN_WINDOW_RECHECK_S = 20.0    # how often to look for new signals while parked
STATUS_INTERVAL_S = 1.0

# dft_detect type-name prefixes -> the decoder that handles them (it reports
# e.g. "DFM9" for every DFM06/09/17 variant). Types dft_detect can recognize
# but that aren't listed here (RS92, LMS6, MEISEI, ...) are reported but not
# decoded.
DETECT_TYPE_PREFIXES = [
    ("RS41", "RS41"),
    ("DFM", "DFM"),
    ("M10", "M10"),
    ("M20", "M10"),
    ("IMET4", "IMET4"),
    ("IMET1RS", "IMET4"),
]


def decoder_type_for(detected: str) -> str | None:
    for prefix, sonde_type in DETECT_TYPE_PREFIXES:
        if detected.startswith(prefix):
            return sonde_type
    return None


def position_valid(frame: dict) -> bool:
    """False for frames sent before the sonde's GPS has a fix -- M10s report
    lat 90 / lon 0, others 0/0, all with sats 0."""
    lat, lon = frame.get("lat"), frame.get("lon")
    if lat is None or lon is None or abs(lat) >= 90 or (lat == 0 and lon == 0):
        return False
    return frame.get("sats") != 0 or frame.get("type") == "IMET"  # iMet-4 doesn't report sats


def decoder_cmd(decoder_dir: Path, sonde_type: str) -> list[str]:
    rate = str(CHANNEL_RATE)
    iq_in = ["--IQ", "0.0", "--lpIQ", "--dc", "-", rate, "16"]
    if sonde_type == "RS41":
        return [str(decoder_dir / "rs41mod"), "--ptu2", "--json", "--jsnsubfrm1", *iq_in]
    if sonde_type == "DFM":
        return [str(decoder_dir / "dfm09mod"), "--ecc", "--json", "--dist", "--auto", *iq_in]
    if sonde_type == "M10":  # m10m20mod handles both M10 and M20
        return [str(decoder_dir / "m10m20mod"), "--json", "--ptu", *iq_in]
    if sonde_type == "IMET4":
        return [str(decoder_dir / "imet4iq"), "--json", "--iq", "0.0", "--lpIQ", "--dc", "-", rate, "16"]
    raise ValueError(f"no decoder for sonde type {sonde_type}")


_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False
    spectrum_scan._running = False  # makes StreamReader.read_n() bail out too


# --------------------------------------------------------------------------
# Geometry: where the sonde is as seen from the station.

def look_angles(st_lat, st_lon, st_alt_m, lat, lon, alt_m):
    """(ground distance km, azimuth deg, elevation deg) from the station to
    a point, via ECEF vectors -- elevation needs the real 3-D geometry
    since a sonde 30 km up and 300 km out is still above the horizon."""
    a, e2 = 6378137.0, 6.69437999014e-3

    def ecef(la, lo, h):
        la, lo = math.radians(la), math.radians(lo)
        n = a / math.sqrt(1 - e2 * math.sin(la) ** 2)
        return ((n + h) * math.cos(la) * math.cos(lo),
                (n + h) * math.cos(la) * math.sin(lo),
                (n * (1 - e2) + h) * math.sin(la))

    sx, sy, sz = ecef(st_lat, st_lon, st_alt_m)
    px, py, pz = ecef(lat, lon, alt_m)
    dx, dy, dz = px - sx, py - sy, pz - sz
    la, lo = math.radians(st_lat), math.radians(st_lon)
    east = -math.sin(lo) * dx + math.cos(lo) * dy
    north = -math.sin(la) * math.cos(lo) * dx - math.sin(la) * math.sin(lo) * dy + math.cos(la) * dz
    up = math.cos(la) * math.cos(lo) * dx + math.cos(la) * math.sin(lo) * dy + math.sin(la) * dz
    az = (math.degrees(math.atan2(east, north)) + 360) % 360
    el = math.degrees(math.atan2(up, math.hypot(east, north)))

    dlat, dlon = math.radians(lat - st_lat), math.radians(lon - st_lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(la) * math.cos(math.radians(lat)) * math.sin(dlon / 2) ** 2
    ground_km = 2 * 6371.0 * math.asin(math.sqrt(h))
    return ground_km, az, el


# --------------------------------------------------------------------------
# Channelizer: wideband 2 MSPS -> one 50 kHz complex baseband channel.

class FirDecimator:
    """Stateful polyphase FIR decimator. upfirdn() only computes the
    outputs that survive decimation (so it's ~D times cheaper than filtering
    then slicing), but it's stateless -- carrying the last len(taps)-1 input
    samples over and prepending them gives a seamless stream across chunks.
    Tap counts are chosen as k*D+1 so that the first fully-overlapped output
    lands exactly on a decimation phase."""

    def __init__(self, taps: np.ndarray, factor: int):
        assert (len(taps) - 1) % factor == 0
        self.taps = taps.astype(np.float32)
        self.factor = factor
        self.hist = np.zeros(len(taps) - 1, np.complex64)

    def process(self, x: np.ndarray) -> np.ndarray:
        assert len(x) % self.factor == 0
        xx = np.concatenate([self.hist, x])
        y = upfirdn(self.taps, xx, 1, self.factor)
        first = (len(self.taps) - 1) // self.factor
        self.hist = xx[len(xx) - (len(self.taps) - 1):]
        return y[first:first + len(x) // self.factor].astype(np.complex64)


class Channelizer:
    def __init__(self, offset_hz: float):
        self.offset_hz = offset_hz
        self._phase = 1.0 + 0j
        self._rot = None
        self.stage1 = FirDecimator(firwin(DEC1 * 8 + 1, 60e3, fs=SAMPLE_RATE), DEC1)
        self.stage2 = FirDecimator(firwin(DEC2 * 24 + 1, 18e3, fs=SAMPLE_RATE / DEC1), DEC2)

    def retune(self, offset_hz: float) -> None:
        self.offset_hz = offset_hz
        self._rot = None

    def process(self, iq: np.ndarray) -> np.ndarray:
        n = len(iq)
        if self._rot is None or len(self._rot) != n:
            self._rot = np.exp(-2j * np.pi * self.offset_hz * np.arange(n) / SAMPLE_RATE).astype(np.complex64)
        mixed = iq * (self._rot * np.complex64(self._phase))
        self._phase *= np.exp(-2j * np.pi * self.offset_hz * n / SAMPLE_RATE)
        self._phase /= abs(self._phase)
        return self.stage2.process(self.stage1.process(mixed))


def to_cs16(x: np.ndarray, gain: float) -> bytes:
    out = np.empty(2 * len(x), np.int16)
    out[0::2] = np.clip(x.real * gain, -32767, 32767)
    out[1::2] = np.clip(x.imag * gain, -32767, 32767)
    return out.tobytes()


# --------------------------------------------------------------------------
# External decoder process plumbing.

class PipedProcess:
    """An external decoder fed IQ on stdin from a bounded queue on its own
    writer thread (same reasoning as dmr.py: the SDR read loop must never
    block on a slow consumer), with stdout lines collected on a reader
    thread. `lossless` makes feed() block instead of dropping when the
    queue is full -- only used for --replay, where there is no real-time
    SDR to overflow and dropping would just corrupt the test."""

    def __init__(self, cmd: list[str], lossless: bool = False):
        self.cmd = cmd
        self.lossless = lossless
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, bufsize=0)
        self.lines: "queue.Queue[str]" = queue.Queue()
        self._in: "queue.Queue[bytes | None]" = queue.Queue(maxsize=50)
        self._closed = False
        threading.Thread(target=self._writer, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()

    def _writer(self) -> None:
        while True:
            item = self._in.get()
            if item is None:
                break
            try:
                self.proc.stdin.write(item)
            except (BrokenPipeError, OSError):
                return
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def _reader(self) -> None:
        for raw in self.proc.stdout:
            self.lines.put(raw.decode("utf-8", "replace").strip())

    def feed(self, data: bytes) -> None:
        if self._closed:
            return
        if self.lossless:
            self._in.put(data)
            return
        try:
            self._in.put_nowait(data)
        except queue.Full:
            pass

    def close_input(self) -> None:
        """Graceful EOF once everything already queued has been written."""
        if self._closed:
            return
        self._closed = True
        try:
            self._in.put(None, timeout=2)
        except queue.Full:
            pass  # writer is gone (process died); nothing left to flush

    def drain_lines(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                return out

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        self._closed = True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def parse_dft_detect(line: str):
    """"RS41: 0.9870 , -231.5Hz" -> ("RS41", 0.987, -231.5). A negative
    score means an inverted-polarity match, which the decoders handle."""
    if ":" not in line:
        return None
    name, rest = line.split(":", 1)
    parts = [p.strip() for p in rest.split(",")]
    try:
        score = float(parts[0])
        offset = float(parts[1].replace("Hz", "")) if len(parts) > 1 and parts[1] else 0.0
    except ValueError:
        return None
    return name.strip(), score, offset


# --------------------------------------------------------------------------
# One sonde channel.

class SondeChannel:
    def __init__(self, freq_hz: float, lo_hz: float, now: float, forced_type: str | None,
                 decoder_dir: Path, lossless: bool, fixed: bool):
        self.freq_hz = freq_hz
        self.chan = Channelizer(freq_hz - lo_hz)
        self.decoder_dir = decoder_dir
        self.lossless = lossless
        self.fixed = fixed  # --freq channel: never given up on permanently
        self.opened_at = now
        self.state_since = now
        self.sonde_type = None
        self.detected_as = None
        self.serial = None
        self.frames = 0
        self.last_frame_at = None
        self.last_frame = None
        self.snr_db = None
        self.closed_reason = None
        self._iq_gain = 3000.0
        self._last_line = None
        self._prev_alt = None  # (stream time, alt) for deriving vel_v when the decoder doesn't send it
        self.proc = None
        if forced_type:
            self._start_decode(forced_type, now)
        else:
            self.state = "detect"
            self.proc = PipedProcess([str(decoder_dir / "dft_detect"), "--IQ", "0.0", "--dc",
                                       "-", str(CHANNEL_RATE), "16"], lossless)

    def _start_decode(self, sonde_type: str, now: float) -> None:
        self.state = "decode"
        self.state_since = now
        self.sonde_type = sonde_type
        self.proc = PipedProcess(decoder_cmd(self.decoder_dir, sonde_type), self.lossless)

    def feed(self, iq_wide: np.ndarray) -> None:
        if self.state == "closed":
            return
        x = self.chan.process(iq_wide)
        # Slow AGC so the int16 stream uses a sane part of its range whatever
        # the RF gain -- the decoders are FSK/phase based and don't care about
        # absolute level, only about not clipping or drowning in quantization.
        rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) or 1e-9
        self._iq_gain = 0.9 * self._iq_gain + 0.1 * (4000.0 / rms)
        self.proc.feed(to_cs16(x, self._iq_gain))

    def close(self, reason: str, now: float) -> None:
        if self.proc is not None:
            self.proc.kill()
        self.state = "closed"
        self.state_since = now
        self.closed_reason = reason

    def status(self) -> dict:
        return {
            "freq_hz": self.freq_hz, "state": self.state, "type": self.sonde_type,
            "detected_as": self.detected_as, "serial": self.serial, "frames": self.frames,
            "snr_db": None if self.snr_db is None else round(self.snr_db, 1),
            "last_frame_at": self.last_frame_at,
            "last_frame": self.last_frame,
        }


# --------------------------------------------------------------------------
# IQ sources: the real SDR, or a recorded sonde for offline testing.

class SdrSource:
    def __init__(self, gain_db, antenna):
        self.sdr = open_sdr(gain_db, sample_rate=SAMPLE_RATE, bandwidth_hz=1_536_000, antenna=antenna)
        self.sdr.setFrequency(spectrum_scan.SOAPY_SDR_RX, 0, 403e6)
        self.rx = self.sdr.setupStream(spectrum_scan.SOAPY_SDR_RX, spectrum_scan.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx)
        self.reader = StreamReader(self.sdr, self.rx, CHUNK_SAMPLES)
        self.realtime = True

    def tune(self, lo_hz: float) -> None:
        self.sdr.setFrequency(spectrum_scan.SOAPY_SDR_RX, 0, lo_hz)
        time.sleep(0.05)  # PLL settle
        self.reader.flush()

    def read(self, n: int) -> np.ndarray:
        return self.reader.read_n(n)

    def close(self) -> None:
        self.reader.close()
        self.sdr.deactivateStream(self.rx)
        self.sdr.closeStream(self.rx)


class ReplaySource:
    """Plays back a recorded complex-float32 baseband sonde capture (e.g.
    auto_rx's test samples) as if it were transmitting at `signal_hz`,
    upsampled to SAMPLE_RATE and buried in noise at `snr_db` -- exercises
    the sweep, detection, channelizer and decoders end to end without
    needing a sonde overhead. Linear interpolation is fine for the upsample:
    its spectral images of a <=20 kHz-wide signal land >=75 kHz away and
    are removed by the channelizer's own filters."""

    def __init__(self, path: Path, file_rate: float, signal_hz: float, snr_db: float):
        self.data = np.fromfile(path, np.complex64)
        self.data /= float(np.sqrt(np.mean(np.abs(self.data) ** 2))) or 1.0
        self.file_rate = file_rate
        self.signal_hz = signal_hz
        # Noise power in the full 2 MHz such that SNR is snr_db within ~10 kHz.
        self.noise_sigma = math.sqrt(10 ** (-snr_db / 10) * SAMPLE_RATE / 10e3 / 2)
        self.lo_hz = 0.0
        self.pos = 0.0          # position in the file, in file samples
        self.t = 0              # output samples produced, for the mixer phase
        self.rng = np.random.default_rng(1)
        self.realtime = False

    def tune(self, lo_hz: float) -> None:
        self.lo_hz = lo_hz

    def read(self, n: int) -> np.ndarray:
        if self.pos >= len(self.data) - 2:
            raise _StopSurvey()
        noise = (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) * self.noise_sigma
        out = noise.astype(np.complex64)
        offset = self.signal_hz - self.lo_hz
        idx = self.pos + np.arange(n) * (self.file_rate / SAMPLE_RATE)
        self.pos = float(idx[-1] + self.file_rate / SAMPLE_RATE)
        if abs(offset) < SAMPLE_RATE / 2 - 50e3:
            idx = np.minimum(idx, len(self.data) - 1)
            i0 = np.floor(idx).astype(np.int64)
            i1 = np.minimum(i0 + 1, len(self.data) - 1)
            frac = (idx - i0).astype(np.float32)
            sig = self.data[i0] * (1 - frac) + self.data[i1] * frac
            tt = (self.t + np.arange(n)) / SAMPLE_RATE
            out += (sig * np.exp(2j * np.pi * offset * tt)).astype(np.complex64)
        self.t += n
        return out

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Signal finding.

def find_signals(iq: np.ndarray, lo_hz: float, threshold_db: float) -> list[tuple[float, float]]:
    """Candidate sonde signals in one capture, as (freq_hz, snr_db), strongest
    first. Same local-spectral-floor idea as spectrum_scan.py's
    detect_hits(): sondes are always on, so they have to be found by
    standing out from the neighboring spectrum, not by switching on."""
    power = power_spectrum(iq, SCAN_FFT_SIZE)
    if power is None:
        return []
    bin_hz = SAMPLE_RATE / SCAN_FFT_SIZE
    freqs = lo_hz + (np.arange(SCAN_FFT_SIZE) - SCAN_FFT_SIZE // 2) * bin_hz
    floor = spectral_floor(power, int(SCAN_FLOOR_WINDOW_HZ / bin_hz))
    usable = (np.abs(freqs - lo_hz) <= USABLE_HALF_SPAN_HZ) & (np.abs(freqs - lo_hz) >= DC_GUARD_HZ / 2)
    active = usable & (power > floor * 10 ** (threshold_db / 10))
    found = []
    for lo, hi in contiguous_runs(active) if active.any() else []:
        width = (hi - lo) * bin_hz
        if not (SCAN_MIN_WIDTH_HZ <= width <= SCAN_MAX_WIDTH_HZ):
            continue
        p = power[lo:hi]
        centroid = float(np.sum(freqs[lo:hi] * p) / np.sum(p))
        snr = 10 * math.log10(float(np.max(p)) / float(np.mean(floor[lo:hi])))
        found.append((centroid, snr))
    return sorted(found, key=lambda c: -c[1])


def channel_snr(power: np.ndarray, lo_hz: float, freq_hz: float) -> float | None:
    bin_hz = SAMPLE_RATE / len(power)
    center = int(round((freq_hz - lo_hz) / bin_hz)) + len(power) // 2
    half = int(10e3 / bin_hz)  # M10/M20 spread energy out to ~+-10 kHz
    if center - half < 0 or center + half >= len(power):
        return None
    floor = float(np.median(power))
    return 10 * math.log10(max(float(np.max(power[center - half:center + half + 1])), 1e-30) / floor)


def choose_lo(freqs: list[float]) -> float:
    """An LO that has all of freqs within the usable span and none of them
    near the DC spike at the center."""
    lo = (min(freqs) + max(freqs)) / 2
    for nudge in (0, 60e3, -60e3, 120e3, -120e3, 180e3, -180e3):
        cand = lo + nudge
        if all(DC_GUARD_HZ <= abs(f - cand) <= USABLE_HALF_SPAN_HZ for f in freqs):
            return cand
    return lo + 60e3


# --------------------------------------------------------------------------
# Main tracker.

class Tracker:
    def __init__(self, source, args):
        self.src = source
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.flights_dir = self.out_dir / "flights"
        self.flights_dir.mkdir(parents=True, exist_ok=True)
        self.decoder_dir = Path(args.decoder_dir)
        self.fixed_freqs = [float(f) for f in (args.freq or [])]
        self.forced_type = None if args.type == "auto" else args.type
        self.channels: list[SondeChannel] = []
        self.ignored: dict[float, float] = {}  # freq -> stream time it's ignored until
        self.lo_hz = None
        self.mode = "idle"
        self.stream_t = 0.0  # seconds of IQ processed; drives every timeout (so --replay runs faster than real time)
        self.last_sweep = None
        self._last_status = 0.0
        self._last_recheck = 0.0
        self._sweeps_without_hits = 0
        self._power_acc = None

    # -- helpers ----------------------------------------------------------

    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def _read(self, n: int) -> np.ndarray:
        iq = self.src.read(n)
        self.stream_t += len(iq) / SAMPLE_RATE
        return iq

    def _is_ignored(self, f: float) -> bool:
        return any(abs(f - g) < SAME_SIGNAL_HZ and until > self.stream_t for g, until in self.ignored.items())

    def _is_tracked(self, f: float) -> bool:
        return any(abs(f - c.freq_hz) < SAME_SIGNAL_HZ for c in self.channels if c.state != "closed")

    def _open(self, freq_hz: float, fixed: bool = False) -> None:
        ch = SondeChannel(freq_hz, self.lo_hz, self.stream_t, self.forced_type, self.decoder_dir,
                           lossless=not self.src.realtime, fixed=fixed)
        self.channels.append(ch)
        self.log(f"{freq_hz / 1e6:.4f} MHz: "
                 + (f"decoding as {self.forced_type}" if self.forced_type else "detecting sonde type..."))

    # -- sweeping ---------------------------------------------------------

    def sweep(self) -> list[tuple[float, float]]:
        # Windows overlap by half, so the DC-guarded middle of each one is
        # covered by its neighbors -- otherwise a sonde sitting right on a
        # window center would never be found.
        step = USABLE_HALF_SPAN_HZ
        centers = []
        c = self.args.band_start_hz + step
        while c - USABLE_HALF_SPAN_HZ < self.args.band_end_hz:
            centers.append(c)
            c += step
        found = []
        for center in centers:
            if not _running:
                break
            self.src.tune(center)
            self._read(CHUNK_SAMPLES)  # discard settling samples
            iq = self._read(int(SAMPLE_RATE * SCAN_DWELL_S))
            for f, snr in find_signals(iq, center, self.args.scan_threshold_db):
                if self.args.band_start_hz <= f <= self.args.band_end_hz and not self._is_ignored(f):
                    found.append((f, snr))
        found.sort(key=lambda c: -c[1])
        merged = []  # the same signal seen from two overlapping windows: keep the stronger reading
        for f, snr in found:
            if all(abs(f - g) >= SAME_SIGNAL_HZ for g, _ in merged):
                merged.append((f, snr))
        found = merged
        self.last_sweep = {"at": datetime.now(timezone.utc).isoformat(),
                           "candidates": [{"freq_hz": round(f), "snr_db": round(s, 1)} for f, s in found]}
        return found

    def park(self, freqs: list[float]) -> None:
        self._power_acc = None
        self.lo_hz = choose_lo(freqs)
        self.src.tune(self.lo_hz)
        self.mode = "tracking"
        self.log(f"parked at {self.lo_hz / 1e6:.4f} MHz covering "
                 + ", ".join(f"{f / 1e6:.4f}" for f in freqs) + " MHz")

    # -- per-chunk work ---------------------------------------------------

    def _handle_channel_output(self, ch: SondeChannel) -> None:
        for line in ch.proc.drain_lines():
            if ch.state == "detect":
                parsed = parse_dft_detect(line)
                if parsed is None:
                    continue
                name, score, offset = parsed
                ch.detected_as = name
                sonde_type = decoder_type_for(name)
                if sonde_type is None:
                    self.log(f"{ch.freq_hz / 1e6:.4f} MHz: detected {name} (score {score:.2f}), "
                             "but no decoder is wired up for that type")
                    ch.proc.kill()
                    ch.state = "unsupported"
                    return
                ch.proc.kill()
                ch.freq_hz += offset
                ch.chan.retune(ch.freq_hz - self.lo_hz)
                self.log(f"{ch.freq_hz / 1e6:.4f} MHz: detected {name} (score {score:.2f}, "
                         f"offset {offset:+.0f} Hz), decoding")
                ch._start_decode(sonde_type, self.stream_t)
                return
            elif ch.state == "decode" and line.startswith("{"):
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if line == ch._last_line:
                    continue
                ch._last_line = line
                self._on_frame(ch, frame)

    def _on_frame(self, ch: SondeChannel, frame: dict) -> None:
        serial = frame.get("id")
        if not serial or "xxxx" in serial:
            return  # DFMs send frames before their serial has been fully assembled
        now = datetime.now(timezone.utc)
        if frame.get("type") == "IMET":
            # iMet-4s don't transmit a serial number (id is always "iMet");
            # same workaround as auto_rx: name the flight by date + frequency.
            serial = f"IMET-{now:%Y%m%d}-{ch.freq_hz / 1e3:.0f}"
            frame["id"] = serial
        valid = position_valid(frame)
        frame["position_valid"] = valid
        if frame.get("vel_v") is None and frame.get("alt") is not None:
            # Over >=5 s: iMet altitudes are whole metres, so frame-to-frame
            # differences are mostly quantization noise.
            if ch._prev_alt is None or self.stream_t - ch._prev_alt[0] > 30:
                ch._prev_alt = (self.stream_t, frame["alt"])
            elif self.stream_t - ch._prev_alt[0] >= 5:
                ch._vel_v = (frame["alt"] - ch._prev_alt[1]) / (self.stream_t - ch._prev_alt[0])
                ch._prev_alt = (self.stream_t, frame["alt"])
            if getattr(ch, "_vel_v", None) is not None:
                frame["vel_v"] = round(ch._vel_v, 2)
        frame["rx_time"] = now.isoformat()
        frame["freq_mhz"] = round(ch.freq_hz / 1e6, 5)
        if ch.snr_db is not None:
            frame["snr_db"] = round(ch.snr_db, 1)
        if valid and frame.get("alt") is not None:
            dist, az, el = look_angles(self.args.station_lat, self.args.station_lon, self.args.station_alt,
                                        frame["lat"], frame["lon"], frame["alt"])
            frame["distance_km"] = round(dist, 2)
            frame["azimuth_deg"] = round(az, 1)
            frame["elevation_deg"] = round(el, 2)
        safe_serial = "".join(c for c in str(serial) if c.isalnum() or c in "-_")
        with open(self.flights_dir / f"{safe_serial}.jsonl", "a") as fh:
            fh.write(json.dumps(frame) + "\n")
        if ch.serial != serial:
            self.log(f"{ch.freq_hz / 1e6:.4f} MHz: receiving {frame.get('subtype') or frame.get('type')} {serial}")
        ch.serial = serial
        ch.frames += 1
        ch.last_frame_at = now.isoformat()
        ch._last_frame_t = self.stream_t
        ch.last_frame = {k: frame.get(k) for k in (
            "datetime", "frame", "lat", "lon", "alt", "vel_v", "vel_h", "heading", "temp", "humidity",
            "pressure", "batt", "sats", "distance_km", "azimuth_deg", "elevation_deg", "position_valid")}
        if ch.frames == 1 or ch.frames % 30 == 0:
            pos = f"{frame['lat']:.5f},{frame['lon']:.5f}" if valid else "no GPS fix yet"
            self.log(f"  {serial} #{frame.get('frame')}: {pos} "
                     f"alt {frame.get('alt', 0):.0f} m, vV {frame.get('vel_v', 0):+.1f} m/s"
                     + (f", T {frame['temp']:.1f} C" if frame.get("temp") not in (None, -273.0) else "")
                     + (f", {frame['distance_km']:.0f} km @ {frame['azimuth_deg']:.0f} deg" if "distance_km" in frame else ""))

    def _lifecycle(self, ch: SondeChannel) -> None:
        age = self.stream_t - ch.state_since
        if ch.state == "detect":
            if age >= DETECT_SECONDS and ch.proc.alive():
                ch.proc.close_input()  # EOF -> dft_detect gives up if it hasn't matched yet
            if age >= DETECT_SECONDS + 3 and not ch.proc.lines.qsize():
                self._give_up(ch, "no sonde detected")
        elif ch.state == "decode":
            last = getattr(ch, "_last_frame_t", None)
            if last is None and age > FIRST_FRAME_TIMEOUT_S:
                self._give_up(ch, f"{ch.sonde_type} detected but no frames decoded")
            elif last is not None and self.stream_t - last > self.args.lost_timeout:
                self._give_up(ch, f"signal lost (no frames for {self.args.lost_timeout:.0f} s)")
            elif not ch.proc.alive() and not ch.proc.lines.qsize():
                self._give_up(ch, "decoder exited")
        elif ch.state == "unsupported" and age > DETECT_SECONDS:
            self._give_up(ch, f"unsupported type {ch.detected_as}")

    def _give_up(self, ch: SondeChannel, reason: str) -> None:
        self.log(f"{ch.freq_hz / 1e6:.4f} MHz: {reason}" + (", retrying" if ch.fixed else ""))
        ch.close(reason, self.stream_t)
        if ch.fixed:
            ch.__init__(ch.freq_hz, self.lo_hz, self.stream_t, self.forced_type, self.decoder_dir,
                        lossless=not self.src.realtime, fixed=True)
        elif ch.frames == 0:
            self.ignored[ch.freq_hz] = self.stream_t + self.args.ignore_time

    def _recheck_window(self, iq: np.ndarray) -> None:
        if len([c for c in self.channels if c.state != "closed"]) >= self.args.max_channels:
            return
        for f, snr in find_signals(iq, self.lo_hz, self.args.scan_threshold_db):
            if not (self.args.band_start_hz <= f <= self.args.band_end_hz):
                continue
            if self._is_tracked(f) or self._is_ignored(f):
                continue
            self.log(f"new signal in window: {f / 1e6:.4f} MHz ({snr:.1f} dB)")
            self._open(f)
            return

    def write_status(self) -> None:
        status = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "lo_hz": self.lo_hz,
            "fixed_freqs": self.fixed_freqs,
            "forced_type": self.forced_type,
            "channels": [c.status() for c in self.channels if c.state != "closed"],
            "last_sweep": self.last_sweep,
            "ignored": sorted(round(f) for f, until in self.ignored.items() if until > self.stream_t),
        }
        tmp = self.out_dir / "status.json.tmp"
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.out_dir / "status.json")

    # -- main loop --------------------------------------------------------

    def run(self) -> None:
        if self.fixed_freqs:
            if max(self.fixed_freqs) - min(self.fixed_freqs) > 2 * USABLE_HALF_SPAN_HZ - 2 * DC_GUARD_HZ:
                raise SystemExit("--freq values must all fit within one ~1.35 MHz window")
            self.park(self.fixed_freqs)
            for f in self.fixed_freqs:
                self._open(f, fixed=True)

        while _running and (self.args.duration is None or self.stream_t < self.args.duration):
            if not self.fixed_freqs and not [c for c in self.channels if c.state != "closed"]:
                self.channels = []
                self.lo_hz = None
                self.mode = "sweeping"
                self.write_status()
                found = self.sweep()
                if not found:
                    self._sweeps_without_hits += 1
                    if self._sweeps_without_hits % 10 == 1:
                        self.log(f"sweep {self.args.band_start_hz / 1e6:.1f}-{self.args.band_end_hz / 1e6:.1f} MHz: "
                                 "no candidate signals, still sweeping")
                    self.write_status()
                    continue
                self._sweeps_without_hits = 0
                self.log("sweep found: " + ", ".join(f"{f / 1e6:.4f} MHz ({s:.1f} dB)" for f, s in found))
                best = found[0][0]
                group = [f for f, _ in found if abs(f - best) <= USABLE_HALF_SPAN_HZ - DC_GUARD_HZ]
                group = group[:self.args.max_channels]
                self.park(group)
                for f in group:
                    self._open(f)
                self._last_recheck = self.stream_t

            iq = self._read(CHUNK_SAMPLES)
            if self.channels:
                # Averaged over the whole status interval for the SNR readout --
                # a single 0.1 s chunk can land between an M10's bursts.
                power = power_spectrum(iq, SCAN_FFT_SIZE)
                if power is not None:
                    self._power_acc = power if self._power_acc is None else self._power_acc + power
            for ch in self.channels:
                ch.feed(iq)
                if ch.proc is not None:
                    self._handle_channel_output(ch)
                self._lifecycle(ch)

            if self.stream_t - self._last_status >= STATUS_INTERVAL_S:
                if self._power_acc is not None:
                    for ch in self.channels:
                        if ch.state != "closed":
                            ch.snr_db = channel_snr(self._power_acc, self.lo_hz, ch.freq_hz)
                    self._power_acc = None
                self.write_status()
                self._last_status = self.stream_t

            if not self.fixed_freqs and self.stream_t - self._last_recheck >= IN_WINDOW_RECHECK_S:
                self._last_recheck = self.stream_t
                self._recheck_window(np.concatenate([iq, self._read(CHUNK_SAMPLES * 4)]))

    def shutdown(self) -> None:
        for ch in self.channels:
            if ch.state != "closed":
                ch.close("stopped", self.stream_t)
        self.mode = "stopped"
        self.channels = []
        self.write_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freq", type=float, action="append",
                         help="sonde frequency in Hz (repeatable, all within ~1.35 MHz); default: sweep the band")
    parser.add_argument("--type", default="auto", choices=["auto", "RS41", "DFM", "M10", "IMET4"],
                         help="sonde type; 'auto' (default) identifies it with dft_detect first")
    parser.add_argument("--band-start-mhz", type=float, default=BAND_START_HZ / 1e6)
    parser.add_argument("--band-end-mhz", type=float, default=BAND_END_HZ / 1e6)
    parser.add_argument("--scan-threshold-db", type=float, default=6.0,
                         help="how far above the neighboring spectrum a signal must be to try it (default: 6)")
    parser.add_argument("--max-channels", type=int, default=4,
                         help="max sondes decoded at once within the 1.4 MHz window (default: 4)")
    parser.add_argument("--lost-timeout", type=float, default=DEFAULT_LOST_TIMEOUT_S,
                         help=f"seconds without a frame before a sonde is dropped (default: {DEFAULT_LOST_TIMEOUT_S:.0f})")
    parser.add_argument("--ignore-time", type=float, default=DEFAULT_IGNORE_S,
                         help=f"seconds a signal that isn't a sonde is skipped by the sweep (default: {DEFAULT_IGNORE_S:.0f})")
    parser.add_argument("--gain", type=float, default=40.0,
                         help="manual RF gain in dB, 0-66 (default: 40); pass --gain=-1 for AGC")
    parser.add_argument("--antenna", choices=["A", "B", "C"], default=None,
                         help="RSPdx antenna input to use (default: device default, Antenna A)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                         help=f"where flights/<serial>.jsonl and status.json go (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--duration", type=float, default=None,
                         help="total seconds to run (default: until Ctrl+C)")
    parser.add_argument("--decoder-dir", default=str(DEFAULT_DECODER_DIR),
                         help=f"directory with the built auto_rx decoders (default: {DEFAULT_DECODER_DIR})")
    parser.add_argument("--station-lat", type=float, default=EBAW_LAT)
    parser.add_argument("--station-lon", type=float, default=EBAW_LON)
    parser.add_argument("--station-alt", type=float, default=EBAW_ELEV_M, help="station altitude, m")
    parser.add_argument("--replay", default=None,
                         help="test mode: complex-float32 IQ recording of a sonde to play back instead of the SDR")
    parser.add_argument("--replay-rate", type=float, default=96000.0, help="--replay file sample rate (default: 96000)")
    parser.add_argument("--replay-freq", type=float, default=403.5e6, help="frequency to place the --replay signal at")
    parser.add_argument("--replay-snr-db", type=float, default=20.0, help="--replay signal SNR in ~10 kHz (default: 20)")
    args = parser.parse_args()
    args.band_start_hz = args.band_start_mhz * 1e6
    args.band_end_hz = args.band_end_mhz * 1e6

    for tool in ("dft_detect", "rs41mod", "dfm09mod", "m10m20mod", "imet4iq"):
        if not (Path(args.decoder_dir) / tool).exists():
            raise SystemExit(f"{tool} not found in {args.decoder_dir} -- build the decoders first "
                             "(see Install in this script's docstring)")

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    if args.replay:
        source = ReplaySource(Path(args.replay), args.replay_rate, args.replay_freq, args.replay_snr_db)
    else:
        gain = None if args.gain is not None and args.gain < 0 else args.gain
        source = SdrSource(gain, args.antenna)

    tracker = Tracker(source, args)
    what = (", ".join(f"{f / 1e6:.4f}" for f in tracker.fixed_freqs) + " MHz") if tracker.fixed_freqs \
        else f"sweeping {args.band_start_mhz:.1f}-{args.band_end_mhz:.1f} MHz"
    tracker.log(f"Radiosonde tracker: {what}, telemetry -> {tracker.flights_dir}/ -- Ctrl+C to stop")
    try:
        tracker.run()
    except _StopSurvey:
        pass
    finally:
        tracker.shutdown()
        source.close()
        tracker.log("Stopped.")


if __name__ == "__main__":
    main()
