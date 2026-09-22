#!/usr/bin/env python3
"""Capture a satellite pass (continuous IQ, no squelch) on an SDRplay device
and decode it with satdump.

Unlike record.py/scan.py's squelch-triggered voice capture, a satellite pass
is a single continuous transmission of known duration -- this just tunes,
records baseband IQ for --duration seconds, and hands the result to satdump.
Decode stays external (satdump) rather than hand-rolled: Meteor-M2's LRPT
downlink is digital QPSK, not the simple analog FM/AM that record.py's demod
chain is built for.

Usage:
    scripts/satellite.py --freq 137.9e6 --duration 600 --prefix METEOR_M2_4 \\
        --out-dir output/satellite
"""

import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

from record import SAMPLE_RATE, open_sdr

CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
# 1e6 matches meteor_m2-x_lrpt's own default samplerate in
# /usr/share/satdump/pipelines/Meteor-M.json -- 140 kHz (originally assumed
# here) only gives ~1.9x oversampling of the 72 kSym/s OQPSK signal, too
# marginal for reliable carrier/clock recovery. 2e6 / 1e6 is an exact 2:1
# decimation.
DEFAULT_BASEBAND_RATE = 1_000_000
# OQPSK pipeline for M2-3/M2-4 -- see satellites.py's comment; meteor_m2_lrpt
# (plain QPSK) is for the older, now-dead M2/M2-2.
DEFAULT_PIPELINE = "meteor_m2-x_lrpt"
# Confirmed via `satdump`'s own usage banner (no binary --help exists; a bad
# invocation just reprints this): accepted --baseband_format values are
# cf32/cs16/cs8/cu8, with its own sample command using "s16" for a cs16 file
# -- there is no wav/wav-container option, unlike record.py/fm_record.py's
# audio WAVs.
BASEBAND_FORMAT = "s16"
BASEBAND_EXT = ".s16"

_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


def write_iq_raw(out_path: Path, iq: np.ndarray) -> None:
    """Write complex baseband samples as headerless interleaved 16-bit
    signed I/Q (satdump's "cs16"/"s16" --baseband_format) -- there's no
    container/header, just raw samples, so the sample rate has to be passed
    to satdump separately via --samplerate at decode time."""
    peak = max(np.max(np.abs(iq.real)), np.max(np.abs(iq.imag))) or 1.0
    interleaved = np.empty(len(iq) * 2, dtype=np.float32)
    interleaved[0::2] = iq.real
    interleaved[1::2] = iq.imag
    pcm = np.clip(interleaved / peak * 0.95 * 32767, -32768, 32767).astype(np.int16)
    out_path.write_bytes(pcm.tobytes())


def capture(freq_hz: float, duration: float, gain_db, antenna=None) -> np.ndarray:
    sdr = open_sdr(gain_db, antenna=antenna)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    global _running
    signal.signal(signal.SIGINT, _handle_sigint)

    buff = np.empty(CHUNK_SAMPLES, np.complex64)
    captured = []
    total_samples = int(duration * SAMPLE_RATE)
    read = 0
    next_log = time.monotonic()

    print(f"Tuning to {freq_hz / 1e6:.4f} MHz, capturing {duration:.0f}s "
          f"at {SAMPLE_RATE / 1e6:.3f} MS/s -- Ctrl+C to stop early", file=sys.stderr)
    try:
        while _running and read < total_samples:
            want = min(CHUNK_SAMPLES, total_samples - read)
            sr = sdr.readStream(rx, [buff], want, timeoutUs=1_000_000)
            if sr.ret > 0:
                captured.append(buff[: sr.ret].copy())
                read += sr.ret
            elif sr.ret < 0:
                print(f"readStream error: {sr.ret}", file=sys.stderr)
            now = time.monotonic()
            if now >= next_log:
                print(f"  captured {read / SAMPLE_RATE:.0f}s / {duration:.0f}s", file=sys.stderr)
                next_log = now + 30
    finally:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)

    return np.concatenate(captured) if captured else np.empty(0, np.complex64)


def decode(raw_path: Path, out_dir: Path, pipeline: str, baseband_rate: int) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["satdump", pipeline, "baseband", str(raw_path), str(out_dir),
           "--samplerate", str(baseband_rate), "--baseband_format", BASEBAND_FORMAT]
    print(f"Decoding: {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"satdump failed to run: {exc}", file=sys.stderr)
        return False
    if result.stdout:
        print(result.stdout.strip(), file=sys.stderr)
    if result.returncode != 0:
        print(f"satdump exited with code {result.returncode}: "
              f"{result.stderr.strip()[-500:]}", file=sys.stderr)
        return False
    print(f"Decode products written to {out_dir}/", file=sys.stderr)
    return True


def run(freq_hz: float, duration: float, out_dir: Path, prefix: str, gain_db,
        baseband_rate: int = DEFAULT_BASEBAND_RATE, pipeline: str = DEFAULT_PIPELINE,
        skip_decode: bool = False, antenna=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    iq = capture(freq_hz, duration, gain_db, antenna)
    if len(iq) == 0:
        print("No samples captured, nothing to decode.", file=sys.stderr)
        return

    frac = Fraction(baseband_rate, int(SAMPLE_RATE)).limit_denominator(1000)
    baseband = resample_poly(iq, up=frac.numerator, down=frac.denominator,
                              window=("kaiser", 5.0))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"{prefix}_{ts}{BASEBAND_EXT}"
    write_iq_raw(raw_path, baseband)
    print(f"Wrote {len(baseband) / baseband_rate:.1f}s of baseband IQ to {raw_path}",
          file=sys.stderr)

    if not skip_decode:
        decode(raw_path, out_dir / f"{prefix}_{ts}_decoded", pipeline, baseband_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freq", type=float, required=True, help="downlink frequency in Hz")
    parser.add_argument("--duration", type=float, required=True,
                         help="capture duration in seconds (pass length, e.g. LOS - AOS)")
    parser.add_argument("--out-dir", default="output/satellite", help="output directory")
    parser.add_argument("--prefix", default="SAT", help="filename prefix")
    parser.add_argument("--gain", type=float, default=40.0,
                         help="manual RF gain in dB (default: 40); pass --gain=-1 for AGC")
    parser.add_argument("--antenna", choices=["A", "B", "C"], default=None,
                         help="RSPdx antenna input to use (default: device default, Antenna A)")
    parser.add_argument("--baseband-rate", type=int, default=DEFAULT_BASEBAND_RATE,
                         help=f"decimated IQ sample rate captured to the raw baseband file "
                              f"(default: {DEFAULT_BASEBAND_RATE})")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE,
                         help=f"satdump pipeline id to decode with (default: {DEFAULT_PIPELINE})")
    parser.add_argument("--skip-decode", action="store_true",
                         help="capture only, don't invoke satdump")
    args = parser.parse_args()

    gain = None if args.gain is not None and args.gain < 0 else args.gain
    run(args.freq, args.duration, Path(args.out_dir), args.prefix, gain,
        args.baseband_rate, args.pipeline, args.skip_decode, args.antenna)


if __name__ == "__main__":
    main()
