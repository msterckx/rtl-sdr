#!/usr/bin/env python3
"""Listen to a DMR (Digital Mobile Radio) channel on an SDRplay device and
decode it live via dsd-fme (an external digital-voice decoder --
https://github.com/lwvmobile/dsd-fme, built locally with mbelib for AMBE+2
voice decode -- see Install_Notes below).

DMR is TDMA + a proprietary-derived vocoder, not something record.py's
hand-rolled AM/FM discriminator chain can turn into audio on its own, so
this script only does the RF front end (tune, wideband FM-discriminate,
decimate) and hands the resulting audio stream to dsd-fme, which does the
actual DMR sync/slot/AMBE decode. There is no squelch here -- unlike
record.py/scan.py's power-based voice-activated squelch on analog audio,
the discriminator output is streamed to dsd-fme continuously and dsd-fme
itself detects sync bursts and writes one WAV file per completed call
(embedding the talkgroup/radio IDs from the DMR frame headers into the
filename). Each finished call WAV is then transcribed with whisper.cpp,
same as record.py/scan.py.

Install (no root needed beyond the one apt line for headers/libs):
    sudo apt install libpulse-dev libsndfile1-dev libfftw3-dev liblapack-dev \\
        socat libusb-1.0-0-dev rtl-sdr librtlsdr-dev libncurses-dev \\
        libcodec2-dev pulsemixer pulseaudio

    cd ~/src
    git clone https://github.com/lwvmobile/mbelib && cd mbelib
    git checkout ambe_tones
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=~/src/dsd-fme-local .. && make -j4 && make install
    cd ~/src
    git clone https://github.com/lwvmobile/dsd-fme && cd dsd-fme
    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=~/src/dsd-fme-local -DCMAKE_INSTALL_PREFIX=~/src/dsd-fme-local .. \\
        && make -j4 && make install

Usage:
    scripts/dmr.py --freq 439900000 --out-dir output
"""

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

from record import (
    SAMPLE_RATE, CHUNK_SECONDS, CHUNK_SAMPLES,
    DEFAULT_WHISPER_BIN, DEFAULT_WHISPER_MODEL,
    open_sdr, FmDiscriminator, transcribe_file,
)

# Discriminator output rate fed to dsd-fme's stdin. 2,000,000 / 40 = 50,000
# exactly -- dsd-fme takes an explicit --samplerate so it doesn't have to be
# its own 48000 default, just a clean decimation of SAMPLE_RATE.
DMR_AUDIO_RATE = 50_000
DMR_DECIMATION = int(SAMPLE_RATE // DMR_AUDIO_RATE)

# DMR's 4FSK signal (9600 baud raw, 4800 symbols/s) needs much more bandwidth
# than record.py's 4 kHz voice-band filter -- this passes the sidebands
# while staying well under DMR_AUDIO_RATE/2 (25 kHz).
DISC_LOWPASS_HZ = 8_000

# Peak DMR symbol deviation is +-1944 Hz (ETSI TS 102 361-1 / MOTOTRBO).
# Scaling instantaneous frequency by this (with headroom) keeps the 4FSK
# levels well separated without clipping. This is the one parameter most
# likely to need tuning against a real signal -- if dsd-fme reports sync
# but garbled/no audio, try raising or lowering it via --disc-full-scale-hz.
DISC_FULL_SCALE_HZ = 2_500.0

SRC_DIR = Path("/home/michel/src")
DEFAULT_DSDFME_BIN = SRC_DIR / "dsd-fme-local" / "bin" / "dsd-fme"
DEFAULT_DSDFME_LIB_DIR = SRC_DIR / "dsd-fme-local" / "lib"

_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


def discriminator_to_pcm16(theta: np.ndarray, lp_b, lp_a, lp_zi, full_scale_hz: float):
    """Convert quadrature-discriminator phase-difference samples (radians
    per sample, at SAMPLE_RATE) into int16 PCM at DMR_AUDIO_RATE for
    dsd-fme's stdin, via instantaneous frequency in Hz -- a fixed,
    physically-grounded gain rather than record.py's per-clip peak
    normalization, since this is a live unbounded stream."""
    freq_hz = theta.astype(np.float64) * (SAMPLE_RATE / (2 * np.pi))
    filtered, lp_zi = lfilter(lp_b, lp_a, freq_hz, zi=lp_zi)
    decimated = filtered[::DMR_DECIMATION]
    pcm = np.clip(decimated / full_scale_hz * 32767, -32768, 32767).astype(np.int16)
    return pcm, lp_zi


def start_dsdfme(bin_path: Path, lib_dir: Path, calls_dir: Path, event_log: Path) -> subprocess.Popen:
    calls_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    cmd = [
        str(bin_path),
        "-fs",              # DMR TDMA BS/MS simplex -- fixed non-trunked channel, no trunking logic
        "-i", "-",           # raw PCM16LE mono from stdin
        "-s", str(DMR_AUDIO_RATE),
        "-o", "null",         # no live audio sink; we only want the per-call WAV files
        "-7", str(calls_dir),  # per-call WAV output directory
        "-P",                 # enable per-call WAV writing
        "-J", str(event_log),  # human-readable call event journal (supplementary; not parsed here)
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, bufsize=0)


def _forward_dsdfme_output(proc: subprocess.Popen) -> None:
    """Relay dsd-fme's own log lines to our stderr -- when this script is
    itself run as a subprocess by webserver.py, that surfaces them in the
    web UI's log panel the same way record.py's/scan.py's own prints do."""
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", "replace").rstrip("\n")
        if line:
            print(f"  [dsd-fme] {line}", file=sys.stderr)


def watch_calls(calls_dir: Path, maybe_transcribe, poll_interval: float = 1.0) -> None:
    """Poll calls_dir for WAV files dsd-fme has finished writing. dsd-fme
    writes each call under a TEMP_ name while it's in progress and renames
    it to the final name (with talkgroup/radio IDs embedded) only once the
    call ends, so anything not named TEMP_* is already complete."""
    seen = set()
    global _running
    while _running:
        time.sleep(poll_interval)
        try:
            wavs = list(calls_dir.glob("*.wav"))
        except OSError:
            continue
        for path in wavs:
            if path.name.startswith("TEMP_") or path.name in seen:
                continue
            seen.add(path.name)
            print(f"  wrote {path.name}", file=sys.stderr)
            maybe_transcribe(path)


def listen(freq_hz: float, out_dir: Path, gain_db, dsdfme_bin: Path, dsdfme_lib_dir: Path,
           transcribe: bool, whisper_bin: str, whisper_model: str, whisper_threads: int,
           disc_full_scale_hz: float = DISC_FULL_SCALE_HZ, duration=None, antenna=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    event_log = out_dir / "dsd-fme_events.log"

    if not Path(dsdfme_bin).exists():
        raise RuntimeError(
            f"dsd-fme binary not found at {dsdfme_bin} -- build it first "
            "(see the Install section in this script's docstring)")

    executor = None
    whisper_available = Path(whisper_bin).exists() and Path(whisper_model).exists()
    if transcribe:
        executor = ThreadPoolExecutor(max_workers=2)
        if not whisper_available:
            print(f"Transcription disabled: whisper-cli or model not found "
                  f"({whisper_bin}, {whisper_model})", file=sys.stderr)

    def maybe_transcribe(path: Path) -> None:
        if executor is not None and whisper_available:
            executor.submit(transcribe_file, path, whisper_bin, whisper_model, whisper_threads)

    dsdfme = start_dsdfme(Path(dsdfme_bin), Path(dsdfme_lib_dir), out_dir, event_log)
    threading.Thread(target=_forward_dsdfme_output, args=(dsdfme,), daemon=True).start()
    threading.Thread(target=watch_calls, args=(out_dir, maybe_transcribe), daemon=True).start()

    # dsd-fme's stdin is fed from a bounded queue by a separate writer
    # thread rather than written to directly from the SDR read loop below --
    # same rationale as record.py's transcription offload: readStream() has
    # to keep being serviced every ~0.1s or the driver drops samples, and a
    # write() that blocked on dsd-fme falling behind would stall it. If the
    # queue fills up (dsd-fme can't keep up), we drop that chunk of DMR
    # audio rather than block -- losing part of one call beats losing SDR
    # samples for every channel.
    stdin_queue: "queue.Queue[bytes | None]" = queue.Queue(maxsize=50)  # ~5s of audio headroom

    def _stdin_writer():
        while True:
            item = stdin_queue.get()
            if item is None:
                return
            try:
                dsdfme.stdin.write(item)
            except (BrokenPipeError, OSError):
                return

    threading.Thread(target=_stdin_writer, daemon=True).start()

    sdr = open_sdr(gain_db, antenna=antenna)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    lp_b, lp_a = butter(4, DISC_LOWPASS_HZ / (SAMPLE_RATE / 2), btype="low")
    lp_zi = np.zeros(max(len(lp_a), len(lp_b)) - 1, dtype=np.float64)
    fm_disc = FmDiscriminator()

    buff = np.empty(CHUNK_SAMPLES, np.complex64)

    global _running
    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"Listening on {freq_hz / 1e6:.4f} MHz (DMR), decoding via dsd-fme, "
          f"per-call WAVs written to {out_dir}/ -- Ctrl+C to stop", file=sys.stderr)

    elapsed = 0.0
    try:
        while _running and (duration is None or elapsed < duration) and dsdfme.poll() is None:
            sr = sdr.readStream(rx, [buff], CHUNK_SAMPLES, timeoutUs=1_000_000)
            if sr.ret == 0:
                continue
            if sr.ret < 0:
                print(f"readStream error: {sr.ret}", file=sys.stderr)
                continue
            iq = buff[: sr.ret]
            elapsed += sr.ret / SAMPLE_RATE

            theta = fm_disc.apply(iq)
            pcm, lp_zi = discriminator_to_pcm16(theta, lp_b, lp_a, lp_zi, disc_full_scale_hz)
            try:
                stdin_queue.put_nowait(pcm.tobytes())
            except queue.Full:
                pass
        if dsdfme.poll() is not None:
            print(f"dsd-fme exited unexpectedly (code {dsdfme.returncode})", file=sys.stderr)
    finally:
        _running = False
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        try:
            stdin_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            dsdfme.stdin.close()
        except OSError:
            pass
        try:
            dsdfme.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dsdfme.terminate()
        if executor is not None:
            print("Waiting for pending transcriptions to finish...", file=sys.stderr)
            executor.shutdown(wait=True)
        print("Stopped.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freq", type=float, required=True, help="frequency in Hz")
    parser.add_argument("--out-dir", default="output", help="output directory for per-call WAV files")
    parser.add_argument("--gain", type=float, default=40.0,
                         help="manual RF gain in dB, 0-66 (default: 40); pass --gain=-1 for AGC")
    parser.add_argument("--antenna", choices=["A", "B", "C"], default=None,
                         help="RSPdx antenna input to use (default: device default, Antenna A)")
    parser.add_argument("--duration", type=float, default=None,
                         help="total seconds to listen (default: run until Ctrl+C)")
    parser.add_argument("--transcribe", dest="transcribe", action="store_true", default=True,
                         help="transcribe each call with local whisper.cpp (default: on)")
    parser.add_argument("--no-transcribe", dest="transcribe", action="store_false",
                         help="disable transcription")
    parser.add_argument("--whisper-bin", default=str(DEFAULT_WHISPER_BIN),
                         help=f"path to whisper-cli binary (default: {DEFAULT_WHISPER_BIN})")
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER_MODEL),
                         help=f"path to whisper.cpp ggml model (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--whisper-threads", type=int, default=4,
                         help="threads for whisper.cpp to use per transcription")
    parser.add_argument("--dsd-fme-bin", default=str(DEFAULT_DSDFME_BIN),
                         help=f"path to dsd-fme binary (default: {DEFAULT_DSDFME_BIN})")
    parser.add_argument("--dsd-fme-lib-dir", default=str(DEFAULT_DSDFME_LIB_DIR),
                         help="directory containing dsd-fme's shared libs (libmbe etc.) for "
                              f"LD_LIBRARY_PATH (default: {DEFAULT_DSDFME_LIB_DIR})")
    parser.add_argument("--disc-full-scale-hz", type=float, default=DISC_FULL_SCALE_HZ,
                         help="instantaneous frequency deviation (Hz) mapped to full-scale "
                              f"int16 before handing audio to dsd-fme (default: {DISC_FULL_SCALE_HZ}); "
                              "tune this if dsd-fme syncs but audio is garbled or absent")
    args = parser.parse_args()

    gain = None if args.gain is not None and args.gain < 0 else args.gain
    listen(args.freq, Path(args.out_dir), gain, args.dsd_fme_bin, args.dsd_fme_lib_dir,
           args.transcribe, args.whisper_bin, args.whisper_model, args.whisper_threads,
           args.disc_full_scale_hz, args.duration, args.antenna)


if __name__ == "__main__":
    main()
