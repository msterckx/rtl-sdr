#!/usr/bin/env python3
"""Cycle through a list of receiver frequencies on an SDRplay device and
record whatever transmission it finds -- a generic scanner usable for the
ATC channels, PMR446, or any other AM/FM channel list.

Channels come from presets.py by default:
  --group atc|pmr|marine|amateur|cb|all   scan a named group (default: all)
  --channels KEY,KEY                      scan specific preset keys instead of a whole group

Or bypass presets.py entirely with an ad hoc list:
  --freqs FREQ_HZ:MODE:PREFIX[:LABEL],...
    e.g. --freqs 446006250:fm:PMR1,135205000:am:TWR

For each channel it dwells briefly, checking RF power against a per-channel
noise floor; if a signal is found it stops hopping, demodulates and records
the transmission (same squelch open/close logic as record.py), then resumes
scanning from the next channel.

DMR channels aren't scannable this way -- there's no RF-power squelch to
hop on, dsd-fme itself has to run continuously against one fixed frequency.
Use scripts/dmr.py for those instead.
"""

import argparse
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

from presets import PRESETS, GROUPS
from record import (
    SAMPLE_RATE, AUDIO_RATE, DECIMATION, CHUNK_SECONDS, CHUNK_SAMPLES,
    DEFAULT_WHISPER_BIN, DEFAULT_WHISPER_MODEL,
    DEFAULT_RUNPOD_ENDPOINT, RUNPOD_API_KEY_ENV, RUNPOD_BATCH_SIZE,
    open_sdr, DcBlocker, FmDiscriminator, write_wav, transcribe_file,
    RunpodAtcBatcher, QualityLogger,
)

_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


class _StopScan(Exception):
    """Raised out of read_chunk() once Ctrl+C has been requested."""


class Channel:
    __slots__ = ("key", "label", "freq_hz", "mode", "prefix", "category")

    def __init__(self, key, label, freq_hz, mode, prefix, category=None):
        self.key = key
        self.label = label
        self.freq_hz = freq_hz
        self.mode = mode
        self.prefix = prefix
        self.category = category


def resolve_channels(group: str, channels_arg: str | None, freqs_arg: str | None) -> list[Channel]:
    if freqs_arg:
        channels = []
        for i, spec in enumerate(freqs_arg.split(",")):
            parts = spec.split(":")
            freq_hz = float(parts[0])
            mode = parts[1] if len(parts) > 1 else "am"
            prefix = parts[2] if len(parts) > 2 else f"CH{i + 1}"
            label = parts[3] if len(parts) > 3 else prefix
            if mode not in ("am", "fm"):
                raise ValueError(f"invalid mode in --freqs entry {spec!r}: {mode}")
            channels.append(Channel(f"custom{i + 1}", label, freq_hz, mode, prefix))
        return channels

    if channels_arg:
        keys = [k.strip() for k in channels_arg.split(",")]
    elif group == "all":
        keys = list(PRESETS.keys())
    else:
        keys = GROUPS[group]

    return [Channel(k, PRESETS[k]["label"], PRESETS[k]["freq"], PRESETS[k]["mode"],
                    PRESETS[k]["prefix"], PRESETS[k]["category"])
            for k in keys]


def demod_chunk(iq, mode, lp_b, lp_a, lp_zi, dc_blocker, fm_disc):
    envelope = np.abs(iq).astype(np.float64)
    chunk_power = float(np.mean(envelope))
    if mode == "am":
        filtered, lp_zi = lfilter(lp_b, lp_a, envelope, zi=lp_zi)
        audio_chunk = dc_blocker.apply(filtered[::DECIMATION].astype(np.float32))
    else:  # fm
        demod = fm_disc.apply(iq).astype(np.float64)
        filtered, lp_zi = lfilter(lp_b, lp_a, demod, zi=lp_zi)
        audio_chunk = filtered[::DECIMATION].astype(np.float32)
    return chunk_power, audio_chunk, lp_zi


def scan(channels, out_dir, gain_db, dwell, open_ratio, close_ratio,
         hang_time, min_duration, pre_roll, transcribe,
         whisper_bin, whisper_model, whisper_threads,
         runpod_api_key=None, runpod_endpoint=DEFAULT_RUNPOD_ENDPOINT,
         runpod_batch_size=RUNPOD_BATCH_SIZE,
         quality_log=True, quality_log_path=None, quality_log_interval=60.0, antenna=None):
    if not channels:
        raise ValueError("no channels to scan")

    out_dir.mkdir(parents=True, exist_ok=True)

    quality_logger = None
    if quality_log:
        qpath = Path(quality_log_path) if quality_log_path else out_dir / "rf_quality.jsonl"
        quality_logger = QualityLogger(qpath, quality_log_interval)

    executor = None
    runpod_batcher = None
    whisper_available = Path(whisper_bin).exists() and Path(whisper_model).exists()
    if transcribe:
        executor = ThreadPoolExecutor(max_workers=2)
        if runpod_api_key:
            runpod_batcher = RunpodAtcBatcher(executor, runpod_api_key, runpod_endpoint,
                                               runpod_batch_size)
        if runpod_batcher is None and not whisper_available:
            print(f"Transcription disabled: whisper-cli or model not found "
                  f"({whisper_bin}, {whisper_model})", file=sys.stderr)

    def maybe_transcribe(path: Path, category, skip_ms: float = 0.0) -> None:
        if category is not None and category.startswith("atc") and runpod_batcher is not None:
            runpod_batcher.add(path, skip_ms)
        elif executor is not None and whisper_available:
            executor.submit(transcribe_file, path, whisper_bin, whisper_model,
                             whisper_threads, skip_ms)

    sdr = open_sdr(gain_db, antenna=antenna)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    buff = np.empty(CHUNK_SAMPLES, np.complex64)

    def read_chunk():
        while _running:
            sr = sdr.readStream(rx, [buff], CHUNK_SAMPLES, timeoutUs=1_000_000)
            if sr.ret > 0:
                return buff[: sr.ret].copy()
            if sr.ret < 0:
                print(f"readStream error: {sr.ret}", file=sys.stderr)
        raise _StopScan()

    dwell_chunks = max(1, int(round(dwell / CHUNK_SECONDS)))
    hang_chunks = max(1, int(round(hang_time / CHUNK_SECONDS)))
    pre_roll_chunks = max(0, int(round(pre_roll / CHUNK_SECONDS)))
    min_samples = int(min_duration * AUDIO_RATE)
    noise_floor = {ch.key: 1e-6 for ch in channels}

    global _running
    signal.signal(signal.SIGINT, _handle_sigint)

    chan_list = ", ".join(f"{c.label} ({c.freq_hz / 1e6:.4f} MHz {c.mode.upper()})" for c in channels)
    print(f"Scanning {len(channels)} channel(s): {chan_list}", file=sys.stderr)
    print(f"Writing transmissions to {out_dir}/ -- Ctrl+C to stop", file=sys.stderr)

    n_written = 0
    idx = 0
    try:
        while True:
            chan = channels[idx % len(channels)]
            idx += 1

            sdr.setFrequency(SOAPY_SDR_RX, 0, chan.freq_hz)
            lp_b, lp_a = butter(4, 4_000 / (SAMPLE_RATE / 2), btype="low")
            lp_zi = np.zeros(max(len(lp_a), len(lp_b)) - 1, dtype=np.float64)
            dc_blocker = DcBlocker()
            fm_disc = FmDiscriminator()

            read_chunk()  # discard one chunk right after retuning to let filters/PLL settle

            dwell_audio = []
            locked = False
            dwell_powers = []
            for _ in range(dwell_chunks):
                iq = read_chunk()
                chunk_power, audio_chunk, lp_zi = demod_chunk(
                    iq, chan.mode, lp_b, lp_a, lp_zi, dc_blocker, fm_disc)
                dwell_powers.append(chunk_power)
                dwell_audio.append(audio_chunk)
                if chunk_power > noise_floor[chan.key] * open_ratio:
                    locked = True

            if not locked:
                noise_floor[chan.key] = 0.98 * noise_floor[chan.key] + 0.02 * min(dwell_powers)
                if quality_logger is not None:
                    quality_logger.maybe_log_idle(chan.key, chan.prefix, chan.freq_hz,
                                                   chan.mode, noise_floor[chan.key])
                continue

            tx_start_time = datetime.now()
            print(f"[{tx_start_time:%H:%M:%S}] activity on {chan.label} "
                  f"({chan.freq_hz / 1e6:.4f} MHz)", file=sys.stderr)
            tx_buffer = dwell_audio[-pre_roll_chunks:] if pre_roll_chunks else []
            tx_preroll_ms = len(tx_buffer) * CHUNK_SECONDS * 1000
            tx_peak_power = max(dwell_powers) if dwell_powers else 0.0
            tx_power_sum = sum(dwell_powers)
            tx_power_n = len(dwell_powers)
            close_counter = 0
            while True:
                iq = read_chunk()
                chunk_power, audio_chunk, lp_zi = demod_chunk(
                    iq, chan.mode, lp_b, lp_a, lp_zi, dc_blocker, fm_disc)
                tx_buffer.append(audio_chunk)
                tx_peak_power = max(tx_peak_power, chunk_power)
                tx_power_sum += chunk_power
                tx_power_n += 1
                if chunk_power < noise_floor[chan.key] * close_ratio:
                    close_counter += 1
                else:
                    close_counter = 0
                if close_counter >= hang_chunks:
                    break

            audio = np.concatenate(tx_buffer)
            if len(audio) >= min_samples:
                path = write_wav(out_dir, chan.prefix, tx_start_time, audio)
                n_written += 1
                print(f"[{tx_start_time:%H:%M:%S}] wrote {path.name} "
                      f"({len(audio) / AUDIO_RATE:.1f}s)", file=sys.stderr)
                maybe_transcribe(path, chan.category, tx_preroll_ms)
                if quality_logger is not None:
                    quality_logger.log_tx(
                        chan.key, chan.prefix, chan.freq_hz, chan.mode,
                        noise_floor[chan.key], tx_peak_power,
                        tx_power_sum / tx_power_n if tx_power_n else tx_peak_power,
                        len(audio) / AUDIO_RATE, path.name)
    except _StopScan:
        pass
    finally:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        if runpod_batcher is not None:
            runpod_batcher.flush()
        if executor is not None:
            print("Waiting for pending transcriptions to finish...", file=sys.stderr)
            executor.shutdown(wait=True)
        print(f"Stopped. {n_written} transmission(s) written to {out_dir}/", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=list(GROUPS.keys()) + ["all"], default="all",
                         help="preset channel group to scan (default: all)")
    parser.add_argument("--channels", default=None,
                         help="comma-separated presets.py keys to scan instead of --group, "
                              "e.g. pmr1,pmr2,twr")
    parser.add_argument("--freqs", default=None,
                         help="ad hoc channel list instead of presets.py: comma-separated "
                              "freq_hz:mode:prefix[:label], e.g. "
                              "446006250:fm:PMR1,135205000:am:TWR")
    parser.add_argument("--out-dir", default="output", help="output directory for WAV files")
    parser.add_argument("--gain", type=float, default=40.0,
                         help="manual RF gain in dB, 0-66 (default: 40); pass --gain=-1 for AGC")
    parser.add_argument("--antenna", choices=["A", "B", "C"], default=None,
                         help="RSPdx antenna input to use (default: device default, Antenna A)")
    parser.add_argument("--dwell", type=float, default=0.3,
                         help="seconds to sample each idle channel before moving on")
    parser.add_argument("--open-ratio", type=float, default=4.0,
                         help="signal/noise power ratio to treat a channel as active")
    parser.add_argument("--close-ratio", type=float, default=2.0,
                         help="signal/noise power ratio to close squelch (hysteresis)")
    parser.add_argument("--hang-time", type=float, default=1.0,
                         help="seconds below threshold before closing squelch")
    parser.add_argument("--min-duration", type=float, default=0.4,
                         help="discard transmissions shorter than this (seconds)")
    parser.add_argument("--pre-roll", type=float, default=0.3,
                         help="seconds of dwell audio to keep before the point of detection")
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
    parser.add_argument("--runpod-api-key", default=os.environ.get(RUNPOD_API_KEY_ENV),
                         help=f"RunPod API key for ATC transcription (default: ${RUNPOD_API_KEY_ENV}); "
                              "when set, channels with category 'atc' are transcribed via the "
                              "hosted ATC Whisper model in batches instead of local whisper.cpp")
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
                         help="minimum seconds between idle noise-floor samples per channel "
                              "(default: 60)")
    args = parser.parse_args()

    channels = resolve_channels(args.group, args.channels, args.freqs)
    gain = None if args.gain is not None and args.gain < 0 else args.gain
    scan(channels, Path(args.out_dir), gain, args.dwell,
         args.open_ratio, args.close_ratio, args.hang_time,
         args.min_duration, args.pre_roll, args.transcribe,
         args.whisper_bin, args.whisper_model, args.whisper_threads,
         args.runpod_api_key, args.runpod_endpoint, args.runpod_batch_size,
         args.quality_log, args.quality_log_path, args.quality_log_interval, args.antenna)


if __name__ == "__main__":
    main()
