#!/usr/bin/env python3
"""Tune an SDRplay device to an FM broadcast channel and record demodulated
audio to a local WAV file."""

import argparse
import sys
import wave

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
from scipy.signal import resample_poly, lfilter

SAMPLE_RATE = 2_400_000.0
AUDIO_RATE = 48_000
DECIMATION = int(SAMPLE_RATE // AUDIO_RATE)  # 50
DEEMPHASIS_US = 50.0  # 50us EU / Netherlands, use 75.0 for US/Korea


def fm_demod(iq: np.ndarray) -> np.ndarray:
    """Quadrature (phase-difference) FM discriminator."""
    prod = iq[1:] * np.conj(iq[:-1])
    return np.angle(prod).astype(np.float32)


def deemphasis(audio: np.ndarray, sample_rate: float, tau_us: float) -> np.ndarray:
    """Single-pole de-emphasis IIR filter."""
    dt = 1.0 / sample_rate
    tau = tau_us * 1e-6
    alpha = dt / (tau + dt)
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, audio).astype(np.float32)


def record(freq_hz: float, seconds: float, out_path: str, gain_db: float | None) -> None:
    candidates = [r for r in SoapySDR.Device.enumerate() if r["driver"] == "sdrplay"]
    if not candidates:
        raise RuntimeError("no sdrplay device found")
    sdr = SoapySDR.Device(candidates[0])

    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
    sdr.setBandwidth(SOAPY_SDR_RX, 0, 200_000)

    if gain_db is None:
        sdr.setGainMode(SOAPY_SDR_RX, 0, True)  # AGC
    else:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, gain_db)

    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    total_samples = int(seconds * SAMPLE_RATE)
    chunk = 65536
    buff = np.empty(chunk, np.complex64)
    captured = []
    read = 0

    print(f"Tuning to {freq_hz / 1e6:.3f} MHz, capturing {seconds:.1f}s "
          f"at {SAMPLE_RATE / 1e6:.3f} MS/s...", file=sys.stderr)

    try:
        while read < total_samples:
            want = min(chunk, total_samples - read)
            sr = sdr.readStream(rx, [buff], want, timeoutUs=1_000_000)
            if sr.ret > 0:
                captured.append(buff[: sr.ret].copy())
                read += sr.ret
            elif sr.ret < 0:
                print(f"readStream error: {sr.ret}", file=sys.stderr)
                break
    finally:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)

    iq = np.concatenate(captured)
    print(f"Captured {len(iq)} IQ samples, demodulating...", file=sys.stderr)

    audio = fm_demod(iq)

    # anti-alias lowpass + decimation to audio rate
    audio = resample_poly(audio, up=1, down=DECIMATION, window=("kaiser", 5.0))

    audio = deemphasis(audio, AUDIO_RATE, DEEMPHASIS_US)

    # normalize and convert to 16-bit PCM
    peak = np.max(np.abs(audio)) or 1.0
    pcm = np.clip(audio / peak * 0.95 * 32767, -32768, 32767).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(pcm.tobytes())

    print(f"Wrote {len(pcm) / AUDIO_RATE:.2f}s of audio to {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", type=float, default=100.9e6,
                         help="FM station frequency in Hz (default: 100.9 MHz)")
    parser.add_argument("--seconds", type=float, default=10.0,
                         help="duration to record in seconds")
    parser.add_argument("--out", default="output/fm_100.9.wav",
                         help="output WAV file path")
    parser.add_argument("--gain", type=float, default=None,
                         help="manual RF gain in dB (default: AGC)")
    args = parser.parse_args()

    record(args.freq, args.seconds, args.out, args.gain)


if __name__ == "__main__":
    main()
