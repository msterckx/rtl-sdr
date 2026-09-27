#!/usr/bin/env python3
"""Read-side helpers for radiosonde tracking, used by webserver.py's Sondes
tab: summarize the per-flight telemetry logs sonde.py writes
(output/sonde/flights/<serial>.jsonl), return one flight's track, and look
up which sondes are currently airborne nearby according to SondeHub
(https://sondehub.org -- the community radiosonde tracking network, whose
API needs no key), so it's clear what's up and on what frequency before
spending time sweeping for it.

Kept separate from sonde.py so the web server doesn't import SoapySDR/scipy
just to read some JSON.

Usage (CLI, for a quick look without the web UI):
    python3 scripts/sonde_data.py flights
    python3 scripts/sonde_data.py nearby
"""

import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from flight_lookup import EBAW_LAT, EBAW_LON

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SONDE_DIR = PROJECT_ROOT / "output" / "sonde"
FLIGHTS_DIR = SONDE_DIR / "flights"
STATUS_PATH = SONDE_DIR / "status.json"

SONDEHUB_URL = "https://api.v2.sondehub.org/sondes"
SONDEHUB_TRACKER_URL = "https://sondehub.org/"
NEARBY_RADIUS_KM = 400
NEARBY_MAX_AGE_S = 3 * 3600
NEARBY_CACHE_S = 60
MAX_TRACK_POINTS = 2000

TRACK_FIELDS = ("rx_time", "datetime", "lat", "lon", "alt", "vel_v", "vel_h", "temp", "humidity",
                "pressure", "snr_db")

_summary_cache: dict[str, tuple[float, int, dict]] = {}
_summary_lock = threading.Lock()
_nearby_cache: dict = {"at": 0.0, "data": None}
_nearby_lock = threading.Lock()


def _ground_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _read_frames(path: Path) -> list[dict]:
    frames = []
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line still being written
    except OSError:
        pass
    return frames


def _summarize(path: Path) -> dict | None:
    frames = _read_frames(path)
    if not frames:
        return None
    last = frames[-1]
    positioned = [f for f in frames if f.get("position_valid", True) and f.get("alt") is not None]
    max_alt_frame = max(positioned, key=lambda f: f["alt"]) if positioned else None
    last_pos = positioned[-1] if positioned else None
    # Burst = it has come down meaningfully from its peak altitude.
    burst = bool(max_alt_frame and last_pos and max_alt_frame["alt"] - last_pos["alt"] > 500)
    return {
        "serial": last.get("id") or path.stem,
        "type": last.get("type"),
        "subtype": last.get("subtype"),
        "freq_mhz": last.get("freq_mhz"),
        "frames": len(frames),
        "first_rx": frames[0].get("rx_time"),
        "last_rx": last.get("rx_time"),
        "max_alt": max_alt_frame["alt"] if max_alt_frame else None,
        "burst": burst,
        "last": {k: last.get(k) for k in (
            "lat", "lon", "alt", "vel_v", "vel_h", "heading", "temp", "humidity", "pressure", "batt",
            "sats", "snr_db", "distance_km", "azimuth_deg", "elevation_deg", "position_valid")},
        "last_position": {k: last_pos.get(k) for k in ("lat", "lon", "alt", "rx_time")} if last_pos else None,
    }


def list_flights(flights_dir: Path | None = None) -> list[dict]:
    """One summary per flight log, most recently heard first. Summaries are
    cached per file by (mtime, size), so polling this every few seconds only
    re-parses the log(s) currently being appended to."""
    flights_dir = flights_dir or FLIGHTS_DIR
    out = []
    if not flights_dir.is_dir():
        return out
    for path in flights_dir.glob("*.jsonl"):
        try:
            st = path.stat()
        except OSError:
            continue
        key = str(path)
        with _summary_lock:
            cached = _summary_cache.get(key)
        if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
            summary = cached[2]
        else:
            summary = _summarize(path)
            if summary is None:
                continue
            with _summary_lock:
                _summary_cache[key] = (st.st_mtime, st.st_size, summary)
        out.append(summary)
    out.sort(key=lambda s: s.get("last_rx") or "", reverse=True)
    return out


def safe_serial(serial: str) -> str:
    return "".join(c for c in serial if c.isalnum() or c in "-_")


def flight_path(serial: str, flights_dir: Path | None = None) -> Path | None:
    path = (flights_dir or FLIGHTS_DIR) / f"{safe_serial(serial)}.jsonl"
    return path if path.is_file() else None


def read_track(serial: str, flights_dir: Path | None = None) -> list[dict] | None:
    """Positioned frames of one flight, thinned to at most MAX_TRACK_POINTS
    (always keeping the latest one) so a multi-hour flight stays light
    enough to poll."""
    path = flight_path(serial, flights_dir)
    if path is None:
        return None
    points = [{k: f.get(k) for k in TRACK_FIELDS}
              for f in _read_frames(path) if f.get("position_valid", True) and f.get("lat") is not None]
    if len(points) > MAX_TRACK_POINTS:
        step = math.ceil(len(points) / MAX_TRACK_POINTS)
        points = points[::step] + ([points[-1]] if (len(points) - 1) % step else [])
    return points


def read_status(status_path: Path | None = None) -> dict | None:
    try:
        return json.loads((status_path or STATUS_PATH).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def fetch_nearby(lat: float = EBAW_LAT, lon: float = EBAW_LON, radius_km: float = NEARBY_RADIUS_KM,
                 max_age_s: int = NEARBY_MAX_AGE_S) -> dict:
    """Sondes SondeHub has heard within radius_km over the last max_age_s,
    nearest first. Cached for NEARBY_CACHE_S; on a network error the last
    good result is returned with an "error" field instead."""
    with _nearby_lock:
        if _nearby_cache["data"] is not None and time.time() - _nearby_cache["at"] < NEARBY_CACHE_S:
            return _nearby_cache["data"]
    url = f"{SONDEHUB_URL}?lat={lat}&lon={lon}&distance={int(radius_km * 1000)}&last={int(max_age_s)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rtl-sdr-sonde-tracker"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        with _nearby_lock:
            stale = dict(_nearby_cache["data"] or {"sondes": []})
        stale["error"] = f"SondeHub unreachable: {exc}"
        return stale
    now = datetime.now(timezone.utc)
    sondes = []
    for serial, s in raw.items():
        if s.get("lat") is None or s.get("lon") is None:
            continue
        try:
            age_s = (now - datetime.fromisoformat(s["datetime"].replace("Z", "+00:00"))).total_seconds()
        except (KeyError, ValueError):
            age_s = None
        sondes.append({
            "serial": serial,
            "type": s.get("type"),
            "subtype": s.get("subtype"),
            "freq_mhz": s.get("frequency"),
            "lat": s["lat"], "lon": s["lon"], "alt": s.get("alt"),
            "vel_v": s.get("vel_v"),
            "last_heard": s.get("datetime"),
            "age_s": age_s,
            "distance_km": round(_ground_km(lat, lon, s["lat"], s["lon"]), 1),
            "uploader": s.get("uploader_callsign"),
            "tracker_url": f"{SONDEHUB_TRACKER_URL}{serial}",
        })
    sondes.sort(key=lambda s: s["distance_km"])
    data = {"fetched_at": now.isoformat(), "radius_km": radius_km, "sondes": sondes}
    with _nearby_lock:
        _nearby_cache.update(at=time.time(), data=data)
    return data


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "flights"
    if what == "nearby":
        for s in fetch_nearby()["sondes"]:
            print(f"{s['serial']:12} {s['type'] or '?':7} {s['freq_mhz'] or 0:9.4f} MHz  "
                  f"{(s['alt'] or 0):6.0f} m  {s['distance_km']:5.0f} km  heard {s['last_heard']}")
    else:
        for f in list_flights():
            last = f["last"]
            print(f"{f['serial']:16} {f['type'] or '?':6} {f['freq_mhz'] or 0:9.4f} MHz  {f['frames']:5} frames  "
                  f"last {f['last_rx']}  alt {last.get('alt') or 0:.0f} m  max {f['max_alt'] or 0:.0f} m"
                  + ("  (burst)" if f["burst"] else ""))


if __name__ == "__main__":
    main()
