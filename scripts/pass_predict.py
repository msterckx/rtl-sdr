#!/usr/bin/env python3
"""Predict upcoming Meteor-M2 LRPT passes visible from the receiver's
location (Antwerp -- reuses flight_lookup.py's EBAW_LAT/EBAW_LON/EBAW_ELEV_M),
using python3-skyfield for orbital propagation against Celestrak TLEs.

Usage:
    scripts/pass_predict.py [--hours 48] [--min-elevation 5]
"""

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from flight_lookup import EBAW_LAT, EBAW_LON, EBAW_ELEV_M
from satellites import SATELLITES

CELESTRAK_WEATHER_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "output" / "satellite" / "tle_cache.txt"


def fetch_tles(cache_path: Path = DEFAULT_CACHE_PATH, max_age_hours: float = 24.0) -> str:
    """Return cached Celestrak weather-group TLE text, refreshing it from the
    network if the cache is missing or older than max_age_hours. Never
    raises for a network failure if a (possibly stale) cache already exists
    -- an old TLE is still far better than no pass prediction at all."""
    if cache_path.is_file():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return cache_path.read_text()

    try:
        with urllib.request.urlopen(CELESTRAK_WEATHER_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        if cache_path.is_file():
            print(f"TLE refresh failed ({exc}), using stale cache", file=sys.stderr)
            return cache_path.read_text()
        raise

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def _parse_satellites(tle_text: str, ts, earth_satellite_cls) -> dict:
    """Map catalog name (as it appears in the TLE's name line) -> EarthSatellite,
    for every three-line group in tle_text."""
    lines = [l.rstrip("\n") for l in tle_text.splitlines() if l.strip()]
    by_name = {}
    for i in range(0, len(lines) - 2, 3):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            continue
        by_name[name.strip()] = earth_satellite_cls(line1, line2, name.strip(), ts)
    return by_name


def upcoming_passes(hours: float = 48.0, min_elevation_deg: float = 5.0,
                     cache_path: Path = DEFAULT_CACHE_PATH) -> list[dict]:
    """List predicted passes (AOS/max-elevation/LOS) for every satellite in
    SATELLITES, over the next `hours`, above `min_elevation_deg`. A satellite
    whose catalog_name isn't found in the current TLE set (e.g. it's been
    deorbited, or Celestrak renamed it) is silently skipped rather than
    raising -- see satellites.py's note on Meteor-M2-3 sometimes being off
    the air.

    Imports skyfield lazily so the rest of this project (webserver.py's
    existing radio features included) still works before `python3-skyfield`
    is installed -- only this call fails, with a clear message, until then.
    """
    try:
        from skyfield.api import EarthSatellite, load, wgs84
    except ImportError as exc:
        raise RuntimeError(
            "python3-skyfield not installed -- run: sudo apt install python3-skyfield") from exc

    ts = load.timescale()
    by_name = _parse_satellites(fetch_tles(cache_path), ts, EarthSatellite)
    observer = wgs84.latlon(EBAW_LAT, EBAW_LON, elevation_m=EBAW_ELEV_M)

    t0 = ts.now()
    t1 = t0 + hours / 24.0

    passes = []
    for key, sat_info in SATELLITES.items():
        sat = by_name.get(sat_info["catalog_name"])
        if sat is None:
            continue
        times, events = sat.find_events(observer, t0, t1, altitude_degrees=min_elevation_deg)
        i = 0
        while i + 2 < len(events):
            if events[i] == 0 and events[i + 1] == 1 and events[i + 2] == 2:
                aos_t, culm_t, los_t = times[i], times[i + 1], times[i + 2]
                alt, _, _ = (sat - observer).at(culm_t).altaz()
                passes.append({
                    "key": key,
                    "label": sat_info["label"],
                    "freq_hz": sat_info["freq_hz"],
                    "aos": aos_t.utc_iso(),
                    "los": los_t.utc_iso(),
                    "max_elevation_deg": round(alt.degrees, 1),
                    "duration_s": round((los_t.tt - aos_t.tt) * 86400),
                })
                i += 3
            else:
                i += 1

    passes.sort(key=lambda p: p["aos"])
    return passes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=48.0,
                         help="prediction window in hours (default: 48)")
    parser.add_argument("--min-elevation", type=float, default=5.0,
                         help="minimum peak elevation in degrees to list a pass (default: 5)")
    args = parser.parse_args()

    passes = upcoming_passes(args.hours, args.min_elevation)
    if not passes:
        print("No passes found -- check the TLE cache and satellites.py's catalog_name entries.",
              file=sys.stderr)
        return
    for p in passes:
        aos_local = datetime.fromisoformat(p["aos"].replace("Z", "+00:00")).astimezone()
        print(f"{aos_local:%Y-%m-%d %H:%M} local  {p['label']:<20} "
              f"max_el={p['max_elevation_deg']:>5.1f}deg  dur={p['duration_s']}s  "
              f"freq={p['freq_hz']/1e6:.3f} MHz")


if __name__ == "__main__":
    main()
