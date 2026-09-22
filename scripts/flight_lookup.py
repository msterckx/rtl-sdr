#!/usr/bin/env python3
"""Look up the real flight (callsign, position, aircraft) behind an ATC
transmission, by cross-referencing the recording's timestamp and a spoken
airline name against OpenSky Network's ADS-B data.

ATC callsigns in the transcripts are unreliable -- noisy AM audio plus a
Whisper model that isn't fully tuned to aviation phraseology often drops or
garbles the flight number -- but the airline name and the transmission's
timestamp usually come through fine. That's enough to narrow down which
aircraft near EBAW (Antwerp Deurne) was transmitting.

Usage:
    scripts/flight_lookup.py output/EBAW_APP_20260906_095045_081.txt
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Antwerp Deurne (EBAW) reference point, used to build a bounding box for
# the OpenSky query. Approach traffic can be tens of km out; Tower/Ground
# traffic is on the field. The default radius is sized for Approach.
EBAW_LAT = 51.1894
EBAW_LON = 4.4603

# Matches datetime.now() in record.py's write_wav -- filenames encode local
# (not UTC) time.
RECORDING_TZ = ZoneInfo("Europe/Brussels")

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)

# Airline (as it's likely to be said/transcribed) -> ICAO callsign prefix.
# Extend as you recognize more operators in EBAW traffic.
AIRLINE_PREFIXES = {
    "lufthansa": "DLH",
    "klm": "KLM",
    "brussels": "BEL",
    "air france": "AFR",
    "airfrance": "AFR",
    "eurowings": "EWG",
    "swiss": "SWR",
    "austrian": "AUA",
    "ryanair": "RYR",
    "easyjet": "EZY",
    "wizz": "WZZ",
    "vueling": "VLG",
    "tui": "TUI",
    "jet2": "EXS",
    "scandinavian": "SAS",
    "sas": "SAS",
    "scan": "SAS",
    "turkish": "THY",
    "iberia": "IBE",
    "alitalia": "AZA",
    "ita": "ITY",
    "cityjet": "CJT",
    "vlm": "VLM",
    "netjets": "NJE",
    "dhl": "DHK",
    "fedex": "FDX",
    "ups": "UPS",
}

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9",
}

FILENAME_TS_RE = re.compile(r"(\d{8})_(\d{6})_(\d{3})")

EARTH_RADIUS_KM = 6371.0

# Deurne field elevation, for slant range (not just ground distance) -- small
# next to typical approach altitudes but keeps close-in/on-ground geometry sane.
EBAW_ELEV_M = 12.0

STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]


def parse_recording_time(path: Path) -> datetime:
    """Extract the local recording start time encoded in a WAV/TXT filename
    (YYYYMMDD_HHMMSS_mmm, as written by write_wav in record.py) and return
    it as a timezone-aware UTC datetime."""
    m = FILENAME_TS_RE.search(path.stem)
    if not m:
        raise ValueError(f"no timestamp found in filename: {path.name}")
    date_s, time_s, _ms = m.groups()
    naive = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S")
    local = naive.replace(tzinfo=RECORDING_TZ)
    return local.astimezone(ZoneInfo("UTC"))


def slant_range_km(lat: float, lon: float, altitude_m: float | None) -> float:
    """Great-circle ground distance from EBAW to (lat, lon) via the
    haversine formula, combined with the altitude difference into a 3D
    slant range -- the actual straight-line distance the VHF signal
    traveled, which is what a radio-horizon check needs (ground distance
    alone understates range for aircraft several km up).

    `altitude_m` is OpenSky's barometric altitude and can be missing
    (aircraft on ground, or state not reported); treated as field
    elevation in that case.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, (EBAW_LAT, EBAW_LON, lat, lon))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    ground_km = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
    alt_m = altitude_m if altitude_m is not None else EBAW_ELEV_M
    height_km = (alt_m - EBAW_ELEV_M) / 1000.0
    return math.hypot(ground_km, height_km)


def extract_airline_and_digits(transcript: str):
    """Best-effort scan of a transcript for a known airline name, and any
    spoken digits immediately following it (e.g. "Lufthansa seven zero" ->
    ("lufthansa", "DLH", "70")). Digits are frequently where Whisper drops
    or garbles output, so this is a ranking hint, not a hard filter."""
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    for i, word in enumerate(words):
        prefix = AIRLINE_PREFIXES.get(word)
        if prefix is None and i + 1 < len(words):
            prefix = AIRLINE_PREFIXES.get(f"{word} {words[i + 1]}")
        if prefix is None:
            continue
        digits = []
        for w in words[i + 1: i + 7]:
            if w in DIGIT_WORDS:
                digits.append(DIGIT_WORDS[w])
            elif digits:
                break
        return word, prefix, "".join(digits) or None
    return None, None, None


def opensky_token(client_id: str, client_secret: str) -> str:
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(OPENSKY_TOKEN_URL, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["access_token"]


def query_opensky(epoch: int, radius_km: float, token: str | None) -> list:
    """Query OpenSky's states/all endpoint. The `time` parameter (for a
    historical snapshot) is only honored for authenticated requests -- the
    API returns 403 if an anonymous request includes it -- so anonymous
    callers get the current live snapshot instead, regardless of `epoch`."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.1, abs(math.cos(math.radians(EBAW_LAT)))))
    params = {
        "lamin": EBAW_LAT - dlat,
        "lamax": EBAW_LAT + dlat,
        "lomin": EBAW_LON - dlon,
        "lomax": EBAW_LON + dlon,
    }
    if token:
        params["time"] = epoch
    url = f"{OPENSKY_STATES_URL}?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return body.get("states") or []


def locate_flight(txt_path: Path, radius_km: float = 50.0,
                   client_id: str | None = None, client_secret: str | None = None) -> dict:
    """Core lookup used by both the CLI below and webserver.py's /api/locate
    endpoint. Never raises for expected failure modes (no airline matched,
    OpenSky unreachable/rate-limited) -- those come back as an "error" key
    instead, since a bad transcript or a network hiccup shouldn't be fatal
    to a page load."""
    client_id = client_id or os.environ.get("OPENSKY_CLIENT_ID")
    client_secret = client_secret or os.environ.get("OPENSKY_CLIENT_SECRET")

    transcript = txt_path.read_text().strip()
    rec_time_utc = parse_recording_time(txt_path)
    epoch = int(rec_time_utc.timestamp())
    age_s = datetime.now(ZoneInfo("UTC")).timestamp() - epoch

    result = {
        "recording_time_utc": rec_time_utc.isoformat(),
        "age_minutes": round(age_s / 60, 1),
        "transcript": transcript,
        "airline": None,
        "prefix": None,
        "digits": None,
        "authenticated": False,
        "candidates": [],
        "warning": None,
        "error": None,
    }

    airline_word, prefix, digits = extract_airline_and_digits(transcript)
    if prefix is None:
        result["error"] = ("no known airline name recognized in transcript -- "
                            "add it to AIRLINE_PREFIXES in flight_lookup.py")
        return result
    result["airline"], result["prefix"], result["digits"] = airline_word, prefix, digits

    token = None
    if client_id and client_secret:
        try:
            token = opensky_token(client_id, client_secret)
            result["authenticated"] = True
        except (urllib.error.URLError, OSError, KeyError) as exc:
            result["warning"] = f"OpenSky OAuth token request failed ({exc}), using anonymous access"
    else:
        result["warning"] = ("no OpenSky credentials set -- anonymous requests only return "
                              "the current live snapshot, not a historical one at the "
                              "recording's timestamp")

    try:
        states = query_opensky(epoch, radius_km, token)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            result["error"] = (f"OpenSky request failed ({exc.code}): anonymous access is "
                                f"rate-limited and time-restricted -- set OPENSKY_CLIENT_ID/"
                                f"OPENSKY_CLIENT_SECRET")
        else:
            result["error"] = f"OpenSky request failed: {exc}"
        return result
    except (urllib.error.URLError, OSError) as exc:
        result["error"] = f"OpenSky request failed: {exc}"
        return result

    candidates = [dict(zip(STATE_FIELDS, raw)) for raw in states
                  if (raw[1] or "").strip().startswith(prefix)]
    for s in candidates:
        lat, lon = s.get("latitude"), s.get("longitude")
        s["range_km"] = (round(slant_range_km(lat, lon, s.get("baro_altitude")), 1)
                          if lat is not None and lon is not None else None)
    if digits:
        candidates.sort(key=lambda s: digits not in (s.get("callsign") or ""))
    result["candidates"] = candidates
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcript", type=Path, help="path to a .txt transcript (or its .wav)")
    parser.add_argument("--radius-km", type=float, default=50.0,
                         help="bounding box half-width around EBAW to search (default: 50)")
    parser.add_argument("--client-id", default=None,
                         help="OpenSky OAuth2 client id (default: $OPENSKY_CLIENT_ID)")
    parser.add_argument("--client-secret", default=None,
                         help="OpenSky OAuth2 client secret (default: $OPENSKY_CLIENT_SECRET)")
    args = parser.parse_args()

    path = args.transcript
    txt_path = path if path.suffix == ".txt" else path.with_suffix(".txt")
    if not txt_path.exists():
        print(f"error: {txt_path} not found", file=sys.stderr)
        sys.exit(1)

    r = locate_flight(txt_path, args.radius_km, args.client_id, args.client_secret)

    print(f"Recording time: {r['recording_time_utc']} ({r['age_minutes']} min ago)", file=sys.stderr)
    print(f"Transcript: {r['transcript']}", file=sys.stderr)
    if r["warning"]:
        print(f"warning: {r['warning']}", file=sys.stderr)
    if r["error"]:
        print(r["error"], file=sys.stderr)
        sys.exit(1)
    print(f"Matched airline: {r['airline']!r} -> ICAO prefix {r['prefix']!r}"
          + (f", spoken digits: {r['digits']!r}" if r["digits"] else ""), file=sys.stderr)

    candidates = r["candidates"]
    if not candidates:
        print(f"No {r['prefix']} flights found near EBAW within {args.radius_km} km "
              f"at that time.", file=sys.stderr)
        return

    print(f"\n{len(candidates)} candidate(s):")
    for s in candidates:
        range_s = f"{s['range_km']}km" if s.get("range_km") is not None else "?"
        print(f"  callsign={s.get('callsign', '').strip():<10} icao24={s.get('icao24')}  "
              f"range={range_s:<8} alt={s.get('baro_altitude')}m  speed={s.get('velocity')}m/s  "
              f"track={s.get('true_track')}deg  pos=({s.get('latitude')},{s.get('longitude')})  "
              f"origin={s.get('origin_country')}")
    print("\nLook up icao24 (the aircraft's 24-bit hex address) on "
          "https://opensky-network.org/aircraft-database or "
          "https://www.flightradar24.com to get registration/type/route.")


if __name__ == "__main__":
    main()
