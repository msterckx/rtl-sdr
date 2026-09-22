#!/usr/bin/env python3
"""Identify which shortwave broadcaster is likely on a given frequency, by
cross-referencing against EiBi's freely published shortwave schedule
(https://www.eibispace.de/dx/) -- frequency + UTC time + day of week against
a season's worth of known broadcasts.

Unlike FM (see the RDS decoder, once it exists), AM/shortwave carries no
station identifier in the signal itself -- this is the standard workaround
the shortwave-listening (SWL) hobby has always used: look up what's
*scheduled* to be there and use your own ears (language, content) to confirm
which of the (often several) candidates it actually is. See EiBi's own
README for this same caveat: "The ONLY 100% ID is that which you hear on the
radio yourself."

Usage:
    scripts/shortwave_id.py --freq-khz 13700
    scripts/shortwave_id.py --freq-khz 13700 --time 1830 --day We
"""

import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

EIBI_BASE_URL = "https://www.eibispace.de/dx"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "output" / "shortwave" / "eibi_schedule.csv"

# eibispace.de's TLS cert has lapsed as of this writing (verified: every
# fetch_schedule() call below fails cert verification, not just an
# occasional blip) -- EiBi is a small hobbyist site, not a target where
# unverified TLS carries real stakes, and the alternative is no schedule
# data at all. Revisit if/when their cert gets renewed.
_EIBI_SSL_CONTEXT = None


def _ssl_context():
    global _EIBI_SSL_CONTEXT
    if _EIBI_SSL_CONTEXT is None:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _EIBI_SSL_CONTEXT = ctx
    return _EIBI_SSL_CONTEXT


def current_season_code(when: datetime | None = None) -> str:
    """EiBi's 'AYY'/'BYY' season file-naming scheme: A = summer (roughly
    late March - late October), B = winter (late October - late March,
    named after the year it STARTS in, e.g. B25 spans Oct 2025-Mar 2026).
    The real transition dates track each hemisphere's DST changeover and
    move a few days year to year, so this is a calendar-month approximation
    -- exact enough except for a short window right at the boundary, where
    fetch_schedule()'s fallback to the previous season's file covers the
    case this guess picks a file EiBi hasn't published yet."""
    when = when or datetime.now(timezone.utc)
    if 4 <= when.month <= 9:
        return f"A{when.year % 100:02d}"
    year = when.year if when.month >= 10 else when.year - 1
    return f"B{year % 100:02d}"


def _previous_season_code(code: str) -> str:
    letter, year = code[0], int(code[1:])
    if letter == "A":
        return f"B{(year - 1) % 100:02d}"
    return f"A{year % 100:02d}"


def fetch_schedule(cache_path: Path = DEFAULT_CACHE_PATH, max_age_hours: float = 168.0) -> str:
    """Return the current season's EiBi schedule CSV text, refreshing from
    the network if the cache is missing or older than max_age_hours (default
    a week -- the schedule only changes seasonally, with occasional
    corrections). Never raises for a network failure if a cache already
    exists, same rationale as pass_predict.py's fetch_tles: a stale schedule
    still beats no schedule at all."""
    if cache_path.is_file():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return cache_path.read_text(encoding="utf-8", errors="replace")

    code = current_season_code()
    tried = [code, _previous_season_code(code)]
    last_exc = None
    for candidate in tried:
        url = f"{EIBI_BASE_URL}/sked-{candidate.lower()}.csv"
        try:
            with urllib.request.urlopen(url, timeout=30, context=_ssl_context()) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return text

    if cache_path.is_file():
        print(f"EiBi schedule refresh failed ({last_exc}), using stale cache", file=sys.stderr)
        return cache_path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError(f"could not fetch EiBi schedule (tried {tried}): {last_exc}")


_DAY_DIGITS = "1234567"  # Mon=1 .. Sun=7, EiBi's digit-list day convention
_DAY_ABBR = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]


def _day_matches(days_field: str, weekday: int) -> bool:
    """weekday: Python convention, Monday=0..Sunday=6. Empty field = daily.
    Handles EiBi's 'Mo,Tu,We' and digit-list '1245' forms; anything fancier
    (nth-weekday-of-month, 'alt', 'irr', specific calendar dates, etc.) is
    treated as a permissive match rather than excluded, since under- rather
    than over-filtering is the safer failure mode for an ID aid -- a
    schedule that says "maybe" is more useful here than one that silently
    drops a real candidate."""
    days_field = (days_field or "").strip()
    if not days_field:
        return True
    abbr = _DAY_ABBR[weekday]
    if abbr in days_field.split(","):
        return True
    if "-" in days_field and all(p in _DAY_ABBR for p in days_field.replace("-", ",").split(",")):
        lo, hi = days_field.split("-")
        lo_i, hi_i = _DAY_ABBR.index(lo), _DAY_ABBR.index(hi)
        return lo_i <= weekday <= hi_i
    if all(c in _DAY_DIGITS for c in days_field) and days_field:
        return str(weekday + 1) in days_field
    return True  # unrecognized special code (alt/irr/tent/nth-weekday/...) -- don't exclude


def _time_matches(time_field: str, hhmm: int) -> bool:
    """time_field is EiBi's 'HHMM-HHMM' UTC range; handles wrap past
    midnight (e.g. '2200-0600')."""
    try:
        start_s, end_s = time_field.split("-")
        start, end = int(start_s), int(end_s)
    except (ValueError, AttributeError):
        return True
    if start == 0 and end == 2400:
        return True
    if start <= end:
        return start <= hhmm < end
    return hhmm >= start or hhmm < end  # wraps past midnight


def parse_schedule(csv_text: str) -> list[dict]:
    """Parse EiBi's semicolon-separated sked-*.csv into a list of entry
    dicts. The first line is a column-width header for their own EiBiView
    tool (not data), detected and skipped by its non-numeric frequency
    field rather than assuming a fixed line count."""
    entries = []
    reader = csv.reader(io.StringIO(csv_text), delimiter=";")
    for row in reader:
        if len(row) < 9:
            continue
        try:
            freq_khz = float(row[0])
        except ValueError:
            continue  # header row, or a malformed line -- skip either way
        entries.append({
            "freq_khz": freq_khz,
            "time": row[1].strip(),
            "days": row[2].strip(),
            "country": row[3].strip(),
            "station": row[4].strip(),
            "language": row[5].strip(),
            "target": row[6].strip(),
            "site": row[7].strip(),
        })
    return entries


def lookup(schedule: list[dict], freq_hz: float, when_utc: datetime | None = None,
           tolerance_khz: float = 2.0) -> list[dict]:
    """Candidate broadcasts for freq_hz at when_utc (default: now), within
    tolerance_khz of the listed frequency (SDR tuning/display precision and
    minor transmitter drift both mean an exact-kHz match is too strict).
    Returns candidates sorted by how close their listed frequency is to the
    query, closest first -- multiple results are normal and expected (see
    this module's docstring); the caller picks based on language/content."""
    when_utc = when_utc or datetime.now(timezone.utc)
    freq_khz = freq_hz / 1000.0
    hhmm = when_utc.hour * 100 + when_utc.minute
    weekday = when_utc.weekday()

    matches = []
    for entry in schedule:
        if abs(entry["freq_khz"] - freq_khz) > tolerance_khz:
            continue
        if not _time_matches(entry["time"], hhmm):
            continue
        if not _day_matches(entry["days"], weekday):
            continue
        matches.append(entry)
    matches.sort(key=lambda e: abs(e["freq_khz"] - freq_khz))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freq-khz", type=float, required=True, help="frequency in kHz")
    parser.add_argument("--time", default=None,
                         help="UTC time as HHMM to check instead of right now (e.g. 1830)")
    parser.add_argument("--day", choices=_DAY_ABBR, default=None,
                         help="day of week to check instead of today (Mo/Tu/We/Th/Fr/Sa/Su)")
    parser.add_argument("--tolerance-khz", type=float, default=2.0,
                         help="how far from the listed frequency still counts as a match "
                              "(default: 2.0)")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH),
                         help=f"schedule cache path (default: {DEFAULT_CACHE_PATH})")
    args = parser.parse_args()

    when = datetime.now(timezone.utc)
    if args.time is not None:
        hh, mm = int(args.time[:-2] or 0), int(args.time[-2:])
        when = when.replace(hour=hh, minute=mm)
    if args.day is not None:
        target_weekday = _DAY_ABBR.index(args.day)
        when = when + timedelta(days=target_weekday - when.weekday())

    schedule = parse_schedule(fetch_schedule(Path(args.cache)))
    matches = lookup(schedule, args.freq_khz * 1000, when, args.tolerance_khz)

    if not matches:
        print(f"No scheduled broadcasts found near {args.freq_khz:.1f} kHz "
              f"at {when:%H:%M} UTC {_DAY_ABBR[when.weekday()]}.", file=sys.stderr)
        return
    print(f"{len(matches)} candidate(s) for {args.freq_khz:.1f} kHz "
          f"at {when:%H:%M} UTC {_DAY_ABBR[when.weekday()]}:")
    for m in matches:
        print(f"  {m['freq_khz']:>9.1f} kHz  {m['time']} UTC  {m['days'] or 'daily':<5}  "
              f"{m['station']:<28} lang={m['language']:<4} target={m['target']:<5} "
              f"country={m['country']:<4} site={m['site']}")


if __name__ == "__main__":
    main()
