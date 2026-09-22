#!/usr/bin/env python3
"""Background monitor for Belgian Brandmeister DMR network activity, backed
by the community-run ham-dmr.be dashboard rather than Brandmeister's own
site.

This is unrelated to dmr.py's local RF decode -- it doesn't touch the SDR
at all, it just tracks the Brandmeister *network* (internet-linked DMR)
so the web UI can show a rolling history of Belgian talkgroup activity.

WHY HAM-DMR.BE, NOT BRANDMEISTER DIRECTLY: Brandmeister's own real-time
feed (wss://api.brandmeister.network/lh/) can be connected to with the
exact protocol a real browser uses -- same path, same Origin/User-Agent
headers, same Socket.IO handshake -- and still receives zero events over
a script, while a genuine browser tab gets several a second. That points
to fingerprinting below the HTTP layer (TLS/JA3 or similar), which isn't
something to try to defeat from a script. ham-dmr.be
(https://www.ham-dmr.be/, the official Belgian BM206 community dashboard
run by ON3YH/ON7LDS) runs its own server-side connection to Brandmeister
and republishes recent calls as plain JSON with no such protection, so
this polls that instead:
    http://dashboard.ham-dmr.be/DBlastheard.php

That endpoint returns roughly the last ~7 completed calls network-wide
for the BM206 (Belgium) master, refreshed continuously -- not filtered to
Belgian talkgroups server-side, so filtering happens here. It's an
undocumented endpoint of a small volunteer-run site (found by reading
that dashboard's own lastheard.js), not a stable public API, so this
polls it politely (POLL_INTERVAL_S) rather than hammering it.
"""

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

# Belgian Brandmeister talkgroups (network BM206). TG206 itself is the
# little-used national group; most real traffic is on the two regional
# groups or the six on-demand groups. See
# https://wiki.brandmeister.network/index.php/TalkGroup/206 and
# https://www.ham-dmr.be/2022/01/28/brandmeister-belgium-and-tg206/.
BELGIAN_TALKGROUPS = {
    206: "Belgium",
    2061: "Flanders",
    2062: "Wallonia",
    2064: "OnDemand 1",
    2065: "OnDemand 2",
    2066: "OnDemand 3",
    2067: "OnDemand 4",
    2068: "OnDemand 5",
    2069: "OnDemand 6",
}

LASTHEARD_URL = "http://dashboard.ham-dmr.be/DBlastheard.php"
POLL_INTERVAL_S = 15
REQUEST_TIMEOUT_S = 10
BRUSSELS_TZ = ZoneInfo("Europe/Brussels")

MAX_AGE_HOURS = 24
MAX_CALLS = 5000


class BrandmeisterMonitor:
    """Polls ham-dmr.be's last-heard feed forever in a background thread
    (run_forever, meant to be started as a daemon thread once at server
    startup), keeping a rolling, disk-persisted log of Belgian-talkgroup
    calls. calls_since() queries it."""

    def __init__(self, log_path: Path, talkgroups: dict[int, str] = BELGIAN_TALKGROUPS):
        self._log_path = log_path
        self._talkgroups = talkgroups
        self._lock = threading.Lock()
        self._calls = deque(maxlen=MAX_CALLS)
        self._last_poll_at = None
        self._last_poll_ok = None
        self._error = None
        self._load()

    def _load(self) -> None:
        if not self._log_path.is_file():
            return
        try:
            data = json.loads(self._log_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        cutoff = time.time() - MAX_AGE_HOURS * 3600
        with self._lock:
            self._calls.extend(c for c in data if c.get("stop", 0) >= cutoff)

    def _save_locked(self) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text(json.dumps(list(self._calls)))
        except OSError:
            pass

    def status(self) -> dict:
        with self._lock:
            return {
                "call_count": len(self._calls),
                "last_poll_at": self._last_poll_at,
                "last_poll_ok": self._last_poll_ok,
                "error": self._error,
            }

    def talkgroups(self) -> dict[int, str]:
        return dict(self._talkgroups)

    def calls_since(self, hours: float, talkgroup: int | None = None) -> list[dict]:
        cutoff = time.time() - hours * 3600
        with self._lock:
            calls = [c for c in self._calls if c["stop"] >= cutoff]
        if talkgroup is not None:
            calls = [c for c in calls if c["destination_id"] == talkgroup]
        return sorted(calls, key=lambda c: c["stop"], reverse=True)

    def _parse_entry(self, raw: dict) -> dict | None:
        try:
            dest_id = int(raw["dst"])
        except (KeyError, TypeError, ValueError):
            return None
        if dest_id not in self._talkgroups:
            return None
        try:
            duration = float(raw["duur"])
        except (KeyError, TypeError, ValueError):
            return None
        if duration < 0:
            return None  # still in progress -- ham-dmr.be marks ongoing calls this way
        try:
            start = (datetime.strptime(raw["start"], "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=BRUSSELS_TZ).timestamp())
        except (KeyError, TypeError, ValueError):
            return None
        source_id = raw.get("src")
        try:
            source_id = int(source_id)
        except (TypeError, ValueError):
            pass
        return {
            "start": start,
            "stop": start + duration,
            "duration": duration,
            "source_call": (raw.get("scall") or "").strip(),
            "source_name": (raw.get("talker-alias") or "").strip(),
            "source_id": source_id,
            "destination_id": dest_id,
            "destination_name": self._talkgroups.get(dest_id) or (raw.get("dcall") or ""),
            "slot": raw.get("slot"),
            # Which repeater/hotspot actually relayed this call -- e.g. "ON0AN"
            # means it really did go out over that repeater's RF, vs. most
            # BM206 traffic which is hotspot-to-hotspot over the internet and
            # never touches any physical repeater at all.
            "via": (raw.get("vianaam") or "").strip(),
        }

    def _poll_once(self) -> None:
        resp = requests.get(LASTHEARD_URL, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        entries = resp.json()
        added = 0
        with self._lock:
            existing_keys = {
                (c["start"], c["destination_id"], c.get("source_id")) for c in self._calls
            }
            for raw in entries:
                call = self._parse_entry(raw)
                if call is None:
                    continue
                key = (call["start"], call["destination_id"], call["source_id"])
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                self._calls.append(call)
                added += 1
            if added:
                self._save_locked()

    def run_forever(self) -> None:
        """Never returns -- polls at POLL_INTERVAL_S regardless of success
        or failure, so a transient network error just delays the next
        successful update rather than stopping polling altogether."""
        while True:
            try:
                self._poll_once()
                with self._lock:
                    self._last_poll_ok = True
                    self._error = None
            except Exception as exc:  # background thread must never die silently
                with self._lock:
                    self._last_poll_ok = False
                    self._error = str(exc)
            with self._lock:
                self._last_poll_at = time.time()
            time.sleep(POLL_INTERVAL_S)
