"""Weather satellite catalog for pass prediction and scheduled decode
(pass_predict.py, satellite.py, webserver.py's Satellites tab).

NOAA POES 15/18/19 (the classic APT target) were all decommissioned in
mid-2025 and no longer transmit -- confirmed against NOAA's own OSPO status
page. The current real-world beginner target on the same 137 MHz band is
Meteor-M2, transmitting LRPT (a digital QPSK signal, decoded via satdump
rather than a simple FM/AM demod).

`catalog_name` is the substring matched against Celestrak's TLE name lines
(see pass_predict.py's fetch_tles) -- verify it against the actual fetched
TLE file if a satellite stops resolving, since Celestrak's naming isn't
perfectly consistent release to release. Meteor-M2-4 is currently the more
reliably active transmitter of the two; Meteor-M2-3 is kept here too but may
be off the air for stretches -- if upcoming_passes() comes back empty for it,
that's likely why, not a bug.
"""

SATELLITES = {
    "meteor_m2_4": {
        "label": "Meteor-M2-4 LRPT",
        "catalog_name": "METEOR-M2 4",
        "freq_hz": 137.9e6,
        # meteor_m2-x_lrpt (OQPSK, 72 kSym/s) is the M2-3/M2-4 pipeline --
        # meteor_m2_lrpt (plain QPSK) is for the older, now-dead M2/M2-2.
        # Confirmed against /usr/share/satdump/pipelines/Meteor-M.json's own
        # frequency table, which lists this pipeline's Primary/Backup as
        # exactly 137.9/137.1 MHz.
        "pipeline": "meteor_m2-x_lrpt",
    },
    "meteor_m2_3": {
        "label": "Meteor-M2-3 LRPT",
        "catalog_name": "METEOR-M2 3",
        "freq_hz": 137.1e6,
        "pipeline": "meteor_m2-x_lrpt",
    },
}
