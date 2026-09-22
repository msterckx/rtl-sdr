"""Shared receiver frequency/mode presets, used by the web UI (webserver.py)
and the frequency scanner (scan.py).

Presets are grouped into CATEGORIES so the UI and the scanner can offer
"scan this whole band" without listing every channel in one flat control.
Broadcast FM (87.5-108 MHz) is deliberately not included -- this is for
narrowband voice traffic, not music stations.

Frequencies below are centered on Antwerp, Belgium (EBAW / the Scheldt
port area):
  - atc: real published frequencies for Antwerp Deurne airport (EBAW),
    plus the international VHF air distress/guard channel.
  - marine: internationally standardized VHF marine channels. Antwerp's
    port and Scheldt VTS use specific working channels that vary by
    stretch of river -- these are the generic ITU calling/safety/port-ops
    channels as a starting point; add local working channels via
    --freqs or a new preset once you know them.
  - amateur: the international 2m/70cm FM calling (simplex) channels.
  - cb: the standard 40-channel CEPT/EU CB band plan (27 MHz, FM).

DMR (Digital Mobile Radio) is deliberately not modeled as a preset category
here: unlike the analog services above, there's no generic calling/simplex
channel -- DMR repeater and business-PMR frequencies are locally licensed
and vary by area, so there's nothing generic to list. Once you know a local
DMR frequency, listen to it with scripts/dmr.py directly (or the web UI's
"Custom..." band with mode DMR) rather than adding it as a "mode" here --
DMR needs dsd-fme's live decode pipeline, not the am/fm discriminator chain
record.py/scan.py use for these presets. DMR_REPEATERS below is just a
quick-pick list of known local repeaters' downlink (TX) frequencies for
that Custom/DMR UI field -- not part of PRESETS/GROUPS since scan.py's
scan-a-whole-category flow doesn't apply to DMR at all.
"""

CB_CHANNEL_MHZ = [
    26.965, 26.975, 26.985, 27.005, 27.015, 27.025, 27.035, 27.055, 27.065, 27.075,
    27.085, 27.105, 27.115, 27.125, 27.135, 27.155, 27.165, 27.175, 27.185, 27.205,
    27.215, 27.225, 27.255, 27.235, 27.245, 27.265, 27.275, 27.285, 27.295, 27.305,
    27.315, 27.325, 27.335, 27.345, 27.355, 27.365, 27.375, 27.385, 27.395, 27.405,
]

CATEGORY_LABELS = {
    "atc_ebaw": "EBAW Air Traffic Control (Antwerp)",
    "atc_ebbr": "EBBR Air Traffic Control (Brussels)",
    "pmr": "PMR446",
    "marine": "Marine VHF",
    "amateur": "Amateur Radio (2m/70cm)",
    "cb": "CB Radio (27 MHz)",
}

PRESETS = {
    "twr": {"label": "EBAW Tower (135.205 MHz, AM)", "freq": 135.205e6, "prefix": "EBAW_TWR", "mode": "am", "category": "atc_ebaw"},
    "gnd": {"label": "EBAW Ground (121.905 MHz, AM)", "freq": 121.905e6, "prefix": "EBAW_GND", "mode": "am", "category": "atc_ebaw"},
    "atis": {"label": "EBAW ATIS (124.205 MHz, AM)", "freq": 124.205e6, "prefix": "EBAW_ATIS", "mode": "am", "category": "atc_ebaw"},
    "app": {"label": "EBAW Approach (118.255 MHz, AM)", "freq": 118.255e6, "prefix": "EBAW_APP", "mode": "am", "category": "atc_ebaw"},
    "guard": {"label": "VHF Air Guard/Emergency (121.500 MHz, AM)", "freq": 121.500e6, "prefix": "GUARD", "mode": "am", "category": "atc_ebaw"},

    # Brussels Airport (EBBR / Zaventem), ~32 km south of EBAW -- well within
    # VHF line-of-sight range from Antwerp given typical approach altitudes.
    # UHF military frequencies (362.30, 369.20 MHz) are deliberately left
    # out -- different band, not civil AM airband.
    "ebbr_arr1": {"label": "EBBR Arrival 1 (118.250 MHz, AM)", "freq": 118.250e6, "prefix": "EBBR_ARR1", "mode": "am", "category": "atc_ebbr"},
    "ebbr_arr2": {"label": "EBBR Arrival 2 / Radar (120.100 MHz, AM)", "freq": 120.100e6, "prefix": "EBBR_ARR2", "mode": "am", "category": "atc_ebbr"},
    "ebbr_fin1": {"label": "EBBR Final Approach 1 (127.570 MHz, AM)", "freq": 127.570e6, "prefix": "EBBR_FIN1", "mode": "am", "category": "atc_ebbr"},
    "ebbr_fin2": {"label": "EBBR Final Approach 2 (129.730 MHz, AM)", "freq": 129.730e6, "prefix": "EBBR_FIN2", "mode": "am", "category": "atc_ebbr"},
    "ebbr_dep": {"label": "EBBR Departure (126.630 MHz, AM)", "freq": 126.630e6, "prefix": "EBBR_DEP", "mode": "am", "category": "atc_ebbr"},
    "ebbr_twr1": {"label": "EBBR Tower 1 (118.600 MHz, AM)", "freq": 118.600e6, "prefix": "EBBR_TWR1", "mode": "am", "category": "atc_ebbr"},
    "ebbr_twr2": {"label": "EBBR Tower 2 (120.780 MHz, AM)", "freq": 120.780e6, "prefix": "EBBR_TWR2", "mode": "am", "category": "atc_ebbr"},
    "ebbr_twr3": {"label": "EBBR Tower 3 (127.150 MHz, AM)", "freq": 127.150e6, "prefix": "EBBR_TWR3", "mode": "am", "category": "atc_ebbr"},
    "ebbr_gnd1": {"label": "EBBR Ground 1 (118.050 MHz, AM)", "freq": 118.050e6, "prefix": "EBBR_GND1", "mode": "am", "category": "atc_ebbr"},
    "ebbr_gnd2": {"label": "EBBR Ground 2 (121.700 MHz, AM)", "freq": 121.700e6, "prefix": "EBBR_GND2", "mode": "am", "category": "atc_ebbr"},
    "ebbr_gnd3": {"label": "EBBR Ground 3 (121.880 MHz, AM)", "freq": 121.880e6, "prefix": "EBBR_GND3", "mode": "am", "category": "atc_ebbr"},
    "ebbr_del": {"label": "EBBR Clearance Delivery (121.950 MHz, AM)", "freq": 121.950e6, "prefix": "EBBR_DEL", "mode": "am", "category": "atc_ebbr"},
    "ebbr_atis_dep": {"label": "EBBR Departure ATIS (121.750 MHz, AM)", "freq": 121.750e6, "prefix": "EBBR_ATIS_DEP", "mode": "am", "category": "atc_ebbr"},
    "ebbr_atis_arr1": {"label": "EBBR Arrival ATIS 1 (110.600 MHz, AM)", "freq": 110.600e6, "prefix": "EBBR_ATIS_ARR1", "mode": "am", "category": "atc_ebbr"},
    "ebbr_atis_arr2": {"label": "EBBR Arrival ATIS 2 (112.050 MHz, AM)", "freq": 112.050e6, "prefix": "EBBR_ATIS_ARR2", "mode": "am", "category": "atc_ebbr"},
    "ebbr_atis_arr3": {"label": "EBBR Arrival ATIS 3 (114.600 MHz, AM)", "freq": 114.600e6, "prefix": "EBBR_ATIS_ARR3", "mode": "am", "category": "atc_ebbr"},
    "ebbr_atis_arr4": {"label": "EBBR Arrival ATIS 4 (114.900 MHz, AM)", "freq": 114.900e6, "prefix": "EBBR_ATIS_ARR4", "mode": "am", "category": "atc_ebbr"},
    "ebbr_atis_arr5": {"label": "EBBR Arrival ATIS 5 (117.550 MHz, AM)", "freq": 117.550e6, "prefix": "EBBR_ATIS_ARR5", "mode": "am", "category": "atc_ebbr"},
    "ebbr_atis_arr6": {"label": "EBBR Arrival ATIS 6 (132.480 MHz, AM)", "freq": 132.480e6, "prefix": "EBBR_ATIS_ARR6", "mode": "am", "category": "atc_ebbr"},

    "pmr1": {"label": "PMR446 CH1 (446.006 MHz, FM)", "freq": 446.00625e6, "prefix": "PMR1", "mode": "fm", "category": "pmr"},
    "pmr2": {"label": "PMR446 CH2 (446.019 MHz, FM)", "freq": 446.01875e6, "prefix": "PMR2", "mode": "fm", "category": "pmr"},
    "pmr3": {"label": "PMR446 CH3 (446.031 MHz, FM)", "freq": 446.03125e6, "prefix": "PMR3", "mode": "fm", "category": "pmr"},
    "pmr4": {"label": "PMR446 CH4 (446.044 MHz, FM)", "freq": 446.04375e6, "prefix": "PMR4", "mode": "fm", "category": "pmr"},
    "pmr5": {"label": "PMR446 CH5 (446.056 MHz, FM)", "freq": 446.05625e6, "prefix": "PMR5", "mode": "fm", "category": "pmr"},
    "pmr6": {"label": "PMR446 CH6 (446.069 MHz, FM)", "freq": 446.06875e6, "prefix": "PMR6", "mode": "fm", "category": "pmr"},
    "pmr7": {"label": "PMR446 CH7 (446.081 MHz, FM)", "freq": 446.08125e6, "prefix": "PMR7", "mode": "fm", "category": "pmr"},
    "pmr8": {"label": "PMR446 CH8 (446.094 MHz, FM)", "freq": 446.09375e6, "prefix": "PMR8", "mode": "fm", "category": "pmr"},

    "marine6": {"label": "Marine CH06 - ship-to-ship safety (156.300 MHz, FM)", "freq": 156.300e6, "prefix": "MARINE_CH06", "mode": "fm", "category": "marine"},
    "marine9": {"label": "Marine CH09 - secondary calling (156.450 MHz, FM)", "freq": 156.450e6, "prefix": "MARINE_CH09", "mode": "fm", "category": "marine"},
    "marine12": {"label": "Marine CH12 - port operations (156.600 MHz, FM)", "freq": 156.600e6, "prefix": "MARINE_CH12", "mode": "fm", "category": "marine"},
    "marine13": {"label": "Marine CH13 - bridge-to-bridge navigation (156.650 MHz, FM)", "freq": 156.650e6, "prefix": "MARINE_CH13", "mode": "fm", "category": "marine"},
    "marine14": {"label": "Marine CH14 - port operations (156.700 MHz, FM)", "freq": 156.700e6, "prefix": "MARINE_CH14", "mode": "fm", "category": "marine"},
    "marine16": {"label": "Marine CH16 - distress/safety/calling (156.800 MHz, FM)", "freq": 156.800e6, "prefix": "MARINE_CH16", "mode": "fm", "category": "marine"},
    "marine18": {"label": "Marine CH18 - port operations (156.900 MHz, FM)", "freq": 156.900e6, "prefix": "MARINE_CH18", "mode": "fm", "category": "marine"},

    "ham2m": {"label": "2m FM Calling (145.500 MHz, FM)", "freq": 145.500e6, "prefix": "HAM_2M", "mode": "fm", "category": "amateur"},
    "ham70cm": {"label": "70cm FM Calling (433.500 MHz, FM)", "freq": 433.500e6, "prefix": "HAM_70CM", "mode": "fm", "category": "amateur"},
}

for _i, _mhz in enumerate(CB_CHANNEL_MHZ, start=1):
    PRESETS[f"cb{_i}"] = {
        "label": f"CB Channel {_i} ({_mhz:.3f} MHz, FM)",
        "freq": _mhz * 1e6,
        "prefix": f"CB{_i:02d}",
        "mode": "fm",
        "category": "cb",
    }
del _i, _mhz

GROUPS = {
    cat: [key for key, preset in PRESETS.items() if preset["category"] == cat]
    for cat in CATEGORY_LABELS
}

# Quick-pick band ranges for spectrum_scan.py's wideband survey (webserver.py's
# Spectrum tab), as opposed to PRESETS/GROUPS above which are individual
# channels for scan.py's per-channel squelch. Amateur is split into 2m/70cm
# since they're two disjoint bands nearly 300 MHz apart -- sweeping "amateur"
# as one range would waste most of the sweep on the dead space between them.
SPECTRUM_RANGES = {
    "all": {"label": "All bands (26-470 MHz)", "start_mhz": 26.0, "end_mhz": 470.0},
    "cb": {"label": "CB Radio (26.9-27.5 MHz)", "start_mhz": 26.9, "end_mhz": 27.5},
    "aviation": {"label": "Aviation (108-137 MHz)", "start_mhz": 108.0, "end_mhz": 137.0},
    "amateur_2m": {"label": "Amateur 2m (144-148 MHz)", "start_mhz": 144.0, "end_mhz": 148.0},
    "marine": {"label": "Marine VHF (156-163 MHz)", "start_mhz": 156.0, "end_mhz": 163.0},
    "pmr": {"label": "PMR446 (446.0-446.2 MHz)", "start_mhz": 446.0, "end_mhz": 446.2},
    "amateur_70cm": {"label": "Amateur 70cm (430-440 MHz)", "start_mhz": 430.0, "end_mhz": 440.0},
}

# Known Belgian DMR (Brandmeister BM206) repeaters within ~20 km of Antwerp,
# from the Brandmeister repeater directory (api.brandmeister.network/v2/device).
# Frequency given is each repeater's downlink/TX -- what to tune a receiver
# to, since dmr.py listens, it doesn't transmit. Personal MMDVM hotspots
# (ON3/ON4/etc. callsigns, milliwatt power) are deliberately excluded --
# only receivable from meters away, not useful as a regional preset. "ON0"
# is Belgium's dedicated repeater callsign block.
DMR_REPEATERS = {
    "on0an": {"label": "ON0AN Antwerp (439.350 MHz)", "freq": 439.350e6},
    "on0kb": {"label": "ON0KB Bornem (438.925 MHz)", "freq": 438.925e6},
    "on0snw": {"label": "ON0SNW Sint-Niklaas (431.800 MHz)", "freq": 431.800e6},
    "on0wal": {"label": "ON0WAL Walem (439.075 MHz, multimode DMR/YSF/FM/D-STAR)", "freq": 439.075e6},
}
