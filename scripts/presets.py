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
"""

CB_CHANNEL_MHZ = [
    26.965, 26.975, 26.985, 27.005, 27.015, 27.025, 27.035, 27.055, 27.065, 27.075,
    27.085, 27.105, 27.115, 27.125, 27.135, 27.155, 27.165, 27.175, 27.185, 27.205,
    27.215, 27.225, 27.255, 27.235, 27.245, 27.265, 27.275, 27.285, 27.295, 27.305,
    27.315, 27.325, 27.335, 27.345, 27.355, 27.365, 27.375, 27.385, 27.395, 27.405,
]

CATEGORY_LABELS = {
    "atc": "EBAW Air Traffic Control",
    "pmr": "PMR446",
    "marine": "Marine VHF",
    "amateur": "Amateur Radio (2m/70cm)",
    "cb": "CB Radio (27 MHz)",
}

PRESETS = {
    "twr": {"label": "EBAW Tower (135.205 MHz, AM)", "freq": 135.205e6, "prefix": "EBAW_TWR", "mode": "am", "category": "atc"},
    "gnd": {"label": "EBAW Ground (121.905 MHz, AM)", "freq": 121.905e6, "prefix": "EBAW_GND", "mode": "am", "category": "atc"},
    "atis": {"label": "EBAW ATIS (124.205 MHz, AM)", "freq": 124.205e6, "prefix": "EBAW_ATIS", "mode": "am", "category": "atc"},
    "app": {"label": "EBAW Approach (118.255 MHz, AM)", "freq": 118.255e6, "prefix": "EBAW_APP", "mode": "am", "category": "atc"},
    "guard": {"label": "VHF Air Guard/Emergency (121.500 MHz, AM)", "freq": 121.500e6, "prefix": "GUARD", "mode": "am", "category": "atc"},

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
