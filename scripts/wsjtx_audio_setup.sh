#!/bin/bash
# Creates a persistent virtual audio cable ("wsjtx_in") that gqrx's
# demodulated audio can be routed into, and that WSJT-X can record from
# as its input device -- avoids needing a physical loopback cable.
#
# Idempotent: safe to run on every login (installed as a systemd --user
# service, see wsjtx-audio-loopback.service).
set -euo pipefail

SINK_NAME="wsjtx_in"

if ! pactl list sinks short | grep -q "\b${SINK_NAME}\b"; then
    pactl load-module module-null-sink \
        sink_name="${SINK_NAME}" \
        sink_properties=device.description=WSJTX_Input
fi
