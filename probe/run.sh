#!/usr/bin/env bash
# The hand-over's line in open boxes: the held solve at sqrt(K)/h 0.01 and 0.03 with the 0.1h interior on gate 2's four
# open cases (50 draws each), the forced and gap cases at the 0.01h interior's line, and the open vessel's leak.
set -euo pipefail
mkdir -p probe/out
export JAX_PLATFORMS=cpu
O="probe/out/${PROBE_ARM}.json"
case "$PROBE_ARM" in
  *_0p01h) python probe/line.py "${PROBE_ARM%_0p01h}" 0.01 0.01h "$O" ;;
  *_0p01) python probe/line.py "${PROBE_ARM%_0p01}" 0.01 0.1h "$O" ;;
  *_0p03) python probe/line.py "${PROBE_ARM%_0p03}" 0.03 0.1h "$O" ;;
  leak_*) python probe/leak.py "${PROBE_ARM#leak_}" "$O" ;;
esac
