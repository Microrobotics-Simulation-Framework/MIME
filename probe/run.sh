#!/usr/bin/env bash
# The contact-loops pull request's plants (a loop of 7 steps, a carry from two steps back, a stop at 1e-5 of the hull's size)
# rated under the pull request's rule and the base's (arms cyl_*: cylinder draws, mix_*: all kinds), the clean arm hunted
# for false flags (hunt_*), and two flagged draws replayed (replay_*).
set -euo pipefail
mkdir -p probe/out
export OPENBLAS_NUM_THREADS=1
case "$PROBE_ARM" in
  replay_*) python probe/replay.py ;;
  *) python probe/rates.py ;;
esac
