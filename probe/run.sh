#!/usr/bin/env bash
# Spike spike/tie-chatter: one arm (PROBE_ARM) of the tie-window chatter measurement. See scripts/spikes/tie_chatter/.
set -u
mkdir -p probe/out
export JAX_PLATFORMS=cpu PROBE_OUT=probe/out
timeout 100m python scripts/spikes/tie_chatter/run_arm.py "$PROBE_ARM" > probe/out/log.txt 2>&1
rc=$?
tail -5 probe/out/log.txt
exit $rc
