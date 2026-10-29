#!/usr/bin/env bash
# The spend guard (no agent): run a command (a `modal run`) while watching Modal's spend for the day.
#
#   spend_guard.sh LOG BASELINE COST_SCRIPT -- COMMAND...
#
# BASELINE holds the day's spend before the first run (written if absent, so later runs share it). Every 5 minutes
# while COMMAND runs, and once when it ends, the spend over the baseline is appended to LOG; passing $3 (the guide) is
# noted once; past $20 (a runaway stop) COMMAND is killed. COST_SCRIPT prints "Modal spend for 'today': $X".
set -u
log=$1 baseline_file=$2 cost=$3
shift 4
spend() { python3 "$cost" --for today 2>/dev/null | sed -n 's/^Modal spend[^$]*\$\([0-9.]*\).*/\1/p'; }
over() { python3 -c "import sys; print(f'{float(sys.argv[1] or 0) - float(sys.argv[2] or 0):.4f}')" "$1" "$2"; }
past() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$1" "$2"; }
[ -s "$baseline_file" ] || spend > "$baseline_file"
base=$(cat "$baseline_file")
echo "$(date -Is) start, baseline \$${base}: $*" >> "$log"
"$@" &
pid=$!
echo "$(date -Is) command pid $pid" >> "$log"
noted=0
check() {
  local now extra
  now=$(spend)
  extra=$(over "$now" "$base")
  echo "$(date -Is) spend over baseline \$${extra} ($1)" >> "$log"
  if [ "$noted" = 0 ] && past "$extra" 3; then echo "$(date -Is) passed \$3, the guide" >> "$log"; noted=1; fi
  if past "$extra" 20 && kill -0 "$pid" 2>/dev/null; then
    echo "$(date -Is) past \$20: killing $pid" >> "$log"
    kill "$pid"; sleep 15; kill -9 "$pid" 2>/dev/null
  fi
}
while kill -0 "$pid" 2>/dev/null; do
  for _ in $(seq 60); do sleep 5; kill -0 "$pid" 2>/dev/null || break; done
  kill -0 "$pid" 2>/dev/null && check running
done
wait "$pid"
status=$?
check "finished, exit $status"
exit "$status"
