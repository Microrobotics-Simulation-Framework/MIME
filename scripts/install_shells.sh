#!/usr/bin/env bash
# zsh and fish for tests/cli/test_completion.py, in the CI job that runs that file alone (.github/workflows/ci.yml).
#
# `apt-get install` takes 12 s at the median and, in 23 of 1043 shard jobs of 21 runs, over 120 s (121 to 320 s, each of
# them successful on its one attempt), once 8 min 19 s and once 9 min 51 s (both lost to the job's 20 minutes), nearly all
# of it reading apt's package lists and dpkg's database before anything is fetched. So:
#
#   - the first attempts are cut at SHELLS_ATTEMPT_SECONDS (120) and made again, after `dpkg --configure -a` (itself cut at
#     SHELLS_DPKG_SECONDS, 60) and a pause of SHELLS_PAUSE_SECONDS (15, which also rides out a held apt lock);
#   - the last attempt is cut at SHELLS_FINAL_SECONDS (360), above the longest install that passed (320 s), so that no
#     install that passed before is failed here: the cut only ever ends an attempt the history says is stalled.
#
# The worst case, printed first, is (attempt cut + 10 + dpkg cut + 10 + pause) x (attempts - 1) + final cut + 10 =
# (120 + 10 + 60 + 10 + 15) x 2 + 370 = 800 s (13.3 min). The job that runs it holds the completion tests alone
# (its limit is 30 minutes: tests/test_completion_job.py holds that above this worst case with a margin), where the step
# before this one ran beside a shard's 11 to 14 minutes of tests and lost the job to installs of 8 and 10 minutes.
# Whether a second attempt on a runner whose first stalled is quick is not known; the log says how long each took.
set -u

attempts=${SHELLS_ATTEMPTS:-3}
limit=${SHELLS_ATTEMPT_SECONDS:-120}
final=${SHELLS_FINAL_SECONDS:-360}
dpkg_limit=${SHELLS_DPKG_SECONDS:-60}
pause=${SHELLS_PAUSE_SECONDS:-15}
kill_after=10

echo "zsh and fish: worst case $(( (limit + kill_after + dpkg_limit + kill_after + pause) * (attempts - 1) + final + kill_after )) s"

for attempt in $(seq 1 "$attempts"); do
  cut=$limit
  [ "$attempt" -eq "$attempts" ] && cut=$final
  started=$SECONDS
  if sudo timeout --kill-after="$kill_after" "$cut" apt-get install -y --no-install-recommends zsh fish; then
    echo "zsh and fish installed on attempt $attempt of $attempts in $((SECONDS - started)) s"
    exit 0
  fi
  echo "::warning::apt-get install zsh fish: attempt $attempt of $attempts failed or ran over $cut s (ran $((SECONDS - started)) s)"
  if [ "$attempt" -lt "$attempts" ]; then
    sudo timeout --kill-after="$kill_after" "$dpkg_limit" dpkg --configure -a || true
    sleep "$pause"
  fi
done
echo "::error::zsh and fish were not installed in $attempts attempts"
exit 1
