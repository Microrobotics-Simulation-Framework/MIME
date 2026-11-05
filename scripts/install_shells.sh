#!/usr/bin/env bash
# zsh and fish for tests/cli/test_completion.py, on the CI shard that holds that file (.github/workflows/ci.yml).
#
# `apt-get install` takes 12 s at the median and, in 22 of 943 shard jobs of 20 runs, over 100 s, once 8 min 19 s and once
# 9 min 51 s, nearly all of it reading apt's package lists and dpkg's database before anything is fetched. Each attempt
# is therefore cut at SHELLS_ATTEMPT_SECONDS (120) and made again, SHELLS_ATTEMPTS (3) times, so that a runner whose first
# attempt stalls costs minutes and not the job's 20. Whether a second attempt on such a runner is quick is not known; the
# log says how long each took.
set -u

attempts=${SHELLS_ATTEMPTS:-3}
limit=${SHELLS_ATTEMPT_SECONDS:-120}

for attempt in $(seq 1 "$attempts"); do
  started=$SECONDS
  if sudo timeout --kill-after=10 "$limit" apt-get install -y --no-install-recommends zsh fish; then
    echo "zsh and fish installed on attempt $attempt of $attempts in $((SECONDS - started)) s"
    exit 0
  fi
  echo "::warning::apt-get install zsh fish: attempt $attempt of $attempts failed or ran over $limit s (ran $((SECONDS - started)) s)"
  sudo dpkg --configure -a || true
done
echo "::error::zsh and fish were not installed in $attempts attempts of $limit s"
exit 1
