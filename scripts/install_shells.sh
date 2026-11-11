#!/usr/bin/env bash
# zsh and fish for tests/cli/test_completion.py, in the CI job that runs that file alone (.github/workflows/ci.yml).
#
# `apt-get install` takes 12 s at the median and, in 23 of 1043 shard jobs of 21 runs, over 120 s (121 to 320 s, each of
# them successful on its one attempt), once 8 min 19 s and once 9 min 51 s (both lost to the job's 20 minutes), nearly all
# of it reading apt's package lists and dpkg's database before anything is fetched. So:
#
#   - the first attempt is cut at SHELLS_ATTEMPT_SECONDS (120) and made again, after `dpkg --configure -a` (itself cut at
#     SHELLS_DPKG_SECONDS, 60) and a pause of SHELLS_PAUSE_SECONDS (15, which also rides out a held apt lock);
#   - the last attempt is cut at SHELLS_FINAL_SECONDS (360), above the longest install that passed (320 s), so that no
#     install that passed before is failed here: the cut only ever ends an attempt the history says is stalled.
#
# The runner's configured sources (archive.ubuntu.com and its security and Azure mirrors) can be unreachable altogether: on
# 2026-11-11 the runners could not resolve any of them for hours, and no number of attempts at them installs anything. So
# when the attempts fail, or at once when one fails to resolve a host (a resolve failure is fast, not a stall, and the
# remaining attempts would meet the same answer), the same Ubuntu packages are installed from other Ubuntu mirrors, on
# other domains (SHELLS_FALLBACK_MIRRORS, in this order): apt is pointed (`-o Dir::Etc::sourcelist`) at that mirror's
# release, -updates and -security pockets alone, `apt-get update` reads only those, and zsh and fish are installed from
# there, each step cut (SHELLS_FALLBACK_UPDATE_SECONDS 60, SHELLS_FALLBACK_INSTALL_SECONDS 120). Nothing under /etc/apt is
# changed, so there is nothing to restore (the job ends), and the log names the source that installed the shells.
#
# The worst case, printed first, is the stall path, every cut reached:
#   (attempt cut + 10 + dpkg cut + 10 + pause) x (attempts - 1) + final cut + 10         = (120 + 10 + 60 + 10 + 15) + 370 = 585
#   + the repair of a half-done install before the fallback (dpkg cut + 10)               = 70
#   + (update cut + 10 + install cut + 10) x mirrors                                      = (60 + 10 + 120 + 10) x 2 = 400
#                                                                                         = 1055 s (17.6 min).
# The job that runs it holds the completion tests alone (its limit is 30 minutes: tests/test_completion_job.py holds that
# above this worst case with 10 minutes to spare), where the step before this one ran beside a shard's 11 to 14 minutes of
# tests and lost the job to installs of 8 and 10 minutes. The first source gets two attempts, not three: a third on the same
# source is worth less than the same time on another. Whether a second attempt on a runner whose first stalled is quick
# is not known; the log says how long each took.
set -u

attempts=${SHELLS_ATTEMPTS:-2}
limit=${SHELLS_ATTEMPT_SECONDS:-120}
final=${SHELLS_FINAL_SECONDS:-360}
dpkg_limit=${SHELLS_DPKG_SECONDS:-60}
pause=${SHELLS_PAUSE_SECONDS:-15}
mirrors=${SHELLS_FALLBACK_MIRRORS-https://mirrors.edge.kernel.org/ubuntu http://mirror.us.leaseweb.net/ubuntu}
update_limit=${SHELLS_FALLBACK_UPDATE_SECONDS:-60}
install_limit=${SHELLS_FALLBACK_INSTALL_SECONDS:-120}
kill_after=10
mirror_count=$(wc -w <<<"$mirrors")
resolve_failure='Temporary failure resolving|Could not resolve|Name or service not known'

echo "zsh and fish: worst case $(( (limit + kill_after + dpkg_limit + kill_after + pause) * (attempts - 1) + final + kill_after \
  + (mirror_count > 0 ? dpkg_limit + kill_after : 0) + (update_limit + kill_after + install_limit + kill_after) * mirror_count )) s"

log=$(mktemp)
trap 'rm -f "$log"' EXIT

# One apt-get run under its cut as root, its output shown and kept in $log; the status is apt's (124 or 137: the cut).
run_apt() {
  local cut=$1
  shift
  sudo timeout --kill-after="$kill_after" "$cut" apt-get "$@" >"$log" 2>&1
  local status=$?
  cat "$log"
  return "$status"
}

for attempt in $(seq 1 "$attempts"); do
  cut=$limit
  [ "$attempt" -eq "$attempts" ] && cut=$final
  started=$SECONDS
  if run_apt "$cut" install -y --no-install-recommends zsh fish; then
    echo "zsh and fish installed on attempt $attempt of $attempts in $((SECONDS - started)) s, from the runner's configured sources"
    exit 0
  fi
  echo "::warning::apt-get install zsh fish: attempt $attempt of $attempts failed or ran over $cut s (ran $((SECONDS - started)) s)"
  if grep -Eq "$resolve_failure" "$log"; then
    echo "::warning::the runner's configured sources do not resolve; the remaining attempts on them are skipped"
    break
  fi
  if [ "$attempt" -lt "$attempts" ]; then
    sudo timeout --kill-after="$kill_after" "$dpkg_limit" dpkg --configure -a || true
    sleep "$pause"
  fi
done

if [ -z "$mirrors" ]; then
  echo "::error::zsh and fish were not installed from the runner's configured sources and no fallback mirror is set"
  exit 1
fi

# The same packages from other Ubuntu mirrors: this release's release, -updates and -security pockets, main and universe.
codename=$(. "${SHELLS_OS_RELEASE:-/etc/os-release}" && echo "${VERSION_CODENAME:-}")
if [ -z "$codename" ]; then
  echo "::error::the release's codename is unknown, so no fallback mirror can be chosen"
  exit 1
fi
sudo timeout --kill-after="$kill_after" "$dpkg_limit" dpkg --configure -a || true
for mirror in $mirrors; do
  sources=$(mktemp)
  for pocket in "$codename" "$codename-updates" "$codename-security"; do
    echo "deb $mirror $pocket main universe" >>"$sources"
  done
  # Only this mirror's sources are read, and only its index files are fetched; the system's lists and sources stay as they were.
  options=(-o "Dir::Etc::sourcelist=$sources" -o "Dir::Etc::sourceparts=-" -o APT::Get::List-Cleanup=false
           -o Acquire::Languages=none -o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20)
  started=$SECONDS
  if run_apt "$update_limit" "${options[@]}" update \
      && run_apt "$install_limit" "${options[@]}" install -y --no-install-recommends zsh fish; then
    echo "zsh and fish installed from the fallback mirror $mirror ($codename, -updates and -security pockets) in $((SECONDS - started)) s"
    rm -f "$sources"
    exit 0
  fi
  echo "::warning::apt-get from the fallback mirror $mirror failed or ran over its cut (ran $((SECONDS - started)) s)"
  rm -f "$sources"
done
echo "::error::zsh and fish were not installed: $attempts attempts on the configured sources, then $mirror_count fallback mirrors"
exit 1
