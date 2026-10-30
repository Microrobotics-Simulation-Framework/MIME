"""The launch wrapper for ``m3b_modal.py``: starts ``modal run --detach`` at most 3 times, never while an app named
``mime-7ja-m3b`` is running or while the results Volume holds ``started.json``, ``done.json`` or ``failed.json``, with
backoffs of 60 s and 120 s between attempts, and polls until the run is done. It only reads state through the modal CLI
and never stops an app (the tracker does that).

    python3 m3b_launch.py --out DIR --log FILE     (MIME_7JA_WHEEL, MIME_7JA_REQUIREMENTS, MIME_7JA_TABLE set)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

APP, VOLUME = "mime-7ja-m3b", "mime-7ja-m3-results"
MAX_ATTEMPTS, BACKOFFS, POLL_S, GRACE_S = 3, (60, 120), 30, 60
HERE = Path(__file__).resolve().parent


def say(log: Path, text: str) -> None:
    line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {text}"
    print(line, flush=True)
    with log.open("a") as handle:
        handle.write(line + "\n")


def app_running() -> bool | None:
    """Whether an app named ``APP`` is running; None when the CLI could not say (a network failure)."""
    try:
        done = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            return None
        return any(r.get("description") == APP and not str(r.get("state", "")).lower().startswith("stop")
                   for r in json.loads(done.stdout or "[]"))
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def volume_files() -> set[str] | None:
    """The Volume's top-level files; empty when the Volume does not exist yet; None when the CLI could not say."""
    try:
        done = subprocess.run(["modal", "volume", "ls", VOLUME, "/", "--json"], capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            text = (done.stdout + done.stderr).lower()
            return set() if "not found" in text or "no volume" in text else None
        return {Path(str(r.get("Filename") or r.get("filename") or r.get("path") or "")).name
                for r in json.loads(done.stdout or "[]")}
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    log = Path(args.log)
    command = ["modal", "run", "--detach", str(HERE / "m3b_modal.py"), "--out", args.out]
    attempts, process, exited_at = 0, None, None
    while True:
        running, files = app_running(), volume_files()
        if running is None or files is None:
            say(log, f"state unknown (app list {running}, volume {files}); waiting")
            time.sleep(POLL_S)
            continue
        if "done.json" in files or "failed.json" in files:
            say(log, f"finished: volume holds {sorted(files)}; launch attempts {attempts}")
            return 0 if "done.json" in files else 2
        if process is not None and process.poll() is not None and exited_at is None:
            exited_at = time.time()
            say(log, f"launch process {attempts} exited with {process.returncode}")
        alive = process is not None and process.poll() is None
        if running or alive or "started.json" in files:
            say(log, f"in progress: app running {running}, local process alive {alive}, volume {sorted(files)}")
            if not running and not alive and "started.json" in files and exited_at and time.time() - exited_at > 600:
                say(log, "started.json present, nothing running and no done.json for 10 minutes: stopping, no relaunch")
                return 3
            time.sleep(POLL_S)
            continue
        if exited_at is not None and time.time() - exited_at < GRACE_S:
            time.sleep(10)
            continue
        if attempts >= MAX_ATTEMPTS:
            say(log, f"no run after {attempts} launch attempts: giving up")
            return 4
        if attempts > 0:
            wait = BACKOFFS[attempts - 1]
            say(log, f"attempt {attempts} left no run; backing off {wait} s")
            time.sleep(wait)
            if app_running() or "started.json" in (volume_files() or set()):
                continue
        attempts += 1
        say(log, f"launching attempt {attempts}: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=open(log.with_suffix(f".modal{attempts}.log"), "w"), stderr=subprocess.STDOUT)
        exited_at = None
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
