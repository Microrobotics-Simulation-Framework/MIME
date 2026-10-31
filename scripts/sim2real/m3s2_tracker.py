"""The spend tracker for M3's two Modal apps (``mime-m3-rerun`` on an A100-40GB, ``mime-m3-stage2`` on CPU), adapted from
the cap fix's. Runs locally (it dies with the network: the functions' 1800 s ``timeout`` is the real backstop). Every
90 s it lists the account's apps and containers, keeps only these two apps' (absent from the start list), logs a line and a
live upper bound on the combined spend (each container's age times its app's hourly rate, including finished
containers), and every 10 minutes reads the billed spend for the two apps. Guide $2 (a warning), kill $5 on the
combined figure.

It stops the runs (``modal app stop`` on their app ids only) when the live bound or the billed spend reaches $5, a
container has run over 2100 s, more than one container of an app runs at once, or 3 hours have passed. It ends when
``DONE`` exists in its folder and no app of the runs is running (a run still up 5 minutes after ``DONE`` is stopped).

    python3 m3s2_tracker.py FOLDER
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

APPS = {"mime-m3-rerun": 2.10 + 4 * 0.0473 + 16 * 0.008, "mime-m3-stage2": 32 * 0.0473 + 64 * 0.008}  # $/h, rates read 2026-10-31
GUIDE, KILL = 2.0, 5.0
CONTAINER_LIMIT_S, TOTAL_S, DONE_GRACE_S = 2100.0, 10800.0, 300.0
INTERVAL_S, BILLED_EVERY_S = 90.0, 600.0
HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
LOG, DONE_FILE = HERE / "tracker_log.txt", HERE / "DONE"


def say(text: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}Z {text}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def modal_json(*args: str) -> list[dict]:
    try:
        done = subprocess.run(["modal", *args, "--json"], capture_output=True, text=True, timeout=60)
        return json.loads(done.stdout) if done.returncode == 0 and done.stdout.strip() else []
    except (subprocess.SubprocessError, ValueError, OSError):
        return []


def strings(row: dict) -> list[str]:
    return [str(v) for v in row.values() if isinstance(v, (str, int, float))]


def app_id(row: dict) -> str | None:
    return next((v for v in strings(row) if re.fullmatch(r"ap-\w+", v)), None)


def app_name(row: dict) -> str | None:
    return next((n for n in APPS if n in strings(row)), None)


def billed() -> float:
    rows = modal_json("billing", "report", "--for", "today", "--show-resources")
    return sum(float(r["cost"]) for r in rows if r.get("description") in APPS)


def stop(reason: str, ids: set[str]) -> None:
    (HERE / "STOP").write_text(reason + "\n")
    say(f"STOP: {reason}; apps {sorted(ids)}")
    for ident in ids:
        out = subprocess.run(["modal", "app", "stop", "-y", ident], capture_output=True, text=True)
        say(f"modal app stop {ident}: exit {out.returncode} {out.stdout.strip()[-200:]} {out.stderr.strip()[-200:]}")


def main() -> None:
    LOG.write_text("")
    started = time.time()
    baseline = {app_id(r) for r in modal_json("app", "list")}
    say(f"start; apps already on the account (never touched): {sorted(i for i in baseline if i)}")
    say(f"billed for the two apps today at start: {billed():.4f}; rates {APPS}; guide ${GUIDE}, kill ${KILL}")
    seen: dict[str, tuple[float, str]] = {}  # container id -> (first seen, app)
    last_age: dict[str, float] = {}
    used, last_billed, warned, done_at = 0.0, 0.0, False, None
    while True:
        now = time.time()
        apps = [r for r in modal_json("app", "list") if app_name(r) and app_id(r) not in baseline]
        ids = {i for i in (app_id(r) for r in apps) if i}
        by_id = {app_id(r): app_name(r) for r in apps}
        containers = [r for r in modal_json("container", "list") if any(i in strings(r) for i in ids)] if ids else []
        present, ages = set(), []
        for row in containers:
            ident = next((v for v in strings(row) if re.fullmatch(r"ta-\w+", v)), str(row)[:40])
            owner = next((by_id[i] for i in ids if i in strings(row)), "mime-m3-stage2")
            present.add(ident)
            seen.setdefault(ident, (now, owner))
            last_age[ident] = now - seen[ident][0]
            ages.append((last_age[ident], owner))
        rate = {i: seen[i][1] for i in seen}
        for ident in [i for i in last_age if i not in present]:  # a finished container: its cost stays in the bound
            used += last_age.pop(ident) / 3600.0 * APPS[rate[ident]]
            seen.pop(ident, None)
        bound = used + sum(a / 3600.0 * APPS[o] for a, o in ages)
        say(f"apps {[(app_id(r), app_name(r)) for r in apps]} containers {len(containers)} "
            f"ages_s {[(round(a), o) for a, o in ages]} live_bound ${bound:.3f}")
        reason = None
        if bound >= KILL:
            reason = f"live bound ${bound:.2f} >= ${KILL}"
        elif any(a > CONTAINER_LIMIT_S for a, _ in ages):
            reason = f"a container has run over {CONTAINER_LIMIT_S:.0f} s"
        elif any(sum(1 for _, o in ages if o == name) > 1 for name in APPS):
            reason = "more than one container of an app at once"
        elif now - started > TOTAL_S:
            reason = "3 hours since the tracker started"
        if now - last_billed >= BILLED_EVERY_S and ids:
            last_billed = now
            spend = billed()
            say(f"billed for the two apps today: ${spend:.3f}")
            if spend >= KILL:
                reason = f"billed ${spend:.2f} >= ${KILL}"
            elif spend >= GUIDE and not warned:
                warned = True
                say(f"WARNING: billed spend ${spend:.2f} passed the guide of ${GUIDE}")
        if bound >= GUIDE and not warned:
            warned = True
            say(f"WARNING: live bound ${bound:.2f} passed the guide of ${GUIDE}")
        if reason:
            stop(reason, ids)
            return
        if DONE_FILE.exists():
            done_at = done_at or now
            running = [r for r in apps if "stopped" not in " ".join(strings(r)).lower()]
            if not running and not containers:
                say(f"DONE; final billed ${billed():.3f}")
                return
            if now - done_at > DONE_GRACE_S:
                stop("DONE written but an app is still running after 5 minutes", ids)
                return
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("interrupted")
        sys.exit(1)
