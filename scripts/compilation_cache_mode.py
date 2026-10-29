#!/usr/bin/env python
"""How a CI shard job uses JAX's persistent compilation cache, written to ``$GITHUB_OUTPUT`` as ``mode``, ``save`` and
``key`` (``.github/workflows/ci.yml``). Standard library only.

- A pull request runs **warm**: it restores the cache its base branch's last push saved for the same lane, shard and
  runner CPU, and never saves one. A program the pull request adds or changes is not in it, so it compiles in full and
  the time budget sees what it costs; if pull requests saved, a second push would read the first push's programs and a
  new slow test would pass the budget fast. A pull request runs **cold** instead when its head commit's message holds
  ``[cold-ci]`` (before-and-after numbers), when its diff edits ``tests/duration_allowlist.txt``, the shards' table or
  ``tests/_sharding.py`` (a test moved, slow-marked or deleted hands its compiles to another test, which would read
  them from the base cache and look fast until the cold run after the merge), or when the message cannot be read.
- A push (after a merge) runs **cold**, from an empty cache, and saves what it compiled: the cache the next pull
  requests restore.
- A manual run is **cold** and saves, unless its ``compilation_cache`` input asks for **warm**, which restores the
  branch's last saved cache and saves nothing: two manual runs measure the cache's saving.

The key holds the lane, the jaxlib release and a digest of the runner CPU's feature flags. XLA compiles for the CPU it
runs on, and JAX's own cache key holds no CPU features (only the platform, ``cpu``), so a cache built on a runner with
features another lacks must never be restored there. A cold job still writes the cache as it runs, so its xdist workers
and the processes its tests start read each other's programs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

#: What a pull request's diff may not touch and still run warm.
COLD_PATHS = ("tests/duration_allowlist.txt", "tests/shard_weights.json", "tests/_sharding.py")
COLD_REQUEST = "[cold-ci]"


def decide(event: str, choice: str, message: str | None, changed: list[str]) -> tuple[str, bool, str]:
    """``(mode, save, why)`` for a run's event, its manual choice, its head commit's message (``None`` when unreadable)
    and the paths its pull request changes."""
    if event == "pull_request":
        if message is None:
            return "cold", False, "the head commit's message could not be read, so a [cold-ci] cannot be ruled out"
        if COLD_REQUEST in message:
            return "cold", False, f"the head commit asks for {COLD_REQUEST}"
        touched = sorted(set(changed) & set(COLD_PATHS))
        if touched:
            return "cold", False, f"the pull request edits {', '.join(touched)}"
        return "warm", False, "a pull request restores its base branch's cache"
    if event == "workflow_dispatch" and choice == "warm":
        return "warm", False, "the manual run asks for a warm cache"
    return "cold", True, f"a {event} starts empty and saves the cache"


def cpu_features(cpuinfo: str) -> str:
    """A digest of the first processor's feature flags."""
    flags = next((line.split(":", 1)[1].split() for line in cpuinfo.splitlines() if line.startswith("flags")), [])
    return hashlib.sha256(" ".join(sorted(flags)).encode("utf-8")).hexdigest()[:12]


def _git(*arguments: str) -> str | None:
    done = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--choice", default="", help="a manual run's compilation_cache input")
    parser.add_argument("--lane", required=True, help="the lane and its Python, as the key names them")
    parser.add_argument("--jaxlib", required=True)
    parser.add_argument("--shard", required=True, help="i/N")
    args = parser.parse_args(argv)
    message, changed = None, []
    if args.event == "pull_request":
        # A pull request checks out its merge commit two deep: the base's tip, then the head commit.
        message = _git("log", "-1", "--format=%B", "HEAD^2")
        changed = (_git("diff", "--name-only", "HEAD^1", "HEAD") or "").split()
    mode, save, why = decide(args.event, args.choice, message, changed)
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), "?")
    shard = args.shard.replace("/", "of")
    key = f"jaxcc-v1-{args.lane}-jaxlib{args.jaxlib}-cpu{cpu_features(cpuinfo)}-shard{shard}"
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        out.write(f"mode={mode}\nsave={'true' if save else 'false'}\nkey={key}\n")
    print(f"compilation cache: {mode}{' and saved' if save else ''} ({why}); {model}; key {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
