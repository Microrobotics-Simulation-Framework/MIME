#!/usr/bin/env python
"""What a stress run (``.github/workflows/stress.yml``) runs, from the pull request's changed files and its description,
written to ``$GITHUB_OUTPUT``. Standard library only, except ``cache``, which reads ``tests/_sharding.py`` and so runs
inside the synced environment.

``select`` reads the description from the environment variable ``PR_BODY`` and the changed files from git. The runs are
the test files the pull request changes (``tests/**/test_*.py``, deleted ones left out) and whatever a line of the
description lists::

    stress: tests/oracles/test_spinner.py tests/test_x.py::test_y[a] runs=30 python=3.14 seed=7 only

- a bare token is a test file or node id under ``tests/``, added to the changed files; ``only`` drops the changed files;
- ``runs=N``: 1 to 40 runs, 20 if absent. ``python=3.x``: a Python of the default lane, 3.12 (the floor) if absent;
- ``seed=N``: the first run's Hypothesis seed, a fresh random one if absent. Run *k* uses ``N + k - 1``.

Anything it cannot use is refused by name, and so is an empty selection: a stress run that ran nothing is not a pass.
``cache`` names the compilation caches of the default lane's shards that hold the selected files (the ones a base
branch's push saved), at most ``CACHE_SLOTS`` of them, most files first.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RUNS, MAX_RUNS = 20, 40
SUPPORTED_PYTHONS = ("3.12", "3.13", "3.14")
DEFAULT_PYTHON = SUPPORTED_PYTHONS[0]
CACHE_SLOTS = 3
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compilation_cache_mode  # noqa: E402
import shard_matrix  # noqa: E402

CHANGED_TEST = re.compile(r"tests/(?:[^/]+/)*test_[^/]+\.py")
#: A file under tests/ (no `..`, no leading `-`), then an optional `::` node id of printable characters without spaces.
TARGET = re.compile(r"(?P<file>tests/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_-][A-Za-z0-9_.-]*\.py)(?P<node>::[!-~]+)?")
OPTION = re.compile(r"(?P<name>runs|python|seed)=(?P<value>.*)", re.IGNORECASE)
LINE = re.compile(r"^[ \t>*+-]*stress:(?P<rest>.*)$", re.IGNORECASE | re.MULTILINE)


class Refused(ValueError):
    """A request that cannot be run, saying why."""


@dataclass(frozen=True)
class Request:
    targets: tuple[str, ...]
    runs: int
    python: str
    seed: int

    def matrix(self) -> dict[str, list[dict[str, int]]]:
        return {"include": [{"run": k, "seed": self.seed + k - 1} for k in range(1, self.runs + 1)]}


def _whole(name: str, value: str, low: int, high: int) -> int:
    if not re.fullmatch(r"[0-9]{1,12}", value):
        raise Refused(f"{name}={value!r} is not a whole number")
    number = int(value)
    if not low <= number <= high:
        raise Refused(f"{name}={number} is outside {low} to {high}")
    return number


def parse_description(body: str) -> tuple[list[str], dict[str, str], bool]:
    """``(listed targets, options, only)`` from the ``stress:`` lines of a pull request's description."""
    listed: list[str] = []
    options: dict[str, str] = {}
    only = False
    for line in LINE.finditer(body):
        for token in (t.strip("`,;") for t in line.group("rest").split()):
            option = OPTION.fullmatch(token)
            if token.lower() == "only":
                only = True
            elif option:
                name, value = option.group("name").lower(), option.group("value")
                if options.setdefault(name, value) != value:
                    raise Refused(f"{name} is given twice, as {options[name]!r} and {value!r}")
            elif token:
                listed.append(token)
    return listed, options, only


def select(body: str, changed: list[str], exists: Callable[[str], bool], fresh_seed: Callable[[], int]) -> Request:
    """The stress run a description and a diff ask for; ``exists`` says whether a listed file is in the checkout."""
    listed, options, only = parse_description(body)
    for token in listed:
        found = TARGET.fullmatch(token)
        if found is None:
            raise Refused(f"{token!r} is not a test file or node id under tests/ (options are runs=, python=, seed=, only)")
        if not exists(found.group("file")):
            raise Refused(f"{found.group('file')} is not in this checkout")
    targets = list(dict.fromkeys(listed + ([] if only else sorted(p for p in changed if CHANGED_TEST.fullmatch(p)))))
    if not targets:
        raise Refused("nothing to run: the pull request changes no tests/**/test_*.py and its description lists none "
                      "(a line `stress: <path or node id> ...`)")
    python = options.get("python", DEFAULT_PYTHON)
    if python not in SUPPORTED_PYTHONS:
        raise Refused(f"python={python!r} is not one of the default lane's {', '.join(SUPPORTED_PYTHONS)}")
    runs = _whole("runs", options["runs"], 1, MAX_RUNS) if "runs" in options else DEFAULT_RUNS
    seed = _whole("seed", options["seed"], 0, 2**31 - 1) if "seed" in options else fresh_seed()
    return Request(tuple(targets), runs, python, seed)


def cache_shards(targets: list[str], count: int) -> list[str]:
    """``["i/N", ...]`` for the shards holding the targets' files, the shards with most files first, at most
    ``CACHE_SLOTS``; the split is the one CI runs (``tests/_sharding.py``)."""
    sys.path.insert(0, str(ROOT / "tests"))
    _sharding = importlib.import_module("_sharding")

    dealt = _sharding.deal(_sharding.load_weights(), count)
    held: dict[int, set[str]] = {}
    for target in targets:
        held.setdefault(_sharding.shard_of(target, count, dealt), set()).add(target.split("::", 1)[0])
    ranked = sorted(held, key=lambda shard: (-len(held[shard]), shard))
    return [f"{shard}/{count}" for shard in ranked[:CACHE_SLOTS]]


def _output(values: dict[str, str]) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        for name, value in values.items():
            out.write(f"{name}={value}\n")


def _changed_files() -> list[str]:
    """The files of the pull request: the merge commit against its base's tip, as ``compilation_cache_mode.py`` reads."""
    done = subprocess.run(["git", "diff", "--name-only", "--diff-filter=d", "HEAD^1", "HEAD"],
                          capture_output=True, text=True, check=True)
    return done.stdout.split("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select", help="reads PR_BODY and git; writes targets, runs, python, seed, matrix")
    cache = commands.add_parser("cache", help="reads TARGETS (JSON); writes key1..key3, the default lane's cache keys")
    cache.add_argument("--python", required=True)
    cache.add_argument("--jaxlib", required=True)
    cache.add_argument("--variable", default="", help="the repository variable MIME_CI_SHARDS, empty if unset")
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            request = select(os.environ.get("PR_BODY", ""), _changed_files(),
                             lambda path: (ROOT / path).is_file(), lambda: secrets.randbelow(2**31))
            _output({"targets": json.dumps(request.targets), "runs": str(request.runs), "python": request.python,
                     "seed": str(request.seed), "matrix": json.dumps(request.matrix())})
            print(f"stress: {len(request.targets)} targets, {request.runs} runs, Python {request.python}, "
                  f"first seed {request.seed}")
            for target in request.targets:
                print(f"  {target}")
            return 0
        count = shard_matrix.resolve(args.variable, "")[0]
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        shards = cache_shards(json.loads(os.environ["TARGETS"]), count)
        keys = [compilation_cache_mode.cache_key(f"default-py{args.python}", args.jaxlib, cpuinfo, s) for s in shards]
        _output({f"key{slot}": keys[slot - 1] if slot <= len(keys) else "" for slot in range(1, CACHE_SLOTS + 1)})
        print(f"compilation caches restored, read-only: shards {', '.join(shards)}")
        return 0
    except (Refused, shard_matrix.Refused) as error:
        print(f"::error title=Stress::{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
