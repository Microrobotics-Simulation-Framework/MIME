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

**The split.** Every run of the selection (one Hypothesis seed) is split across parallel jobs, so that a large pull request's
changed files fit a job's limit: the files are dealt longest first onto the lightest of *n* parts by their seconds in
``tests/shard_weights.json`` (the table the default lane's shards are dealt by; a file it does not list takes
``new_file_weight``, the 90th percentile of its file weights, as ``tests/_sharding.py`` assumes), *n* being the total over
``PART_SECONDS``, rounded up, and at most the number of units and ``MAX_JOBS`` over the runs. Each part runs in its own job for
every run, each run with its own seed, so every selected test runs as many times as it did unsplit. A file is not divided
(a node id of a file weighs the file's seconds, an upper bound), so a file over ``PART_SECONDS`` is a part of its own, and a
description that narrows the set with ``only`` and a few node ids is one part, as before.

**A file the table splits** (its ``split``: one so heavy that, whole, it would not fit a job, as the default lane knows) is
dealt by test instead: each test the table lists is a unit of its listed seconds, dealt with the other files' units, and goes
into its part as a node id. The tests the table does not list (each under its ``tests_over``, and any added since) have no
unit of their own: they ride with the one part, among those that hold a listed test of the file, with the least seconds, as
that part's pytest is given the file itself and a ``--deselect=<node id>`` for each listed test another part runs (so a part
that held only the unlisted tests, and so ran nothing, can never occur). Every test of the file is in exactly one part. Two
limits: ``--deselect`` matches by prefix, so a listed test whose name is a prefix of another test's name would drop that test
from the part that has the file (a file with such a pair among its listed tests is dealt whole instead, and
``test_stress_select.py`` holds the table to it); and a listed test the file no longer has is a node id pytest cannot find,
which fails the part loudly (the table is rebuilt by ``scripts/shard_weights.py``). A part's seconds are the units' by the
table; each part that gets tests of a split file compiles the programs they share once more, which the table does not count.
``cache`` names the compilation caches of the default lane's shards that hold the selected files (the ones a base
branch's push saved), at most ``CACHE_SLOTS`` of them, most files first.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_RUNS, MAX_RUNS = 20, 40
SUPPORTED_PYTHONS = ("3.12", "3.13", "3.14")
DEFAULT_PYTHON = SUPPORTED_PYTHONS[0]
CACHE_SLOTS = 3
#: The seconds of the shard table (summed test time, run on four workers) a job is given, and the most jobs one stress run may
#: start: GitHub caps a matrix at 256 entries. On a runner the changed set of a large pull request took 0.4 to 0.6 wall
#: seconds for a table second, so 1000 is 7 to 10 minutes, and a file over it is the one job that long, of the 20 minutes.
PART_SECONDS, MAX_JOBS = 1000, 250
WEIGHTS = Path(__file__).resolve().parents[1] / "tests" / "shard_weights.json"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tests"))
import _shard_table  # noqa: E402
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
    #: ``parts[i]`` is the targets of part ``i + 1``, each in directory order; every part runs for every run.
    parts: tuple[tuple[str, ...], ...] = ()

    def matrix(self) -> dict[str, list[dict[str, int]]]:
        """One job per run and part: ``index`` is the part's place in the ``parts`` output, ``part`` its number."""
        return {"include": [{"run": k, "seed": self.seed + k - 1, "part": i + 1, "index": i}
                            for k in range(1, self.runs + 1) for i in range(len(self.parts))]}


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


def in_directory_order(targets: list[str]) -> list[str]:
    """``targets`` with each directory's files together, the directories in tree order; the order within a directory is
    kept. pytest 9.1.1 loses a directory's conftest fixtures (``tests/saved``'s ``x64``) for the files it meets the
    directory with again after a file of a directory above it, which a listed target, then a changed file elsewhere, then
    another changed file of the first's directory would do (``tests/_collection_guard.py``)."""
    return sorted(targets, key=lambda target: PurePosixPath(target.split("::", 1)[0]).parts[:-1])


def table_seconds(path: Path = WEIGHTS) -> tuple[dict[str, int], int]:
    """``({test file: seconds}, the seconds of a file the table does not list)`` from the shard table, read as
    ``tests/_sharding.py`` reads it (its ``new_file_weight``: the 90th percentile, nearest rank, of the file weights)."""
    seconds = json.loads(path.read_text(encoding="utf-8"))["seconds"]
    ordered = sorted(seconds.values())
    return seconds, ordered[math.ceil(0.9 * len(ordered)) - 1]


def test_ids(source: str) -> set[str]:
    """The ids, as pytest forms them without the file and without parameters, of the ``test*`` functions and methods of
    ``source``: ``test_x`` and ``TestC::test_x``, a class's name leading its methods'. A name that is not collected is
    included too: a name too many only pins a test."""
    ids: set[str] = set()

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}::")
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
                ids.add(prefix + node.name)

    walk(ast.parse(source).body, "")
    return ids


def defined_tests(file: str, root: Path = ROOT) -> set[str]:
    """The ids (``test_ids``) of the tests ``file`` defines, read from its source (none if it cannot be read)."""
    try:
        return test_ids((root / file).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set()


def pinned_tests(tests: dict[str, int], defined: set[str]) -> set[str]:
    """The listed tests of a split file that ``--deselect`` could not leave out alone: those whose name is a prefix of
    another test's id, listed or defined in the file, by the whole id as ``--deselect`` matches it (``TestC::test_a`` is a
    prefix of ``TestC::test_ab``, and not of ``TestD::test_a``; a parameter's id ends in ``]``, so it never is one)."""
    return {name for name in tests if any(other != name and other.startswith(name) for other in {*tests, *defined})}


def split(targets: list[str], seconds: dict[str, int], unlisted: int, runs: int,
          by_test: dict[str, dict[str, int]] | None = None, defined: dict[str, set[str]] | None = None) -> list[list[str]]:
    """``targets`` (in directory order) dealt into parts by their seconds: longest unit first onto the lightest part, ties on
    the unit's name and the lower part. A unit is a file, kept whole, or, for a bare file of ``by_test`` (the table's
    ``split``), one of its listed tests, except the one that carries the file's other tests (the module's docstring): its
    unlisted tests and its pinned ones (``pinned_tests``) with the file's lightest listed test that is not pinned (a file
    with none is dealt whole). Each part keeps ``targets``' order, with its ``--deselect`` options last."""
    by_test, defined = by_test or {}, defined or {}
    files = dict.fromkeys(target.split("::", 1)[0] for target in targets)
    units: dict[tuple[str, ...], int] = {}
    carrier: dict[str, str] = {}  # a split file's carrying listed test
    pinned: dict[str, set[str]] = {}
    for file in files:
        tests = by_test.get(file, {})
        pinned[file] = pinned_tests(tests, defined.get(file, set()))
        free = sorted((sec, name) for name, sec in tests.items() if name not in pinned[file])
        if file in targets and free:
            carrier[file] = free[0][1]
            units.update({(file, name): sec for name, sec in tests.items() if name not in pinned[file]})
            others = sum(sec for name, sec in tests.items() if name in pinned[file]) + max(
                seconds.get(file, unlisted) - sum(tests.values()), 1)
            units[(file, carrier[file])] += others
        else:
            units[(file,)] = seconds.get(file, unlisted)
    count = max(1, min(len(units), math.ceil(sum(units.values()) / PART_SECONDS), MAX_JOBS // runs))
    load, held = [0] * count, [set[tuple[str, ...]]() for _ in range(count)]
    for unit in sorted(units, key=lambda u: (-units[u], u)):
        lightest = min(range(count), key=lambda k: (load[k], k))
        held[lightest].add(unit)
        load[lightest] += units[unit]
    parts: list[list[str]] = []
    for k in range(count):
        arguments, deselected = [], []
        for file in files:
            if file in carrier:
                mine = [name for name in by_test[file] if (file, name) in held[k] and name not in pinned[file]]
                if (file, carrier[file]) in held[k]:
                    arguments.append(file)
                    deselected += [f"--deselect={file}::{name}" for name in by_test[file]
                                   if name not in pinned[file] and (file, name) not in held[k]]
                else:
                    arguments += [f"{file}::{name}" for name in mine]
            elif (file,) in held[k]:
                arguments += [t for t in targets if t.split("::", 1)[0] == file]
        if arguments:
            parts.append(arguments + deselected)
    return parts


def select(body: str, changed: list[str], exists: Callable[[str], bool], fresh_seed: Callable[[], int],
           weights: tuple[dict[str, int], int] | None = None,
           by_test: dict[str, dict[str, int]] | None = None,
           defined: dict[str, set[str]] | None = None) -> Request:
    """The stress run a description and a diff ask for; ``exists`` says whether a listed file is in the checkout."""
    listed, options, only = parse_description(body)
    for token in listed:
        found = TARGET.fullmatch(token)
        if found is None:
            raise Refused(f"{token!r} is not a test file or node id under tests/ (options are runs=, python=, seed=, only)")
        if not exists(found.group("file")):
            raise Refused(f"{found.group('file')} is not in this checkout")
    chosen = dict.fromkeys(listed + ([] if only else sorted(p for p in changed if CHANGED_TEST.fullmatch(p))))
    targets = in_directory_order(list(chosen))
    if not targets:
        raise Refused("nothing to run: the pull request changes no tests/**/test_*.py and its description lists none "
                      "(a line `stress: <path or node id> ...`)")
    python = options.get("python", DEFAULT_PYTHON)
    if python not in SUPPORTED_PYTHONS:
        raise Refused(f"python={python!r} is not one of the default lane's {', '.join(SUPPORTED_PYTHONS)}")
    runs = _whole("runs", options["runs"], 1, MAX_RUNS) if "runs" in options else DEFAULT_RUNS
    seed = _whole("seed", options["seed"], 0, 2**31 - 1) if "seed" in options else fresh_seed()
    seconds, unlisted = weights if weights is not None else table_seconds()
    by_name = by_test if by_test is not None else ({} if weights is not None else _shard_table.read_split(WEIGHTS))
    if defined is None:
        defined = {file: defined_tests(file) for file in by_name if file in targets}
    parts = tuple(tuple(part) for part in split(targets, seconds, unlisted, runs, by_name, defined))
    return Request(tuple(targets), runs, python, seed, parts)


def cache_shards(targets: list[str], count: int) -> list[str]:
    """``["i/N", ...]`` for the shards holding the targets' files, the shards with most files first, at most
    ``CACHE_SLOTS``; the split is the one CI runs (``tests/_sharding.py``), with the files the table does not list read from
    the checkout's ``test_*.py`` files, as CI reads them from its collection."""
    _sharding = importlib.import_module("_sharding")

    weights, listed = _sharding.load_weights(), _sharding.load_split()
    split = set(listed)
    on_disk = {path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").rglob("test_*.py")}
    dealt = _sharding.deal(weights, count, on_disk - weights.keys() - split, _sharding.new_file_weight())
    held: dict[int, set[str]] = {}
    skipped = {target.split("=", 1)[1] for target in targets if target.startswith("--deselect=")}
    for target in (t for t in targets if not t.startswith("-")):
        file = target.split("::", 1)[0]
        for nodeid in [f"{file}::{name}" for name in listed[file]] if file == target and file in split else [target]:  # a bare split file: the shards of the listed tests this part runs
            if nodeid not in skipped:
                held.setdefault(_sharding.shard_of(nodeid, count, dealt, split), set()).add(file)
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
                     "seed": str(request.seed), "parts": json.dumps(request.parts), "part_count": str(len(request.parts)),
                     "matrix": json.dumps(request.matrix())})
            print(f"stress: {len(request.targets)} targets, {request.runs} runs of {len(request.parts)} parts, "
                  f"Python {request.python}, first seed {request.seed}")
            for number, part in enumerate(request.parts, 1):
                print(f" part {number}:")
                for target in part:
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
