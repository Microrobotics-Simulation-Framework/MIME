#!/usr/bin/env python
"""Measure each test's peak memory and write the ones over the line to ``tests/memory_peaks.json``, the file
``tests/test_memory_heavy.py`` guards the ``memory_heavy`` marker with.

    python scripts/measure_test_memory.py --out peaks.jsonl -- -n 4 tests/oracles

Everything after ``--`` goes to pytest, which runs here with this file loaded as a plugin. The plugin resets the
process's peak resident size before each test (``/proc/self/clear_refs``, Linux) and reads it after, so a test's peak
is its own process's high-water mark during that test: the memory its worker held when it began, which a worker keeps
from the tests before it, is in it. ``--merge FILE`` instead reads such a file (the lines of one or more runs) and
rewrites ``tests/memory_peaks.json`` from it with ``--line`` as the line, keeping a function's largest peak over its
parameters; ``--date`` and ``--basis`` fill the file's account of the measurement. One process each is the cleaner
number for a test over the line: ``/usr/bin/time -f %M pytest -p no:xdist <test id>``."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "tests" / "memory_peaks.json"
START: dict[str, float] = {}


def status() -> dict[str, float]:
    found: dict[str, float] = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmHWM", "VmRSS")):
                key, value = line.split(":")
                found[key] = int(value.split()[0]) / 1024  # MiB
    return found


def pytest_runtest_protocol(item: object, nextitem: object) -> None:
    with open("/proc/self/clear_refs", "w", encoding="utf-8") as handle:
        handle.write("5")  # reset the peak resident size
    START[getattr(item, "nodeid")] = status()["VmRSS"]


def pytest_runtest_logfinish(nodeid: str, location: object) -> None:
    now = status()
    with open(os.environ["MIME_PEAKS_OUT"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": nodeid, "peak_mib": round(now["VmHWM"]), "start_mib": round(START.pop(nodeid, 0)),
                                 "pid": os.getpid()}) + "\n")


def function_of(nodeid: str) -> str:
    return re.sub(r"\[.*\]$", "", nodeid)


def merge(files: list[str], line: int, date: str, basis: str) -> None:
    largest: dict[str, int] = {}
    for file in files:
        for text in Path(file).read_text(encoding="utf-8").splitlines():
            row = json.loads(text)
            name = function_of(row["id"])
            largest[name] = max(largest.get(name, 0), row["peak_mib"])
    peaks = {name: peak for name, peak in sorted(largest.items()) if peak > line}
    TABLE.write_text(json.dumps({"measured": date, "line_mib": line, "basis": basis, "peaks_mib": peaks}, indent=2) + "\n",
                     encoding="utf-8")
    print(f"{len(peaks)} tests over {line} MiB of {len(largest)} measured -> {TABLE}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="the JSON lines file this run appends its peaks to")
    parser.add_argument("--merge", nargs="+", help="rewrite tests/memory_peaks.json from these runs' files")
    parser.add_argument("--line", type=int, default=2048, help="MiB, the line above which a test is memory_heavy")
    parser.add_argument("--date", default="")
    parser.add_argument("--basis", default="")
    split = argv.index("--") if "--" in argv else len(argv)
    own, pytest_arguments = argv[1:split], argv[split + 1:]
    args = parser.parse_args(own)
    if args.merge:
        merge(args.merge, args.line, args.date, args.basis)
        return 0
    if not args.out:
        parser.error("--out or --merge")
    environment = {**os.environ, "MIME_PEAKS_OUT": str(Path(args.out).resolve()),
                   "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])}
    return subprocess.call([sys.executable, "-m", "pytest", "-p", "measure_test_memory", *pytest_arguments],
                           env=environment, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
