#!/usr/bin/env python
"""Write ``tests/shard_weights.json``, the table ``tests/_sharding.py`` deals CI's test files by, from CI's JUnit
reports, and print the seconds each shard is then predicted to hold.

    gh run download <run id> --pattern 'durations-*' --dir reports
    python scripts/shard_weights.py reports/* --shards 6 --source "run <run id>"

A file's weight is the median over the lanes of its tests' summed times, rounded to whole seconds (each directory is
one artifact, and a shard's, named ``...-shard-<n>`` or ``...-shard-<n>-attempt-<k>``, is its lane's); the table lists
the files at or over ``--over`` seconds, and every other file goes by its hash. A file named by ``--split`` (or already
under the table's ``split``) is listed by test as well: its tests at or over ``--over`` seconds, each the median over the
lanes of that test's time, are what the deal places, so the file's tests spread over the shards. Run it in the project's
environment: it loads ``tests/_sharding.py``, which imports pytest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "tests" / "shard_weights.json"


def node_id(case: ET.Element) -> str:
    """The pytest node id of an xunit1 ``<testcase>``: its file, its class (the last part of ``classname`` when that
    is not the module's), and its name."""
    file, name = case.get("file", ""), case.get("name", "")
    module = file.removesuffix(".py").replace("/", ".")
    classname = case.get("classname", "")
    cls = classname[len(module) + 1:] if classname.startswith(module + ".") else ""
    return "::".join(part for part in (file, cls.replace(".", "::"), name) if part)


def file_seconds(reports: list[Path]) -> dict[str, float]:
    """Each test file's summed time over ``reports``, one lane's."""
    seconds: dict[str, float] = {}
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            path = node_id(case).split("::", 1)[0]
            seconds[path] = seconds.get(path, 0.0) + float(case.get("time") or 0.0)
    return seconds


def test_seconds(reports: list[Path]) -> dict[str, float]:
    """Each test's time over ``reports``, one lane's."""
    seconds: dict[str, float] = {}
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            nodeid = node_id(case)
            seconds[nodeid] = seconds.get(nodeid, 0.0) + float(case.get("time") or 0.0)
    return seconds


def medians(lanes: list[list[Path]]) -> dict[str, float]:
    """Each file's median over the lanes (a file a lane did not run counts as 0 s there)."""
    per_lane = [file_seconds(reports) for reports in lanes]
    files = sorted(set().union(*per_lane))
    return {path: statistics.median(lane.get(path, 0.0) for lane in per_lane) for path in files}


def lanes_of(directories: list[Path]) -> list[list[Path]]:
    """The JUnit reports of each lane, from the artifacts' directories. A directory is a shard's, named
    ``<lane>-shard-<n>`` or ``<lane>-shard-<n>-attempt-<k>``; a shard that ran again is counted once, by its last attempt
    (its first is a job that was lost or failed)."""
    last: dict[tuple[str, str], tuple[int, Path]] = {}
    for directory in directories:
        match = re.fullmatch(r"(.*)-shard-([0-9]+)(?:-attempt-([0-9]+))?", directory.name)
        lane, shard, attempt = (match.group(1), match.group(2), int(match.group(3) or 0)) if match else (
            directory.name, "", 0)
        if (lane, shard) not in last or attempt >= last[lane, shard][0]:
            last[lane, shard] = (attempt, directory)
    grouped: dict[str, list[Path]] = {}
    for (lane, _), (_, directory) in last.items():
        grouped.setdefault(lane, []).extend(sorted(directory.glob("*.xml")))
    return [reports for _, reports in sorted(grouped.items()) if reports]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="+", type=Path,
                        help="the downloaded artifacts' directories, each holding a lane's or a shard's JUnit XML")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--over", type=float, default=1.0, help="list the files at or over this many seconds")
    parser.add_argument("--split", action="append", default=[], metavar="FILE",
                        help="deal this test file by test (repeat for more); the files already split stay split")
    parser.add_argument("--source", required=True, help="the runs the times come from, for the table's record")
    args = parser.parse_args(argv)
    lanes = lanes_of(args.directories)
    if not lanes:
        print("no JUnit XML in the directories given", file=sys.stderr)
        return 2
    times = medians(lanes)
    seconds = {path: max(1, round(t)) for path, t in times.items() if t >= args.over}
    alone, split_files = {}, set(args.split)
    if TABLE.exists():  # the files that hold a shard of their own, and why, are kept while they are still listed
        old = json.loads(TABLE.read_text(encoding="utf-8"))
        alone = {path: why for path, why in old.get("alone", {}).items() if path in seconds}
        split_files |= set(old.get("split", {}))
    split_files &= set(seconds)
    per_test = [test_seconds(reports) for reports in lanes]
    split = {}
    for file in sorted(split_files - set(alone)):
        names = sorted({n for lane in per_test for n in lane if n.split("::", 1)[0] == file})
        tests = {n.split("::", 1)[1]: max(1, round(statistics.median(lane.get(n, 0.0) for lane in per_test)))
                 for n in names if "::" in n}
        split[file] = {n: t for n, t in tests.items() if t >= args.over}
    table = {"source": args.source, "over": args.over, "seconds": seconds, **({"alone": alone} if alone else {}),
             **({"split": {f: t for f, t in split.items() if t}} if split else {})}
    TABLE.write_text(json.dumps(table, indent=1) + "\n",
                     encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_sharding", ROOT / "tests" / "_sharding.py")
    assert spec is not None and spec.loader is not None
    sharding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sharding)
    dealt = sharding.deal(sharding.load_weights(TABLE), args.shards)
    split_names = set(sharding.load_split(TABLE))
    load = [0.0] * args.shards
    for path, t in times.items():
        if path not in split_names:
            load[sharding.shard_of(path, args.shards, dealt) - 1] += t
    for lane in per_test:  # a split file's tests are placed one by one; each lane counts for its share of the median
        for nodeid, t in lane.items():
            if nodeid.split("::", 1)[0] in split_names:
                load[sharding.shard_of(nodeid, args.shards, dealt, split_names) - 1] += t / len(per_test)
    print(f"{len(seconds)} files listed from {len(lanes)} lanes; predicted seconds per shard: "
          + ", ".join(f"{t:.0f}" for t in load)
          + (f"; split by test: {', '.join(sorted(split_names))}" if split_names else ""))
    mean = sum(w for path, w in seconds.items() if path not in alone) / max(1, args.shards - len(alone))
    heavy = sorted(path for path, w in seconds.items() if w > mean and path not in alone and path not in split_names)
    if heavy:
        print(f"WARNING: heavier than a shard's mean ({mean:.0f} s) and dealt whole, so a shard holds it alone at best: "
              + ", ".join(heavy) + " (--split FILE deals a file by test; each shard that gets some of its tests "
              "compiles the programs they share once more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
