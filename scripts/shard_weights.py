#!/usr/bin/env python
"""Write ``tests/shard_weights.json``, the table ``tests/_sharding.py`` deals CI's test files by, from CI's JUnit
reports, and print the seconds each shard is then predicted to hold.

    gh run download <run id> --pattern 'durations-*' --dir reports
    python scripts/shard_weights.py reports/* --shards 6 --source "run <run id>"

A file's weight is the median over the lanes of its tests' summed times, rounded to whole seconds (each directory is
one artifact, and a shard's, named ``...-shard-<n>``, is its lane's); the table lists the files at or over ``--over`` seconds, and every other
file goes by its hash. Run it in the project's
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


def medians(lanes: list[list[Path]]) -> dict[str, float]:
    """Each file's median over the lanes (a file a lane did not run counts as 0 s there)."""
    per_lane = [file_seconds(reports) for reports in lanes]
    files = sorted(set().union(*per_lane))
    return {path: statistics.median(lane.get(path, 0.0) for lane in per_lane) for path in files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="+", type=Path,
                        help="the downloaded artifacts' directories, each holding a lane's or a shard's JUnit XML")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--over", type=float, default=1.0, help="list the files at or over this many seconds")
    parser.add_argument("--source", required=True, help="the runs the times come from, for the table's record")
    args = parser.parse_args(argv)
    grouped: dict[str, list[Path]] = {}
    for directory in args.directories:
        grouped.setdefault(re.sub(r"-shard-[0-9]+$", "", directory.name), []).extend(sorted(directory.glob("*.xml")))
    lanes = [reports for _, reports in sorted(grouped.items()) if reports]
    if not lanes:
        print("no JUnit XML in the directories given", file=sys.stderr)
        return 2
    times = medians(lanes)
    seconds = {path: max(1, round(t)) for path, t in times.items() if t >= args.over}
    TABLE.write_text(json.dumps({"source": args.source, "over": args.over, "seconds": seconds}, indent=1) + "\n",
                     encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_sharding", ROOT / "tests" / "_sharding.py")
    assert spec is not None and spec.loader is not None
    sharding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sharding)
    dealt = sharding.deal(seconds, args.shards)
    load = [0.0] * args.shards
    for path, t in times.items():
        load[sharding.shard_of(path, args.shards, dealt) - 1] += t
    print(f"{len(seconds)} files listed from {len(lanes)} lanes; predicted seconds per shard: "
          + ", ".join(f"{t:.0f}" for t in load))
    return 0


if __name__ == "__main__":
    sys.exit(main())
