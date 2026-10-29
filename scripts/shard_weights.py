#!/usr/bin/env python
"""Write ``tests/shard_weights.json``, the table ``tests/_sharding.py`` deals CI's heaviest tests by, from CI's JUnit
reports, and print the seconds each shard is then predicted to hold.

    gh run download <run id> --pattern 'durations-*' --dir reports
    python scripts/shard_weights.py reports --shards 4 --source "run <run id>"

A test's weight is its median time over the reports (each lane's, each shard's), rounded to whole seconds; the table
lists the tests at or over ``--over`` seconds, and every other test goes by its hash. Run it in the project's
environment: it loads ``tests/_sharding.py``, which imports pytest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def medians(reports: list[Path]) -> dict[str, float]:
    times: dict[str, list[float]] = {}
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            if case.find("skipped") is None:
                times.setdefault(node_id(case), []).append(float(case.get("time") or 0.0))
    return {nodeid: statistics.median(seconds) for nodeid, seconds in times.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="+", type=Path, help="directories holding the runs' JUnit XML")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--over", type=float, default=2.0, help="list the tests at or over this many seconds")
    parser.add_argument("--source", required=True, help="the runs the times come from, for the table's record")
    args = parser.parse_args(argv)
    reports = sorted(path for directory in args.directories for path in directory.rglob("*.xml"))
    if not reports:
        print("no JUnit XML under the directories given", file=sys.stderr)
        return 2
    times = medians(reports)
    seconds = {nodeid: max(1, round(t)) for nodeid, t in sorted(times.items()) if t >= args.over}
    TABLE.write_text(json.dumps({"source": args.source, "over": args.over, "seconds": seconds}, indent=1) + "\n",
                     encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_sharding", ROOT / "tests" / "_sharding.py")
    assert spec is not None and spec.loader is not None
    sharding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sharding)
    dealt = sharding.deal(seconds, args.shards)
    load = [0.0] * args.shards
    for nodeid, t in times.items():
        load[sharding.shard_of(nodeid, args.shards, dealt) - 1] += t
    print(f"{len(seconds)} tests listed from {len(reports)} reports; predicted seconds per shard: "
          + ", ".join(f"{t:.0f}" for t in load))
    return 0


if __name__ == "__main__":
    sys.exit(main())
