#!/usr/bin/env python
"""A lane's shards ran the lane: run by a lane's own job after its shard jobs (``.github/workflows/ci.yml``).

Each shard job uploads a directory holding ``shard.json`` (what it collected and what it kept, written by
``tests/_sharding.py``) and ``outcome.txt`` (its job's status). The lane passes only if there are exactly ``--shards``
of them, one per shard number, each passed, all collected the same tests, and their selections partition that
collection: no test run by no shard, none by two. A shard that never uploaded (it failed to install, was cancelled or
never started) is missing, and fails the lane. Standard library only, so the lane's job needs no install to run it.

    python3 scripts/check_shards.py --shards 4 results/*/

Exits 0 when the lane holds, 1 naming each problem otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def problems(directories: list[Path], shards: int) -> list[str]:
    found: list[str] = []
    records: dict[int, dict[str, object]] = {}
    for directory in directories:
        try:
            record = json.loads((directory / "shard.json").read_text(encoding="utf-8"))
            outcome = (directory / "outcome.txt").read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as error:
            found.append(f"{directory}: no readable shard record and outcome ({error})")
            continue
        number = record.get("shard")
        if record.get("of") != shards or not isinstance(number, int) or not 1 <= number <= shards:
            found.append(f"{directory}: shard {number!r} of {record.get('of')!r}; this lane runs {shards}")
            continue
        if number in records:
            found.append(f"{directory}: a second record for shard {number}")
            continue
        records[number] = record
        if outcome != "success":
            found.append(f"shard {number} of {shards}: its job's status is {outcome!r}")
    missing = sorted(set(range(1, shards + 1)) - set(records))
    if missing:
        found.append(f"no record from shard(s) {', '.join(map(str, missing))} of {shards}: the job failed before it "
                     f"uploaded one, or never ran")
    if not records:
        return found
    collections = {tuple(record["collected"]) for record in records.values()}  # type: ignore[arg-type]
    if len(collections) != 1:
        found.append(f"the shards collected {len(collections)} different sets of tests; one collection is split")
        return found
    (collected,) = collections
    selected: dict[str, list[int]] = {}
    for number, record in sorted(records.items()):
        for nodeid in record["selected"]:  # type: ignore[union-attr]
            selected.setdefault(nodeid, []).append(number)
    twice = sorted(nodeid for nodeid, numbers in selected.items() if len(numbers) > 1)
    stray = sorted(set(selected) - set(collected))
    unrun = sorted(set(collected) - set(selected)) if not missing else []
    found += [f"{nodeid}: kept by shards {selected[nodeid]}" for nodeid in twice[:20]]
    found += [f"{nodeid}: kept by a shard and not collected" for nodeid in stray[:20]]
    found += [f"{nodeid}: collected and kept by no shard" for nodeid in unrun[:20]]
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="*", type=Path, help="one directory per shard job's upload")
    parser.add_argument("--shards", type=int, required=True, help="how many shards the lane runs")
    args = parser.parse_args(argv)
    found = problems(args.directories, args.shards)
    for problem in found:
        print(f"::error title=Shards::{problem}")
    if not found:
        kept = len(json.loads((args.directories[0] / "shard.json").read_text(encoding="utf-8"))["collected"])
        print(f"{args.shards} shards passed and together ran the {kept} tests collected")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
