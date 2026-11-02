#!/usr/bin/env python
"""Keep each shard's newest attempt: run by a lane's own job (``.github/workflows/ci.yml``) between downloading the
shards' reports and checking them.

A shard job uploads its report as ``<lane>-shard-<n>-attempt-<k>``, ``k`` being the run attempt (``github.run_attempt``).
Re-running one failed shard job adds ``...-attempt-2`` for that shard and leaves the first attempt's report in the run, so
the run holds two reports of the shard. Under one name the download took the first listed, the failed attempt, and the
lane stayed red after the shard passed on its re-run. This deletes, in the downloaded directory, every attempt of a
shard but its highest, so that ``scripts/check_shards.py`` reads each shard once, from the newest attempt.

It does not decide whether a shard passed, and a re-run cannot hide a failure through it: the newest attempt's own
``outcome.txt`` and JUnit report are what the check reads, so a shard that fails again fails the lane again. A
directory whose name is not of that form fails this script, since a report it cannot place is one it cannot choose
between.

    python3 scripts/latest_attempt.py "$RUNNER_TEMP"/shards

Exits 0 after printing what it kept and dropped, 1 naming each directory it cannot place.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

NAME = re.compile(r"^(?P<shard>.+-shard-\d+)-attempt-(?P<attempt>\d+)$")


def newest(directory: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """The directories in ``directory`` to keep, to drop (an older attempt of a shard kept) and that fit no shard."""
    attempts: dict[str, dict[int, Path]] = {}
    unplaced: list[Path] = []
    for child in sorted(directory.iterdir()):
        match = NAME.match(child.name) if child.is_dir() else None
        if match is None:
            unplaced.append(child)
        else:
            attempts.setdefault(match["shard"], {})[int(match["attempt"])] = child
    kept = [by_attempt[max(by_attempt)] for _, by_attempt in sorted(attempts.items())]
    dropped = [path for by_attempt in attempts.values() for attempt, path in sorted(by_attempt.items())
               if attempt != max(by_attempt)]
    return kept, sorted(dropped), unplaced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="the directory the shards' reports were downloaded into")
    arguments = parser.parse_args(argv)
    if not arguments.directory.is_dir():
        print(f"::error title=Shards::{arguments.directory} is not a directory; no shard's report was downloaded")
        return 1
    kept, dropped, unplaced = newest(arguments.directory)
    for path in unplaced:
        print(f"::error title=Shards::{path.name} is not a shard's report named <lane>-shard-<n>-attempt-<k>")
    if unplaced:
        return 1
    for path in dropped:
        print(f"dropped {path.name}: a newer attempt of that shard is in this run")
        shutil.rmtree(path)
    print(f"kept {len(kept)} shard report(s), the newest attempt of each: {', '.join(path.name for path in kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
