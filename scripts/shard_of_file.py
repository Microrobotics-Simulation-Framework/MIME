#!/usr/bin/env python
"""The CI shard that holds a test file, for ``.github/workflows/ci.yml``: a job that only one shard of a lane needs (the
shells the completion tests run under) asks which shard that is and does the work there alone.

    uv run --no-sync python scripts/shard_of_file.py tests/cli/test_completion.py --of 10

prints the shard, 1-based, that ``--mime-shard i/10`` keeps the file's tests in (``tests/_sharding.py``: the same table
and the same deal, so the answer is the plugin's, never a copy of its rule). It imports pytest, as the plugin does, and
so runs inside the lane's environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import _sharding  # noqa: E402


def shard_of_file(path: str, of: int) -> int:
    """The shard of ``of`` that holds the tests of ``path`` (relative to the root, with ``/``)."""
    return _sharding.shard_of(path, of, _sharding.deal(_sharding.load_weights(), of))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="a test file, relative to the repository root")
    parser.add_argument("--of", type=int, required=True, help="the lane's shard count")
    arguments = parser.parse_args(argv)
    if arguments.of < 1:
        parser.error("--of must be at least 1")
    if not (Path(__file__).resolve().parents[1] / arguments.path).is_file():
        parser.error(f"{arguments.path} is not a file of this repository")
    print(shard_of_file(arguments.path, arguments.of))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
