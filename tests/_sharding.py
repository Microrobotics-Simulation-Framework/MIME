"""Split the suite across CI jobs by test file, stably: ``--mime-shard i/N`` keeps shard ``i``'s files (1-based).

Each lane runs as N jobs (``.github/workflows/ci.yml``). Every job collects the whole suite, so every ``conftest.py`` is
imported as in an unsplit run, and deselects the other shards' files after the markers have deselected their tests. A
file's shard is a function of its path and of the committed table ``shard_weights.json``, never of the test list:

- the files the table lists, with their median seconds on CI, are dealt longest first, each onto the shard with the
  least time so far, ties broken on the path and the lower shard;
- every other file goes by a hash of its path.

Adding or deleting a test moves no file, and shard ``i`` of a pull request holds the files of shard ``i`` of its base
unless the table changed, so the compilation cache the base saved for a shard is the one its programs are in. A file's
tests stay together, as MADDENING's split keeps them: a test that reuses the programs another test of its file
compiled (a control beside the property it controls) stays in the shard that compiles them, and module fixtures are
built in one shard. ``scripts/shard_weights.py`` rewrites the table from a run's JUnit reports.

The split is a command-line option, not a variable, so a pytest that a test starts in a subprocess inherits no shard.
``--mime-shard-record FILE`` writes what the job collected and what it kept, as JSON, which the lane's own job reads
(``scripts/check_shards.py``): the shards' selections must partition one collection, so no test is run by none of
them and none by two.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

#: The options, which tests/conftest.py adds.
OPTION = "--mime-shard"
RECORD_OPTION = "--mime-shard-record"
WEIGHTS_FILE = Path(__file__).with_name("shard_weights.json")


def parse(spec: str) -> tuple[int, int]:
    """``"i/N"`` -> ``(i, N)``, 1 <= i <= N; anything else is a usage error."""
    if not re.fullmatch(r"[0-9]+/[0-9]+", spec):
        raise pytest.UsageError(f"{OPTION} {spec!r}: expected 'i/N', e.g. '2/4'")
    shard, of = (int(x) for x in spec.split("/"))
    if not 1 <= shard <= of:
        raise pytest.UsageError(f"{OPTION} {spec!r}: need 1 <= i <= N")
    return shard, of


def by_hash(path: str, of: int) -> int:
    # sha256, not hash(): Python's string hash is salted per process.
    return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % of + 1


def load_weights(path: Path = WEIGHTS_FILE) -> dict[str, int]:
    """``{test file: seconds}`` from the committed table; a bad table is an error, never an unweighted split."""
    try:
        seconds = json.loads(path.read_text(encoding="utf-8"))["seconds"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise pytest.UsageError(f"{path}: cannot read the shards' table ({error!r})") from None
    if not isinstance(seconds, dict) or not seconds:
        raise pytest.UsageError(f"{path}: 'seconds' must be a non-empty {{test file: seconds}} table")
    bad = {k: v for k, v in seconds.items() if type(v) is not int or v <= 0}
    if bad:
        raise pytest.UsageError(f"{path}: every weight must be a positive whole number of seconds: {bad}")
    return seconds


def deal(weights: dict[str, int], of: int) -> dict[str, int]:
    """``{test file: shard}`` for the listed files: longest first, each onto the lightest shard."""
    load = [0] * of
    shards: dict[str, int] = {}
    for path, seconds in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        lightest = min(range(of), key=lambda k: (load[k], k))
        shards[path] = lightest + 1
        load[lightest] += seconds
    return shards


def shard_of(nodeid: str, of: int, dealt: dict[str, int]) -> int:
    """The shard of a test, by its file: the node id's path, relative to the root with ``/`` on every platform."""
    path = nodeid.split("::", 1)[0]
    return dealt[path] if path in dealt else by_hash(path, of)


class Plugin:
    def __init__(self, shard: int, of: int, weights: dict[str, int], record: str | None):
        self.shard, self.of, self.record = shard, of, record
        self.dealt = deal(weights, of)

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        collected = [item.nodeid for item in items]
        keep = [item for item in items if shard_of(item.nodeid, self.of, self.dealt) == self.shard]
        drop = [item for item in items if shard_of(item.nodeid, self.of, self.dealt) != self.shard]
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep
        # Every xdist worker collects the same items; one of them writes the record.
        if self.record and os.environ.get("PYTEST_XDIST_WORKER", "gw0") == "gw0":
            Path(self.record).write_text(json.dumps({
                "shard": self.shard, "of": self.of, "collected": sorted(collected),
                "selected": sorted(item.nodeid for item in keep)}), encoding="utf-8")

    def pytest_report_header(self, config: pytest.Config) -> str:
        return f"test shard: {self.shard}/{self.of} (by file; tests/_sharding.py)"


def register(config: pytest.Config, spec: str) -> None:
    shard, of = parse(spec)
    config.pluginmanager.register(Plugin(shard, of, load_weights(), config.getoption(RECORD_OPTION)),
                                  "mime-test-shard")
