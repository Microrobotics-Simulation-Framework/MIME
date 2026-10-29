"""Split the suite across CI jobs, test by test, stably: ``MIME_TEST_SHARD=i/N`` keeps shard ``i``'s tests (1-based).

Each lane runs as N jobs (``.github/workflows/ci.yml``). Every job collects the whole suite, so every ``conftest.py`` is
imported as in an unsplit run, and deselects the other shards' tests after the markers have deselected theirs. A test's
shard is a function of its node id and of the committed table ``shard_weights.json``, never of the test list:

- the tests the table lists (each over a couple of seconds on CI, with its median seconds) are dealt longest first,
  each onto the shard with the least time so far, ties broken on the node id and the lower shard;
- every other test goes by a hash of its node id.

Adding or deleting a test moves no other, and shard ``i`` of a pull request holds the tests of shard ``i`` of its base
unless the table changed, so the compilation cache the base saved for a shard is the one its programs are in. The split
is by test, not by file as MADDENING's is, because two of MIME's files (the registry walk and the node walk) each hold
more than a quarter of a lane; a file's module fixtures are built again in each shard that holds its tests, as they
already are in each xdist worker. ``scripts/shard_weights.py`` rewrites the table from a run's JUnit reports.

``MIME_TEST_SHARD_RECORD=<file>`` writes what the job collected and what it kept, as JSON, which the lane's summary job
reads (``scripts/check_shards.py``): the shards' selections must partition one collection, so no test is run by none
of them and none by two.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

ENV_VAR = "MIME_TEST_SHARD"
RECORD_VAR = "MIME_TEST_SHARD_RECORD"
WEIGHTS_FILE = Path(__file__).with_name("shard_weights.json")


def parse(spec: str) -> tuple[int, int]:
    """``"i/N"`` -> ``(i, N)``, 1 <= i <= N; anything else is a usage error."""
    if not re.fullmatch(r"[0-9]+/[0-9]+", spec):
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: expected 'i/N', e.g. '2/4'")
    shard, of = (int(x) for x in spec.split("/"))
    if not 1 <= shard <= of:
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: need 1 <= i <= N")
    return shard, of


def by_hash(nodeid: str, of: int) -> int:
    # sha256, not hash(): Python's string hash is salted per process.
    return int(hashlib.sha256(nodeid.encode("utf-8")).hexdigest(), 16) % of + 1


def load_weights(path: Path = WEIGHTS_FILE) -> dict[str, int]:
    """``{node id: seconds}`` from the committed table; a bad table is an error, never an unweighted split."""
    try:
        seconds = json.loads(path.read_text(encoding="utf-8"))["seconds"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise pytest.UsageError(f"{path}: cannot read the shards' table ({error!r})") from None
    if not isinstance(seconds, dict) or not seconds:
        raise pytest.UsageError(f"{path}: 'seconds' must be a non-empty {{node id: seconds}} table")
    bad = {k: v for k, v in seconds.items() if type(v) is not int or v <= 0}
    if bad:
        raise pytest.UsageError(f"{path}: every weight must be a positive whole number of seconds: {bad}")
    return seconds


def deal(weights: dict[str, int], of: int) -> dict[str, int]:
    """``{node id: shard}`` for the listed tests: longest first, each onto the lightest shard."""
    load = [0] * of
    shards: dict[str, int] = {}
    for nodeid, seconds in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        lightest = min(range(of), key=lambda k: (load[k], k))
        shards[nodeid] = lightest + 1
        load[lightest] += seconds
    return shards


def shard_of(nodeid: str, of: int, dealt: dict[str, int]) -> int:
    return dealt[nodeid] if nodeid in dealt else by_hash(nodeid, of)


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
        return f"test shard: {self.shard}/{self.of} (by test; tests/_sharding.py)"


def register(config: pytest.Config, spec: str) -> None:
    shard, of = parse(spec)
    config.pluginmanager.register(Plugin(shard, of, load_weights(), os.environ.get(RECORD_VAR)), "mime-test-shard")
