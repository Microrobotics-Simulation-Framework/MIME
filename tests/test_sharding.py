"""CI's shards (``tests/_sharding.py``, ``scripts/check_shards.py``, ``scripts/compilation_cache_mode.py``,
``scripts/shard_weights.py``): a lane's shards, collecting for real, keep a partition of its tests, and the check a lane
runs over their records passes it and fails each way a shard can drop, double or miss a test; the deal is by file, a
function of the table alone and balanced; a pull request restores a cache only when nothing it changes moves a compile.

The controls are fixed plants, each failing on every run."""

import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import _sharding
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_shards = _script("check_shards")
cache_mode = _script("compilation_cache_mode")
shard_weights = _script("shard_weights")
#: Two files that import nothing heavy, collected for real under each shard: the split is by file, so some shard of
#: two must hold one and another the other (their paths' hashes differ at two shards).
SMALL = ("tests/test_canary.py", "tests/test_prune_jax_cache.py")


def record_shards(tmp_path: Path, of: int) -> list[Path]:
    """Collect ``SMALL`` once per shard of ``of``, as a shard job does, each record beside a passing outcome."""
    directories = []
    for shard in range(1, of + 1):
        directory = tmp_path / f"shard-{shard}"
        directory.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_XDIST")}
        done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                               "--mime-shard", f"{shard}/{of}", "--mime-shard-record", str(directory / "shard.json"),
                               *SMALL], cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=300)
        assert done.returncode in (0, 5), done.stdout + done.stderr  # 5: this shard holds none of the file's tests
        (directory / "outcome.txt").write_text("success\n", encoding="utf-8")
        directories.append(directory)
    return directories


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> list[Path]:
    return record_shards(tmp_path_factory.mktemp("shards"), 2)


def test_the_shards_collecting_for_real_partition_the_tests_and_the_check_passes_them(recorded):
    records = [json.loads((d / "shard.json").read_text(encoding="utf-8")) for d in recorded]
    assert records[0]["collected"] == records[1]["collected"]
    assert {nodeid.split("::")[0] for nodeid in records[0]["collected"]} == set(SMALL)
    assert sorted(records[0]["selected"] + records[1]["selected"]) == records[0]["collected"]
    assert records[0]["selected"] and records[1]["selected"]
    assert all(len({nodeid.split("::")[0] for nodeid in record["selected"]}) == 1 for record in records)
    assert check_shards.problems(recorded, 2) == []


def _rewrite(directory: Path, change) -> None:
    record = json.loads((directory / "shard.json").read_text(encoding="utf-8"))
    change(record)
    (directory / "shard.json").write_text(json.dumps(record), encoding="utf-8")


def test_control_a_shard_that_dropped_a_test_fails_naming_it(recorded, tmp_path):
    copies = [Path(_copy(d, tmp_path)) for d in recorded]
    donor = next(d for d in copies if json.loads((d / "shard.json").read_text(encoding="utf-8"))["selected"])
    dropped = json.loads((donor / "shard.json").read_text(encoding="utf-8"))["selected"][0]
    _rewrite(donor, lambda record: record["selected"].remove(dropped))
    assert check_shards.problems(copies, 2) == [f"{dropped}: collected and kept by no shard"]


def test_control_a_test_kept_by_two_shards_fails_naming_it(recorded, tmp_path):
    copies = [Path(_copy(d, tmp_path)) for d in recorded]
    first, second = (json.loads((d / "shard.json").read_text(encoding="utf-8")) for d in copies)
    taker, giver = (copies[1], first) if first["selected"] else (copies[0], second)
    doubled = giver["selected"][0]
    _rewrite(taker, lambda record: record["selected"].append(doubled))
    assert check_shards.problems(copies, 2) == [f"{doubled}: kept by shards [1, 2]"]


def test_control_a_missing_or_red_shard_or_another_collection_fails(recorded, tmp_path):
    copies = [Path(_copy(d, tmp_path)) for d in recorded]
    assert check_shards.problems(copies[:1], 2) == [
        "no record from shard(s) 2 of 2: the job failed before it uploaded one, or never ran"]
    (copies[1] / "outcome.txt").write_text("failure\n", encoding="utf-8")
    assert check_shards.problems(copies, 2) == ["shard 2 of 2: its job's status is 'failure'"]
    (copies[1] / "outcome.txt").write_text("success\n", encoding="utf-8")
    _rewrite(copies[1], lambda record: record["collected"].pop())
    assert check_shards.problems(copies, 2) == ["the shards collected 2 different sets of tests; one collection is "
                                                "split"]
    assert check_shards.problems(copies, 3)[0].endswith("shard 1 of 2; this lane runs 3")


def _copy(directory: Path, into: Path) -> str:
    target = into / directory.name
    target.mkdir()
    for name in ("shard.json", "outcome.txt"):
        (target / name).write_bytes((directory / name).read_bytes())
    return str(target)


def test_the_check_exits_with_its_verdict(recorded, tmp_path):
    assert check_shards.main(["--shards", "2", *map(str, recorded)]) == 0
    assert check_shards.main(["--shards", "2", str(recorded[0])]) == 1


def test_the_deal_is_a_function_of_the_table_and_balanced():
    weights = _sharding.load_weights()
    for of in (2, 4, 6):
        dealt = _sharding.deal(weights, of)
        assert dealt == _sharding.deal(dict(reversed(list(weights.items()))), of)
        assert set(dealt.values()) == set(range(1, of + 1))
        load = [sum(s for path, s in weights.items() if dealt[path] == shard) for shard in range(1, of + 1)]
        # Longest first onto the lightest: no shard is more than the longest file above the lightest.
        assert max(load) - min(load) <= max(weights.values())


def test_a_files_tests_share_its_shard_and_a_file_outside_the_table_goes_by_its_hash():
    dealt = _sharding.deal(_sharding.load_weights(), 6)
    listed = next(iter(dealt))
    assert {_sharding.shard_of(f"{listed}::test_{i}[{i}]", 6, dealt) for i in range(9)} == {dealt[listed]}
    added = "tests/test_new.py"
    assert added not in dealt and _sharding.shard_of(f"{added}::test_added", 6, dealt) == _sharding.by_hash(added, 6)


@pytest.mark.parametrize("spec", ["0/4", "5/4", "1/0", "a/4", "1/4/2", " 1/4", "1/4:weighted"])
def test_a_malformed_shard_is_a_usage_error(spec):
    with pytest.raises(pytest.UsageError):
        _sharding.parse(spec)


def test_a_bad_table_is_an_error_never_an_unweighted_split(tmp_path):
    for content in ("", "{}", '{"seconds": {}}', '{"seconds": {"a": 1.5}}', '{"seconds": {"a": 0}}'):
        (tmp_path / "w.json").write_text(content, encoding="utf-8")
        with pytest.raises(pytest.UsageError):
            _sharding.load_weights(tmp_path / "w.json")


@pytest.mark.parametrize("event, choice, message, changed, expected", [
    ("pull_request", "", "feat: a change", ["src/mime/core/_node.py"], ("warm", False)),
    ("pull_request", "", "perf: before and after [cold-ci]", [], ("cold", False)),
    ("pull_request", "", None, [], ("cold", False)),
    ("pull_request", "", "test: a test slow-marked", ["tests/duration_allowlist.txt"], ("cold", False)),
    ("pull_request", "", "ci: the shards rebalanced", ["tests/shard_weights.json"], ("cold", False)),
    ("pull_request", "", "ci: the split changed", ["tests/_sharding.py"], ("cold", False)),
    ("push", "", None, [], ("cold", True)),
    ("workflow_dispatch", "cold", None, [], ("cold", True)),
    ("workflow_dispatch", "warm", None, [], ("warm", False)),
    ("workflow_dispatch", "", None, [], ("cold", True)),
])
def test_only_a_pull_request_that_moves_no_compile_restores_a_cache_and_none_saves(event, choice, message, changed,
                                                                                   expected):
    assert cache_mode.decide(event, choice, message, changed)[:2] == expected


def test_the_cache_key_follows_the_cpus_features_not_their_order():
    one = cache_mode.cpu_features("processor\t: 0\nflags\t\t: fpu sse2 avx2\n")
    assert one == cache_mode.cpu_features("flags\t: avx2 fpu sse2\nflags\t: other\n")
    assert one != cache_mode.cpu_features("flags\t: fpu sse2 avx2 avx512f\n")


def test_the_weights_read_a_reports_node_ids_as_pytest_writes_them():
    report = ET.fromstring(
        '<testsuites><testsuite><testcase classname="tests.saved.test_x" name="test_a[1-2]" file="tests/saved/'
        'test_x.py" time="3.5"/><testcase classname="tests.saved.test_x.TestMachine" name="runTest" file="tests/saved/'
        'test_x.py" time="1"/></testsuite></testsuites>')
    assert [shard_weights.node_id(case) for case in report.iter("testcase")] == [
        "tests/saved/test_x.py::test_a[1-2]", "tests/saved/test_x.py::TestMachine::runTest"]
