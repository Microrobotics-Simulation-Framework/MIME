"""CI's shards (``tests/_sharding.py``, ``scripts/check_shards.py``, ``scripts/compilation_cache_mode.py``,
``scripts/shard_weights.py``): a lane's shards, running for real, keep a partition of its tests, and the check a lane
runs over their records and JUnit reports passes it and fails each way a shard can drop, double, skip or miss a test, or
a record can differ from the report; the deal is by file, a
function of the table alone and balanced; a pull request restores a cache only when nothing it changes moves a compile.

The controls are fixed plants, each failing on every run."""

import importlib.util
import json
import os
import shutil
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
#: The shard counts the recorded artifacts are made for: one, a pair, a count that does not divide the files, and
#: today's default. A lane may run as any whole number of shards from ``scripts/shard_matrix.py``'s bounds.
COUNTS = (1, 2, 3, 6)
CONFTEST = """import sys
sys.path.insert(0, {tests!r})
import _sharding


def pytest_addoption(parser):
    parser.addoption("--mime-shard")
    parser.addoption("--mime-shard-record")


def pytest_configure(config):
    if config.getoption("--mime-shard"):
        _sharding.register(config, config.getoption("--mime-shard"))
"""


def shard_files(of: int) -> list[str]:
    """One test file name per shard of ``of``, found by the hash that places a file outside the weights' table: the
    split is by file, so a project of these files has a shard that holds each."""
    names: dict[int, str] = {}
    candidate = 0
    while len(names) < of:
        name = f"test_s{candidate}.py"
        names.setdefault(_sharding.by_hash(name, of), name)
        candidate += 1
    return [names[shard] for shard in range(1, of + 1)]


def record_shards(tmp_path: Path, of: int) -> list[Path]:
    """Run a project of ``of`` small test files once per shard of ``of``, as a shard job does, with the shard plugin of
    ``tests/_sharding.py``: its record, its JUnit report and a passing outcome in a directory each. Every shard holds
    one file of two tests."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "conftest.py").write_text(CONFTEST.format(tests=str(ROOT / "tests")), encoding="utf-8")
    for name in shard_files(of):
        (project / name).write_text("def test_a():\n    pass\n\n\ndef test_b():\n    pass\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_XDIST", "MIME_CI_PLANT", "MIME_TEST_JAX"))}
    directories = []
    for shard in range(1, of + 1):
        directory = tmp_path / f"shard-{shard}"
        directory.mkdir()
        done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
                               "junit_family=xunit1", "--mime-shard", f"{shard}/{of}", "--mime-shard-record",
                               str(directory / "shard.json"),
                               f"--junitxml={directory / f'test-results-shard{shard}.xml'}"],
                              cwd=project, env=env, capture_output=True, text=True, check=False, timeout=300)
        assert done.returncode == 0, done.stdout + done.stderr
        (directory / "outcome.txt").write_text("success\n", encoding="utf-8")
        directories.append(directory)
    return directories


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    """``recordings(count)``: the artifacts of a lane of ``count`` shards, made once per count."""
    made: dict[int, list[Path]] = {}

    def get(count: int) -> list[Path]:
        if count not in made:
            made[count] = record_shards(tmp_path_factory.mktemp(f"shards{count}"), count)
        return made[count]
    return get


@pytest.mark.parametrize("of", COUNTS)
def test_the_shards_running_for_real_partition_the_tests_and_the_check_passes_them(of, recordings):
    recorded = recordings(of)
    records = [json.loads((d / "shard.json").read_text(encoding="utf-8")) for d in recorded]
    assert {tuple(record["collected"]) for record in records} == {tuple(records[0]["collected"])}
    assert len(records[0]["collected"]) == 2 * of
    assert sorted(nodeid for record in records for nodeid in record["selected"]) == records[0]["collected"]
    assert all(len(record["selected"]) == 2 for record in records)
    assert all(len({nodeid.split("::")[0] for nodeid in record["selected"]}) == 1 for record in records)
    for directory, record in zip(recorded, records, strict=True):
        assert sorted(_outcomes(directory)) == record["selected"]
    assert check_shards.problems(recorded, of, 0) == []


def _rewrite(directory: Path, change) -> None:
    record = json.loads((directory / "shard.json").read_text(encoding="utf-8"))
    change(record)
    (directory / "shard.json").write_text(json.dumps(record), encoding="utf-8")


def _report(directory: Path) -> Path:
    return next(directory.glob("test-results-shard*.xml"))


def _edit_report(directory: Path, change) -> None:
    tree = ET.parse(_report(directory))
    change(tree.getroot())
    tree.write(_report(directory), encoding="utf-8")


def _outcomes(directory: Path) -> dict[str, str]:
    return {shard_weights.node_id(case): ("skipped" if case.find("skipped") is not None else "ran")
            for case in ET.parse(_report(directory)).getroot().iter("testcase")}


def _drop_case(root: ET.Element, position: int = 0) -> None:
    case = list(root.iter("testcase"))[position]
    for suite in root.iter("testsuite"):
        if case in list(suite):
            suite.remove(case)


def _skip(case: ET.Element) -> None:
    ET.SubElement(case, "skipped", {"message": "planted"})


def _copy(directory: Path, into: Path) -> Path:
    target = into / directory.name
    shutil.copytree(directory, target)
    return target


def _kept(directory: Path) -> list[str]:
    return json.loads((directory / "shard.json").read_text(encoding="utf-8"))["selected"]


def _both(copies: list[Path], change) -> None:
    for directory in copies:
        _rewrite(directory, change)


def _first_case(root: ET.Element) -> ET.Element:
    return next(iter(root.iter("testcase")))


def _append_case(root: ET.Element, case: ET.Element) -> None:
    next(root.iter("testsuite")).append(ET.fromstring(ET.tostring(case)))


def _ghost_case(root: ET.Element) -> None:
    _append_case(root, ET.Element("testcase", {"classname": "tests.ghost", "name": "test_never_collected",
                                               "file": "tests/ghost.py", "time": "0"}))


def _fail_case(root: ET.Element) -> None:
    ET.SubElement(_first_case(root), "failure", {"message": "planted"})


def _twice(root: ET.Element) -> None:
    _append_case(root, _first_case(root))


def _stale(copies: list[Path]) -> tuple[list[Path], int]:
    stale = copies[0].parent / "shard-stale"
    shutil.copytree(copies[0], stale)
    _rewrite(stale, lambda record: record.update(shard=len(copies) + 1))
    return [*copies, stale], len(copies)


def _repeated(copies: list[Path]) -> tuple[list[Path], int]:
    repeated = copies[0].parent / "shard-1-again"
    shutil.copytree(copies[0], repeated)
    return [*copies, repeated], len(copies)


def _no_outcome(copies: list[Path]) -> tuple[list[Path], int]:
    (copies[-1] / "outcome.txt").unlink()
    return copies, len(copies)


def _status(value: str):
    def plant(copies: list[Path]) -> tuple[list[Path], int]:
        (copies[-1] / "outcome.txt").write_text(f"{value}\n", encoding="utf-8")
        return copies, len(copies)
    return plant


def _on(change):
    """A plant that changes the copies in place."""
    def plant(copies: list[Path]) -> tuple[list[Path], int]:
        change(copies)
        return copies, len(copies)
    return plant


def _last_keeps_a_test_of_the_first(copies: list[Path]) -> None:
    _rewrite(copies[-1], lambda record: record["selected"].append(_kept(copies[0])[0]))


def _first_reports_a_case_of_the_last(copies: list[Path]) -> None:
    other = ET.parse(_report(copies[-1])).getroot()
    _edit_report(copies[0], lambda root: _append_case(root, _first_case(other)))


def _record_names_a_test_the_report_lacks(copies: list[Path]) -> None:
    ghost = "tests/ghost.py::test_named_by_the_record_only"
    _both(copies, lambda record: record["collected"].append(ghost))
    _rewrite(copies[0], lambda record: record["selected"].append(ghost))


def _nothing_collected(copies: list[Path]) -> None:
    for directory in copies:
        _rewrite(directory, lambda record: record.update(collected=[], selected=[]))
        _edit_report(directory, lambda root: [_drop_case(root) for _ in list(root.iter("testcase"))])


#: Each plant, made in copies of the recorded artifacts: ``(fewest shards it can be made with, the plant, what the
#: check must say)``, ``{n}`` the lane's shard count and ``{n1}`` one more. Two of the plants need a second shard, to
#: give a test to or take one from. All are fixed, so each fails on every run.
PLANTS = {
    "a shard that never uploaded": (1, lambda copies: (copies[:-1], len(copies)),
                                    "no record from shard(s) {n} of {n}"),
    "a cancelled shard": (1, _status("cancelled"), "its job's status is 'cancelled'"),
    "a failed shard": (1, _status("failure"), "its job's status is 'failure'"),
    "a shard with no outcome file": (1, _no_outcome, "no readable shard record and outcome"),
    "a shard that kept no test": (1, _on(lambda c: _rewrite(c[0], lambda r: r.update(selected=[]))),
                                  "collected and kept by no shard"),
    "a test dropped by its shard": (1, _on(lambda c: _rewrite(c[0], lambda r: r["selected"].pop())),
                                    "collected and kept by no shard"),
    "a test kept by two shards": (2, _on(_last_keeps_a_test_of_the_first), "kept by shards [1, {n}]"),
    "a test kept and never collected": (1, _on(lambda c: _rewrite(c[0], lambda r: r["selected"].append(
        "tests/ghost.py::test_x"))), "kept by a shard and not collected"),
    "two different collections": (2, _on(lambda c: _rewrite(c[-1], lambda r: r["collected"].pop())),
                                  "2 different sets of tests"),
    "one shard collected nothing": (2, _on(lambda c: _rewrite(c[-1], lambda r: r.update(collected=[]))),
                                    "2 different sets of tests"),
    "a lane told more shards than ran": (1, lambda copies: (copies, len(copies) + 1), "this lane runs {n1}"),
    "a repeated shard number": (1, _repeated, "a second record for shard 1"),
    "an extra, stale shard directory": (1, _stale, "shard {n1} of {n}"),
    "a record that names a test the report lacks": (1, _on(_record_names_a_test_the_report_lacks),
                                                    "its JUnit report has no outcome for it"),
    "a report that lacks a test its record keeps": (1, _on(lambda c: _edit_report(c[0], _drop_case)),
                                                    "its JUnit report has no outcome for it"),
    "a test no report holds": (1, _on(lambda c: _edit_report(c[0], _drop_case)),
                               "collected and reported by no shard's JUnit"),
    "a shard whose tests were all skipped": (1, _on(lambda c: _edit_report(c[0], lambda root: [
        _skip(case) for case in root.iter("testcase")])), "of its tests were skipped"),
    "a skip above the lane's ceiling": (1, _on(lambda c: _edit_report(c[0], lambda root: _skip(_first_case(root)))),
                                        "tests were skipped; this lane allows 0"),
    "all shards collecting nothing": (1, _on(_nothing_collected), "the lane collected no tests"),
    "a shard with no report": (1, _on(lambda c: _report(c[0]).unlink()), "0 JUnit reports"),
    "a test with two outcomes in a report": (1, _on(lambda c: _edit_report(c[0], _twice)),
                                             "has 2 outcomes in its JUnit report"),
    "a test that failed in a report of a passing job": (1, _on(lambda c: _edit_report(c[0], _fail_case)),
                                                        "reported failure"),
    "a test reported by two shards": (2, _on(_first_reports_a_case_of_the_last), "reported by shards [1, {n}]"),
    "a test reported and never collected": (1, _on(lambda c: _edit_report(c[0], _ghost_case)),
                                            "reported by a shard and not collected"),
}


@pytest.mark.parametrize("of", COUNTS)
def test_the_recorded_artifacts_copied_pass_before_a_plant(of, recordings, tmp_path):
    assert check_shards.problems([_copy(d, tmp_path) for d in recordings(of)], of, 0) == []


@pytest.mark.parametrize("of, name", [(of, name) for of in COUNTS for name in PLANTS if of >= PLANTS[name][0]],
                         ids=lambda value: str(value) if isinstance(value, int) else value.replace(" ", "-"))
def test_control_each_planted_fault_in_the_recorded_artifacts_fails_the_check_naming_it(of, name, recordings,
                                                                                         tmp_path):
    _, plant, expected = PLANTS[name]
    copies = [_copy(d, tmp_path) for d in recordings(of)]
    directories, shards = plant(copies)
    found = check_shards.problems(directories, shards, 0)
    wanted = expected.format(n=of, n1=of + 1)
    assert any(wanted in problem for problem in found), (wanted, found)


def test_every_plant_is_made_with_every_count_that_can_make_it():
    """The plants that need a second shard are the only ones left out of a count, and only of one shard."""
    assert {name for name, (fewest, _, _) in PLANTS.items() if fewest > 1} == {
        "a test kept by two shards", "two different collections", "one shard collected nothing",
        "a test reported by two shards"}
    assert max(fewest for fewest, _, _ in PLANTS.values()) == 2


@pytest.mark.parametrize("of", COUNTS)
def test_the_check_exits_with_its_verdict(of, recordings, tmp_path):
    recorded = recordings(of)
    of = str(of)
    assert check_shards.main(["--shards", of, "--max-skipped", "0", *map(str, recorded)]) == 0
    assert check_shards.main(["--shards", of, "--max-skipped", "0", *map(str, recorded[:-1])]) == 1
    copies = [_copy(d, tmp_path) for d in recorded]
    _edit_report(copies[0], lambda root: _skip(_first_case(root)))
    assert check_shards.main(["--shards", of, "--max-skipped", "0", *map(str, copies)]) == 1
    assert check_shards.main(["--shards", of, "--max-skipped", "1", *map(str, copies)]) == 0


def test_the_deal_is_a_function_of_the_table_and_balanced():
    weights = _sharding.load_weights()
    for of in (1, 2, 3, 4, 6, 12):
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


#: Two runner CPUs of one flag set and two models, as /proc/cpuinfo prints them (the flags are shortened).
ZEN3 = ("processor\t: 0\nvendor_id\t: AuthenticAMD\ncpu family\t: 25\nmodel\t\t: 1\n"
        "model name\t: AMD EPYC 7763 64-Core Processor\nflags\t\t: fpu sse2 avx2\n\nprocessor\t: 1\nmodel\t\t: 1\n")
ZEN4 = ZEN3.replace("model\t\t: 1\n", "model\t\t: 17\n", 1).replace("7763", "9V74")


def test_the_cache_key_follows_the_cpus_model_and_features_not_their_order():
    one = cache_mode.cpu_digest(ZEN3)
    assert one == cache_mode.cpu_digest(ZEN3.replace("fpu sse2 avx2", "avx2 fpu sse2"))
    assert one != cache_mode.cpu_digest(ZEN3.replace("fpu sse2 avx2", "fpu sse2 avx2 avx512f"))


def test_control_two_cpu_models_with_the_same_flags_get_different_keys():
    assert ZEN3.split("flags")[1] == ZEN4.split("flags")[1]
    assert cache_mode.cpu_digest(ZEN3) != cache_mode.cpu_digest(ZEN4)
    # The marketing name alone, and the family and model numbers alone, each tell them apart.
    assert cache_mode.cpu_digest(ZEN3) != cache_mode.cpu_digest(ZEN3.replace("7763", "9V74"))
    assert cache_mode.cpu_digest(ZEN3) != cache_mode.cpu_digest(ZEN3.replace("model\t\t: 1\n", "model\t\t: 17\n", 1))


def test_a_second_processors_lines_do_not_move_the_key():
    assert cache_mode.cpu_digest(ZEN3) == cache_mode.cpu_digest(ZEN3.replace("processor\t: 1\nmodel\t\t: 1",
                                                                              "processor\t: 1\nmodel\t\t: 99"))


def test_the_weights_read_a_reports_node_ids_as_pytest_writes_them():
    report = ET.fromstring(
        '<testsuites><testsuite><testcase classname="tests.saved.test_x" name="test_a[1-2]" file="tests/saved/'
        'test_x.py" time="3.5"/><testcase classname="tests.saved.test_x.TestMachine" name="runTest" file="tests/saved/'
        'test_x.py" time="1"/></testsuite></testsuites>')
    assert [shard_weights.node_id(case) for case in report.iter("testcase")] == [
        "tests/saved/test_x.py::test_a[1-2]", "tests/saved/test_x.py::TestMachine::runTest"]
