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
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_shards = _script("check_shards")
latest_attempt = _script("latest_attempt")
check_shard_jobs = _script("check_shard_jobs")
cache_mode = _script("compilation_cache_mode")
shard_weights = _script("shard_weights")
shard_matrix = _script("shard_matrix")
#: The shard counts the recorded artifacts are made for: one, a pair, a count that does not divide the files, and
#: today's default. A lane may run as any whole number of shards from ``scripts/shard_matrix.py``'s bounds.
COUNTS = (1, 2, 3, 6)
def shard_files(of: int) -> list[str]:
    """``of`` test file names, none in the project's table: each is a new file, and the plugin places new files one at a
    time on the lightest shard, so a project of these files has a shard that holds each (the table's two listed files,
    which the project does not hold, weigh what a new file is assumed to, and so take two shards' places first)."""
    return [f"test_s{number}.py" for number in range(of)]


def record_shards(tmp_path: Path, of: int) -> list[Path]:
    """Run a project of ``of`` small test files once per shard of ``of``, as a shard job does, with the shard plugin of
    ``tests/_sharding.py``, on a table that lists none of the project's files: its record, its JUnit report and a passing
    outcome in a directory each. Every shard holds one file of two tests."""
    project = tmp_path / "project"
    project.mkdir()
    table = _split_table(tmp_path, {"seconds": {"test_listed_a.py": 1, "test_listed_b.py": 1}})
    (project / "conftest.py").write_text(SPLIT_CONFTEST.format(tests=str(ROOT / "tests"), table=str(table)),
                                         encoding="utf-8")
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


def test_a_file_the_table_gives_a_shard_of_its_own_has_it_to_itself_and_would_not_without():
    """The arm battery's longest tests are a worker's share each; beside other files they start last and leave the
    rest idle. The control: dealt by its seconds alone, the same file shares its shard."""
    table = json.loads(_sharding.WEIGHTS_FILE.read_text(encoding="utf-8"))
    alone = list(table["alone"])
    assert alone and all(table["alone"][path] for path in alone), "each file that holds a shard says why"
    weights = _sharding.load_weights()
    for of in (*range(2, 13), 16, 20, 24, 32):  # the cap of scripts/shard_matrix.py is 32
        dealt = _sharding.deal(weights, of)
        for path in alone:
            assert [other for other, shard in dealt.items() if shard == dealt[path]] == [path]
        assert set(dealt.values()) == set(range(1, of + 1))
    plain = _sharding.deal(table["seconds"], 9)
    assert any(shard == plain[alone[0]] and other != alone[0] for other, shard in plain.items())


def test_a_files_tests_share_its_shard_and_a_file_nobody_passed_as_new_goes_by_its_hash():
    dealt = _sharding.deal(_sharding.load_weights(), 6)
    listed = next(iter(dealt))
    assert {_sharding.shard_of(f"{listed}::test_{i}[{i}]", 6, dealt) for i in range(9)} == {dealt[listed]}
    added = "tests/test_new.py"
    assert added not in dealt and _sharding.shard_of(f"{added}::test_added", 6, dealt) == _sharding.by_hash(added, 6)


def test_a_file_the_table_does_not_list_goes_to_the_lightest_shard_at_the_assumed_weight_not_by_its_hash(tmp_path):
    """The table lists three files (100, 60 and 50 s on shards 1, 2 and 3); the weight a new file is assumed to have is the
    table's 90th percentile (nearest rank), 100 s. The new files, in the order of their paths, take the lightest shard each
    time: shard 3 (50 s, then 150), then 2 (60 s, then 160), then 1 (100 s). The control is the hash, which places them
    otherwise: a file by hash ignores how loaded its shard already is, which is how a push's shard ran past its limit."""
    path = _split_table(tmp_path, {"seconds": {"tests/a.py": 100, "tests/b.py": 60, "tests/c.py": 50}})
    weights, assumed = _sharding.load_weights(path), _sharding.new_file_weight(path)
    assert assumed == 100
    new = ["tests/new_1.py", "tests/new_2.py", "tests/new_3.py"]
    dealt = _sharding.deal(weights, 3, new, assumed)
    assert [dealt[file] for file in ("tests/a.py", "tests/b.py", "tests/c.py")] == [1, 2, 3]
    assert [dealt[file] for file in new] == [3, 2, 1]
    assert [_sharding.shard_of(f"{file}::test_x", 3, dealt) for file in new] == [3, 2, 1]
    assert [_sharding.by_hash(file, 3) for file in new] != [3, 2, 1], "the control: by hash these would not be placed so"
    assert dealt == _sharding.deal(weights, 3, reversed(new), assumed), "the order the files are given in moves none"
    assert _sharding.deal(weights, 3, ["tests/a.py"], assumed)["tests/a.py"] == 1, "a listed file is never re-placed"


def test_the_table_assumes_for_a_new_file_the_weight_of_its_90th_percentile_file():
    seconds = _sharding.load_seconds()
    ranked = sorted(seconds.values())
    assumed = _sharding.new_file_weight()
    assert assumed == ranked[-(-9 * len(ranked) // 10) - 1]
    assert sum(1 for weight in ranked if weight > assumed) <= 0.1 * len(ranked) <= sum(1 for w in ranked if w >= assumed)


def test_a_new_file_never_lands_on_the_shard_of_a_file_the_table_gives_to_itself_and_goes_to_the_lightest():
    """With the committed table, at the counts a lane may run as: a hundred new files are placed after the listed ones,
    none on a shard that holds a file alone, and the first on the shard the listed files load least."""
    weights, assumed = _sharding.load_weights(), _sharding.new_file_weight()
    alone = list(_sharding.load_alone())
    new = [f"tests/new_{number:03d}.py" for number in range(100)]
    for of in (*range(2, 13), 16, 20, 24, 32):
        listed = _sharding.deal(weights, of)
        dealt = _sharding.deal(weights, of, new, assumed)
        assert all(dealt[path] == listed[path] for path in listed), "the listed files stay where the table puts them"
        taken = {dealt[path] for path in alone}
        assert not taken & {dealt[file] for file in new}
        load = [sum(w for path, w in weights.items() if listed[path] == shard) for shard in range(1, of + 1)]
        assert dealt[new[0]] == 1 + min(range(of), key=lambda k: (load[k], k))
        placed = [sum(assumed for file in new if dealt[file] == shard) for shard in range(1, of + 1)]
        if of > 2:  # the new files are spread: the shards they land on end within one assumed weight of each other
            ends = [load[k] + placed[k] for k in range(of) if k + 1 not in taken]
            assert max(ends) - min(ends) <= assumed


SPLIT_TABLE = {"seconds": {"tests/big.py": 60, "tests/mid.py": 20, "tests/small.py": 10},
               "split": {"tests/big.py": {f"test_{i}[{i}]": 10 for i in range(6)}}}

#: A project's conftest that registers the shard plugin with a given table, as ``_sharding.register`` does with the committed one.
SPLIT_CONFTEST = """import sys
from pathlib import Path
sys.path.insert(0, {tests!r})
import _sharding


def pytest_addoption(parser):
    parser.addoption("--mime-shard")
    parser.addoption("--mime-shard-record")


def pytest_configure(config):
    if config.getoption("--mime-shard"):
        shard, of = _sharding.parse(config.getoption("--mime-shard"))
        table = Path({table!r})
        config.pluginmanager.register(_sharding.Plugin(
            shard, of, _sharding.load_weights(table), config.getoption("--mime-shard-record"),
            _sharding.load_split(table), _sharding.new_file_weight(table)))
"""


def _split_table(tmp_path: Path, table: dict | None = None) -> Path:
    path = tmp_path / "w.json"
    path.write_text(json.dumps(table or SPLIT_TABLE), encoding="utf-8")
    return path


def test_a_split_file_is_dealt_by_test_and_stably_and_an_unlisted_test_of_it_goes_by_its_node_id(tmp_path):
    """The heaviest file, split: its listed tests are dealt one by one beside the other files, longest first, and a test
    the table does not list goes by the hash of its node id, so adding a test moves no other. The control is the same
    file unsplit, which stays whole on one shard."""
    path = _split_table(tmp_path)
    weights, split = _sharding.load_weights(path), set(_sharding.load_split(path))
    assert split == {"tests/big.py"} and "tests/big.py" not in weights
    assert {f"tests/big.py::test_{i}[{i}]" for i in range(6)} <= set(weights)
    dealt = _sharding.deal(weights, 3)
    where = [_sharding.shard_of(f"tests/big.py::test_{i}[{i}]", 3, dealt, split) for i in range(6)]
    assert set(where) == {1, 2, 3}
    assert dealt == _sharding.deal(dict(reversed(list(weights.items()))), 3)
    added = "tests/big.py::test_added[x]"
    assert added not in dealt and _sharding.shard_of(added, 3, dealt, split) == _sharding.by_hash(added, 3)
    assert [_sharding.shard_of(f"tests/big.py::test_{i}[{i}]", 3, dealt, split) for i in range(6)] == where
    whole = _sharding.deal(SPLIT_TABLE["seconds"], 3)
    assert {_sharding.shard_of(f"tests/big.py::test_{i}[{i}]", 3, whole) for i in range(6)} == {whole["tests/big.py"]}


def test_a_split_file_running_for_real_is_kept_by_the_shards_between_them_and_the_check_passes_them(tmp_path):
    """The split, in the plugin and in pytest: three shards of a project whose big file is split. Every test is run by
    one shard, the big file's by more than one, and ``scripts/check_shards.py`` passes the lane."""
    project = tmp_path / "project"
    project.mkdir()
    table = _split_table(tmp_path, {"seconds": {"big.py": 60, "mid.py": 20, "small.py": 10},
                                    "split": {"big.py": {f"test_big[{i}]": 10 for i in range(6)}}})
    (project / "conftest.py").write_text(SPLIT_CONFTEST.format(tests=str(ROOT / "tests"), table=str(table)),
                                         encoding="utf-8")
    (project / "test_big.py").write_text("import pytest\n\n\n@pytest.mark.parametrize('i', range(6))\n"
                                         "def test_big(i):\n    pass\n", encoding="utf-8")
    (project / "test_mid.py").write_text("def test_mid():\n    pass\n", encoding="utf-8")
    (project / "test_small.py").write_text("def test_small():\n    pass\n", encoding="utf-8")
    table.write_text(json.dumps({"seconds": {"test_big.py": 60, "test_mid.py": 20, "test_small.py": 10},
                                 "split": {"test_big.py": {f"test_big[{i}]": 10 for i in range(6)}}}),
                     encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_XDIST", "MIME_CI_PLANT", "MIME_TEST_JAX"))}
    directories = []
    for shard in (1, 2, 3):
        directory = tmp_path / f"shard-{shard}"
        directory.mkdir()
        done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
                               "junit_family=xunit1", "--mime-shard", f"{shard}/3",
                               "--mime-shard-record", str(directory / "shard.json"),
                               f"--junitxml={directory / f'test-results-shard{shard}.xml'}"],
                              cwd=project, env=env, capture_output=True, text=True, check=False, timeout=300)
        assert done.returncode == 0, done.stdout + done.stderr
        (directory / "outcome.txt").write_text("success\n", encoding="utf-8")
        directories.append(directory)
    kept = [json.loads((d / "shard.json").read_text(encoding="utf-8"))["selected"] for d in directories]
    assert sum(len(each) for each in kept) == 8 and len({nodeid for each in kept for nodeid in each}) == 8
    assert sum(any(nodeid.startswith("test_big.py") for nodeid in each) for each in kept) > 1
    assert check_shards.problems(directories, 3, 0) == []


def test_the_table_cannot_name_a_split_file_it_does_not_list_or_one_that_is_also_alone(tmp_path):
    base = {"seconds": {"a": 5, "b": 5}}
    for extra in ({"split": {"c": {"t": 1}}}, {"split": {"a": {}}}, {"split": {"a": {"t": 0}}},
                  {"split": {"a": {"t": 1.5}}}, {"split": ["a"]}, {"split": {"a": {"t": 1}}, "alone": {"a": "why"}}):
        path = _split_table(tmp_path, {**base, **extra})
        with pytest.raises(pytest.UsageError):
            _sharding.load_weights(path)


def test_no_file_heavier_than_a_shards_mean_is_dealt_whole_at_the_default_count():
    """The reason a file is split: whole, it would be a shard's floor however many shards there are. When the suite
    grows a file past the mean at ``scripts/shard_matrix.py``'s default, this fails and names it, and
    ``scripts/shard_weights.py --split FILE`` deals it by test (it prints the same list)."""
    table = json.loads(_sharding.WEIGHTS_FILE.read_text(encoding="utf-8"))
    alone, split = set(table.get("alone", {})), set(table.get("split", {}))
    ordinary = {path: seconds for path, seconds in table["seconds"].items() if path not in alone}
    mean = sum(ordinary.values()) / (shard_matrix.DEFAULT - len(alone))
    assert sorted(path for path, seconds in ordinary.items() if seconds > mean and path not in split) == []
    for file, tests in table.get("split", {}).items():
        assert abs(sum(tests.values()) - table["seconds"][file]) <= 0.1 * table["seconds"][file] + len(tests)


def _lane_reports(tmp_path: Path) -> list[Path]:
    """Two lanes of one shard each: a file of six 10 s tests, a 20 s file and a 10 s file."""
    cases = ([("tests/h.py", f"test_h[{i}]", 10) for i in range(6)] + [("tests/m.py", "test_m", 20),
                                                                      ("tests/s.py", "test_s", 10)])
    directories = []
    for lane in ("durations-a", "durations-b"):
        directory = tmp_path / f"{lane}-shard-1-attempt-1"
        directory.mkdir()
        body = "".join(f'<testcase classname="{file[:-3].replace("/", ".")}" name="{name}" file="{file}" time="{t}"/>'
                       for file, name, t in cases)
        (directory / "r.xml").write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>", encoding="utf-8")
        directories.append(directory)
    return directories


def test_the_weights_script_splits_the_files_it_is_told_to_keeps_them_split_and_warns_of_the_heavy_ones(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shard_weights, "TABLE", tmp_path / "table.json")
    reports = [str(path) for path in _lane_reports(tmp_path)]
    assert shard_weights.main([*reports, "--shards", "4", "--source", "x"]) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "tests/h.py" in out and "split" not in json.loads((tmp_path / "table.json").read_text())
    assert shard_weights.main([*reports, "--shards", "4", "--split", "tests/h.py", "--source", "x"]) == 0
    out = capsys.readouterr().out
    table = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert "WARNING" not in out and table["split"] == {"tests/h.py": {f"test_h[{i}]": 10 for i in range(6)}}
    assert table["seconds"]["tests/h.py"] == 60
    assert "predicted seconds per shard: 30, 20, 20, 20" in out
    assert shard_weights.main([*reports, "--shards", "4", "--source", "x"]) == 0  # a file stays split
    assert "split" in json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))


def test_the_weights_script_lists_every_file_the_reports_hold_so_that_a_file_the_table_lacks_is_a_new_one(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shard_weights, "TABLE", tmp_path / "table.json")
    reports = [str(path) for path in _lane_reports(tmp_path)]
    for directory in (tmp_path / "durations-a-shard-1-attempt-1", tmp_path / "durations-b-shard-1-attempt-1"):
        report = directory / "r.xml"
        report.write_text(report.read_text(encoding="utf-8").replace(
            "</testsuite>", '<testcase classname="tests.quick" name="test_q" file="tests/quick.py" time="0.2"/></testsuite>'),
            encoding="utf-8")
    assert shard_weights.main([*reports, "--shards", "4", "--source", "x"]) == 0
    table = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert table["seconds"]["tests/quick.py"] == 1 and set(table["seconds"]) == {
        "tests/h.py", "tests/m.py", "tests/s.py", "tests/quick.py"}
    assert shard_weights.main([*reports, "--shards", "4", "--over", "5", "--source", "x"]) == 0
    assert "tests/quick.py" not in json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))["seconds"]


@pytest.mark.parametrize("spec", ["0/4", "5/4", "1/0", "a/4", "1/4/2", " 1/4", "1/4:weighted"])
def test_a_malformed_shard_is_a_usage_error(spec):
    with pytest.raises(pytest.UsageError):
        _sharding.parse(spec)


def test_a_bad_table_is_an_error_never_an_unweighted_split(tmp_path):
    for content in ("", "{}", '{"seconds": {}}', '{"seconds": {"a": 1.5}}', '{"seconds": {"a": 0}}',
                    '{"seconds": {"a": 1}, "alone": {"b": "x"}}', '{"seconds": {"a": 1}, "alone": {"a": ""}}',
                    '{"seconds": {"a": 1}, "alone": ["a"]}', '{"seconds": {"a": 1}, "split": {"b": {"t": 1}}}'):
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


@pytest.mark.parametrize("event, choice", [("push", ""), ("workflow_dispatch", "cold"), ("workflow_dispatch", "warm"),
                                           ("pull_request", "")])
def test_a_lane_that_keeps_no_compilation_cache_runs_cold_and_saves_nothing_in_every_event(event, choice):
    """Python 3.13's shards (a decision of 2026-11-08): the control is the same event with the lane keeping a cache."""
    assert cache_mode.decide(event, choice, "feat: a change", [], keeps_cache=False)[:2] == ("cold", False)


def test_the_same_push_saves_a_cache_where_its_lane_keeps_one():
    assert cache_mode.decide("push", "", None, [])[:2] == ("cold", True)
    assert cache_mode.decide("push", "", None, [], keeps_cache=False)[:2] == ("cold", False)


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


def test_the_weights_count_a_shard_that_ran_again_once_by_its_last_attempt_and_group_the_lanes_by_name(tmp_path):
    """CI names a shard's artifact ``<lane>-shard-<n>-attempt-<k>``: the lane is what is left of that name, so a lane's
    shards are one lane (the weights' median is over lanes), and a lost first attempt is not counted beside its re-run."""
    names = ["durations-a-python-3.12-shard-1-attempt-1", "durations-a-python-3.12-shard-2-attempt-1",
             "durations-a-python-3.12-shard-2-attempt-2", "durations-b-shard-1", "durations-b-shard-2-attempt-1"]
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "report.xml").write_text("<testsuites/>", encoding="utf-8")
    lanes = shard_weights.lanes_of([tmp_path / name for name in names])
    assert [sorted(path.parent.name for path in lane) for lane in lanes] == [
        ["durations-a-python-3.12-shard-1-attempt-1", "durations-a-python-3.12-shard-2-attempt-2"],
        ["durations-b-shard-1", "durations-b-shard-2-attempt-1"]]


# A re-run of one shard job (scripts/latest_attempt.py): the run then holds that shard's report twice, one per attempt.

LANE = "durations-default-python-3.12"
#: The step runs after a red shard, and in a reduced run only for the lane that ran (tests/test_ci_ready_label.py).
LANE_RAN = "${{ !cancelled() && env.MIME_CI_RUNS == 'true' }}"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _downloaded(tmp_path: Path, recorded: list[Path], attempts: dict[int, list[str]]) -> Path:
    """What the lane's job downloads: a directory per shard and attempt. ``attempts[shard]`` lists the outcomes of the
    shard's attempts, in order; a shard not in it ran once and passed. A failed attempt is a failed job's outcome and a
    failing test in its report, as ``job.status`` and the JUnit report of a shard whose test failed."""
    downloaded = tmp_path / "shards"
    downloaded.mkdir()
    for number, directory in enumerate(recorded, start=1):
        for attempt, outcome in enumerate(attempts.get(number, ["success"]), start=1):
            copy = _copy(directory, downloaded)
            copy = copy.rename(downloaded / f"{LANE}-shard-{number}-attempt-{attempt}")
            (copy / "outcome.txt").write_text(f"{outcome}\n", encoding="utf-8")
            if outcome != "success":
                _edit_report(copy, _fail_case)
    return downloaded


def _lane_passes(downloaded: Path, of: int) -> bool:
    return check_shards.main(["--shards", str(of), "--max-skipped", "0",
                              *map(str, sorted(path for path in downloaded.iterdir() if path.is_dir()))]) == 0


def test_control_without_the_pruning_a_shard_re_run_alone_leaves_the_lane_red(recordings, tmp_path):
    """The defect: shard 2 failed and passed on its re-run, and the lane checked both attempts' reports."""
    downloaded = _downloaded(tmp_path, recordings(3), {2: ["failure", "success"]})
    assert not _lane_passes(downloaded, 3)


def test_a_shard_re_run_alone_clears_the_lane(recordings, tmp_path):
    downloaded = _downloaded(tmp_path, recordings(3), {2: ["failure", "success"]})
    assert latest_attempt.main([str(downloaded)]) == 0
    assert sorted(path.name for path in downloaded.iterdir()) == [
        f"{LANE}-shard-1-attempt-1", f"{LANE}-shard-2-attempt-2", f"{LANE}-shard-3-attempt-1"]
    assert _lane_passes(downloaded, 3)


@pytest.mark.parametrize("attempts", [["failure", "failure"], ["failure", "cancelled"], ["success", "failure"],
                                      ["failure", "success", "failure"]],
                         ids=lambda each: "-then-".join(each))
def test_a_re_run_does_not_hide_a_failure_that_persists(attempts, recordings, tmp_path):
    """The newest attempt is the one read: a shard whose newest attempt did not pass keeps the lane red, whatever an
    older attempt did."""
    downloaded = _downloaded(tmp_path, recordings(3), {2: attempts})
    assert latest_attempt.main([str(downloaded)]) == 0
    assert not _lane_passes(downloaded, 3)


def test_the_newest_attempt_is_the_highest_number_not_the_last_name(recordings, tmp_path):
    downloaded = _downloaded(tmp_path, recordings(2), {1: ["failure"] * 9 + ["success"]})
    assert latest_attempt.main([str(downloaded)]) == 0
    assert (downloaded / f"{LANE}-shard-1-attempt-10").is_dir()
    assert sorted(path.name for path in downloaded.iterdir()) == [
        f"{LANE}-shard-1-attempt-10", f"{LANE}-shard-2-attempt-1"]
    assert _lane_passes(downloaded, 2)


def test_a_report_it_cannot_place_fails_the_pruning(recordings, tmp_path):
    downloaded = _downloaded(tmp_path, recordings(2), {})
    (downloaded / f"{LANE}-shard-2-attempt-1").rename(downloaded / f"{LANE}-shard-2")
    assert latest_attempt.main([str(downloaded)]) == 1
    assert latest_attempt.main([str(downloaded / "absent")]) == 1


def _steps(job: dict) -> list[dict]:
    return job["steps"]


# The pruning keeps the newest attempt of a shard that was re-run, and cannot see a re-run that failed before it
# uploaded (the first attempt's report is then the newest): the default lane's job reads the shard jobs' result too.
def _workflow_problems(workflow: dict) -> list[str]:
    """What is wrong with how the workflow names a shard's report by attempt and prunes them in each lane's job."""
    found: list[str] = []
    jobs = workflow["jobs"]
    for tests, lane in (("default-tests", "default"), ("openmaterica-main-tests", "openmaterica-main"),
                        ("maddening-main-tests", "maddening-main")):
        names = [step["with"]["name"] for step in _steps(jobs[tests]) if "upload-artifact" in step.get("uses", "")]
        if len(names) != 1 or not names[0].endswith("-shard-${{ matrix.shard }}-attempt-${{ github.run_attempt }}"):
            found.append(f"{tests}: the shard's report is not named by the run attempt ({names})")
        steps = _steps(jobs[lane])
        pruning = [at for at, step in enumerate(steps) if "scripts/latest_attempt.py" in str(step.get("run", ""))]
        checking = [at for at, step in enumerate(steps) if "scripts/check_shards.py" in str(step.get("run", ""))]
        downloading = [at for at, step in enumerate(steps) if "download-artifact" in step.get("uses", "")]
        if len(pruning) != 1 or len(checking) != 1 or len(downloading) != 1 or not downloading[0] < pruning[0] < checking[0]:
            found.append(f"{lane}: the newest attempt is not kept between the download and the check")
    default_steps = _steps(jobs["default"])
    if not any("scripts/check_shard_jobs.py" in str(step.get("run", "")) and "default tests (Python" in str(step.get("run", ""))
               and "filter=latest" in str(step.get("env", {}))
               and "pipefail" in str(step.get("run", "")) and step.get("if") in ("${{ !cancelled() }}", LANE_RAN)
               for step in default_steps):
        found.append("default: no step fails the lane when a shard job's latest attempt did not pass")
    if (jobs["default"].get("permissions") or {}).get("actions") != "read":
        found.append("default: cannot read the run's jobs (permissions: actions: read)")
    return found


def test_the_workflow_names_a_report_by_its_attempt_and_prunes_before_it_checks():
    assert _workflow_problems(yaml.safe_load(CI.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("old, new, expected", [
    ("-shard-${{ matrix.shard }}-attempt-${{ github.run_attempt }}\n          path: ${{ runner.temp }}/report\n"
     "          if-no-files-found: error\n\n  # The lane's result",
     "-shard-${{ matrix.shard }}\n          path: ${{ runner.temp }}/report\n"
     "          if-no-files-found: error\n\n  # The lane's result", "default-tests: the shard's report is not named"),
    ("run: python3 scripts/latest_attempt.py \"$RUNNER_TEMP\"/shards\n      - name: Gather the default lane's JUnit",
     "run: 'true'\n      - name: Gather the default lane's JUnit", "default: the newest attempt is not kept"),
    ("run: |\n          set -o pipefail\n          python3 scripts/run_jobs.py \"$JOBS_URL\" \\\n            | python3 scripts/check_shard_jobs.py --prefix \"default tests",
     "run: |\n          python3 scripts/run_jobs.py \"$JOBS_URL\" \\\n            | python3 scripts/check_shard_jobs.py --prefix \"default tests",
     "default: no step fails the lane when a shard job's latest attempt"),
    ("      actions: read\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    strategy:\n",
     "    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    strategy:\n", "default: cannot read the run's jobs"),
], ids=["a-report-named-without-its-attempt", "no-pruning", "the-jobs-pipe-without-pipefail", "no-permission-to-read-jobs"])
def test_control_a_break_in_the_attempt_wiring_is_named(old, new, expected):
    text = CI.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    found = _workflow_problems(yaml.safe_load(text.replace(old, new)))
    assert any(expected in problem for problem in found), found


def _jobs(conclusions: dict[int, str | None], python: str = "3.12", of: int = 9) -> list[str]:
    lines = [json.dumps({"name": f"default tests (Python {python}, shard {n} of {of})", "conclusion": c})
             for n, c in conclusions.items()]
    return [*lines, json.dumps({"name": "default (Python 3.12)", "conclusion": None}),
            json.dumps({"name": "shard count", "conclusion": "success"})]


def test_the_shard_jobs_at_their_latest_attempts_all_passed():
    assert check_shard_jobs.problems(_jobs({n: "success" for n in range(1, 10)}), "default tests (Python 3.12, ", 9) == []


@pytest.mark.parametrize("bad", ["failure", "cancelled", "skipped", None], ids=str)
def test_control_a_shard_job_that_did_not_pass_fails_the_lane_even_with_a_passing_report_beside_it(bad):
    """The case the pruning cannot see: a re-run that failed before it uploaded leaves the first attempt's passing report
    as the newest, and the job list says what the job ended as."""
    found = check_shard_jobs.problems(_jobs({n: "success" if n != 7 else bad for n in range(1, 10)}),
                                      "default tests (Python 3.12, ", 9)
    assert any("shard 7 of 9" in problem and repr(bad) in problem for problem in found), found


def test_another_pythons_failed_shard_does_not_fail_this_lane_and_a_missing_job_does():
    lines = _jobs({n: "success" for n in range(1, 10)}) + _jobs({2: "failure"}, python="3.14")
    assert check_shard_jobs.problems(lines, "default tests (Python 3.12, ", 9) == []
    assert any("8 shard jobs" in problem for problem in check_shard_jobs.problems(
        _jobs({n: "success" for n in range(1, 9)}), "default tests (Python 3.12, ", 9))
