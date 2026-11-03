"""The saved form's specification: each normative statement cites its cases, every case resolves through pytest's
collection, and the statements not yet covered are a list (``mime.docs._specification``).

The real specification passes against a collection of the files it cites. Each control doctors a copy of the
specification's text or of its list and runs the real check on it; the inputs are deterministic, so each fails on every
run.
"""

from pathlib import Path

import pytest
from checkout import PLANTED_LANES, planted_checkout

from mime.docs._check import LISTS
from mime.docs._evidence import collect
from mime.docs._lists import read_list
from mime.docs._specification import SPECIFICATIONS, specification_report, statements

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / SPECIFICATIONS[0]
LIST = ROOT / "docs" / "ratchets" / "not_yet_covered.txt"


@pytest.fixture(scope="module")
def collection():
    held, _ = statements(SPEC.read_text(encoding="utf-8"), SPECIFICATIONS[0])
    files = sorted({case.split("::", 1)[0] for statement in held for case in statement.cases})
    return collect(ROOT, files)


def doctored(tmp_path: Path, spec: str | None = None, listed: str | None = None) -> Path:
    """A checkout holding the specification and its list (as given, or as they are), its tests the real ones."""
    (tmp_path / SPECIFICATIONS[0]).parent.mkdir(parents=True)
    (tmp_path / SPECIFICATIONS[0]).write_text(spec if spec is not None else SPEC.read_text(encoding="utf-8"),
                                              encoding="utf-8")
    (tmp_path / "docs" / "ratchets").mkdir(parents=True)
    (tmp_path / "docs" / "ratchets" / LIST.name).write_text(listed if listed is not None else
                                                            LIST.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
    return tmp_path


def problems_of(root: Path, collection) -> list[str]:
    return specification_report(root, collection, read_list(root / "docs" / "ratchets" / LIST.name,
                                                            LISTS["not_yet_covered"])).problems


def test_the_specification_cites_resolving_cases_and_lists_what_it_does_not_cover(collection):
    report = specification_report(ROOT, collection, read_list(LIST, LISTS["not_yet_covered"]))
    assert report.problems == []
    held = report.statements[SPECIFICATIONS[0]]
    assert len(held) == len({s.key for s in held}) and sum(not s.covered for s in held) == len(read_list(LIST))
    assert collection.errors == ()


def line_of(marker: str) -> int:
    return next(number for number, line in enumerate(SPEC.read_text(encoding="utf-8").splitlines(), start=1)
                if marker in line)


def edit(old: str, new: str) -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_control_a_statement_without_its_cases_fails(tmp_path, collection):
    spec = edit("  *Cases:* `tests/scene/test_saved_scene.py::test_a_changed_asset_file_is_refused_at_load`.\n", "")
    problems = problems_of(doctored(tmp_path, spec), collection)
    assert [p for p in problems if "[files.layer]" in p] == [
        f"{SPECIFICATIONS[0]}:{line_of('[files.layer]')}: the statement [files.layer] ends with no '*Cases:*': "
        f"cite the tests that hold it, or write '*Cases:* not yet covered.'"], problems


def test_control_a_case_that_is_not_collected_fails(tmp_path, collection):
    spec = edit("test_a_changed_asset_file_is_refused_at_load`", "test_a_changed_asset_file_is_refused_at_loading`")
    problems = problems_of(doctored(tmp_path, spec), collection)
    assert any("[files.layer] cites tests/scene/test_saved_scene.py::test_a_changed_asset_file_is_refused_at_loading"
               in p and "is not collected" in p for p in problems), problems


def test_control_an_id_used_twice_fails(tmp_path, collection):
    spec = edit("- **[files.layer]**", "- **[files.stage]**")
    problems = problems_of(doctored(tmp_path, spec), collection)
    assert any("the statement id [files.stage] is used 2 times" in p for p in problems), problems


def test_control_a_statement_not_yet_covered_and_not_listed_fails_and_a_covered_one_listed_fails(tmp_path, collection):
    listed = LIST.read_text(encoding="utf-8").replace("rebuild.namespace  #", "rebuild.namespaces  #")
    problems = problems_of(doctored(tmp_path, listed=listed), collection)
    assert any(p.startswith("rebuild.namespace: a normative statement not yet covered, and it is not on")
               for p in problems), problems
    assert any("'rebuild.namespaces' now passes; delete this line" in p for p in problems), problems


def test_control_cases_that_are_not_node_ids_in_backticks_fail(tmp_path, collection):
    spec = edit("  *Cases:* `tests/scene/test_saved_scene.py::test_a_changed_asset_file_is_refused_at_load`.\n",
                "  *Cases:* the scene's tests.\n")
    problems = problems_of(doctored(tmp_path, spec), collection)
    assert any("the cases of [files.layer] are pytest node ids in backticks" in p for p in problems), problems


def lanes_checkout(tmp_path: Path, listed: str = "") -> Path:
    """A checkout of its own: three statements citing a planted test CI runs, an xfail test and a slow test, and the
    list of statements not covered (``listed``)."""
    spec = "# Planted\n\n" + "".join(
        f"- **[planted.{name}]** A sentence.\n  *Cases:* `tests/test_planted.py::test_{name}`.\n"
        for name in ("runs", "xfail", "slow"))
    return planted_checkout(tmp_path, {
        str(SPECIFICATIONS[0]): spec, "docs/ratchets/not_yet_covered.txt": f"# The planted list.\n{listed}",
        "tests/test_planted.py": "import pytest\n\n\n" + "\n\n".join(PLANTED_LANES.values())})


def lanes_report(root: Path):
    return specification_report(root, collect(root, ["tests/test_planted.py"]),
                                read_list(root / "docs" / "ratchets" / LIST.name, LISTS["not_yet_covered"]))


def test_control_a_statement_citing_only_an_xfail_or_only_a_slow_test_is_not_covered(tmp_path):
    """Evidence is a test CI runs: a statement whose only case is expected to fail, or is slow, is shown as citing
    only tests CI does not run, never as covered, and fails until it is listed; the one citing a test CI runs is the
    baseline, and passes."""
    report = lanes_report(lanes_checkout(tmp_path))
    flagged = sorted(key for key in ("runs", "xfail", "slow")
                     if any(f"[planted.{key}] cites only tests CI does not run" in p for p in report.problems))
    assert flagged == ["slow", "xfail"], report.problems
    assert any("tests/test_planted.py::test_xfail: not run in CI (xfail: expected to fail)" in p
               for p in report.problems), report.problems
    assert len(report.problems) == 2, report.problems
    assert "1 covered by a test CI runs" in report.summary[0] and "2 citing only tests CI does not run" in \
        report.summary[0], report.summary


def test_a_statement_citing_only_tests_ci_does_not_run_passes_once_listed_with_its_reason(tmp_path):
    listed = "planted.slow  # its case is slow\nplanted.xfail  # its case is expected to fail\n"
    assert lanes_report(lanes_checkout(tmp_path, listed)).problems == []
