"""The evidence counts page (``mime.docs._evidence_page``): every claim and every specification statement with its tests'
outcomes in the default lane's JUnit report, read through the resolver and never resolved again.

The planted checkouts hold a registry, a specification, the page's source and floor, and a test file of their own that a
real pytest run reports on as CI's default lane does (``-m "not slow"``), so the outcomes are pytest's, not a fixture's.
The gate's planted breaks, each of which fails on every run (they are deterministic: nothing is drawn):

- the fixed sentence removed from the source, and altered by one word (the build fails; the page without it fails);
- a claim whose cited test is deleted (its row fails the build);
- a claim resting only on a skipped, xfail, slow or marked-skip test is not shown as passed, nor counted;
- an acceptance claim placed under an accuracy heading;
- the count floor undercut (a pinned claim or statement removed);
- the page edited by hand, one character, and a copy of it committed under ``docs/``.

The fix round's (each fails without its fix, every run): a bound's held-out test read like any other evidence; the
floor lowered against ``--base``; the page's source deleted; a shard's missing report read as "not read"; a test id
twice in the reports taking the worse; the Device column saying it is declared.

Each has a passing arm that passes on every run, so that none of them passes for want of a page."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from checkout import ROOT, copy_checkout, docs_check, docs_run, edit, failures, planted_checkout
from test_numbers import committed

from mime.docs._evidence import DEFAULT_LANE, junit_outcomes, read_claims
from mime.docs._evidence_page import (
    BODY_LINE,
    DIGEST_LABEL,
    DEVICE_NOTE,
    FIXED_SENTENCE,
    FLOOR,
    FLOOR_REMOVED,
    LANE_DEVICE,
    ROLE_GROUPS,
    SOURCE,
    STATUSES,
    committed_copy_problems,
    device_of,
    floor_form,
    evidence_pages,
    generation_problems,
    floor_base_problems,
    page_problems,
    status_of,
    title_of,
    Inputs,
    lane_reports,
)
from mime.docs._lists import read_list
from mime.docs._specification import SPECIFICATIONS, statements

#: The decision's sentence, typed again here so that a change to the constant fails the test that reads the page.
SENTENCE = "This document is not a regulatory record or certificate; bounds and estimates are as labelled."
TESTS = "tests/test_planted.py"
PLANTED_TESTS = '''\
import pytest


def test_runs():
    assert True


def test_fails():
    assert False


def test_skipped_in_body():
    pytest.skip("planted")


@pytest.mark.slow
def test_slow():
    assert True


@pytest.mark.xfail(reason="planted")
def test_xfail():
    assert False


@pytest.mark.skip(reason="planted")
def test_marked_skip():
    assert True


@pytest.mark.gpu
def test_gpu():
    assert True


def test_to_delete():
    assert True
'''
FRAMING = "\nThe framing, which says what bound and estimate mean.\n"


def claim(name: str, node: str | None, **fields: object) -> dict[str, object]:
    """A claim record resting on the planted test ``node``: a computed estimate unless the fields say otherwise."""
    held: dict[str, object] = {"id": name, "quantity": f"the quantity of {name}", "value": 0.5, "unit": "1",
                               "basis": "computed", "role": "estimate", "oracle_kind": "analytic",
                               "evidence": [f"{TESTS}::{node}"] if node else [], "n": 3, "says": f"what {name} says"}
    held.update(fields)
    if held["basis"] == "declared":
        held.pop("oracle_kind"), held.pop("n"), held.pop("value", None)
    return held


def registry(claims: list[dict[str, object]]) -> str:
    anomaly = {"anomaly_id": "MIME-ANO-900", "title": "planted", "description": "planted", "severity": "minor",
               "safety_relevance": "none", "safety_relevance_rationale": "planted", "resolution_status": "open",
               "affected_versions": ">=0.3", "claims": claims}
    return yaml.safe_dump({"schema_version": "1", "generated_date": "2026-11-03", "anomalies": [anomaly]},
                          sort_keys=False)


def specification(rows: dict[str, list[str]]) -> str:
    """A specification of the statement ids in ``rows``, each with its cases (none: not yet covered)."""
    lines = ["# The planted specification", ""]
    for key, nodes in rows.items():
        cases = ", ".join(f"`{TESTS}::{n}`" for n in nodes) + "." if nodes else "not yet covered."
        lines += [f"- **[{key}]** The planted statement {key}.", f"  *Cases:* {cases}"]
    return "\n".join(lines) + "\n"


def floor(*items: str) -> str:
    return "".join(f"{item}  # pinned for the test\n" for item in items)


def planted(into: Path, claims: list[dict[str, object]] | None = None,
            rows: dict[str, list[str]] | None = None, pinned: tuple[str, ...] | None = None,
            source: str | None = None) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    claims = [claim("runs", "test_runs")] if claims is None else claims
    rows = {"doc.runs": ["test_runs"]} if rows is None else rows
    items = pinned if pinned is not None else (
        *(f"claim MIME-ANO-900.{c['id']}" for c in claims), *(f"statement {key}" for key in rows))
    return planted_checkout(into, {
        TESTS: PLANTED_TESTS, "docs/validation/known_anomalies.yaml": registry(claims),
        str(SPECIFICATIONS[0]): specification(rows), str(SOURCE): SENTENCE + "\n" + FRAMING if source is None else source,
        "docs/ratchets/evidence_floor.txt": floor(*items)})


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The default lane's JUnit report of a real run of the planted tests, in a directory of its own."""
    root = planted_checkout(tmp_path_factory.mktemp("lane"), {TESTS: PLANTED_TESTS})
    out = tmp_path_factory.mktemp("junit")
    done = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={out / 'a.xml'}",
                           TESTS], cwd=root, capture_output=True, text=True, check=False)
    assert (out / "a.xml").is_file(), done.stdout + done.stderr
    return out


@pytest.fixture(scope="module")
def partial_reports(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A report of the planted tests with ``test_runs`` and ``test_to_delete`` missing from it, as a lane's report is
    that lost the shard holding them."""
    root = planted_checkout(tmp_path_factory.mktemp("lane"), {TESTS: PLANTED_TESTS})
    out = tmp_path_factory.mktemp("junit")
    done = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={out / 'a.xml'}",
                           "--deselect", f"{TESTS}::test_runs", "--deselect", f"{TESTS}::test_to_delete", TESTS],
                          cwd=root, capture_output=True, text=True, check=False)
    assert (out / "a.xml").is_file(), done.stdout + done.stderr
    return out


def page_of(root: Path, reports: Path | None = None):
    inputs = Inputs(True, None, lane_reports(reports) if reports is not None else None)
    return evidence_pages(root, inputs)


def text_of(root: Path, reports: Path | None = None) -> str:
    built = page_of(root, reports)
    assert built.problems == [] and built.unresolved == [], built
    (page,) = built.pages
    return page.text


# -- the real page ----------------------------------------------------------------------------------------------------

def test_the_real_page_opens_with_the_fixed_sentence_and_says_counts():
    built = evidence_pages(ROOT, Inputs())
    assert built.problems == [] and built.unresolved == []
    (page,) = built.pages
    lines = page.text.splitlines()
    assert lines[:3] == ["# Evidence counts", "", SENTENCE]
    assert BODY_LINE in lines and "a test's body is not checked" in BODY_LINE.lower()


def test_the_real_page_holds_every_claim_and_every_statement_once_and_the_floor_is_their_count():
    built = evidence_pages(ROOT, Inputs())
    (page,) = built.pages
    claims = read_claims(ROOT / "docs/validation/known_anomalies.yaml")
    statement = {s.key for path in SPECIFICATIONS for s in statements((ROOT / path).read_text(encoding="utf-8"), path)[0]}
    assert page_problems(page.text, list(claims.values()), {SPECIFICATIONS[0]: statements(
        (ROOT / SPECIFICATIONS[0]).read_text(encoding="utf-8"), SPECIFICATIONS[0])[0]}) == []
    pinned = (ROOT / "docs/ratchets/evidence_floor.txt").read_text(encoding="utf-8")
    assert {line.split()[1] for line in pinned.splitlines() if line.startswith("statement ")} <= statement
    assert all(f"claim {key}" in pinned for key in claims) and len(claims) >= 1 and len(statement) >= 38


# -- the outcomes, from a real run's report ---------------------------------------------------------------------------

def test_the_passing_arm_a_claim_on_a_test_that_ran_and_passed_counts_and_prints_its_device(tmp_path, reports):
    page = text_of(planted(tmp_path), reports)
    assert ("Claims 1; with a test that ran and passed in the default lane 1; not run 0; failed 0; skipped 0; "
            "covered by no test 0; outcome not read 0.") in page
    assert ("Statements 1; with a test that ran and passed in the default lane 1; not run 0; failed 0; skipped 0; "
            "covered by no test 0; outcome not read 0.") in page
    row = next(line for line in page.splitlines() if line.startswith("| `MIME-ANO-900.runs`"))
    assert f"| passed | {DEVICE_NOTE} |" in row
    assert f"passed in {DEFAULT_LANE}" in row


@pytest.mark.parametrize(("node", "status"), [("test_skipped_in_body", "skipped"), ("test_xfail", "not run"),
                                              ("test_slow", "not run"), ("test_marked_skip", "not run"),
                                              ("test_gpu", "not run"), ("test_fails", "failed")])
def test_a_claim_resting_only_on_a_test_that_did_not_pass_is_not_shown_as_passed_nor_counted(tmp_path, reports, node,
                                                                                               status):
    page = text_of(planted(tmp_path, [claim("one", node)], {"doc.one": [node]}), reports)
    assert "with a test that ran and passed in the default lane 0;" in page
    rows = [line for line in page.splitlines() if line.startswith(("| `MIME-ANO-900.one`", "| [`doc.one`]"))]
    assert len(rows) == 2 and all(f"| {status} | " in row and "| passed |" not in row for row in rows), rows


def test_a_row_with_one_test_that_passed_and_one_that_did_not_takes_the_worse(tmp_path, reports):
    root = planted(tmp_path, [claim("two", "test_runs", evidence=[f"{TESTS}::test_runs", f"{TESTS}::test_slow"])])
    page = text_of(root, reports)
    assert "Claims 1; with a test that ran and passed in the default lane 0; not run 1;" in page


def test_without_a_lane_report_every_row_reads_outcome_not_read_and_none_counts(tmp_path):
    page = text_of(planted(tmp_path))
    assert "Claims 1; with a test that ran and passed in the default lane 0;" in page
    assert page.count("outcome not read (no lane report)") >= 2
    assert "Outcome not read (no lane report): this build was given none" in page


def test_an_empty_directory_of_reports_is_a_problem_not_a_page_that_says_not_read(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    built = page_of(planted(tmp_path / "root"), empty)
    assert built.pages == [] and any("no JUnit report was found" in problem for problem in built.problems)


def test_through_the_command_a_directory_with_no_report_fails_the_build_and_writes_nothing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    done = docs_run("build", "--root", str(ROOT), "--out", str(tmp_path / "out"), "--junit", str(empty))
    assert done.returncode == 1 and not (tmp_path / "out").exists(), done.stdout
    assert "no JUnit report was found" in done.stdout


def test_evidence_not_resolved_reads_not_read_and_collects_nothing(tmp_path):
    built = evidence_pages(planted(tmp_path), Inputs())
    (page,) = built.pages
    assert "Evidence not resolved (`--skip-evidence`)" in page.text and "outcome not read 1." in page.text


def test_a_declared_claim_and_a_statement_not_yet_covered_count_as_covered_by_no_test(tmp_path, reports):
    root = planted(tmp_path, [claim("edge", None, basis="declared", role="limit")], {"doc.open": []})
    page = text_of(root, reports)
    assert "Claims 1; with a test that ran and passed in the default lane 0; not run 0; failed 0; skipped 0; " \
           "covered by no test 1;" in page
    assert "Statements 1; with a test that ran and passed in the default lane 0; not run 0; failed 0; skipped 0; " \
           "covered by no test 1;" in page
    assert "| not yet covered |" in page


def test_status_and_device_are_the_lanes_and_a_row_that_did_not_run_has_none():
    assert status_of((), {}, resolved=True) == "no test"
    assert status_of(("x",), {}, resolved=False) == "not read"
    ran = DEVICE_NOTE
    assert {status: device_of(status) for status in STATUSES} == {
        "passed": ran, "failed": ran, "skipped": ran, "not run": "none", "no test": "none", "not read": "not read"}


# -- a claim whose cited test is deleted -------------------------------------------------------------------------------

def test_a_claim_whose_cited_test_is_deleted_fails_its_row(tmp_path, reports):
    root = planted(tmp_path, [claim("gone", "test_to_delete")], {"doc.gone": ["test_runs"]})
    assert page_of(root, reports).problems == []                       # the passing arm: the test is there
    edit(root / TESTS, "def test_to_delete():\n    assert True\n", "")
    built = page_of(root, reports)
    assert built.pages == []
    assert built.unresolved == [f"evidence counts: {TESTS}::test_to_delete: is not collected: pytest finds no test of "
                                f"that id"]


def test_a_statement_whose_cited_test_is_deleted_fails_its_row_too(tmp_path, reports):
    root = planted(tmp_path, [claim("runs", "test_runs")], {"doc.gone": ["test_to_delete"]})
    edit(root / TESTS, "def test_to_delete():\n    assert True\n", "")
    built = page_of(root, reports)
    assert built.pages == [] and built.unresolved == [
        f"evidence counts: {TESTS}::test_to_delete: is not collected: pytest finds no test of that id"]


# -- the fixed sentence -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("source", [
    FRAMING,                                                                   # removed
    SENTENCE.replace("regulatory", "legal") + "\n" + FRAMING,                  # one word
    SENTENCE.replace("labelled", "labeled") + "\n" + FRAMING,                  # one letter
    "A paragraph first.\n\n" + SENTENCE + "\n" + FRAMING,                      # not first
    SENTENCE.removesuffix(".") + "\n" + FRAMING,                               # the full stop
], ids=["removed", "one word", "one letter", "not first", "no full stop"])
def test_a_source_without_the_fixed_sentence_exactly_first_fails_the_build(tmp_path, source):
    built = evidence_pages(planted(tmp_path, source=source), Inputs())
    assert built.pages == []
    (problem,) = built.problems
    assert problem.startswith(f"{SOURCE}: the page opens with the fixed sentence, exactly: ")


def test_the_fixed_sentence_is_the_decisions(tmp_path):
    assert FIXED_SENTENCE == SENTENCE
    assert text_of(planted(tmp_path)).splitlines()[:3] == ["# Evidence counts", "", SENTENCE]


def test_a_page_without_the_fixed_sentence_fails_the_page_check(tmp_path):
    root = planted(tmp_path)
    page = text_of(root)
    claims = list(read_claims(root / "docs/validation/known_anomalies.yaml").values())
    specs = {SPECIFICATIONS[0]: statements((root / SPECIFICATIONS[0]).read_text(encoding="utf-8"),
                                           SPECIFICATIONS[0])[0]}
    assert page_problems(page, claims, specs) == []
    altered = page.replace("regulatory", "legal", 1)
    assert any("the fixed sentence is not the page's first line" in p for p in page_problems(altered, claims, specs))
    removed = page.replace(SENTENCE + "\n", "", 1)
    assert any("the fixed sentence is not the page's first line" in p for p in page_problems(removed, claims, specs))


def test_through_the_command_a_copy_of_the_checkout_with_the_sentence_altered_fails_the_build(tmp_path):
    root = copy_checkout(tmp_path)
    edit(root / SOURCE, "regulatory", "legal")
    built = docs_run("build", "--root", str(root), "--out", str(tmp_path / "out"), "--skip-evidence")
    assert built.returncode == 1 and not (tmp_path / "out").exists()
    assert f"{SOURCE}: the page opens with the fixed sentence, exactly" in built.stdout


def test_through_the_command_a_copy_of_the_checkout_with_the_sentence_altered_fails_the_check(tmp_path):
    root = copy_checkout(tmp_path)
    assert evidence_pages(root, Inputs()).problems == []                       # the passing arm
    edit(root / SOURCE, "regulatory", "legal")
    assert any(p.startswith(f"{SOURCE}: the page opens with the fixed sentence") for p in failures(docs_check(root)))


# -- the groups: an acceptance record never prints under an accuracy heading ---------------------------------------

def measured_acceptance() -> dict[str, object]:
    return claim("agrees", "test_runs", basis="measured", role="acceptance", oracle_kind="measurement")


def test_an_acceptance_claim_prints_under_acceptance_tests_and_the_page_is_called_a_report(tmp_path, reports):
    root = planted(tmp_path, [claim("runs", "test_runs"), measured_acceptance()])
    page = text_of(root, reports)
    assert page.splitlines()[0] == "# Evidence report"
    assert "It holds no claim with a measured basis" not in page
    section = {}
    heading = None
    for line in page.splitlines():
        if line.startswith("### "):
            heading = line[4:]
        elif line.startswith("| `MIME-ANO-900."):
            section[line.split("`")[1]] = heading
    assert section == {"MIME-ANO-900.runs": "Estimates", "MIME-ANO-900.agrees": "Acceptance tests"}


def test_without_a_measured_claim_the_page_says_counts_and_that_it_is_not_a_report(tmp_path):
    page = text_of(planted(tmp_path))
    assert page.splitlines()[0] == "# Evidence counts" and "Report" not in page.splitlines()[0]
    assert "It holds no claim with a measured basis, so it is a count and not a report." in page
    assert title_of(list(read_claims(tmp_path / "docs/validation/known_anomalies.yaml").values())) == "Evidence counts"


def test_an_acceptance_claim_placed_under_an_accuracy_heading_fails_the_page_check(tmp_path, reports):
    root = planted(tmp_path, [claim("runs", "test_runs"), measured_acceptance()])
    page = text_of(root, reports)
    claims = list(read_claims(root / "docs/validation/known_anomalies.yaml").values())
    specs = {SPECIFICATIONS[0]: statements((root / SPECIFICATIONS[0]).read_text(encoding="utf-8"),
                                           SPECIFICATIONS[0])[0]}
    assert page_problems(page, claims, specs) == []                      # the passing arm
    rows = [line for line in page.splitlines() if line.startswith("| `MIME-ANO-900.agrees`")]
    (row,) = rows
    moved = page.replace(row + "\n", "", 1).replace("### Estimates\n", "### Estimates\n\n" + row + "\n", 1)
    problems = [p for p in page_problems(moved, claims, specs) if "agrees" in p]
    assert problems == ["the claim MIME-ANO-900.agrees (acceptance) is under 'Estimates', not once under "
                        "'Acceptance tests'"]


def test_every_role_has_its_own_heading_and_an_empty_group_says_none(tmp_path):
    page = text_of(planted(tmp_path))
    for role, (heading, _) in ROLE_GROUPS.items():
        assert f"### {heading}\n" in page
    assert page.count("\nNone.\n") == len(ROLE_GROUPS) - 1


def test_a_rule_prints_its_cases_and_no_basis(tmp_path, reports):
    page = text_of(planted(tmp_path), reports)
    statements_part = page.split("## Specification statements")[1]
    header = next(line for line in statements_part.splitlines() if line.startswith("| Statement"))
    assert "Basis" not in header and "Its cases" in header
    row = next(line for line in statements_part.splitlines() if "doc.runs" in line)
    assert f"{TESTS}::test_runs" in row and "computed" not in row and "measured" not in row


# -- the floor on N ---------------------------------------------------------------------------------------------------

def test_the_passing_arm_every_pinned_claim_and_statement_present_builds(tmp_path):
    assert evidence_pages(planted(tmp_path), Inputs()).problems == []


def test_the_count_floor_undercut_a_pinned_claim_removed_fails(tmp_path):
    root = planted(tmp_path, [claim("runs", "test_runs"), claim("other", "test_runs")],
                   pinned=("claim MIME-ANO-900.runs", "claim MIME-ANO-900.other", "statement doc.runs"))
    assert evidence_pages(root, Inputs()).problems == []
    (root / "docs/validation/known_anomalies.yaml").write_text(registry([claim("runs", "test_runs")]), encoding="utf-8")
    built = evidence_pages(root, Inputs())
    assert built.pages == []
    (problem,) = built.problems
    assert problem.endswith("the pinned claim or statement 'claim MIME-ANO-900.other' is missing; restore it, or "
                            "delete the line with its reason logged")


def test_the_count_floor_undercut_a_pinned_statement_removed_fails(tmp_path):
    root = planted(tmp_path, rows={"doc.runs": ["test_runs"], "doc.more": ["test_runs"]},
                   pinned=("claim MIME-ANO-900.runs", "statement doc.runs", "statement doc.more"))
    assert evidence_pages(root, Inputs()).problems == []
    (root / SPECIFICATIONS[0]).write_text(specification({"doc.runs": ["test_runs"]}), encoding="utf-8")
    (problem,) = evidence_pages(root, Inputs()).problems
    assert "'statement doc.more' is missing" in problem


def test_an_item_present_and_not_pinned_is_no_failure_and_a_malformed_floor_line_fails(tmp_path):
    root = planted(tmp_path, pinned=("statement doc.runs",))
    assert evidence_pages(root, Inputs()).problems == []
    (root / "docs/ratchets/evidence_floor.txt").write_text(floor("statement doc.runs", "a claim"), encoding="utf-8")
    problems = evidence_pages(root, Inputs()).problems
    assert any("'a claim' is malformed: an item is 'claim <anomaly id>.<claim id>' or 'statement <statement id>'" in p
               for p in problems)


def test_the_real_floor_through_the_command_a_pinned_statement_removed_from_the_specification_fails(tmp_path):
    root = copy_checkout(tmp_path)
    assert evidence_pages(root, Inputs()).problems == []                       # the passing arm
    spec = root / SPECIFICATIONS[0]
    text = spec.read_text(encoding="utf-8")
    start = text.index("- **[folder.missing]**")
    end = text.index("- **[", start + 1)
    spec.write_text(text[:start] + text[end:], encoding="utf-8")
    done = docs_run("build", "--root", str(root), "--out", str(tmp_path / "out"), "--skip-evidence")
    assert done.returncode == 1 and "'statement folder.missing' is missing" in done.stdout


# -- generation: the page edited by hand --------------------------------------------------------------------------

def test_the_page_edited_by_hand_fails_the_generation_check_and_the_unedited_one_passes(tmp_path, reports):
    page = text_of(planted(tmp_path), reports)
    assert generation_problems(page) == []
    assert page.splitlines()[-1].startswith(DIGEST_LABEL)
    for edited in (page.replace("ran and passed in the default lane 1", "ran and passed in the default lane 2", 1),
                   page.replace("| passed |", "| failed |", 1), page.replace("Claims 1", "Claims 9", 1),
                   page.rstrip("\n").rsplit("\n", 2)[0] + "\n"):
        (problem,) = generation_problems(edited)
        assert "edited by hand" in problem or "no generation digest" in problem
    assert generation_problems(page.replace("a", "b", 1))


def test_a_copy_of_the_generated_page_under_docs_fails_whether_or_not_it_was_edited(tmp_path, reports):
    page = text_of(planted(tmp_path / "root"), reports)
    docs = tmp_path / "docs"
    docs.mkdir()
    assert committed_copy_problems(docs) == []                                 # the passing arm
    (docs / "copy.md").write_text(page, encoding="utf-8")
    (copy_problem,) = committed_copy_problems(docs)
    assert copy_problem.endswith("a copy of a generated page; pages are built and never committed, delete it")
    (docs / "copy.md").write_text(page.replace("Claims 1", "Claims 9"), encoding="utf-8")
    (edited,) = committed_copy_problems(docs)
    assert edited.endswith("delete it (it is also edited by hand)")


def test_through_the_command_a_copy_of_the_page_in_the_checkout_fails_the_check(tmp_path):
    root = copy_checkout(tmp_path)
    built = evidence_pages(root, Inputs())
    (root / "docs" / "evidence_copy.md").write_text(built.pages[0].text, encoding="utf-8")
    problems = failures(docs_check(root))
    assert any("evidence_copy.md: a copy of a generated page" in problem for problem in problems)


def test_the_page_is_the_same_text_whatever_the_hash_seed(tmp_path, reports):
    root = planted(tmp_path, [claim("a", "test_runs"), claim("b", "test_fails"), claim("c", "test_slow")],
                   {"doc.a": ["test_runs"], "doc.b": ["test_fails"], "doc.c": ["test_skipped_in_body"]})
    program = ("import sys\nfrom pathlib import Path\nfrom mime.docs._evidence_page import Inputs, evidence_pages, "
               "lane_reports\nbuilt = evidence_pages(Path(sys.argv[1]), Inputs(True, None, lane_reports(Path(sys.argv[2]))))\n"
               "print(built.pages[0].text if built.pages else built.problems + built.unresolved)")
    texts = {subprocess.run([sys.executable, "-c", program, str(root), str(reports)], capture_output=True, text=True,
                            check=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu"}
                            ).stdout for seed in ("1", "2", "3")}
    assert len(texts) == 1 and "Claims 3;" in next(iter(texts))


# -- a bound's held-out test is read like the rest of its evidence -------------------------------------------------

def bound(name: str, held_out: str) -> dict[str, object]:
    """A bound resting on ``test_runs`` and held out on ``held_out``."""
    return claim(name, "test_runs", role="bound", domain="the unit cube", held_out=f"{TESTS}::{held_out}")


def row_of(page: str, key: str) -> str:
    return next(line for line in page.splitlines() if line.startswith(f"| `{key}`"))


def test_a_bound_whose_held_out_test_passed_reads_passed_and_prints_the_held_out_test(tmp_path, reports):
    page = text_of(planted(tmp_path, [bound("held", "test_to_delete")]), reports)
    row = row_of(page, "MIME-ANO-900.held")
    assert "| bound |" in row and "| passed |" in row
    assert f"held out: {TESTS}::test_to_delete: {DEFAULT_LANE}; passed in {DEFAULT_LANE}" in row
    assert "Claims 1; with a test that ran and passed in the default lane 1;" in page


@pytest.mark.parametrize(("held_out", "status"), [("test_fails", "failed"), ("test_skipped_in_body", "skipped"),
                                                  ("test_slow", "not run"), ("test_xfail", "not run"),
                                                  ("test_marked_skip", "not run"), ("test_gpu", "not run")])
def test_a_bound_whose_held_out_test_did_not_pass_takes_that_status_though_its_fit_test_passed(tmp_path, reports,
                                                                                             held_out, status):
    page = text_of(planted(tmp_path, [bound("held", held_out)]), reports)
    row = row_of(page, "MIME-ANO-900.held")
    assert f"| {status} | " in row and "| passed |" not in row, row
    assert "Claims 1; with a test that ran and passed in the default lane 0;" in page


def test_a_bound_whose_held_out_test_is_in_no_report_reads_not_read_and_one_not_collected_fails_the_build(
        tmp_path, reports, partial_reports):
    root = planted(tmp_path, [claim("fit", "test_fails"), bound("held", "test_to_delete")])
    row = row_of(text_of(root, partial_reports), "MIME-ANO-900.held")
    assert "| not read |" in row and "| passed |" not in row, row           # the fit test is in the report
    assert "| passed |" in row_of(text_of(root, reports), "MIME-ANO-900.held")   # the passing arm
    edit(root / TESTS, "def test_to_delete():\n    assert True\n", "")
    assert page_of(root, reports).unresolved == [
        f"evidence counts: {TESTS}::test_to_delete: is not collected: pytest finds no test of that id"]


# -- the floor may only grow against the base ---------------------------------------------------------------------

FLOORED = "statement folder.layout"
FLOOR_PROBLEM = (f"{FLOOR}: the pinned '{FLOORED}' is on the floor at HEAD and not here; the floor only grows: "
                 f"restore the line, or log the removal as '{FLOORED}  # <reason>' in {FLOOR_REMOVED}")


def without_line(root: Path, item: str) -> None:
    path = root / FLOOR
    kept = [line for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
            if not line.startswith(f"{item}  #")]
    assert len(kept) == len(path.read_text(encoding="utf-8").splitlines(keepends=True)) - 1
    path.write_text("".join(kept), encoding="utf-8")


def test_a_pinned_line_deleted_from_the_floor_fails_against_the_base_and_not_without_one(tmp_path):
    root = committed(copy_checkout(tmp_path))
    assert docs_check(root, base="HEAD").returncode == 0                     # the passing arm
    without_line(root, FLOORED)
    assert FLOOR_PROBLEM in failures(docs_check(root, base="HEAD"))
    unbased = docs_check(root)
    assert unbased.returncode == 0 and "evidence floor: not compared with a base revision" in unbased.stdout


def test_a_statement_removed_with_its_floor_line_fails_against_the_base(tmp_path):
    """The review's attack: the specification's statement and its floor line deleted in one change, which the
    floor's own check (a pinned item missing) cannot see."""
    root = committed(copy_checkout(tmp_path))
    spec = root / SPECIFICATIONS[0]
    text = spec.read_text(encoding="utf-8")
    start = text.index("- **[folder.missing]**")
    spec.write_text(text[:start] + text[text.index("- **[", start + 1):], encoding="utf-8")
    without_line(root, "statement folder.missing")
    problems = failures(docs_check(root, base="HEAD"))
    assert any("the pinned 'statement folder.missing' is on the floor at HEAD and not here" in p for p in problems)
    assert not any("is missing; restore it" in p for p in problems), problems     # the floor's own check is satisfied
    assert docs_check(root).returncode == 0                      # with no base nothing fails: why it passed before


def test_a_removal_with_a_reason_logged_is_excused_and_the_summary_counts_it(tmp_path):
    root = committed(copy_checkout(tmp_path))
    pinned = len(read_list(root / FLOOR).lines)
    without_line(root, FLOORED)
    removed = root / FLOOR_REMOVED
    removed.write_text(removed.read_text(encoding="utf-8") + f"{FLOORED}  # a planted removal, logged\n",
                       encoding="utf-8")
    done = docs_check(root, base="HEAD")
    assert done.returncode == 0, done.stdout
    assert (f"evidence floor: compared with HEAD's, {pinned} lines there and {pinned - 1} here, 1 removed, 1 logged"
            in done.stdout)


def test_a_removal_logged_without_a_reason_or_still_on_the_floor_fails(tmp_path):
    root = committed(copy_checkout(tmp_path))
    removed = root / FLOOR_REMOVED
    removed.write_text(removed.read_text(encoding="utf-8") + f"{FLOORED}  # still pinned\n", encoding="utf-8")
    assert any("is logged as removed from the floor and is still on it" in p for p in failures(docs_check(root)))
    without_line(root, FLOORED)
    removed.write_text(removed.read_text(encoding="utf-8").replace("  # still pinned", ""), encoding="utf-8")
    assert any(f"{FLOORED!r} has no reason" in p for p in failures(docs_check(root, base="HEAD")))


def test_the_floor_against_a_base_that_cannot_be_read_or_has_no_floor_says_so(tmp_path):
    root = copy_checkout(tmp_path)
    floor, removed = (read_list(root / name, floor_form) for name in (FLOOR, FLOOR_REMOVED))
    assert floor_base_problems(root, None, floor, removed)[1].startswith("evidence floor: not compared")
    held = committed(root)
    (problem,), summary = floor_base_problems(held, "no-such-revision", floor, removed)
    assert "cannot be read" in problem and "which cannot be read" in summary
    subprocess.run(["git", "-C", str(held), "rm", "-q", "--cached", str(FLOOR)], check=True)
    subprocess.run(["git", "-C", str(held), "-c", "user.name=A", "-c", "user.email=a@example.org", "commit", "-q",
                    "-m", "no floor"], check=True)
    assert floor_base_problems(held, "HEAD", floor, removed) == ([], "evidence floor: HEAD holds no such list, so it "
                                                                 "is compared with nothing: this change adds it")


# -- the page's source deleted -------------------------------------------------------------------------------------

def test_a_missing_source_fails_the_build_and_the_check_while_a_checkout_with_nothing_to_count_builds_none(tmp_path):
    root = copy_checkout(tmp_path)
    (root / SOURCE).unlink()
    built = evidence_pages(root, Inputs())
    assert built.pages == [] and len(built.problems) == 1 and "the evidence counts page's source is missing" in built.problems[0]
    done = docs_run("build", "--root", str(root), "--out", str(tmp_path / "out"), "--skip-evidence")
    assert done.returncode == 1 and not (tmp_path / "out").exists()
    assert "the evidence counts page's source is missing" in done.stdout
    assert any("the evidence counts page's source is missing" in p for p in failures(docs_check(root)))
    nothing = tmp_path / "nothing"
    nothing.mkdir()
    assert evidence_pages(nothing, Inputs()).problems == [] and evidence_pages(nothing, Inputs()).pages == []


# -- a lane report that does not hold a test the lane runs ---------------------------------------------------------

def test_a_test_the_lane_runs_and_no_report_holds_reads_not_read_and_never_not_run(tmp_path, reports, partial_reports):
    root = planted(tmp_path)
    page = text_of(root, partial_reports)
    assert "Claims 1; with a test that ran and passed in the default lane 0; not run 0; failed 0; skipped 0; " \
           "covered by no test 0; outcome not read 1." in page
    assert "| not read | not read |" in row_of(page, "MIME-ANO-900.runs")
    assert "outcome not read 0." in text_of(root, reports)                              # the passing arm
    assert "not run 1" in text_of(planted(tmp_path / "slow", [claim("slow", "test_slow")], {"doc.s": ["test_slow"]}),
                                  partial_reports)                                   # what CI does not run stays so


# -- a test id in two reports ----------------------------------------------------------------------------------------

def report_xml(path: Path, *cases: str) -> Path:
    path.write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest">' + "".join(cases)
                    + "</testsuite></testsuites>", encoding="utf-8")
    return path


PASSED_CASE = '<testcase classname="tests.test_x" name="test_a" time="0.1"/>'
FAILED_CASE = '<testcase classname="tests.test_x" name="test_a" time="0.1"><failure message="m"/></testcase>'
ERROR_CASE = '<testcase classname="tests.test_x" name="test_a" time="0.1"><error message="m"/></testcase>'
SKIPPED_CASE = '<testcase classname="tests.test_x" name="test_a"><skipped type="pytest.skip" message="m"/></testcase>'


@pytest.mark.parametrize(("cases", "outcome"), [
    ((PASSED_CASE, FAILED_CASE), "failed"), ((FAILED_CASE, PASSED_CASE), "failed"),
    ((PASSED_CASE, ERROR_CASE), "error"), ((ERROR_CASE, PASSED_CASE), "error"),
    ((PASSED_CASE, SKIPPED_CASE), "skipped"), ((SKIPPED_CASE, PASSED_CASE), "skipped"),
    ((PASSED_CASE, PASSED_CASE), "passed")])
def test_a_test_id_twice_takes_the_worse_whichever_report_sorts_last(tmp_path, cases, outcome):
    first, second = (report_xml(tmp_path / name, case) for name, case in zip(("a.xml", "b.xml"), cases, strict=True))
    same = report_xml(tmp_path / "c.xml", *cases)
    assert junit_outcomes([first, second]) == {"tests/test_x.py::test_a": outcome}
    assert junit_outcomes([second, first]) == {"tests/test_x.py::test_a": outcome}
    assert junit_outcomes([same]) == {"tests/test_x.py::test_a": outcome}


# -- the device is declared -----------------------------------------------------------------------------------------

def test_the_device_is_said_to_be_declared_by_the_workflow_and_not_observed(tmp_path, reports):
    page = text_of(planted(tmp_path), reports)
    assert "(declared by the workflow)" in DEVICE_NOTE and "the default lane)" not in DEVICE_NOTE
    assert "a report records none, so the Device column is that one declared value for every row that ran" in page
    assert "the lane runs on" not in page
    assert f"in {DEFAULT_LANE}" in row_of(page, "MIME-ANO-900.runs") and f"| {DEVICE_NOTE} |" in row_of(
        page, "MIME-ANO-900.runs")
    assert "tests in 1 report)" in page
