"""The evidence resolver and the claims' shape (``mime.docs._evidence``): controls E1 to E5, every refused cell of the
basis, role and oracle-kind matrix, and what counts as evidence.

E1 and E2 plant tests in a checkout of their own, a temporary directory holding MIME's own pytest configuration
(copied from ``pyproject.toml``), and resolve ids against pytest's real collection of it, and against the JUnit report
of a real run of it as CI's default lane runs (``-m "not slow"``) where the outcome is the point. Beside them, one real
test of this checkout must resolve in the default lane. Evidence is a test that ran and passed in a named lane: a test
marked ``skip``, ``xfail``, ``slow`` or ``gpu`` prints "not run in CI" and does not count at collection, and in a lane's
report only a pass counts. Each case is deterministic: it fails, or prints its lane, on every run.
"""

import itertools
import subprocess
import sys
from pathlib import Path

import pytest
from checkout import PLANTED_LANES, planted_checkout

from mime.docs._evidence import (
    BASES,
    DEFAULT_LANE,
    LANES,
    ORACLE_KINDS,
    ROLES,
    Collection,
    claim,
    claim_problems,
    claim_rows,
    collect,
    junit_outcomes,
    read_claims,
    resolve,
)

ROOT = Path(__file__).resolve().parents[2]
REAL = "tests/test_anomaly_registry.py::test_the_real_registry_passes"
OTHER_REAL = "tests/test_anomaly_registry.py::test_the_script_runs_as_ci_runs_it"
PLANTED = """\
import pytest


def test_renamed_new():  # was test_renamed_old
    assert 1 + 1 == 2


def test_documented():
    \"\"\"test_docstring_only: this name is written only here.\"\"\"
    assert True


def test_skipped_in_body():
    pytest.skip("planted")


def test_helper(): ...


def test_pass_body():
    pass


def test_wrong_threshold():
    error = 0.2
    assert error < 0.30  # its claim states 0.05


@pytest.mark.skipif(False, reason="planted, false")
def test_skipif_false():
    assert True


@pytest.mark.skipif(True, reason="planted, true")
def test_skipif_true():
    assert True


@pytest.mark.parametrize("case", [1, pytest.param(2, marks=pytest.mark.skip(reason="planted"))])
def test_one_case_skipped(case):
    assert case


""" + "\n\n".join(PLANTED_LANES.values())
FILE = "tests/test_planted.py"


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> Path:
    return planted_checkout(tmp_path_factory.mktemp("planted"), {FILE: PLANTED})


@pytest.fixture(scope="module")
def junit(planted) -> Path:
    """A JUnit report of a real run of the planted file as CI's default lane runs it (``-m "not slow"``)."""
    report = planted / "junit.xml"
    done = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={report}", FILE],
                          cwd=planted, capture_output=True, text=True, check=False)
    assert report.is_file(), done.stdout + done.stderr
    return report


def node(name: str) -> str:
    return f"{FILE}::{name}"


def test_e1_the_real_test_resolves_in_the_default_lane_and_counts():
    resolution = resolve([REAL], ROOT)[REAL]
    assert resolution.problem is None and resolution.lane == DEFAULT_LANE and resolution.outcomes is None
    assert resolution.nodeids == (REAL,) and resolution.runs_in_ci and resolution.counts


@pytest.mark.parametrize("name", ["test_renamed_old", "test_docstring_only"])
def test_e1_a_renamed_test_and_a_name_only_in_a_docstring_fail_as_not_collected(planted, name):
    resolution = resolve([node(name)], planted)[node(name)]
    assert resolution.problem is not None and "is not collected" in resolution.problem and not resolution.counts


@pytest.mark.parametrize("name", ["test_slow", "test_xfail", "test_marked_skip", "test_gpu"])
def test_control_a_test_ci_does_not_run_prints_not_run_in_ci_and_does_not_count(planted, name):
    """A slow, an xfail, a skip-marked and a gpu test each resolve (they are collected) and print why CI does not run
    them, never "default lane"; none counts as evidence without a lane's report."""
    resolution = resolve([node(name)], planted)[node(name)]
    assert resolution.problem is None and not resolution.runs_in_ci and not resolution.counts
    assert resolution.lane == LANES[name.removeprefix("test_").removeprefix("marked_")]
    assert resolution.text().startswith(f"{node(name)}: not run in CI (")


def test_the_baseline_a_test_ci_runs_counts(planted):
    resolution = resolve([node("test_runs")], planted)[node("test_runs")]
    assert resolution.lane == DEFAULT_LANE and resolution.counts


def test_a_skipif_whose_condition_is_false_runs_and_one_whose_condition_holds_here_says_so(planted):
    """CI's lanes allow no skipped test, so a skipif's condition is false on CI's runner: both count at collection, and
    the one whose condition holds on this machine says so; the lane's report then shows it skipped, which does not
    count."""
    resolved = resolve([node("test_skipif_false"), node("test_skipif_true")], planted)
    assert resolved[node("test_skipif_false")].lane == DEFAULT_LANE
    assert resolved[node("test_skipif_true")].lane == f"{DEFAULT_LANE}, skipped on this machine (planted, true)"
    assert all(r.counts for r in resolved.values())


def test_an_id_naming_a_skipped_parametrisation_does_not_count(planted):
    resolution = resolve([node("test_one_case_skipped")], planted)[node("test_one_case_skipped")]
    assert len(resolution.nodeids) == 2 and not resolution.runs_in_ci and not resolution.counts
    assert resolution.lane == f"{DEFAULT_LANE} and {LANES['skip']}"


def test_in_a_named_lanes_report_only_a_pass_counts(planted, junit):
    """Read from the default lane's report: a pass counts; a skip in the body (which collection never sees), an
    expected failure, a skip-marked test and a slow test the lane deselected do not."""
    names = ["test_runs", "test_skipped_in_body", "test_xfail", "test_marked_skip", "test_slow", "test_skipif_true"]
    resolved = resolve([node(name) for name in names], planted, lanes={"default lane": [junit]})
    shown = {name: (resolved[node(name)].outcomes, resolved[node(name)].counts) for name in names}
    assert shown == {
        "test_runs": ((("default lane", "passed"),), True),
        "test_skipped_in_body": ((("default lane", "skipped"),), False),
        "test_xfail": ((("default lane", "xfailed"),), False),
        "test_marked_skip": ((("default lane", "skipped"),), False),
        "test_slow": ((("default lane", "not in the lane's report"),), False),
        "test_skipif_true": ((("default lane", "skipped"),), False)}
    assert resolved[node("test_xfail")].text().endswith("xfailed in default lane")
    assert junit_outcomes([junit])[node("test_xfail")] == "xfailed"


def test_e1_a_test_whose_file_does_not_exist_fails(planted):
    missing = "tests/test_absent.py::test_x"
    resolution = resolve([missing], planted)[missing]
    assert resolution.problem == "the file tests/test_absent.py does not exist"


@pytest.mark.parametrize("name", ["test_helper", "test_pass_body", "test_wrong_threshold"])
def test_e2_the_defects_collection_cannot_see_resolve_as_passing(planted, junit, name):
    """An expected gap, asserted, so that closing it changes this test: a same-name helper and a body of ``pass`` are
    held by no rule here, and a wrong threshold by one authority per number (the test reads its claim)."""
    resolution = resolve([node(name)], planted, lanes={"default lane": [junit]})[node(name)]
    assert resolution.problem is None and resolution.lane == DEFAULT_LANE
    assert resolution.outcomes == (("default lane", "passed"),) and resolution.counts


def test_e1_of_the_cases_those_that_are_not_evidence_differ_from_the_real_test(planted, junit):
    """With the lane's report read, what did not run and pass there does not count. The planted ``gpu`` test ran and
    passed in it (the planted checkout skips nothing without CUDA), so it counts there, and only there."""
    real = resolve([REAL], ROOT)[REAL]
    cases = ["test_renamed_old", "test_docstring_only", "test_slow", "test_marked_skip", "test_skipped_in_body",
             "test_xfail", "test_one_case_skipped", "test_gpu", "test_helper", "test_pass_body",
             "test_wrong_threshold", "test_runs", "test_skipif_false"]
    resolved = resolve([node(name) for name in cases], planted, lanes={"default lane": [junit]})
    differ = [name for name in cases if not resolved[node(name)].counts]
    assert real.counts and differ == cases[:7], {name: resolved[node(name)].text() for name in cases}
    assert not resolve([node("test_gpu")], planted)[node("test_gpu")].counts


def test_a_collection_round_trips_through_its_json_and_names_every_id_once(planted):
    collection = collect(planted, [FILE])
    assert Collection.from_json(collection.to_json()) == collection
    assert collection.xfail == {node("test_xfail"): "planted"}
    assert set(collection.skipped) == {node("test_skipif_true"), node("test_marked_skip"),
                                       node("test_one_case_skipped[2]")}
    assert sorted(collection.items) == sorted(node(name) for name in (
        "test_renamed_new", "test_documented", "test_skipped_in_body", "test_helper", "test_pass_body",
        "test_wrong_threshold", "test_skipif_false", "test_skipif_true", "test_one_case_skipped[1]",
        "test_one_case_skipped[2]", *PLANTED_LANES))
    assert junit_outcomes([]) == {}


def test_an_id_without_its_parameters_names_each_parametrisation_and_one_with_wrong_ones_is_not_collected():
    base = "tests/saved/test_json_schema.py::test_on_drawn_files_and_their_changes_the_readers_agree"
    resolved = resolve([base, f"{base}[lock]", f"{base}[any-id]"], ROOT)
    assert resolved[base].nodeids == (f"{base}[document]", f"{base}[lock]")
    assert resolved[f"{base}[lock]"].nodeids == (f"{base}[lock]",)
    assert resolved[f"{base}[any-id]"].problem is not None


def good(**changes):
    """A claim that passes, changed by ``changes`` (a ``None`` deletes the field)."""
    held = {"id": "x", "quantity": "an error", "value": 0.1, "unit": "1", "basis": "computed", "role": "bound",
            "oracle_kind": "analytic", "evidence": [REAL], "domain": "the whole range", "n": 10, "held_out": OTHER_REAL,
            "says": "it bounds the error over the range"}
    held.update(changes)
    return {key: value for key, value in held.items() if value is not None}


def test_the_baseline_claim_passes_so_the_controls_can_discriminate():
    assert claim_problems(good(), "MIME-ANO-900") == []


def test_e3_a_declared_claim_that_is_a_bound_fails():
    problems = claim_problems(good(basis="declared", oracle_kind=None, evidence=[]), "MIME-ANO-900")
    assert any("a declared claim is not a bound" in p for p in problems), problems


def test_e4_an_acceptance_that_is_computed_fails():
    problems = claim_problems(good(role="acceptance"), "MIME-ANO-900")
    assert any("a computed claim is not an acceptance" in p and "so it is measured" in p for p in problems), problems


def test_e5_a_bound_without_n_fails():
    problems = claim_problems(good(n=None), "MIME-ANO-900")
    assert any("a bound carries" in p and "it lacks n" in p for p in problems), problems


#: The cells the design's matrix allows, written here apart from the code's table: (basis, role, oracle kind).
ALLOWED = {
    *(("declared", role, None) for role in ("limit", "rule")),
    *(("computed", role, kind) for role in ("bound", "estimate", "limit")
      for kind in ("analytic", "manufactured", "software oracle")),
    *(("computed", role, "implementation") for role in ("estimate", "limit")),
    *(("measured", role, "measurement") for role in ("bound", "estimate", "acceptance", "limit")),
    *(("fitted", role, kind) for role in ("bound", "estimate", "limit")
      for kind in ("analytic", "manufactured", "software oracle", "measurement")),
}


def cell_claim(basis: str, role: str, kind: str | None) -> dict:
    """A claim in one cell, carrying everything else a cell needs (a bound's domain, n and held-out evidence, a fit's
    dataset; evidence where the basis is not declared), so that only the cell decides."""
    return good(basis=basis, role=role, oracle_kind=kind, evidence=[] if basis == "declared" else [REAL],
                dataset="dataset-1" if basis == "fitted" else None,
                held_out=OTHER_REAL if basis != "declared" else None)


@pytest.mark.parametrize(("basis", "role", "kind"), list(itertools.product(BASES, ROLES, (None, *ORACLE_KINDS))))
def test_every_cell_of_the_matrix_is_refused_unless_the_design_allows_it(basis, role, kind):
    """S-18: every disallowed (basis, role, kind) cell is refused, and every allowed one passes."""
    problems = claim_problems(cell_claim(basis, role, kind), "MIME-ANO-900")
    assert (problems == []) == ((basis, role, kind) in ALLOWED), problems


def test_a_fitted_bound_without_held_out_evidence_and_a_fit_without_its_dataset_fail():
    problems = claim_problems(good(basis="fitted", dataset="d", oracle_kind="measurement", held_out=None),
                              "MIME-ANO-900")
    assert any("only on held-out evidence" in p for p in problems), problems
    problems = claim_problems(good(basis="fitted", role="estimate", oracle_kind="measurement"), "MIME-ANO-900")
    assert any("names the fit's dataset" in p for p in problems), problems


@pytest.mark.parametrize("kind", ["software oracle", "manufactured", "analytic"])
def test_control_a_maximum_over_five_points_without_held_out_evidence_is_not_a_bound(kind):
    """The review's P1: a bound from five points with no held-out evidence is a maximum over those points, which the
    design files as an estimate, whatever the oracle (the design's other route, n covering the domain, has no field)."""
    problems = claim_problems(good(oracle_kind=kind, n=5, domain="0.2 to 0.32 R, continuous", held_out=None),
                              "MIME-ANO-900")
    assert any("a computed claim is a bound only on held-out evidence" in p for p in problems), problems
    assert claim_problems(good(oracle_kind=kind, n=5, domain="0.2 to 0.32 R, continuous"), "MIME-ANO-900") == []


def test_control_held_out_evidence_that_is_among_the_claims_own_is_refused():
    for basis, kind in (("computed", "software oracle"), ("fitted", "measurement")):
        problems = claim_problems(good(basis=basis, oracle_kind=kind, dataset="d" if basis == "fitted" else None,
                                       held_out=REAL), "MIME-ANO-900")
        assert any(f"held_out {REAL!r} is among its own evidence" in p for p in problems), problems


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_control_a_value_that_is_not_finite_is_refused(value):
    problems = claim_problems(good(role="estimate", value=value, held_out=None), "MIME-ANO-900")
    assert any("not a finite number" in p for p in problems), problems
    assert claim_problems(good(role="estimate", held_out=None), "MIME-ANO-900") == []


def test_an_implementation_oracle_prints_not_accuracy_evidence_and_an_estimate_its_points():
    rows = claim_rows(read_claims_of([good(role="estimate", oracle_kind="implementation", n=5)]), {})
    shown = rows[0][-1]
    assert "not accuracy evidence" in shown and "from 5 points over the whole range, bounding nothing" in shown


def read_claims_of(claims: list[dict]):
    from mime.docs._evidence import claim_of

    return [claim_of(held, "MIME-ANO-900") for held in claims]


def test_the_registrys_claim_is_the_number_its_test_reads():
    held = claim("MIME-ANO-114.max_error", ROOT / "docs" / "validation" / "known_anomalies.yaml")
    assert (held.role, held.basis, held.n) == ("estimate", "computed", 5)
    assert "MIME-ANO-114.max_error" in read_claims(ROOT / "docs" / "validation" / "known_anomalies.yaml")
    with pytest.raises(KeyError, match="no claim"):
        claim("MIME-ANO-114.nothing", ROOT / "docs" / "validation" / "known_anomalies.yaml")
