"""The anomaly registry and its validator, ``scripts/check_anomalies.py``.

The real registry must pass. Each control is a small registry written to ``tmp_path`` that breaks one rule, beside a
baseline that passes, so a refusal is the rule's and not an accident of the file's shape; the conversion check's gate
control breaks a copy of the real registry. Evidence resolves through pytest's collection of the files the registry
names, made once for the module.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_anomalies.py"
REGISTRY = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"
THIS_TEST = "tests/test_anomaly_registry.py::test_the_real_registry_passes"
PARAMETRISED = "tests/saved/test_json_schema.py::test_on_drawn_files_and_their_changes_the_readers_agree"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_gate_check_anomalies", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def collection(gate):
    return gate.collect(REPO_ROOT, ["tests/test_anomaly_registry.py", "tests/saved/test_json_schema.py"])


@pytest.fixture(scope="module")
def components(gate):
    return gate.registered_components()


def entry(**overrides):
    fields = {
        "anomaly_id": "MIME-ANO-900",
        "title": "A control",
        "description": "A control entry.",
        "severity": "minor",
        "safety_relevance": "not_safety_relevant",
        "safety_relevance_rationale": "A control.",
        "resolution_status": "resolved",
        "affected_versions": ">=0.1.0, <0.3.0.dev1",
        "resolution_version": "0.3.0.dev1",
        "verification": [THIS_TEST],
    }
    fields.update(overrides)
    return {key: value for key, value in fields.items() if value is not None}


def write(tmp_path, anomaly):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "2.0", "generated_date": "2026-10-23", "anomalies": [anomaly]}))
    return path


@pytest.fixture
def errors_of(gate, tmp_path, collection, components):
    def errors(anomaly):
        return gate.validate_anomaly_registry(write(tmp_path, anomaly), prefix="MIME-ANO-", repository=REPO_ROOT,
                                              collection=collection, components=components)
    return errors


def test_the_real_registry_passes(gate, components):
    assert gate.validate_anomaly_registry(REGISTRY, prefix="MIME-ANO-", repository=REPO_ROOT,
                                          components=components) == []


def test_the_real_registry_keeps_every_entry_and_gives_each_a_status_and_a_range():
    anomalies = yaml.safe_load(REGISTRY.read_text())["anomalies"]
    assert [a["anomaly_id"] for a in anomalies] == (
        [f"MIME-ANO-{n:03d}" for n in (1, 2, 3)] + [f"MIME-ANO-{n}" for n in range(100, 157)]
    )
    assert all(a["resolution_status"] and a["affected_versions"] for a in anomalies)


def test_the_script_runs_as_ci_runs_it():
    done = subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("OK:") and "evidence ids collected" in done.stdout


def test_the_baseline_passes_so_the_controls_below_can_discriminate(errors_of):
    assert errors_of(entry()) == []


def test_a_missing_affected_versions_is_refused(errors_of):
    errors = errors_of(entry(affected_versions=None))
    assert any("MIME-ANO-900" in e and "affected_versions" in e for e in errors), errors


def test_the_old_status_field_is_refused(errors_of):
    old = entry(resolution_status=None)
    old["status"] = "open"
    errors = errors_of(old)
    assert any("MIME-ANO-900" in e and "'status'" in e and "resolution_status" in e for e in errors), errors


def test_the_old_affected_nodes_field_is_refused(errors_of):
    errors = errors_of(entry(affected_nodes=["umr_ode"]))
    assert any("MIME-ANO-900" in e and "'affected_nodes'" in e and "affected_components" in e for e in errors), errors


def test_a_status_of_closed_is_refused(errors_of):
    errors = errors_of(entry(resolution_status="closed"))
    assert any("MIME-ANO-900" in e and "'closed'" in e for e in errors), errors


def test_an_unknown_field_is_refused(errors_of):
    errors = errors_of(entry(closing_test="tests/x.py::test_x"))
    assert any("MIME-ANO-900" in e and "unknown field 'closing_test'" in e for e in errors), errors


def test_a_resolved_entry_without_resolution_version_is_refused(errors_of):
    errors = errors_of(entry(resolution_version=None))
    assert any("MIME-ANO-900" in e and "resolution_version" in e for e in errors), errors


def test_a_resolved_entry_without_a_verification_list_is_refused(errors_of):
    errors = errors_of(entry(verification=None))
    assert any("MIME-ANO-900" in e and "verification" in e for e in errors), errors


def test_an_empty_verification_list_is_accepted_on_a_resolved_entry(errors_of):
    assert errors_of(entry(verification=[])) == []


def test_a_verification_id_naming_a_missing_test_function_is_refused(errors_of):
    errors = errors_of(entry(verification=["tests/test_canary.py::test_that_is_not_there"]))
    assert any("MIME-ANO-900" in e and "test_that_is_not_there" in e for e in errors), errors


def test_a_verification_id_naming_a_missing_file_is_refused(errors_of):
    errors = errors_of(entry(verification=["tests/no_such_file.py::test_x"]))
    assert any("MIME-ANO-900" in e and "no_such_file.py" in e for e in errors), errors


def test_a_parametrised_verification_id_may_drop_its_bracket_part_and_names_each_parametrisation(errors_of):
    assert errors_of(entry(verification=[PARAMETRISED, f"{PARAMETRISED}[lock]"])) == []


def test_a_parametrised_verification_id_whose_bracket_is_not_collected_is_refused(errors_of):
    errors = errors_of(entry(verification=[f"{PARAMETRISED}[any-id]"]))
    assert any("MIME-ANO-900" in e and "is not collected" in e for e in errors), errors


def test_a_verification_id_naming_a_test_only_in_a_comment_is_refused(errors_of):
    """The text rule this replaced passed a name left in a comment; collection does not."""
    errors = errors_of(entry(verification=["tests/test_anomaly_registry.py::test_left_in_a_comment"]))
    assert any("MIME-ANO-900" in e and "is not collected" in e for e in errors), errors
    # test_left_in_a_comment


def test_a_claim_with_a_shape_error_and_one_whose_evidence_is_not_collected_are_refused(errors_of):
    held = {"id": "error", "quantity": "an error", "value": 0.1, "unit": "1", "basis": "declared", "role": "bound",
            "says": "a planted claim"}
    errors = errors_of(entry(claims=[held]))
    assert any("MIME-ANO-900.error: a declared claim is not a bound" in e for e in errors), errors
    held = {**held, "basis": "computed", "role": "estimate", "oracle_kind": "analytic",
            "evidence": ["tests/test_anomaly_registry.py::test_no_such_test"]}
    errors = errors_of(entry(claims=[held]))
    assert any("MIME-ANO-900.error: evidence" in e and "is not collected" in e for e in errors), errors
    assert errors_of(entry(claims=[{**held, "evidence": [THIS_TEST]}])) == []


@pytest.fixture(scope="module")
def lanes(gate, tmp_path_factory):
    """A checkout of its own holding planted tests CI does not run (slow, xfail, skip, gpu) beside one it runs, and
    its collection."""
    sys.path.insert(0, str(REPO_ROOT / "tests" / "docs"))
    try:
        from checkout import PLANTED_LANES, planted_checkout
    finally:
        sys.path.remove(str(REPO_ROOT / "tests" / "docs"))
    root = planted_checkout(tmp_path_factory.mktemp("lanes"), {
        "tests/test_planted.py": "import pytest\n\n\n" + "\n\n".join(PLANTED_LANES.values())})
    return root, gate.collect(root, ["tests/test_planted.py"])


@pytest.mark.parametrize("name", ["test_xfail", "test_slow", "test_marked_skip", "test_gpu"])
def test_control_a_claim_or_a_resolution_resting_on_a_test_ci_does_not_run_is_refused(gate, tmp_path, lanes, name):
    """Evidence is a test that ran and passed in a named lane: a claim whose evidence is an xfail, a slow, a skip-marked
    or a gpu test, and a resolved entry verified only by one, are each refused as not run in CI; the same entry citing
    the planted test CI runs is the baseline, and passes."""
    root, collection = lanes
    planted = f"tests/test_planted.py::{name}"
    held = {"id": "error", "quantity": "an error", "value": 0.1, "unit": "1", "basis": "computed", "role": "estimate",
            "oracle_kind": "analytic", "says": "a planted claim"}

    def errors(cited):
        anomaly = entry(verification=[cited], claims=[{**held, "evidence": [cited]}])
        return gate.validate(write(tmp_path, anomaly), prefix="MIME-ANO-", repository=root, collection=collection,
                             components=gate.components_from({"entries": {}, "names": [], "nodes": []})).errors

    assert errors("tests/test_planted.py::test_runs") == []
    found = errors(planted)
    assert any(e.startswith("MIME-ANO-900.error: its evidence is tests CI does not run") and planted in e
               for e in found), found
    assert any(e.startswith("MIME-ANO-900: a resolved entry's verification is tests CI does not run") for e in found)
    assert all("not run in CI (" in e for e in found), found


def test_a_legacy_spelling_of_a_registered_component_is_refused_and_of_an_unregistered_one_is_allowed(errors_of):
    errors = errors_of(entry(affected_components=["mime.nodes.actuation.motor"]))
    assert any("MIME-ANO-900" in e and "legacy spelling of motor" in e for e in errors), errors
    errors = errors_of(entry(affected_components=["SurrogateDragNode (the step's output)"]))
    assert any("legacy spelling of SurrogateDragNode" in e for e in errors), errors
    assert errors_of(entry(affected_components=["umr_ode", "MIME-NODE-101", "mime.core.tables (evaluate)"])) == []


def test_the_controlled_form_names_a_registered_entry_rung_and_node(errors_of):
    good = ["entry: interaction drag 0.3.0, rung mrs; a note", "entry: dynamics motor", "node: MRSDragNode; a note"]
    assert errors_of(entry(affected_components=good)) == []
    for bad, why in (("entry: interaction no_such_law", "names no registered entry"),
                     ("entry: interaction drag, rung lbm", "names the rung 'lbm'"),
                     ("node: NoSuchNode", "names no node class"),
                     ("entry: drag", "is not 'entry:")):
        errors = errors_of(entry(affected_components=[bad]))
        assert any(why in e for e in errors), (bad, errors)


def test_control_a_converted_component_written_back_in_its_legacy_spelling_fails_the_real_registry(
        gate, tmp_path, components):
    """The conversion check's gate clause, on a copy of the real registry: one converted item of MIME-ANO-122 put back
    as it was spelt before, which names a node class registered entries build."""
    text = REGISTRY.read_text(encoding="utf-8")
    assert text.count('- "node: DynamicMotorNode"') == 1
    planted = tmp_path / "known_anomalies.yaml"
    planted.write_text(text.replace('- "node: DynamicMotorNode"', '- "motor (rung dynamic), DynamicMotorNode"'),
                       encoding="utf-8")
    errors = gate.validate_anomaly_registry(planted, prefix="MIME-ANO-", repository=REPO_ROOT, components=components)
    assert errors and all("MIME-ANO-122" in e and "legacy spelling of DynamicMotorNode, motor" in e
                          for e in errors), errors

