"""The anomaly registry and its validator, ``scripts/check_anomalies.py``.

The real registry must pass. Each control is a small registry written to ``tmp_path`` that breaks one rule, beside a
baseline that passes, so a refusal is the rule's and not an accident of the file's shape.
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


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_gate_check_anomalies", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def errors_of(gate, tmp_path, anomaly):
    return gate.validate_anomaly_registry(write(tmp_path, anomaly), prefix="MIME-ANO-", repository=REPO_ROOT)


def test_the_real_registry_passes(gate):
    assert gate.validate_anomaly_registry(REGISTRY, prefix="MIME-ANO-", repository=REPO_ROOT) == []


def test_the_real_registry_keeps_every_entry_and_gives_each_a_status_and_a_range():
    anomalies = yaml.safe_load(REGISTRY.read_text())["anomalies"]
    assert [a["anomaly_id"] for a in anomalies] == (
        [f"MIME-ANO-{n:03d}" for n in (1, 2, 3)] + [f"MIME-ANO-{n}" for n in range(100, 129)]
    )
    assert all(a["resolution_status"] and a["affected_versions"] for a in anomalies)


def test_the_script_runs_as_ci_runs_it():
    done = subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("OK:")


def test_the_baseline_passes_so_the_controls_below_can_discriminate(gate, tmp_path):
    assert errors_of(gate, tmp_path, entry()) == []


def test_a_missing_affected_versions_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(affected_versions=None))
    assert any("MIME-ANO-900" in e and "affected_versions" in e for e in errors), errors


def test_the_old_status_field_is_refused(gate, tmp_path):
    old = entry(resolution_status=None)
    old["status"] = "open"
    errors = errors_of(gate, tmp_path, old)
    assert any("MIME-ANO-900" in e and "'status'" in e and "resolution_status" in e for e in errors), errors


def test_the_old_affected_nodes_field_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(affected_nodes=["umr_ode"]))
    assert any("MIME-ANO-900" in e and "'affected_nodes'" in e and "affected_components" in e for e in errors), errors


def test_a_status_of_closed_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(resolution_status="closed"))
    assert any("MIME-ANO-900" in e and "'closed'" in e for e in errors), errors


def test_an_unknown_field_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(closing_test="tests/x.py::test_x"))
    assert any("MIME-ANO-900" in e and "unknown field 'closing_test'" in e for e in errors), errors


def test_a_resolved_entry_without_resolution_version_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(resolution_version=None))
    assert any("MIME-ANO-900" in e and "resolution_version" in e for e in errors), errors


def test_a_resolved_entry_without_a_verification_list_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(verification=None))
    assert any("MIME-ANO-900" in e and "verification" in e for e in errors), errors


def test_an_empty_verification_list_is_accepted_on_a_resolved_entry(gate, tmp_path):
    assert errors_of(gate, tmp_path, entry(verification=[])) == []


def test_a_verification_id_naming_a_missing_test_function_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(verification=["tests/test_canary.py::test_that_is_not_there"]))
    assert any("MIME-ANO-900" in e and "test_that_is_not_there" in e for e in errors), errors


def test_a_verification_id_naming_a_missing_file_is_refused(gate, tmp_path):
    errors = errors_of(gate, tmp_path, entry(verification=["tests/no_such_file.py::test_x"]))
    assert any("MIME-ANO-900" in e and "no_such_file.py" in e for e in errors), errors


def test_a_parametrised_verification_id_may_drop_its_bracket_part(gate, tmp_path):
    parametrised = f"{THIS_TEST}[any-id]"
    assert errors_of(gate, tmp_path, entry(verification=[parametrised])) == []

