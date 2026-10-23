"""A golden output per registered name and version: the first steps of the nodes an entry builds (or, for one that
builds none, of every dynamics in its case), in the first round-trip case that uses it, committed as JSON. A change to
what a version computes shows here as a failure, so a changed law takes a new version; an entry registered without a
golden output fails, as one without a round-trip case fails the registry walk.

Compared within 1e-10 of each field's largest magnitude: across machines XLA may fuse a multiply into an add
differently, which moves the last bits and leaves a residue where an exact zero was, never more over these few steps.
To accept a reviewed change, run once with ``MIME_REGENERATE_GOLDEN=1``: the files are rewritten and the test fails.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from cases import CASES
from node_cases import NODE_CASES

from mime.build._plan import Section
from mime.testing import covered, first_steps, installed_entries, nodes_by_entry

GOLDEN = Path(__file__).parent / "golden" / "outputs"
STEPS = 8
RTOL = 1e-10


def path_of(key: tuple[str, str, str]) -> Path:
    kind, name, version = key
    return GOLDEN / kind.replace(" ", "_") / f"{name}@{version}.json"


def first_case_using(key):
    for case in CASES:
        if case.name == "nudged_toy" and importlib.util.find_spec("mime_lab_fixture") is None:
            if os.environ.get("CI"):
                pytest.fail("the lab fixture package is not installed; every CI lane installs it from ./lab_fixture")
            pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")
        if key in covered(case.make().document())[0]:
            return case
    pytest.fail(f"no round-trip case uses {' '.join(key)}; the registry walk requires one")


def dumped(output: dict) -> str:
    """The output as JSON, one line per field, so a change shows as a diff of the fields it moved."""
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in output.items() if k != "outputs"]
    if "outputs" not in output:
        lines[-1] = lines[-1].rstrip(",")
        return "\n".join([*lines, "}"]) + "\n"
    lines.append(' "outputs": {')
    nodes = list(output["outputs"].items())
    for i, (node, fields) in enumerate(nodes):
        lines.append(f"  {json.dumps(node)}: {{")
        for j, (field, value) in enumerate(fields.items()):
            lines.append(f"   {json.dumps(field)}: {json.dumps(value)}{',' if j < len(fields) - 1 else ''}")
        lines.append("  }" + ("," if i < len(nodes) - 1 else ""))
    return "\n".join([*lines, " }", "}"]) + "\n"


def output_of(key) -> dict:
    case = first_case_using(key)
    if case.refused:
        return {"entry": " ".join(key), "case": case.name, "refused": list(case.refused)}
    built = case.make().build()
    nodes = sorted(nodes_by_entry(built).get(key, {})) or [
        name for plan in built.plans.entries if plan.section is Section.DYNAMICS for name in plan.node_names()]
    history = first_steps(built, STEPS)
    return {"entry": " ".join(key), "case": case.name, "steps": STEPS,
            "outputs": {node: {field: np.asarray(value).tolist() for field, value in sorted(history[node].items())}
                        for node in nodes}}


def differences(found: dict, golden: dict) -> list[str]:
    """Every way ``found`` moved from ``golden``: its case, its refusal, its nodes and fields, or a value beyond
    ``RTOL`` of its field's largest magnitude."""
    if {k: v for k, v in found.items() if k != "outputs"} != {k: v for k, v in golden.items() if k != "outputs"}:
        return [f"the case or its refusal changed: {found.get('case')} {found.get('refused')}"]
    if "outputs" not in golden:
        return []
    if found["outputs"].keys() != golden["outputs"].keys():
        return [f"other nodes: {sorted(found['outputs'])}"]
    moved: list[str] = []
    for node, fields in golden["outputs"].items():
        if found["outputs"][node].keys() != fields.keys():
            moved.append(f"{node}: other fields")
            continue
        for field, value in fields.items():
            want, got = np.asarray(value), np.asarray(found["outputs"][node][field])
            scale = float(np.max(np.abs(want), initial=0.0))
            if want.shape != got.shape or float(np.max(np.abs(got - want), initial=0.0)) > RTOL * scale:
                moved.append(f"{node}.{field} moved from its golden output")
    return moved


@pytest.fixture(scope="module")
def registered():
    found = installed_entries()
    assert ("interaction", "toy.spring", "0.1.0") in found, "the walk would pass over an empty registry"
    return found


def test_every_registered_name_and_version_has_a_golden_output(registered):
    missing = [" ".join(key) for key in map(tuple, registered) if not path_of(key).exists()]
    assert not missing, ("no golden output for: " + "; ".join(missing)
                         + ". Run tests/saved/test_golden_outputs.py with MIME_REGENERATE_GOLDEN=1.")


def test_a_golden_output_left_out_is_named():
    assert not path_of(("interaction", "toy.uncased", "0.1.0")).exists()


@pytest.mark.parametrize("key", sorted(NODE_CASES), ids=" ".join)
def test_each_name_and_version_reproduces_its_golden_output(key):
    found, path = output_of(key), path_of(key)
    if os.environ.get("MIME_REGENERATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumped(found))
        pytest.fail(f"rewrote {path.name}; review the diff and rerun without MIME_REGENERATE_GOLDEN")
    assert differences(found, json.loads(path.read_text())) == []


def test_a_moved_output_is_caught_and_a_rounding_residue_is_not():
    """The control: one value of the torque's golden output moved by 1e-8 of its field's scale is caught; moved by
    1e-12, it is within the tolerance across machines."""
    golden = json.loads(path_of(("interaction", "magnetic.torque", "0.1.0")).read_text())
    for shift, caught in ((1e-8, True), (1e-12, False)):
        moved = json.loads(json.dumps(golden))
        rows = moved["outputs"]["magnetic.torque:umr"]["wrench"]
        scale = max(abs(v) for row in rows for v in row)
        rows[-1][3] += shift * scale
        assert bool(differences(moved, golden)) is caught
