"""The document: what it holds, that it holds no code, and that its hash is the experiment's and not its formatting's.
"""

import json
import math

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.experiment import Experiment
from mime.saved import (
    DOCUMENT_FILE,
    LoadRefused,
    document_hash,
    dump_document,
    parse_yaml,
    read_document,
)
from mime.testing import declare_in_order
from mime.testing.toys import ToyDrag, coupled_toy, full_declarations, full_toy, toy_experiment

PROPERTY = settings(max_examples=100, deadline=None)


def test_an_alias_saves_as_its_family_and_rung():
    experiment = coupled_toy()
    assert ToyDrag.Quadratic() == ToyDrag(rung="quadratic")
    (drag,) = [d for d in experiment.document()["interactions"] if d["id"] == "toy.drag:probe"]
    assert drag == {"id": "toy.drag:probe", "name": "toy.drag", "version": "0.1.0", "binds": {"body": "probe"},
                    "settings": {"rung": "quadratic"}}


def test_the_document_holds_names_versions_settings_and_bindings_and_empty_slots():
    document = full_toy().document()
    assert (document["schema"], document["experiment"]) == ("mime.document/1", "full_toy")
    assert document["run"] == {"dt": {"value": 0.02, "unit": "s"}, "seed": 0}
    assert document["frame"] == {"up": [0.0, 0.0, 1.0], "gravity": {"value": 9.80665, "unit": "m/s^2"}}
    assert document["things"]["probe"] == {
        "kind": "body", "dynamics": {"motion": {"name": "toy.body", "version": "0.1.0"}},
        "initial": {"position": {"value": [0.3, -0.2, 0.1], "unit": "m"},
                    "velocity": {"value": [0.5, 0.1, -0.3], "unit": "m/s"}}}
    assert document["things"]["source"]["fields"] == {"push": {"name": "toy.field", "version": "0.1.0"}}
    (recorder,) = document["observations"]
    assert recorder["settings"] == {"rate": {"value": 0.04, "unit": "s"}} and recorder["binds"] == {"body": "probe"}
    assert (document["materials"], document["scene"], document["controller"]) == ({}, None, None)


@PROPERTY
@given(order=st.permutations(range(len(full_declarations()))))
def test_the_order_of_a_scripts_calls_changes_nothing_in_the_document(order):
    reference = full_toy().document()
    found = declare_in_order(lambda: toy_experiment("full_toy"), full_declarations(), order).document()
    assert dump_document(found) == dump_document(reference)


def test_formatting_never_changes_the_hash():
    document = full_toy().document()
    text = dump_document(document)
    restyled = yaml.safe_dump(json.loads(json.dumps(document)), default_flow_style=True, sort_keys=True, width=40)
    commented = "# a comment\n" + text.replace("experiment: full_toy", "experiment:    'full_toy'   # quoted")
    for other in (restyled, commented, json.dumps(document, indent=3)):
        assert other != text
        assert document_hash(parse_yaml(other)) == document_hash(document)


def test_a_change_of_content_changes_the_hash():
    document = full_toy().document()
    changed = parse_yaml(dump_document(document).replace("rung: quadratic", "rung: linear"))
    assert document_hash(changed) != document_hash(document)


@PROPERTY
@given(value=st.floats(allow_nan=False, allow_infinity=False))
def test_every_finite_float_survives_the_document_bit_for_bit(value):
    document = coupled_toy().document()
    document["run"]["dt"]["value"] = value
    found = parse_yaml(dump_document(document))["run"]["dt"]["value"]
    assert found == value and math.copysign(1.0, found) == math.copysign(1.0, value) and type(found) is float


def _write(tmp_path, text):
    path = tmp_path / DOCUMENT_FILE
    path.write_text(text)
    return path


def test_a_python_tag_is_refused_and_constructs_nothing(tmp_path):
    """A tag that, read by an unsafe loader, opens (and so creates) a file. The control: ``yaml.unsafe_load`` of the
    same text does create it, so the payload is live and its absence after MIME's load means something."""
    target = tmp_path / "written_by_the_payload"
    payload = dump_document(coupled_toy().document()).replace(
        "controller: null", f"controller: !!python/object/apply:builtins.open ['{target}', 'w']")
    _write(tmp_path, payload)
    with pytest.raises(LoadRefused) as refused:
        Experiment.from_document(tmp_path / DOCUMENT_FILE)
    print(f"\n{refused.value}")
    assert refused.value.codes() == ["load.code-in-document"]
    assert "a document never holds code" in refused.value.refusals[0].message
    assert not target.exists()
    yaml.unsafe_load(payload).get("controller").close()
    assert target.exists(), "the control: an unsafe loader runs the payload"


def test_a_key_given_twice_is_refused(tmp_path):
    text = dump_document(coupled_toy().document()).replace("seed: 0", "seed: 0\n  seed: 1")
    document, refusals = read_document(_write(tmp_path, text))
    assert document is None and [r.code for r in refusals] == ["load.document-yaml"]
    assert "the key 'seed' appears twice" in refusals[0].message


def test_an_unknown_key_and_a_value_of_another_type_are_refused_by_path(tmp_path):
    text = dump_document(coupled_toy().document()).replace("seed: 0", "seed: zero\n  speed: 3")
    document, refusals = read_document(_write(tmp_path, text))
    assert document is None
    messages = [r.message for r in refusals]
    assert any("$.run: unknown key 'speed'" in m for m in messages)
    assert any("$.run.seed: 'zero' is not an integer" in m for m in messages)


def test_a_slot_that_is_not_built_yet_stays_empty(tmp_path):
    text = dump_document(coupled_toy().document()).replace("materials: {}", "materials: {water: {id: x}}")
    _, refusals = read_document(_write(tmp_path, text))
    assert [r.message.split(": ", 1)[1] for r in refusals] == [
        "$.materials.water: nothing may be written here yet (the slot is kept empty until it is built)"]
