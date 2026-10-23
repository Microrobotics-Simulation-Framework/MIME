"""The saved form of script A's body side: the document holds the tube's geometry kind, the region the water fills,
the drive of each command, the directions written by reference, and every value declared in the experiment with its
unit and reason; the lock's resolved record labels each value "declared in the experiment" and pins the geometry kind
and the drive; reading the document back gives the same document. And what a load refuses of such a document."""

import pytest
from script_a import script_a

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LoadRefused, document_hash, dump_document, parse_yaml, read_document


def test_the_document_holds_the_geometry_the_fill_the_drives_and_the_declared_values():
    document = script_a().document()
    assert document["things"]["tube"] == {"kind": "structure", "prim": "/World/tube", "geometry": {
        "name": "tube", "version": "0.1.0", "settings": {
            "axis": [1.0, 0.0, 0.0], "inner_radius": {"value": 0.003175, "unit": "m"},
            "wall": {"value": 0.0016, "unit": "m"}, "length": {"value": 0.1, "unit": "m"}}}}
    assert document["things"]["water"] == {"kind": "medium", "prim": "/World/water", "fills": "tube.lumen"}
    assert document["things"]["field"]["commands"] == {
        "amplitude": {"name": "drive.constant", "version": "0.1.0", "settings": {"value": {"value": 0.0012,
                                                                                          "unit": "T"}}},
        "frequency": {"name": "drive.constant", "version": "0.1.0", "settings": {"value": {"value": 10.0,
                                                                                          "unit": "Hz"}}}}
    assert document["things"]["field"]["fields"]["magnetic"]["settings"]["rotation_axis"] == "tube.axis"
    assert document["declared"]["umr.mass"] == {"value": 8.491116020725203e-05, "unit": "kg",
                                                "reason": "A's reference provenance (phase 0, MIME-port041 at "
                                                          "6485516)"}
    assert document["declared"]["umr.inertia"]["unit"] == "kg m^2"
    assert document["materials"] == {}


def test_read_back_the_document_is_the_same_document(tmp_path):
    document = script_a().document()
    (tmp_path / DOCUMENT_FILE).write_text(dump_document(document), encoding="utf-8")
    loaded = Experiment.from_document(tmp_path / DOCUMENT_FILE)
    assert document_hash(loaded.document()) == document_hash(document)
    assert loaded.build().report.derived_hash == script_a().build().report.derived_hash


def test_the_lock_labels_each_value_and_pins_the_geometry_and_the_drive(tmp_path):
    lock = script_a().build().save(tmp_path)
    resolved = lock["resolved"]
    assert resolved["values"]["water.density"] == {"value": 1000.0, "unit": "kg/m^3",
                                                   "source": "declared in the experiment",
                                                   "reason": "A's reference provenance (phase 0, MIME-port041 at "
                                                             "6485516)", "read_by": ["gravity:umr"]}
    assert resolved["geometry"]["tube"]["settings"]["centre"] == {"value": [0.0, 0.0, 0.0], "unit": "m"}
    assert resolved["fills"] == {"water": "tube.lumen"}
    assert {("geometry", "tube"), ("drive", "drive.constant")} <= {(e["kind"], e["name"]) for e in lock["entries"]}
    assert resolved["order"][-2:] == ["drive:field.amplitude", "drive:field.frequency"]


@pytest.mark.parametrize(("edit", "code"), [
    (lambda text: text.replace("unit: kg/m^3", "unit: furlong"), "load.units"),
    (lambda text: text.replace("    fills: tube.lumen", "    fills: tube.lumen\n    commands: {}"),
     "load.document-thing"),
    (lambda text: text.replace("name: tube\n", "name: tubing\n"), "load.unknown-entry"),
])
def test_a_load_refuses_an_edited_document_naming_what_to_do(tmp_path, edit, code):
    text = edit(dump_document(script_a().document()))
    assert parse_yaml(text) is not None
    (tmp_path / DOCUMENT_FILE).write_text(text, encoding="utf-8")
    document, refusals = read_document(tmp_path / DOCUMENT_FILE)
    if refusals:
        assert code in [r.code for r in refusals]
        return
    with pytest.raises(LoadRefused) as refused:
        Experiment.from_document(tmp_path / DOCUMENT_FILE)
    print(f"\n{refused.value}")
    assert code in refused.value.codes()


# -- the drag's asset and its group of settings ----------------------------------------------------------------------

SHA = "ac1fca5c109fa3e2f27520cbd0ee1c0ce5e85ac31a0b1e93e7b8085c68f0a345"


def test_the_document_writes_the_weights_by_name_and_hash_and_the_lubrication_as_its_group():
    (drag,) = [d for d in script_a().document()["interactions"] if d["name"] == "drag"]
    assert drag["settings"]["weights"] == {"asset": "dejongh_mlp_v2", "sha256": SHA}
    assert drag["settings"]["lubrication"] == {"group": "lubrication",
                                               "settings": {"epsilon": {"value": 0.00015, "unit": "m"}}}


def test_the_lock_records_the_asset_by_source_and_hash(tmp_path):
    lock = script_a().build().save(tmp_path)
    assert lock["resolved"]["assets"] == {"dejongh_mlp_v2": {"source": "mime.data/mlp_cholesky_weights_v2.npz",
                                                             "sha256": SHA}}
    statics = lock["derived"]["physics"]["static"]["drag:umr"]
    assert statics["mlp_W0"]["shape"] == [8, 128] and len(statics["mlp_W0"]["sha256"]) == 64


def test_a_load_refuses_an_installation_that_ships_other_weights_under_the_same_name(tmp_path, monkeypatch):
    """The saved experiment recorded the weights' sha256; an installation whose file under that name hashes otherwise
    is refused on load, naming both hashes. The control: the same folder loads with the packaged file."""
    import hashlib

    import mime.data

    script_a().build().save(tmp_path)
    Experiment.load(tmp_path)
    other = b"other weights"
    monkeypatch.setattr(mime.data, "_contents", lambda file: (other, hashlib.sha256(other).hexdigest()))
    with pytest.raises(BuildRefused) as refused:
        Experiment.load(tmp_path)
    print(f"\n{refused.value}")
    assert refused.value.codes() == ["setting.value"]
    assert f"has sha256 {hashlib.sha256(other).hexdigest()}, and the experiment asks for {SHA}" in str(refused.value)
