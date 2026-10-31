"""Records as pinned files, and the saved form's references to them.

- A record file edited by one byte is refused: at build, when the experiment pins it; at load, when the saved folder
  carries it. The control: the unedited file builds and loads.
- The saved document holds references, never a value copied out of a record: a search of it finds none of the
  records' values, and the same search finds them in the lock (the control that the search can find them).
- The lock pins each record's file and its id, version and content hash, and holds every read as sent and answered.
- Only ``mime.materials`` imports OpenMaterica (the control: the scan finds the seam's own imports, and a planted
  one elsewhere).
"""

import ast
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from records import N45_BOUNDS, WATER_VALUES, water_experiment
from script_a import CONDITIONS, WATER_RECORD, record_arm

import mime
from mime import Experiment
from mime.build import BuildRefused
from mime.data import asset, asset_bytes
from mime.materials import record_file
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused

PACKAGE = Path(mime.__file__).parent


def copied_water(folder: Path) -> Path:
    path = folder / "water.json"
    path.write_bytes(asset_bytes(asset("openmaterica_example_water")))
    return path


def edited_by_one_byte(path: Path) -> None:
    held = bytearray(path.read_bytes())
    index = held.index(b"Liquid water")  # in the record's name: one letter, the record otherwise intact
    held[index] = ord("l")
    path.write_bytes(bytes(held))


def test_a_pinned_record_file_edited_by_one_byte_is_refused_at_build(tmp_path):
    path = copied_water(tmp_path)
    pinned = replace(WATER_RECORD, file=record_file(path, sha256=asset("openmaterica_example_water").sha256))
    assert water_experiment(pinned, CONDITIONS).plan().values  # the control: the unedited copy builds
    edited_by_one_byte(path)
    with pytest.raises(BuildRefused) as caught:
        water_experiment(pinned, CONDITIONS).plan()
    assert [r.code for r in caught.value.refusals] == ["record.changed"]


def test_a_record_file_that_cannot_be_read_is_refused_at_build_before_its_contents_are_judged(tmp_path):
    path = copied_water(tmp_path)
    pinned = replace(WATER_RECORD, file=record_file(path, sha256=asset("openmaterica_example_water").sha256))
    assert water_experiment(pinned, CONDITIONS).plan().values  # the control: the copy that is there builds
    path.unlink()
    with pytest.raises(BuildRefused) as caught:
        water_experiment(pinned, CONDITIONS).plan()
    assert [r.code for r in caught.value.refusals] == ["record.file"]


def test_a_saved_record_file_edited_by_one_byte_is_refused_at_load(tmp_path):
    record_arm().save(tmp_path)
    assert Experiment.load(tmp_path).load_report is not None  # the control: the saved folder loads
    edited_by_one_byte(tmp_path / "records" / "water.json")
    with pytest.raises(LoadRefused) as caught:
        Experiment.load(tmp_path)
    assert [r.code for r in caught.value.refusals] == ["load.record-changed"]


def numbers(text: str) -> set[float]:
    """Every number in a YAML or JSON text, as the loaders read them."""
    import yaml

    found: set[float] = set()

    def walk(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(yaml.safe_load(text))
    return found


def test_the_saved_document_holds_no_record_value_and_the_lock_holds_them(tmp_path):
    record_arm().save(tmp_path)
    document = (tmp_path / DOCUMENT_FILE).read_text(encoding="utf-8")
    lock = (tmp_path / LOCK_FILE).read_text(encoding="utf-8")
    values = {*WATER_VALUES, *N45_BOUNDS}
    assert numbers(document) & values == set()
    for value in WATER_VALUES[:1] + WATER_VALUES[2:3]:
        assert repr(value) not in document
    assert {998.207, 0.0010016, *N45_BOUNDS} <= numbers(lock)  # the control: the search finds them where they are


def test_the_lock_pins_every_record_and_holds_every_read_as_sent_and_answered(tmp_path):
    lock = record_arm().save(tmp_path)
    materials = lock["resolved"]["materials"]
    assert materials["records"]["water"] == {
        "file": "records/water.json", "sha256": asset("openmaterica_example_water").sha256, "id": "example/water",
        "version": "1.0.1", "content_hash": "sha256:bafb89c2805a1cb7b042f62a3973f3d16778fb5f6d7bfac641a054ccc510c914"}
    assert lock["records"]["n45"] == {"file": "records/ndfeb_n45.json",
                                      "sha256": asset("openmaterica_example_ndfeb_n45").sha256}
    reads = {(read["material"], read["field"]): read for read in materials["reads"]}
    viscosity = reads[("water", "fluid.dynamic_viscosity")]
    assert viscosity["status"] == "unknown" and viscosity["request"]["target"]["record"]["id"] == "example/water"
    assert viscosity["request"]["operating_conditions"] == {"temperature": {"value": 293.15, "unit": "K"}}
    assert viscosity["request"]["requirement"]["conditions"]["temperature"] == {"tolerance": {"value": 0.5,
                                                                                              "unit": "K"}}
    assert reads[("n45", "magnetic.remanence")]["bounds"] == list(N45_BOUNDS)
    assert lock["environment"]["openmaterica"]
    json.dumps(lock)


def test_a_record_file_missing_from_the_folder_is_refused_at_load(tmp_path):
    record_arm().save(tmp_path)
    (tmp_path / "records" / "ndfeb_n45.json").unlink()
    with pytest.raises(LoadRefused) as caught:
        Experiment.load(tmp_path)
    assert "load.record-changed" in [r.code for r in caught.value.refusals]


def importers_of_materica() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if imports_materica(path.read_text(encoding="utf-8")):
            relative = path.relative_to(PACKAGE)
            found.setdefault(relative.parts[0], []).append(str(relative))
    return found


def imports_materica(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "materica" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "materica":
            return True
    return False


def test_only_mime_materials_imports_openmaterica():
    found = importers_of_materica()
    assert "materials" in found, "the scan found no importer at all, so it could not fail"
    assert set(found) == {"materials"}, found
    assert imports_materica("from materica import read") and not imports_materica("import materials")


def test_the_saved_folder_carries_its_records_byte_for_byte(tmp_path):
    record_arm().save(tmp_path)
    for name, file in (("openmaterica_example_water", "water.json"), ("openmaterica_example_ndfeb_n45",
                                                                      "ndfeb_n45.json")):
        assert (tmp_path / "records" / file).read_bytes() == asset_bytes(asset(name))
    shutil.rmtree(tmp_path)


def test_every_declaration_of_a_material_round_trips_through_the_saved_form(tmp_path):
    """A material with each of its terms (required tokens, an acceptance, an extrapolation) and the conditions saves
    and loads to the same document and the same graph; the record arm's (a nominal, parts) is the registry walk's."""
    from records import density_table

    from mime.materials import Acceptance, Extrapolation, Record
    from mime.saved import document_hash
    from mime.units import K, Condition

    folder = tmp_path / "records"
    folder.mkdir()
    table = Record("table", record_file(density_table(folder)), require={"density_kind": ("apparent",)},
                   accept={"bulk.density": Acceptance(("condition_unstated",), reason="not needed here")},
                   extrapolation={"bulk.density": Extrapolation("linear", basis="the end segment", reason="a test")})
    experiment = water_experiment(table, {"temperature": Condition(K(350.0), K(0.5))})
    built = experiment.build()
    built.save(tmp_path / "saved")
    loaded = Experiment.load(tmp_path / "saved")
    assert document_hash(loaded.document()) == document_hash(experiment.document())
    assert loaded.build().record.hashes() == built.record.hashes()
    assert loaded.document()["materials"]["table"]["extrapolation"] == {
        "bulk.density": {"method": "linear", "basis": "the end segment", "reason": "a test"}}
