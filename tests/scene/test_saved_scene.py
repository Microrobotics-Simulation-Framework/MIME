"""The scene in the saved form: the round trip keeps the scene's settings, every thing's prim path beside its name
(scene paths are experiment names), and the asset's pin; the folder holds the asset's bytes beside the stage, and a
load refuses an asset file or a stage changed after the lock was written. The control for each refusal: the same
folder, unchanged, loads."""

import pytest
from robot_asset import bound_robot
from script_a import script_a

from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LoadRefused
from mime.scene import STAGE_FILE, Scene, Z
from mime.units import ms, standard_gravity


def test_the_round_trip_keeps_the_scene_the_prim_paths_and_the_assets_pin(tmp_path):
    built = bound_robot().build()
    lock = built.save(tmp_path)
    loaded = Experiment.load(tmp_path)
    document = loaded.document()
    assert document == built.record.document
    reference = document["scene"]["references"][0]
    assert (reference["at"], reference["layer"]) == ("/World/UMR", "scene/layers/fl9_robot.usda")
    assert reference["sha256"] == lock["stage"]["layers"]["scene/layers/fl9_robot.usda"]
    assert {name: thing["prim"] for name, thing in document["things"].items()} == {
        "UMR": "/World/UMR", "coils": "/World/coils", "tube": "/World/tube", "water": "/World/water"}
    rebuilt = loaded.build()
    assert rebuilt.record.resolved["scene"] == lock["resolved"]["scene"]
    assert rebuilt.report.stage_hash == lock["stage"]["sha256"]
    assert loaded.load_report is not None and "reproduced exactly" in loaded.load_report.text()


def test_a_python_first_scene_round_trips_with_its_up_and_units(tmp_path):
    experiment = script_a()
    experiment._settings = type(experiment._settings)(experiment.name, Scene.new(up=(0.0, 0.6, 0.8), units="mm"),
                                                      standard_gravity(), ms(0.5), 0)
    built = experiment.build()
    built.save(tmp_path)
    document = Experiment.load(tmp_path).document()
    assert document["scene"] == {"up": [0.0, 0.6, 0.8], "units": "mm", "references": []}
    assert document["things"]["umr"]["prim"] == "/World/umr"


def test_a_changed_asset_file_is_refused_at_load(tmp_path):
    bound_robot().build().save(tmp_path)
    Experiment.load(tmp_path)
    layer = tmp_path / "scene" / "layers" / "fl9_robot.usda"
    layer.write_text(layer.read_text().replace("(1, 0, 0)", "(0, 0, 1)"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(tmp_path)
    print(refused.value)
    assert "load.asset-changed" in refused.value.codes()


def test_a_stage_edited_after_its_lock_is_refused_at_load(tmp_path):
    script_a().build().save(tmp_path)
    Experiment.load(tmp_path)
    stage = tmp_path / STAGE_FILE
    stage.write_text(stage.read_text().replace('upAxis = "Z"', 'upAxis = "Y"'))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(tmp_path)
    assert refused.value.codes() == ["load.stage-changed"]


def test_a_document_read_alone_reads_its_assets_from_beside_it(tmp_path):
    bound_robot().build().save(tmp_path)
    loaded = Experiment.from_document(tmp_path / DOCUMENT_FILE)
    (magnet,) = loaded.plan().scene.magnets
    assert magnet.source == "the MimeMagnetAPI schema on /World/UMR/Magnets"


def test_a_document_whose_python_first_thing_moved_prim_is_refused(tmp_path):
    script_a().build().save(tmp_path)
    document = tmp_path / DOCUMENT_FILE
    document.write_text(document.read_text().replace("prim: /World/tube", "prim: /World/pipe"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.from_document(document)
    assert refused.value.codes() == ["load.document-thing"]


def test_the_scene_up_is_declared_once_in_the_document():
    document = Experiment("frame", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(),
                          dt=ms(1)).document()
    assert "frame" not in document and document["scene"]["up"] == [0.0, 0.0, 1.0]
