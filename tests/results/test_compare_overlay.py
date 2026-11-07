"""``compare.usda``: the ghosted comparison of a study's members, composed with usd-core. Its time samples are each
member's recorded poses at each time code, its ``timeCodesPerSecond`` is samples per simulated second, each member sits
under its own prim ``/Compare/<member>`` with a ``visibility`` attribute and metadata naming the member, its coordinates
and whether it is the reference, one sublayer that can be muted whole, and everything it references lies inside the study
folder. The stage samples are what the members recorded: a ``stage_every`` that is no whole number of a member's base step
is refused with the nearest valid one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import FREQUENCIES, STAGE_EVERY
from pxr import Sdf, Usd, UsdGeom
from script_a import reproduction_arm

from mime import Study
from mime.build._refusals import ArgumentRefused
from mime.results import StudyResult
from mime.study import grid
from mime.units import Hz, ms, s


@pytest.fixture(scope="module")
def overlay(study_folder: Path) -> tuple[Path, StudyResult]:
    result = StudyResult.read(study_folder)
    return result.write_compare(0), result


def test_its_samples_are_each_members_recorded_poses_at_each_time_code(overlay):
    path, result = overlay
    layer = Sdf.Layer.FindOrOpen(str(path))
    assert layer.timeCodesPerSecond == pytest.approx(1.0 / STAGE_EVERY) == layer.framesPerSecond
    for index in range(len(result)):
        record = result.member(index).stage
        rows = record.rows(record.entries()[0])
        base = "/World/umr" if index == 0 else f"/Compare/m{index}/umr"
        codes = layer.ListTimeSamplesForPath(Sdf.Path(f"{base}.xformOp:translate"))
        assert list(codes) == list(range(len(rows))) == list(range(11))
        for code in codes:
            assert tuple(layer.QueryTimeSample(Sdf.Path(f"{base}.xformOp:translate"), code)) == tuple(rows[int(code), :3])
            orient = layer.QueryTimeSample(Sdf.Path(f"{base}.xformOp:orient"), code)
            assert (orient.GetReal(), *orient.GetImaginary()) == tuple(rows[int(code), 3:7])


def test_control_a_members_samples_shifted_by_one_time_code_differ_from_its_recorded_poses(overlay):
    path, result = overlay
    layer = Sdf.Layer.FindOrOpen(str(path))
    detected = trials = 0
    for index in range(1, len(result)):
        rows = result.member(index).stage.rows(result.member(index).stage.entries()[0])
        for field, take in (("translate", slice(0, 3)), ("orient", slice(3, 7))):
            shifted = [layer.QueryTimeSample(Sdf.Path(f"/Compare/m{index}/umr.xformOp:{field}"), code + 1) for code in range(10)]
            flat = [tuple(v) if field == "translate" else (v.GetReal(), *v.GetImaginary()) for v in shifted]
            trials += 1
            detected += any(tuple(rows[code, take]) != flat[code] for code in range(10))
    assert detected == trials == 4


def test_each_member_is_a_prim_of_its_own_with_metadata_and_a_visibility_that_hides_only_it(overlay):
    path, result = overlay
    stage = Usd.Stage.Open(str(path))
    for index, frequency in enumerate(FREQUENCIES):
        prim = stage.GetPrimAtPath(f"/Compare/m{index}")
        held = prim.GetCustomData()["mime_compare"]
        assert (held["name"], held["member"], held["reference"]) == (f"m{index}", index, index == 0)
        axis = held["coordinates"]["axis0"]
        assert (axis["path"], axis["unit"], axis["value"]) == ("field.commands.frequency", "Hz", frequency)
        assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == "inherited"
    layer_data = Sdf.Layer.FindOrOpen(str(path)).customLayerData["mime:compare"]
    assert list(layer_data["members"]) == ["m0", "m1", "m2"] and layer_data["reference"] == "m0"
    UsdGeom.Imageable(stage.GetPrimAtPath("/Compare/m1")).GetVisibilityAttr().Set("invisible")
    seen = {m: UsdGeom.Imageable(stage.GetPrimAtPath(f"/Compare/{m}/umr")).ComputeVisibility(0) for m in ("m1", "m2")}
    assert seen == {"m1": "invisible", "m2": "inherited"}
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/umr")).ComputeVisibility(0) == "inherited"   # the reference stays
    assert not stage.GetPrimAtPath("/Compare/m0/umr")                                                 # its things are /World's


def test_it_is_one_sublayer_that_mutes_whole_and_references_only_what_the_study_folder_holds(overlay):
    path, result = overlay
    layer = Sdf.Layer.FindOrOpen(str(path))
    assert list(layer.subLayerPaths) == ["experiment/scene/scene.usda"]
    assert (path.parent / layer.subLayerPaths[0]).resolve().is_relative_to(path.parent.resolve())
    for prim in ("/Compare/m1/umr", "/Compare/m2/umr"):
        references = layer.GetPrimAtPath(prim).referenceList.prependedItems
        assert [(r.assetPath, str(r.primPath)) for r in references] == [("", "/World/umr")]
    root = Usd.Stage.Open(Sdf.Layer.CreateAnonymous(".usda"))
    root.GetRootLayer().subLayerPaths.append(str(path))
    assert root.GetPrimAtPath("/Compare/m1")
    root.MuteLayer(str(path))
    assert not root.GetPrimAtPath("/Compare")


def test_the_overlay_is_not_the_scene_writers_and_the_study_layout_holds_it(overlay):
    from mime.results._study_folder import layout_problems

    path, result = overlay
    assert layout_problems(result.folder) == []
    assert path.name == "compare.usda"


def test_a_study_in_memory_has_no_folder_for_the_overlay_to_lie_in():
    from rigs_plot import run_of, study_of

    with pytest.raises(ArgumentRefused) as error:
        study_of([run_of([1.0, 2.0], 0.001), run_of([1.0, 2.0], 0.001)], [1.0, 2.0]).write_compare()
    assert error.value.refusals[0].code == "compare.folder"


def test_a_stage_every_that_is_no_whole_number_of_a_members_dt_is_refused_with_the_nearest_valid_one():
    exp = reproduction_arm()
    field = exp.handle("field")
    refused = 0
    for bad, nearest in ((0.0007, 0.0005), (0.0052, 0.005), (0.0003, 0.0005)):
        sweep = Study(exp).sweep(grid({field.commands.frequency: [Hz(10.0)]}), duration=s(0.05), stage_every=s(bad))  # type: ignore[union-attr]
        with pytest.raises(ArgumentRefused) as error:
            sweep.plan()
        refusal = error.value.refusals[0]
        refused += refusal.code == "study.stage-every" and f"stage_every=s({nearest!r})" in refusal.action
    assert refused == 3
    Study(exp).sweep(grid({field.commands.frequency: [Hz(10.0)]}), duration=s(0.05), stage_every=ms(5)).plan()  # type: ignore[union-attr]


def test_a_study_across_a_dt_axis_takes_the_step_that_is_a_whole_number_of_every_dt():
    from mime.study._stage import check_stage_every, common_step

    assert common_step([0.0005, 0.00025]) == pytest.approx(0.0005)
    assert common_step([0.0005, 0.0004]) == pytest.approx(0.002)
    with pytest.raises(ArgumentRefused) as error:
        check_stage_every(s(0.0005), [0.0005, 0.0004])
    assert "stage_every=s(0.002)" in error.value.refusals[0].action
    check_stage_every(s(0.002), [0.0005, 0.0004])
    assert np.isclose(0.002 / 0.0004, 5)


def test_a_member_that_kept_no_stage_samples_names_the_route():
    from rigs_plot import run_of, study_of

    from mime.viz._compare import compare_layer

    with pytest.raises(ArgumentRefused) as error:
        compare_layer(study_of([run_of([1.0, 2.0], 0.001), run_of([1.0, 2.0], 0.001)], [1.0, 2.0]), 0)
    refusal = error.value.refusals[0]
    assert refusal.code == "compare.stage" and "stage_every" in refusal.action


def test_the_stage_samples_are_in_the_group_file_and_the_manifest_records_the_spacing(study_folder):
    import json
    import zipfile

    document = json.loads((study_folder / "study.json").read_text())
    assert document["run"]["stage_every"]["value"] == STAGE_EVERY
    names = zipfile.ZipFile(study_folder / "groups/0/outputs.npz").namelist()
    assert {"m0/stage/steps.npy", "m0/stage/meta.npy", "m0/stage/fields/umr/pose.npy"} <= set(names)
    assert "stage_every: " in (study_folder / "study.mime.yaml").read_text()


def test_a_member_drawn_with_a_borrowed_geometry_says_so_where_a_viewer_reads_it_and_warns_at_write_time(tmp_path):
    import warnings

    from mime.study import DT

    out = tmp_path / "study"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Study(reproduction_arm()).sweep(grid({DT: [ms(0.5), ms(1.0)]}), duration=s(0.01), every=ms(2.0), stage_every=ms(2.0),
                                        record=["swim_speed:umr.speed"]).run_to(out)
        result = StudyResult.read(out)
        assert len(set(result.groups)) == 2          # the two steps are two compiled programs: two groups
        caught.clear()
        path = result.write_compare(0)
    assert [str(w.message) for w in caught if "geometry" in str(w.message)] != []
    layer = Sdf.Layer.FindOrOpen(str(path))
    borrowed = layer.GetPrimAtPath("/Compare/m1").customData["mime_compare"]
    reference = layer.GetPrimAtPath("/Compare/m0").customData["mime_compare"]
    assert borrowed["own_build"] is False and borrowed["geometry_from"] == "m0"
    assert reference["own_build"] is True and reference["geometry_from"] == "m0"


def test_a_member_of_the_references_group_has_its_own_build_and_the_write_does_not_warn(overlay):
    import warnings

    path, result = overlay
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result.write_compare(0)                       # control: one group, nothing borrowed, nothing warned
    layer = Sdf.Layer.FindOrOpen(str(path))
    assert all(layer.GetPrimAtPath(f"/Compare/m{i}").customData["mime_compare"]["own_build"] is True for i in range(3))
