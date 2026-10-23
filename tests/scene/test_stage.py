"""The stage the build writes, read back with pxr: the declared prims, up axis, metersPerUnit and defaultPrim; the
tube's settings as ``mime:`` attributes; the body at its initial pose; usd-core's validators finding no error; the
same declarations writing the same bytes, whose hash the lock records.

The controls: a stage written without ``defaultPrim``, by a planted switch, fails the read-back check and the
validators both; a changed declaration changes the stage's bytes and hash."""

import hashlib
import math
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom
from script_a import script_a

from mime import Body, Experiment, Scene, State
from mime.dynamics import RigidInertial
from mime.scene import STAGE_FILE, Tube, X, Y, Z, check_stage
from mime.scene import _testing as stage_switches
from mime.things import Structure
from mime.units import kg, kg_m2, m, m_per_s, mm, ms, rad_per_s, standard_gravity

#: USD stores points as float32: a point read back is within this of what was written, relative to its size.
FLOAT32 = 1e-7

PRIMS = {"/World/tube": "structure", "/World/water": "medium", "/World/umr": "body", "/World/field": "apparatus"}


def tube_and_body(scene: Scene) -> Experiment:
    """The smallest experiment the stage writer needs: a tube, and a body so that the graph has a node."""
    experiment = Experiment("stage", scene=scene, gravity=standard_gravity(), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3), wall=mm(1), length=mm(9))))
    experiment.add_body(Body("probe", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("probe.mass", kg(1e-4), reason="the stage tests' body")
    experiment.declare("probe.inertia", kg_m2((1e-10,) * 3), reason="the stage tests' body")
    return experiment


def saved(experiment: Experiment, folder: Path) -> Path:
    experiment.build().save(folder)
    return folder / STAGE_FILE


def stage_problems(path: Path, prims: dict[str, str], up: str = "Z", meters_per_unit: float = 1.0) -> list[str]:
    """Every way the stage at ``path`` read back with pxr differs from what was declared: its prims with their
    kinds, its up axis, metersPerUnit and defaultPrim."""
    stage = Usd.Stage.Open(str(path))
    found: list[str] = []
    if UsdGeom.GetStageUpAxis(stage) != up:
        found.append(f"upAxis {UsdGeom.GetStageUpAxis(stage)}, not {up}")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), meters_per_unit, rel_tol=1e-15):
        found.append(f"metersPerUnit {UsdGeom.GetStageMetersPerUnit(stage)}, not {meters_per_unit}")
    if not stage.HasDefaultPrim() or stage.GetDefaultPrim().GetPath() != "/World":
        found.append("defaultPrim is not /World")
    for path_, kind in prims.items():
        prim = stage.GetPrimAtPath(path_)
        if not prim.IsValid():
            found.append(f"no prim {path_}")
        elif prim.GetAttribute("mime:thing").Get() != kind:
            found.append(f"{path_} is a {prim.GetAttribute('mime:thing').Get()}, not a {kind}")
    return found


def test_the_stage_read_back_has_the_declared_prims_up_axis_units_and_default_prim(tmp_path):
    assert stage_problems(saved(script_a(), tmp_path), PRIMS) == []


def test_a_stage_written_without_default_prim_fails_the_read_back(tmp_path):
    with stage_switches.without_default_prim():
        path = saved(script_a(), tmp_path)
    assert stage_problems(path, PRIMS) == ["defaultPrim is not /World"]


def test_usd_cores_validators_find_no_error_and_the_planted_stage_an_error(tmp_path):
    findings = check_stage(saved(script_a(), tmp_path / "written"))
    print("\n".join(f"{f.severity} {f.validator}: {f.message}" for f in findings) or "no findings")
    assert [f for f in findings if f.severity == "error"] == []
    with stage_switches.without_default_prim():
        planted = check_stage(saved(script_a(), tmp_path / "planted"))
    assert [(f.severity, f.validator) for f in planted] == [("error", "usdValidation:StageMetadataChecker")]


def test_the_tube_is_its_wall_with_its_settings_on_the_prim_in_si(tmp_path):
    stage = Usd.Stage.Open(str(saved(script_a(), tmp_path)))
    tube = stage.GetPrimAtPath("/World/tube")
    assert tube.GetTypeName() == "Mesh"
    settings = {name: tube.GetAttribute(f"mime:geometry:{name}").Get() for name in
                ("inner_radius", "wall", "length")}
    assert settings == {"inner_radius": 0.003175, "wall": 0.0016, "length": 0.1}
    assert tuple(tube.GetAttribute("mime:geometry:axis").Get()) == (1.0, 0.0, 0.0)
    assert tube.GetAttribute("mime:geometry:inner_radius").GetCustomDataByKey("units") == "m"
    assert (tube.GetAttribute("mime:geometry").Get(), tube.GetAttribute("mime:geometry:version").Get()) == (
        "tube", "0.1.0")
    mesh = UsdGeom.Mesh(tube)
    points = mesh.GetPointsAttr().Get()
    for point in points:
        radius = math.hypot(point[0], point[1])
        assert min(abs(radius - 0.003175), abs(radius - 0.004775)) <= FLOAT32 * 0.004775
        assert abs(abs(point[2]) - 0.05) <= FLOAT32 * 0.05
    world = UsdGeom.Xformable(tube).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    axis = world.TransformDir((0.0, 0.0, 1.0))
    assert max(abs(a - b) for a, b in zip(axis, (1.0, 0.0, 0.0), strict=True)) < 1e-15


def test_the_body_stands_at_its_initial_pose(tmp_path):
    stage = Usd.Stage.Open(str(saved(script_a(), tmp_path)))
    body = UsdGeom.Xformable(stage.GetPrimAtPath("/World/umr"))
    ops = {op.GetOpName(): (op.GetPrecision(), op.Get()) for op in body.GetOrderedXformOps()}
    assert ops["xformOp:translate"] == (UsdGeom.XformOp.PrecisionDouble, (0.01, 0.0, -0.001))
    assert ops["xformOp:orient"][0] == UsdGeom.XformOp.PrecisionDouble
    assert stage.GetPrimAtPath("/World/umr/extent").GetAttribute("radius").Get() == pytest.approx(1.33 * 1.56e-3)


def test_a_micro_scale_mesh_far_from_the_origin_keeps_its_points_local(tmp_path):
    """USD's points are float32: a 30 um tube 1 km from the origin keeps its points within its own size, and its
    place is a double-precision translate. Script A's body side: the drag surrogate refuses a tube that tight."""
    experiment = script_a(drag=False)
    experiment._declarations[0] = type(experiment._declarations[0])(Structure("tube", geometry=Tube(
        axis=+X, inner_radius=mm(0.03), wall=mm(0.01), length=mm(0.5), centre=m((1000.0, 0.0, 0.0)))))
    stage = Usd.Stage.Open(str(saved(experiment, tmp_path)))
    tube = stage.GetPrimAtPath("/World/tube")
    assert max(abs(c) for p in UsdGeom.Mesh(tube).GetPointsAttr().Get() for c in p) <= 0.25e-3 * (1 + FLOAT32)
    translate = tube.GetAttribute("xformOp:translate")
    assert (translate.GetTypeName().type.typeName, tuple(translate.Get())) == ("GfVec3d", (1000.0, 0.0, 0.0))


@pytest.mark.parametrize(("up", "axis"), [(Z, "Z"), (Y, "Y"), ((0.0, 2.0, 0.0), "Y")])
def test_up_is_the_stages_up_axis_and_world_carries_no_rotation(tmp_path, up, axis):
    path = saved(tube_and_body(Scene.new(up=up, units="m")), tmp_path)
    assert stage_problems(path, {"/World/tube": "structure"}, up=axis) == []
    stage = Usd.Stage.Open(str(path))
    world = stage.GetPrimAtPath("/World")
    assert UsdGeom.Xformable(world).GetOrderedXformOps() == []
    assert UsdGeom.Xformable(world).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) == Gf.Matrix4d(1.0)
    assert not world.HasAttribute("mime:up")


@pytest.mark.parametrize("up", [X, -Z, -Y, (0.0, 0.6, 0.8), (1e-9, 0.0, 1.0)])
def test_an_up_other_than_plus_y_or_plus_z_is_refused_at_declaration(up):
    """The control for the test above: the directions phase 6 wrote with a turned ``/World`` are refused before any
    build, with the message naming the two ups and how to re-express the scene."""
    with pytest.raises(ValueError, match=r"neither \+Y nor \+Z") as refused:
        Scene.new(up=up, units="m")
    message = str(refused.value)
    for words in ("up is +Y or +Z only", "a choice of coordinates", "swapping two axes", "a declared frame transform"):
        assert words in message, (words, message)


def test_the_stage_is_written_in_the_scenes_units_and_its_settings_stay_si(tmp_path):
    path = saved(tube_and_body(Scene.new(up=Z, units="mm")), tmp_path)
    assert stage_problems(path, {"/World/tube": "structure"}, meters_per_unit=0.001) == []
    stage = Usd.Stage.Open(str(path))
    tube = stage.GetPrimAtPath("/World/tube")
    assert max(abs(p[0]) for p in UsdGeom.Mesh(tube).GetPointsAttr().Get()) == pytest.approx(4.0, rel=FLOAT32)
    assert not stage.GetPrimAtPath("/World/probe/extent").IsValid()  # no extent is declared, so no sphere
    assert tube.GetAttribute("mime:geometry:inner_radius").Get() == 0.003


def test_two_builds_write_the_same_bytes_and_the_lock_holds_their_hash(tmp_path):
    first, second = script_a().build(), script_a().build()
    assert first.stage_text.encode() == second.stage_text.encode()
    lock = first.save(tmp_path / "first")
    second.save(tmp_path / "second")
    written = (tmp_path / "first" / STAGE_FILE).read_bytes()
    assert written == (tmp_path / "second" / STAGE_FILE).read_bytes()
    assert lock["stage"]["sha256"] == hashlib.sha256(written).hexdigest() == first.report.stage_hash
    assert f"stage: {lock['stage']['sha256']}" in first.report.text()


def test_a_changed_declaration_changes_the_stage_and_its_hash():
    experiment = script_a()
    experiment._declarations[0] = type(experiment._declarations[0])(Structure("tube", geometry=Tube(
        axis=+X, inner_radius=mm(3.175), wall=mm(1.6), length=mm(101))))
    changed = experiment.build()
    assert changed.report.stage_hash != script_a().build().report.stage_hash
    assert "mime:geometry:length = 0.101" in changed.stage_text
