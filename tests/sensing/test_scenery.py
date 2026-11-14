"""The reader of the composed stage's scenery: the ``Mesh`` prims no thing owns, with the display primvars they inherit,
refused by name until the camera has a mesh tracer, hidden by prim path; and the stage's display primvars read for the
parts of a bound body that declares no appearance.

**Controls** (``CAMERA_PLANT``, each a planted error in the target; a test fails under its own, and every one is a code
patch with no draw, so a run either catches it or does not):

- ``own-primvar``: display primvars read from the prim itself, not inherited;
- ``own-mesh-scenery``: a mesh below a body's prim counted as scenery;
- ``derived-scenery``: a kind's derived mesh counted as scenery;
- ``drop-mesh``: a scenery mesh neither imaged nor refused (dropped silently);
- ``hide-names-only``: ``hide=`` taking names, not prim paths;
- ``no-defaulted-note``: the note of the subjects whose appearance was defaulted left out;
- ``display-over-declared``: the stage's display opacity overriding an appearance the user declared;
- ``forget-hide``: the saved form losing ``hide=``.

The fix round's tests (indexed and non-constant display primvars, every unowned gprim, a time-sampled mesh) were each run
against the head before it (``7149f420``) and fail there: it read the first value of a primvar's array whatever its indices
and interpolation, found only ``Mesh`` prims and read a time-sampled mesh at the default time, as no points."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest
from in_loop_cases import planted
from scenery_cases import (BALL_DISPLAY, VESSEL_DISPLAY, VESSEL_MESH, assets, scenery_scene, write_faces,
                           write_gprims)

from mime import Experiment
from mime.build import BuildRefused
from mime.observations import Camera
from mime.scene import Composition, asset, compose
from mime.scene import _compose, _scenery
from mime.scene._scene import Reference
from mime.sensing import _scene as scene
from mime.units import ms

PATH = re.compile(r"^/")

if planted("own-primvar"):
    from pxr import UsdGeom

    from mime.scene import DisplayRead

    def own_only(prim):
        values = [UsdGeom.PrimvarsAPI(prim).GetPrimvar(name) for name in ("displayColor", "displayOpacity")]
        found = [p.Get()[0] if p and p.HasAuthoredValue() else None for p in values]
        colour = None if found[0] is None else tuple(float(c) for c in found[0])
        return DisplayRead(colour, None if found[1] is None else float(found[1]))

    _scenery.read_display = own_only
    _compose.read_display = own_only
    _scenery.display_authors = lambda prim: (str(prim.GetPath()), str(prim.GetPath()))

if planted("own-mesh-scenery"):
    original_scenery = Composition.scenery
    Composition.scenery = lambda self, owned, mpu: original_scenery(self, (), mpu)  # type: ignore[method-assign]

if planted("derived-scenery"):
    _scenery.DERIVED_FROM = "mime:never-authored"

if planted("drop-mesh"):
    original_subjects = scene.subjects_of

    def without_meshes(roster, hide, absorbers):
        import dataclasses

        return original_subjects(dataclasses.replace(roster, scenery=()), hide, absorbers)

    scene.subjects_of = without_meshes

if planted("hide-names-only"):
    original_found = Camera._found  # noqa: SLF001

    def names_only(self, ctx):
        return scene.subjects_of(ctx.roster, tuple(n for n in self.hide if not PATH.match(n)), dict(self.absorbers))

    Camera._found = names_only  # noqa: SLF001
    assert original_found is not None

if planted("no-defaulted-note"):
    original_notes = Camera._notes  # noqa: SLF001
    Camera._notes = lambda self, *a: tuple(  # type: ignore[method-assign]  # noqa: SLF001
        line for line in original_notes(self, *a) if not line.startswith("appearance defaulted"))

if planted("display-over-declared"):
    original_of = scene.subject_of
    original_camera_found = Camera._found  # noqa: SLF001

    def display_first(spec):
        subject = original_of(spec)
        held = spec.get("display")
        if held is None:
            return subject
        import dataclasses

        from mime.sensing._subjects import Display, from_stage
        return dataclasses.replace(subject, appearance=from_stage(Display(
            tuple(held["colour"]) if held["colour"] is not None else None, held["opacity"])))

    def with_declared_display(self, ctx):
        found = original_camera_found(self, ctx)
        shown = {f"{n}" if len(b.parts) == 1 else f"{n}.{p.name}": p.display
                 for n, b in ctx.roster.bodies.items() for p in b.parts if p.display is not None}
        from mime.sensing._subjects import Display
        import dataclasses
        return dataclasses.replace(found, display={k: Display(*v) for k, v in shown.items()})

    scene.subject_of = display_first
    Camera._found = with_declared_display  # noqa: SLF001
    # the spec writer takes a declared subject's display too
    original_spec = scene.subject_spec

    def spec_with_display(subject, window, display=None):
        held = original_spec(subject, window, display)
        if display is not None and "display" not in held:
            held["display"] = {"colour": list(display.colour) if display.colour else None, "opacity": display.opacity}
        return held

    scene.subject_spec = spec_with_display

if planted("forget-hide"):
    original_save = Experiment.save

    def save_without_hide(self, folder, *args, **kwargs):
        lock = original_save(self, folder, *args, **kwargs)
        document = Path(folder) / "experiment.mime.yaml"
        document.write_text("\n".join(line for line in document.read_text(encoding="utf-8").splitlines()
                                      if "hide:" not in line) + "\n", encoding="utf-8")
        return lock

    Experiment.save = save_without_hide  # type: ignore[method-assign]


def refusals(experiment: Experiment):
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    return refused.value


def notes(experiment: Experiment) -> str:
    return experiment.build().report.text()


def composed() -> Composition:
    vessel, ball = assets()
    return compose([Reference(asset(vessel), "/World/Vessel"), Reference(asset(ball), "/World/Ball")], 1.0)


# -- what the reader finds -------------------------------------------------------------------------------------------

def test_the_scenery_is_the_meshes_no_thing_owns():
    """The ball's own skin (below the body's prim) and the vessel's derived proxy are not scenery; the wall is."""
    meshes = composed().scenery(["/World/Ball"], 1.0)
    assert [m.path for m in meshes] == [VESSEL_MESH]
    (wall,) = meshes
    assert wall.points.shape == (8, 3) and wall.triangles.shape == (12, 3)
    assert np.allclose(np.abs(wall.points), 1e-3)
    # a prim nobody binds is scenery whatever its asset: the ball's shell and skin once its body is not bound
    assert sorted(m.path for m in composed().scenery([], 1.0)) == ["/World/Ball/Shell", "/World/Ball/Skin", VESSEL_MESH]


def test_the_scenery_is_read_in_metres_in_the_world():
    stage_wall = composed().scenery(["/World/Ball"], 0.001)[0]
    assert np.allclose(np.abs(stage_wall.points), 1e-6)


def test_a_mesh_inherits_its_display_primvars_from_its_visual_prim():
    (wall,) = composed().scenery(["/World/Ball"], 1.0)
    assert wall.colour == pytest.approx(VESSEL_DISPLAY[0], abs=1e-6) and wall.opacity == pytest.approx(VESSEL_DISPLAY[1])
    assert (wall.colour_from, wall.opacity_from) == ("/World/Vessel", "/World/Vessel")


def test_a_mesh_whose_path_authors_no_display_has_none():
    vessel, ball = assets(vessel=None)
    stage = compose([Reference(asset(vessel), "/World/Vessel")], 1.0)
    (wall,) = stage.scenery([], 1.0)
    assert (wall.colour, wall.opacity, wall.colour_from, wall.opacity_from) == (None, None, None, None)


def test_a_body_part_takes_its_stage_display_from_its_visual_prim():
    shown = composed().display("/World/Ball/Shell")
    assert shown.colour == pytest.approx(BALL_DISPLAY[0], abs=1e-6) and shown.opacity == pytest.approx(BALL_DISPLAY[1])
    assert shown.reduced == () and composed().display("/World/Nothing") == (None, None, ())


# -- the camera: refusal, hide, notes -----------------------------------------------------------------------------

def test_a_scenery_mesh_is_refused_by_name_never_left_unimaged():
    error = refusals(scenery_scene())
    assert error.codes() == ["camera.mesh-untraced"]
    (refusal,) = error.refusals
    assert VESSEL_MESH in refusal.message and "12 triangles" in refusal.message
    assert f"decimation distance not recorded for: {VESSEL_MESH}" in refusal.message
    assert f"hide=({VESSEL_MESH!r},)" in refusal.action


def test_a_scenery_mesh_is_hidden_by_prim_path_and_the_report_says_so():
    text = notes(scenery_scene(hide=(VESSEL_MESH,)))
    assert f"hidden: {VESSEL_MESH}" in text
    assert "route 1 (closed form) for every subject: ball (sphere)" in text


def test_hiding_a_path_that_is_no_scenery_is_refused_listing_the_paths():
    error = refusals(scenery_scene(hide=("/World/Vessel/Nothing",)))
    assert "camera.hide" in error.codes()
    (refusal,) = [r for r in error.refusals if r.code == "camera.hide"]
    assert VESSEL_MESH in refusal.action


def test_without_scenery_nothing_is_refused_or_noted():
    text = notes(scenery_scene(vessel=False))
    assert "validity of camera: (computed at build): hidden:" not in text and "camera.mesh" not in text


def test_a_body_part_whose_stage_authors_a_display_takes_it_and_the_note_says_so():
    text = notes(scenery_scene(hide=(VESSEL_MESH,)))
    assert "appearance from the stage's display primvars for: ball" in text
    assert "appearance defaulted" not in text


def test_the_subjects_with_no_appearance_anywhere_are_defaulted_and_noted():
    text = notes(scenery_scene(hide=(VESSEL_MESH,), ball_display=None))
    assert "appearance defaulted for: ball" in text
    assert "display primvars for" not in text


# -- the stage's display does not touch what the user declared -----------------------------------------------------

def frame_of(experiment: Experiment) -> np.ndarray:
    built = experiment.build()
    built.run(ms(3))
    return np.asarray(built.current_state()["camera:"]["image"])


def test_a_declared_appearance_is_bit_for_bit_the_same_with_or_without_a_stage_display(dtype):
    declared = (("ball", 3000.0),)
    with_display = scenery_scene(hide=(VESSEL_MESH,), absorbers=declared)
    without = scenery_scene(hide=(VESSEL_MESH,), absorbers=declared, ball_display=None)
    assert np.array_equal(frame_of(with_display), frame_of(without))
    assert "appearance from the stage" not in notes(with_display)


def test_an_undeclared_appearance_takes_the_stage_display_and_the_frame_changes(dtype):
    shown = frame_of(scenery_scene(hide=(VESSEL_MESH,)))
    plain = frame_of(scenery_scene(hide=(VESSEL_MESH,), ball_display=None))
    assert not np.array_equal(shown, plain)
    # half-transparent: the ball blocks half the backlight where it covers a pixel, an opaque one all of it
    assert shown.min() > plain.min() + 100.0


def test_the_spec_a_scene_without_stage_display_writes_has_no_display_key():
    exp = scenery_scene(hide=(VESSEL_MESH,), ball_display=None)
    def keys(experiment):
        node = experiment.build().graph.get_node("camera:")
        return {key for each in node._spec()["subjects"] for key in each}  # noqa: SLF001

    assert "display" not in keys(exp)
    assert "display" in keys(scenery_scene(hide=(VESSEL_MESH,)))


# -- the saved form -------------------------------------------------------------------------------------------------

def test_hide_by_prim_path_is_saved_and_loaded_with_no_new_field(tmp_path: Path):
    exp = scenery_scene(hide=(VESSEL_MESH,))
    exp.save(tmp_path / "scenery")
    text = (tmp_path / "scenery" / "experiment.mime.yaml").read_text(encoding="utf-8")
    assert f"hide: [{VESSEL_MESH}]" in text
    loaded = Experiment.load(tmp_path / "scenery")
    assert f"hidden: {VESSEL_MESH}" in loaded.build().report.text()


# -- display primvars as USD composes them -------------------------------------------------------------------------

def faces(tmp_path: Path, **options):
    """The scenery of ``write_faces``'s mesh (two triangles, areas 1 and 3 mm^2), read from its own layer."""
    file = write_faces(tmp_path / "faces.usda", **options)
    (mesh,) = compose([Reference(asset(file), "/World/Faces")], 1.0).scenery([], 1.0)
    return mesh


def test_an_indexed_constant_primvar_is_read_through_its_indices(tmp_path: Path):
    """A constant displayOpacity [0.9, 0.1] with indices [1] is 0.1; the first element of the array is 0.9."""
    mesh = faces(tmp_path, opacities=[0.9, 0.1], indices=[1], interpolation="constant")
    assert mesh.opacity == pytest.approx(0.1) and mesh.reduced == ()


def test_an_indexed_uniform_primvar_is_read_through_its_indices(tmp_path: Path):
    """A per-face opacity [0.9, 0.1, 0.5] with indices [1, 1] is 0.1 on both faces, so its mean is 0.1."""
    mesh = faces(tmp_path, opacities=[0.9, 0.1, 0.5], indices=[1, 1])
    assert mesh.opacity == pytest.approx(0.1)
    assert mesh.reduced == (f"displayOpacity per-face (uniform) on /World/Faces/Mesh reduced to its area-weighted mean",)


def test_a_per_face_primvar_is_reduced_to_its_area_weighted_mean_and_said(tmp_path: Path):
    """Faces of areas 1 and 3 coloured red and blue: the mean is a quarter red and three quarters blue."""
    mesh = faces(tmp_path, colours=[(1, 0, 0), (0, 0, 1)])
    assert mesh.colour == pytest.approx((0.25, 0.0, 0.75))
    assert mesh.reduced == ("displayColor per-face (uniform) on /World/Faces/Mesh reduced to its area-weighted mean",)


def test_a_per_vertex_primvar_is_weighted_by_the_area_each_vertex_stands_for(tmp_path: Path):
    """Each of the two triangles gives a third of its area to each of its corners; the vertices are not shared here, so
    the mean is the same as the per-face one over the corners (a quarter of the first triangle's three, three quarters of
    the second's)."""
    first, second = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    mesh = faces(tmp_path, colours=[first] * 3 + [second] * 3, interpolation="vertex")
    assert mesh.colour == pytest.approx((0.25, 0.0, 0.75))
    assert "per-vertex" in mesh.reduced[0]


def test_a_reduced_primvar_is_in_the_build_note_naming_the_prim_and_the_interpolation():
    shell = (("displayOpacity", "vertex", [0.2, 0.6], None),)
    text = notes(scenery_scene(hide=(VESSEL_MESH,), shell=shell))
    assert ("appearance: displayOpacity per-vertex on /World/Ball/Shell reduced to its mean for: ball") in text


def test_an_indexed_opacity_on_a_bound_part_is_read_through_its_indices_in_the_frame(dtype):
    """The same part, an opacity [0.9, 0.1] indexed to its second element and one authored 0.1: the same frame."""
    indexed = (("displayOpacity", "constant", [0.9, 0.1], [1]),)
    plain = (("displayOpacity", "constant", [0.1], None),)
    assert np.array_equal(frame_of(scenery_scene(hide=(VESSEL_MESH,), shell=indexed)),
                          frame_of(scenery_scene(hide=(VESSEL_MESH,), shell=plain)))


# -- every unowned gprim is scenery that is read or refused by name ------------------------------------------------

EXTRAS = "/World/Extras"


def extras(tmp_path: Path):
    file = write_gprims(tmp_path / "extras.usda")
    return compose([Reference(asset(file), EXTRAS)], 1.0).scenery([], 1.0)


def test_every_unowned_gprim_is_found_by_type(tmp_path: Path):
    found = {held.path.rsplit("/", 1)[-1]: held.type_name for held in extras(tmp_path)}
    assert found == {"Animated": "Mesh", "Capsule": "Capsule", "Cube": "Cube", "Curves": "BasisCurves"}


@pytest.mark.parametrize("name,type_name", [("Cube", "Cube"), ("Capsule", "Capsule"), ("Curves", "BasisCurves")])
def test_an_unowned_gprim_that_is_no_mesh_is_refused_by_name_and_hides_by_path(tmp_path: Path, name: str, type_name: str):
    file = write_gprims(tmp_path / "extras.usda")
    path = f"{EXTRAS}/{name}"
    others = tuple(f"{EXTRAS}/{n}" for n in ("Cube", "Capsule", "Curves", "Animated") if n != name)
    error = refusals(scenery_scene(vessel=False, extras=file, hide=others))
    assert error.codes() == ["camera.mesh-untraced"]
    (refusal,) = error.refusals
    assert path in refusal.message and f"holds this {type_name}" in refusal.message
    assert f"hide=({path!r},)" in refusal.action
    notes(scenery_scene(vessel=False, extras=file, hide=(*others, path)))


def test_a_mesh_with_only_time_sampled_points_is_read_at_its_earliest_sample(tmp_path: Path):
    (animated,) = [held for held in extras(tmp_path) if held.path.endswith("Animated")]
    assert animated.points.shape == (3, 3) and animated.triangles.shape == (1, 3) and animated.animated
    assert np.allclose(animated.points[1], (1e-3, 0.0, 0.0))
    assert not [held for held in composed().scenery(["/World/Ball"], 1.0) if getattr(held, "animated", False)]
