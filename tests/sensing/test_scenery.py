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
- ``forget-hide``: the saved form losing ``hide=``."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest
from in_loop_cases import planted
from scenery_cases import (BALL_DISPLAY, VESSEL_DISPLAY, VESSEL_MESH, assets, scenery_scene)

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

    def own_only(prim):
        values = [UsdGeom.PrimvarsAPI(prim).GetPrimvar(name) for name in ("displayColor", "displayOpacity")]
        found = [p.Get()[0] if p and p.HasAuthoredValue() else None for p in values]
        colour = None if found[0] is None else tuple(float(c) for c in found[0])
        return colour, None if found[1] is None else float(found[1])

    _scenery.display_primvars = own_only
    _compose.display_primvars = own_only
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
    # a mesh nobody binds is scenery whatever its asset: the ball's skin once its body is not bound
    assert sorted(m.path for m in composed().scenery([], 1.0)) == ["/World/Ball/Skin", VESSEL_MESH]


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
    colour, opacity = composed().display("/World/Ball/Shell")
    assert colour == pytest.approx(BALL_DISPLAY[0], abs=1e-6) and opacity == pytest.approx(BALL_DISPLAY[1])
    assert composed().display("/World/Nothing") == (None, None)


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
