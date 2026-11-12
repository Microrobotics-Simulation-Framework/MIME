"""The contract changes the in-loop camera needed, each reached and each shown load-bearing:

* ``Observation.output_shape`` with ``Output(shape_from_settings=True)``: the layout and the node's state give the image the
  detector's shape, rows by columns. **Control** (``CAMERA_PLANT=shape-ignored``): the entry's override ignored, the build
  refuses the camera (its image declared a scalar, held as 120 by 160).
* ``Plan.mounted_on`` and ``ObservationBuilder.mount``: the mounted frame's poses reach the node. **Control**
  (``CAMERA_PLANT=mount-unwired``): the builder's ``mount`` does nothing, and the build refuses the unwired input.
  A plan that names a mount on anything but an observation is a defect, named (**control**: the check removed, the
  entry vanishes from the graph with no word, and the test fails).
* ``PlanContext.roster``, ``Read(of=)`` and ``Observation.sees_scene``: one pose read per body of the roster, none for a
  structure; the registry refuses an observation that sees the scene and binds a role."""

from __future__ import annotations

import dataclasses
import importlib

import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, ballistic, planted
from test_in_loop_mounts import moving_gantry

from mime.build import BuildRefused, ObservationBuilder
from mime.build._maddening import unreadable_leaves
from mime.study import Study, grid
from mime.observations import Camera, Mounted
from mime.registry import EntryKind, Observation, RoleKind
from mime.registry._registry import _problems  # noqa: PLC2701
from mime.units import ms, um

if planted("shape-ignored"):
    Camera.output_shape = lambda self, output: output.shape  # type: ignore[method-assign]
if planted("mount-unwired"):
    ObservationBuilder.mount = lambda self, into: None  # type: ignore[method-assign]


def test_the_image_has_the_detectors_shape_in_the_layout_and_in_the_node(dtype):
    built = ballistic(ms(3)).build()
    shapes = {o.name: o.shape for held in built.layout.observations for o in held.outputs}
    assert shapes["image"] == shapes["expected"] == (DETECTOR.height, DETECTOR.width)
    assert shapes["frame_step"] == shapes["age"] == shapes["new_frame"] == ()
    assert tuple(built.current_state()["camera:"]["image"].shape) == shapes["image"]


def test_a_mounted_camera_receives_its_frames_poses(dtype):
    exp = moving_gantry(ballistic(ms(3), mount=Mounted(frame="stage.gantry.carriage", orientation=LOOK_ACROSS)))
    built = exp.build()
    wired = [edge for edge in built.graph.edges if edge.target_node == "camera:" and edge.target_field == "mount_poses"]
    assert len(wired) == 1


def test_one_pose_is_read_for_each_body_of_the_roster_and_none_for_a_structure(dtype):
    built = ballistic(ms(3), balls=2).build()
    fields = sorted(edge.target_field for edge in built.graph.edges if edge.target_node == "camera:")
    assert fields == ["pose_ball0", "pose_ball1"]


def test_a_law_whose_plan_names_a_mount_is_refused_by_name(monkeypatch):
    from mime.testing.toys import ToySpring, coupled_toy

    original = ToySpring.plan
    monkeypatch.setattr(ToySpring, "plan", lambda self, ctx: dataclasses.replace(original(self, ctx), mounted_on="a.b.c"))
    with pytest.raises(BuildRefused) as raised:
        coupled_toy().build()
    assert raised.value.codes() == ["entry.defect"]
    assert "Plan.mounted_on" in raised.value.refusals[0].message


def test_an_observation_that_sees_the_scene_binds_no_role():
    class Binding(Camera):
        roles = {"body": RoleKind.BODY}

    assert any("binds no role" in problem for problem in _problems(EntryKind.OBSERVATION, Binding))
    assert not any("binds no role" in problem for problem in _problems(EntryKind.OBSERVATION, Camera))
    assert Observation.sees_scene is False and Camera.sees_scene is True


def test_the_scenes_numbers_are_held_as_built_an_override_is_refused_and_a_sweep_builds_each_value(dtype):
    """The camera consumes the scene's numbers (a tube's and a ball's sizes, the absorbers') when it is built, into its
    subjects and its response: they are one leaf its step does not read, so a run on the built step cannot take another
    value. An override is refused (``override.structural``) and a sweep builds each value. (The walk's control, a value the
    step reads overridden and planned as one group, cannot run through a camera: a batched group of it is refused,
    ``camera.batched``, which ``test_in_loop_refusals`` holds.)"""
    exp = ballistic(ms(3))
    built = exp.build()
    assert ("camera:", "scene_numbers") in unreadable_leaves(built.graph)
    with pytest.raises(BuildRefused) as raised:
        built.run(ms(2.0), override={"tube.inner_radius": um(61.0)})
    assert raised.value.codes() == ["override.structural"]
    swept = Study(exp).sweep(grid({"tube.inner_radius": [um(60.0), um(61.0)]}), duration=ms(2.0)).plan()
    assert [group.members for group in swept.groups] == [[0], [1]] and not swept.refused, swept.text()
