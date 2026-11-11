"""The contract changes the in-loop camera needed, each reached and each shown load-bearing:

* ``Observation.output_shape`` with ``Output(shape_from_settings=True)``: the layout and the node's state give the image the
  detector's shape, rows by columns. **Control** (``CAMERA_PLANT=shape-ignored``): the entry's override ignored, the build
  refuses the camera (its image declared a scalar, held as 120 by 160).
* ``Plan.mounted_on`` and ``ObservationBuilder.mount``: the mounted frame's poses reach the node. **Control**
  (``CAMERA_PLANT=mount-unwired``): the builder's ``mount`` does nothing, and the build refuses the unwired input.
* ``PlanContext.roster``, ``Read(of=)`` and ``Observation.sees_scene``: one pose read per body of the roster, none for a
  structure; the registry refuses an observation that sees the scene and binds a role."""

from __future__ import annotations

import importlib

import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, ballistic, planted
from test_in_loop_mounts import moving_gantry

from mime.build import BuildRefused, ObservationBuilder
from mime.observations import Camera, Mounted
from mime.registry import EntryKind, Observation, RoleKind
from mime.registry._registry import _problems  # noqa: PLC2701
from mime.units import ms

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


def test_an_observation_that_sees_the_scene_binds_no_role():
    class Binding(Camera):
        roles = {"body": RoleKind.BODY}

    assert any("binds no role" in problem for problem in _problems(EntryKind.OBSERVATION, Binding))
    assert not any("binds no role" in problem for problem in _problems(EntryKind.OBSERVATION, Camera))
    assert Observation.sees_scene is False and Camera.sees_scene is True
