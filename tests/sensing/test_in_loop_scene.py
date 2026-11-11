"""The whole scene: the camera is told no body. Every body of the experiment appears in its frame; a body named in
``hide`` is absent and the build report says so; a body added to the experiment appears; a name ``hide`` or ``absorbers``
holds that is no subject is refused.

**The control** (``CAMERA_PLANT=drop-body``) drops the first body from the plan's wiring: it is then absent from the
frame, and every case that expects it fails."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, ballistic, planted

from mime.build import BuildRefused
from mime.observations import Camera
from mime.sensing import CameraPose, project

if planted("drop-body"):
    original = Camera._found  # noqa: SLF001

    def dropped(self, ctx):
        found = original(self, ctx)
        name = sorted(found.bodies)[0]
        return dataclasses.replace(found, subjects=tuple(s for s in found.subjects if s.name != name),
                                   bodies={k: v for k, v in found.bodies.items() if k != name})

    Camera._found = dropped  # noqa: SLF001


def first_frame(exp):
    """The frame after the first step and the centre of each body in it, in pixels, with the backlight's counts."""
    built = exp.build()
    built.run(__import__("mime.units", fromlist=["ms"]).ms(1))
    state = built.current_state()
    camera = DETECTOR.pinhole()
    pose = CameraPose(np.array([0.0, -1e-3, 0.0]), np.array(LOOK_ACROSS))
    centres = {name: np.asarray(project(camera, pose, np.asarray(state[name]["pose"])[:3])[0])
               for name in state if name.startswith("ball")}
    return built, np.asarray(state["camera:"]["image"]), centres


def darkness(image: np.ndarray, centre: np.ndarray) -> float:
    """The mean counts in a 3 x 3 patch at ``centre`` (an opaque ball's shadow is near zero; the open tube transmits a
    large share of the 1000 counts of full scale)."""
    u, v = int(round(float(centre[0]))), int(round(float(centre[1])))
    return float(image[v - 1:v + 2, u - 1:u + 2].mean())


def test_every_body_appears_unlisted(dtype):
    built, image, centres = first_frame(ballistic(balls=2))
    assert sorted(centres) == ["ball0", "ball1"]
    for name, centre in centres.items():
        assert darkness(image, centre) < 50.0, (name, darkness(image, centre))
    text = built.report.text()
    assert "route 1 (closed form) for every subject: ball0 (sphere), ball1 (sphere), tube (tube)" in text


def test_a_hidden_body_is_absent_and_reported(dtype):
    built, image, centres = first_frame(ballistic(balls=2, hide=("ball0",)))
    assert darkness(image, centres["ball0"]) > 300.0
    assert darkness(image, centres["ball1"]) < 50.0
    assert "hidden: ball0" in built.report.text()
    assert "ball0 (sphere)" not in built.report.text().split("route 1 (closed form)")[1].splitlines()[0]


def test_a_body_added_to_the_experiment_appears(dtype):
    _, alone, centres = first_frame(ballistic(balls=1))
    _, both, centres_two = first_frame(ballistic(balls=2))
    # ball1 sits at the second place along the tube: in the one-ball experiment nothing is there
    assert darkness(alone, centres_two["ball1"]) > 300.0
    assert darkness(both, centres_two["ball1"]) < 50.0


def test_a_name_that_is_no_body_is_refused(dtype):
    with pytest.raises(BuildRefused) as error:
        ballistic(hide=("ghost",)).build()
    assert "camera.hide" in str(error.value)
    with pytest.raises(BuildRefused) as error:
        ballistic(absorbers=(("ghost", 5.0),)).build()
    assert "camera.hide" in str(error.value)
