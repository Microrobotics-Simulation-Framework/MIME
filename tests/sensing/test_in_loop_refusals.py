"""G11 for the in-loop camera: each refusal comes with its code, and the experiment without the refused thing builds (the
case is not refused for another reason).

frame_period not a whole number of base steps (naming the nearest two), more sub-frames than the period, ``rate=``, a
body or structure without a closed form (until the sampled route), a ``hide`` or an absorber that names nothing (in
``test_in_loop_scene.py``), a mount naming no frame, recording frames (``record=``; ``history=True`` is the all-fields verification path and is not refused), reverse mode
through the camera and a batched study.

**The controls** plant the three refusals a user meets first: ``CAMERA_PLANT=frame-period`` rounds any period,
``CAMERA_PLANT=recordable`` makes the image recordable, and ``CAMERA_PLANT=box-form`` registers a closed form for the box;
each fails its case every run. The others are deterministic checks of one condition each, without a planted control."""

from __future__ import annotations

import dataclasses

import pytest
from in_loop_cases import LOOK_ACROSS, ballistic, planted

from mime import Body, State
from mime.build import ArgumentRefused, BuildRefused
from mime.dynamics import RigidInertial
from mime.observations import Camera, Mounted
from mime.scene import Box
from mime.study import Study, along
from mime.things import Part
from mime.units import kg_per_m3, m_per_s, ms, rad_per_s, um

if planted("frame-period"):
    import mime.observations._camera as camera_module

    camera_module.period_steps = lambda period, dt: max(1, round(period / dt))
if planted("recordable"):
    Camera.outputs = tuple(dataclasses.replace(o, recordable=True) for o in Camera.outputs)
if planted("box-form"):
    from mime.sensing import register_closed_form
    from mime.sensing._subjects import _cylinder

    register_closed_form("box", _cylinder)


def refused(build) -> str:
    with pytest.raises((BuildRefused, ArgumentRefused)) as error:
        build()
    return str(error.value)


def test_a_period_that_is_no_whole_number_of_steps_is_refused_naming_the_nearest_two(dtype):
    ballistic(ms(3)).build()
    text = refused(lambda: ballistic(ms(2.5)).build())
    assert "camera.frame-period" in text and "2 steps" in text and "3 steps" in text


def test_more_sub_frames_than_the_period_is_refused(dtype):
    ballistic(ms(3), sub_frames=3).build()
    assert "camera.frame-period" in refused(lambda: ballistic(ms(3), sub_frames=4).build())


def test_rate_is_refused(dtype):
    assert "camera.rate" in refused(lambda: ballistic(ms(3), rate=ms(2)).build())


def test_a_mount_that_names_no_frame_is_refused_with_the_frames_there_are(dtype):
    text = refused(lambda: ballistic(ms(3), mount=Mounted(frame="stage.gantry.nope", orientation=LOOK_ACROSS)).build())
    assert "camera.mount" in text


def with_box(hide=()):
    exp = ballistic(ms(3), hide=hide)
    exp.add_body(Body("crate", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Box(size=um((10.0, 10.0, 10.0))))}))
    exp.declare("crate.parts.shell.density", kg_per_m3(2000.0), reason="a test body")
    return exp


def test_a_body_without_a_closed_form_is_refused_with_its_reason_until_it_is_hidden(dtype):
    text = refused(lambda: with_box().build())
    assert "camera.no-closed-form" in text and "box" in text
    with_box(hide=("crate",)).build()


def test_frames_are_not_recorded_in_the_loop(dtype):
    built = ballistic(ms(3)).build()
    built.run(ms(2), record=["camera:.frame_step"])
    assert "camera.recorded-frames" in refused(lambda: built.run(ms(2), record=["camera:.image"]))
    assert "camera.recorded-frames" in refused(lambda: built.run(ms(2), record=["camera:.expected"]))


def test_reverse_mode_through_the_camera_is_refused_naming_the_offline_renderer(dtype):
    density = kg_per_m3(7800.0)
    study = Study(ballistic(ms(3))).sweep(along({"ball.parts.steel.density": [density]}), duration=ms(2))

    def of(final):
        return final["camera:"]["expected"].sum()

    text = refused(lambda: study.gradient(of, wrt=["ball.parts.steel.density"], mode="reverse",
                                          acknowledged={"camera:": "a test"}))
    assert "camera.reverse" in text and "mime.sensing.render" in text


def test_a_batched_study_is_refused_naming_the_offline_renderer(dtype):
    one = m_per_s((2e-3, 0.0, 0.0))
    two = m_per_s((3e-3, 0.0, 0.0))
    Study(ballistic(ms(3))).sweep(along({"ball.initial.velocity": [one]}), duration=ms(2)).plan()
    study = Study(ballistic(ms(3))).sweep(along({"ball.initial.velocity": [one, two]}), duration=ms(2))
    text = refused(lambda: study.plan())
    assert "camera.batched" in text and "mime.sensing.render" in text
