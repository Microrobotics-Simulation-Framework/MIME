"""The study contract change the camera's calibration needed: an observation's setting is an axis, as a law's is
(``"swim_speed:ball.after"``, ``"camera:.detector.fx"``). Swept (the members differ as the setting does), fitted (the
sphere grid's gate), differentiated in forward mode along a camera's setting (against a central difference), reverse mode
refused with its code, an unknown setting refused with the settings there are, and the law and setting axes that were
axes before unchanged (``tests/study/test_axes.py`` is their test and is not edited).

**The control** (``CAMERA_PLANT=laws-only``) makes the resolver list laws alone, as it did: every case here that names
an observation's setting is refused, every run."""

from __future__ import annotations

import numpy as np
import pytest
from in_loop_cases import STATED_MEMORY, TOO_LITTLE_MEMORY, scene_without_camera, stated_memory

import mime.study._axes as axes_module
from mime import Study
from mime.build._refusals import ArgumentRefused, BuildRefused
from mime.observations import SwimSpeed
from mime.scene import X
from mime.study import AxisKind, along
from mime.study._axes import resolve
from mime.units import m_per_s, ms, s
from in_loop_cases import ballistic, planted

if planted("laws-only"):
    held = axes_module._laws  # noqa: SLF001
    axes_module._laws = lambda declarations: {k: v for k, v in held(declarations).items()  # type: ignore[assignment]
                                              if not k.startswith(("swim_speed", "camera"))}


def swimming():
    exp = scene_without_camera()
    exp.observe(SwimSpeed("ball", along=X, after=s(0.0)))
    return exp


def test_an_observations_setting_is_an_axis_and_a_sweep_along_it_differs_by_it(dtype):
    study = Study(swimming()).sweep(along({"swim_speed:ball.after": [s(0.0), s(0.004)]}), duration=ms(8))
    plan = study.plan()
    assert [axis.kind for axis in plan.axes] == [AxisKind.SETTING]
    result = study.run()
    observed = [float(np.asarray(result.member(i)["swim_speed:ball"].observed.value)) for i in range(2)]
    assert observed[0] == pytest.approx(8e-3, rel=1e-3) and observed[1] == pytest.approx(4e-3, rel=1e-3), observed


def test_an_unknown_observation_setting_is_refused_with_the_settings_there_are(dtype):
    _, declarations, _ = swimming().declared_as()
    assert resolve("swim_speed:ball.after", declarations).kind is AxisKind.SETTING
    with pytest.raises(ArgumentRefused) as refused:
        resolve("swim_speed:ball.afte", declarations)
    assert "study.unknown-path" in str(refused.value) and "swim_speed:ball.after" in refused.value.refusals[0].action


def _along_fx(fx: float = 750.0) -> Study:
    return Study(ballistic(ms(1), detector=_detector(fx))).sweep(along({"camera:.detector.fx": [fx]}), duration=ms(1))


def _expected_counts(final):
    return final["camera:"]["expected"].sum()


@pytest.mark.memory_heavy
def test_the_derivative_along_a_cameras_setting_agrees_with_central_differences_in_both_modes(float64, monkeypatch):
    """Reverse mode runs for real, on a stated memory the check admits it on (the host's is not read: see
    ``in_loop_cases.STATED_MEMORY``); the refusal on too little is the next test's."""
    stated_memory(monkeypatch, STATED_MEMORY)
    found = _along_fx().gradient(_expected_counts, wrt=["camera:.detector.fx"], mode="forward",
                                 acknowledged={"camera:": "the derivative along the camera's setting is the gate's"})
    derivative = float(np.asarray(found.members[0].gradient["camera:.detector.fx"]).reshape(-1)[0])
    h = 0.5

    def level(fx: float) -> float:
        built = ballistic(ms(1), detector=_detector(fx)).build()
        built.run(ms(1))
        return float(np.asarray(built.current_state()["camera:"]["expected"]).sum())

    assert derivative == pytest.approx((level(750.0 + h) - level(750.0 - h)) / (2 * h), rel=1e-3), derivative
    # Reverse mode sums the same terms in another order: within T1's 128 eps of their scale (test_in_loop_reverse.py).
    reverse = _along_fx().gradient(_expected_counts, wrt=["camera:.detector.fx"], mode="reverse",
                                   acknowledged={"camera:": "a test"})
    assert float(np.asarray(reverse.members[0].gradient["camera:.detector.fx"]).reshape(-1)[0]) == pytest.approx(
        derivative, rel=1e-9)


def test_reverse_mode_along_a_cameras_setting_is_refused_on_too_little_memory_and_admitted_on_the_stated_memory(
        float64, monkeypatch):
    """The study the test above differentiates, on a stated memory: admitted on ``STATED_MEMORY`` and refused by name,
    before anything is compiled, on ``TOO_LITTLE_MEMORY`` (the control: the stated memory is what decides, and
    ``Study.gradient`` reads the planted one). Forward mode keeps no tape and is refused on neither."""
    from mime.study import _gradient

    def compiled(*_: object, **__: object) -> None:
        raise AssertionError("the derivative was compiled")

    node = ballistic(ms(1), detector=_detector(750.0)).build().graph.get_node("camera:")
    stated_memory(monkeypatch, STATED_MEMORY)
    assert node.derivative_problem("reverse", 1, 1, 1) is None
    stated_memory(monkeypatch, TOO_LITTLE_MEMORY)
    problem = node.derivative_problem("reverse", 1, 1, 1)
    assert problem is not None and problem.code == "camera.reverse-memory"
    assert node.derivative_problem("forward", 1, 1, 1) is None
    monkeypatch.setattr(_gradient, "prepare", compiled)
    with pytest.raises((BuildRefused, ArgumentRefused)) as refused:
        _along_fx().gradient(_expected_counts, wrt=["camera:.detector.fx"], mode="reverse",
                             acknowledged={"camera:": "a test"})
    text = str(refused.value)
    assert "camera.reverse-memory" in text and "mime.sensing.render" in text and "a stated memory" in text


def _detector(fx: float):
    from in_loop_cases import DETECTOR
    import dataclasses

    return dataclasses.replace(DETECTOR, fx=fx)
