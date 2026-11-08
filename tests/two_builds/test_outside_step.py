"""MIME-ANO-146's detection: a callback controller's setting (A0's tracking controller's ``home``, ``tracked_rpm``)
reaches the run through the controller's held writes, the graph's external inputs, and not through the step's
parameters. The trace form sees the step only, so it is quiet; the run form, which writes the leaves that differ into
the base's program but feeds it the base's own inputs, fires; and the check says where the value went
(``inputs_moved``) and gives the verdict ``outside-step``. Its control: the same experiment built twice is quiet by
both forms. The anomaly stays open: its resolution is controller settings as leaves (lane S's gradient phase); this is
how the check reports it meanwhile, on every run.

The ladder: ``home`` moved by a relative 1e-6 (the spike's), 1e-5 and 1e-4, every joint at once. The controller holds
its pose in float32 (its writes are the arm's joint positions in float32), so a move below about 6e-8 does not reach
the run at all, and the check then rightly finds nothing moved."""

from __future__ import annotations

import pytest
from chain_cases import AR4, AR4_BASE, AR4_HOME, tracked_rpm

from mime.control import ArmTracking
from mime.scene import X, Z, asset
from mime.testing import two_builds
from mime.units import m, rad


def tracking_at(home: tuple[float, ...]) -> ArmTracking:
    """``chain_cases.tracking``'s controller with its ``home`` written."""
    return ArmTracking(description=asset(AR4), end_effector="link_6", home=rad(home), axis=X, up=Z, at=m(AR4_BASE))


@pytest.fixture(scope="module")
def base():
    return tracked_rpm().build()


@pytest.mark.parametrize("move", [1e-6, 1e-5, 1e-4])
def test_a_callback_controllers_setting_is_reported_outside_the_step(base, move: float):
    other = tracked_rpm(controller=tracking_at(tuple(h * (1.0 + move) for h in AR4_HOME))).build()
    found = two_builds(base, other)
    assert not found.trace_fired, "the step's program is the same"
    assert not found.moved, "no leaf moved"
    assert found.inputs_moved, "the controller's held writes moved"
    assert found.verdict == "outside-step"
    assert found.run is not None and found.run.ratio >= 4.0, "four times k resolutions above the run form's line"


def test_control_the_same_controller_built_twice_is_quiet(base):
    found = two_builds(base, tracked_rpm().build())
    assert found.verdict == "quiet" and not found.inputs_moved
