"""``UniformRotating`` 0.1.0 turns its field by the present frequency times the time since the start, not by the
integral of the frequency (MIME-ANO-139), as an experiment builds it. A characterisation of a known defect, not a pass.

The node's phase is ``2 pi f(t) t``. For a constant frequency that is the integral ``2 pi f t``, so a script that
declares one (script A's and ``screw_in_tube``'s) turns the field exactly. For a frequency a controller changes in a
run it is not: the field turns at ``2 pi (f + t df/dt)``, not at ``2 pi f``. A linear ramp from zero turns the field
twice as fast as commanded, and a step from ``f1`` to ``f2`` at ``t_s`` moves the field's angle by
``2 pi (f2 - f1) t_s`` in the step that makes it, a jump the field does not make at the rate it then turns at.

Each test builds the experiment through the front door, with a test controller that writes the field's frequency as a
function of time, runs it in float32, and reads the field's direction from the recorded flux density. The measure is
the field's angle against ``2 pi`` times the cumulative sum of the frequency the controller wrote, step by step (the
integral the field would turn by if it integrated). The control is the same experiment at a constant frequency: the
two agree to float32's rounding, so the measure agrees where the node is right and the ramp's 2 is the node's.
The ramp's and the step's tests fail on a node that integrates (checked with one planted, in place of the node: the
ramp's ratio goes to 1 and the step's offset to zero), and are the ones to turn round when the field is fixed."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from control_cases import _ABSENT  # importing control_cases registers the test-only namespace this controller joins

from mime import Apparatus, Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.registry import ControlLaw, Controller, Signal, register_controller
from mime.scene import Scene, Sphere, X, Z
from mime.things import Part
from mime.units import A_m2, kg_per_m3, m_per_s, m_per_s2, mPa_s, ms, mT, rad_per_s, s, um

DT = 1e-3
STEPS = 1000
#: The ramp's slope (Hz per s), and the step's two frequencies (Hz) and its time (s).
SLOPE, LOW, HIGH, STEP_AT = 2.0, 1.0, 2.0, 0.8


@register_controller
@dataclass(frozen=True, kw_only=True)
class Schedule(Controller):
    """Writes ``f(t)`` as the field's frequency at its tick ``k`` (``t = (k + 1) dt``, the time of the step it acts on
    in the field's own count): ``ramp`` is ``SLOPE t``, ``step`` is ``LOW`` before ``STEP_AT`` and ``HIGH`` from it,
    ``constant`` is ``LOW``."""

    name = "oracle.frequency_schedule"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"frequency": Signal("Hz", ())}
    shape: str = "constant"

    def law(self, ctx: Any) -> ControlLaw:
        dtype, shape = jnp.zeros(()).dtype, self.shape

        def at(n: Any) -> Any:
            t = (n + 1).astype(dtype) * DT
            if shape == "ramp":
                return SLOPE * t
            if shape == "step":
                return jnp.where(t < STEP_AT, LOW, HIGH).astype(dtype)
            return jnp.asarray(LOW, dtype=dtype) + 0.0 * t

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[Any, Any]:
            n = state["n"] + 1
            return {"n": n}, {"frequency": at(n)}

        return ControlLaw(state={"n": jnp.asarray(0, dtype=jnp.int32)}, held={"frequency": at(jnp.asarray(-1))},
                          step=step)


def run(shape: str) -> tuple[np.ndarray, np.ndarray]:
    """The field's angle about its axis at each step (radians, unwrapped except across the step), and the frequency the
    controller wrote at each step, from one run of ``STEPS`` steps."""
    exp = Experiment("phase", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(DT * 1e3))
    exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
    exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
    exp.declare("ball.magnetic_moment", A_m2((0.0, 1e-9, 0.0)), reason="a magnetised sphere")
    exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(rotation_axis=X,
                                                                              initial_direction=(0.0, 1.0, 0.0))},
                                commands={"amplitude": mT(1.2)}))
    exp.attach(Magnetic.Torque(), body=ball, field="field.magnetic")
    exp.set_controller(Schedule(shape=shape), reads={"pose": "ball.pose"}, writes={"frequency": "field.frequency"},
                       mode="graph", rate=ms(DT * 1e3))
    history = exp.build().run(s(STEPS * DT), history=True).history
    field = next(name for name in history if name.startswith("field."))
    flux = np.asarray(history[field][next(k for k in history[field] if k.startswith("flux_density"))], np.float64)
    written = np.asarray(history["controller"]["write_frequency"], np.float64)
    return np.arctan2(flux[:, 2], flux[:, 1]), written


def integral(written: np.ndarray) -> np.ndarray:
    """``2 pi`` times the integral of the frequency the controller wrote, one step of ``DT`` for each write."""
    return 2.0 * math.pi * np.cumsum(written) * DT


def turns_apart(angle: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """The angle's distance from ``expected`` on the circle, in turns (a whole turn is the same direction)."""
    return np.abs((angle - expected + math.pi) % (2.0 * math.pi) - math.pi) / (2.0 * math.pi)


def test_control_a_constant_frequency_turns_the_field_by_its_integral():
    angle, written = run("constant")
    assert np.all(written == LOW)
    assert np.max(turns_apart(angle, integral(written))) < 1e-4, "the measure agrees where the node is right"


def test_a_ramp_from_zero_turns_the_field_twice_as_fast_as_commanded():
    angle, written = run("ramp")
    t = (np.arange(STEPS) + 1) * DT
    turned = np.unwrap(angle)
    assert np.max(turns_apart(angle, 2.0 * math.pi * written * t)) < 1e-4, "the node's phase is f(t) t"
    late = slice(STEPS // 2, STEPS)
    assert np.allclose(turned[late] / integral(written)[late], 2.0, rtol=5e-3), "twice the commanded integral"
    assert 1.9 < turned[-1] / (2.0 * math.pi) < 2.1 and 0.95 < integral(written)[-1] / (2.0 * math.pi) < 1.05


def test_a_step_in_frequency_moves_the_fields_angle_by_two_pi_df_t_and_the_offset_stays():
    angle, written = run("step")
    step = int(np.argmax(written > 0.5 * (LOW + HIGH)))
    # The node's HIGH (step + 1) DT against the sum's LOW step DT + HIGH DT.
    offset = 2.0 * math.pi * (HIGH - LOW) * step * DT
    before, after = slice(0, step), slice(step, STEPS)
    assert np.max(turns_apart(angle[before], integral(written)[before])) < 1e-4
    held = integral(written)[after] + offset
    assert np.max(turns_apart(angle[after], held)) < 1e-4, "a constant offset from the step"
    assert 0.1 < (offset / (2.0 * math.pi)) % 1.0 < 0.9, "the step is placed where the offset is not a whole turn"
