"""The uniform rotating field's phase under a frequency that changes in a run, as an experiment builds it
(MIME-ANO-139, fixed in ``uniform_rotating`` 0.2.0; 0.1.0 kept and guarded).

Version 0.2.0 turns the field by the integral of its commanded frequency, each step's frequency held over the step:
the angle turned over step ``n`` is ``2 pi f_n h``. Version 0.1.0 made the phase as the present frequency times the
time since the start, ``2 pi f(t) t``, so a ramp turned the field twice as fast as commanded and a step in frequency
jumped its angle; it stays registered for a folder saved with it, and refuses any writer of its frequency but the
constant drive (``command.held-constant``), naming 0.2.0.

**The ramp's phase (gate 6).** ``drives.Ramp`` from ``f0`` at ``k`` per second writes each step its value at the step's
middle, the ramp's mean over the step, so 0.2.0's field must point along ``2 pi (f0 t + k t^2 / 2)`` at every step's
end, to rounding on the run's scale: float64 within 1e-11 rad, float32 within 2e-5 rad, as committed in
``tests/study/predictions_drives.json`` before this ran. The field's direction is read from the recorded flux
density. The control: 0.1.0's formula, the present frequency times the time, misses the closed form by ``pi k t^2``,
6.3 rad at 1 s, on every run; and the node planted with that formula in place of its own (a mutant) fails the gate.

**A controller's writes.** The same field under a test controller that writes the frequency each step (a ramp from
zero, a step from 1 to 2 Hz at 0.8 s): the angle is ``2 pi`` times the cumulative sum of what it wrote, one step of
``h`` per write (the integral), so the ramp's ratio to the integral reads 1 and the step leaves no offset, where 0.1.0
read 2 and an offset of 0.799 of a turn (the anomaly's measure, turned round). The control: a constant frequency
agrees with the integral to float32's rounding, so the measure agrees where the node is right.

**The guard.** 0.1.0 refuses the ramp and the controller on its frequency, and builds with a constant drive (no false
refusal), computing for it, to the last bit, what 0.2.0 computes (the same process, so at one usable CPU count and one
OpenBLAS thread count). A planted 0.1.0 that guards nothing builds the ramp: the guard's test fails on it."""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from control_cases import _ABSENT  # importing control_cases registers the test-only namespace this controller joins
from script_a import script_a

from mime import Apparatus, Body, Experiment, Medium, State, drives
from mime.build import BuildRefused, ThingDeclaration
from mime.dynamics import RigidInertial
from mime.fields import KEPT, UniformRotating
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.nodes.actuation import IntegratingRotatingNode
from mime.registry import ControlLaw, Controller, Signal, register_controller
from mime.scene import Scene, Sphere, X, Z
from mime.things import Part
from mime.units import A_m2, Hz, kg_per_m3, m_per_s, m_per_s2, mPa_s, ms, mT, rad_per_s, s, um

DT = 1e-3
STEPS = 1000
#: The ramp's slope (Hz per s), and the step's two frequencies (Hz) and its time (s).
SLOPE, LOW, HIGH, STEP_AT = 2.0, 1.0, 2.0, 0.8
#: The drive's ramp: from F0 at K per second (Hz, Hz/s), to well past the run's end.
F0, K = 1.0, 2.0
#: The closed form's tolerance at every step, rad, per precision (committed before the gate ran).
CLOSED_FORM = {"float64": 1e-11, "float32": 2e-5}
OLD, _ = KEPT["uniform_rotating", "0.1.0"]


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


def sphere(field: Any = None, frequency: Any = None) -> Experiment:
    """A magnetised sphere in a uniform rotating field (``field``, 0.2.0 when not given), its frequency driven by
    ``frequency`` (a quantity or a drive; none for a controller to write)."""
    exp = Experiment("phase", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(DT * 1e3))
    exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
    exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
    exp.declare("ball.magnetic_moment", A_m2((0.0, 1e-9, 0.0)), reason="a magnetised sphere")
    provider = (field or UniformRotating)(rotation_axis=X, initial_direction=(0.0, 1.0, 0.0))
    commands = {"amplitude": mT(1.2), **({"frequency": frequency} if frequency is not None else {})}
    exp.add_apparatus(Apparatus("field", fields={"magnetic": provider}, commands=commands))
    exp.attach(Magnetic.Torque(), body=ball, field="field.magnetic")
    return exp


def angle_of(history: Mapping[str, Any]) -> np.ndarray:
    """The field's angle about its axis (+x) at each step, from +y towards +z (radians in (-pi, pi])."""
    field = next(name for name in history if name.startswith("field."))
    flux = np.asarray(history[field][next(k for k in history[field] if k.startswith("flux_density"))], np.float64)
    return np.arctan2(flux[:, 2], flux[:, 1])


def run(shape: str) -> tuple[np.ndarray, np.ndarray]:
    """The field's angle at each step, and the frequency the controller wrote at each step, from one run of ``STEPS``
    steps of 0.2.0 under the controller, in the process's precision."""
    exp = sphere()
    exp.set_controller(Schedule(shape=shape), reads={"pose": "ball.pose"}, writes={"frequency": "field.frequency"},
                       mode="graph", rate=ms(DT * 1e3))
    history = exp.build().run(s(STEPS * DT), history=True).history
    assert history is not None
    return angle_of(history), np.asarray(history["controller"]["write_frequency"], np.float64)


def integral(written: np.ndarray) -> np.ndarray:
    """``2 pi`` times the integral of the frequency written, one step of ``DT`` for each write."""
    return 2.0 * math.pi * np.cumsum(written) * DT


def apart(angle: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """The angle's distance from ``expected`` on the circle, in radians (a whole turn is the same direction)."""
    return np.abs((angle - expected + math.pi) % (2.0 * math.pi) - math.pi)


@pytest.fixture(params=["float32", "float64"])
def precision(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    try:
        yield request.param
    finally:
        jax.config.update("jax_enable_x64", held)


def ramped(field: Any = None) -> Experiment:
    return sphere(field, drives.Ramp(Hz(F0), to=Hz(1e3), rate=Hz(K) / s(1.0)))


def closed_form() -> np.ndarray:
    t = np.arange(1, STEPS + 1, dtype=np.float64) * DT
    return 2.0 * math.pi * (F0 * t + K * t * t / 2.0)


def ramp_angle() -> np.ndarray:
    history = ramped().build().run(s(STEPS * DT), history=True).history
    assert history is not None
    return angle_of(history)


def test_the_ramps_field_turns_by_its_closed_form_integral_at_every_step(precision, record_property):
    miss = float(np.max(apart(ramp_angle(), closed_form())))
    record_property("largest_miss_rad", miss)
    print(f"\n{precision}: the field's angle against 2 pi (f0 t + k t^2 / 2): largest miss {miss:.3e} rad")
    assert miss <= CLOSED_FORM[precision]


def test_control_the_present_frequency_times_the_time_misses_the_closed_form():
    """0.1.0's formula, applied to what the ramp commands at each step: off by ``pi k t^2`` (6.3 rad at 1 s)."""
    t = np.arange(1, STEPS + 1, dtype=np.float64) * DT
    commanded = F0 + K * (t - DT / 2)
    miss = float(np.max(apart(2.0 * math.pi * commanded * t, closed_form())))
    assert miss > 1.0 and miss > 1e5 * CLOSED_FORM["float32"]


def test_mutant_the_node_with_the_present_frequency_times_the_time_fails_the_gate(monkeypatch):
    """The 0.2.0 node planted with 0.1.0's phase (the mutant ``the ramp's phase as the present frequency times the
    time``) fails the closed form on the first run, in float64. Each build constructs its nodes afresh, so no step
    traced before the plant is reused; JAX's caches are cleared on each side of it."""

    def planted(self: Any, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        zero = jnp.zeros((), dtype=state["previous"].dtype)
        steps = state["steps"] + 1
        frequency = boundary_inputs.get("frequency", zero)
        phase = 2 * jnp.pi * frequency * steps.astype(zero.dtype) * dt
        return self._fields(steps, frequency, state["correction"],
                            self.field_at(p, phase, boundary_inputs.get("amplitude", zero)))

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        monkeypatch.setattr(IntegratingRotatingNode, "advance", planted)
        jax.clear_caches()
        angle = ramp_angle()
    finally:
        jax.config.update("jax_enable_x64", held)
        jax.clear_caches()
    assert float(np.max(apart(angle, closed_form()))) > 1.0


def test_control_a_constant_frequency_turns_the_field_by_its_integral():
    angle, written = run("constant")
    assert np.all(written == LOW)
    assert np.max(apart(angle, integral(written))) / (2.0 * math.pi) < 1e-4, "the measure agrees where it is right"


def test_a_ramp_from_zero_turns_the_field_by_the_commanded_integral():
    angle, written = run("ramp")
    turned = np.unwrap(angle)
    late = slice(STEPS // 2, STEPS)
    assert np.max(apart(angle, integral(written))) / (2.0 * math.pi) < 1e-4, "the integral at every step"
    assert np.allclose(turned[late] / integral(written)[late], 1.0, rtol=5e-4), "once the commanded integral, not 2"
    assert 0.95 < turned[-1] / (2.0 * math.pi) < 1.05


def test_a_step_in_frequency_turns_the_field_at_each_rate_and_leaves_no_offset():
    angle, written = run("step")
    step = int(np.argmax(written > 0.5 * (LOW + HIGH)))
    assert 0 < step < STEPS - 1
    assert np.max(apart(angle, integral(written))) / (2.0 * math.pi) < 1e-4, "no offset, before or after the step"
    rates = np.diff(np.unwrap(angle)) / (2.0 * math.pi * DT)
    assert np.allclose(rates[:step - 1], LOW, rtol=1e-3) and np.allclose(rates[step:], HIGH, rtol=1e-3)


def test_the_kept_version_refuses_a_ramp_on_its_frequency_naming_the_current_one():
    with pytest.raises(BuildRefused) as refused:
        ramped(OLD).build()
    assert refused.value.codes() == ["command.held-constant"]
    held = refused.value.refusals[0]
    assert "uniform_rotating 0.1.0" in held.message and "drive.ramp 0.1.0" in held.message
    assert "uniform_rotating 0.2.0" in held.action and "UniformRotating(...)" in held.action


def test_the_kept_version_refuses_a_controller_on_its_frequency():
    exp = sphere(OLD)
    exp.set_controller(Schedule(shape="constant"), reads={"pose": "ball.pose"},
                       writes={"frequency": "field.frequency"}, mode="graph", rate=ms(DT * 1e3))
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    assert refused.value.codes() == ["command.held-constant"]
    assert "the controller changes field.frequency during a run" in refused.value.refusals[0].message


def test_mutant_a_kept_version_that_guards_nothing_builds_the_ramp_and_the_guard_test_fails_on_it(monkeypatch):
    monkeypatch.setattr(OLD, "constant_commands", {})
    ramped(OLD).build()
    with pytest.raises(pytest.fail.Exception):
        test_the_kept_version_refuses_a_ramp_on_its_frequency_naming_the_current_one()


def on_the_kept_field(experiment: Experiment) -> Experiment:
    """``experiment`` with each uniform rotating field at its kept version, 0.1.0, and nothing else changed."""
    held = cast(Any, experiment)
    declarations = []
    for d in held._declarations:
        if isinstance(d, ThingDeclaration) and isinstance(d.thing, Apparatus):
            fields = {name: OLD(**{f.name: getattr(v, f.name) for f in dataclasses.fields(v)})
                      if isinstance(v, UniformRotating) else v for name, v in d.thing.fields.items()}
            d = ThingDeclaration(dataclasses.replace(d.thing, fields=fields))
        declarations.append(d)
    return cast(Experiment, held._declared(held._settings, declarations))


def test_with_a_constant_drive_the_kept_version_builds_and_the_current_one_computes_its_field_to_the_last_bit(
        precision):
    """No false refusal; and script A's every field, over 200 steps, the same on either version, bit for bit (in one
    process: one usable CPU count, one OpenBLAS thread count)."""
    old = on_the_kept_field(script_a()).build()
    assert next(p for p in old.plans.entries if p.owner == "field.magnetic").registered.cls is OLD
    before = old.run(s(0.1), history=True).history
    after = script_a().build().run(s(0.1), history=True).history
    assert before is not None and after is not None
    moved = [f"{node}.{name}" for node, fields in before.items() for name, value in fields.items()
             if not np.array_equal(np.asarray(value), np.asarray(after[node][name]))]
    assert not moved, moved
    assert set(after["field.magnetic"]) - set(before["field.magnetic"]) == {"previous", "correction"}
    assert not np.any(np.asarray(after["field.magnetic"]["correction"])), "a constant frequency leaves it zero"
