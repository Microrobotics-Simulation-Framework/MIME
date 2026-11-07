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
The float32 arm's resolution is its tolerance, 2e-5 rad: a ramp value read 1/500 of a step early (1.3e-5 rad) is below
it and passes there (the float64 arm catches it, and 1/5000 of a step too); the planted mutants are far above it.

**A controller's writes.** The same field under a test controller that writes the frequency each step (a ramp from
zero, a step from 1 to 2 Hz at 0.8 s): the angle is ``2 pi`` times the cumulative sum of what it wrote, one step of
``h`` per write (the integral), so the ramp's ratio to the integral reads 1 and the step leaves no offset, where 0.1.0
read 2 and an offset of 0.799 of a turn (the anomaly's measure, turned round). The control: a constant frequency
agrees with the integral to float32's rounding, so the measure agrees where the node is right.

**The long run (MIME-ANO-139's rounding).** The phase is held as a pair of floating numbers to which each step adds the
step's turns as an exact product by an error-free sum, so its rounding does not grow with the turns made. Read for 1000
s of 0.5 ms steps (2e6 steps, up to 165 000 turns) under a constant frequency, a ramp and a controller-like noisy writer,
the field's angle is within ``LONG_RUN["float32"]`` of the integral of the frequencies written, an exact sum, at every
step, in float32, and the error at 100 s and at 1000 s alike: float32's resolution of an angle in one turn, which the
output's cosine and sine round to (the previous scheme, the present frequency times the time less a correction, was
0.003 to 0.13 rad off by 100 to 1000 s, growing with the phase). The control: that scheme's main term in float32, the
product of the frequency and the time, misses by more than ``CONTROL_MISS`` at 100 s, so the bound can fail. In float64
the field is within ``LONG_RUN["float64"]`` of the exact turns at both (read one step before a whole 100 s and 1000 s,
where the constant and the ramp stand at a whole turn).

**The guard.** 0.1.0 refuses the ramp and the controller on its frequency, and builds with a constant drive (no false
refusal), computing for it the field 0.2.0 computes to 0.1.0's own rounding of the phase (not bit for bit: the pair
avoids the rounding of ``2 pi f t``). A planted 0.1.0 that guards nothing builds the ramp: the guard's test fails on it."""

import dataclasses
import math
from fractions import Fraction
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
from mime.nodes.actuation._turns import advance_phase, turns_of_step
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
#: The long run: 1000 s of 0.5 ms steps, read at 100 s and at the end; the angle's bound per precision (rad): float32's
#: spacing of an angle in a turn, 4.8e-7, doubled (the output's cosine and sine round to it); float64's, the exact
#: turns' reading to 1e-13 (the angle's own rounding at pi is 4e-16). The control must miss by more than CONTROL_MISS at 100 s.
LONG_STEP, LONG_STEPS, LONG_CHECK = 5e-4, 2_000_000, 200_000
LONG_RUN = {"float32": 1e-6, "float64": 1e-13}
CONTROL_MISS = 1e-3


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
        zero = jnp.zeros((), dtype=state["phase"].dtype)
        steps = state["steps"] + 1
        frequency = boundary_inputs.get("frequency", zero)
        phase = 2 * jnp.pi * frequency * steps.astype(zero.dtype) * dt
        return self._fields(steps, state["phase"], state["residual"],
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


def field_angle(history: Mapping[str, Any]) -> np.ndarray:
    return angle_of({name: fields for name, fields in history.items() if name.startswith("field.")})


def test_with_a_constant_drive_the_kept_version_builds_and_the_current_one_computes_the_same_field(precision):
    """No false refusal; and script A's field, over 200 steps, the same on either version to the precision's own
    rounding of a turn: ``2 pi f t`` as the kept version computes it carries a relative rounding of the phase, the
    current version's phase does not, so the two differ by up to the kept version's rounding (a few float32 places of
    the angle over one turn), and are not bit for bit the same."""
    old = on_the_kept_field(script_a()).build()
    assert next(p for p in old.plans.entries if p.owner == "field.magnetic").registered.cls is OLD
    before = old.run(s(0.1), history=True).history
    after = script_a().build().run(s(0.1), history=True).history
    assert before is not None and after is not None
    miss = float(np.max(apart(field_angle(before), field_angle(after))))
    print(f"\n{precision}: the kept and the current field's angle over script A's 200 steps differ by {miss:.2e} rad")
    assert miss < {"float32": 2e-6, "float64": 1e-13}[precision]
    assert set(after["field.magnetic"]) - set(before["field.magnetic"]) == {"phase", "residual"}


def long_writer(shape: str, steps: np.ndarray) -> np.ndarray:
    """The frequency written at each (1-based) step of the long run, float64: a constant 40 Hz, a ramp from 40 Hz at
    0.25 Hz/s (the ramp's value at the step's middle) and a controller-like writer, 40 Hz and two bounded oscillations."""
    middle = (steps - 0.5) * LONG_STEP
    if shape == "constant":
        return np.full(steps.shape, 40.0)
    if shape == "ramp":
        return 40.0 + 0.25 * middle
    return 40.0 + 3.0 * np.sin(0.37 * steps) + 0.5 * np.sin(1.7 * steps)


def long_run_angles(frequencies: np.ndarray) -> np.ndarray:
    """The integrating node's field angle (rad, float64) at each step of a run that writes ``frequencies``, from the
    node's own ``advance`` scanned, in the process's precision."""
    node = IntegratingRotatingNode("f", LONG_STEP, slots=("s",), rotation_axis=(1.0, 0.0, 0.0),
                                   initial_direction=(0.0, 1.0, 0.0))
    p = {"rotation_axis": jnp.asarray((1.0, 0.0, 0.0)), "initial_direction": jnp.asarray((0.0, 1.0, 0.0))}
    start = node.initial_state()

    def body(state: Any, written: Any) -> Any:
        out = node.advance(p, state, {"frequency": written, "amplitude": jnp.asarray(1.0, dtype=written.dtype)}, LONG_STEP)
        flux = out["flux_density_s"]
        return out, jnp.arctan2(flux[2], flux[1])

    dtype = start["phase"].dtype
    _, angle = jax.jit(lambda f: jax.lax.scan(body, start, f))(jnp.asarray(frequencies, dtype=dtype))
    return np.asarray(angle, np.float64)


def exact_turns(written: np.ndarray, count: int) -> float:
    """The turns the first ``count`` of ``written`` (floats) make, ``sum(f) h`` mod 1, by exact integer and rational
    arithmetic (each float is a multiple of 2^-60 at these sizes)."""
    total = sum(int(math.ldexp(float(value), 60)) for value in written[:count])
    return float((Fraction(total, 1 << 60) * Fraction(LONG_STEP)) % 1)


@pytest.mark.parametrize("shape", ["constant", "ramp", "noisy"])
def test_the_phase_does_not_grow_with_the_run_in_float32(shape, record_property):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        written = long_writer(shape, np.arange(1, LONG_STEPS + 1, dtype=np.float64)).astype(np.float32)
        angle = long_run_angles(written)
    finally:
        jax.config.update("jax_enable_x64", held)
    turns = np.cumsum(written.astype(np.longdouble)) * np.longdouble(LONG_STEP)  # exact sums: float32 inputs
    wrapped = np.asarray(turns - np.floor(turns), np.float64)
    miss = apart(angle, 2.0 * math.pi * wrapped)
    early, late = float(np.max(miss[:LONG_CHECK])), float(np.max(miss[-LONG_CHECK:]))
    record_property("largest_miss_rad", float(np.max(miss)))
    print(f"\n{shape}, float32, {float(turns[-1]):.0f} turns: the angle's largest miss over the first 100 s {early:.2e} "
          f"rad, over the last 100 s {late:.2e} rad, over the run {float(np.max(miss)):.2e} rad")
    assert float(np.max(miss)) <= LONG_RUN["float32"], "the run's largest miss, 1000 s"
    assert late <= LONG_RUN["float32"] and early <= LONG_RUN["float32"], "the same at 100 s and at 1000 s"


def test_control_float32s_product_of_the_frequency_and_the_time_is_off_by_100_s():
    """The scheme's main term: ``2 pi f t`` in float32 at 40 Hz misses by 0.003 rad at 100 s (and 0.025 at 1000 s): the
    long-run bound can fail, and does for the scheme whose error grows with the phase."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        count = jnp.arange(LONG_CHECK - 999, LONG_CHECK + 1, dtype=jnp.float32)
        angle = np.asarray(2 * jnp.pi * jnp.float32(40.0) * count * jnp.float32(LONG_STEP), np.float64)
    finally:
        jax.config.update("jax_enable_x64", held)
    exact = 2.0 * math.pi * ((40.0 * np.arange(LONG_CHECK - 999, LONG_CHECK + 1, dtype=np.float64) * LONG_STEP) % 1.0)
    assert float(np.max(apart(angle, exact))) > CONTROL_MISS


@pytest.mark.parametrize("shape", ["constant", "ramp", "noisy"])
def test_the_phase_is_exact_in_float64_over_the_long_run(shape, record_property):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        written = long_writer(shape, np.arange(1, LONG_STEPS + 1, dtype=np.float64))
        angle = long_run_angles(written)
    finally:
        jax.config.update("jax_enable_x64", held)
    misses = [float(apart(angle[count - 1:count], np.asarray([2.0 * math.pi * exact_turns(written, count)]))[0])
              for count in (LONG_CHECK - 1, LONG_STEPS - 1)]
    record_property("miss_100s_rad", misses[0])
    record_property("miss_1000s_rad", misses[1])
    print(f"\n{shape}, float64: the angle's miss of the exact turns, at 100 s {misses[0]:.2e} rad, at 1000 s "
          f"{misses[1]:.2e} rad")
    assert max(misses) <= LONG_RUN["float64"]


def test_the_phase_pair_adds_the_exact_product_of_the_frequency_and_the_step_in_float32():
    """The building block: ``frequency * timestep`` as a pair is exact to ``eps^2``, where the plain float32 product is
    off by ``eps`` (the control)."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        frequency = np.random.default_rng(7).uniform(0.5, 600.0, 20000).astype(np.float32)
        p, e = jax.jit(jax.vmap(lambda f: turns_of_step(f, LONG_STEP)))(jnp.asarray(frequency))
        plain = np.asarray(jnp.asarray(frequency) * jnp.float32(LONG_STEP), np.float64)
    finally:
        jax.config.update("jax_enable_x64", held)
    exact = np.asarray(frequency, np.longdouble) * np.longdouble(LONG_STEP)
    pair = np.abs((np.asarray(p, np.longdouble) + np.asarray(e, np.longdouble) - exact) / exact)
    control = np.abs((plain - np.asarray(exact, np.float64)) / np.asarray(exact, np.float64))
    print(f"\nrelative error of the pair {float(np.max(pair)):.2e}; of the plain float32 product {float(np.max(control)):.2e}")
    assert float(np.max(pair)) < 1e-13 and float(np.max(control)) > 1e-8


def test_the_derivative_of_the_phase_with_respect_to_each_frequency_written_is_the_step():
    """``d Theta_n / d f_j = h`` for every ``j <= n`` (the fraction's reduction drops whole turns, not slope), through
    the pair's arithmetic: its tangent is given, so the arithmetic is not differentiated."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        written = jnp.asarray(np.linspace(30.0, 50.0, 64))

        def phase(frequencies: Any) -> Any:
            def body(state: Any, f: Any) -> Any:
                hi, lo = advance_phase(state[0], state[1], f, LONG_STEP)
                return (hi, lo), None

            (hi, lo), _ = jax.lax.scan(body, (jnp.asarray(0.0), jnp.asarray(0.0)), frequencies)
            return hi + lo

        grad = np.asarray(jax.grad(phase)(written))
        slope = float(jax.jvp(phase, (written,), (jnp.ones_like(written),))[1])
    finally:
        jax.config.update("jax_enable_x64", held)
    assert np.allclose(grad, LONG_STEP, rtol=1e-14, atol=0.0) and slope == pytest.approx(64 * LONG_STEP, rel=1e-14)


def test_mutant_a_phase_held_as_one_float_wrapped_each_step_fails_the_long_run_bound(monkeypatch):
    """The pair planted as a single float32 fraction, ``frac(phase + f h)`` each step (the naive accumulator): the
    float32 ramp misses the bound within 100 s on every run."""
    import mime.nodes.actuation._uniform_field as field_module

    def naive(hi: Any, lo: Any, frequency: Any, timestep: float) -> Any:
        total = hi + frequency * jnp.asarray(timestep, dtype=hi.dtype)
        return total - jnp.floor(total), lo

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        monkeypatch.setattr(field_module, "advance_phase", naive)
        written = long_writer("ramp", np.arange(1, LONG_CHECK + 1, dtype=np.float64)).astype(np.float32)
        angle = long_run_angles(written)
    finally:
        jax.config.update("jax_enable_x64", held)
    turns = np.cumsum(written.astype(np.longdouble)) * np.longdouble(LONG_STEP)
    miss = float(np.max(apart(angle, 2.0 * math.pi * np.asarray(turns - np.floor(turns), np.float64))))
    print(f"\nthe naive accumulator: largest miss over 100 s {miss:.2e} rad, against the bound {LONG_RUN['float32']:.0e}")
    assert miss > 100 * LONG_RUN["float32"]
