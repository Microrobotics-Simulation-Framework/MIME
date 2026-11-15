"""``examples/camera_loop.py``: camera spike R2's checks of the differentiable closed loop, on R2's case and draws.

**What these bound, and what they estimate.** They hold the loop's derivative to the derivative of what it computes
(reverse mode against forward mode, float32 against float64, both against central differences whose order is measured)
and its allocation to an independent solver (SciPy's BVLS) at every tick; that the two controller modes run one law; that
a lost target is held and flagged, never a NaN; and that as the latencies go to nothing and the frame rate to the step
rate the loop meets the true-state run, within what the tracker's own error and the loop's linear response bound. None
of them says the loop tracks well: the camera does not see depth, and on a draw where a coil sits at its bound the
sphere drifts off the plane (R2: 4.6 mm in 0.2 s), which the tracker flags. The tracker's accuracy has its own gates
(``test_camera_loop_tracker.py``).

**Tolerances** (from mechanisms, never from the results; the predictions in ``predictions_camera_3c.json`` were
committed before any probe ran):

- **reverse against forward**: ``16 eps N S``, ``S`` the per-step terms' absolute sum of the derivative (from the forward
  tangents), ``N`` the steps: one program's rounding, accumulated linearly, sixteen operations a step (R2's line).
- **float32 against float64**: the objective within ``16 eps32 N J``; each derivative within ``(16 N eps32 + 1.5
  sqrt(W eps32)) S``, ``W`` the detector's width: the chain's rounding, and the derivative's own change as float32's
  rounding moves an edge by about ``W eps32`` pixels, the exact disc coverage rising as a power 3/2 past a pixel's side
  (the camera's reverse-mode gate T3, ``tests/sensing/reverse_gates.py``).
- **central differences**, float64, from ``h`` (a relative 1e-4 of the parameter) to ``h/16``: the order ``p`` measured
  from the last three levels and clamped to [1/2, 2] (the camera makes the objective only C^(1,1/2)), the truncation of
  the finest ``tau = |D(h/8) - D(h/16)| / (2^p - 1)``, its rounding ``rho = 3 sigma_J / (h/16)`` with ``sigma_J`` the
  objective's own noise (ECnoise: fourth differences of seven values), never under the objective's rounding bound (its
  terms' absolute sum times eps, the positions rounded as the run holds them), the line ``3 tau + 3 rho``; where the
  second difference falls as ``h`` rather than ``h^2`` (a kink inside ``h``) the line is the kink's, ``|D2(h/16)| /
  (h/8) + 3 rho`` (``camera_loop_cases.difference_verdict``). A parameter whose line is above a tenth of its derivative
  does not discriminate, and the test says so.
- **BVLS**: camera spike R2's line, sixteen times the least-squares perturbation bound of the free currents
  (``tests/control/allocation_draws.py``), on the rows the law solved (the controller keeps them, ``state_rows``). They
  are the coils' model in the run's arithmetic: in float32 they differ from float64's by 1.6e-5 to 3.7e-5 relative (the
  multipole's sum, rounded), a perturbation some 300 times what the line allows for, and two float32 evaluations of them
  differ by up to 5e-6 (op by op against compiled), so the reference is made from the rows the law had, never made
  again in float64 or by another program.
- **the two modes**: the writes within ``16 eps N_t`` of the limit, ``N_t`` the steps run: the host's arithmetic and the
  compiled step's differ by rounding, which the loop carries forward.

**The limit's floor.** At frame period, controller period and latencies of one, one, nothing and nothing, the camera
loop is the true-state loop with its measurement moved by the tracker's error ``delta_n`` and its force by what the rows
at the measured point miss at the true one, ``dF_n``. Unsaturated, the allocation makes the force asked exactly, so the
true-state loop is linear (a double integrator closed by the PID, which the test checks against the run), and the gap
is bounded by ``|h_y|_1 max |delta| + |h_F|_1 max |dF|``, ``h_y`` and ``h_F`` that linear loop's impulse responses from
the measurement and the force to the position: computed, not fitted."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

CONTROL = str(Path(__file__).resolve().parents[1] / "control")
if CONTROL not in sys.path:
    sys.path.append(CONTROL)

import camera_loop_cases as cases  # noqa: E402
from allocation_draws import Problem, verdict  # noqa: E402

from mime.build import RunFailed  # noqa: E402
from mime.units import ms  # noqa: E402

jnp: Any = jax.numpy
EX = cases.example()
NAMES = (*cases.GAINS, cases.FX)
#: The true-state arm reads no camera, so its controller holds no believed focal length.
STATE_NAMES = cases.GAINS
#: The run's relative central-difference step, and the spacing of the noise table, in float64.
RELATIVE_H, NOISE_SPACING = 1e-4, 1e-9
#: A derivative is judged by its central differences only where their line is under this share of it.
DISCRIMINATES = 0.1


#: Each test that reads :func:`measured` is marked ``fixture_first`` (``tests/_fixture_first.py``): the draw's run and
#: its derivatives, about a minute a dtype to build and compile, are paid once a job, on one worker.


@dataclass
class Measured:
    """One draw's run in one dtype: the objective, its reverse and forward derivatives, the per-step absolute sums,
    and the run's kept outputs (float64 arrays)."""

    dtype: str
    gains: dict[str, float]
    step: tuple[float, float]
    value: float
    reverse: dict[str, float]
    forward: dict[str, float]
    absum: dict[str, float]
    out: dict[str, dict[str, np.ndarray]]
    theta: dict[str, float]
    evaluate: Any


@cache
def measured(dtype: str, seed: int = cases.SEED, arm: str = "camera") -> Measured:
    """The draw ``seed``'s run in ``dtype``, the camera arm or the true-state arm (``arm``), with its derivatives."""
    gains, step = cases.drawn(seed)
    with cases.in_dtype(dtype), cases.looser_bounds(cases.planted("limits")), \
            cases.stopped_branch(cases.planted("stop_branch")):
        built = cases.leafy(EX.experiment(arm=arm, step_x=step[0], step_y=step[1], **gains), gains)
        theta = cases.theta_of(built, gains)
        forward = cases.objective(built, step, checkpoint=False)
        stop = math.ceil(math.sqrt(cases.STEPS)) if cases.planted("reverse_chunk") else 0
        backward = cases.objective(built, step, checkpoint=True, stop_last=stop)
        value, grad = jax.jit(jax.value_and_grad(lambda th: backward(th)[0]))(theta)
        tangents, (_, out) = jax.jit(jax.jacfwd(lambda th: forward(th)[1][0], has_aux=False))(theta), \
            jax.jit(lambda th: forward(th)[1])(theta)
        once = jax.jit(lambda th: forward(th)[0])

    def evaluate(name: str, value_: float) -> float:
        with cases.in_dtype(dtype):
            moved = {**theta, name: jnp.asarray(value_, dtype=theta[name].dtype)}
            return float(once(moved))

    return Measured(dtype, gains, step, float(value), {k: float(v) for k, v in grad.items()},
                    {k: float(np.sum(np.asarray(v, dtype=np.float64))) for k, v in tangents.items()},
                    {k: float(np.sum(np.abs(np.asarray(v, dtype=np.float64)))) for k, v in tangents.items()},
                    {node: {f: np.asarray(v, dtype=np.float64) for f, v in fields.items()} for node, fields in out.items()},
                    {k: float(v) for k, v in theta.items()}, evaluate)


def eps(dtype: str) -> float:
    return float(np.finfo(dtype).eps)


# -- reverse mode against forward mode ----------------------------------------------------------------------------

@pytest.mark.fixture_first
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_reverse_mode_is_forward_mode_through_the_whole_loop(dtype):
    run = measured(dtype)
    missed = {}
    for name in NAMES:
        line = 16 * eps(dtype) * cases.STEPS * run.absum[name]
        gap = abs(run.reverse[name] - run.forward[name])
        if not (math.isfinite(run.reverse[name]) and gap <= line):
            missed[name] = (run.reverse[name], run.forward[name], gap / line if line else math.inf)
    assert not missed, (dtype, run.gains, run.step, missed)


# -- float32 against float64 --------------------------------------------------------------------------------------

@pytest.mark.fixture_first
def test_float32_is_judged_against_float64_at_the_same_draw():
    f64, f32 = measured("float64"), measured("float32")
    e32, width = eps("float32"), EX.WIDTH
    missed = {}
    if abs(f32.value - f64.value) > 16 * e32 * cases.STEPS * f64.value:
        missed["J"] = (f32.value, f64.value)
    for name in NAMES:
        line = (16 * cases.STEPS * e32 + 1.5 * math.sqrt(width * e32)) * f64.absum[name]
        gap = abs(f32.reverse[name] - f64.reverse[name])
        if not (math.isfinite(f32.reverse[name]) and gap <= line):
            missed[name] = (f32.reverse[name], f64.reverse[name], gap / line if line else math.inf)
    assert not missed, (f64.gains, f64.step, missed)


# -- central differences, their order measured --------------------------------------------------------------------

def quantum(run: Measured) -> float:
    """The objective's rounding bound: ``eps`` times its terms' absolute sum, each term's miss ``p - r`` carrying the
    rounding of the absolute positions it is the difference of (about 0.03 m, against a miss of millimetres)."""
    with cases.in_dtype("float64"):
        track = np.asarray(cases.reference_track(EX, run.step, cases.STEPS, jnp.float64))
    poses = run.out["sphere"]["pose"][:, :3]
    miss = np.linalg.norm(poses - track, axis=1)
    held = np.linalg.norm(poses, axis=1) + np.linalg.norm(track, axis=1)
    return eps(run.dtype) * float(np.sum(miss * miss + 2.0 * miss * held)) / 1e-6 / cases.STEPS


def ladder(run: Measured, name: str) -> dict[str, Any]:
    """The central differences of ``name`` at ``h .. h/16``, the measured order, the line and its verdict."""
    base = run.theta[name]
    h = RELATIVE_H * abs(base)
    value = run.value
    rows = []
    for k in range(5):
        step = h / 2 ** k
        up, down = run.evaluate(name, base + step), run.evaluate(name, base - step)
        rows.append(((up - down) / (2 * step), up - 2 * value + down, step))
    spacing = NOISE_SPACING * abs(base)
    noise = cases.ecnoise([run.evaluate(name, base + k * spacing) for k in range(-3, 4)])
    found = cases.difference_verdict(rows, noise, quantum(run), eps(run.dtype))
    return {**found, "cd": [r[0] for r in rows], "reverse": run.reverse[name], "noise": noise,
            "miss": abs(run.reverse[name] - found["cd"]),
            "discriminates": found["line"] <= DISCRIMINATES * abs(found["cd"])}


@pytest.mark.fixture_first
def test_the_derivative_meets_central_differences_whose_order_is_measured():
    run = measured("float64")
    rows = {name: ladder(run, name) for name in NAMES}
    judged = {name: row for name, row in rows.items() if row["discriminates"]}
    assert judged, ("no parameter's central differences discriminate: no evidence", rows)
    missed = {name: row for name, row in judged.items() if not row["miss"] <= row["line"]}
    assert not missed, (run.gains, run.step, missed)


# -- the allocation against SciPy's BVLS, every tick --------------------------------------------------------------

@pytest.mark.fixture_first
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_every_ticks_allocation_is_scipys_bvls_at_r2s_line(dtype):
    controller = measured(dtype).out["controller"]
    ticks = np.arange(0, cases.STEPS, 5)
    found: list[str] = []
    judged = 0
    for t in ticks:
        lost = controller["state_lost"][t] > (controller["state_lost"][t - 1] if t else 0.0)
        if lost:
            continue
        rows, wanted = controller["state_rows"][t], controller["state_wanted"][t]
        problem = Problem("navion_2", "force", rows, wanted, np.full(3, 1e-3), EX.LIMIT)
        found += [f"tick at step {t}: {problem_}" for problem_ in verdict(problem, controller["write_currents"][t], dtype)]
        judged += 1
    assert judged >= 30, f"only {judged} ticks judged"
    assert not found, found[:5]


# -- the two modes ------------------------------------------------------------------------------------------------

def test_the_callback_and_graph_modes_write_the_same_currents():
    gains, step = cases.drawn(cases.SEED)
    steps = 50
    with cases.in_dtype("float64"):
        callback = EX.experiment(mode="callback", step_x=step[0], step_y=step[1], **gains).build()
        written_cb = np.asarray(callback.run(ms(steps), history=True).history["ports:coils"]["written_currents"])
        with cases.looser_bounds(cases.planted("mode_bounds")):  # the law is made again inside the traced step
            graph = EX.experiment(step_x=step[0], step_y=step[1], **gains).build()
            written = np.asarray(graph.run(ms(steps), history=True).history["ports:coils"]["written_currents"])
    assert any("none pass through it in mode 2" in line for line in callback.report.controller)
    line = 16 * eps("float64") * steps * EX.LIMIT
    assert np.max(np.abs(written - written_cb)) <= line, (np.max(np.abs(written - written_cb)), line)
    assert np.any(np.abs(written) >= EX.LIMIT * (1 - 1e-9)), "the draw holds no coil at its bound: the plant is inert"


# -- the lightly damped loop's lost target ------------------------------------------------------------------------

@cache
def damped_run(seed: int) -> Any:
    """The lightly damped loop of draw ``seed`` (``kd`` times 0.3, the step within 30 degrees of +y, 4.5 to 6 mm) in
    float64, run for the example's 200 steps; where the run stops non-finite, ``RunFailed``."""
    gains, _ = cases.drawn(seed)
    gains = {**gains, "kd": 0.3 * gains["kd"]}
    rng = np.random.default_rng(8000 + seed)
    angle, size = math.pi / 2 + rng.uniform(-math.pi / 6, math.pi / 6), rng.uniform(4.5e-3, 6e-3)
    step = (size * math.cos(angle), size * math.sin(angle))
    with cases.in_dtype("float64"):
        return EX.experiment(step_x=step[0], step_y=step[1], **gains).build().run(ms(cases.STEPS), history=True)


def test_the_lightly_damped_loops_lost_target_is_held_and_flagged():
    """R2's lightly damped loop (``kd`` times 0.3) overshoots out of the frame: its soft centroid was 0/0 from step 51.
    The guarded tracker holds its last point, the law writes no current, and the run stays finite. The step is drawn
    within 30 degrees of +y (the frame's short side, 9 mm from its centre), where the overshoot leaves the frame on
    every draw measured (45 to 65 steps in, ten draws on a laptop); R2's own directions leave it on about half.

    The plant (``unguarded``) fails this in one of two ways, and each says so: the loop goes non-finite and the run
    stops (13 draws of 20), or it stays finite and the first tick's empty frame is not flagged lost (the other 7). A
    clean draw too gentle to leave the frame is a third thing, and its message says that."""
    try:
        result = damped_run(cases.SEED)
    except RunFailed as failed:
        assert failed.codes() == ["run.non-finite"], failed.codes()
        pytest.fail("guard removed: non-finite (the run stopped on run.non-finite)")
    history = result.history
    controller = history["controller"]
    pose = np.asarray(history["sphere"]["pose"])
    assert np.all(np.isfinite(pose)) and all(np.all(np.isfinite(np.asarray(v))) for v in controller.values()), \
        "guard removed: non-finite"
    lost = np.asarray(controller["state_lost"])
    assert lost[-1] >= 1, "guard removed: the first tick's empty frame was not flagged lost"
    assert lost[-1] > 1, "draw too gentle to test the guard: the target never left the frame after the first tick"
    ticks = np.flatnonzero(np.diff(lost) > 0) + 1
    points = np.asarray(controller["state_point"])
    held = [t for t in ticks if np.array_equal(points[t], points[t - 1])]
    assert held == list(ticks), "a lost tick moved the measured point"
    written = np.asarray(controller["write_currents"])
    assert np.all(written[ticks] == 0.0), "a lost tick wrote currents"
    assert float(result.final["controller"]["state_first_lost"]) == 0.0, "the first tick reads no frame yet"


def test_a_lost_tick_freezes_the_integral():
    """The PID's integral is held across a lost tick (``pid(..., frozen=)``), so that a frame that shows nothing adds no
    error to it. The first tick always reads the camera's empty initial frame and is lost, with the reference a step
    from the held point: a law that integrates it anyway has an integral of the step times the period after one tick, and
    this one has exactly zero. The damped run's later lost ticks (where the target left the frame) are held alike. Every
    draw has the first lost tick, so the plant (``integral_unfrozen``) fails on every run."""
    history = damped_run(cases.SEED).history["controller"]
    lost = np.concatenate([[0.0], np.asarray(history["state_lost"])])
    integral = np.concatenate([np.zeros((1, 3)), np.asarray(history["state_integral"])])
    flagged = np.flatnonzero(np.diff(lost) > 0) + 1
    assert flagged.size and flagged[0] == 1, "the first tick's empty frame was not flagged lost: no lost tick to test"
    moved = np.flatnonzero(np.any(np.diff(integral, axis=0) != 0.0, axis=1)) + 1
    assert moved.size, "the integral never moved on a tick that read a frame: the check reads the wrong field"
    changed = [int(t) for t in flagged if np.any(integral[t] != integral[t - 1])]
    assert not changed, f"the integral moved on lost ticks at steps {changed}: it is not frozen"


# -- the limit: latency to nothing, the frame rate to the step rate -----------------------------------------------

RUNGS = ((4, 4, 1), (2, 2, 1), (1, 1, 1), (1, 1, 0))
LIMIT_PLANTS = {"extra_port": {"port_latency_ms": 1}, "extra_controller": {"latency_ms": 1},
                "extra_command": {"command_latency_ms": 1}}


def gentle(seed: int) -> tuple[dict[str, float], tuple[float, float]]:
    """A draw whose loop binds no coil: R2's gains, a step of 0.5 to 1.5 mm in a drawn direction."""
    rng = np.random.default_rng(9000 + seed)
    defaults = {"kd": 80.0, "ki": 3000.0, "kp": 2500.0}
    gains = {name: float(defaults[name] * rng.uniform(0.8, 1.2)) for name in cases.GAINS}
    angle, size = rng.uniform(0, 2 * math.pi), rng.uniform(0.5e-3, 1.5e-3)
    return gains, (size * math.cos(angle), size * math.sin(angle))


def trajectory(arm: str, gains: dict[str, float], step: tuple[float, float], **timing: float) -> Any:
    built = EX.experiment(arm=arm, step_x=step[0], step_y=step[1], settle=1e-3, **gains, **timing).build()
    history = built.run(ms(cases.STEPS), history=True).history
    return built, np.asarray(history["sphere"]["pose"])[:, :3], history


def linear_loop(gains: dict[str, float], steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """The true-state loop's impulse responses to the position (one axis) from the measurement, from the force and
    from a push of the position itself, and the loop as a function of the reference, for the check that it is the run:
    a double integrator (the rigid body's step, velocity then position) closed every base step by the PID, its tick
    reading the state at the start of the step it acts on."""
    kp, ki, kd, dt, m = gains["kp"], gains["ki"], gains["kd"], 1e-3, EX.MASS
    zero = np.zeros(steps)

    def loop(reference: np.ndarray, offset: np.ndarray, force: np.ndarray, push: np.ndarray = zero) -> np.ndarray:
        x = v = integral = 0.0
        previous = 0.0
        found = np.zeros(steps)
        for n in range(steps):
            y = x + offset[n]
            error = reference[n] - y
            integral += error * dt
            wanted = m * (kp * error + ki * integral - kd * (y - previous) / dt)
            previous = y
            v += (wanted + force[n]) / m * dt
            x += v * dt + push[n]
            found[n] = x
        return found

    impulse = np.zeros(steps)
    impulse[0] = 1.0
    return (np.abs(loop(zero, impulse, zero)), np.abs(loop(zero, zero, impulse)),
            np.abs(loop(zero, zero, zero, impulse)), loop)


def test_the_loop_meets_the_true_state_run_as_latency_goes_to_nothing_and_frames_to_every_step():
    gains, step = gentle(cases.SEED)
    with cases.in_dtype("float64"):
        _, truth, truth_history = trajectory("state", gains, step, frame_ms=1, rate_ms=1, latency_ms=0)
        gaps, chains = [], []
        for frame, rate, latency in RUNGS:
            extra = LIMIT_PLANTS.get(cases.PLANT, {}) if (frame, rate, latency) == RUNGS[-1] else {}
            built, camera, history = trajectory("camera", gains, step, frame_ms=frame, rate_ms=rate,
                                                **{"latency_ms": latency, **extra})
            written = np.asarray(history["controller"]["write_currents"])
            assert not np.any(np.abs(written) >= EX.LIMIT * (1 - 1e-9)), "a coil reached its bound: depth is not held"
            gaps.append(float(np.max(np.linalg.norm(camera - truth, axis=1))))
            chains.append(next(line for line in built.report.controller if "imaging.frames -> coils.currents" in line))
    assert all(later < earlier for earlier, later in zip(gaps, gaps[1:])), gaps
    assert "0 base steps (0.0 to 0.0 s)" in chains[-1], chains[-1]
    # The floor at the last rung, from the tracker's own error (against the pose of the frame it read, whatever that
    # frame's age) and the linear loop's responses.
    points = np.asarray(history["controller"]["state_point"])
    # The frame read at the tick of step n was made at step n - 1 - age, and shows the pose after that step.
    made = np.arange(cases.STEPS) - 1 - np.asarray(history["controller"]["state_age"]).astype(int)
    seen = made >= 0
    delta = float(np.max(np.linalg.norm(points[seen] - camera[made[seen]], axis=1)))
    currents = np.asarray(history["controller"]["write_currents"])
    miss = max(float(np.linalg.norm((cases.force_rows(EX, camera[n - 1]) - cases.force_rows(EX, points[n])) @ currents[n]))
               for n in range(1, cases.STEPS))
    h_y, h_f, h_x, loop = linear_loop(gains, cases.STEPS)
    floor = float(np.sum(h_y)) * delta + float(np.sum(h_f)) * miss
    # The premise: unsaturated, the true-state run is the linear loop, axis by axis, to its rounding. In exact
    # arithmetic they are one (the allocation makes the force asked, the body's step is the loop's, the moment holds its
    # direction); the run rounds positions of about 0.03 m, sixteen operations a step at eps64 of the largest, which
    # reach the position through the loop as a push of the state and as an offset of the measurement, and the force it
    # makes, sixteen operations of the largest sum it is made of (each coil's force at its current), through the force's
    # response. The reference is the run's own float64 (outside the float64 block it is float32, whose rounding of
    # 0.03 m is 1e-9 m).
    with cases.in_dtype("float64"):
        times = (np.arange(cases.STEPS)) * 1e-3
        reference = np.asarray(EX.reference(jnp.asarray(times), step, 1e-3)).T - np.asarray(EX.START)
    largest = float(np.max(np.abs(truth)))
    held = truth_history["controller"]
    force = float(np.max(np.einsum("nij,nj->ni", np.abs(np.asarray(held["state_rows"])),
                                   np.abs(np.asarray(held["write_currents"])))))
    premise = 16 * eps("float64") * (largest * float(np.sum(h_x) + np.sum(h_y)) + force * float(np.sum(h_f)))
    for axis in range(3):
        modelled = loop(reference[:, axis], np.zeros(cases.STEPS), np.zeros(cases.STEPS))
        assert np.max(np.abs(modelled - (truth[:, axis] - EX.START[axis]))) <= premise, (axis, premise)
    assert gaps[-1] <= floor, (gaps, floor, delta, miss)
