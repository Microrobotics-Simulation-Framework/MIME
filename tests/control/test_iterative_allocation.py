"""The iterative allocation, ``mime.control.iterative_allocation``: against the exact path where both apply, against
SciPy's BVLS above the exact path's size, inverse kinematics against the library's, its implicit derivative against
central differences and reverse mode against forward, float32 against float64, and "converged" held to what it says.

**Converged means converged.** Every test that compares commands first asks the solve's own account, and judges it:
``converged`` must be true, its ``stationarity`` within the declared line :data:`LINE` (read when this module is
imported, before any plant), and the commands within the derived distance of the oracle's: the oracle's own line (the
least-squares perturbation bound of the exact path's and BVLS's, ``allocation_draws.line``) plus the natural
residual's error bound, ``|v - v*| <= (1 + L) / mu |pi|`` in the commands scaled to unit columns (``mu`` and ``L`` the
scaled Hessian's least and largest eigenvalues, ``pi`` the projected gradient the stationarity measures). A solve that
claims to converge where it did not is caught by one of the three.

**Its plants** (``ITERATIVE_PLANT``): ``early``, the solve stopped after two iterations (a solve within its tolerance
by then goes on to its local phase, as the clean one does); ``loose``, its tolerance loosened to one with no local
phase, so that it stops at its start and claims to have converged (with the local phase a loose tolerance is only an
earlier start of Newton's full steps, which on an affine model end at the minimiser: no wrong claim to catch); ``gauss_newton``, the root's gradient taken with its Jacobian
held, so that its linearisation is the Gauss-Newton Hessian, which is wrong wherever the rows are not met (an
unreachable target); ``bounds``, the root's bounds held, so that a command at a bound has no derivative in it;
``unpolished``, no polishing (``_POLISHING_STEPS`` zero), so that a warm start near the minimiser returns the first
iterate the tolerance admits, where the derivative of a weakly regularised problem is not the minimiser's. Each must be
caught on every run: ``early`` and ``loose`` by the agreement tests, the other three by the derivative's."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from functools import cache
from typing import Any

import jax
import numpy as np
import pytest
from allocation_draws import SLACK, Problem, bvls, line, problems, residual_line, tie_window, unique
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import mime.control._iterative as module
from mime.control import exact_allocation, iterative_allocation

jnp: Any = jax.numpy
lax: Any = jax.lax
PLANT = os.environ.get("ITERATIVE_PLANT", "")
#: The declared tolerance, read before any plant replaces the module's.
LINE = dict(module.TOLERANCE)
OWNER = "iterative_tests.active_set"
TRUE_GRADIENT, TRUE_BOUNDS = module._gradient, module._bounds


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def _gauss_newton(rows: Callable[[Any], Any], u: Any) -> Any:
    return lax.stop_gradient(jax.jacfwd(rows)(u)).T @ rows(u)


@contextlib.contextmanager
def planted(names: tuple[str, ...]) -> Iterator[None]:
    """The plant ``PLANT`` names, where it is one of ``names``: what the test it is measured on sees."""
    with pytest.MonkeyPatch.context() as patch:
        if PLANT in names:
            if PLANT == "early":
                patch.setattr(module, "ITERATIONS", 2)
            elif PLANT == "loose":
                patch.setattr(module, "TOLERANCE", dict.fromkeys(module.TOLERANCE, 1.0))
                patch.setattr(module, "_POLISHING_STEPS", 0)
            elif PLANT == "gauss_newton":
                patch.setattr(module, "_gradient", _gauss_newton)
            elif PLANT == "bounds":
                patch.setattr(module, "_bounds", lambda low, high: (lax.stop_gradient(low), lax.stop_gradient(high)))
            elif PLANT == "unpolished":
                patch.setattr(module, "_POLISHING_STEPS", 0)
        yield


# -- the agreement, and what "converged" claims ----------------------------------------------------------------------

def scaled_problem(matrix: np.ndarray, target: np.ndarray, scales: np.ndarray, effort: float | None,
                   limit: float) -> tuple[np.ndarray, np.ndarray]:
    """The rows over their scales, with the effort's rows below them (towards zero), in float64."""
    a, b = matrix / scales[:, None], target / scales
    if effort is None:
        return a, b
    k = matrix.shape[1]
    return np.vstack([a, np.eye(k) / effort]), np.concatenate([b, np.zeros(k)])


def claim_problems(found: Any, a: np.ndarray, b: np.ndarray, reference: np.ndarray, oracle_line: float,
                   dtype: str) -> list[str]:
    """What is wrong with the solve's claim against the oracle's commands ``reference`` (see the module)."""
    problems_: list[str] = []
    commands = np.asarray(found.commands, dtype=np.float64)
    stationarity = float(found.stationarity)
    if not bool(found.converged):
        problems_.append(f"not converged: stationarity {stationarity:.3g}, {int(found.iterations)} iterations")
    if stationarity > LINE[np.dtype(dtype)]:
        problems_.append(f"stationarity {stationarity:.3g} past the declared line {LINE[np.dtype(dtype)]:.3g}")
    columns = np.linalg.norm(a, axis=0)
    unit = a / columns
    values = np.linalg.eigvalsh(unit.T @ unit)
    size = float(np.linalg.norm(a @ commands - b)) + float(np.sum(columns * np.abs(commands)))
    bound = (1.0 + values[-1]) / values[0] * np.sqrt(len(commands)) * stationarity * size / columns
    miss = np.abs(commands - reference)
    held = oracle_line + SLACK * bound
    if np.any(miss > held):
        problems_.append(f"commands {commands} from the oracle's {reference}: {miss} past {held}")
    return problems_


def iterative(a: np.ndarray, b: np.ndarray, limit: float, dtype: str) -> Any:
    held = [jnp.asarray(v, dtype=dtype) for v in (a, b)]
    return iterative_allocation(lambda u: held[0] @ u - held[1], jnp.zeros((a.shape[1],), dtype=dtype),
                                lower=-limit, upper=limit, owner=OWNER)


def efforts() -> st.SearchStrategy[float | None]:
    """None (unregularised) half the time, else a scale from 1 to 100 A."""
    return st.one_of(st.none(), st.floats(1.0, 100.0))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@settings(max_examples=100, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(problem=problems(), effort=efforts())
def test_the_iterative_path_meets_the_exact_path_where_both_apply(dtype, problem, effort):
    """On the arrays MIME ships (three to eight coils, the exact path's size), the iterative path's commands are the
    exact path's within the derived distance, and its claim holds. An unregularised problem whose minimiser is not
    unique (more coils than rows) takes an effort of 30 A, as ``ctx.allocator`` requires of the iterative path."""
    effort = effort if effort is not None or unique(problem, dtype) else 30.0
    a, b = scaled_problem(problem.matrix, problem.target, problem.scales, effort, problem.limit)
    limit = float(np.float32(problem.limit)) if dtype == "float32" else problem.limit
    with in_dtype(dtype), planted(("early", "loose")):
        found = iterative(a, b, limit, dtype)
    with in_dtype(dtype):
        exact = exact_allocation(jnp.asarray(a, dtype=dtype), jnp.asarray(b, dtype=dtype), lower=-limit, upper=limit,
                                 scales=jnp.ones((a.shape[0],), dtype=dtype), owner=OWNER)
    reference = np.asarray(exact.commands, dtype=np.float64)
    rounded = Problem(problem.system, problem.kind, a, b, np.ones(a.shape[0]), limit)
    found_problems = claim_problems(found, a, b, reference, line(rounded, reference, dtype), dtype)
    if found_problems and tied(rounded, np.asarray(found.commands, dtype=np.float64), reference, dtype):
        # The exact path ties faces whose residuals are within its window and takes the least effort among them
        # (``allocation_draws``' verdict): float32 resolves residuals more finely than the window, so its commands
        # can be another face's by whole amperes (a stress run's CardioMag draw: 0.51 A off, its residual 3.6e-5 over
        # the minimum's within a window of 1.1e-4, the iterative path on BVLS's to 3.7e-6). There it is no oracle of
        # the minimiser, and SciPy's BVLS on the same rows is, at its own line.
        reference = bvls(rounded, dtype)
        found_problems = claim_problems(found, a, b, reference, line(rounded, reference, dtype), dtype)
    assert found_problems == []


def tied(problem: Problem, found: np.ndarray, exact: np.ndarray, dtype: str) -> bool:
    """Whether the exact path's residual is within its tie windows (at its commands and at ``found``) and the
    residual's rounding of the iterative path's: two faces the exact path cannot tell apart by its own rule."""
    a, b = problem.scaled(dtype)
    gap = abs(float(np.linalg.norm(a @ exact - b)) - float(np.linalg.norm(a @ found - b)))
    return gap <= residual_line(problem, dtype) + tie_window(problem, exact, dtype) + tie_window(problem, found, dtype)


def synthetic(rng: np.random.Generator, commands: int, rows: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Gaussian rows of singular values 0.3 to 3, a target of a drawn direction whose size is a tenth to four times
    what the limit reaches, and a limit from 5 to 50 A."""
    left, _ = np.linalg.qr(rng.normal(size=(rows, commands)))
    right, _ = np.linalg.qr(rng.normal(size=(commands, commands)))
    matrix = left @ np.diag(np.exp(rng.uniform(np.log(0.3), np.log(3.0), commands))) @ right.T
    limit = float(rng.uniform(5.0, 50.0))
    direction = rng.normal(size=rows)
    reach = float(np.linalg.norm(matrix, axis=0).sum()) * limit
    return matrix, direction / np.linalg.norm(direction) * reach * float(np.exp(rng.uniform(np.log(0.1),
                                                                                             np.log(4.0)))), limit


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@settings(max_examples=60, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2 ** 32 - 1), commands=st.integers(9, 16), extra=st.integers(0, 6),
       effort=efforts())
def test_the_iterative_path_meets_bvls_above_the_exact_paths_size(dtype, seed, commands, extra, effort):
    """Nine to sixteen commands, past the exact path's eight: SciPy's BVLS on the same rows (tall, or regularised) is
    the oracle, at camera spike R2's line, and the solve's claim holds."""
    from scipy.optimize import lsq_linear

    rng = np.random.default_rng(seed)
    matrix, target, limit = synthetic(rng, commands, commands + extra)
    a, b = scaled_problem(matrix, target, np.ones(len(target)), effort, limit)
    a = np.asarray(a, dtype=dtype).astype(np.float64)
    b = np.asarray(b, dtype=dtype).astype(np.float64)
    limit = float(np.float32(limit)) if dtype == "float32" else limit
    solved = lsq_linear(a, b, bounds=(-limit, limit), method="bvls", max_iter=3 ** 8)
    assert solved.status > 0, "the oracle did not converge"
    reference = np.asarray(solved.x)
    with in_dtype(dtype), planted(("early", "loose")):
        found = iterative(a, b, limit, dtype)
    rounded = Problem("synthetic", "rows", a, b, np.ones(a.shape[0]), limit)
    assert claim_problems(found, a, b, reference, line(rounded, reference, dtype), dtype) == []


def test_the_plants_of_the_claim_are_each_caught_at_one_problem(monkeypatch):
    """At one problem whose solve takes more than two iterations, the early stop leaves it unconverged and the loose
    tolerance, with no local phase, claims a stationarity past the line: the two plants' mechanisms, held here; their
    rates are measured."""
    rng = np.random.default_rng(11)
    matrix, target, limit = synthetic(rng, 12, 14)
    with in_dtype("float64"):
        clean = iterative(matrix, target, limit, "float64")
        assert int(clean.iterations) > 2 and bool(clean.converged)
        reference = np.asarray(clean.commands)
        monkeypatch.setattr(module, "ITERATIONS", 2)
        early = iterative(matrix, target, limit, "float64")
        monkeypatch.setattr(module, "ITERATIONS", module.ITERATIONS)
        monkeypatch.undo()
        monkeypatch.setattr(module, "TOLERANCE", dict.fromkeys(module.TOLERANCE, 1.0))
        monkeypatch.setattr(module, "_POLISHING_STEPS", 0)
        loose = iterative(matrix, target, limit, "float64")
    problem = Problem("synthetic", "rows", matrix, target, np.ones(len(target)), limit)
    assert claim_problems(early, matrix, target, reference, line(problem, reference, "float64"), "float64") != []
    assert claim_problems(loose, matrix, target, reference, line(problem, reference, "float64"), "float64") != []


def test_a_solve_that_cannot_converge_in_its_cap_says_so():
    """Capped at one iteration from far away, the solve returns its iterate inside the box and says it did not
    converge, with a stationarity past the line: never a silent value."""
    rng = np.random.default_rng(3)
    matrix, target, limit = synthetic(rng, 10, 12)
    with in_dtype("float64"), pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "ITERATIONS", 1)
        found = iterative(matrix, target * 3.0, limit, "float64")
    commands = np.asarray(found.commands)
    assert not bool(found.converged) and float(found.stationarity) > LINE[np.dtype("float64")]
    assert np.all(np.abs(commands) <= limit)


def curved_rows(u: Any) -> Any:
    """Rows no point meets, in a curved valley (Rosenbrock's, with a turn and a product added)."""
    return jnp.stack([10.0 * (u[1] - u[0] ** 2), 1.0 - u[0], 2.0 * (u[2] - jnp.sin(u[1])), 0.5 * u[0] * u[2] - 0.7])


@pytest.mark.parametrize("kind", ["affine", "nonlinear"])
def test_a_solve_whose_tolerance_cannot_be_met_ends_within_its_cap_and_says_so(kind):
    """With the tolerance below zero, which no iterate can meet (at zero, the curved valley's solve met it: its
    gradient rounds to exactly zero there), the solve still ends: within ``ITERATIONS`` (its local phase never
    starts), its commands finite and inside the box, ``converged`` false. On an affine problem and on a curved valley's
    rows, whose line searches are the nonlinear ones: every loop of the solve is bounded, and at its bound the solve
    says so."""
    with in_dtype("float64"), pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "TOLERANCE", dict.fromkeys(module.TOLERANCE, -1.0))
        if kind == "affine":
            matrix, target, limit = synthetic(np.random.default_rng(3), 10, 12)
            found = iterative(matrix, target * 3.0, limit, "float64")
        else:
            limit = 2.0
            found = iterative_allocation(curved_rows, jnp.asarray([-1.5, 1.0, 0.0]), lower=-limit, upper=limit,
                                         owner=OWNER)
    commands = np.asarray(found.commands)
    low, high = -limit, limit
    assert not bool(found.converged) and int(found.iterations) <= module.ITERATIONS, int(found.iterations)
    assert np.all(np.isfinite(commands)) and np.all(commands >= low) and np.all(commands <= high)


# -- the arm: inverse kinematics and the implicit derivative ------------------------------------------------------

#: The arm's objective: link_6's position over 1 mm and its orientation's rotation vector over 10 mrad; the effort's
#: scale on unreachable targets (rad), and the half-width of the slew box around the last joints (rad).
POSITION_SCALE, TURN_SCALE, EFFORT, BOX = 1e-3, 1e-2, 20.0, 0.05


@cache
def arm() -> tuple[Any, int, np.ndarray, np.ndarray]:
    from believed_cases import model_of
    from chain_cases import AR4, kinematic_rpm

    from mime.core.multibody import read_urdf

    tree = read_urdf(AR4).tree
    return (model_of(kinematic_rpm(), "rpm.arm"), tree.frame_names.index("link_6"),
            np.asarray(tree.articulation.lower, dtype=float), np.asarray(tree.articulation.upper, dtype=float))


def arm_solve(tangent: Any, target: Any, last: Any, reachable: bool) -> Any:
    """The arm's allocation at ``target`` moved by ``tangent`` (its position in mm, its turn in 10 mrad, and ``last``
    in mrad), regularised towards ``last`` where it is unreachable, within the joints' limits and a box of ``BOX``
    around ``last``, from ``last``."""
    from mime.core import quaternion
    from mime.core.multibody._se3 import relative_turn, rotation_vector

    model, frame, lower, upper = arm()
    dtype = target.dtype
    moved_last = last + tangent[6:] * 1e-3
    goal = jnp.concatenate([target[:3] + tangent[:3] * POSITION_SCALE,
                            quaternion.multiply(quaternion.exponential(tangent[3:6] * TURN_SCALE), target[3:])])

    def rows(q: Any) -> Any:
        reached = model({"joint_positions": q}, {})["frame_poses"][frame]
        miss = jnp.concatenate([(reached[:3] - goal[:3]) / POSITION_SCALE,
                                rotation_vector(relative_turn(goal, reached), owner=OWNER) / TURN_SCALE])
        return miss if reachable else jnp.concatenate([miss, (q - moved_last) / EFFORT])

    low = jnp.maximum(jnp.asarray(lower, dtype=dtype), moved_last - BOX)
    high = jnp.minimum(jnp.asarray(upper, dtype=dtype), moved_last + BOX)
    return iterative_allocation(rows, moved_last, lower=low, upper=high, owner=OWNER)


@cache
def arm_programs(dtype: str, plant: str, reachable: bool) -> dict[str, Any]:
    """The arm's solve and its two derivatives, compiled once per dtype, plant and kind of target."""
    def commands(t: Any, a: Any, l: Any) -> Any:
        return arm_solve(t, a, l, reachable).commands

    return {"solve": jax.jit(lambda t, a, l: arm_solve(t, a, l, reachable)), "forward": jax.jit(jax.jacfwd(commands)),
            "reverse": jax.jit(jax.jacrev(commands))}


def arm_call(name: str, dtype: str, reachable: bool, *args: Any, plant: bool = True) -> Any:
    held = PLANT if plant and PLANT in ("gauss_newton", "bounds", "unpolished") else ""
    with in_dtype(dtype), planted((held,) if held else ()):
        return arm_programs(dtype, held, reachable)[name](*(jnp.asarray(a, dtype=dtype) for a in args))


def arm_draw(seed: int, reachable: bool) -> tuple[np.ndarray, np.ndarray]:
    """A target (the frame's pose at joints in the central 80 % of their ranges, moved 1.5 m where it is unreachable)
    and the last joints, 0.15 rad (normal) from those joints, within the limits, at least one of them twice the box away."""
    model, frame, lower, upper = arm()
    rng = np.random.default_rng(seed)
    joints = rng.uniform(lower + 0.1 * (upper - lower), upper - 0.1 * (upper - lower))
    noise = rng.normal(scale=0.15, size=joints.size)
    # The box (BOX) must bind a joint for the bounds' derivative to be tested: a draw whose six offsets all fall within it
    # (about 1 in 3000) has its offsets scaled up to twice the box, a transformation of the draw and no rejection of it.
    noise *= max(1.0, 2.0 * BOX / float(np.max(np.abs(noise))))
    last = np.clip(joints + noise, lower, upper)
    with in_dtype("float64"):
        target = np.array(model({"joint_positions": jnp.asarray(joints)}, {})["frame_poses"][frame])
    if not reachable:
        direction = rng.normal(size=3)
        target[:3] += 1.5 * direction / np.linalg.norm(direction)
    return target, last


@pytest.mark.memory_heavy
@pytest.mark.parametrize("reachable", [True, False], ids=["reachable", "unreachable"])
@settings(max_examples=40, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2 ** 32 - 1))
def test_the_implicit_derivative_is_central_differences_and_reverse_mode_is_forward(reachable, seed):
    """float64: the derivative of the arm's joints in its target's position and turn and in the last joints (so in a
    slew box's bounds, which the box makes active) against central differences of reference solves (a Richardson pair,
    the solves' tolerance near the floor), within 1e-6 of its largest entry; reverse mode against forward to 1e-12.
    Unreachable targets leave the rows unmet, where the Gauss-Newton Hessian's derivative is wrong."""
    target, last = arm_draw(seed, reachable)
    zero = np.zeros(12)
    found = arm_call("solve", "float64", reachable, zero, target, last)
    assert bool(found.converged), float(found.stationarity)
    forward = np.asarray(arm_call("forward", "float64", reachable, zero, target, last))
    reverse = np.asarray(arm_call("reverse", "float64", reachable, zero, target, last))
    v, w = np.random.default_rng(seed).normal(size=(2, 12))
    v = v[:6]
    assert abs(v @ forward @ w - (reverse.T @ v) @ w) <= 1e-12 * np.linalg.norm(forward) * np.linalg.norm(v) * \
        np.linalg.norm(w)
    assert np.any(np.asarray(found.active) != 0), "the box holds no joint at a bound: the bounds' derivative is untested"
    differences = central_differences(found, forward, target, last, reachable)
    miss = float(np.max(np.abs(forward - differences)))
    assert miss <= 1e-6 * float(np.max(np.abs(differences))), (miss, float(np.max(np.abs(differences))))


def central_differences(found: Any, forward: np.ndarray, target: np.ndarray, last: np.ndarray,
                        reachable: bool) -> np.ndarray:
    """Each column of the derivative by a Richardson pair of central differences of reference solves, over the widest
    of three step sizes whose solves stay on ``found``'s branch: its active set, and commands within ten times the
    step's first-order move of its own (a local minimiser reached on another branch of a nonlinear model jumps by far
    more). A draw near a kink or a branch's end is differenced over a smaller segment, never left out. The reference
    solves are the library's own, unplanted, each converged and polished to the commands' rounding (a tolerance
    patched here would not reach them: the clean solve is compiled once, before them, and a patch after a trace is
    ignored)."""
    columns = []
    base = np.asarray(found.commands)
    for i in range(12):
        reach = float(np.max(np.abs(forward[:, i])))
        for step in (0.2, 0.02, 0.002):
            pair, kept = [], True
            for h in (step, step / 2):
                e = np.zeros(12)
                e[i] = h
                for sign in (1.0, -1.0):
                    moved = arm_call("solve", "float64", reachable, sign * e, target, last, plant=False)
                    assert bool(moved.converged), (i, sign * h, float(moved.stationarity))
                    kept = kept and np.array_equal(np.asarray(moved.active), np.asarray(found.active)) and \
                        float(np.max(np.abs(np.asarray(moved.commands) - base))) <= 10.0 * h * reach + 1e-9
                    pair.append(np.asarray(moved.commands) * sign / (2 * h))
            if kept:
                break
        assert kept, f"the solution leaves its branch within {step / 2} of the draw along input {i}, both ways"
        first, second = pair[0] + pair[1], pair[2] + pair[3]
        columns.append((4 * second - first) / 3)
    return np.stack(columns, axis=1)


def follower_solve(tangent: Any, target: Any, last: Any) -> Any:
    """A0's follower's problem: link_6's position at ``target`` over 1 mm, regularised towards ``last`` at ``EFFORT``
    rad (a weak effort, whose rows set the scaled Hessian's least eigenvalues near 1e-8 of its largest), within the
    joints' limits, from ``last``; moved by ``tangent`` (the target in mm, ``last`` in mrad)."""
    model, frame, lower, upper = arm()
    moved_last = last + tangent[3:] * 1e-3
    goal = target + tangent[:3] * POSITION_SCALE

    def rows(q: Any) -> Any:
        reached = model({"joint_positions": q}, {})["frame_poses"][frame]
        return jnp.concatenate([(reached[:3] - goal) / POSITION_SCALE, (q - moved_last) / EFFORT])

    dtype = target.dtype
    return iterative_allocation(rows, moved_last, lower=jnp.asarray(lower, dtype=dtype),
                                upper=jnp.asarray(upper, dtype=dtype), owner=OWNER)


@cache
def follower_programs(plant: str) -> dict[str, Any]:
    def commands(t: Any, a: Any, l: Any) -> Any:
        return follower_solve(t, a, l).commands

    return {"solve": jax.jit(follower_solve), "forward": jax.jit(jax.jacfwd(commands))}


def follower_call(name: str, *args: Any, plant: bool = True) -> Any:
    held = PLANT if plant and PLANT == "unpolished" else ""
    with in_dtype("float64"), planted((held,) if held else ()):
        return follower_programs(held)[name](*(jnp.asarray(a, dtype="float64") for a in args))


@settings(max_examples=40, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2 ** 32 - 1))
def test_a_warm_start_near_the_minimiser_gives_the_minimisers_derivative(seed):
    """float64, the follower's operating point: the joints read are the start and the effort's reference, and the
    target a fraction of a millimetre to two from where they put the flange, so that the solve meets its tolerance in a
    step or two. Its derivative in the target and the joints read is central differences of reference solves (a
    Richardson pair; the library's own, unplanted, each converged) within 1e-6 of its largest entry: the local phase
    takes the commands to the minimiser, at which the derivative is taken (the tolerance alone left A0's follower 4e-8
    rad from it and its derivative 8e-5 off, and a reference solve here 2e-3 rad from it, its objective 37 % above the
    minimum, where a full step first raises the objective). The differences' steps, 0.2 and 0.1 mm (and mrad), keep
    the reference solves' rounding, which smaller steps amplify (1e-5 of the derivative at 2 um), near 1e-7."""
    model, frame, lower, upper = arm()
    rng = np.random.default_rng(seed)
    joints = rng.uniform(lower + 0.1 * (upper - lower), upper - 0.1 * (upper - lower))
    direction = rng.normal(size=3)
    with in_dtype("float64"):
        flange = np.asarray(model({"joint_positions": jnp.asarray(joints)}, {})["frame_poses"][frame])[:3]
    target = flange + direction / np.linalg.norm(direction) * rng.uniform(2e-4, 2e-3)
    zero = np.zeros(9)
    found = follower_call("solve", zero, target, joints)
    assert bool(found.converged) and not np.any(np.asarray(found.active)), (float(found.stationarity), found.active)
    forward = np.asarray(follower_call("forward", zero, target, joints))
    columns = []
    for i in range(9):
        pair = []
        for h in (0.2, 0.1):
            e = np.zeros(9)
            e[i] = h
            moved = [follower_call("solve", sign * e, target, joints, plant=False) for sign in (1.0, -1.0)]
            assert all(bool(each.converged) for each in moved), (i, h, [float(each.stationarity) for each in moved])
            pair.append((np.asarray(moved[0].commands) - np.asarray(moved[1].commands)) / (2 * h))
        columns.append((4 * pair[1] - pair[0]) / 3)
    differences = np.stack(columns, axis=1)
    miss = float(np.max(np.abs(forward - differences)))
    assert miss <= 1e-6 * float(np.max(np.abs(differences))), (miss, float(np.max(np.abs(differences))))


def inverse_kinematics_rows(target: Any) -> Callable[[Any], Any]:
    """link_6's pose error from ``target``, its position over 1 mm and its turn over 10 mrad."""
    from mime.core.multibody._se3 import pose_error

    model, frame, _, _ = arm()

    def rows(q: Any) -> Any:
        reached = model({"joint_positions": q}, {})["frame_poses"][frame]
        return pose_error(target, reached, owner=OWNER) * jnp.asarray([1e3] * 3 + [1e2] * 3)

    return rows


@cache
def inverse_kinematics_program(plant: str) -> Any:
    """The arm's unregularised pose solve, compiled once per plant (``early`` is this test's), and the frame it puts
    link_6 at."""
    model, frame, lower, upper = arm()

    def solve(target: Any, last: Any) -> tuple[Any, Any]:
        found = iterative_allocation(inverse_kinematics_rows(target), last, lower=jnp.asarray(lower),
                                     upper=jnp.asarray(upper), owner=OWNER)
        return found, model({"joint_positions": found.commands}, {})["frame_poses"][frame]

    return jax.jit(solve)


@settings(max_examples=40, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2 ** 32 - 1))
def test_iterative_inverse_kinematics_meets_the_librarys_on_reachable_poses(seed):
    """Unregularised, from 0.15 rad away, with the limits but no box: the iterative solve converges; where it holds no
    joint at a limit it puts link_6 on the reachable pose to 1e-9 m, and where ``mime.core.multibody.inverse_kinematics``
    reaches the pose inside the limits from the same start, the two reach the same joints on the same branch (the AR4's
    six joints fix a pose up to its wrist's flip, away from its singular poses); where it holds a joint at a limit, it
    is stationary there, judged from the objective's own gradient (the minimiser reached from the start, which need
    not be the library's: see the comment below)."""
    from mime.core.multibody import compose, inverse, inverse_kinematics, read_urdf

    from chain_cases import AR4

    model, _, lower, upper = arm()
    target, last = arm_draw(seed, True)
    tree = read_urdf(AR4).tree
    held = PLANT if PLANT == "early" else ""
    with in_dtype("float64"), planted((held,) if held else ()):
        found, reached = inverse_kinematics_program(held)(jnp.asarray(target), jnp.asarray(last))
    with in_dtype("float64"):
        placed = model({"joint_positions": jnp.zeros(6)}, {})["frame_poses"][0]
        local = np.asarray(compose(inverse(placed), jnp.asarray(target)))
        theirs = inverse_kinematics(tree, "link_6", jnp.asarray(local), jnp.asarray(last), position_tolerance=1e-9,
                                    rotation_tolerance=1e-9, owner=OWNER, iterations=200)
    reached = np.asarray(reached)
    assert bool(found.converged)
    if np.any(np.asarray(found.active) != 0):
        # Held at a limit: what `converged` claims, judged outside the solve. Unregularised, near the wrist's singular
        # pose Newton's step along the direction the pose barely fixes is long, and the solve can end at a limit far
        # from its start, off the pose, where the library's damped steps met the pose close by (draws 451, 57135088
        # and 12766: joint 4 at its limit 1.8 rad from the start, link_6 2 to 3 mm off; with no limits the same solve
        # ends 215 rad round). The minimiser reached from the start is the one claimed, not the nearest.
        with in_dtype("float64"):
            assert stationary_within_the_limits(inverse_kinematics_rows(jnp.asarray(target)), found, lower, upper) == []
        return
    # Held at no limit, the solve reaches the reachable pose.
    assert np.linalg.norm(reached[:3] - target[:3]) <= 1e-9, np.linalg.norm(reached[:3] - target[:3])
    if bool(theirs.reached) and bool(theirs.within_limits):
        apart = float(np.max(np.abs(np.asarray(found.commands) - np.asarray(theirs.q))))
        # The same joints on the library's branch; the AR4's wrist has a second solution for a pose (joints 4 and 6
        # turned by pi, joint 5 negated), which a solve from near the wrist's singular pose can reach instead (seed
        # 149, before this path's line search changed as well): the pose is then met, as asserted, on that branch.
        assert apart <= 1e-6 or apart > 0.5, apart
    # Otherwise the library's solve is no oracle here: its step ignores the limits (from some starts it ends outside
    # them), or its damped steps stop short of its 1e-9 tolerances in its iterations (seed 729201: 1.2e-7 m off
    # after 200, inside the limits, the solve here on the pose to 1e-16 m).


def stationary_within_the_limits(rows: Callable[[Any], Any], found: Any, lower: np.ndarray,
                                 upper: np.ndarray) -> list[str]:
    """What is wrong with ``found``'s claim to be stationary within the limits, computed here from the objective's
    own gradient (``jax.grad``, not the solve's): each free joint's scaled gradient, and each held joint's scaled
    distance from its limit, within the declared line of the problem's size, and each held joint's gradient pushing
    out of the box (the first-order conditions of a minimiser at a limit)."""
    commands = jnp.asarray(found.commands)
    made = np.asarray(rows(commands))
    columns = np.linalg.norm(np.asarray(jax.jacfwd(rows)(commands)), axis=0)
    gradient = np.asarray(jax.grad(lambda q: 0.5 * jnp.sum(rows(q) ** 2))(commands))
    q, active = np.asarray(commands), np.asarray(found.active)
    size = float(np.linalg.norm(made)) + float(np.sum(columns * np.abs(q)))
    line = LINE[np.dtype("float64")] * size
    problems_: list[str] = []
    for k in range(q.size):
        bound = lower[k] if active[k] < 0 else upper[k]
        if active[k] == 0 and abs(gradient[k]) / columns[k] > line:
            problems_.append(f"joint {k} free, its scaled gradient {gradient[k] / columns[k]:.3g} past {line:.3g}")
        if active[k] != 0 and columns[k] * abs(q[k] - bound) > line:
            problems_.append(f"joint {k} held {columns[k] * abs(q[k] - bound):.3g} (scaled) from its limit")
        if active[k] != 0 and active[k] * gradient[k] > 0.0:
            problems_.append(f"joint {k} held, its gradient {gradient[k]:.3g} pulling it into the box")
    return problems_


@pytest.mark.memory_heavy
@settings(max_examples=40, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@example(seed=510511, reachable=False)  # the stress run's draw (float32 stopped at 4 iterations by a premature stall)
@given(seed=st.integers(0, 2 ** 32 - 1), reachable=st.booleans())
def test_float32_is_judged_against_float64(seed, reachable):
    """The arm's solve in float32 converges where float64's does, its joints within 1e-3 rad of float64's and its
    forward-mode derivative within 1e-2 of float64's largest entry."""
    target, last = arm_draw(seed, reachable)
    zero = np.zeros(12)
    wide = arm_call("solve", "float64", reachable, zero, target, last)
    narrow = arm_call("solve", "float32", reachable, zero, target, last)
    assert bool(wide.converged) and bool(narrow.converged), (float(wide.stationarity), float(narrow.stationarity))
    assert np.max(np.abs(np.asarray(narrow.commands, np.float64) - np.asarray(wide.commands))) <= 1e-3
    if np.array_equal(np.asarray(narrow.active), np.asarray(wide.active)):
        d64 = np.asarray(arm_call("forward", "float64", reachable, zero, target, last))
        d32 = np.asarray(arm_call("forward", "float32", reachable, zero, target, last), np.float64)
        assert np.max(np.abs(d32 - d64)) <= 1e-2 * np.max(np.abs(d64))


__all__: list[str] = []
