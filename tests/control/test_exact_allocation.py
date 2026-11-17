"""The exact allocation, ``mime.control.exact_allocation``: against SciPy's BVLS, its derivative, its guards, its
refusal, and ``allocate()`` re-based on it.

**The swap test** draws problems on the three arrays MIME ships (``allocation_draws``) and holds the law to BVLS at
camera spike R2's line in both dtypes, limits inactive and active. Its plant (``ALLOCATION_PLANT=limits``): the law
is handed limits 1 % looser. Where two faces are within the law's tie window the test lets the law take either, the one of
less effort: a unique minimiser binds it only past that window (a recorded draw, below, pins this).

**The derivative**, within a draw's active set, is affine in the wanted values and the limit, so central differences
over a segment whose ends hold the same active set are exact up to rounding, and judge it to the line over the step;
forward and reverse mode agree to rounding; float32's is judged against float64's at the same problem. Its plants:
``free_tangent`` (the commands' derivative the unconstrained least squares', a wrong active set) and ``stop_branch``
(the derivative of what the bound commands leave the free ones stopped). At an active-set change the derivative jumps
and stays finite.

**The derivative in every input** (``stop_q`` and ``stop_inverse`` plant a stopped gradient on the matrix's path): the
first test above differentiates in the wanted values and the limit alone; this one differentiates in the matrix, the
wanted values, both limits, the offset, the reference, its scales and the rows' scales, on nine shapes of 3 to 8
commands, tall, wide and regularised, in both dtypes.

**The column spread**: the one corner in which the law is not exact, stated on its page, held at its edge in each dtype.

**The guards** are tested structurally: each ``domain_guard`` of the law is reached with its condition false by one of
``allocation_cases.GUARD_POINTS``, and replacing its fallback there changes no output and no derivative. Its plant
(``single_where``): the stand-ins removed, which no point then reaches."""

from __future__ import annotations

import contextlib
import inspect
import os
import re
import sys
from collections import Counter
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import jax
import numpy as np
import pytest
from allocation_cases import (GUARD_POINTS, LIMIT, OWNER, allocating_field, allocating_sphere, guard_point_derivative,
                              guard_point_value)
from allocation_draws import (DECLARED_TIE_ULPS, SLACK, General, Problem, bvls, column_oracle, general_at, generals, line, problems,
                              provider, reach_of, residual_line, two_sided_window, unique, verdict)
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

import mime.control._allocation as law_module
from mime.build._refusals import ArgumentRefused
from mime.control import EXACT_COMMANDS, exact_allocation
from mime.core import switch
from mime.assets import asset
from mime.fields import CalibratedMultipole, allocate
from mime.testing import fallback_problems
from mime.units import A, T_per_m, m, mT, ms, uT

sys.path.append(str(Path(__file__).resolve().parents[1] / "solve_sites"))
from solve_walk import sites_reached, solve_problems  # noqa: E402

jnp: Any = jax.numpy
lax: Any = jax.lax
PLANT = os.environ.get("ALLOCATION_PLANT", "")
#: The law's own share of what the bound commands leave, which the float32 test's float64 reference always runs (the
#: stop_branch plant replaces it for the law under test).
TRUE_LEFT = law_module._left


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def solved(problem: Problem, dtype: str) -> np.ndarray:
    with in_dtype(dtype):
        a, b, s = (jnp.asarray(v, dtype=dtype) for v in (problem.matrix, problem.target, problem.scales))
        held = jnp.asarray(problem.limit * (1.01 if PLANT == "limits" else 1.0), dtype=dtype)
        found = exact_allocation(a, b, lower=-held, upper=held, scales=s, owner=OWNER)
        return np.asarray(found.commands, dtype=np.float64)


# -- the swap test -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
@settings(max_examples=150, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(problem=problems())
def test_the_exact_allocation_meets_scipys_bvls_within_the_line(dtype, problem):
    assert verdict(problem, solved(problem, dtype), dtype) == []


def test_control_limits_one_percent_looser_fail_the_swap_test():
    """The plant the detection rate is measured with, held here at one problem with a coil at its limit."""
    problem = next(p for p in fixed_problems() if np.any(np.abs(bvls(p)) >= p.limit * (1 - 1e-9)))
    a, b, s = (jnp.asarray(v) for v in (problem.matrix, problem.target, problem.scales))
    with in_dtype("float64"):
        found = exact_allocation(a, b, lower=-problem.limit * 1.01, upper=problem.limit * 1.01, scales=s, owner=OWNER)
        assert verdict(problem, np.asarray(found.commands), "float64") != []


def tie_draw() -> Problem:
    """Release run 43740791320 (shard 29, Python 3.13), the swap test's draw: OctoMag at the centre of its region asked
    for a field of 2.72 times what its coils reach at 5 A, along y, with the field rows over 10 mT and the gradient's over
    1 T/m."""
    matrix = provider("octomag").actuation(m((0.0, 0.0, 0.0))).matrix()
    wanted = np.zeros(8)
    wanted[1] = reach_of(matrix, 5.0) * float(np.exp(1.0))
    return Problem("octomag", "both", matrix, wanted, np.asarray([1e-2] * 3 + [1.0] * 5), 5.0)


def test_a_unique_minimiser_binds_the_law_only_past_its_tie_window_in_float32():
    """The draw failed the swap test in float32: the law put command 1 at +5 A where BVLS puts it at -5 A. The problem is
    unique (its columns are alike, spread 1.4, condition 12: nothing near the stated column-spread limit), and the two
    faces' residuals differ by 85 units in the last place of the target's norm, which float32 resolves (its computed
    residuals are good to a third of one), but inside the law's tie window of 64 units each side, so the law ties the
    faces, and their effort is equal. The law's own rule then holds either, and the verdict holds it to the residual
    past its windows, not to BVLS's commands. The controls: float64, whose windows are 1e-9 of that, holds the law to
    BVLS's commands, and a face off by a command whose flip costs far past the float32 windows is still refused."""
    problem = tie_draw()
    a, b = problem.scaled("float32")
    reference = bvls(problem, "float32")
    flipped = reference.copy()
    flipped[0] *= -1  # the tied face
    gap = float(np.linalg.norm(a @ flipped - b) - np.linalg.norm(a @ reference - b))
    window = two_sided_window(problem, reference, flipped, "float32")
    # the premise of the draw: unique, resolvable by float32's rounding, inside the windows
    assert unique(problem, "float32") and 8 * float(np.finfo("float32").eps) * float(np.linalg.norm(b)) < gap < window
    assert verdict(problem, flipped, "float32") == []
    assert verdict(problem, solved(problem, "float32"), "float32") == []
    far = reference.copy()
    far[1] *= -1  # a command whose flip costs far more than the windows
    assert verdict(problem, far, "float32") != []
    assert verdict(problem, solved(problem, "float64"), "float64") == []
    assert verdict(problem, bvls(problem) * np.asarray([-1, 1, 1, 1, 1, 1, 1, 1]), "float64") != []


def cardiomag_tie_draw() -> Problem:
    """Release run 43748061840's second swap red (PR 325's CI, float32): CardioMag at the centre of its region asked for a
    field along z and a gradient's last entries, the field rows over 10 mT and the gradient's over 1 T/m, 5 A."""
    matrix = provider("cardiomag").actuation(m((0.0, 0.0, 0.0))).matrix()
    wanted = np.asarray([0.0, 0.0, 0.03585362502957054, 0.0, 0.0, 0.0, 0.057365800047312863, 0.043024350035484646])
    return Problem("cardiomag", "both", matrix, wanted, np.asarray([1e-2] * 3 + [1.0] * 5), 5.0)


def test_a_unique_minimiser_with_a_command_off_its_bound_is_held_to_the_tie_window_too():
    """The second draw of the same cause. Unique (columns alike), float64 agrees with BVLS to 3e-14 A; in float32 the law
    puts command 5 at its +5 A bound where BVLS leaves it at 4.95 A, and the commands differ by 0.062 A against a line of
    0.00083. The residuals differ by 3.4e-5, inside the windows (1.3e-4), and the law's effort is the lower. The law's
    float32 commands at the draw are held here as numbers, so that the case does not move with the law's rounding."""
    problem = cardiomag_tie_draw()
    a, b = problem.scaled("float32")
    reference = bvls(problem, "float32")
    law = np.asarray([5.0, -4.20883465, 4.74467897, -4.38866854, 5.0, -4.01990557, 3.06744909, -3.12709117])
    gap = float(np.linalg.norm(a @ law - b) - np.linalg.norm(a @ reference - b))
    window = two_sided_window(problem, reference, law, "float32")
    assert unique(problem, "float32") and float(np.max(np.abs(law - reference))) > line(problem, reference, "float32")
    assert 0.0 < gap < window
    assert verdict(problem, law, "float32") == []
    far = reference.copy()
    far[2] *= -1  # a command whose flip costs far more than the windows
    assert verdict(problem, far, "float32") != []
    assert verdict(problem, solved(problem, "float64"), "float64") == []


# -- the tie window's pin ------------------------------------------------------------------------------------------

@contextlib.contextmanager
def tie_planted() -> Iterator[None]:
    """The law's ``TIE_ULPS`` 16 times wider (``ALLOCATION_PLANT=tie_x16``) or a quarter as wide
    (``ALLOCATION_PLANT=tie_div4``) for the context, if ``PLANT`` names one: the mutants the pin below is rated with
    (a test-side patch of the module's constant, never a ``src/`` change)."""
    with pytest.MonkeyPatch.context() as patch:
        if PLANT == "tie_x16":
            patch.setattr(law_module, "TIE_ULPS", law_module.TIE_ULPS * 16)
        elif PLANT == "tie_div4":
            patch.setattr(law_module, "TIE_ULPS", law_module.TIE_ULPS / 4)
        yield


def test_the_laws_tie_window_is_the_oracles_declared_64_ulps():
    """The oracle carries its own copy of the window (``allocation_draws.DECLARED_TIE_ULPS``, a literal), never the law's
    constant, so a change in the law moves nothing in the oracle and is seen here (x16 in the constant went unseen, 0
    of 8 draws, while the oracle imported it)."""
    assert DECLARED_TIE_ULPS == 64
    with tie_planted():
        assert law_module.TIE_ULPS == DECLARED_TIE_ULPS, (
            f"the law's tie window is {law_module.TIE_ULPS} units in the last place, the declared contract is "
            f"{DECLARED_TIE_ULPS}: changing the window is a contract change that needs the maintainer's word (the "
            f"oracle's own copy is allocation_draws.DECLARED_TIE_ULPS)")


def two_faces_problem(dtype: str, ratio: float) -> tuple[Problem, np.ndarray, np.ndarray]:
    """Two faces whose residuals differ by ``ratio`` times the law's two-sided tie window (the oracle's own, 64 units),
    and the problem's ``(problem, deep, bound)`` faces. The rows are ``diag(0.001, 1)`` over unit scales, the wanted
    value ``(0.001 (1 - g), 0.3)``, the limits +-1 and the reference ``(1, 0.3)``: the first command is free at
    ``1 - g`` (residual zero) or at its upper bound 1 (residual ``0.001 g``, effort zero, the reference). The lower
    residual is the free face's, the lower effort the bound face's: tied, the law takes the bound face."""
    eps, sensitivity, other = float(np.finfo(dtype).eps), 1e-3, 0.3
    window = 4 * DECLARED_TIE_ULPS * eps * float(np.hypot(sensitivity, other))
    gap = ratio * window / sensitivity
    problem = Problem("two_faces", "synthetic", np.diag([sensitivity, 1.0]),
                      np.asarray([sensitivity * (1.0 - gap), other]), np.ones(2), 1.0)
    return problem, np.asarray([1.0 - gap, other]), np.asarray([1.0, other])


def two_faces_law(problem: Problem, dtype: str) -> np.ndarray:
    with in_dtype(dtype), tie_planted():
        a, b, s = (jnp.asarray(v, dtype=dtype) for v in (problem.matrix, problem.target, problem.scales))
        found = exact_allocation(a, b, lower=-jnp.ones(2, dtype=dtype), upper=jnp.ones(2, dtype=dtype), scales=s,
                                 owner=OWNER, reference=jnp.asarray([1.0, 0.3], dtype=dtype))
        return np.asarray(found.commands, dtype=np.float64)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("ratio", [0.95, 1.05])
def test_the_law_ties_two_faces_just_inside_its_window_and_not_just_outside(dtype, ratio):
    """The law's tie window as a contract, on a constructed problem (``two_faces_problem``): the faces' residuals differ
    by 0.95 of the oracle's two-sided window and the law ties them (the lower effort, the bound face, is taken); by 1.05
    of it and the law does not (the lower residual, the free face, is taken). The premise is asserted on the faces as
    rounded in ``dtype``: the oracle's own ratio of the gap to the window, in float64, within 0.04 of the nominal. A
    law whose window is 16 times wider ties the 1.05 pair, one a quarter as wide does not tie the 0.95 pair."""
    problem, free, bound = two_faces_problem(dtype, ratio)
    a, b = problem.scaled(dtype)
    free, bound = (np.asarray(v, dtype=dtype).astype(np.float64) for v in (free, bound))
    gap = float(np.linalg.norm(a @ bound - b) - np.linalg.norm(a @ free - b))
    measured = gap / two_sided_window(problem, free, bound, dtype)
    assert abs(measured - ratio) < 0.04, (measured, ratio)
    taken = two_faces_law(problem, dtype)
    expected = bound if ratio < 1.0 else free
    assert np.allclose(taken, expected, rtol=0.0, atol=0.25 * (bound[0] - free[0])), (taken, expected)
    assert (taken[0] == bound[0]) == (ratio < 1.0), (taken, ratio)


def fixed_problems() -> list[Problem]:
    """A problem on each array and kind, drawn once by a fixed seed, its wanted value three times what the limit
    reaches."""
    rng = np.random.default_rng(7)
    return [_drawn(system, kind, rng) for system in ("navion_2", "cardiomag", "octomag")
            for kind in ("field", "both", "force")]


def _drawn(system: str, kind: str, rng: Any) -> Problem:
    from allocation_draws import FIELD_SCALE, GRADIENT_SCALE, provider, region

    centre, half = region(system)
    point = centre + rng.uniform(-1, 1, 3) * half
    actuation = provider(system).actuation(m(tuple(point)))
    limit = 20.0
    if kind == "force":
        matrix = np.einsum("i,ijk->jk", np.asarray((0.0, 0.0, 5e-4)), actuation.gradient)
        scales = np.full(3, float(np.max(np.linalg.norm(matrix, axis=0))))
    else:
        matrix = actuation.field if kind == "field" else actuation.matrix()
        scales = np.asarray([FIELD_SCALE] * 3 + ([GRADIENT_SCALE] * 5 if kind == "both" else []))
    direction = rng.uniform(-1, 1, len(matrix))
    wanted = direction / np.linalg.norm(direction) * np.linalg.norm(matrix, axis=0).sum() * limit * 3.0
    return Problem(system, kind, matrix, wanted, scales, limit)


# -- the derivative ------------------------------------------------------------------------------------------------

def commands_of(problem: Problem, dtype: str, plant: str = "") -> Any:
    """The commands as a function of the wanted values and the limit, ``p = (wanted, limit)``, in ``dtype``, with the
    derivative plant ``plant``."""
    a, s = (jnp.asarray(v, dtype=dtype) for v in (problem.matrix, problem.scales))
    rows = len(problem.target)

    def f(p: Any) -> Any:
        found = exact_allocation(a, p[:rows], lower=-p[rows], upper=p[rows], scales=s, owner=OWNER).commands
        if plant == "free_tangent":
            free = jnp.linalg.pinv(a / s[:, None]) @ (p[:rows] / s)
            return lax.stop_gradient(found) + free - lax.stop_gradient(free)
        return found

    return f


def active_of(problem: Problem, dtype: str, p: Any) -> np.ndarray:
    a, s = (jnp.asarray(v, dtype=dtype) for v in (problem.matrix, problem.scales))
    rows = len(problem.target)
    return np.asarray(exact_allocation(a, p[:rows], lower=-p[rows], upper=p[rows], scales=s, owner=OWNER).active)


def derivative_problems(problem: Problem, dtype: str, tangent: np.ndarray, cotangent: np.ndarray,
                        plant: str = "") -> list[str]:
    """What is wrong with the derivative at ``problem`` along ``tangent``: forward against reverse mode, and in float64
    forward mode against central differences over a segment that holds one active set; in float32 against float64's."""
    found: list[str] = []
    eps = float(np.finfo(dtype).eps)
    with in_dtype(dtype):
        p = jnp.asarray(np.append(problem.target, problem.limit), dtype=dtype)
        rows = len(problem.target)
        v = jnp.asarray(tangent * np.append(np.full(rows, reach_of(problem.matrix, problem.limit)), problem.limit),
                        dtype=dtype)
        w = jnp.asarray(cotangent, dtype=dtype)
        f = commands_of(problem, dtype, plant)
        value, forward = jax.jvp(f, (p,), (v,))
        _, pullback = jax.vjp(f, p)
        (backward,) = pullback(w)
        jacobian = np.asarray(jax.jacfwd(f)(p), dtype=np.float64)
        forward, backward = np.asarray(forward, np.float64), np.asarray(backward, np.float64)
        if not (np.all(np.isfinite(forward)) and np.all(np.isfinite(backward))):
            return [f"a derivative is not finite: {forward}, {backward}"]
        held = SLACK * eps * float(np.abs(np.asarray(w, np.float64)) @ np.abs(jacobian) @ np.abs(np.asarray(v, np.float64)))
        if abs(float(np.asarray(w, np.float64) @ forward) - float(backward @ np.asarray(v, np.float64))) > held:
            found.append(f"forward and reverse mode differ past {held:.3g}")
        here = active_of(problem, dtype, p)
    if dtype == "float64":
        with in_dtype("float64"):
            reference = bvls(problem)
            for h in (1e-2, 1e-3, 1e-4, 1e-5):
                ends = (p + h * v, p - h * v)
                if all(np.array_equal(active_of(problem, dtype, end), here) for end in ends):
                    difference = (np.asarray(f(ends[0])) - np.asarray(f(ends[1]))) / (2 * h)
                    held = 2 * line(problem, reference, dtype) / (2 * h) + \
                        line(problem, reference, dtype) / problem.limit * float(np.max(np.abs(forward)))
                    miss = float(np.max(np.abs(difference - forward)))
                    if miss > held:
                        found.append(f"forward mode {forward} against central differences {difference} at h {h}: "
                                     f"{miss:.3g} past {held:.3g}")
                    break
    else:
        with in_dtype("float64"):
            a32 = np.asarray(problem.matrix, dtype=np.float32).astype(np.float64)
            rounded = Problem(problem.system, problem.kind, a32,
                              np.asarray(problem.target, np.float32).astype(np.float64),
                              np.asarray(problem.scales, np.float32).astype(np.float64),
                              float(np.float32(problem.limit)))
            p64 = jnp.asarray(np.asarray(p, np.float64))
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(law_module, "_left", TRUE_LEFT)
                same = np.array_equal(active_of(rounded, "float64", p64), here)
                reference = commands_of(rounded, "float64")
                _, wide = jax.jvp(reference, (p64,), (jnp.asarray(np.asarray(v, np.float64)),))
                spread = np.abs(np.asarray(jax.jacfwd(reference)(p64))) @ np.abs(np.asarray(v, np.float64))
            if same:
                held = line(rounded, bvls(rounded, "float32"), "float32") / rounded.limit * \
                    float(np.max(np.abs(np.asarray(wide)))) + SLACK * eps * float(np.max(spread))
                miss = float(np.max(np.abs(np.asarray(wide) - forward)))
                if miss > held:
                    found.append(f"float32's forward mode {forward} against float64's {np.asarray(wide)}: {miss:.3g} "
                                 f"past {held:.3g}")
    return found


def unit(direction: np.ndarray) -> np.ndarray:
    return direction / np.linalg.norm(direction)


def directions(rows: int) -> st.SearchStrategy[tuple[np.ndarray, np.ndarray]]:
    """A tangent on the wanted values and the limit, and a cotangent on eight commands (cut to the problem's), each
    component of magnitude 0.1 to 1 and either sign. No component is near zero: one drawn at float32's least normal
    (1.2e-38) made reverse-mode products below the normal range, which XLA's CPU flushes to zero, so the two modes
    differed by the flushed share, not by the law's rounding."""
    component = st.builds(lambda sign, size: sign * size, st.sampled_from((-1.0, 1.0)), st.floats(0.1, 1.0))
    return st.tuples(st.lists(component, min_size=rows + 1, max_size=rows + 1),
                     st.lists(component, min_size=8, max_size=8)).map(lambda t: (np.asarray(t[0]), np.asarray(t[1])))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@settings(max_examples=100, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(problem=problems(), data=st.data())
def test_the_derivative_is_the_selected_sets_in_both_modes(dtype, problem, data):
    tangent, cotangent = data.draw(directions(len(problem.target)))
    plant = PLANT if PLANT in ("free_tangent", "stop_branch") else ""
    with pytest.MonkeyPatch.context() as patch:
        if plant == "stop_branch":
            patch.setattr(law_module, "_left", lambda t, s, b: t - s @ lax.stop_gradient(b))
        assert derivative_problems(problem, dtype, unit(tangent), unit(cotangent[:problem.matrix.shape[1]]), plant) == []


def test_at_an_active_set_change_the_derivative_jumps_and_stays_finite():
    """The wanted force scaled past what the limit reaches: a coil reaches its limit between two scales, the commands
    stay continuous across it, their derivative in the scale jumps there, and never stops being finite."""
    problem = next(p for p in fixed_problems() if p.system == "navion_2" and p.kind == "force")
    with in_dtype("float64"):
        a, s = jnp.asarray(problem.matrix), jnp.asarray(problem.scales)
        b = jnp.asarray(problem.target)

        def at(scale: Any) -> Any:
            return exact_allocation(a, scale * b, lower=-problem.limit, upper=problem.limit, scales=s, owner=OWNER)

        scales = np.linspace(0.01, 1.0, 400)
        actives = [np.asarray(at(x).active) for x in scales]
        change = next(i for i in range(1, len(scales)) if not np.array_equal(actives[i], actives[i - 1]))
        slope = jax.grad(lambda x: jnp.sum(at(x).commands))
        before, after = float(slope(scales[change - 1])), float(slope(scales[change]))
        assert np.isfinite([float(slope(x)) for x in scales]).all()
        assert abs(after - before) > 1e-6 * max(abs(before), abs(after))
        gap = float(np.max(np.abs(np.asarray(at(scales[change]).commands) - np.asarray(at(scales[change - 1]).commands))))
        assert gap < 4 * (scales[1] - scales[0]) * max(abs(before), abs(after)) * len(actives[0])


# -- the derivative in every input ---------------------------------------------------------------------------------

class _PlantedLinalg:
    """``jnp.linalg`` with one differentiable result of the solve stopped: the orthogonal factor of the tall solve's QR
    (``stop_q``), or the wide solve's pseudo-inverse (``stop_inverse``)."""

    def __init__(self, plant: str) -> None:
        self.plant = plant

    def __getattr__(self, name: str) -> Any:
        return getattr(jnp.linalg, name)

    def qr(self, a: Any) -> Any:
        q, r = jnp.linalg.qr(a)
        return (lax.stop_gradient(q) if self.plant == "stop_q" else q), r

    def pinv(self, a: Any, **kwargs: Any) -> Any:
        found = jnp.linalg.pinv(a, **kwargs)
        return lax.stop_gradient(found) if self.plant == "stop_inverse" else found


class _PlantedJnp:
    """The law's ``jnp`` with the plant applied to its ``linalg``."""

    def __init__(self, plant: str) -> None:
        self.linalg = _PlantedLinalg(plant)

    def __getattr__(self, name: str) -> Any:
        return getattr(jnp, name)


@contextlib.contextmanager
def planted_law(planted: bool = True) -> Iterator[None]:
    """The matrix-path plant ``PLANT`` names, if it is one and ``planted``, for the context (the float32 test's float64
    reference is never planted: planted alike, the two would agree)."""
    with pytest.MonkeyPatch.context() as patch:
        if planted and PLANT in ("stop_q", "stop_inverse"):
            patch.setattr(law_module, "jnp", _PlantedJnp(PLANT))
        yield


@cache
def law_of(names: tuple[str, ...], only: str = "commands", planted: bool = True) -> Any:
    """The law's commands (or its ``only`` output) as a jitted function of the inputs ``names``, in the order given.
    Jitted and kept: the law's operations dispatched one by one, each shape and dtype of them, outgrow JAX's caches and
    every example then compiles afresh (ten seconds in place of one). ``planted`` only keys the cache: the plant is in
    the trace, which the caller's context sets."""
    def run(*values: Any) -> Any:
        given = dict(zip(names, values, strict=True))
        found = exact_allocation(given["matrix"], given["wanted"], lower=given["lower"], upper=given["upper"],
                                 scales=given["scales"], owner=OWNER, offset=given["offset"],
                                 reference=given["reference"], reference_scales=given.get("reference_scales"))
        return found if only == "all" else getattr(found, only)
    return jax.jit(run)


#: Steps of the central difference, tried from the largest: the first at which both ends, and the double step's ends,
#: hold the base's active set, and at which the difference's own rounding is small against the derivative it judges
#: (``ROUNDING_SHARE``). A draw's active set can switch within 1e-7 of its base along one input's tangent (a coil a
#: hair from its bound), which a ladder stopping at 1e-7 reads as "undifferenceable".
STEPS = (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)
#: The share of the law's sensitivity (its largest derivative, times the tangent) that a step's rounding may reach before
#: the step is too small to judge by: the floor of the ladder, set by the dtype's rounding and not by a number of decades.
ROUNDING_SHARE = 1e-3
#: Of the draws of a run (the float64 property's), each input must be differenced in at least this share. A draw that no
#: step clears is not a failure of the law (an input can sit on an active-set switch); a run in which most draws leave an
#: input undifferenced has tested too little of it.
RUN_DIFFERENCED_SHARE = 0.9


class Derivatives(NamedTuple):
    """The law at ``values`` in a dtype: its active set, its forward-mode product along each input's tangent, its
    Jacobian in each input (``(K, size)``), and the pullback of the cotangent (each input's cotangent)."""

    active: np.ndarray
    products: list[np.ndarray]
    jacobians: list[np.ndarray]
    pulled: list[np.ndarray]


def derivatives_at(names: tuple[str, ...], values: list[np.ndarray], tangents: list[np.ndarray],
                   cotangent: np.ndarray, dtype: str, planted: bool = True) -> Derivatives:
    with in_dtype(dtype), planted_law(planted):
        x = [jnp.asarray(v, dtype=dtype) for v in values]
        law = law_of(names, planted=planted)
        products = []
        for i in range(len(names)):
            only = [jnp.asarray(t, dtype=dtype) if j == i else jnp.zeros_like(x[j]) for j, t in enumerate(tangents)]
            products.append(np.asarray(jax.jvp(law, x, only)[1], np.float64))
        _, pullback = jax.vjp(law, *x)
        return Derivatives(
            np.asarray(law_of(names, "active", planted)(*x)), products,
            [np.asarray(jax.jacfwd(law, argnums=j)(*x), np.float64).reshape(len(products[0]), -1)
             for j in range(len(names))],
            [np.asarray(c, np.float64) for c in pullback(jnp.asarray(cotangent, dtype=dtype))])


def derivative_in_every_input_problems(general: General, dtype: str,
                                       differenced_inputs: set[str] | None = None) -> list[str]:
    """What is wrong with the law's derivative in each of its inputs (matrix, wanted, lower, upper, offset, reference,
    reference scales, scales) at ``general``, along one tangent in each.

    Float64: the forward-mode product against the central difference over a step that holds the active set. The
    commands are smooth there, so the difference's error is its truncation, which the step and the double step give
    (an error of ``h^2`` and four times it, so ``|D(2h) - D(h)|`` is three times it, allowed twice over), plus its
    rounding, ``SLACK`` times eps times the selected set's condition number times the commands' largest size at its
    ends over the step (the scale a command's rounding is of, not its own: a small one is a difference of large
    terms). Float32: the forward-mode product against float64's at the same rounded inputs, where the active set is
    the same. Both dtypes: reverse against forward mode, one cotangent's pullback dotted with the tangent against the
    cotangent dotted with the product. Those two are within ``SLACK`` times eps times the condition number times the
    Jacobian's absolute product and the law's own sensitivity (its largest derivative in any input, times the
    tangent): the scale an input's rounding is of where the derivative in it is zero, which the result's own scale
    would call exact. The condition number is the selected set's: the derivative of a least squares or a
    pseudo-inverse carries it, and its square in the matrix's.

    ``differenced_inputs``, if given, is filled with the names of the inputs the float64 central difference judged. An
    input whose every step crossed an active-set change, or whose rounding at the steps that did not was too large to
    judge by, is left out and is no failure of the draw (it can sit on a switch): the property's run counts them, and
    fails where an input was left out of too many draws (``run_guard_problems``)."""
    found: list[str] = []
    rng = np.random.default_rng(general.seed + 1)
    names = tuple(general.inputs())
    commands = general.matrix.shape[1]
    values = [np.asarray(v, dtype=dtype).astype(np.float64) for v in general.inputs().values()]
    tangents = []
    for name, value in zip(names, values, strict=True):
        size = float(np.sqrt(np.mean(value ** 2))) if name in ("matrix", "scales", "reference_scales") else 1.0
        tangents.append(rng.choice((-1.0, 1.0), value.shape) * rng.uniform(0.1, 1.0, value.shape) * size)
    cotangent = rng.choice((-1.0, 1.0), commands) * rng.uniform(0.1, 1.0, commands)
    own = derivatives_at(names, values, tangents, cotangent, dtype)
    # The conditioning of the selected set's solve (its free columns over the row scales, with the regularisation rows):
    # the derivative of a least squares or a pseudo-inverse carries it (its square, in the matrix's), so that rounding
    # in either mode is eps times it times the products' size, not eps times their size alone.
    given = general.inputs()
    rows = given["matrix"] / given["scales"][:, None]
    if "reference_scales" in given:
        rows = np.concatenate([rows, np.diag(1.0 / given["reference_scales"])])
    free = own.active == 0
    eps = float(np.finfo(dtype).eps) * (float(np.linalg.cond(rows[:, free])) if free.any() else 1.0)
    # The law's own sensitivity: its largest derivative in any input, which the products of an input the commands do
    # not depend on (a reference of a command at a bound) cancel from, so that their rounding is of this scale.
    sensitivity = sum(float(np.max(np.abs(jacobian))) for jacobian in own.jacobians)
    wide = derivatives_at(names, values, tangents, cotangent, "float64", planted=False) if dtype == "float32" else own
    differenced = 0
    for i, name in enumerate(names):
        product, tangent = own.products[i], tangents[i]
        if not np.all(np.isfinite(product)):
            found.append(f"{name}: the forward-mode product is not finite: {product}")
            continue
        floor = float(np.max(np.abs(tangent))) * sensitivity
        held_by = SLACK * eps * float(np.abs(cotangent) @ (np.abs(own.jacobians[i]) @ np.abs(tangent).ravel() + floor))
        reverse = float(np.sum(own.pulled[i] * tangent))
        if abs(reverse - float(cotangent @ product)) > held_by:
            found.append(f"{name}: reverse mode {reverse:.6g} against forward {float(cotangent @ product):.6g}, "
                         f"past {held_by:.3g}")
        if dtype == "float32":
            if np.array_equal(wide.active, own.active):
                differenced += 1
                allowed = SLACK * eps * (np.abs(wide.jacobians[i]) @ np.abs(tangent).ravel() + floor)
                if np.any(np.abs(wide.products[i] - product) > allowed):
                    found.append(f"{name}: float32's forward mode {product} against float64's {wide.products[i]}: "
                                 f"{float(np.max(np.abs(wide.products[i] - product))):.3g} past "
                                 f"{float(np.max(allowed)):.3g}")
            continue
        with in_dtype(dtype), planted_law():
            x = [jnp.asarray(v, dtype=dtype) for v in values]
            whole = law_of(names, "all")
            for h in STEPS:
                ends = {k: whole(*[v + k * h * jnp.asarray(tangent) if j == i else v for j, v in enumerate(x)])
                        for k in (1, -1, 2, -2)}
                if not all(np.array_equal(np.asarray(end.active), own.active) for end in ends.values()):
                    continue
                high, low = np.asarray(ends[1].commands), np.asarray(ends[-1].commands)
                one = (high - low) / (2 * h)
                two = (np.asarray(ends[2].commands) - np.asarray(ends[-2].commands)) / (4 * h)
                rounding = SLACK * eps * float(np.max(np.abs(high) + np.abs(low))) / (2 * h)
                if rounding > ROUNDING_SHARE * floor:
                    break  # smaller steps only round worse: the rest of the ladder cannot judge this input
                allowed = 2 * np.abs(two - one) + rounding
                if np.any(np.abs(one - product) > allowed):
                    found.append(f"{name}: forward mode {product} against central differences {one} at h {h}: "
                                 f"{float(np.max(np.abs(one - product))):.3g} past {float(np.max(allowed)):.3g}")
                differenced += 1
                if differenced_inputs is not None:
                    differenced_inputs.add(name)
                break
    if dtype == "float32" and differenced == 0 and not found:
        # float32 and float64 chose different active sets at the rounded inputs (a near tie): the premise of the
        # comparison, one active set, does not hold, so the draw is no evidence either way.
        assume(False)
    if dtype == "float64" and differenced == 0 and not found:
        found.append(f"none of {len(names)} inputs could be differenced: the draw tests nothing")
    return found


def run_guard_problems(draws: int, differenced: Counter[str], present: Counter[str]) -> list[str]:
    """What a run of the float64 property left too little tested: ``present[name]`` draws had the input ``name`` and
    ``differenced[name]`` of them judged it by a central difference."""
    found = [f"{name}: differenced in {differenced[name]} of {count} draws, under the share {RUN_DIFFERENCED_SHARE}"
             for name, count in sorted(present.items()) if differenced[name] < RUN_DIFFERENCED_SHARE * count]
    if draws == 0:
        found.append("the run drew nothing")
    return found


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_derivative_in_every_input_is_the_selected_sets_in_both_modes(dtype):
    """The law's derivative in the matrix, the wanted values, the limits, the offset, the reference, its scales and the
    rows' scales (the first derivative test differentiates in the wanted values and the limit alone, so a wrong or
    stopped dependence on the matrix was invisible to it), on problems of 3 to 8 commands, tall and wide, regularised
    or not. Plants: ``stop_q`` (the orthogonal factor of the tall solve stopped) and ``stop_inverse`` (the wide
    solve's pseudo-inverse stopped).

    The premise that a draw tests the derivative in each input is held over the run, not per draw: an input a draw
    leaves undifferenced (an active-set switch within every step) is counted, reported as a Hypothesis event, and the
    run fails if any input was left out of more than a tenth of the draws (``run_guard_problems``). The property is
    run from inside the test, so that the counts live in one process whatever the shard or worker."""
    draws, differenced, present = [0], Counter[str](), Counter[str]()

    @settings(max_examples=100, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
    @given(general=generals())
    def the_property(general: General) -> None:
        judged: set[str] = set()
        assert derivative_in_every_input_problems(general, dtype, judged) == [], general.form
        if dtype == "float64":
            draws[0] += 1
            present.update(general.inputs().keys())
            differenced.update(judged)
            event(f"{len(judged)} of {len(general.inputs())} inputs differenced")

    the_property()
    if dtype == "float64":
        assert run_guard_problems(draws[0], differenced, present) == []


def test_the_run_guard_fails_a_run_that_differences_too_little():
    """The guard's control: a run in which the matrix was differenced in 80 of 100 draws fails, and one at 95 passes."""
    present = Counter({"matrix": 100, "wanted": 100})
    assert run_guard_problems(100, Counter({"matrix": 80, "wanted": 100}), present) != []
    assert run_guard_problems(100, Counter({"matrix": 95, "wanted": 100}), present) == []
    assert run_guard_problems(0, Counter(), Counter()) != []


#: The draw of release run 43731739240 (seed 3512841, 6 commands, 6 rows, plain): its matrix and upper limit sit within
#: 7e-8 and 1.1e-7 along their tangents of an active-set switch, so that a ladder stopping at a step of 1e-7 (whose double
#: step crosses it) judged only five of the seven inputs and the per-draw count failed.
SWITCH_NEAR_DRAW = (6, 6, False, 3512841)


def test_a_draw_with_the_matrix_a_hair_from_a_switch_is_differenced_in_every_input():
    judged: set[str] = set()
    general = general_at(*SWITCH_NEAR_DRAW)
    assert derivative_in_every_input_problems(general, "float64", judged) == []
    assert judged == set(general.inputs())


# -- the limit on the columns' spread --------------------------------------------------------------------------------

#: The ratio of one column's norm to the others' up to which the law is exact where its commands outnumber its rows,
#: in each dtype (``1 / RCOND`` a decade lower: 1e10 in float64 and 1e5 in float32, where the rank test of the wide
#: subsets reads the whole set as deficient), measured by ``test_the_law_is_exact_to_the_stated_column_spread...``
COLUMN_SPREAD = {"float64": 1e9, "float32": 1e4}


@cache
def column_law(rows: int, dtype: str) -> Any:
    """The law at limits of one and unit row scales, jitted for the matrices of ``rows`` rows in ``dtype``."""
    return jax.jit(lambda a, b: exact_allocation(a, b, lower=-1.0, upper=1.0, scales=jnp.ones(rows, dtype=dtype),
                                                 owner=OWNER).commands)


def column_excess(dtype: str, spread: float, seed: int, commands: int = 4, rows: int = 3) -> float:
    """What the law's commands leave above the optimum, over ``|wanted|^2``: three rows of Gaussian entries against
    four commands (more commands than rows, so the wide subsets are solved), the last column scaled by ``spread``, the
    limits one, the optimum SciPy's BVLS on the columns normalised, both on the matrix as ``dtype`` rounds it."""
    rng = np.random.default_rng(seed)
    scaling = np.ones(commands)
    scaling[-1] = spread
    with in_dtype(dtype):
        matrix = jnp.asarray(rng.normal(size=(rows, commands)) * scaling, dtype=dtype)
        wanted = jnp.asarray(rng.normal(size=rows) * 2.0, dtype=dtype)
        commands_ = np.asarray(column_law(rows, dtype)(matrix, wanted), np.float64)
    a, b = np.asarray(matrix, np.float64), np.asarray(wanted, np.float64)
    return float(np.sum((a @ commands_ - b) ** 2) - column_oracle(a, b, 1.0)) / float(np.sum(b * b))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_law_is_exact_to_the_stated_column_spread_where_commands_outnumber_rows_and_loses_it_past(dtype):
    """The limit the law's page states: with more commands than rows and no regularisation, one column's norm over the
    others' up to ``COLUMN_SPREAD`` leaves the objective at BVLS's optimum to the conditioning's square times rounding
    (the objective is a square of the residual, whose error is the conditioning times eps), and one decade past it the
    rank test of the whole free set reads it deficient and the objective is far above the optimum. The second half is
    the control: it fails if the edge is not where it is stated (a law that normalised its columns would pass it too,
    and the statement would then change). With commands no more than rows, the subsets are solved on unit columns and
    there is no limit: held at 1e12 in float64 and 1e6 in float32."""
    eps, edge = float(np.finfo(dtype).eps), COLUMN_SPREAD[dtype]
    assert max(column_excess(dtype, edge, seed) for seed in range(8)) <= 16 * (eps * edge) ** 2 + 64 * eps
    assert max(column_excess(dtype, edge * 10 ** 1.5, seed) for seed in range(8)) > 1e-2
    unlimited = 1e12 if dtype == "float64" else 1e6
    assert max(column_excess(dtype, unlimited, seed, commands=3) for seed in range(8)) <= 64 * eps


# -- the guards ----------------------------------------------------------------------------------------------------

def _law_guards() -> int:
    """How many ``domain_guard`` calls the law's source makes."""
    return len(re.findall(r"switch\.domain_guard\(", inspect.getsource(law_module)))


def test_every_guard_of_the_law_is_reached_false_and_its_fallback_reaches_no_output_or_derivative():
    reached: dict[str, bool] = {}
    problems_found: list[str] = []
    with in_dtype("float64"), pytest.MonkeyPatch.context() as patch:
        if PLANT == "single_where":
            patch.setattr(law_module.switch, "domain_guard", lambda condition, a, b: a)
        for name, point in GUARD_POINTS.items():
            held = fallback_problems(lambda point=point: guard_point_value(point),
                                     lambda point=point: guard_point_derivative(point), name)
            problems_found += held.problems
            for site, false in held.guard_sites.items():
                if site.startswith(law_module.__file__):
                    reached[site] = reached.get(site, False) or false
    assert problems_found == []
    assert len([site for site, false in reached.items() if false]) == _law_guards(), reached


def test_a_command_with_no_effect_is_held_at_its_reference_within_its_bounds():
    rows = np.asarray([[1.0, 0.0, 2.0], [0.5, 0.0, -1.0]])
    with in_dtype("float64"):
        for reference, expected in ((None, 0.0), (0.4, 0.4), (3.0, 1.0), (-3.0, -0.5)):
            found = exact_allocation(rows, np.asarray([10.0, 3.0]), lower=-0.5, upper=1.0, scales=np.ones(2),
                                     owner=OWNER, reference=reference)
            assert float(found.commands[1]) == expected and int(found.active[1]) == 0


def test_more_commands_than_the_bound_is_refused_by_name_with_its_route():
    with pytest.raises(ArgumentRefused) as refused:
        exact_allocation(np.ones((3, EXACT_COMMANDS + 1)), np.ones(3), lower=-1.0, upper=1.0, scales=np.ones(3),
                         owner=OWNER)
    (refusal,) = refused.value.refusals
    assert refusal.code == "allocation.exact-size" and "iterative_allocation" in refusal.action
    with pytest.raises(ArgumentRefused, match="allocation.wanted"):
        exact_allocation(np.ones((3, 2)), np.ones(4), lower=-1.0, upper=1.0, scales=np.ones(3), owner=OWNER)


def test_every_switch_the_law_evaluates_carries_its_callers_owner_and_the_selection_is_an_argmin():
    with in_dtype("float64"), switch.collecting() as rows:
        exact_allocation(np.asarray([[1.0, 2.0, 0.5], [0.3, -1.0, 2.0]]), np.asarray([4.0, 1.0]), lower=-1.0,
                         upper=1.0, scales=np.ones(2), owner=OWNER)
    assert {row.owner for row in rows} == {OWNER} and sum(row.function == "argmin" for row in rows) == 2


@pytest.mark.parametrize("rows", [3, 2])
def test_every_solve_of_the_law_runs_in_its_registered_site(rows):
    """Three commands against three rows (the QR) and against two (the selected set's pseudo-inverse too): each solve
    primitive sits in a site that owns it, within its bound, and both sites are reached where they are used."""
    with in_dtype("float64"):
        traced = jax.make_jaxpr(lambda a, b: exact_allocation(a, b, lower=-1.0, upper=1.0, scales=jnp.ones(rows),
                                                                owner=OWNER).commands)(jnp.ones((rows, 3)), jnp.ones(rows))
    assert solve_problems(traced.jaxpr) == []
    assert sites_reached(traced.jaxpr) == ({"allocation_exact"} if rows == 3 else
                                           {"allocation_exact", "allocation_least_norm"})


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("make", [allocating_sphere, allocating_field], ids=["force_3_coils", "field_8_coils"])
def test_the_law_runs_inside_the_compiled_step(make, dtype):
    with in_dtype(dtype):
        built = make().build()
        currents = np.asarray(built.run(ms(60), history=True).history["controller"]["write_currents"])
    assert np.all(np.isfinite(currents)) and np.all(np.abs(currents) <= LIMIT) and np.any(np.abs(currents) > 0)


# -- allocate(), re-based ------------------------------------------------------------------------------------------

def test_allocate_reaches_the_optimum_for_a_field_near_a_tenth_of_a_microtesla():
    """A wanted field near a tenth of a microtesla, on rows in T/A: the law's tests are relative, so it meets the field
    to rounding. (BVLS meets this one too: the base's allocate() missed it by 5e-23 T. Where BVLS stops short, on SI
    force rows, is the next test's.)"""
    actuation = CalibratedMultipole(calibration=asset("navion_2")).actuation(m((0.01, -0.005, 0.0)))
    found = allocate(actuation, field=uT((0.1, -0.05, 0.08)), limits=A(25.0))
    assert found.field_miss is not None and found.field_miss < 1e-12 * 0.14e-6 * 1e3


def test_the_law_on_si_force_rows_reaches_the_optimum_where_bvls_unscaled_stops_short():
    """Force rows in N/A (about 1e-6), unscaled: SciPy's BVLS judges its KKT conditions to an absolute 1e-10 and
    stops short on some (the control: one of these is found, its iterations capped at 3^K so that the stop is its
    tolerance's); the law, whose tests are relative, reaches the optimum of every one, as BVLS does on the same rows
    scaled to order one."""
    from scipy.optimize import lsq_linear

    stopped = 0
    rng = np.random.default_rng(11)
    for problem in [p for p in fixed_problems() if p.kind == "force"]:
        for size in (0.05, 0.2, 0.5, 0.9):
            a = problem.matrix
            b = problem.target / np.linalg.norm(problem.target) * float(np.linalg.norm(a, axis=0).sum()) * \
                problem.limit * size * rng.uniform(0.5, 1.0)
            with in_dtype("float64"):
                mine = np.asarray(exact_allocation(a, b, lower=-problem.limit, upper=problem.limit, scales=np.ones(3),
                                                   owner=OWNER).commands)
            best = bvls(Problem(problem.system, problem.kind, a, b, np.ones(3), problem.limit))
            short = lsq_linear(a, b, bounds=(-problem.limit, problem.limit), method="bvls", max_iter=3 ** a.shape[1]).x
            floor = residual_line(Problem(problem.system, problem.kind, a, b, np.ones(3), problem.limit), "float64")
            stopped += np.linalg.norm(a @ short - b) > np.linalg.norm(a @ best - b) + floor
            assert np.linalg.norm(a @ mine - b) <= np.linalg.norm(a @ best - b) + floor
    assert stopped > 0


def test_allocate_weighs_its_rows_by_the_scales_given_and_refuses_one_of_two():
    actuation = CalibratedMultipole(calibration=asset("navion_2")).actuation(m((0.0, 0.0, 0.0)))
    gradient = T_per_m((1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5))
    field_first = allocate(actuation, field=mT((40.0, 0.0, 0.0)), gradient=gradient, limits=A(10.0),
                           field_scale=mT(0.01), gradient_scale=T_per_m(10.0))
    gradient_first = allocate(actuation, field=mT((40.0, 0.0, 0.0)), gradient=gradient, limits=A(10.0),
                              field_scale=mT(100.0), gradient_scale=T_per_m(0.001))
    assert field_first.field_miss is not None and gradient_first.field_miss is not None
    assert field_first.gradient_miss is not None and gradient_first.gradient_miss is not None
    assert field_first.field_miss < gradient_first.field_miss and \
        gradient_first.gradient_miss < field_first.gradient_miss
    with pytest.raises(ArgumentRefused, match="one row scale of two"):
        allocate(actuation, field=mT((1.0, 0.0, 0.0)), gradient=gradient, limits=A(10.0), field_scale=mT(1.0))
    with pytest.raises(ArgumentRefused, match="above zero"):
        allocate(actuation, field=mT((1.0, 0.0, 0.0)), limits=A(10.0), field_scale=mT(0.0))


def test_allocate_on_eight_coils_asked_for_a_field_alone_takes_the_least_current_minimiser():
    """Eight coils and three rows: many currents make the field; the law returns the one of least |i|^2, no more than
    BVLS's own, and no coil is driven to its limit for a field well within reach."""
    from scipy.optimize import lsq_linear

    for system in ("cardiomag", "octomag"):
        actuation = CalibratedMultipole(calibration=asset(system)).actuation(m((0.005, -0.004, 0.003)))
        found = allocate(actuation, field=mT((4.0, -2.0, 3.0)), limits=A(35.0))
        currents = np.asarray(found.currents.value)
        least = np.linalg.pinv(actuation.field) @ np.asarray((4e-3, -2e-3, 3e-3))
        assert not any(found.at_limit) and found.field_miss is not None and found.field_miss < 1e-12
        assert np.max(np.abs(currents - least)) < 1e-9 * 35.0, (system, currents, least)
        held = lsq_linear(actuation.field / 1e-3, np.asarray((4.0, -2.0, 3.0)), bounds=(-35.0, 35.0), method="bvls",
                          max_iter=3 ** 8).x
        assert np.sum(currents ** 2) <= np.sum(held ** 2) * (1 + 1e-9)
