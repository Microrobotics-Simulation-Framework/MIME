"""The exact allocation, ``mime.control.exact_allocation``: against SciPy's BVLS, its derivative, its guards, its
refusal, and ``allocate()`` re-based on it.

**The swap test** draws problems on the three arrays MIME ships (``allocation_draws``) and holds the law to BVLS at
camera spike R2's line in both dtypes, limits inactive and active. Its plant (``ALLOCATION_PLANT=limits``): the law
is handed limits 1 % looser.

**The derivative**, within a draw's active set, is affine in the wanted values and the limit, so central differences
over a segment whose ends hold the same active set are exact up to rounding, and judge it to the line over the step;
forward and reverse mode agree to rounding; float32's is judged against float64's at the same problem. Its plants:
``free_tangent`` (the commands' derivative the unconstrained least squares', a wrong active set) and ``stop_branch``
(the derivative of what the bound commands leave the free ones stopped). At an active-set change the derivative jumps
and stays finite.

**The guards** are tested structurally: each ``domain_guard`` of the law is reached with its condition false by one of
``allocation_cases.GUARD_POINTS``, and replacing its fallback there changes no output and no derivative. Its plant
(``single_where``): the stand-ins removed, which no point then reaches."""

from __future__ import annotations

import contextlib
import inspect
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from allocation_cases import (GUARD_POINTS, LIMIT, OWNER, allocating_field, allocating_sphere, guard_point_derivative,
                              guard_point_value)
from allocation_draws import SLACK, Problem, bvls, line, problems, reach_of, residual_line, verdict
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import mime.control._allocation as law_module
from mime.build._refusals import ArgumentRefused
from mime.control import EXACT_COMMANDS, exact_allocation
from mime.core import switch
from mime.data import asset
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
    assert refusal.code == "allocation.exact-size" and "iterative path" in refusal.action
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
    """BVLS's absolute tolerance stops it short on rows this small; the law's tests are relative."""
    actuation = CalibratedMultipole(calibration=asset("navion_2")).actuation(m((0.01, -0.005, 0.0)))
    found = allocate(actuation, field=uT((0.1, -0.05, 0.08)), limits=A(25.0))
    assert found.field_miss is not None and found.field_miss < 1e-12 * 0.14e-6 * 1e3


def test_the_law_on_si_force_rows_reaches_the_optimum_where_bvls_unscaled_stops_short():
    """Force rows in N/A (about 1e-6), unscaled: SciPy's BVLS judges its KKT conditions to an absolute 1e-10 and
    stops short on some (the control: one of these is found); the law, whose tests are relative, reaches the optimum
    of every one, as BVLS does on the same rows scaled to order one."""
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
            short = lsq_linear(a, b, bounds=(-problem.limit, problem.limit), method="bvls").x
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
        held = lsq_linear(actuation.field / 1e-3, np.asarray((4.0, -2.0, 3.0)), bounds=(-35.0, 35.0), method="bvls").x
        assert np.sum(currents ** 2) <= np.sum(held ** 2) * (1 + 1e-9)
