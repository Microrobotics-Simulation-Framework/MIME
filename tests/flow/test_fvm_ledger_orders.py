"""CI check (b) (row #301, R-19), on the ledger harness's order check (``mime.testing.term_orders``): each numerical
term the flow's step identity books falls at the order its scheme declares (``TimeScheme.ledger_orders``), so a
physical loss written into an ``N`` formula, which the identity books whatever it is called and closes all the same, is
caught; and every scheme declares an order, or why it has none, for every term it books.

On a smooth swirl in the periodic tube with its Poiseuille drive (IMEX Euler, convection on, D/h 8, a fixed time of
0.05 in units of D and nu, the solve at each dtype's default tolerance), each term's magnitude summed over the run, at
the rig's step and tolerance and with each halved apart: ``N_BE`` and ``N_conv``, declared second order in dt, fall by
1.6 to 2.4 as dt halves (``2^(2 - 1)`` within the harness's band); ``N_Krylov``, declared at least first order in the
tolerance (its stop bounds it), falls by at least 1.6 as the tolerance halves. The steady limit's two zero terms and
RK3's convective power (zero to rounding on the box) declare no order, with their reasons.

Every arm starts from one warmed state (the swirl and its drive after a short run at a quarter of the step), so the
stiff interior's start, damped in a step whatever the step, is not in the sums; and each arm's books close
(``ledger_closes``).

**The controls, each failing every seed**: ``N_BE`` and ``N_conv`` moved to the wrong order, each scaled by ``(dt /
dt_rig)^k`` (``k = -1``: first order; ``k = +1``: third), the difference moved into the viscous row so the identity
still closes; and the viscous loss written into ``N_Krylov`` (#225's control), of order zero in the tolerance. Over-budget
cases go on the duration allowlist, never under ``slow``.

**The resolution** (#225's, on this rig, three seeds): a physical loss of ``k`` times the term's own coarse sum, moved
from the viscous row into the term, leaves the band on one seed of the three from ``k = 0.1`` (``N_BE`` at 1.58,
``N_conv`` at 1.45), which fails the test, and on all three from about 0.6 for ``N_BE``. **A gift a term's own formula
books** (convection's, at mixed time levels) is booked under that term, where the step identity cannot see it: the
order check and the convection flag (``mime.nodes.flow.fvm._flags``) are what see it."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import METERED, in_dtype, tube_node

from mime.core import ledger as books
from mime.nodes.flow.fvm import SCHEMES, ImplicitFlowNode
from mime.testing import ledger_closes, term_orders
from mime.testing._ledger_harness import expected_ratio

#: The fixed time each term is summed over (D = 1, nu = 1: about a fifth of the swirl's decay time).
TIME = 0.05
#: The step of the arms, and the warm-up's: a twentieth of the run at a quarter of the step, which the stiff interior
#: and the drive's start settle in (the penalty damps the solid's start in a step whatever the step is).
STEP, WARM_STEP, WARM_STEPS = 0.0025, 0.000625, 20
#: The rig's tolerance in each dtype: its default.
TOLERANCE = {"float64": 1e-8, "float32": 1e-5}
#: The drive, constant: the tube's Poiseuille body force.
DRIVE = 8.0
#: Each term's index in the records (``ImplicitFlowNode.LEDGER_TERMS``).
TERMS = {name: index for index, name in enumerate(ImplicitFlowNode.LEDGER_TERMS)}


def cellular(node: ImplicitFlowNode, seed: int) -> dict[str, Any]:
    """A smooth swirl across the tube, uniform along it (a cellular flow of the box's width with a drawn phase in
    each component), projected: convection does work at mixed time levels, and every mode is resolved in time."""
    rng = np.random.default_rng(seed)
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    a, b, c = rng.uniform(0.0, 2.0 * np.pi, 3)
    width = grid.cells[0] * grid.spacing
    points = [np.asarray(grid.face_points(k, "float64")) for k in range(3)]
    u = np.sin(2.0 * np.pi * points[0][..., 1] / width + a)
    v = np.sin(2.0 * np.pi * points[1][..., 0] / width + b)
    w = np.sin(2.0 * np.pi * (points[2][..., 0] + points[2][..., 1]) / width + c)
    swirl = grid.project((u, v, w))
    return {f"initial_velocity_{k}": np.asarray(swirl[i]) for i, k in enumerate("xyz")}


def wrong_order(term: str, argument: str, power: int | None):
    """The plant: ``term`` scaled by ``(dt / STEP)^power`` (or by the tolerance over the rig's), the difference moved
    into the viscous row, so that the identity still closes and the term falls at its order plus ``power``; with
    ``power`` None, the viscous loss moved into the term whole (#225's control: a physical loss written into an ``N``
    formula, of order zero in the tolerance)."""
    real = ImplicitFlowNode.energy_terms

    def planted(self, p, grid, before, stage, sigma, fields, dt):
        terms = dict(real(self, p, grid, before, stage, sigma, fields, dt))
        if power is None:
            terms[term] = terms[term] + terms["viscous"]
            terms["viscous"] = 0.0 * terms["viscous"]
            return terms
        rig = STEP if argument == "dt" else TOLERANCE[str(jnp.asarray(terms[term]).dtype)]
        setting = dt if argument == "dt" else p["tolerance"]
        moved = terms[term] * (setting / rig) ** power
        terms["viscous"] = terms["viscous"] + terms[term] - moved
        terms[term] = moved
        return terms

    return planted


def sums_of(dtype: str, seed: int) -> tuple[Any, list[Any]]:
    """The rig's ``sums(dt_scale, tolerance_scale)`` for ``seed`` (each term's summed magnitude over :data:`TIME`), and
    the list each arm's ended records are put in. One compiled program for every arm: the step, the run's length and
    the tolerance are its arguments."""
    tolerance = TOLERANCE[dtype]
    with in_dtype(dtype):
        node = tube_node(8, length_cells=4, dt=STEP, convection=True, tolerance=tolerance, ledger=METERED)
        p0 = node._step_constants(None)  # noqa: SLF001
        held = books.deliveries(node.constants())
        given = {books.law_input(d.field, index): jnp.asarray([0.0, 0.0, DRIVE]) if d.field == "body_force"
                 else jnp.zeros(7) for index, d in enumerate(held) if d.kind == books.LOAD}

        def arm(state, dt, steps, tolerance):
            p = {**p0, "tolerance": tolerance}

            def body(_, carry):
                s, sums = carry
                s = node.advance(p, s, given, dt)
                return s, sums + jnp.abs(s[f"{books.RECORDS}step_terms"])

            return jax.lax.fori_loop(0, steps, body, (state, jnp.zeros(len(TERMS), dtype=dt.dtype)))

        compiled = jax.jit(arm)
        dtype_of = jnp.asarray(0.0).dtype
        start = node.initial_state()
        swirl = cellular(node, seed)
        begin = books.with_records(node, books.unrecorded({**start, **{f"velocity_{c}": jnp.asarray(
            swirl[f"initial_velocity_{c}"], dtype=dtype_of) for c in "xyz"}}))
        warm, _ = compiled(begin, jnp.asarray(WARM_STEP, dtype=dtype_of), WARM_STEPS, jnp.asarray(tolerance,
                                                                                                  dtype=dtype_of))
        fresh = books.with_records(node, books.unrecorded({**start, **{f"velocity_{c}": warm[f"velocity_{c}"]
                                                                        for c in "xyz"}}))
    ended: list[Any] = []

    def sums(dt_scale: float, tolerance_scale: float) -> dict[str, float]:
        with in_dtype(dtype):
            dt = STEP * dt_scale
            state, summed = compiled(fresh, jnp.asarray(dt, dtype=dtype_of), int(round(TIME / dt)),
                                     jnp.asarray(tolerance * tolerance_scale, dtype=dtype_of))
        ended.append(state)
        return {name: float(summed[index]) for name, index in TERMS.items()}

    return sums, ended


def _report(dtype: str, seed: int):
    sums, ended = sums_of(dtype, seed)
    scheme = SCHEMES["imex_euler"]
    return term_orders(sums, scheme.ledger_terms, scheme.ledger_orders), ended


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_each_numerical_term_falls_at_its_declared_order(dtype):
    for seed in range(3):
        report, ended = _report(dtype, seed)
        assert report.passed, (seed, report.failures(), report.checks)
        assert [(c.term, c.argument) for c in report.checks] == [("backward_euler", "dt"), ("convection", "dt"),
                                                                 ("krylov", "tolerance")]
        for state in ended:
            judged = ledger_closes(state)
            assert judged.passed, judged.detail


PLANTS = [("backward_euler", "dt", -1), ("backward_euler", "dt", 1), ("convection", "dt", -1),
          ("convection", "dt", 1), ("krylov", "tolerance", None)]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(("term", "argument", "power"), PLANTS,
                         ids=[f"{t}-{a}{'-loss' if p is None else f'{p:+d}'}" for t, a, p in PLANTS])
def test_control_a_term_of_the_wrong_order_fails_on_every_seed(dtype, term, argument, power, monkeypatch):
    """The identity still closes with the term moved, and its check fails on every seed: at order 1 or 3 where 2 is
    declared (ratios about 1 and 4 against 1.6 to 2.4), or near order 0 where at least 1 is."""
    monkeypatch.setattr(ImplicitFlowNode, "energy_terms", wrong_order(term, argument, power))
    for seed in range(3):
        report, ended = _report(dtype, seed)
        failed = [check for check in report.checks if not check.passed]
        assert [(c.term, c.argument) for c in failed] == [(term, argument)], (seed, report.checks)
        assert all(ledger_closes(state).passed for state in ended)


def test_every_scheme_declares_an_order_or_its_reason_for_every_term_it_books():
    for scheme in SCHEMES.values():
        assert set(scheme.ledger_orders) == set(scheme.ledger_terms), scheme.name
        for term, order in scheme.ledger_orders.items():
            assert order.dt is not None or order.tolerance is not None or order.reason, (scheme.name, term)


# -- the harness itself, on exact power laws -------------------------------------------------------------------------

def _laws(orders: dict[str, tuple[float, float]]):
    """``sums`` of terms whose sums are exact powers: ``(p, q)`` the dt and tolerance orders of each, summed over a
    fixed time (``dt^(p - 1)``) and a fixed count of steps (``tol^q``)."""
    return lambda dt, tol: {term: dt ** (p - 1) * tol ** q for term, (p, q) in orders.items()}


def test_the_harness_passes_exact_orders_and_fails_one_off_and_an_undeclared_term():
    from mime.core.ledger import TermOrder

    declared = {"a": TermOrder(dt=2), "b": TermOrder(tolerance=1, at_least=True), "c": TermOrder(reason="zero")}
    assert term_orders(_laws({"a": (2, 0), "b": (0, 1), "c": (0, 0)}), ["a", "b", "c"], declared).passed
    assert term_orders(_laws({"a": (2, 0), "b": (0, 3), "c": (0, 0)}), ["a", "b", "c"], declared).passed
    for wrong in ({"a": (1, 0), "b": (0, 1), "c": (0, 0)}, {"a": (3, 0), "b": (0, 1), "c": (0, 0)},
                  {"a": (2, 0), "b": (0, 0), "c": (0, 0)}):
        assert not term_orders(_laws(wrong), ["a", "b", "c"], declared).passed
    report = term_orders(_laws({"a": (2, 0), "d": (2, 0)}), ["a", "d"], declared)
    assert report.undeclared == ("d",) and not report.passed
    assert expected_ratio("dt", 2) == 2.0 and expected_ratio("tolerance", 1) == 2.0
