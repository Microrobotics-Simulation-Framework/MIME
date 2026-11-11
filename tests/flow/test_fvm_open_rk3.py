"""RK3 with ports (F2b PR 4b; predictions in ``predictions_rk3_ports.json``): the open node of the explicit stage kind
(``OpenExplicitFlowNode``) on drawn pulsatile open tubes through a narrowing (``rk3_open_cases``).

FB11: the laws held at RK3's stages, the exchange workless to rounding (the law's own books against the flow's, every
step) and route 1's check that every step dissipated, the laws' storage counted with the flow's; its controls, the law
stepped once a step (a first-order exchange residual) and a gift on one step (counted on that step). The inflow's face
flux is its waveform at the step's end; R-8: the Windkessel's ``N_WK`` under RK3 is first order (``term_orders``); the
open tube's RK3 fixed point is the steady open solve's (the stage's port forces at the pressure's weight; its control,
the drive's weight); the start consistent with the openings against the zero start; the backflow term in its lagged
implicit form (R-15), pointwise in each stage, on draws whose whole flow reverses: it acts and never gives energy (its
control, the coefficient negated, flagged)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from open_cases import draw
from rk3_open_cases import PLANTS, STEPS, backflow_books, backflow_fails, books_of, drawn, fixed_point_distance, \
    fixed_point_pair, inflow_reading, point_node, restart_distance, reversing, scan, start_node, transverse_kick

from mime.core import ledger as books
from mime.nodes.flow.fvm import Windkessel
from mime.testing import ledger_closes, term_orders

SEEDS = (0, 1, 2)
#: The fixed point's line over the largest speed: the steady solve's tolerance, 1e-11, carried through the operator.
FIXED_POINT_LINE = 1e-9
#: The start's line on the fluid's transverse speed over the mean, the spike's question 7 point (a straight tube, so the
#: flow stays axial): rounding (1e-14 measured while writing at R_p 100 R_tube; the moves taken before the penalty, 0.13).
KICK_LINE = 1e-10
#: A restart's line on its first step against the uninterrupted run's, over the largest speed: the consistent pressure
#: is not the carried one by RK3's split, first order (2e-4 to 2.7e-3 on draws 0 to 2 while writing); the interior at
#: zero against the openings' prediction 1.8e-2 to 5.2e-2.
RESTART_LINE = 6e-3


@pytest.mark.parametrize("seed", SEEDS)
def test_every_step_dissipates_and_the_exchange_is_workless(dtype, seed):
    """Route 1 on every step (no step gave energy, the Windkessel's storage counted with the flow's), the laws' own
    identity to its floor at every step (the flow's ``-sum_k alpha_k dt p_k q_k`` against the law's books over the
    stages), nothing unattributed, and the inflow's face flux its waveform at the step's end."""
    node = drawn(seed)
    final, inflow = scan(node, STEPS, inflow_reading(node))
    found = books_of(final)
    assert found["negative_steps"] == 0.0 and found["route_one_worst"] > 0.0, found
    assert found["law_identity"] <= 1.0 and found["unattributed"] == 0.0 and found["identity"] == 0.0, found
    assert float(np.max(np.asarray(inflow))) <= 1.0
    judged = ledger_closes(final)
    assert judged.passed, judged.detail


@pytest.mark.parametrize("seed", SEEDS)
def test_control_the_law_stepped_once_a_step_leaves_a_first_order_residual(dtype, monkeypatch, seed):
    """FB11's control: the law held over the whole step and advanced once, the flow still booking its stages: the law's
    own identity fails (a first-order residual, far above its floor)."""
    PLANTS["law_once"](monkeypatch.setattr)
    final, _ = scan(drawn(seed), STEPS)
    assert books_of(final)["law_identity"] > 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_control_a_gift_on_one_step_is_counted_on_that_step(dtype, monkeypatch, seed):
    """Route 1's control (F2's rows, 4, item 8): a gift of twice one step's dissipation, the laws' losses included, is
    counted on that step and on no other."""
    PLANTS["gift_once"](monkeypatch.setattr)
    final, _ = scan(drawn(seed), STEPS)
    assert books_of(final)["negative_steps"] == 1.0


def test_control_an_inflow_off_its_waveform_is_caught(dtype, monkeypatch):
    PLANTS["inflow_scaled"](monkeypatch.setattr)
    node = drawn(0)
    _, inflow = scan(node, STEPS, inflow_reading(node))
    assert float(np.min(np.asarray(inflow))) > 1.0


def _n_wk_sum(point):
    def sums(dt_scale, tolerance_scale):
        del tolerance_scale
        final, _ = scan(point_node(point, dt_scale=dt_scale), int(round(20 / dt_scale)))
        rows = books.laws(books.deliveries(dict(point_node(point).constants())))
        index = rows.index("windkessel:tube.end")
        return {"windkessel": abs(float(np.asarray(final[f"{books.RECORDS}law_numerical"])[index]))}

    return sums


def test_r8_the_windkessel_under_rk3_is_first_order(float64):
    """R-8: backward Euler at each stage, ``N_WK`` a stage's capacitor change squared: summed over a fixed time it
    falls as dt (``term_orders``, Windkessel's declared order, one ensemble of three drawn cases at D/h 8)."""
    cases = [_n_wk_sum(dataclasses.replace(draw(seed), cells=8)) for seed in SEEDS]
    report = term_orders(cases, ["windkessel"], Windkessel.orders)
    assert report.passed, report.failures()


@pytest.mark.parametrize("plant", [None, "forces_at_gamma"])
def test_the_open_tubes_rk3_fixed_point_is_the_steady_open_solve(float64, monkeypatch, plant):
    """RK3's fixed point is the discrete Brinkman problem whatever dt (B2): through a narrowing, Stokes, pressures given
    at both ends, it is the steady open solve's on the same grid to the solve's tolerance (7.1e-13 while writing, 500
    steps); the openings' forces at the drive's weight in place of the pressure's move it by order one (0.72)."""
    if plant:
        PLANTS[plant](monkeypatch.setattr)
    distance = fixed_point_distance(*fixed_point_pair(steps=500))
    assert (distance <= FIXED_POINT_LINE) == (plant is None), distance


@pytest.mark.parametrize("plant", [None, "unit_penalised"])
def test_the_start_and_the_stages_moves_kick_no_transverse_flow(float64, monkeypatch, plant):
    """Decision 14 at the open-ports spike's question 7 point (R_p 100 R_tube, a pressure inlet 30 % above its mean at
    the start): from the start consistent with the velocity and the openings the fluid's transverse speed stays at
    rounding (a straight tube), the openings' moves taken after the penalty's solve as the stage's own pressure
    increment; taken before it (the unit fields divided by the penalty), the plane's solid faces damped and the solid
    behind them not, the fluid is kicked sideways (0.13 of the mean speed while writing)."""
    if plant:
        PLANTS[plant](monkeypatch.setattr)
    node, speed = start_node(100.0)
    kick = transverse_kick(node, speed)
    assert (kick <= KICK_LINE) == (plant is None), kick


def test_the_zero_start_under_the_gauge_is_measured_not_assumed(float64):
    """The record's divergence from a zero start (the spike's, at R_p 100 R_tube) does not occur here: the field sees
    the openings' pressures relative to the gauge's reference, so the level never reaches it, and the zero start's
    increments are the stage's own. Measured: its transverse speed stays at rounding as the consistent start's does; what
    it costs is the restart's offset (below)."""
    node, speed = start_node(100.0)
    assert transverse_kick(node, speed, zero=True) <= KICK_LINE


@pytest.mark.parametrize("seed", SEEDS)
def test_a_state_given_from_outside_restarts_from_its_consistent_pressure(float64, seed):
    """A state given from outside with its pressure unmade restarts from the pressure consistent with its velocity and
    its openings, within the split's first-order offset of the carried pressure; the interior pressure at zero against
    the openings' prediction (the spike's arm that diverged) is off by an order more, on every draw."""
    node = drawn(seed, ledger=False)
    assert restart_distance(node, "consistent") <= RESTART_LINE
    assert restart_distance(node, "mismatched") > RESTART_LINE


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("plant", [None, "backflow_negated"])
def test_the_backflow_term_damps_the_reversed_outlet_in_every_stage(dtype, monkeypatch, seed, plant):
    """R-15's form under RK3: on a draw whose whole flow reverses (the inflow's pulsatile part 1.5 of its mean), the
    backflow term's coefficient lagged at each stage's start and the term implicit in the stage's pointwise solve acts
    on the reversed outlet (its booked work below zero on some steps) and never gives energy (never flagged, no work
    above zero), the books holding; its coefficient negated, an anti-damping, it gives energy and is flagged."""
    if plant:
        PLANTS[plant](monkeypatch.setattr)
    reading = backflow_books(reversing(seed))
    assert reading["outlet_least"] < 0.0, reading
    assert backflow_fails(reading) == (plant is not None), reading
    if plant:
        assert reading["flagged"] > 0.0, reading
