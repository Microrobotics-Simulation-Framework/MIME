"""RK3 with ports (F2b PR 4b; predictions in ``predictions_rk3_ports.json``): the open node of the explicit stage kind
(``OpenExplicitFlowNode``) on drawn pulsatile open tubes through a narrowing (``rk3_open_cases``).

FB11: the laws held at RK3's stages, the exchange workless to rounding (the law's own books against the flow's, every
step) and route 1's check that every step dissipated, the laws' storage counted with the flow's; its controls, the law
stepped once a step (a first-order exchange residual) and a gift on one step (counted on that step). The inflow's face
flux is its waveform at the step's end; R-8: the Windkessel's ``N_WK`` under RK3 is first order (``term_orders``); the
open tube's RK3 fixed point is the steady open solve's (the stage's port forces at the pressure's weight; its control,
the drive's weight); the start consistent with the openings against the zero start."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from open_cases import draw
from rk3_open_cases import PLANTS, STEPS, books_of, drawn, inflow_reading, point_node, scan

from mime.core import ledger as books
from mime.nodes.flow.fvm import Windkessel
from mime.testing import ledger_closes, term_orders

SEEDS = (0, 1, 2)


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
