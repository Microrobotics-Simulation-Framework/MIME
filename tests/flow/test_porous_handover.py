"""The porous plug's hand-over below the resolved line (``tests/flow/predictions_porous_handover.json`` holds their
predictions, committed before they ran): below ``sqrt(K)/h`` at the line the solve holds the plug at the line's
permeability and its throughflow is read one way from its face pressures.

1. **The overlap**: where both routes hold, the hand-over route's reading agrees with the resolved ``lumen_flux``;
   controls, the reading's permeability doubled, the face pressures' signs flipped.
2. **The forced-flow test**: a fixed inflow through a plug below the line closing a single tube is flagged; a
   pressure-driven tube, a plug leaving a gap, and two parallel tubes with one closed and a fixed inflow upstream are
   not; controls, the comparison disabled, the topology answering "forced" everywhere.
3. **The carried depth**: far below the line the reading's error against the series closed form is constant in K;
   control, the reading's permeability doubled.
4. **Gradients**, float64, in the open vessel: the reading in K and in the inflow, forward and reverse against central
   differences, and the dot-product check; controls, ``stop_gradient`` on the face pressures and on K in the reading.
5. **Float32**: the reading below the line within its rounding of float64.
6. **Beside a resolved body**: a plug and a ``Drag.Penalty`` body in one flow build, close their books and leave the
   plug's throughflow; control, a body's loss taken off the walls' penalty twice.

Each plant is switched by ``MIME_PLANT_<NAME>``, each drawn case by ``DETECTION_SEED`` (``porous_handover_cases``)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import in_dtype
from porous_handover_cases import Forced, beside_a_body, body_twice, forced_draw, gradient_vessel, occluded_twin, \
    overlap, planted, quiet_draw, seeded, tube_reading, vessel

import mime.nodes.flow.fvm._porous as porous
from mime.testing import ledger_closes
from mime.units import ms

#: Gate 1's line beyond the held plug's feedback ``f``, by D/h (the reading's face error, #264's within 3.5 %).
OVERLAP_LINE = {16: 0.02, 32: 0.01}
#: Gate 3's band on the reading against the series closed form at D/h 16, and its spread across K.
CARRIED_BAND, CARRIED_SPREAD = (-0.04, 0.01), 1e-3
#: Gate 4's lines against the central difference: in K (the reading's own, linear in K times Brinkman's factor) and in
#: the inflow (through the solve at its tolerance of 1e-12); the dot product's.
PERMEABILITY_LINE, INFLOW_LINE, DOT_LINE = 1e-6, 1e-5, 1e-10
FLOAT32_LINE = 1e-3
#: The plug's throughflow beside a body at rest far from it, against the run without it.
BESIDE_LINE = 0.02


def feedback(root: float, dh: int) -> float:
    """The held plug's feedback in gate 1's arm: the flux nine times the plug's K passes, less its own, over the open
    tube's, ``32 (9 - 1) K / R^2`` for a plug filling a quarter of the period."""
    return 1024.0 * root ** 2 / dh ** 2


@pytest.mark.parametrize("interior", ["0.1h", "0.01h"])
def test_the_hand_overs_reading_agrees_with_the_resolved_route_where_both_hold(float64, interior):
    dh, root = 16, 0.03
    found = overlap(dh, root, interior)
    assert found["flagged"] == 0.0
    assert abs(found["ratio"] - 1.0) < feedback(root, dh) + OVERLAP_LINE[dh], found


def test_a_fixed_inflow_through_a_plug_below_the_line_is_flagged(float64):
    """One drawn plug closing the open vessel below the line, a fixed inflow upstream (the probe runs the draws)."""
    final = vessel(forced_draw(seeded()))
    assert float(final["forced_flagged"]) == 1.0


@pytest.mark.parametrize("case", ["periodic", "pressure", "gap", "twin"])
def test_a_driven_tube_a_gap_and_an_occluded_branch_are_not_flagged(float64, case):
    """Below the line: the periodic tube driven by a body force (the loop present), the open vessel between a given
    pressure and a resistance, a coaxial plug of half the lumen's radius with a fixed inflow, and two parallel tubes
    from one inlet with the second closed and a fixed inflow upstream."""
    drawn = forced_draw(seeded())
    if case == "periodic":
        found = tube_reading(16, math.sqrt(drawn.ratio) * porous.HANDOVER_LINE["0.01h"], "0.01h")
        flagged, solve = found["forced_flagged"], found["solve_flagged"]
    else:
        if case == "twin":
            point, ratio = quiet_draw(seeded())
            # The 0.01h interior: at the 0.1h line the rig's solve (the held plug's, as the resolved one's at the line)
            # went not finite at D/h 12, flagged (the PR states it); the case tests the topology, not that solve.
            held = occluded_twin(point, ratio, interior="0.01h")
        else:
            held = vessel(drawn, laws=("pressure", "windkessel")) if case == "pressure" else vessel(drawn, radius=0.25)
        flagged, solve = float(held["forced_flagged"]), float(held["solve_flagged"])
    # A quiet run is quiet with its solve converged: a flagged or non-finite solve is no evidence of quiet.
    assert (flagged, solve) == (0.0, 0.0)


def test_the_vessels_topology_says_where_a_law_fixes_the_flux():
    """``_porous.isolated`` on a grid: a single tube cut across isolates its inlet; two parallel tubes from one inlet
    plane, one cut, do not; with no cut nothing is isolated."""
    single = np.zeros((6, 6, 10), dtype=bool)
    single[2:4, 2:4, :] = True
    touches = [np.zeros_like(single), np.zeros_like(single)]
    touches[0][:, :, 0], touches[1][:, :, -1] = single[:, :, 0], single[:, :, -1]
    disc = np.zeros((6, 6), dtype=bool)
    disc[2, 2] = True
    assert porous.isolated(single, touches, [0], porous.branch_cut(single, disc, 5))
    assert not porous.isolated(single, touches, [0], np.zeros_like(single))
    twins = np.zeros((10, 6, 10), dtype=bool)
    twins[1:3, 2:4, :] = twins[7:9, 2:4, :] = True
    twins[:, 2:4, 0] = True  # one inlet plane joining both
    inlet, first, second = (np.zeros_like(twins) for _ in range(3))
    inlet[:, :, 0] = twins[:, :, 0]
    first[:5, :, -1], second[5:, :, -1] = twins[:5, :, -1], twins[5:, :, -1]
    disc = np.zeros((10, 6), dtype=bool)
    disc[7, 2] = True
    assert not porous.isolated(twins, [inlet, first, second], [0], porous.branch_cut(twins, disc, 5))


@pytest.mark.parametrize("dh", [16])
def test_far_below_the_line_the_readings_error_is_constant_in_k(float64, dh):
    """Two plugs a decade apart far below the 0.01h interior's line (sqrt(K)/h 1e-3 and 1e-4): the reading against
    the series closed form within the band, the two within the spread of each other, far under the carried region's
    share of the lumen (4 h / R, a half at D/h 16)."""
    errors = [tube_reading(dh, root, "0.01h")["reading"] - 1.0 for root in (1e-3, 1e-4)]
    assert all(CARRIED_BAND[0] < error < CARRIED_BAND[1] for error in errors), errors
    assert abs(errors[0] - errors[1]) < CARRIED_SPREAD, errors


@pytest.mark.parametrize("leaf", [0, 1], ids=["permeability", "inflow"])
def test_the_reading_below_the_line_is_differentiable_in_k_and_the_inflow(float64, leaf):
    """The open vessel's plug at 0.3 of the line's K: forward (``jvp``) and reverse (``vjp``) of its reading against the
    central difference, and the dot product of the two over the reading and the face pressures."""
    with planted():
        outputs, x, _ = gradient_vessel()
        direction = jnp.zeros(2, dtype=jnp.float64).at[leaf].set(1.0)
        step = 1e-4 * x[leaf]
        _, forward = jax.jvp(outputs, (x,), (direction,))
        _, pull = jax.vjp(outputs, x)
        central = (outputs(x + step * direction) - outputs(x - step * direction)) / (2.0 * step)
        reverse = pull(jnp.zeros(3, dtype=jnp.float64).at[0].set(1.0))[0][leaf]
        v, w = jnp.asarray([0.5 * x[0], 0.2], dtype=jnp.float64), jnp.asarray([1.0, 0.3, -0.7], dtype=jnp.float64)
        _, along = jax.jvp(outputs, (x,), (v,))
        left, right = float(jnp.vdot(along, w)), float(jnp.vdot(v, pull(w)[0]))
    line = PERMEABILITY_LINE if leaf == 0 else INFLOW_LINE
    scale = abs(float(central[0]))
    assert abs(float(forward[0]) - float(central[0])) < line * scale
    assert abs(float(reverse) - float(central[0])) < line * scale
    assert abs(left - right) < DOT_LINE * max(abs(left), abs(right))


def test_float32_reads_a_plug_below_the_line_within_its_rounding_of_float64():
    """IMEX Euler, six steps of a warm-started solve (float32's steady cold solve at the line stagnates near 1e-4,
    flagged; the probe states float32's range)."""
    read = []
    for dtype in ("float64", "float32"):
        with in_dtype(dtype):
            found = tube_reading(16, 1e-3, "0.01h", steps=6, scheme="imex_euler", dt=0.002)
            assert found["solve_flagged"] == 0.0
            read.append(found["reading"])
    assert abs(read[1] / read[0] - 1.0) < FLOAT32_LINE


@pytest.mark.parametrize("permeability", [1e-10, 1e-15], ids=["above", "below"])
def test_a_plug_beside_a_resolved_body_closes_its_books_and_keeps_its_throughflow(float64, permeability):
    """The plug and a bead resolved on the grid (``Drag.Penalty``) in one periodic tube at one spacing, the bead at rest
    at the first step a millimetre from the plug: the books close; the plug's throughflow is the run's without the
    bead within the line; below the line ``porous_flagged`` is raised (the forced-flow test does not see a body's
    displacement), and not without the bead."""
    finals = []
    with body_twice():
        for bead in (True, False):
            result = beside_a_body(permeability, bead=bead).build().run(ms(0.25))
            judged = ledger_closes(result)
            assert judged.passed, judged.detail
            finals.append(result.final["water.flow"])
    with_bead, without = (float(np.asarray(final["porous_flux"])[0]) for final in finals)
    assert abs(with_bead / without - 1.0) < BESIDE_LINE
    assert [float(final["porous_flagged"]) for final in finals] == [float(permeability < 1e-13), 0.0]


def test_the_flow_forced_through_a_plug_is_warned_of_through_the_experiment(float64):
    """The forced-flow flag is laid out for a flow holding a plug and warned of after a run that raised it."""
    import warnings

    from porous_cases import open_experiment

    K = 0.1 * (porous.HANDOVER_LINE["0.1h"] * 1e-3 / 16) ** 2
    built = open_experiment(K).build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final = built.run(ms(0.5)).final["water.flow"]
    assert float(final["forced_flagged"]) == 1.0
    assert any(porous.FORCED_FLAG in str(w.message) for w in caught)
    line = next(line for line in built.report.text().splitlines() if "the porous region" in line)
    assert "below the hand-over's line" in line and "43123280055" in line


def test_drawn_forced_cases_are_physical():
    """The draws stay where the gate's claim is: below the line by at least a half, inside the vessel."""
    for index in range(50):
        drawn = forced_draw(index)
        assert 1e-4 <= drawn.ratio <= 0.5 and 0.7 <= drawn.at <= 1.3 and drawn.at + drawn.length / 2 < 2.0
    assert isinstance(forced_draw(0), Forced)
