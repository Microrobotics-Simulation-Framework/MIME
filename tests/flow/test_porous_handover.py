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
7. **A held solve gone not finite** is refused by name (``flow.porous-solve-not-finite``), at the node and through the
   experiment, never a flag on a finished run, and only where the solve itself went so (a NaN in the drive is the
   run's own refusal, run.non-finite); controls, that refusal left silent, and every step's start read as finite.
8. **Plugs in series**: two resolved plugs in one tube read the series closed form; one below the line beside another
   is refused (``flow.porous-series``), where unrefused both readings would be far off it; control, that refusal left
   silent.

Each plant is switched by ``MIME_PLANT_<NAME>``, each drawn case by ``DETECTION_SEED`` (``porous_handover_cases``)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import in_dtype
from open_cases import scan
from porous_handover_cases import Forced, beside_a_body, body_twice, forced_draw, gradient_vessel, line_at, line_of, \
    not_finite_draw, occluded_twin, overlap, planted, quiet_draw, seeded, series_tube, tube_reading, twin_node, vessel

import mime.nodes.flow.fvm._porous as porous
from mime.build import RunFailed
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
#: A closing plug's mid-section flux against the box's whole flux, where both are the solve's (above the line), and
#: below it lumen_flux against the reading it counts the plug at: the solve's continuity to its tolerance (1e-8) over
#: the held flux's share (up to about 1e3 times the reading far below the line).
SECTION_LINE, COUNTED_LINE = 1e-6, 1e-4
#: How far two plugs in series, both resolved, may read from the series closed form (gate 1's band at D/h 16).
SERIES_LINE = 0.04
#: The plug's throughflow beside a body at rest far from it, against the run without it.
BESIDE_LINE = 0.02


def feedback(root: float, dh: int) -> float:
    """The held plug's feedback in gate 1's arm: the flux nine times the plug's K passes, less its own, over the open
    tube's, ``32 (9 - 1) K / R^2`` for a plug filling a quarter of the period."""
    return 1024.0 * root ** 2 / dh ** 2


@pytest.mark.parametrize("interior", ["0.1h", "0.01h"])
def test_the_hand_overs_reading_agrees_with_the_resolved_route_where_both_hold(float64, interior):
    """Above the line a closing plug's porous_flux is the solve's flux through its mid-section, which is the box's
    whole flux, lumen_flux, to the solve's tolerance; the hand-over route's reading of the same plug agrees with it."""
    dh, root = 16, 0.03
    resolved = tube_reading(dh, root, interior)
    assert abs(resolved["reading"] / resolved["lumen"] - 1.0) < SECTION_LINE, resolved
    found = overlap(dh, root, interior)
    assert found["flagged"] == 0.0
    assert abs(found["ratio"] - 1.0) < feedback(root, dh) + OVERLAP_LINE[dh], found


def test_a_fixed_inflow_through_a_plug_below_the_line_is_flagged(float64):
    """One drawn plug closing the open vessel below the line, a fixed inflow upstream (the probe runs the draws)."""
    final = vessel(forced_draw(seeded()))
    assert float(final["forced_flagged"]) == 1.0


@pytest.mark.parametrize("case", ["periodic", "pressure", "gap", "twin"])
def test_a_driven_tube_a_gap_and_an_occluded_branch_are_not_flagged(float64, case):
    """Below the line: the periodic tube driven by a body force (the loop present), and at the open vessel's 0.1h line
    the open vessel between a given pressure and a resistance, a coaxial plug of half the lumen's radius with a fixed
    inflow, and two parallel tubes from one inlet with the second closed and a fixed inflow upstream."""
    drawn = forced_draw(seeded())
    if case == "periodic":
        found = tube_reading(16, math.sqrt(drawn.ratio) * line_of(16, "0.01h", "periodic"), "0.01h")
        flagged, solve = found["forced_flagged"], found["solve_flagged"]
    else:
        if case == "twin":
            point, ratio = quiet_draw(seeded())
            held = occluded_twin(point, ratio)
        elif case == "pressure":
            held = vessel(drawn, laws=("pressure", "windkessel"))
        else:
            held = vessel(drawn, radius=0.25)
        # Every step's solve, the cold first among them: the run's counts of a cap, a stagnation and a NaN.
        flagged, solve = float(held["forced_flagged"]), float(np.sum(np.asarray(held["solve_report_sum"])[2:5]))
    # A quiet run is quiet with its solve converged: a flagged or non-finite solve is no evidence of quiet.
    assert (flagged, solve) == (0.0, 0.0)


def _tubes(width: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Two parallel tubes along z on a grid ten cells long from one inlet plane (layer 0), each its own outlet."""
    lumen = np.zeros((width, 6, 10), dtype=bool)
    lumen[1:3, 2:4, :] = lumen[width - 3:width - 1, 2:4, :] = True
    lumen[:, 2:4, 0] = True
    inlet, first, second = (np.zeros_like(lumen) for _ in range(3))
    inlet[:, :, 0] = lumen[:, :, 0]
    first[:width // 2, :, -1], second[width // 2:, :, -1] = lumen[:width // 2, :, -1], lumen[width // 2:, :, -1]
    return lumen, [inlet, first, second]


def _cut(lumen: np.ndarray, x: int, layer: int) -> np.ndarray:
    disc = np.zeros(lumen.shape[:2], dtype=bool)
    disc[x, 2] = True
    return porous.branch_cut(lumen, disc, layer)


def test_the_vessels_topology_says_where_a_law_fixes_the_flux_and_which_plugs_share_a_path():
    """``_porous.forced_by`` and ``series_pairs`` on a grid. A single tube cut across, a fixed inflow upstream, is
    forced, a given pressure upstream not, and no cut nothing; two parallel tubes from one inflow, one cut, are not
    forced, the other's outlet a closed valve (a flux law) they are, and both cut they are both forced (cut together,
    not one at a time). Two cuts in one tube share a path; one in each tube does not."""
    single = np.zeros((6, 6, 10), dtype=bool)
    single[2:4, 2:4, :] = True
    ends = [np.zeros_like(single), np.zeros_like(single)]
    ends[0][:, :, 0], ends[1][:, :, -1] = single[:, :, 0], single[:, :, -1]
    cut = _cut(single, 2, 5)
    assert porous.forced_by(single, ends, [True, False], [cut]) == [True]
    assert porous.forced_by(single, ends, [False, False], [cut]) == [False]
    assert porous.forced_by(single, ends, [True, False], []) == []
    lumen, touches = _tubes(10)
    second = _cut(lumen, 7, 5)
    assert porous.forced_by(lumen, touches, [True, False, False], [second]) == [False]
    assert porous.forced_by(lumen, touches, [True, True, False], [second]) == [True]
    both = [_cut(lumen, 1, 5), second]
    assert porous.forced_by(lumen, touches, [True, False, False], both) == [True, True]
    table = porous.forced_table(lumen, touches, [True, False, False], both)
    assert table.tolist() == [[False, False], [False, False], [False, False], [True, True]]
    assert not porous.series_pairs(lumen, touches, both)[0, 1]
    assert porous.series_pairs(lumen, touches, [second, _cut(lumen, 7, 8)])[0, 1]
    assert porous.series_pairs(single, ends, [cut, _cut(single, 2, 2)])[0, 1]


@pytest.mark.parametrize("case", ["valve", "parallel"])
def test_a_closed_valve_and_parallel_plugs_below_one_inflow_are_flagged(float64, case):
    """The twin rig (drawn, D/h 8 or 12), a fixed inflow at its start, at the open line's 0.1h interior: the second tube
    plugged below the line and the first's outlet a closed valve, so every bit of the inflow crosses the plug; or each
    tube plugged, the outlets resistances, so the inflow crosses the two together."""
    point, ratio = quiet_draw(seeded())
    settings = {"valve": dict(outlets="valve"), "parallel": dict(tubes=(0, 1))}[case]
    final = occluded_twin(point, ratio, **settings)
    assert float(np.sum(np.asarray(final["solve_report_sum"])[2:5])) == 0.0
    assert float(final["forced_flagged"]) == 1.0


def test_two_resolved_plugs_in_series_read_the_closed_form_and_one_below_the_line_is_refused(float64):
    """A periodic tube at D/h 16 (the 0.01h interior's line 0.01) closed by two plugs in series: both above the line
    (sqrt(K)/h 0.03 and 0.1) read the series closed form in porous_flux and lumen_flux; the second below it (1e-3,
    held at the line) is refused flow.porous-series, and its readings, unrefused, are far off the closed form (the
    refusal's premise: the held plug passes the line's flux, which the resolved one reports)."""
    node, final, closed = series_tube(16, (0.03, 0.1))
    assert float(np.sum(np.asarray(final["solve_report_sum"])[2:5])) == 0.0
    assert all(abs(float(q) / closed - 1.0) < SERIES_LINE for q in [*np.asarray(final["porous_flux"]),
                                                                     final["lumen_flux"]]), (final["porous_flux"], closed)
    assert float(final["porous_series"]) == 0.0 and node.stop_problem(final) is None
    with planted():
        node, final, closed = series_tube(16, (0.03, 1e-3))
        problem = node.stop_problem(final)
    readings = np.asarray(final["porous_flux"]) / closed
    assert readings[0] > 10.0, readings
    assert problem is not None and problem.code == "flow.porous-series", problem


@pytest.mark.parametrize("dh", [16])
def test_far_below_the_line_the_readings_error_is_constant_in_k(float64, dh):
    """Two plugs a decade apart far below the 0.01h interior's line (sqrt(K)/h 1e-3 and 1e-4): the reading against
    the series closed form within the band, the two within the spread of each other, far under the carried region's
    share of the lumen (4 h / R, a half at D/h 16)."""
    found = [tube_reading(dh, root, "0.01h") for root in (1e-3, 1e-4)]
    # Below the line lumen_flux counts the plug at its reading, not at the held flux through it.
    assert all(abs(f["lumen"] / f["reading"] - 1.0) < COUNTED_LINE and f["solve_flagged"] == 0.0 for f in found), found
    errors = [f["reading"] - 1.0 for f in found]
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
    bead within the line, each run's solve converged (a stagnated pair is no evidence of it: below the line at D/h 20
    the held solve takes the D/h 32 row's line, where it converges); below the line ``porous_flagged`` is raised (the
    forced-flow test does not see a body's displacement), and not without the bead."""
    finals = []
    with body_twice():
        for bead in (True, False):
            result = beside_a_body(permeability, bead=bead).build().run(ms(0.25))
            judged = ledger_closes(result)
            assert judged.passed, judged.detail
            finals.append(result.final["water.flow"])
    assert all(float(np.sum(np.asarray(final["solve_report_sum"])[2:5])) == 0.0 for final in finals)
    with_bead, without = (float(np.asarray(final["porous_flux"])[0]) for final in finals)
    assert abs(with_bead / without - 1.0) < BESIDE_LINE
    assert [float(final["porous_flagged"]) for final in finals] == [float(permeability < 1e-13), 0.0]


def test_the_flow_forced_through_a_plug_is_warned_of_through_the_experiment(float64):
    """The forced-flow flag is laid out for a flow holding a plug and warned of after a run that raised it."""
    import warnings

    from porous_cases import open_experiment

    K = 0.1 * (porous.HANDOVER_LINE["open"]["0.1h"] * 1e-3 / 16) ** 2
    built = open_experiment(K).build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final = built.run(ms(0.5)).final["water.flow"]
    assert float(final["forced_flagged"]) == 1.0
    assert any(porous.FORCED_FLAG in str(w.message) for w in caught)
    line = next(line for line in built.report.text().splitlines() if "the porous region" in line)
    assert "below the hand-over's line" in line and "an open vessel" in line and "43213789033" in line


def test_drawn_forced_cases_are_physical():
    """The draws stay where the gate's claim is: below the line by at least a half, inside the vessel."""
    for index in range(50):
        drawn = forced_draw(index)
        assert 1e-4 <= drawn.ratio <= 0.5 and 0.7 <= drawn.at <= 1.3 and drawn.at + drawn.length / 2 < 2.0
    assert isinstance(forced_draw(0), Forced)


def test_a_held_solve_that_goes_not_finite_is_refused_by_name(float64):
    """The twin rig at D/h 12, its line moved to the periodic box's old 0.1h line (sqrt(K)/h 0.003), where its held
    solve went not finite on every such draw (run 43201390806): the node's stop names it, with its line and box type."""
    point, ratio = not_finite_draw(seeded())
    with line_at(0.003), planted():
        node = twin_node(point, ratio)
        final, _ = scan(node, 2)
        problem = node.stop_problem(final)
    # The premise: the held solve went not finite from a finite start (a draw on which it did not is no evidence).
    assert float(final["porous_not_finite"]) > 0.0
    assert problem is not None and problem.code == "flow.porous-solve-not-finite", problem
    assert "an open vessel" in problem.message and "0.003" in problem.message


@pytest.mark.parametrize("cause", ["drive", "solve"])
def test_a_porous_flows_non_finite_run_is_named_by_its_cause_through_the_experiment(float64, cause):
    """The open vessel holding a plug below the line, its velocity gone not finite: by a NaN in the step's drive from
    its third step (planted as the flow's own NaN tests plant it), which is the run's own refusal, run.non-finite,
    naming the first node and field; or by the solve itself, its solution NaN from a finite start and drive (planted
    at every step), which is refused flow.porous-solve-not-finite, naming the region."""
    import jax.numpy as jnp
    from porous_cases import open_experiment

    import mime.nodes.flow.fvm._node as flow_node
    import mime.nodes.flow.fvm._open_node as open_node

    made, solved = flow_node.ImplicitFlowNode.forcing, open_node.projected_solve

    def forcing(self, fields, boundary_inputs, scheme, dtype_):
        force = made(self, fields, boundary_inputs, scheme, dtype_)
        return lambda stage, time: tuple(jnp.where(time > 2.5 * self.delta_t, jnp.nan, c) for c in force(stage, time))

    def solve(grid, operator, diagonal, rhs, *rest):
        velocity, report = solved(grid, operator, diagonal, rhs, *rest)
        return tuple(c * jnp.nan for c in velocity), report

    K = 0.1 * (line_of(16, "0.1h", "open") * 1e-3 / 16) ** 2
    with pytest.MonkeyPatch.context() as patched, planted():
        if cause == "drive":
            patched.setattr(flow_node.ImplicitFlowNode, "forcing", forcing)
        else:
            patched.setattr(open_node, "projected_solve", solve)
        built = open_experiment(K).build()
        with pytest.raises(RunFailed) as raised:
            built.run(ms(1.0))
    codes = [r.code for r in raised.value.refusals]
    result = raised.value.result
    assert result is not None
    counted = float(np.asarray(result.final["water.flow"]["porous_not_finite"]))
    if cause == "drive":
        assert codes == ["run.non-finite"] and counted == 0.0, (codes, counted)
    else:
        assert codes == ["flow.porous-solve-not-finite"] and counted > 0.0, (codes, counted)
        assert "darcy" in raised.value.refusals[0].message
