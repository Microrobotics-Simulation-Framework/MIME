"""F4 PR 1's gates: a resolved body on the FVM's grid (``Drag.Penalty``), its coupling lagged and declared. Predictions
committed first (``tests/flow/predictions_f4.json``). The controls honour ``MIME_PLANT_<NAME>``
(``resolved_cases.plant``), for ``.github/probe/detection_rate.py``.

- **The Re -> 0 ladder** (gate 1): the steady sphere's drag against Haberman and Sayre at 3 and 6 cells a radius, the
  radius the drag reads (smaller by a shift in cells, which falls with N: the shape-and-position form, never a bound on
  the nominal radius).
- **The lagged coupling** (gate 2, re-measured at the default N = 6 in PR 259's fix round): R3's case at Re 76 runs
  unstopped and bounded at ratios 0.3 and 1.4; its stability is judged at rest (gate 2b): at the largest step the
  refusal admits (its margin's share of the edge measured in dt/tau) the loop decays by energy; past the edge the
  build refuses (drag.penalty-lagged) and a run that gets there is stopped by its body's motion (gate G). The
  rigid-interior shortcut diverges at 1.2.
- **The displaced share** (gate 3): a neutral sphere in a free box driven by a uniform force accelerates with the
  fluid (control: the share left out).
- **The books** (gate 4): metered, every receiver's books close, the drag judged at one velocity by its host.
- **The ripple and the position derivative** (gates 5 and 6), **float32** (gate 7), the spacing rule and the
  refusals.
- **The fix round's** (predictions in ``tests/flow/predictions_f4_fix.json``): the part's place and the window (F1),
  the share's couple off the centre of mass (F2), FVM() naming no scheme (F3), the moving start (F4), the interior's
  change telescoping (F5), the Reynolds watch (R), the near-wall (W) and density-ratio (D) flags."""

from __future__ import annotations

import contextlib
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from resolved_cases import (TAU_R, TAU_T, WATER_RHO, FreeBox, HeldSphere, a_cell_off, at_rest, deeper_band, free_node,
                            haberman_sayre, indicator_centroids, load_reversed, own_indicator, patched, plant,
                            r3_experiment, r3_history, rigid_interior, seed, share_at_the_centre, shift_cells, stable,
                            start_at_rest)

from dataclasses import dataclass

from mime.build import BuildRefused, RunFailed
from mime.core import run_dtype
from mime.interactions import Drag
from mime.registry import register_interaction
from mime.vocabulary import register_namespace
from mime.nodes.flow.fvm import _resolved_node
from mime.nodes.flow.fvm._resolved_node import ResolvedFlowNode
from mime.units import ms


# -- gate 1: the Re -> 0 ladder ---------------------------------------------------------------------------------------
def test_the_sphere_reads_haberman_and_sayre_at_a_radius_smaller_by_a_shift_that_falls_with_n(float64):
    """Gate 1: a sphere at a/R 0.375 on a periodic tube's axis, Stokes, flux-free with its displaced share: its drag
    against Haberman and Sayre at 3 and 6 cells a radius (R3: -6.9 % and -1.0 %), and the radius it reads, smaller by a
    shift in cells that falls with N. The plant doubles the body's band's penalty depth, a shift the gate must see."""
    readings = {}
    if plant("BODY_BAND"):
        with patched(_resolved_node, "body_faces", deeper_band(2.0)):
            readings = {cells: HeldSphere(0.375, cells).k() for cells in (3.0, 6.0)}
    else:
        readings = {cells: HeldSphere(0.375, cells).k() for cells in (3.0, 6.0)}
    exact = haberman_sayre(0.375)
    errors = {cells: k / exact - 1.0 for cells, k in readings.items()}
    shifts = {cells: shift_cells(k, 0.375, cells) for cells, k in readings.items()}
    print(f"F4 gate 1: K errors {errors}, shifts in cells {shifts}")
    assert -0.085 < errors[3.0] < -0.055, errors
    assert -0.020 < errors[6.0] < -0.003, errors
    assert 0.0 < shifts[6.0] < shifts[3.0], shifts


# -- gate 2: R3 at the default N, and the lagged edge at rest ----------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.3, 1.4])
def test_r3_runs_unstopped_and_bounded_at_the_default_n(dtype, ratio):
    """Gate 2, re-measured at the default N = 6: R3's case (a sphere of 6 cells a radius on a periodic tube's axis, Re
    76's drive, from rest, IMEX Euler at dt 6e-5 s, inside the refusal's line at both ratios: dt/tau_t 0.082 at ratio
    0.3, where the refusal admits 0.8 of the measured 0.119, and 0.018 at 1.4) runs its 200 steps
    unstopped (no motion guard, no CFL stop), finite, the body's speed under twice the flow's largest. The gate's old
    metric, four alternating changes in a row, read the start-up ring at ratio 0.3 as unstable though it decays (the
    refuting review's run); and no growth of the body's velocity can be read in a run driven at Re 76 (the developing
    flow and the cells the body crosses, near the Nyquist frequency, move it by themselves: measured, every velocity
    metric tried flagged every arm, ratio 1.4 at N = 3 included), so the coupling's stability is judged at rest (gate
    2b) and this gate holds the operating point. The plant reverses the body's load."""
    if plant("LOAD_SIGN"):
        with patched(ResolvedFlowNode, "stepped", load_reversed(ResolvedFlowNode.stepped)):
            velocities, top = r3_history(ratio, dt=6e-5, cells=6)
    else:
        velocities, top = r3_history(ratio, dt=6e-5, cells=6)
    ran = bool(np.all(np.isfinite(velocities))) and math.isfinite(top)
    largest = float(np.max(np.abs(velocities))) if ran else math.nan
    print(f"F4 gate 2 [{dtype}] ratio {ratio} at N = 6: ran {ran}, the body's largest speed {largest:.4g} m/s, the "
          f"flow's {top:.4g} m/s")
    assert ran, (velocities[-1], top)
    assert largest < 2.0 * top, (largest, top)


def _admitted(ratio: float, cells: int) -> tuple[float, float]:
    """The largest step the lagged refusal admits for R3's uniform sphere at ``ratio`` and N ``cells`` with the fluid's
    inertia stored, and the measured edge's step past it by a quarter (dt = x tau, the binding mode's)."""
    from mime.dynamics.flow._resolved import LAGGED_MARGIN, lagged_edge

    steps = [edge * ratio * tau for mode, tau in (("translation", TAU_T), ("rotation", TAU_R))
             for edge in (lagged_edge(mode, True, cells, ratio),) if edge is not None]
    return LAGGED_MARGIN * min(steps), 1.25 * min(steps)


def test_the_lagged_loop_at_rest_decays_at_the_largest_step_the_refusal_admits(dtype):
    """Gate 2b (B1): the lagged loop's stability, judged where it can be (at rest, by energy: body and fluid only lose
    it there, so a body whose kinetic energy exceeds its start's had it made by the coupling). R3's uniform sphere at
    a density ratio of 0.3 (the least the rung admits) at the default N = 6, quiescent, every twist component
    perturbed, at the largest step the refusal admits (drag.penalty-lagged: its margin's share of the edge measured
    there): its energy over the last quarter is under its start's. The plant steps a quarter past the measured edge (the
    refusal set aside): its energy grows past its start's, or the motion guard stops it."""
    admitted, past = _admitted(0.3, 6)
    if plant("PAST_EDGE"):
        with _without_the_lagged_refusal():
            energy, stop = at_rest(0.3, past, cells=6)
    else:
        energy, stop = at_rest(0.3, admitted, cells=6)
    print(f"F4 gate 2b [{dtype}]: dt {admitted:.4g} s admitted ({past:.4g} past the edge): the energy over its "
          f"start's {energy:.3g}, stop {stop}")
    assert stop is None and energy < 1.0, (energy, stop)


def test_a_step_past_the_lagged_edge_is_refused_naming_the_step_it_admits(float64):
    """B1's refusal: R3's neutral sphere in Stokes flow at dt 7e-3 s (dt/tau_t 2.9, dt/tau_r 3.0, where the refuting
    review's run went to 6e7 m/s) is refused at build (drag.penalty-lagged), naming the largest step the measured edges
    admit and the affine coupling."""
    found = _refusals(r3_experiment(1.0, dt=7e-3, reynolds=0.1, stokes=True))
    assert "drag.penalty-lagged" in found and "dt=... at most " in found["drag.penalty-lagged"] and \
        "affine" in found["drag.penalty-lagged"], found


def test_the_rigid_interior_shortcut_diverges_where_the_field_formulation_holds(float64):
    """Gate 2's control: the rigid-interior shortcut (the interior's momentum change taken as m_f (V^n - V^n-1)/dt)
    at a density ratio of 1.2 is not smooth: R3 measured it diverging, its amplification about -1.6 a step."""
    with patched(ResolvedFlowNode, "stepped", rigid_interior(1.2, ResolvedFlowNode.stepped)):
        velocities, top = r3_history(1.2)
    held, longest = stable(velocities, top)
    print(f"F4 gate 2's control: the shortcut at 1.2 stable {held}, the longest run of alternating changes {longest}")
    assert not held, (longest, velocities[-1], top)


# -- gate 3: the displaced share ---------------------------------------------------------------------------------------
def test_a_neutral_sphere_in_a_uniformly_driven_free_box_accelerates_with_the_fluid(dtype):
    """Gate 3: a neutral sphere of 3 cells a radius in a free 24^3 box (the tube outside it) driven by a uniform force
    g: over the run's second half its acceleration is g's (the share restores the drive on the fluid it displaces,
    which the field's drive, on the fluid alone, leaves out). The plant leaves the share out (about 16 % short)."""
    box = FreeBox(share=not plant("NO_SHARE"))
    velocities = box.run(160)
    acceleration = (velocities[-1] - velocities[79]) / (80 * box.dt)
    error = acceleration / box.g - 1.0
    print(f"F4 gate 3 [{dtype}]: the second half's acceleration over g - 1 = {error:.3g}")
    assert abs(error) < 0.005, error


def test_the_interiors_momentum_change_telescopes_over_a_run(dtype):
    """The fix round's gate F5 (the review's m8): gate 3's neutral sphere driven harder (g 50 m/s^2), so that it crosses
    about 1.7 cells in 80 steps: the interior's momentum change summed over the run, sum dt (load - penalty), is its
    momentum at the end (the fluid starts at rest), to 1e-6 of it: each step's change is taken from the value the step
    before held. The plant takes it on the step's own indicator (1e-4 off)."""
    box = FreeBox(g=50.0)
    with patched(ResolvedFlowNode, "stepped", own_indicator(ResolvedFlowNode.stepped)) if plant("OWN_INDICATOR") \
            else contextlib.nullcontext():
        box.run(80)
    end = np.asarray(box.state["body_0_momentum"], dtype=np.float64)
    mismatch = float(np.max(np.abs(box.changed - end)) / np.max(np.abs(end)))
    print(f"F4 fix F5 [{dtype}]: the summed change against the end's momentum {mismatch:.3g}")
    assert mismatch < 1e-6, mismatch


def test_a_neutral_sphere_held_in_a_closed_tube_under_a_drive_across_it_takes_no_net_load(dtype):
    """Gate 3's second control: a sphere of 6 cells a radius held at rest on the tube's axis (a/R 0.375, the steady
    Stokes step) under a drive across the axis, which the wall stops: the pressure the drive builds pushes the body
    through the field, and its displaced share cancels that push, so a neutral body in a closed vessel stays at rest.
    The plant leaves the share out."""
    held = HeldSphere(0.375, 6.0)
    g = 1.0
    load, share = held.at_rest((g, 0.0, 0.0))
    total = np.asarray(load[:3]) + (0.0 if plant("NO_SHARE") else np.asarray(share[:3]))
    scale = WATER_RHO * 4.0 / 3.0 * math.pi * held.radius ** 3 * g
    along, across = abs(total[0]) / scale, float(np.max(np.abs(total[1:]))) / scale
    print(f"F4 gate 3b [{dtype}]: the net load over rho V g along the drive {along:.3g}, across {across:.3g}; the "
          f"field's push {float(load[0]) / scale:.4g}, the share {float(share[0]) / scale:.4g}")
    assert along < 0.1, along
    assert across < 0.1, across


@pytest.mark.parametrize("at", [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (0.0, 3.9, 2.1)],
                         ids=["centred", "two radii off", "off on two axes"])
def test_the_lifted_solid_sits_at_its_parts_place_about_the_pose(dtype, at):
    """The fix round's gate F1: a sphere of 3 cells a radius in a free 32^3 box, its part placed ``at`` (cells) off the
    body's centre, at a pose drawn (``DETECTION_SEED``: a sub-cell position and a turn): on every face set the lifted
    indicator's centroid, read from the grid's own fields, is the pose's position plus its turn of the part's place,
    to 0.05 cells (measured 0.003 clean). It holds the part's place (the key the shape's own centre overwrote) and the
    window sized for it (two radii off clips a window sized from the radius alone). The plant lifts the solid a cell
    off its pose on every axis."""
    h = 1e-5
    node = free_node(32, 3.0, tuple(c * h for c in at))
    draws = np.random.default_rng(seed())
    turn = draws.normal(size=4)
    pose = jnp.asarray(np.concatenate([(16.0 + draws.uniform(-1.0, 1.0, 3)) * h, turn / np.linalg.norm(turn)]),
                       dtype=run_dtype())
    if plant("POSE_CELL"):
        with patched(_resolved_node, "body_faces", a_cell_off(_resolved_node.body_faces)):
            centroids = indicator_centroids(node, pose)
    else:
        centroids = indicator_centroids(node, pose)
    print(f"F4 fix F1 [{dtype}] at {at} cells: the indicator's centroid off the solid's place, largest "
          f"{float(np.max(np.abs(centroids))):.3g} cells")
    assert float(np.max(np.abs(centroids))) < 0.05, centroids


def test_a_neutral_sphere_held_off_its_centre_of_mass_in_a_closed_tube_takes_no_net_couple(dtype):
    """The fix round's gate F2 (gate 3b with an offset hull): a sphere of 6 cells a radius held at rest on the tube's
    axis (a/R 0.375, the steady Stokes step), its part placed a quarter radius along the axis from the body's centre,
    under a drive across the axis: the field's push acts at the hull, so its couple about the centre of mass is the
    lever's, and the displaced share, given at the hull's centroid, cancels it. The plant gives the share at the centre
    of mass."""
    lever = 0.25 * 0.5e-3
    held = HeldSphere(0.375, 6.0, at=(0.0, 0.0, lever))
    g = 1.0
    if plant("SHARE_AT_COM"):
        with patched(ResolvedFlowNode, "stepped", share_at_the_centre(ResolvedFlowNode.stepped)):
            load, share = held.at_rest((g, 0.0, 0.0))
    else:
        load, share = held.at_rest((g, 0.0, 0.0))
    total = np.asarray(load) + np.asarray(share)
    weight = WATER_RHO * 4.0 / 3.0 * math.pi * held.radius ** 3 * g
    couple = float(np.max(np.abs(total[3:]))) / (lever * weight)
    along = abs(total[0]) / weight
    print(f"F4 fix F2 [{dtype}]: the net couple over the lever times rho V g {couple:.3g} (the field's "
          f"{float(load[4]) / (lever * weight):.4g}, the share's {float(share[4]) / (lever * weight):.4g}), the net "
          f"force along the drive {along:.3g}")
    assert couple < 0.1, couple
    assert along < 0.1, along


@pytest.mark.parametrize("stokes", [False, True], ids=["fvm", "fvm-stokes"])
def test_an_fvm_naming_no_scheme_is_judged_at_the_scheme_it_chooses(float64, stokes):
    """The fix round's gate F3: ``FVM()`` and ``FVM(stokes=True)`` naming no scheme are planned at the scheme the FVM
    chooses at the run's values, never refused as an entry's defect: at R3's step RK3 (nu dt / h^2 0.038 under its
    0.15), which takes no resolved body (drag.penalty-scheme)."""
    found = _refusals(r3_experiment(1.0, unnamed=True, stokes=stokes))
    assert "drag.penalty-scheme" in found and "entry.defect" not in found, found


def test_a_neutral_body_moving_with_a_uniformly_moving_fluid_keeps_its_speed_over_the_first_step(dtype):
    """The review's moving-start oracle (its m7): a neutral sphere of 3 cells a radius moving at U with a free box's
    fluid already moving at U (the node's initial velocity): the first step's load leaves its speed unchanged, so its
    interior's momentum change at a run's first step is taken from the start's field, not from rest. The plant takes it
    from rest (a kick of about U)."""
    h, cells, speed = 1e-5, 24, 1e-3
    dt = 0.1 * h * h * WATER_RHO / 1e-3
    tolerance = 1e-10 if run_dtype() == jnp.float64 else 1e-5
    node = free_node(cells, 3.0, h=h, dt=dt, tolerance=tolerance,
                     initial_velocity_z=np.full((cells, cells, cells), speed))
    side = cells * h
    pose = jnp.asarray((side / 2 + 0.3 * h, side / 2 + 0.2 * h, side / 2 + 0.1 * h, 1.0, 0.0, 0.0, 0.0),
                       dtype=run_dtype())
    twist = jnp.asarray((0.0, 0.0, speed, 0.0, 0.0, 0.0), dtype=run_dtype())
    with patched(ResolvedFlowNode, "stepped", start_at_rest(ResolvedFlowNode.stepped)) if plant("START_AT_REST") \
            else contextlib.nullcontext():
        state = node.update(dict(node.initial_state()), {"body_0_pose": pose, "body_0_twist": twist,
                                                         "body_force": jnp.zeros(3, dtype=run_dtype())}, dt)
    volume = 4.0 / 3.0 * math.pi * (3.0 * h) ** 3
    change = dt * float(state["body_0_load"][2]) / (WATER_RHO * volume)
    print(f"F4 fix m7 [{dtype}]: the first step's velocity change over the speed {change / speed:.3g}")
    assert abs(change) < 1e-3 * speed, change


def _without_the_lagged_refusal():  # noqa: ANN202 - the build's lagged refusal set aside, so a run reaches the guard
    original = Drag.penalty_value_problems

    def problems(self, values, ctx):  # noqa: ANN001, ANN202
        return [problem for problem in original(self, values, ctx) if problem.code != "drag.penalty-lagged"]
    return patched(Drag, "penalty_value_problems", problems)


def test_a_diverging_lagged_run_is_stopped_naming_the_coupling(float64):
    """The fix round's gate G: the refuting review's Stokes case (R3's tube at 3 cells a radius, IMEX Euler with
    stokes=True, a neutral uniform sphere, dt 7e-3 s: 2.9 times its translation's relaxation time and 3.0 times its
    rotation's), past the lagged coupling's edge, which ran unstopped to 6e7 m/s in float64: with the build's refusal
    set aside the body's own motion stops it, past a cell a step (drag.penalty-motion), naming the body and its
    coupling. The plant removes the guard."""
    from mime.units import s as seconds

    line = math.inf if plant("NO_MOTION_GUARD") else _resolved_node.MOTION_LINE
    with _without_the_lagged_refusal(), patched(_resolved_node, "MOTION_LINE", line):
        built = r3_experiment(1.0, dt=7e-3, reynolds=0.1, stokes=True).build()
        with pytest.raises(RunFailed) as stopped:
            built.run(seconds(100 * 7e-3))
    found = {refusal.code: refusal.message for refusal in stopped.value.refusals}
    print(f"F4 fix G: stopped {sorted(found)}: {found.get('drag.penalty-motion', '')[:160]}")
    assert "drag.penalty-motion" in found, found
    assert "ball" in found["drag.penalty-motion"] and "lagged coupling" in found["drag.penalty-motion"], found


@pytest.mark.parametrize("start", [1.875e-4, 0.0], ids=["two cells from the wall", "on the axis"])
def test_a_body_nearer_the_wall_than_three_cells_is_flagged_each_step(float64, start):
    """The fix round's gate W (M6): a resolved body whose solid comes within 3 cells of the tube's wall, where the
    free body's accuracy was measured to fall (the near-wall resolved-body phase's), raises its drag's flag
    (water.flow.body_0_near_wall) at every step; on the axis (5 cells away at 3 cells a radius) it raises none."""
    from mime.units import s as seconds

    built = r3_experiment(1.0, dt=1.5e-4, reynolds=0.1, stokes=True, start=(start, 0.0, 0.0)).build()
    declared = {flag.path: flag.owner for flag in built.layout.flags}
    result = built.run(seconds(3 * 1.5e-4))
    raised = {read.path for read in result.flags if read.was_raised}
    gap = float(np.asarray(result.final["water.flow"]["body_0_gap"]))
    print(f"F4 fix W: start {start} m, gap {gap:.3g} cells, flags raised {sorted(raised)}")
    assert declared.get("water.flow.body_0_near_wall") == "drag:ball", declared
    assert ("water.flow.body_0_near_wall" in raised) is (start > 0.0), (gap, raised)


@pytest.mark.parametrize("speed", [0.2, 0.0], ids=["past the line", "at rest"])
def test_the_penalty_rungs_reynolds_watch_flags_a_body_past_its_line_at_the_runs_n(float64, speed):
    """The fix round's gate R (M6): the penalty rung's Reynolds line, 150 at the default N (the highest particle
    Reynolds number N = 6 was measured at, where its drag is 6.6 % off N = 9's), falls as N squared below it: at 3 cells
    a radius 37.5. The post-run watch is computed (the rung reads the medium's viscosity) and is raised for a body at
    0.2 m/s (Re about 70 by its diameter), not for one at rest."""
    from mime.units import s as seconds

    result = r3_experiment(1.0, dt=1.5e-4, reynolds=0.1, stokes=True, velocity=(0.0, 0.0, speed)).build().run(
        seconds(2 * 1.5e-4))
    (watch,) = [read for read in result.watches if read.entry.startswith("drag")]
    print(f"F4 fix R: {watch.text()}")
    assert watch.upper is not None and math.isclose(watch.upper, 150.0 * (3.0 / 6.0) ** 2), watch.upper
    assert watch.status == ("above" if speed > 0.0 else "within"), watch.text()


@pytest.mark.parametrize("ratio", [25.0, 1.0], ids=["past the measured", "inside"])
def test_a_density_ratio_outside_the_measured_range_is_flagged_each_step(float64, ratio):
    """The fix round's gate D (M6): the lagged coupling's edge moves with the body's density ratio and was measured over
    a range of them (``LAGGED_RATIOS``); a body outside it (here 25 times the fluid's density) raises its drag's flag at
    every step (its edge taken as the nearest measured ratio's), one inside raises none. The node reads the ratio from
    the body's mass matrix."""
    from mime.units import s as seconds

    built = r3_experiment(ratio, dt=1.5e-5, reynolds=0.1, stokes=True).build()
    declared = {flag.path: flag.owner for flag in built.layout.flags}
    result = built.run(seconds(3 * 1.5e-5))
    raised = {read.path for read in result.flags if read.was_raised}
    read = float(np.asarray(result.final["water.flow"]["body_0_ratio"]))
    print(f"F4 fix D: ratio {ratio}, read {read:.6g}, flags raised {sorted(raised)}")
    assert math.isclose(read, ratio, rel_tol=1e-9), read
    assert declared.get("water.flow.body_0_ratio_unmeasured") == "drag:ball", declared
    assert ("water.flow.body_0_ratio_unmeasured" in raised) is (ratio > 20.0), raised


def test_a_resolved_bodys_solve_that_stops_short_is_flagged_and_warned_of(dtype):
    """The fix round's gate S (the near-wall spike's question, never silent): a resolved body's flow whose solve stops
    above its tolerance (planted below either dtype's rounding, 1e-30, the steady scheme) raises solve_flagged at every
    step, the run warns of it with its count, and the step's report says how the solve stopped; at the default
    tolerance the same run raises neither. The resolved node takes its parent's step and report."""
    import warnings

    from mime.dynamics import Scheme
    from mime.units import s as seconds

    for tolerance, flagged in ((1e-30, True), (None, False)):
        built = r3_experiment(1.0, dt=1.5e-4, reynolds=0.1, stokes=True, scheme=Scheme.Steady(),
                              tolerance=tolerance).build()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = built.run(seconds(3 * 1.5e-4))
        raised = {read.path: read for read in result.flags}
        said = [str(warning.message) for warning in caught if "solve_flagged" in str(warning.message)]
        report = np.asarray(result.final["water.flow"]["solve_report"])
        print(f"F4 fix S [{dtype}] tolerance {tolerance}: flags {sorted(p for p, r in raised.items() if r.was_raised)}, "
              f"warnings {len(said)}, report {report.tolist()}")
        if flagged:
            read = raised["water.flow.solve_flagged"]
            assert read.was_raised and read.raised == read.steps == 3, read
            assert len(said) == 1 and "raised at 3 of 3 steps" in said[0], said
            assert report[2] + report[3] + report[4] > 0.0, report
        else:
            assert "water.flow.solve_flagged" not in raised and not said and report[2:5].sum() == 0.0, report


# -- gate 4: the books ----------------------------------------------------------------------------------------------
def _gift(original):  # noqa: ANN001, ANN202 - the plant: the body's penalty load taken at half the twist solved at
    def body_books(self, p, fields, velocity, index):  # noqa: ANN001, ANN202
        halved = dataclasses.replace(fields, twists=tuple(0.5 * twist for twist in fields.twists))
        return original(self, p, halved, velocity, index)
    return body_books


def test_the_books_close_and_the_drag_is_judged_at_one_velocity_by_its_host(dtype):
    """Gate 4: metered, a sphere (ratio 1.2) carried in R3's driven tube for 40 steps: every receiver's books close at
    every step (``ledger_closes``); the drag booked under its id on the fluid's row (held in its step) and the body's, its
    share under ``drag:ball.displaced``; its sign judged by its host at one velocity, no gift step on either row. The
    plant takes the body's penalty load at half the twist the field was solved at (a load at a velocity the field did
    not see, which gives the body work the fluid never lost: the defect a judgement at one velocity exists for). A gift
    added to the delivered load alone is not this gate's to see: the lagged exchange's unpaired work is booked in the
    drag's rows by design, unjudged until the affine coupling pairs it (measured: such a plant caught 0 of 5 runs in
    each dtype)."""
    from mime.observations import EnergyLedger
    from mime.testing import ledger_closes

    exp = r3_experiment(1.2)
    exp.observe(EnergyLedger())
    stopped: RunFailed | None = None
    with patched(ResolvedFlowNode, "body_books", _gift(ResolvedFlowNode.body_books)) if plant("GIFT") \
            else contextlib.nullcontext():
        try:
            result = exp.build().run(ms(6.0))
        except RunFailed as failed:  # the books up to a stop, so that a gift reads as a gift, not as the stop it causes
            stopped, result = failed, failed.result
    assert result is not None and result.energy is not None, stopped
    rows = {(law.law, law.receiver): law for law in result.energy.laws}
    assert {("drag:ball", "water.flow"), ("drag:ball", "ball"), ("drag:ball.displaced", "ball")} <= set(rows), rows
    gifts = {receiver: rows[("drag:ball", receiver)].gift_steps for receiver in ("ball", "water.flow")}
    print(f"F4 gate 4 [{dtype}]: the drag's gift steps {gifts}" + (" (the run stopped)" if stopped else ""))
    assert gifts == {"ball": 0, "water.flow": 0}, gifts
    assert stopped is None, stopped
    verdict = ledger_closes(result)
    assert verdict.status == "PASS", verdict.detail


# -- gates 5 and 6: the ripple and the derivative in position ------------------------------------------------------------
def test_the_drag_of_a_sphere_moved_through_a_cell_stays_within_the_ripples_line(float64):
    """Gate 5: a sphere of 6 cells a radius at a/R 0.375 at eight offsets through a cell along the axis: the drag's
    spread over them, (max - min)/mean, and the largest change of the drag a cell along it (its derivative times the
    cell, by autodiff), the quality line a derivative in the body's position carries. The plant stiffens the body's
    band (its depth a tenth), the staircase a stiff penalty makes of a smooth band (B3)."""
    if plant("STIFF_BAND"):
        with patched(_resolved_node, "body_faces", deeper_band(0.1)):
            spread, largest = _ripple()
    else:
        spread, largest = _ripple()
    print(f"F4 gate 5: the drag's spread through a cell {spread:.3g}, its largest change a cell {largest:.3g}")
    assert spread < 0.01, spread
    assert largest < 0.05, largest


def _ripple() -> tuple[float, float]:
    held = HeldSphere(0.375, 6.0)
    h = held.layout.spacing
    offsets = np.linspace(0.0, 1.0, 8, endpoint=False)
    drags, slopes = [], []
    for offset in offsets:
        centre = held.centre + np.asarray((0.0, 0.0, offset * h))
        drag, slope = jax.value_and_grad(lambda z: held.column(2, jnp.asarray(centre).at[2].set(z))[2])(centre[2])
        drags.append(float(drag))
        slopes.append(float(slope))
    mean = float(np.mean(drags))
    return (max(drags) - min(drags)) / mean, max(abs(slope) for slope in slopes) * h / mean


def test_a_derivative_in_the_bodys_position_equals_a_central_difference(float64):
    """Gate 6: the axial drag of a sphere off the axis (its centre a fifth of the tube's radius across, 3 cells a
    radius) in its lateral position: autodiff through the steady step against a central difference of a hundredth of
    a cell, the derivative of the discrete drag (the window's corner held out of it, the band smooth). Within a cell the
    derivative itself moves (gate 5's ripple): a difference of a tenth of a cell already reads 5 % off it. The plant
    stops the derivative at the body's position in the lift."""
    held = HeldSphere(0.375, 3.0)
    h = held.layout.spacing
    centre = jnp.asarray(held.centre).at[0].add(0.2 * held.radius / 0.375)

    def drag(x: object) -> object:
        moved = jax.lax.stop_gradient(x) if plant("STOP_POSITION") else x
        return held.column(2, centre.at[0].set(moved))[2]

    derivative = float(jax.grad(drag)(centre[0]))
    step = 0.01 * h
    difference = (float(drag(centre[0] + step)) - float(drag(centre[0] - step))) / (2.0 * step)
    print(f"F4 gate 6: d F_z / d x {derivative:.6g} against {difference:.6g}")
    assert abs(derivative - difference) <= 2e-3 * abs(difference), (derivative, difference)


# -- gate 7: float32 ---------------------------------------------------------------------------------------------------
def test_the_steady_drag_in_float32_is_within_its_line_of_float64s():
    """Gate 7: the steady drag at 6 cells a radius in float32 within 1e-3 of float64's (B10's line for a penalty
    drag)."""
    held = jax.config.jax_enable_x64
    try:
        values = {}
        for dtype in ("float64", "float32"):
            jax.config.update("jax_enable_x64", dtype == "float64")
            values[dtype] = HeldSphere(0.375, 6.0).k()
    finally:
        jax.config.update("jax_enable_x64", held)
    difference = abs(values["float32"] / values["float64"] - 1.0)
    print(f"F4 gate 7: float32's drag against float64's {difference:.3g}")
    assert difference < 1e-3, values


# -- the build: the spacing rule and the refusals -----------------------------------------------------------------------
@pytest.mark.parametrize(("cells", "across"), [(3, 16), (6, 32)])
def test_the_fvm_takes_the_finer_of_the_lumens_spacing_and_n_cells_across_the_bodys_radius(float64, cells, across):
    """B8.2: with no spacing written, the FVM's grid is the finer of D/h ``cells_across`` (16 here) and N cells across
    the resolved body's radius: a 0.1875 mm sphere in a 1 mm lumen at 3 cells a radius takes 16 across, at the rung's
    own 6, 32."""
    built = r3_experiment(1.0, feature_cells=cells, dt=1.5e-4 * 16 / across).build()
    node = built.graph.get_node("water.flow")
    assert isinstance(node, ResolvedFlowNode)
    assert math.isclose(float(node.constants()["spacing"]), 1e-3 / across, rel_tol=1e-3), node.constants()["spacing"]


def _refusals(exp: object) -> dict[str, str]:
    with pytest.raises(BuildRefused) as refused:
        exp.build()  # type: ignore[attr-defined]
    return {refusal.code: f"{refusal.message} | {refusal.action}" for refusal in refused.value.refusals}


def test_a_body_lighter_than_the_lagged_ranges_edge_is_refused_naming_the_route(float64):
    """The lagged coupling is declared over the measured range: a body at 0.2 times the fluid's density is refused,
    naming the routes (a denser body, the MRS, the affine coupling this one comes with next)."""
    found = _refusals(r3_experiment(0.2))
    assert "drag.penalty-lagged" in found, found


def test_a_hull_with_no_exact_solid_region_is_refused_naming_the_mrs(float64):
    from mime.scene import Box
    from mime.units import m

    found = _refusals(r3_experiment(1.0, hull=Box(size=m((2e-4, 2e-4, 2e-4)))))
    assert "drag.penalty-kind" in found, found


def test_one_way_is_refused_on_a_resolved_body(float64):
    found = _refusals(r3_experiment(1.0, one_way=True))
    assert any("one_way" in message for message in found.values()), found


def test_a_grid_coarser_than_the_thinnest_feature_line_is_refused(float64):
    from mime.units import m

    found = _refusals(r3_experiment(1.0, spacing=m(1e-4)))
    assert "drag.penalty-thin" in found, found


def test_feature_cells_written_beside_a_coarser_written_spacing_is_refused(float64):
    """The review's m4: FVM(spacing=...) written wins over the drag's feature_cells, so a feature_cells the written
    spacing does not give would be ignored in silence: refused, naming both routes."""
    from mime.units import m

    found = _refusals(r3_experiment(1.0, feature_cells=6, spacing=m(6.25e-5)))
    assert "drag.penalty-spacing" in found and "drag.penalty-thin" not in found, found


def test_rk3_is_refused_for_a_resolved_body(float64):
    from mime.dynamics import Scheme

    found = _refusals(r3_experiment(1.0, scheme=Scheme.RK3()))
    assert "drag.penalty-scheme" in found, found


register_namespace("resolvedtest", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PointBeside(Drag):
    """The stokes rung under a name of its own, so that a point body's drag (which delivers its Maxey-Riley added mass)
    can be attached beside a resolved body's: the case the added-mass refusal guards."""

    name = "resolvedtest.point_beside"


def test_an_added_mass_delivered_to_a_resolved_body_is_refused_naming_the_added_mass_phase(float64):
    """A6 item 9: a resolved body's grid carries its fluid's inertia, so a law delivering an added mass to it (here a
    point body's drag, whose Maxey-Riley added mass rides on its resistance) is refused, naming the route (I2's
    delivered-inertia port, for a medium whose solver carries no inertia)."""
    exp = r3_experiment(1.0)
    exp.attach(PointBeside(rung="stokes"), body="ball", medium="water")
    found = _refusals(exp)
    assert "drag.penalty-added-mass" in found and "I2" in found["drag.penalty-added-mass"], found


def test_a_body_whose_window_would_meet_its_periodic_image_is_refused(float64):
    from mime.units import m

    found = _refusals(r3_experiment(1.0, length=m(5e-4)))
    assert "drag.penalty-box" in found, found


def test_an_open_vessel_takes_no_resolved_body_yet(float64):
    """The open vessel's one elimination of openings and bodies is the affine coupling's (the next part): until then
    its flow refuses a resolved body, naming the periodic box."""
    found = _refusals(r3_experiment(1.0, opened=True))
    assert "flow.open-resolved" in found, found


def test_a_tube_trees_flow_takes_no_resolved_body_yet(float64):
    """A tube tree's node holds the tree's solid alone: a resolved body in its flow is refused by name, naming the
    straight tube and the MRS (without the refusal the build stopped on an entry defect, its reads unconnected)."""
    found = _refusals(r3_experiment(1.0, tree=True))
    assert "flow.tree-resolved" in found, found
