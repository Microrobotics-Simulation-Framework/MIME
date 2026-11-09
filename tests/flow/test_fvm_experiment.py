"""The FVM through the experiment: ``Medium(fields={"flow": FVM(...)})`` built with ``exp.build``, driven by a
``BodyForce`` law bound to the medium; the gates' physics read through the build (Poiseuille's core curvature on the
built node), the saved form's round trip, the CFL check's stop and refusal of a run, a solve's stop flagged and warned
of through the run, a NaN that arises inside the step refused as not finite, the validity lines' flags, and the plan's
refusals."""

from __future__ import annotations

import functools
import pathlib
import warnings

import numpy as np
import pytest
from fvm_cases import in_dtype
from test_fvm_tube import CURVATURE_LINE, core_viscosity

from mime.build import ArgumentRefused, BuildRefused, RunFailed
from mime.control._run import automatic_chunks
from mime.dynamics import FVM, Band, Scheme
from mime.dynamics.flow import IMEXEulerScheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.nodes.flow.fvm import Stopping
from mime.scene import PeriodicBox, Scene, Tube, X, Z
from mime.things import Medium, Structure
from mime.units import kg_per_m3, m_per_s2, mm, mPa_s, ms

RADIUS, NU, G = 0.5e-3, 1e-6, 1.0


def experiment(*, axis=+Z, fills: str | None = "tube.lumen", dt=ms(20.0), steady=True, g=G, stokes=None,
               length=mm(0.25), cells_across=16, **settings):
    exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=dt)
    exp.add_structure(Structure("tube", geometry=Tube(axis=axis, inner_radius=mm(RADIUS * 1e3), wall=mm(0.2),
                                                      length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=length), cells_across=cells_across,
               stokes=steady if stokes is None else stokes,
               scheme=Scheme.Steady() if steady else Scheme.IMEXEuler(), **settings)
    exp.add_medium(Medium("water", fills=fills, fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(g)), medium="water")
    return exp


def test_the_built_flows_core_curvature_gives_the_viscosity(dtype):
    with in_dtype(dtype):
        built = experiment().build()
        result = built.run(ms(40.0))
        node = built.graph.get_node("water.flow")
    assert abs(core_viscosity(node, result.final["water.flow"], g=G) / NU - 1.0) < CURVATURE_LINE


def test_a_saved_flow_loads_and_runs_bit_for_bit(float64, tmp_path: pathlib.Path):
    """The flow saves in the form it has (dev1's form 10, a medium's ``dynamics.flow``), settings groups included,
    and loads as the same medium: never dropped, never refused."""
    exp = experiment()
    exp.save(tmp_path / "saved")
    loaded = Experiment.load(tmp_path / "saved")
    finals = []
    for each in (exp, loaded):
        finals.append(each.build().run(ms(40.0)).final["water.flow"])
    for name in ("velocity_x", "velocity_y", "velocity_z"):
        assert np.array_equal(np.asarray(finals[0][name]), np.asarray(finals[1][name]))
    text = (tmp_path / "saved" / "experiment.mime.yaml").read_text()
    assert "name: fvm" in text and "group: periodic_box" in text and "group: steady" in text


def test_a_run_past_the_cfl_line_is_refused_before_any_step(float64):
    """Convection on, a drive strong enough that its predicted peak passes the CFL line: refused at the build, before
    any step, with the route (F1b's step rule); the per-step stop behind it is the next test's."""
    exp = experiment(steady=False, g=2000.0, dt=ms(1.0))
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    refusal = next(r for r in raised.value.refusals if r.code == "flow.cfl-predicted")
    assert "dt <=" in refusal.action and "dt='auto'" in refusal.action


@pytest.mark.parametrize("asked", [200, 2000])
def test_a_run_past_the_cfl_line_holds_its_state_and_ends_at_the_chunk(float64, asked, monkeypatch):
    """Convection on, a drive strong enough that the flow passes the CFL line at its second step (the review's case,
    g = 2000 m/s^2 at dt 1 ms): from that step the state is held, the run ends at the end of the chunk the step lies in,
    not at the steps asked for, and is refused after, naming the step and the route; its result comes with the
    refusal. The control is the run's own steps before the crossing, which advance. The build's prediction would
    refuse this run before any step (the test above); it is set aside here (a prediction of no flow), so the per-step
    stop behind it is what runs."""
    import mime.dynamics.flow._fvm as entry

    monkeypatch.setattr(entry, "predicted_peak", lambda *_: 0.0)
    exp = experiment(steady=False, g=2000.0, dt=ms(1.0), cells_across=8)
    with pytest.raises(RunFailed) as raised:
        exp.build().run(ms(float(asked)), history=True)
    assert [r.code for r in raised.value.refusals] == ["flow.cfl-exceeded"]
    assert "at its step" in raised.value.refusals[0].message and "dt <=" in raised.value.refusals[0].action
    result = raised.value.result
    assert result is not None and result.history is not None
    at = int(np.asarray(result.final["water.flow"]["stopped_at"]))
    assert at == 2
    ends = np.cumsum(automatic_chunks(asked))
    assert result.steps == int(ends[np.searchsorted(ends, at)]) < asked
    w = np.asarray(result.history["water.flow"]["velocity_z"])
    assert not np.array_equal(w[0], np.zeros_like(w[0]))  # the step before the crossing advanced
    assert all(np.array_equal(w[k], w[at - 2]) for k in range(at - 1, result.steps))  # the steps after it did not


def test_a_run_inside_the_cfl_line_is_not_refused(float64):
    exp = experiment(steady=False, g=1.0, dt=ms(0.2))
    result = exp.build().run(ms(20.0))
    assert int(np.asarray(result.final["water.flow"]["stopped_at"])) == -1


def test_a_solve_that_stagnates_is_flagged_and_warned_of_with_its_count(dtype):
    """B4.1, never silent: a tolerance below the dtype's rounding (planted) stalls every step's solve; the run's
    result names the flag with its count, the run warns with it, and each step's report says how it stopped. The
    control is the same run at the default tolerance, which raises neither."""
    for tolerance, flagged in ((1e-30, True), (None, False)):
        with in_dtype(dtype):
            built = experiment(cells_across=8, **({"tolerance": tolerance} if tolerance is not None else {})).build()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = built.run(ms(100.0))
        raised = {f.path: f for f in result.flags}
        said = [str(w.message) for w in caught if "solve_flagged" in str(w.message)]
        report = np.asarray(result.final["water.flow"]["solve_report"])
        if flagged:
            read = raised["water.flow.solve_flagged"]
            assert read.raised == read.steps == 5 and read.was_raised
            assert len(said) == 1 and "raised at 5 of 5 steps" in said[0]
            assert report[3] == 1.0 and report[2] == 0.0  # stagnated, before its cap
        else:
            assert "water.flow.solve_flagged" not in raised and not said and report[2:5].sum() == 0.0


def _nan_plant_refused(dtype: str, steady: bool, owner: object, name: str, plant: object, *, in_step: bool) -> None:
    """The plant through ``exp.run`` (D/h 8, IMEX Euler under Stokes or the steady limit): refused ``run.non-finite``
    from step 3, the flow's velocity NaN at the end; and the control, the solve before the fix, which returned a
    finite answer for a non-finite residual (its zero start, as non-finite answers made zero): the same run ends
    finite and unrefused (an in-step plant's velocity zero from its step on)."""
    import jax.numpy as jnp

    import mime.nodes.flow.fvm._solve as solve

    solved = solve.guarded_cg

    def zero_start_returned(*args, **kwargs):
        x, report = solved(*args, **kwargs)
        return tuple(jnp.where(jnp.isfinite(c), c, 0.0) for c in x), report

    dt, span = (ms(20.0), ms(100.0)) if steady else (ms(10.0), ms(60.0))
    for arm in ("fixed", "control"):
        with in_dtype(dtype), pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            patch.setattr(owner, name, plant)
            if arm == "control":
                patch.setattr(solve, "guarded_cg", zero_start_returned)
            built = experiment(steady=steady, stokes=True, dt=dt, cells_across=8).build()
            if arm == "fixed":
                with pytest.raises(RunFailed) as raised:
                    built.run(span, history=True)
                assert [r.code for r in raised.value.refusals] == ["run.non-finite"], raised.value.refusals
                message = raised.value.refusals[0].message
                assert "stopped being finite numbers at step 3" in message, message
                assert not in_step or "velocity_z" in message, message
                result = raised.value.result
                assert result is not None
                assert not np.isfinite(np.asarray(result.final["water.flow"]["velocity_z"])).any()
            else:
                w = np.asarray(built.run(span, history=True).history["water.flow"]["velocity_z"])
                assert np.isfinite(w).all()
                assert not in_step or (w[1].any() and not w[2:].any())  # zeros from the plant's step on


@pytest.mark.parametrize("steady", [True, False], ids=["steady", "imex_euler"])
def test_a_nan_that_arises_inside_the_flows_step_is_refused_as_not_finite(dtype, steady):
    """A NaN that arises inside the flow's step (the drive the step reads, made NaN in the step from its third step
    on) is refused as not finite, never replaced by a velocity: the solve whose residual is not finite returns NaN, so
    the run's own non-finite refusal names the flow's velocity from the step it began, on both monolithic schemes.
    The control is the solve before the fix (:func:`_nan_plant_refused`)."""
    import jax.numpy as jnp

    import mime.nodes.flow.fvm._node as flow_node

    made = flow_node.ImplicitFlowNode.forcing

    def forcing(self, fields, boundary_inputs, scheme, dtype_):
        force = made(self, fields, boundary_inputs, scheme, dtype_)
        return lambda stage, time: tuple(jnp.where(time > 2.5 * self.delta_t, jnp.nan, c) for c in force(stage, time))

    _nan_plant_refused(dtype, steady, flow_node.ImplicitFlowNode, "forcing", forcing, in_step=True)


@pytest.mark.parametrize("steady", [True, False], ids=["steady", "imex_euler"])
def test_a_nan_the_drive_delivers_at_one_step_is_refused_as_not_finite(dtype, steady):
    """The drive's own node NaN at its third step only: the flow's solve returns NaN there and its next step carries
    it on (IMEX Euler's right-hand side holds the velocity, the steady solve starts from it), so the run is refused as
    not finite where, before the fix, the flow restarted from rest and the run ended finite (the control)."""
    import jax.numpy as jnp

    import mime.nodes.flow._body_force as body_force

    delivered = body_force.BodyForceNode.force_at

    def force_at(self, p, time):
        return jnp.where((time > 2.5 * self.delta_t) & (time < 3.5 * self.delta_t), jnp.nan, delivered(self, p, time))

    _nan_plant_refused(dtype, steady, body_force.BodyForceNode, "force_at", force_at, in_step=False)


def test_a_stiff_interior_at_d_h_32_converges_on_every_step(dtype):
    """M7: the 0.01h interior at D/h 32, IMEX Euler under Stokes at dt 10 ms, where the stagnation stop judged against
    the last check stopped 26 of 30 steps in float64 (29 in float32) short of a solve that converges: under the window
    rule every step converges, unflagged. The controls: the same run under the rule before the fix (a window of one
    check at 0.9, at least as strict as judging against the last check) stops a step stagnated; and at a tolerance
    below rounding (the planted stall) every step stagnates, before its cap."""
    import mime.nodes.flow.fvm._node as flow_node

    for arm in ("window", "last_check", "planted"):
        with in_dtype(dtype), pytest.MonkeyPatch.context() as patch:
            if arm == "last_check":
                patch.setattr(flow_node, "stagnation_window", lambda cells, every: 1)
                patch.setattr(flow_node, "Stopping", functools.partial(Stopping, stagnation=0.9))
            built = experiment(steady=False, stokes=True, dt=ms(10.0), length=mm(0.125), cells_across=32,
                               wall=Band(delta_in=0.01), **({"tolerance": 1e-30} if arm == "planted" else {})).build()
            result = built.run(ms(80.0), history=True)
        report = np.asarray(result.history["water.flow"]["solve_report"])
        if arm == "window":
            assert report[:, 2:5].sum() == 0.0, report[:, :4]
            assert report[:, 1].max() <= (1e-8 if dtype == "float64" else 1e-5)
        elif arm == "last_check":
            assert report[:, 3].sum() >= 1.0, report[:, :4]
        else:
            assert report[:, 3].sum() == len(report) and report[:, 2].sum() == 0.0


def test_the_validity_lines_crossed_raise_their_flags(float64):
    """M6: the Stokes line on the lumen's Reynolds number (stokes=True) and the solid's bypass line, each per step,
    and the build's lines fixed at build (the resolution under the band's calibration). The drive of 1 m/s^2 in this
    tube gives a Reynolds number of about 31 at D/h 8, past the Stokes line, and a bypass of 2.6e-3, past its line; the
    controls are the Reynolds flag of the same flow with convection on (no Stokes approximation declared; at dt 0.5 ms,
    inside the step the build's CFL prediction allows, 0.71 ms), and the bypass flag at D/h 16."""
    flags = {}
    for name, exp in (("stokes_8", experiment(cells_across=8)),
                      ("convection_8", experiment(steady=False, stokes=False, dt=ms(0.5), cells_across=8)),
                      ("stokes_16", experiment(cells_across=16))):
        built = exp.build()
        lines = [line for held in built.report.checked.values() for line in held]
        result = built.run(ms(40.0) if name != "convection_8" else ms(4.0))
        flags[name] = ({f.path.rsplit(".", 1)[1]: f.was_raised for f in result.flags}, lines,
                       float(np.asarray(result.final["water.flow"]["reynolds"])))
    assert flags["stokes_8"][0].get("reynolds_flagged") and flags["stokes_8"][0].get("bypass_flagged")
    assert flags["stokes_8"][2] > 1.0 and flags["convection_8"][2] > 0.1
    assert "reynolds_flagged" not in flags["convection_8"][0]
    assert "bypass_flagged" not in flags["stokes_16"][0]
    assert any("8 cells across" in line for line in flags["stokes_8"][1])
    assert any("16 cells across" in line for line in flags["stokes_16"][1])


def test_the_entry_lists_its_validity_lines_and_flags():
    """Each decided line F1a carries is declared, and each flag a step raises is listed for the run to read."""
    text = " ".join(FVM.validity)
    for line in ("CFL", "tolerance", "cells", "bypass", "Reynolds", "eps x tau/dt", "geometry", "stall detector"):
        assert line in text, line
    assert {flag.field for flag in FVM.raises} == {"solve_flagged", "reynolds_flagged", "bypass_flagged"}
    assert [flag.field for flag in FVM.raises if flag.warns] == ["solve_flagged"]


def test_the_flow_builds_with_its_defaults_but_its_domain():
    """B8.1 and B8.2: every setting but the domain has its default (the domain's period is not decided, so it is
    still required), the wall's settings grouped as the wall's (Band, delta_in), and a non-default interior taken."""
    exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=ms(20.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0))))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": FVM(domain=PeriodicBox(length=mm(0.125)))}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water")
    node = exp.build().graph.get_node("water.flow")
    c = node.constants()
    assert (c["cells_x"], c["band"], c["interior"], c["scheme"]) == (43, "smooth_2h_0.3h", "0.1h", "imex_euler")
    assert FVM(domain=PeriodicBox(length=mm(1.0)), wall=Band(delta_in=0.01)).wall.delta_in == 0.01


@pytest.mark.parametrize(("change", "code"), [
    ({"steady": True, "stokes": False}, "flow.steady-convection"),
    ({"wall": Band(delta_in=0.5)}, "flow.interior"),
    ({"preconditioner": "multigrid"}, "flow.preconditioner"),
    ({"cells_across": 4}, "flow.resolution"),
])
def test_a_flow_its_settings_rule_out_is_refused_when_planned(change, code):
    steady = change.pop("steady", True)
    stokes = change.pop("stokes", None)
    exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=ms(20.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=mm(0.25)), stokes=steady if stokes is None else stokes,
               scheme=Scheme.Steady() if steady else IMEXEulerScheme(), **change)
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water")
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    assert code in [r.code for r in raised.value.refusals]


def test_a_flow_in_a_medium_that_fills_no_tube_is_refused():
    with pytest.raises(BuildRefused) as raised:
        experiment(fills=None).build()
    assert "flow.region" in [r.code for r in raised.value.refusals]


def test_a_tube_along_an_axis_the_box_cannot_hold_is_refused():
    with pytest.raises(BuildRefused) as raised:
        experiment(axis=(0.0, 0.6, 0.8)).build()
    assert "flow.box-axis" in [r.code for r in raised.value.refusals]


def test_a_tube_along_x_is_refused_with_its_route():
    with pytest.raises(BuildRefused) as raised:
        experiment(axis=+X).build()
    assert "flow.box-axis" in [r.code for r in raised.value.refusals]


def test_a_medium_takes_its_fields_by_role():
    flow = FVM(domain=PeriodicBox(length=mm(0.25)))
    medium = Medium("water", fields={"flow": flow})
    assert medium.flow is flow and medium.dynamics is flow and dict(medium.fields) == {"flow": flow}
    assert dict(Medium("bath", flow=flow).fields) == {"flow": flow}
    with pytest.raises(ArgumentRefused) as raised:
        Medium("water", fields={"acoustic": flow})
    assert raised.value.refusals[0].code == "medium.field-role"
    with pytest.raises(ArgumentRefused) as raised:
        Medium("water", flow=FVM(domain=PeriodicBox(length=mm(1.0))), fields={"flow": flow})
    assert raised.value.refusals[0].code == "medium.flow-twice"


def test_a_box_margin_under_the_fields_reach_is_refused_when_planned():
    """A margin of solid under the band's and the interior's reach brings an image's fields onto the tube's own:
    refused with the route (flow.box-margin); the control, the floor itself, builds."""
    for margin, refused in ((2, True), (3, False)):
        exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=ms(20.0))
        exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                          length=mm(10.0))))
        flow = FVM(domain=PeriodicBox(length=mm(0.25), margin=margin), cells_across=8, stokes=True,
                   scheme=Scheme.Steady())
        exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
        exp.declare("water.viscosity", mPa_s(1.0), reason="water")
        exp.declare("water.density", kg_per_m3(1000.0), reason="water")
        if refused:
            with pytest.raises(BuildRefused) as raised:
                exp.build()
            assert "flow.box-margin" in [r.code for r in raised.value.refusals]
        else:
            exp.build()


def test_a_run_past_the_reports_rows_is_refused_before_it_runs(float64):
    """Minor 1: the derivative solves' per-step rows are never clamped; a run that would pass the last row is refused
    before it runs (flow.report-rows), naming the rows and the route. The control: a run within them runs."""
    built = experiment(cells_across=8, report_rows=3).build()
    built.run(ms(60.0))  # three steps of 20 ms: within the rows
    with pytest.raises(ArgumentRefused) as raised:
        built.run(ms(20.0))
    assert [r.code for r in raised.value.refusals] == ["flow.report-rows"]
    assert "report_rows=4" in raised.value.refusals[0].action
