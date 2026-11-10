"""The open vessel through the experiment (F2b's PR 2): ``FVM(domain=VesselRegion())`` on a tube cut at its own ends,
an ``Inflow`` at ``tube.inlet`` and a ``Windkessel`` at ``tube.outlet``, built, run and metered; the account names the
inflow as what supplied the work, the Windkessel's stored energy and its loss by name; and the plan's refusals, each a
way the openings cannot be stepped."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mime.build import BuildRefused
from mime.dynamics import FVM, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce, Inflow, Windkessel
from mime.observations import EnergyLedger
from mime.scene import PeriodicBox, Scene, Tube, VesselRegion, Z
from mime.testing import ledger_closes
from mime.things import Medium, Structure
from mime.units import Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_Pa, m3_per_s, m_per_s2, mm, ms

RADIUS, LENGTH, MU = 0.5e-3, 1e-3, 1e-3
#: The inflow at a mean speed of 1 mm/s, and the tube's own Poiseuille resistance.
FLUX = 1e-3 * math.pi * RADIUS ** 2
RESISTANCE = 8.0 * MU * LENGTH / (math.pi * RADIUS ** 4)


def vessel(*, domain=None, axis=+Z, scheme=None, laws=("inlet", "outlet"), inflows=("inlet",)) -> Experiment:
    exp = Experiment("open_tube", scene=Scene.new(up=Z, units="m"), dt=ms(2.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=axis, inner_radius=mm(RADIUS * 1e3), wall=mm(0.2),
                                                      length=mm(LENGTH * 1e3))))
    flow = FVM(domain=VesselRegion() if domain is None else domain, cells_across=8,
               scheme=Scheme.IMEXEuler() if scheme is None else scheme)
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(MU), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    for where in laws:
        if where in inflows:
            exp.attach(Inflow(steady=m3_per_s(FLUX if where == "inlet" else -FLUX)), medium="water",
                       surface=f"tube.{where}")
        else:
            exp.attach(Windkessel(proximal=Pa_s_per_m3(0.3 * RESISTANCE), compliance=m3_per_Pa(1e-13),
                                  distal=Pa_s_per_m3(3.0 * RESISTANCE)), medium="water", surface=f"tube.{where}")
    return exp


def test_an_open_vessel_runs_with_its_laws_and_its_books_close(float64):
    exp = vessel()
    exp.observe(EnergyLedger())
    built = exp.build()
    result = built.run(ms(20.0))
    final = result.final["water.flow"]
    np.testing.assert_allclose(np.asarray(final["opening_flux"]), [-FLUX, FLUX], rtol=1e-12)
    assert ledger_closes(result).passed
    account = result.energy
    assert account is not None
    assert set(account.supplied) == {"inflow:tube.inlet"}
    assert "windkessel:tube.outlet" in account.stored and "windkessel:tube.outlet" in account.lost
    assert account.lost["windkessel:tube.outlet"] > 0.0
    text = built.report.text()
    assert "metered (fluid and opening ports)" in text and "supplied by inflow:tube.inlet" in text


def test_a_saved_open_vessel_loads_and_runs_bit_for_bit(float64, tmp_path):
    exp = vessel()
    exp.save(tmp_path / "saved")
    loaded = Experiment.load(tmp_path / "saved")
    finals = [each.build().run(ms(10.0)).final["water.flow"] for each in (exp, loaded)]
    for name in ("velocity_z", "opening_pressure", "opening_state"):
        assert np.array_equal(np.asarray(finals[0][name]), np.asarray(finals[1][name]))
    text = (tmp_path / "saved" / "experiment.mime.yaml").read_text()
    assert "group: vessel_region" in text and "name: windkessel" in text and "name: inflow" in text


def refused(exp: Experiment) -> list[str]:
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    return [r.code for r in raised.value.refusals]


def test_each_opening_takes_one_law(float64):
    assert "flow.opening-law" in refused(vessel(laws=("inlet",)))
    doubled = vessel()
    doubled.attach(Windkessel(proximal=Pa_s_per_m3(RESISTANCE), compliance=m3_per_Pa(1e-13),
                              distal=Pa_s_per_m3(RESISTANCE)), medium="water", surface="tube.inlet")
    assert "flow.opening-law" in refused(doubled)


def test_a_law_at_a_surface_that_is_no_opening_is_refused(float64):
    exp = vessel()
    exp.attach(Inflow(steady=m3_per_s(FLUX)), medium="water", surface="tube.inner_surface")
    assert "flow.opening-unknown" in refused(exp)


def test_inflows_at_every_opening_must_cancel(float64):
    """Two inflows that cancel (one giving what the other takes) are well posed through the gauge's row; two that do
    not are refused."""
    exp = vessel(inflows=("inlet", "outlet"))
    built = exp.build()
    final = built.run(ms(4.0)).final["water.flow"]
    np.testing.assert_allclose(np.asarray(final["opening_flux"]), [-FLUX, FLUX], rtol=1e-12)
    unbalanced = vessel(laws=("inlet",), inflows=("inlet",))
    unbalanced.attach(Inflow(steady=m3_per_s(FLUX)), medium="water", surface="tube.outlet")
    assert "flow.openings-ill-posed" in refused(unbalanced)


def test_a_periodic_box_takes_no_law_at_an_opening(float64):
    exp = vessel(domain=PeriodicBox(length=mm(0.25)), laws=("outlet",))
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    assert "flow.no-openings" in refused(exp)


def test_an_open_vessel_steps_by_imex_euler_or_its_steady_limit(float64):
    assert "flow.open-scheme" in refused(vessel(scheme=Scheme.RK3()))


def test_a_tilted_open_vessel_is_refused(float64):
    assert "flow.open-axis" in refused(vessel(axis=(1.0, 0.0, 1.0)))
