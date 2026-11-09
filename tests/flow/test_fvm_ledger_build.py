"""The flow's ledger through the build (F2b, D1 to D3): with ``EnergyLedger()`` the build meters the FVM as a receiver of
the ``fluid`` port kind, the ``BodyForce`` driving it is outside and supplies its work (the source rule, D2), and the
viscosity and the wall's band are held in the flow's step under their own names.

**The account closes on a driven Womersley flow** (a steady and a harmonic body force, Womersley number about 2.5,
one period), in both dtypes: IMEX Euler's step identity within its floor at every step, nothing unattributed, the
books' balance within their rounding, the losses under ``water.flow.viscosity`` and ``tube.wall``, the supply under
``body_force:water``; RK3, on route 1, every step judged to have dissipated, its drive's work under the same law.

**The control for the source rule on the flow** (D2's): the same build's plans and edges with one planted read, the
fluid's state into the drive's node, put the drive back inside the components (not outside, its work supplied by
nothing); without the planted read it is outside. The graph is deterministic, so the control fails every run."""

from __future__ import annotations

import numpy as np
import pytest
from fvm_cases import in_dtype

from mime.build import _ledger as wiring
from mime.build._builders import STATE, EdgeDecl
from mime.core import ledger
from mime.dynamics import FVM, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.observations import EnergyLedger
from mime.scene import PeriodicBox, Scene, Tube, Z
from mime.things import Medium, Structure
from mime.units import Hz, kg_per_m3, m_per_s2, mm, mPa_s, ms

#: The Womersley drive: a 1 mm tube of water, 4 Hz (Womersley number R sqrt(omega / nu) about 2.5), a steady part
#: and a harmonic one, at a step under IMEX Euler's CFL line (the predicted peak's 0.32 of it) and RK3's viscous number
#: at eight cells across: one period, 250 steps.
FREQUENCY, PERIOD, STEP = 4.0, ms(250.0), ms(1.0)
LAWS = {"body_force:water": ledger.UNMETERED, "water.flow.viscosity": ledger.DISSIPATIVE,
        "tube.wall": ledger.DISSIPATIVE}


def womersley(scheme) -> Experiment:
    exp = Experiment("womersley_ledger", scene=Scene.new(up=Z, units="m"), dt=STEP)
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                      length=mm(10.0))))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": FVM(
        domain=PeriodicBox(length=mm(0.5)), cells_across=8, scheme=scheme)}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(0.1), amplitude=m_per_s2(0.4),
                         frequency=Hz(FREQUENCY)), medium="water")
    exp.observe(EnergyLedger())
    return exp


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_imex_euler_s_account_closes_on_a_driven_womersley_flow(dtype):
    with in_dtype(dtype):
        built = womersley(Scheme.IMEXEuler()).build()
        result = built.run(PERIOD)
    held = ledger.deliveries(built.graph.get_node("water.flow").constants())
    assert {d.law: d.energy_class for d in held} == LAWS
    account = result.energy
    assert account is not None
    assert account.findings == () and dict(account.unmetered) == {}
    assert float(account.identity_worst["water.flow"]) <= 1.0
    assert float(account.unattributed["water.flow"]) == 0.0
    assert abs(float(account.balance)) <= float(account.balance_floor)
    assert set(account.lost) == {"water.flow.viscosity", "tube.wall"} and all(v > 0.0 for v in account.lost.values())
    assert set(account.supplied) == {"body_force:water"}
    assert set(account.paired_supply) == {"body_force:water"}
    records = result.final["water.flow"]
    assert float(np.sum(records[f"{ledger.RECORDS}gift_steps"])) == 0.0
    assert "not metered" not in result.text()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_rk3_s_every_step_dissipates_on_a_driven_womersley_flow_its_drive_under_its_law(dtype):
    with in_dtype(dtype):
        built = womersley(Scheme.RK3()).build()
        result = built.run(PERIOD)
    account = result.energy
    assert account is not None and account.findings == ()
    records = result.final["water.flow"]
    assert float(records[f"{ledger.RECORDS}route_one_negative_steps"]) == 0.0
    assert float(account.unattributed["water.flow"]) == 0.0
    assert set(account.paired_supply) == {"body_force:water"} and float(account.paired_supply["body_force:water"]) > 0.0
    assert set(account.lost) == {"water.flow.viscosity", "tube.wall"}


def _captured(exp: Experiment):
    """The plans and edges the build's wiring was given, and what it built."""
    real = wiring.attach
    held: dict[str, object] = {}

    def capture(plans, built, edges):
        held.update(plans=plans, edges=list(edges))
        return real(plans, built, edges)

    wiring.attach = capture
    try:
        built = exp.build()
    finally:
        wiring.attach = real
    return held["plans"], held["edges"], built


def test_control_a_drive_whose_node_reads_the_fluid_s_state_is_not_outside():
    plans, edges, built = _captured(womersley(Scheme.IMEXEuler()))
    outside, supplied = wiring.boundary(plans, edges)
    assert outside == ("body_force:water",) and supplied == {"body_force:water": ("body_force:water",)}
    drive = next(e for e in edges if e.by == "body_force:water")
    read = EdgeDecl(by="body_force:water", source=("water.flow", "lumen_flux"), target=(drive.source[0], "flux"),
                    additive=False, units="m^3/s", seq=99, carries=STATE)
    outside, supplied = wiring.boundary(plans, [*edges, read])
    assert outside == () and "body_force:water" not in supplied
