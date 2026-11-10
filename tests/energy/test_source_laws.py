"""The ledger's rule for a law that is a source (F2b, D2): **a law that declares no class, no read it declares is of a
component's state, and that delivers only loads its receivers pair is outside the components**, and what it delivers is
supplied work, named by the law itself (a body force driving a metered flow: its node computes its waveform from its
own parameters). The rule is for laws that declare no class, and only those: a class says how a law's work is booked.

Each test builds a body with one law on it and reads what the build's wiring found. The controls, each a
discrimination that fails every run (the builds are deterministic), each against a planted rule that drops one part of
the real one:

- **a class-less law whose node reads a body's state is not outside**: the same law, class-less, on a body whose centre
  of buoyancy is off its centre of mass, reads the body's pose and is a component's law, not a source; with the centre
  of buoyancy at the centre of mass it reads nothing and is a source;
- **a class-less law reading a field at the body's pose (``at=``) is supplied by the field**: the read's points edge
  runs from the body to the field's provider, into no node of the law's; the rule judging only the reads into the
  law's own nodes (planted, the rule as first built) counts the law outside and books the field's whole supply under
  the law's name;
- **a class-less resistance that reads nothing is no source**: its work is paired with the body's own solution; the
  rule counting every delivery (planted, the rule as first built) books the body's loss as a negative supply;
- **gravity stays a potential**: gravity on a body with no couple reads nothing too, and declares its class, so it is
  not outside and its work is not supplied; with the rule's class restriction taken out (planted), it is counted
  outside, which this test sees; and in every saved case that carries gravity it stays a potential.

The flow's own control (a drive whose node reads the fluid's state is not outside) is in
``tests/flow/test_fvm_ledger_build.py``."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest
from maddening.core.node import SimulationNode

from mime.build import _ledger as wiring
from mime.build._builders import DELIVERY, FIELD, STATE
from mime.core import ledger, node_fields, node_input
from mime.gradients import GradientStatus
from mime.interactions import Gravity
from mime.observations import EnergyLedger
from mime.registry import Delivery, Interaction, Read, RoleKind, register_interaction
from mime.units import ms
from mime.vocabulary import (APPLIED_WRENCH, BACKGROUND_LOAD, FLUX_DENSITY, POSE, RESISTANCE, DeliveryKind,
                             register_namespace)

register_namespace("sourcecase", "0.0.1")


@register_interaction
@dataclasses.dataclass(frozen=True, kw_only=True)
class ClassLessWeight(Gravity):
    """Gravity's weight and buoyancy with no class declared and its reason stated: a law whose node reads the body's
    pose only where the body's centre of buoyancy is off its centre of mass."""

    name = "sourcecase.weight"
    energy_class = None
    unmetered = "a test's law: gravity's weight with its class taken out"


@register_interaction
@dataclasses.dataclass(frozen=True, kw_only=True)
class UnstatedWeight(ClassLessWeight):
    """The class-less weight with no reason stated, as an entry outside MIME may be."""

    name = "sourcecase.unstated_weight"
    unmetered = ""


def body_in_water(law: Interaction, *, couple: bool, profile: str = "exploratory"):
    """A body in water in a tube with ``law`` on it, its centre of buoyancy off its centre of mass where ``couple``."""
    from mime import Body, Experiment, Medium, State, Structure
    from mime.dynamics import RigidInertial
    from mime.scene import Scene, Tube, X, Z
    from mime.units import kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mm, ms, rad_per_s

    exp = Experiment("source_law", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(1.0),
                     profile=profile)
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                             length=mm(100.0))))
    exp.add_medium(Medium("water", fills=tube.lumen))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    centre = m((0.0, 0.0, 1e-4)) if couple else m((0.0, 0.0, 0.0))
    for path, value in (("umr.mass", kg(8.5e-5)), ("umr.inertia", kg_m2((1e-10,) * 3)),
                        ("umr.buoyant_volume", m3(6e-8)), ("umr.centre_of_buoyancy", centre),
                        ("water.density", kg_per_m3(1000.0))):
        exp.declare(path, value, reason="the test's body and water")
    exp.attach(law, body="umr", medium="water")
    exp.observe(EnergyLedger())
    return exp


def delivered(built, law: str) -> ledger.LedgerDelivery:
    return next(d for d in ledger.deliveries(built.graph.get_node("umr").constants()) if d.law == law)


def outside_line(built) -> str:
    return built.report.energy[-1]


def test_a_class_less_law_reading_nothing_of_the_components_is_outside_and_supplies_its_own_work():
    built = body_in_water(ClassLessWeight(), couple=False).build()
    law = "sourcecase.weight:umr"
    assert outside_line(built) == f"outside: {law} (receives no delivery from the components)"
    assert delivered(built, law).supplied_by == law
    assert delivered(built, law).energy_class == ledger.UNMETERED
    assert not any(line.startswith(f"{law}: declares no energy class") for line in built.report.energy)


def test_control_a_class_less_law_whose_node_reads_a_body_s_state_is_not_outside():
    """The same class-less law with the body's centre of buoyancy off its centre of mass: its node reads the body's
    pose, a component's state, so it is not a source; its delivery is booked unmetered, supplied by nothing."""
    built = body_in_water(ClassLessWeight(), couple=True).build()
    law = "sourcecase.weight:umr"
    assert outside_line(built) == "outside: nothing (receives no delivery from the components)"
    assert delivered(built, law).supplied_by == ""
    assert f"{law}: declares no energy class and reads nothing from outside: not metered: " \
           f"{ClassLessWeight.unmetered}" in built.report.energy


def test_gravity_without_a_couple_reads_nothing_and_stays_a_potential():
    built = body_in_water(Gravity(), couple=False).build()
    held = delivered(built, "gravity:umr")
    assert (held.energy_class, held.supplied_by) == (ledger.POTENTIAL, "")
    assert outside_line(built) == "outside: nothing (receives no delivery from the components)"


def planted_rule(*, own_nodes_only: bool = False, every_delivery: bool = False, any_class: bool = False):
    """The source rule with one of its parts taken out: judging only the reads into the law's own nodes (missing the
    points a field is read at), counting a law that delivers anything (not only loads its receivers pair), or holding
    for a law that declares a class."""
    def rule(plans, edges, units, outside):
        owners = wiring._owners(plans)  # noqa: SLF001
        found: set[str] = set()
        for plan in plans.entries:
            law = plan.entry
            if plan.section is not wiring.Section.INTERACTION or not isinstance(law, Interaction):
                continue
            if law.energy_class is not None and not any_class:
                continue
            mine = set(plan.node_names())
            reads = [e for e in edges if e.carries in (STATE, FIELD) and e.source[0] not in mine
                     and (e.target[0] in mine or (e.by == plan.owner and not own_nodes_only))]
            made = [e for e in edges if e.by == plan.owner and e.carries == DELIVERY and e.target[0] not in mine]
            loads = every_delivery or (made and all(
                wiring._paired_as(owners, *e.target) is DeliveryKind.LOAD for e in made))  # noqa: SLF001
            if mine and loads and all(units.get(e.source[0]) in outside for e in reads):
                found.add(plan.owner)
        return found
    return rule


def test_control_without_the_class_restriction_gravity_would_be_counted_outside(monkeypatch: pytest.MonkeyPatch):
    """The plant the test above sees: gravity on the same body, the rule's restriction taken out, is counted outside
    and its work booked as supplied by itself, its potential's change dropped from the components'."""
    monkeypatch.setattr(wiring, "_sources", planted_rule(any_class=True))
    built = body_in_water(Gravity(), couple=False).build()
    held = delivered(built, "gravity:umr")
    assert held.supplied_by == "gravity:umr"
    assert outside_line(built) == "outside: gravity:umr (receives no delivery from the components)"


def test_a_source_that_states_no_reason_is_still_refused_under_research():
    """A source's missing declaration is a declaration missing: the reason for declaring no class is still asked for
    under research (``ledger.unmetered``), though its work is booked as supplied."""
    from mime.build import BuildRefused

    law = "sourcecase.unstated_weight:umr"
    built = body_in_water(UnstatedWeight(), couple=False).build()
    assert f"{law}: declares no energy class and states no reason; it reads no component's state and delivers only " \
           f"loads, so it is outside and supplies its own work" in built.report.energy
    assert delivered(built, law).supplied_by == law
    with pytest.raises(BuildRefused) as refused:
        body_in_water(UnstatedWeight(), couple=False, profile="research").build()
    assert refused.value.codes() == ["ledger.unmetered"]


# -- the review's two shapes of class-less law (M1): a field read at the body's pose, a resistance reading nothing -----

class PushNode(SimulationNode):
    """A force along the field it is given, ``F = k B``."""

    def __init__(self, name, timestep, gain=1e-3):
        super().__init__(name, timestep, gain=float(gain))

    def initial_state(self):
        return node_fields({"wrench": jnp.zeros(6, dtype=jnp.zeros(()).dtype)})

    def boundary_input_spec(self):
        return {"flux_density": node_input("T", (3,))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else params
        b = jnp.asarray(boundary_inputs.get("flux_density", jnp.zeros(3)))
        return node_fields({"wrench": jnp.concatenate([p["gain"] * b, jnp.zeros(3, dtype=b.dtype)])})


@register_interaction
@dataclasses.dataclass(frozen=True, kw_only=True)
class FieldAtTheBody(Interaction):
    """A force along the field read where the body is, ``F = k B(x)``: its only read is the field, at the body's pose."""

    name = "sourcecase.field_push"
    version = "0.0.1"
    unmetered = "a test's law: a force along the field read at the body"
    differentiability = GradientStatus.EXACT
    builds = {"": (PushNode,)}
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    reads = {"flux_density": Read("field", FLUX_DENSITY.name, at=Read("body", POSE.name))}
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}

    def build(self, ctx, b):
        node = b.add_response_node(PushNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("flux_density"), into=node["flux_density"])
        b.connect(node["wrench"], into=b.target("wrench"))


class DamperNode(SimulationNode):
    """A constant isotropic resistance ``c I`` and no background load."""

    def __init__(self, name, timestep, c=1e-4):
        super().__init__(name, timestep, c=float(c))

    def initial_state(self):
        dtype = jnp.zeros(()).dtype
        return node_fields({"resistance": jnp.eye(6, dtype=dtype) * self.params["c"],
                            "background_load": jnp.zeros(6, dtype=dtype)})

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else params
        dtype = state["resistance"].dtype
        return node_fields({"resistance": jnp.eye(6, dtype=dtype) * jnp.asarray(p["c"], dtype=dtype),
                            "background_load": jnp.zeros(6, dtype=dtype)})


@register_interaction
@dataclasses.dataclass(frozen=True, kw_only=True)
class Damper(Interaction):
    """A constant resistance reading nothing: its load ``-c U`` is the body's solve's, paired with its solution."""

    name = "sourcecase.damper"
    version = "0.0.1"
    unmetered = "a test's law: a constant resistance"
    differentiability = GradientStatus.EXACT
    builds = {"": (DamperNode,)}
    roles = {"body": RoleKind.BODY}
    delivers = {"resistance": Delivery("body", RESISTANCE.name),
                "background_load": Delivery("body", BACKGROUND_LOAD.name)}

    def build(self, ctx, b):
        node = b.add_response_node(DamperNode(ctx.node_name(), ctx.timestep))
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))


FIELD_LAW, DAMPER_LAW = "sourcecase.field_push:umr", "sourcecase.damper:umr"


def body_by_a_magnet(law: Interaction, *, magnet: bool = True):
    """A body moving at 1 mm/s in a tube with ``law`` on it and, where ``magnet``, a point dipole on a kinematic motor
    above it (its field outside the components)."""
    from mime import Apparatus, Body, Experiment, State, Structure
    from mime.dynamics import Motor, RigidInertial
    from mime.fields import PointDipole
    from mime.scene import Cylinder, Scene, Tube, X, Z
    from mime.things import Part, PermanentMagnet
    from mime.units import A_m2, kg, kg_m2, m, m_per_s, mm, ms, rad, rad_per_s

    exp = Experiment("source_law_field", scene=Scene.new(up=Z, units="m"), dt=ms(1.0), profile="exploratory")
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                      length=mm(100.0))))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((1e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("umr.mass", kg(8.5e-5)), ("umr.inertia", kg_m2((1.0e-10, 1.2e-10, 0.8e-10))),
                        ("umr.magnetic_moment", A_m2((0.0, 1.7e-3, 0.0)))):
        exp.declare(path, value, reason="the test's body")
    if magnet:
        exp.add_apparatus(Apparatus(
            "spin", chain={"motor": Motor.Kinematic(axis=Z, prescribed="angle", initial_angle=rad(0.0),
                                                    at=m((0.0, 0.0, 0.1)))},
            parts={"magnet": Part(Cylinder(radius=m(17.5e-3), length=m(20e-3), axis=(1.0, 0.0, 0.0)),
                                  mount="motor.rotor")},
            magnets={"magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
            fields={"magnetic": PointDipole(part="magnet")}, commands={"motor.angle": rad(0.25)}))
        exp.declare("spin.magnetic_moment", A_m2(18.89), reason="the test's magnet")
        exp.attach(law, body="umr", field="spin.magnetic")
    else:
        exp.attach(law, body="umr")
    exp.observe(EnergyLedger())
    return exp


def test_a_class_less_law_reading_a_field_at_the_body_s_pose_is_supplied_by_the_field():
    """The read's points edge, from the body to the field's provider, is a read of the body: the law is no source, and
    its work is the field's supply, as any law reading a field outside is."""
    built = body_by_a_magnet(FieldAtTheBody()).build()
    assert delivered(built, FIELD_LAW).supplied_by == "spin.magnetic"
    assert FIELD_LAW not in outside_line(built) and "spin.magnetic" in outside_line(built)


def test_control_judging_only_the_law_s_own_nodes_books_the_field_s_supply_under_the_law(monkeypatch):
    monkeypatch.setattr(wiring, "_sources", planted_rule(own_nodes_only=True))
    built = body_by_a_magnet(FieldAtTheBody()).build()
    assert delivered(built, FIELD_LAW).supplied_by == FIELD_LAW
    assert FIELD_LAW in outside_line(built)


def test_a_class_less_resistance_reading_nothing_is_no_source_and_supplies_nothing():
    """Its resistance is a force the body's step solves for, paired with the body's own solution: no supply, its
    work booked under its name as neither a loss nor a supply (its note in the account)."""
    built = body_by_a_magnet(Damper(), magnet=False).build()
    assert {d.supplied_by for d in ledger.deliveries(built.graph.get_node("umr").constants())} == {""}
    assert outside_line(built) == "outside: nothing (receives no delivery from the components)"
    account = built.run(ms(20.0)).energy
    assert account is not None and dict(account.supplied) == {} and dict(account.paired_supply) == {}


def test_control_counting_every_delivery_books_the_resistance_s_loss_as_a_negative_supply(monkeypatch):
    monkeypatch.setattr(wiring, "_sources", planted_rule(every_delivery=True))
    built = body_by_a_magnet(Damper(), magnet=False).build()
    assert {d.supplied_by for d in ledger.deliveries(built.graph.get_node("umr").constants())} == {DAMPER_LAW}
    account = built.run(ms(20.0)).energy
    assert account is not None and float(account.paired_supply[DAMPER_LAW]) < 0.0


# -- gravity in every saved case that carries it ---------------------------------------------------------------------

#: The saved cases (``tests/saved/cases.py``) that carry gravity.
GRAVITY_CASES = ("script_a_body", "script_a", "script_a_uncoupled", "script_a_kept", "script_a_records", "hull_contact",
                 "hull_contact_looped", "bound_robot", "bound_robot_with_parts", "script_a_ramped", "script_a_metered")


def _gravity_in(name: str) -> tuple[set[tuple[str, str, str]], str]:
    """The saved case ``name`` built with the ledger: each gravity delivery's law, class and supplier, and the outside
    line."""
    import sys
    from pathlib import Path

    saved = str(Path(__file__).resolve().parents[1] / "saved")
    if saved not in sys.path:
        sys.path.append(saved)
    import cases

    make = {case.name: case.make for case in cases.CASES}.get(name) or getattr(cases, name)
    exp = make()
    if name != "script_a_metered":
        exp.observe(EnergyLedger())
    built = exp.build()
    held = {(d.law, d.energy_class, d.supplied_by) for node in built.graph.node_names
            for d in ledger.deliveries(dict(getattr(built.graph.get_node(node), "params", {}) or {}))
            if d.law.startswith("gravity:")}
    return held, outside_line(built)


@pytest.mark.parametrize("name", GRAVITY_CASES)
def test_gravity_stays_a_potential_in_every_saved_case_that_carries_it(name):
    held, outside = _gravity_in(name)
    assert held and all(energy_class == ledger.POTENTIAL and not supplier for _, energy_class, supplier in held), held
    assert "gravity:" not in outside


@pytest.mark.parametrize("name", ("script_a_metered", "bound_robot", "script_a_kept"))
def test_control_without_the_class_restriction_gravity_leaves_the_components_in_a_saved_case(name, monkeypatch):
    """The plant the test above sees, on saved cases that build in about a second: with the rule's class restriction
    taken out, gravity (whose node there reads nothing of the body) is counted outside and self-supplied."""
    monkeypatch.setattr(wiring, "_sources", planted_rule(any_class=True))
    held, outside = _gravity_in(name)
    assert held and all(supplier == law for law, _, supplier in held), held
    assert all(law in outside for law, _, _ in held)
