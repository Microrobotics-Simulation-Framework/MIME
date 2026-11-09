"""The ledger's rule for a law that is a source (F2b, D2): **a law that declares no class and whose own nodes read no
component's state is outside the components**, and what it delivers is supplied work, named by the law (a body force
driving a medium's flow: its node computes its waveform from its own parameters). The rule is for laws that declare no
class, and only those: a class says how a law's work is booked.

Each test builds a body in water with one law on it and reads what the build's wiring found. The controls, each a
discrimination that fails every run (the builds are deterministic):

- **a class-less law whose node reads a body's state is not outside**: the same law, class-less, on a body whose centre
  of buoyancy is off its centre of mass, reads the body's pose and is a component's law, not a source; with the centre
  of buoyancy at the centre of mass it reads nothing and is a source;
- **gravity stays a potential**: gravity on a body with no couple reads nothing too, and declares its class, so it is
  not outside and its work is not supplied; with the rule's class restriction taken out (a planted rule), it is
  counted outside, which this test sees.

The flow's own control (a drive whose node reads the fluid's state is not outside) is in
``tests/flow/test_fvm_ledger_build.py``."""

from __future__ import annotations

import dataclasses

import pytest

from mime.build import _ledger as wiring
from mime.build._builders import FIELD, STATE
from mime.core import ledger
from mime.interactions import Gravity
from mime.observations import EnergyLedger
from mime.registry import Interaction, register_interaction
from mime.vocabulary import register_namespace

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


def _without_the_class_restriction(plans, edges, units, outside):
    """The source rule planted without its restriction to class-less laws: any law with own nodes reading nothing of
    the components."""
    found: set[str] = set()
    for plan in plans.entries:
        if plan.section is not wiring.Section.INTERACTION:
            continue
        mine = set(plan.node_names())
        if mine and all(units.get(e.source[0]) in outside for e in edges
                        if e.target[0] in mine and e.carries in (STATE, FIELD) and e.source[0] not in mine):
            found.add(plan.owner)
    return found


def test_control_without_the_class_restriction_gravity_would_be_counted_outside(monkeypatch: pytest.MonkeyPatch):
    """The plant the test above sees: gravity on the same body, the rule's restriction taken out, is counted outside
    and its work booked as supplied by itself, its potential's change dropped from the components'."""
    monkeypatch.setattr(wiring, "_sources", _without_the_class_restriction)
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
    assert f"{law}: declares no energy class and states no reason; its own nodes read nothing of the components, so " \
           f"it is outside and supplies its own work" in built.report.energy
    assert delivered(built, law).supplied_by == law
    with pytest.raises(BuildRefused) as refused:
        body_in_water(UnstatedWeight(), couple=False, profile="research").build()
    assert refused.value.codes() == ["ledger.unmetered"]
