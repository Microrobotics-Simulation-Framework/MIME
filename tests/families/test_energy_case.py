"""The energy case's controls (``CaseKind.ENERGY``; F0-1 M6.4): each class's exact identity on the law alone catches a law
planted with one defect, on every run. The family walk runs the case on every member of a family that lists it
(``test_family_walk.py``); here each planted law fails it, and only it among the cases that read the same draws.

- gravity with its weight a hundredth heavier (its buoyancy as it is): its wrench is no longer the negative gradient of
  its energy, by 50 floors and more on every draw (P1a-1's measurement on the law alone: 477 floors at the least in
  float32);
- a drag whose resistance has an indefinite part (F0's spike FU1's: 1.1 times the least eigenvalue of its scaled
  symmetric part taken off along its eigenvector), its scaled least eigenvalue about -0.1 against a floor of 16 eps:
  not positive semidefinite, on every draw;
- a contact whose rule at arrival gives energy (the velocity it projects kept 1 % faster): the body placed on its
  surface moving outward, so that the rule acts on almost every draw (about 5 % of float32 draws leave the real rule's
  removal below one floor, and a few exactly zero; the plant keeps the velocity 1.01 times whatever the removal, so it
  is caught on every draw), the rule's removal is negative by 2 % of the body's kinetic energy of translation, about
  1e5 floors in float32;
- a law that declares no class and states no reason: refused by the case, whatever its draws.

**What energy cannot see**, shown: a drag scaled by 0.9, or its coupling's sign flipped, passes the energy case and
every check of script A's books with the ledger, and the hydrodynamic family's oracle case fails each.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax.numpy as jnp
import pytest
from family_cases import a_drag, surrogate, weight_and_buoyancy

from mime.build import InteractionBuilder
from mime.core import VELOCITY_RULES
from mime.interactions import Contact, Drag, Gravity
from mime.interactions.hydrodynamic import uniform_tube
from mime.nodes.body import GravityNode
from mime.nodes.hydro import SurrogateDragNode
from mime.registry import BuildContext, Confinement, Plan, PlanContext, Problem, Value, register_interaction
from mime.testing import CaseKind, conformance
from mime.units import mm
from mime.vocabulary import CONFINEMENT, register_namespace

register_namespace("ledgered", "0.0.1")


class _HeavierNode(GravityNode):
    @override
    def wrench(self, p: Any, pose: Any) -> Any:
        return super().wrench(p, pose) + jnp.concatenate([0.01 * p["mass"] * jnp.asarray(p["gravity"]),
                                                          jnp.zeros(3, dtype=jnp.asarray(p["gravity"]).dtype)])


@register_interaction
@dataclass(frozen=True, kw_only=True)
class HeavierGravity(Gravity):
    """Gravity whose weight is a hundredth heavier than its energy says."""

    name = "ledgered.heavier_gravity"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        centre = ctx.values["centre_of_buoyancy"]
        node = b.add_response_node(_HeavierNode(
            ctx.node_name(), ctx.timestep, mass=ctx.scalar_value("mass"),
            buoyant_volume=ctx.scalar_value("buoyant_volume"), fluid_density=ctx.scalar_value("fluid_density"),
            gravity=ctx.gravity, centre_of_buoyancy=centre if isinstance(centre, tuple) else None))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["wrench"], into=b.target("wrench"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class UnstatedGravity(Gravity):
    """Gravity that declares no class and states no reason."""

    name = "ledgered.unstated_gravity"
    energy_class = None


def indefinite(resistance: Any) -> Any:
    """FU1's plant: 1.1 times the least eigenvalue of the scaled symmetric part taken off along its eigenvector."""
    symmetric = 0.5 * (resistance + resistance.T)
    scale = jnp.sqrt(jnp.diag(symmetric))
    least, vectors = jnp.linalg.eigh(symmetric / (scale[:, None] * scale[None, :]))
    along = vectors[:, 0] * scale
    return resistance - 1.1 * least[0] * jnp.outer(along, along)


class _IndefiniteNode(SurrogateDragNode):
    @override
    def advance(self, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        found = super().advance(p, state, boundary_inputs, dt)
        return {**found, "resistance": indefinite(found["resistance"])}


class _ScaledNode(SurrogateDragNode):
    @override
    def advance(self, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        found = super().advance(p, state, boundary_inputs, dt)
        return {**found, "resistance": 0.9 * found["resistance"]}


class _FlippedNode(SurrogateDragNode):
    @override
    def advance(self, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        found = super().advance(p, state, boundary_inputs, dt)
        flip = jnp.concatenate([jnp.ones(3), -jnp.ones(3)]).astype(found["resistance"].dtype)
        return {**found, "resistance": flip[:, None] * found["resistance"] * flip[None, :]}


@dataclass(frozen=True, kw_only=True)
class _PlantedDrag(Drag):
    """Script A's surrogate drag, built with a planted node."""

    node: ClassVar[type[SurrogateDragNode]] = SurrogateDragNode

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        region, hull, weights = ctx.region("medium"), ctx.geometry(of="body"), self.weights
        tube = uniform_tube(region, ctx.bindings["medium"])
        assert not isinstance(tube, Problem) and region is not None and hull is not None and weights is not None
        family = hull.kind.family_parameters(hull.settings)
        node = b.add_response_node(self.node(
            ctx.node_name(), ctx.timestep, weights=weights.name, weights_sha256=weights.sha256,
            viscosity=ctx.scalar_value("viscosity"), helix_wavenumber=float(family["wavenumber"]),
            length=float(family["length"]), radius=float(family["radius"]),
            outer_radius=hull.kind.radial_reach(hull.settings), vessel_shape=region.shape, vessel_centre=tube.centre,
            vessel_axis=tube.axis, vessel_radius=tube.radius,
            lubrication_epsilon=ctx.scalar("lubrication.epsilon") if self.lubrication is not None else None))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class IndefiniteDrag(_PlantedDrag):
    """Script A's surrogate drag with an indefinite part in its resistance."""

    name = "ledgered.indefinite_drag"
    node = _IndefiniteNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ScaledDrag(_PlantedDrag):
    """Script A's surrogate drag, its resistance scaled by 0.9: what energy cannot see."""

    name = "ledgered.scaled_drag"
    node = _ScaledNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class FlippedDrag(_PlantedDrag):
    """Script A's surrogate drag, the sign of its coupling between translation and rotation flipped: a congruence by
    ``diag(1, -1)``, definite as the drag is, and what energy cannot see."""

    name = "ledgered.flipped_drag"
    node = _FlippedNode


def planted(cls: type[_PlantedDrag]) -> _PlantedDrag:
    held = a_drag()
    return cls.Surrogate(weights=held.weights, lubrication=held.lubrication)


VELOCITY_RULES["ledgered.gain"] = lambda velocity, normal, *, owner: 1.01 * velocity


@register_interaction
@dataclass(frozen=True, kw_only=True)
class GainingContact(Contact):
    """A projection whose rule at arrival keeps the velocity it projects, 1 % faster."""

    name = "ledgered.gaining_contact"

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(nodes=(), values={"extent": Value("body", "extent")},
                    confines=(Confinement("body", CONFINEMENT.name, "surface", ctx.scalar("margin"),
                                          "ledgered.gain"),))


def energy_of(law, family: str, **kwargs):
    (result,) = conformance(law, family, cases=[CaseKind.ENERGY], **kwargs).values()
    print(f"\n{result}")
    return result


def test_gravity_with_its_weight_a_hundredth_heavier_fails_the_energy_case_and_not_the_others():
    assert energy_of(HeavierGravity(), "gravity").failed
    results = conformance(HeavierGravity(), "gravity", oracle=weight_and_buoyancy,
                          cases=[CaseKind.FRAME, CaseKind.GRADIENTS])
    assert all(result.passed for result in results.values()), results


def test_a_drag_with_an_indefinite_part_in_its_resistance_fails_the_energy_case():
    assert energy_of(planted(IndefiniteDrag), "hydrodynamic").failed


@pytest.mark.parametrize("cls", [ScaledDrag, FlippedDrag], ids=["scaled by 0.9", "coupling flipped"])
def test_what_energy_cannot_see_passes_every_ledger_check_and_the_family_s_oracle_fails_it(cls):
    """A drag scaled by 0.9, or its coupling's sign flipped: the energy case passes it, script A's books with the ledger
    close over 0.25 s with no gift and nothing unattributed, and the hydrodynamic family's oracle case (the
    surrogate's formula again, in float64) fails it."""
    from script_a import REFERENCE_START, script_a

    from mime.observations import EnergyLedger
    from mime.units import s as seconds

    law = planted(cls)
    assert energy_of(law, "hydrodynamic").passed
    experiment = script_a("what_energy_cannot_see", drag=law, start=REFERENCE_START)
    experiment.observe(EnergyLedger())
    account = experiment.build().run(seconds(0.25)).energy
    assert account is not None
    assert all(worst <= 1.0 for worst in account.identity_worst.values()), account.text(details=True)
    assert account.unattributed == {"umr": 0.0} and abs(account.balance) <= account.balance_floor
    assert all(line.gift_steps == 0 and line.indefinite_steps == 0 for line in account.laws)
    (oracle,) = conformance(law, "hydrodynamic", oracle=surrogate, cases=[CaseKind.ORACLE]).values()
    print(f"\n{oracle}")
    assert oracle.failed


def test_a_contact_whose_rule_gives_energy_fails_the_energy_case():
    assert energy_of(GainingContact(rung="projection", margin=mm(0.1)), "contact").failed


def test_a_law_that_declares_no_class_and_states_no_reason_fails_the_energy_case():
    result = energy_of(UnstatedGravity(), "gravity")
    assert result.failed and "states no reason" in result.detail


@pytest.mark.parametrize("family", ["magnetic"])
def test_an_unmetered_law_passes_by_its_declaration_alone_and_says_so(family):
    """The kept magnetic torque 0.1.0, unmetered by its anomaly (MIME-ANO-137)."""
    from mime.interactions import KEPT

    result = energy_of(KEPT["magnetic.torque", "0.1.0"][0](), family)
    assert result.passed and result.detail.startswith("unmetered: ")
    assert "a pass by that declaration alone" in result.detail


def test_the_magnetic_pair_s_energy_case_needs_the_member_that_completes_it():
    """The family states its energy once for the pair: the torque alone is not the derivative of an energy, so the
    case asks for its force (together=) rather than judge half of it."""
    from mime.interactions import Magnetic

    result = energy_of(Magnetic.Torque(), "magnetic")
    assert result.failed and "together=" in result.detail


def test_the_kept_0_1_0_pair_taken_for_a_potential_fails_the_pair_s_energy_case_on_every_run(monkeypatch):
    """The magnetic pair's energy case has a control that fails: dev1's pair (MIME-ANO-137: both laws read at the
    magnet's centre of mass) declared a potential, which it is not, is not the negative gradient of ``-m . B`` at the
    magnet: the force differs from it by 2 to 4 % of its scale and the couple by 1 to 4 %, against a floor of 16 eps.
    Three runs of the case, each fails (the 0.2.0 pair passes it in the family walk). Declaring the class is what the
    case exists to catch; the real 0.1.0 declares none, and is unmetered by its anomaly."""
    from mime.interactions import KEPT
    from mime.vocabulary import EnergyClass

    torque, force = (KEPT[f"magnetic.{name}", "0.1.0"][0] for name in ("torque", "force"))
    for law in (torque, force):
        monkeypatch.setattr(law, "energy_class", EnergyClass.POTENTIAL)
        monkeypatch.setattr(law, "unmetered", "")
    for _ in range(3):
        result = energy_of(torque(), "magnetic", together=(force(),))
        assert result.failed and "together=" not in result.detail, result
