"""The registry walk's other half: every registered interaction, MIME's and the lab fixture's, has a family case in
``family_cases.py``, and every one that declares a family (``Interaction.family``) passes that family's whole suite
(``assert_conforms``), while every one that declares none says why in its case. An interaction registered without a
case fails here.

The controls: removing any case is caught, naming its entry; an interaction registered without one is caught; and a
law planted with one defect fails its family's case for it: gravity with its weight reversed fails the sign case, a
projection that keeps the outward velocity fails the contact sign case, even at a speed whose square float32 cannot
hold, and a projection that pushes the body 70 nm too far fails the contact oracle. The magnetic family's control (a
sign flip) is in ``tests/oracles/test_laws.py``, the hydrodynamic family's in ``test_hydrodynamic_family.py``.
"""

from dataclasses import dataclass
from typing import Any, override

import numpy as np
import pytest
from family_cases import FAMILY_CASES, RADIUS, FamilyCase, projected, weight_and_buoyancy

from mime.build import InteractionBuilder
from mime.core import VELOCITY_RULES
from mime.interactions import Contact, Gravity
from mime.nodes.body import GravityNode
from mime.docs import load_namespaces
from mime.registry import (
    BuildContext,
    Confinement,
    EntryKind,
    Interaction,
    Plan,
    PlanContext,
    Value,
    lookup,
    register_interaction,
)
from mime.testing import CaseKind, assert_conforms, conformance, family_named, installed_entries, probe_law
from mime.testing._families import off_the_contact_band, oracle_agrees
from mime.units import mm
from mime.vocabulary import CONFINEMENT, register_namespace


def uncased(cases: dict, registered: list[tuple[str, str, str]]) -> list[str]:
    return [" ".join(key) for key in map(tuple, registered) if key[0] == "interaction" and key not in cases]


@pytest.fixture(scope="module")
def registered():
    found = installed_entries()
    assert ("interaction", "magnetic.torque", "0.1.0") in found, "the walk would pass over an empty registry"
    return found


def test_every_registered_interaction_has_a_family_case(registered):
    missing = uncased(FAMILY_CASES, registered)
    assert not missing, "no family case for: " + "; ".join(missing) + ". Add one to tests/families/family_cases.py."


def test_removing_any_family_case_fails_the_walk_naming_its_entry(registered):
    for key in FAMILY_CASES:
        assert uncased({k: c for k, c in FAMILY_CASES.items() if k != key}, registered) == [" ".join(key)]


def test_an_interaction_registered_without_a_family_case_fails_the_walk(registered):
    assert uncased(FAMILY_CASES, [*registered, ("interaction", "toy.uncased", "0.1.0")]) == [
        "interaction toy.uncased 0.1.0"]


def declared_family(key: tuple[str, str, str]) -> str | None:
    """The family the registered interaction ``key`` declares, with every installed namespace loaded."""
    load_namespaces()
    record = lookup(EntryKind.INTERACTION, key[1], key[2])
    assert record is not None and issubclass(record.cls, Interaction), key
    return record.cls.family


def test_every_interaction_declares_a_family_or_its_case_says_why_it_has_none():
    for key, case in FAMILY_CASES.items():
        assert bool(declared_family(key)) != bool(case.no_family), key
        assert case.no_family or case.law is not None, key


@pytest.mark.parametrize("key", sorted(k for k, c in FAMILY_CASES.items() if not c.no_family), ids=" ".join)
def test_every_interaction_in_a_family_passes_its_suite(key):
    case: FamilyCase = FAMILY_CASES[key]
    assert case.law is not None
    law = case.law()
    family = type(law).family
    assert family is not None and family == declared_family(key), key
    assert_conforms(law, family, oracle=case.oracle, node_case=case.node_case,
                    swap_with=case.swap_with() if case.swap_with else None)


# -- planted defects ----------------------------------------------------------------------------------------------

register_namespace("walked", "0.0.1")


class _ReversedNode(GravityNode):
    @override
    def wrench(self, p: Any, pose: Any) -> Any:
        found = super().wrench(p, pose)
        return found.at[:3].multiply(-1.0)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ReversedGravity(Gravity):
    """Weight minus buoyancy pointing up."""

    name = "walked.reversed_gravity"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        centre = ctx.values["centre_of_buoyancy"]
        node = b.add_response_node(_ReversedNode(
            ctx.node_name(), ctx.timestep, mass=ctx.scalar_value("mass"),
            buoyant_volume=ctx.scalar_value("buoyant_volume"), fluid_density=ctx.scalar_value("fluid_density"),
            gravity=ctx.gravity, centre_of_buoyancy=centre if isinstance(centre, tuple) else None))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["wrench"], into=b.target("wrench"))


def reversed_oracle(read, settings):
    found = weight_and_buoyancy(read, settings)["wrench"]
    return {"wrench": np.concatenate([-found[:3], found[3:]])}


def test_gravity_with_its_weight_reversed_fails_the_sign_case_and_only_it():
    results = conformance(ReversedGravity(), "gravity", oracle=reversed_oracle,
                          cases=[CaseKind.SIGN, CaseKind.FRAME, CaseKind.ORACLE, CaseKind.GRADIENTS])
    print("\n" + "\n".join(f"{case.value}: {result}" for case, result in results.items()))
    assert {case for case, result in results.items() if not result.passed} == {CaseKind.SIGN}


VELOCITY_RULES["walked.keep"] = lambda velocity, normal, *, owner: velocity


@register_interaction
@dataclass(frozen=True, kw_only=True)
class SlidingContact(Contact):
    """A projection whose velocity rule keeps the outward velocity."""

    name = "walked.sliding_contact"

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(nodes=(), values={"extent": Value("body", "extent")},
                    confines=(Confinement("body", CONFINEMENT.name, "surface", ctx.scalar("margin"), "walked.keep"),))


def keeping(read, settings):
    expected = projected(read, settings)
    return {"pose": expected["pose"], "twist": np.asarray(read["twist"])}


def test_a_projection_that_keeps_the_outward_velocity_fails_the_contact_sign_case_and_only_it():
    results = conformance(SlidingContact(rung="projection", margin=mm(0.1)), "contact", oracle=keeping,
                          cases=[CaseKind.SIGN, CaseKind.FRAME, CaseKind.ORACLE])
    print("\n" + "\n".join(f"{case.value}: {result}" for case, result in results.items()))
    assert {case for case, result in results.items() if not result.passed} == {CaseKind.SIGN}


def contact_sign_holds(law, *, twist: float = 0.0, load: float = 0.0, dt: float = 1e-4) -> bool:
    """The contact sign rule on one step of ``law`` in the contact rig, from the draw the sign case failed on in CI
    (run 39912549151): a body 5.5 mm off the tube's axis, turned a third of a turn about (1, 1, 1), with no
    resistance, a velocity of ``twist`` m/s on each axis and a background load of ``load`` N on each (a thousandth of
    it in N m)."""
    contact = family_named("contact")
    probe = probe_law(law, contact)
    assert not isinstance(probe, str), probe
    state = {"pose": np.array([3.90625e-3] * 3 + [0.5] * 4, np.float32),
             "twist": np.array([twist] * 3 + [0.0] * 3, np.float32)}
    boundary = {"applied_wrench": np.zeros(6, np.float32), "resistance": np.zeros((6, 6), np.float32),
                "background_load": np.array([load] * 3 + [1e-3 * load] * 3, np.float32)}
    assert contact.sign is not None
    return contact.sign.holds(probe.response(boundary, probe.node.update(state, boundary, dt), state, dt))


def test_the_contact_sign_case_judges_a_vanishing_velocity_on_its_own_scale():
    """CI's draw: a load of 3.2e-24 N moves the body 3.7e-24 m/s in a step, and the projection leaves 1.7e-31 m/s of it
    outward, a quarter of float32's rounding of that speed. The rule took the speed's norm in float32, where its
    square underflows to zero, and so demanded an outward part of exactly zero. The control: a projection that keeps
    the outward velocity is caught at that speed."""
    assert contact_sign_holds(Contact.Projection(margin=mm(0.1)), load=3.1790528e-24)
    assert not contact_sign_holds(SlidingContact(rung="projection", margin=mm(0.1)), load=3.1790528e-24)


def test_the_contact_sign_case_floors_its_rounding_at_the_smallest_normal_number():
    """A velocity of float32's smallest normal number on each axis, stepped by a microsecond: XLA flushes the rule's
    products (8e-39) to zero, so the projection leaves the whole outward velocity, 1.7e-38 m/s, under the rule's floor
    of eight smallest normal numbers. The control: an outward velocity kept at 1e-36 m/s is caught."""
    assert contact_sign_holds(Contact.Projection(margin=mm(0.1)), twist=float(np.finfo(np.float32).tiny), dt=1e-6)
    assert not contact_sign_holds(SlidingContact(rung="projection", margin=mm(0.1)), twist=1e-36)


def test_a_projection_to_the_wrong_radius_fails_the_contact_oracle():
    """A closed form whose margin is off by 70 nanometres is told apart in every run. Each field has its own rounding
    floor, and each part of a composite its own (a pose's position apart from its quaternion), so a confined position
    is judged on the rounding of metres, about a tenth of a micrometre in float32 on a generic draw, where in 4b the
    whole state's floor (about 15 micrometres, from the quaternion) let a micrometre pass in about one run in eight.
    Measured over independent runs of this oracle case before choosing it: 70 nm caught in 210 of 210 runs, 50 nm in 58
    of 60, 30 nm in 6 of 10, 10 nm in none of 10; so 70 nm is the smallest planted error the oracle resolves in every
    run, and the control sits there."""
    def too_far(read, settings):
        return projected(read, {**settings, "margin": settings["margin"] + 7e-8})

    (result,) = conformance(Contact.Projection(margin=mm(0.1)), "contact", oracle=too_far,
                            cases=[CaseKind.ORACLE]).values()
    print(f"\n{result}")
    assert result.failed


def contact_step(pose, twist, dt):
    """One step of the contact rig's body from ``pose`` and ``twist`` (no wrench, no resistance, no background load,
    in float32 as the oracle case draws), with the oracle case's predicate and the probe's response to it."""
    probe = probe_law(Contact.Projection(margin=mm(0.1)), family_named("contact"))
    assert not isinstance(probe, str), probe
    state = {"pose": np.asarray(pose, np.float32), "twist": np.asarray(twist, np.float32)}
    boundary = {"applied_wrench": np.zeros(6, np.float32), "resistance": np.zeros((6, 6), np.float32),
                "background_load": np.zeros(6, np.float32)}
    response = probe.response(boundary, probe.node.update(state, boundary, dt), state, dt)
    return probe, response, lambda oracle: oracle_agrees(probe, oracle)(response, state, boundary, dt)


#: The oracle case's counterexample in the full local lane (contact 0.1.0, float32, shrunk by Hypothesis): the body
#: 1.4 mm from the tube's axis, which is the shrunk surface (3 mm less the body's 1.5 mm extent and the 0.1 mm
#: margin) to within 30 nm, moving outward and across it at 1 m/s. Neither the speed nor the step (0.1 ms to 10 ms)
#: failed it: it fails at 1e-4 m/s across and outward with a 10 ms step, and passes at 1 m/s a micrometre inside.
THE_FLAKE = ([0.00390625, 0.0, -0.00139997, 1.0, 0.0, 0.0, 0.0], [-1.0] * 6)


def test_a_body_that_starts_its_step_in_the_contact_band_is_held_in_the_solve_and_the_oracle_does_not_know_it():
    """On the flake's draw the node delivers the held step (the outward velocity taken off along the normal at the
    start of the step, less the gap it closes in the step, the body moved by what is left and projected back), which
    with no resistance is a closed form of its own; and the oracle case's closed form, a free step projected back
    along the normal at its end, is 6.6 micrometres and 0.066 m/s away in y at 0.1 ms. The oracle case keeps its
    draws off the band for that reason."""
    dt = 1e-4
    probe, response, agrees = contact_step(*THE_FLAKE, dt)
    pose, twist = (np.asarray(v, np.float64) for v in THE_FLAKE)
    start_normal = np.array([0.0, pose[1], pose[2]]) / np.hypot(pose[1], pose[2])
    gap = -(np.hypot(pose[1], pose[2]) - RADIUS + response.settings["extent"] + response.settings["margin"])
    held = twist[:3] - (twist[:3] @ start_normal - gap / dt) * start_normal
    expected = projected({"pose": np.concatenate([pose[:3] + dt * held, pose[3:]]),
                          "twist": np.concatenate([held, twist[3:]])}, response.settings)
    delivered = {label: np.asarray(v, np.float64) for label, v in response.delivered.items()}
    assert np.allclose(delivered["pose"][:3], expected["pose"][:3], rtol=0.0, atol=1e-8)
    assert np.allclose(delivered["twist"][:3], expected["twist"][:3], rtol=0.0, atol=1e-5)
    assert not agrees(projected), "the oracle agrees here, so the oracle case no longer needs to keep off the band"
    free = projected(response.read, response.settings)
    assert abs(delivered["pose"][1] - free["pose"][1]) > 5e-6 and abs(delivered["twist"][1] - free["twist"][1]) > 0.05


@pytest.mark.parametrize("beyond", [-4e-8, -3e-8, -1e-8, 0.0, 1e-8, 3e-8, 4e-8])
@pytest.mark.parametrize("dt", [1e-4, 1e-3, 1e-2])
def test_the_oracle_case_puts_a_body_drawn_in_the_contact_band_inside_it_and_then_agrees(beyond, dt):
    """The flake's draw, at each place across the band and at each step the case draws, once the oracle case has
    moved the body to eight bands inside the surface; and a body drawn a micrometre inside is not touched."""
    probe = probe_law(Contact.Projection(margin=mm(0.1)), family_named("contact"))
    assert not isinstance(probe, str), probe
    moved = off_the_contact_band(probe)
    assert moved is not None
    pose = np.array(THE_FLAKE[0], np.float32)
    pose[2] = np.float32(-(1.4e-3 + beyond))
    state = moved({"pose": pose, "twist": np.array(THE_FLAKE[1], np.float32)})
    assert abs(float(state["pose"][2]) - float(pose[2])) < 4e-7 and state["pose"].dtype == np.float32
    assert state["pose"][2] > pose[2], "moved inward: z is negative, the wall is at its far end"
    _, _, agrees = contact_step(state["pose"], state["twist"], dt)
    assert agrees(projected)
    clear = np.array(THE_FLAKE[0], np.float32)
    clear[2] = np.float32(-(1.4e-3 - 1e-6))
    assert np.array_equal(moved({"pose": clear, "twist": state["twist"]})["pose"], clear)


def test_a_projection_to_the_wrong_radius_fails_the_contact_oracle_on_the_flakes_draw_too():
    """The control for the case above: the oracle, once the flake's body is off the band, still tells apart a margin
    off by 10 micrometres (above this draw's rounding floor, 0.5 micrometres, which a 1 m/s step sets), so moving the
    draw has not blunted the case."""
    def too_far(read, settings):
        return projected(read, {**settings, "margin": settings["margin"] + 1e-5})

    probe = probe_law(Contact.Projection(margin=mm(0.1)), family_named("contact"))
    assert not isinstance(probe, str), probe
    moved = off_the_contact_band(probe)
    assert moved is not None
    state = moved({"pose": np.array(THE_FLAKE[0], np.float32), "twist": np.array(THE_FLAKE[1], np.float32)})
    _, _, agrees = contact_step(state["pose"], state["twist"], 1e-4)
    assert agrees(projected) and not agrees(too_far)


def test_a_law_that_confines_no_body_has_no_contact_band_to_keep_off():
    probe = probe_law(Gravity(), family_named("gravity"))
    assert not isinstance(probe, str), probe
    assert off_the_contact_band(probe) is None


def test_the_contact_oracle_case_draws_its_states_through_the_map_off_the_band(monkeypatch):
    """The wiring: the state map the oracle case hands MADDENING's draws moves the flake's body off the band, and the
    sign case's does not (it judges a body in the band, where the rule holds too)."""
    import mime.testing._families as families

    maps = {}
    real = families.run_verify_node

    def spying(node, *args, invariants, **kwargs):
        maps[next(iter(invariants))] = kwargs.get("constrain_state")
        return real(node, *args, invariants=invariants, **kwargs)

    monkeypatch.setattr(families, "run_verify_node", spying)
    conformance(Contact.Projection(margin=mm(0.1)), "contact", oracle=projected,
                cases=[CaseKind.ORACLE, CaseKind.SIGN])
    draw = {"pose": np.array(THE_FLAKE[0], np.float32), "twist": np.array(THE_FLAKE[1], np.float32)}
    assert maps["oracle"](draw)["pose"][2] > draw["pose"][2]
    assert np.array_equal(maps["sign"](draw)["pose"], draw["pose"])
