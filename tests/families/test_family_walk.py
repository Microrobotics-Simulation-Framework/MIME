"""The registry walk's other half: every registered interaction, MIME's and the lab fixture's, has a family case in
``family_cases.py``, and every one that belongs to a family passes that family's whole suite (``assert_conforms``).
An interaction registered without a case fails here.

The controls: removing any case is caught, naming its entry; an interaction registered without one is caught; and a
law planted with one defect fails its family's case for it: gravity with its weight reversed fails the sign case, a
projection that keeps the outward velocity fails the contact sign case, and a projection that pushes the body too far
fails the contact oracle. The magnetic family's control (a sign flip) is in ``tests/oracles/test_laws.py``.
"""

from dataclasses import dataclass
from typing import Any, override

import numpy as np
import pytest
from family_cases import FAMILY_CASES, FamilyCase, projected, weight_and_buoyancy

from mime.build import InteractionBuilder
from mime.core import VELOCITY_RULES
from mime.interactions import Contact, Gravity
from mime.nodes.body import GravityNode
from mime.registry import BuildContext, Confinement, Plan, PlanContext, Value, register_interaction
from mime.testing import CaseKind, assert_conforms, conformance, installed_entries
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


def test_every_case_names_a_family_or_says_why_it_has_none():
    for key, case in FAMILY_CASES.items():
        assert bool(case.family) != bool(case.no_family), key
        assert case.no_family or case.law is not None, key


@pytest.mark.parametrize("key", sorted(k for k, c in FAMILY_CASES.items() if c.family), ids=" ".join)
def test_every_interaction_in_a_family_passes_its_suite(key):
    case: FamilyCase = FAMILY_CASES[key]
    assert case.law is not None
    assert_conforms(case.law(), case.family, oracle=case.oracle, node_case=case.node_case)


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


VELOCITY_RULES["walked.keep"] = lambda velocity, normal: velocity


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


def test_a_projection_to_the_wrong_radius_fails_the_contact_oracle():
    """A closed form whose margin is off by another margin (0.1 mm) is told apart in every run. One micrometre off is
    told apart in most runs and not all: the oracle case's floor is the largest rounding effect of a 64-ulp nudge
    across the body's whole state, and in about one run in eight some drawn states raise it past a micrometre."""
    def too_far(read, settings):
        return projected(read, {**settings, "margin": settings["margin"] + 1e-4})

    (result,) = conformance(Contact.Projection(margin=mm(0.1)), "contact", oracle=too_far,
                            cases=[CaseKind.ORACLE]).values()
    print(f"\n{result}")
    assert result.failed
