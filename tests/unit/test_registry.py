from dataclasses import dataclass

import pytest

from mime.gradients import GradientStatus
from mime.registry import (
    Dynamics,
    EntryKind,
    FieldProvider,
    Interaction,
    RoleKind,
    SupportRow,
    lookup,
    register_dynamics,
    register_field_provider,
    register_interaction,
    registered,
    registration_of,
    requires,
    versions_of,
)
from mime.testing.toys import ToyBody, ToyDrag
from mime.things import ThingKind
from mime.vocabulary import Capability, register_namespace

register_namespace("registering", "0.0.1")


def test_entries_are_keyed_by_name_and_version():
    record = lookup(EntryKind.INTERACTION, "toy.drag", "0.1.0")
    assert record is not None and record.cls is ToyDrag and record.namespace == "toy"
    assert registration_of(ToyDrag(rung="linear")) is record
    assert versions_of(EntryKind.DYNAMICS, "toy.body") == ["0.1.0"]
    assert lookup(EntryKind.INTERACTION, "toy.drag", "9.9.9") is None
    assert ToyBody in {r.cls for r in registered(EntryKind.DYNAMICS)}


def _interaction(**attributes):
    body = {"name": "registering.law", "version": "0.0.1", "differentiability": GradientStatus.EXACT,
            "roles": {"body": RoleKind.BODY}, **attributes}
    return dataclass(frozen=True, kw_only=True)(type("Law", (Interaction,), body))


@pytest.mark.parametrize(("attributes", "problem"), [
    ({"name": "Law"}, "lower-case words joined by dots"),
    ({"name": "law"}, "is in MIME's namespace"),
    ({"version": ""}, "version ''"),
    ({"differentiability": None}, "declare a mime.gradients.GradientStatus"),
    ({"roles": {}}, "a non-empty mapping of role names to RoleKind"),
    ({"rungs": ("a", "b")}, "has a 'rung' setting"),
    ({"support": (SupportRow("c", {"body": requires()}),)}, "rung 'c', which is not one of ()"),
])
def test_a_malformed_entry_is_refused_at_registration(attributes, problem):
    with pytest.raises(TypeError, match=problem):
        register_interaction(_interaction(**attributes))


def test_an_entry_is_a_frozen_dataclass():
    class Plain(Interaction):
        name = "registering.plain"
        version = "0.0.1"
        differentiability = GradientStatus.EXACT
        roles = {"body": RoleKind.BODY}

    with pytest.raises(TypeError, match="frozen dataclass"):
        register_interaction(Plain)


def test_a_key_held_by_another_class_is_refused():
    register_interaction(_interaction(name="registering.held"))
    with pytest.raises(ValueError, match="already registered by"):
        @register_interaction
        @dataclass(frozen=True, kw_only=True)
        class Other(Interaction):
            name = "registering.held"
            version = "0.0.1"
            differentiability = GradientStatus.EXACT
            roles = {"body": RoleKind.BODY}


def test_dynamics_claim_flags_from_the_closed_list_and_providers_one_provision():
    with pytest.raises(TypeError, match="the closed list"):
        @register_dynamics
        @dataclass(frozen=True, kw_only=True)
        class Flagged(Dynamics):
            name = "registering.flagged"
            version = "0.0.1"
            differentiability = GradientStatus.EXACT
            applies_to = frozenset({ThingKind.BODY})
            flags = frozenset({"fast"})
    with pytest.raises(TypeError, match="exactly one provision"):
        @register_field_provider
        @dataclass(frozen=True, kw_only=True)
        class Unprovided(FieldProvider):
            name = "registering.unprovided"
            version = "0.0.1"
            differentiability = GradientStatus.EXACT
            flags = frozenset({Capability.RIGID})
            publishes = {"toy.field.push": "push"}
