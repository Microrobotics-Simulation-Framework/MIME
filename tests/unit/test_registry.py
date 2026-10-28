from dataclasses import dataclass
from typing import Annotated

import pytest

from mime.gradients import GradientStatus
from mime.registry import (
    DimensionedSetting,
    Dynamics,
    EntryKind,
    FieldProvider,
    Interaction,
    RoleKind,
    SupportRow,
    dimensioned_settings,
    lookup,
    register_dynamics,
    register_field_provider,
    register_interaction,
    registered,
    registration_of,
    requires,
    versions_of,
)
from mime.testing.toys import ToyBody, ToyDrag, ToyRecorder, ToySpring
from mime.things import ThingKind
from mime.units import FORCE, LENGTH, MASS, Length, Quantity, Time, mm, s
from mime.vocabulary import Capability, register_namespace

register_namespace("registering", "0.0.1")


def test_entries_are_keyed_by_name_and_version():
    record = lookup(EntryKind.INTERACTION, "toy.drag", "0.2.0")
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



def test_registration_reads_the_dimension_each_setting_declares():
    assert registration_of(ToySpring).settings == {"k": DimensionedSetting(FORCE / LENGTH),
                                                   "x0": DimensionedSetting(LENGTH)}
    assert registration_of(ToyBody).settings == {"mass": DimensionedSetting(MASS)}
    # rate, a period or a frequency, is checked against dt by its own rule and declares no marker.
    assert registration_of(ToyRecorder).settings == {}


def _law_with(annotation, *default):
    body = {"name": "registering.dimensioned", "version": "0.0.1", "differentiability": GradientStatus.EXACT,
            "roles": {"body": RoleKind.BODY}, "__annotations__": {"margin": annotation}}
    if default:
        body["margin"] = default[0]
    return dataclass(frozen=True, kw_only=True)(type("Dimensioned", (Interaction,), body))


def test_an_optional_dimensioned_setting_is_read_as_optional():
    assert dimensioned_settings(_law_with(Annotated[Quantity | None, Length], None)) == (
        {"margin": DimensionedSetting(Length, optional=True)}, [])
    assert dimensioned_settings(_law_with(Annotated[Quantity, Length], mm(1))) == (
        {"margin": DimensionedSetting(Length, optional=False)}, [])


@pytest.mark.parametrize(("annotation", "default", "problem"), [
    (Quantity, mm(1), "holds a Quantity and declares no dimension"),
    (Quantity | None, None, "holds a Quantity and declares no dimension"),
    (Annotated[float, Length], 1.0, "a dimensioned setting holds a Quantity"),
    (Annotated[Quantity, Length, Time], mm(1), "declares one dimension"),
    (Annotated[Quantity, Length], s(1), r"its default margin needs units of length \(m\); it was given time"),
    (Annotated[Quantity, Length], 0.001, r"its default margin needs units of length \(m\); it was given the bare "
                                         r"value 0.001"),
    (Annotated[Quantity, Length], None, r"its default margin needs units of length \(m\); it was given the bare "
                                        r"value None"),
])
def test_a_dimensioned_setting_declared_wrongly_is_refused_at_registration(annotation, default, problem):
    with pytest.raises(TypeError, match=problem):
        register_interaction(_law_with(annotation, default))


# -- groups of settings ------------------------------------------------------------------------------------------------


def test_a_groups_dimensioned_settings_are_read_under_the_settings_name_optional_where_it_may_be_none():
    from mime.interactions import Drag, Lubrication

    assert dimensioned_settings(Drag) == ({"lubrication.epsilon": DimensionedSetting(Length, optional=True),
                                           "spacing": DimensionedSetting(Length, optional=True)}, [])
    assert dimensioned_settings(Lubrication) == ({"epsilon": DimensionedSetting(Length)}, [])


def test_a_group_that_is_not_one_is_refused_at_registration():
    from dataclasses import field

    from mime.registry import SettingGroup, register_setting_group

    class Loose(SettingGroup):
        group = "registering.loose"

    with pytest.raises(TypeError, match="a setting group is a frozen dataclass"):
        register_setting_group(Loose)

    @dataclass(frozen=True, kw_only=True)
    class Inner(SettingGroup):
        group = "registering.inner"
        gap: Annotated[Quantity, Length] = field(default_factory=lambda: mm(1))

    @dataclass(frozen=True, kw_only=True)
    class Outer(SettingGroup):
        group = "registering.outer"
        inner: Inner | None = None

    with pytest.raises(TypeError, match="a group holds no group"):
        register_setting_group(Outer)
