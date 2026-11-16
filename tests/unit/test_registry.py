from dataclasses import dataclass
from typing import Annotated, Literal, Optional, get_args

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
from mime.units import DIMENSIONLESS, FORCE, LENGTH, MASS, Angle, Length, Quantity, Time, deg, mm, s, unit_symbol
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


type _LengthAlias = Annotated[Quantity, Length]
type _MaybeQuantity = Quantity | None


def test_an_angle_setting_is_kept_as_an_angle_and_documented_with_its_unit():
    assert dimensioned_settings(_law_with(Annotated[Quantity, Angle], deg(30))) == (
        {"margin": DimensionedSetting(Angle)}, [])
    assert dimensioned_settings(_law_with(Annotated[Quantity | None, Angle], None)) == (
        {"margin": DimensionedSetting(Angle, optional=True)}, [])
    assert str(Angle) == "angle (rad)" and unit_symbol(Angle) == "rad" and unit_symbol(DIMENSIONLESS) == "1"
    assert type(get_args(Annotated[Quantity, Angle])[1]) is not type(get_args(Annotated[Quantity, DIMENSIONLESS])[1]), \
        "typing keeps one Annotated for equal markers of one type: an angle and a number are told apart by type"


@pytest.mark.parametrize(("annotation", "default", "problem"), [
    (Annotated[Quantity, Angle], 0.5, r"its default margin needs units of angle \(rad\); it was given the bare value 0.5"),
    (Annotated[Quantity, Angle], mm(1), r"its default margin needs units of angle \(rad\); it was given length"),
])
def test_an_angle_settings_default_is_checked_as_an_angle(annotation, default, problem):
    with pytest.raises(TypeError, match=problem):
        register_interaction(_law_with(annotation, default))


#: Each form a setting that holds a Quantity can be written in that the reader does not read, and the word its refusal
#: names it by. Each was left out of the entry's settings before registration refused it: the setting planned with its
#: value unchecked, as a number nothing had turned into SI.
UNREAD = {
    "metadata that is no dimension (a string)": Annotated[Quantity, "Length"],
    "metadata that is no dimension (a number)": Annotated[Quantity, 3],
    "Optional around the Annotated": Optional[Annotated[Quantity, Length]],  # noqa: UP045 - the spelling under test
    "| None around the Annotated": Annotated[Quantity, Length] | None,
    "| None around the Annotated, marked again": Annotated[Annotated[Quantity, Length] | None, "note"],
    "a type alias of the Annotated": _LengthAlias,
    "a type alias of the Quantity": _MaybeQuantity,
    "a list of quantities": list[Quantity],
    "a tuple of quantities": tuple[Quantity, ...],
    "a mapping of quantities": dict[str, Quantity],
}


@pytest.mark.parametrize("annotation", list(UNREAD.values()), ids=list(UNREAD))
def test_a_setting_of_a_type_the_reader_cannot_read_is_refused_at_registration_never_left_out(annotation):
    found, problems = dimensioned_settings(_law_with(annotation, None))
    assert found == {} and len(problems) == 1 and "setting 'margin'" in problems[0], \
        "the setting is in neither the settings it reads nor the refusals: it was dropped"
    assert "does not read" in problems[0] or "declares no dimension" in problems[0]
    with pytest.raises(TypeError, match="setting 'margin' holds .* the reader does not read"):
        register_interaction(_law_with(annotation, None))


def test_control_the_readable_forms_are_kept_and_a_setting_that_holds_no_quantity_needs_no_marker():
    """The refusal above is not every Quantity: the forms the reader reads are each kept, as the one that holds none is."""
    kept = [(Annotated[Quantity, Length], mm(1)), (Annotated[Quantity | None, Length], None),
            (Annotated[Annotated[Quantity, Length], "note"], mm(1)), (Annotated[Quantity, Angle], deg(1))]
    for annotation, default in kept:
        found, problems = dimensioned_settings(_law_with(annotation, default))
        assert list(found) == ["margin"] and problems == [], annotation
    for annotation in (float, Literal["a", "b"], Annotated[float, "a note"], tuple[float, ...], str | None):
        assert dimensioned_settings(_law_with(annotation, None)) == ({}, []), annotation


def test_a_setting_group_in_a_container_is_refused_not_left_out_with_its_dimensioned_settings():
    from mime.interactions import Lubrication
    from mime.registry import SettingGroup  # noqa: F401 - the base of the group held

    body = {"name": "registering.grouped", "version": "0.0.1", "differentiability": GradientStatus.EXACT,
            "roles": {"body": RoleKind.BODY}, "__annotations__": {"lubrications": tuple[Lubrication, ...]}}
    found, problems = dimensioned_settings(dataclass(frozen=True, kw_only=True)(type("Grouped", (Interaction,), body)))
    assert found == {} and len(problems) == 1 and "holds a setting group" in problems[0]


def test_quantities_in_a_container_the_entry_checks_itself_are_declared_so_and_the_arm_does():
    from mime.dynamics import Arm
    from mime.registry import CheckedByEntry

    declared = Annotated[tuple[tuple[str, Quantity], ...], CheckedByEntry]
    assert dimensioned_settings(_law_with(declared, ())) == ({}, [])
    assert dimensioned_settings(Arm)[1] == [] and "armature" not in dimensioned_settings(Arm)[0]
    with pytest.raises(TypeError, match="one marker"):
        register_interaction(_law_with(Annotated[tuple[tuple[str, Quantity], ...], CheckedByEntry, Length], ()))


# -- groups of settings ------------------------------------------------------------------------------------------------


def test_a_groups_dimensioned_settings_are_read_under_the_settings_name_optional_where_it_may_be_none():
    from mime.interactions import Drag, ForceSampling, Lubrication

    assert dimensioned_settings(Drag) == ({"lubrication.epsilon": DimensionedSetting(Length, optional=True),
                                           "spacing": DimensionedSetting(Length, optional=True),
                                           "sampling.contact_gap": DimensionedSetting(Length, optional=True)}, [])
    assert dimensioned_settings(Lubrication) == ({"epsilon": DimensionedSetting(Length)}, [])
    assert dimensioned_settings(ForceSampling) == ({"contact_gap": DimensionedSetting(Length, optional=True)}, [])


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
