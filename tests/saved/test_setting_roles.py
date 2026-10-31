"""Every setting declares its role and its default's kind, and every default is a claim (design #191 items 1, 4 and
5): the CI test of the defaults, able to fail.

- **Completeness**: every setting of every registered entry, MIME's and a lab's, declares its role and its default's
  kind, or is required (no default). MIME's own declare every setting. The toys' laws are made up for the tests, and
  their namespace is exempt, as the docs walk exempts it.
- **Consistency**, refused at registration: a default on a physical or design setting that is no convention, a
  numerical default without its rule and basis. Its control: a planted default on a tube's radius.
- **Minimal builds generated from the registry**: each round-trip case's document with every setting that has a
  default taken out builds in exploratory, and research refuses exactly the numerical defaults exploratory stamps.
- **The ratchet on required settings**: each entry's required settings, every one listed with its reason in
  ``docs/ratchets/required_settings.txt``; one not listed fails, and so does a listed one that is no longer required."""

import dataclasses
import re
from pathlib import Path
from typing import Annotated

import pytest
from cases import CASES

from mime.build import BuildRefused
from mime.docs import load_namespaces, read_list
from mime.docs._check import LISTS, list_path
from mime.registry import (
    DefaultKind,
    Interaction,
    Role,
    SettingRole,
    numerical_default,
    registered,
    role_problems,
    setting_names,
)
from mime.scene import Tube
from mime.experiment import Experiment
from mime.units import Length, Quantity, mm
from mime.vocabulary import MIME_NAMESPACE

ROOT = Path(__file__).parents[2]
EXEMPT = frozenset({"toy"})


def judged():
    """MIME's entries and every installed namespace's (a lab's), the toys' aside: not an entry a test registers on its
    own in this process, which is no package's."""
    import importlib.metadata

    load_namespaces()
    installed = {MIME_NAMESPACE, *(point.name for point in importlib.metadata.entry_points(group="mime.entries"))}
    return [record for record in registered() if record.namespace in installed - EXEMPT]


def written(cls: type, name: str) -> object:
    held = next(f for f in dataclasses.fields(cls) if f.name == name)
    if held.default is not dataclasses.MISSING:
        return held.default
    if held.default_factory is not dataclasses.MISSING:
        return held.default_factory()
    return dataclasses.MISSING


def required(record) -> list[str]:
    """The settings of ``record`` with no default: none written, or ``None`` (not given) with no default declared."""
    found = []
    for name in setting_names(record.cls):
        role = record.roles.get(name)
        held = written(record.cls, name)
        if held is dataclasses.MISSING or (held is None and role is not None and role.default is DefaultKind.NONE
                                           and role.by is None):
            found.append(name)
    return found


def item(record, name: str) -> str:
    return f"{record.kind.value.replace(' ', '_')} {record.name} {record.version} {name}"


def test_completeness_every_setting_declares_its_role_or_is_required():
    undeclared, checked = [], 0
    for record in judged():
        for name in setting_names(record.cls):
            checked += 1
            if record.roles.get(name) is None and written(record.cls, name) is not dataclasses.MISSING:
                undeclared.append(item(record, name))
    assert checked > 50, "the scan found too few settings to judge"
    assert undeclared == [], f"a setting with a default declares its role and the default's kind: {undeclared}"


def test_mime_s_entries_declare_every_setting():
    undeclared = [item(record, name) for record in judged() if record.namespace == MIME_NAMESPACE
                  for name in setting_names(record.cls) if record.roles.get(name) is None]
    assert undeclared == []


def test_the_control_a_planted_default_on_a_tube_s_radius_is_refused_at_registration():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class DefaultedTube(Tube):
        inner_radius: Annotated[Quantity, Length] = mm(3.0)

    found = role_problems(DefaultedTube)
    assert len(found) == 1 and "inner_radius" in found[0] and "physical" in found[0], found
    from mime.registry import register_geometry

    with pytest.raises(TypeError, match="has the default"):
        register_geometry(DefaultedTube)
    assert role_problems(Tube) == []


def test_the_controls_a_numerical_default_without_its_rule_and_a_default_on_design_are_refused():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Planted(Interaction):
        setting_roles = {"step": SettingRole(Role.NUMERICAL, DefaultKind.NUMERICAL),
                         "gain": numerical_default("a rule", "a basis", by="no_such_method"),
                         "aim": SettingRole(Role.DESIGN, DefaultKind.NUMERICAL, "a rule", "a basis"),
                         "nothing": SettingRole(Role.PHYSICAL)}
        step: float = 1.0
        gain: float | None = None
        aim: float = 2.0

    found = "\n".join(role_problems(Planted))
    for expected in ("'step' declares a numerical default without its rule and its basis",
                     "computed by 'no_such_method', which is no method",
                     "'aim' is design and declares a numerical default",
                     "names 'nothing', which is not one of its settings"):
        assert expected in found, found


def minimal(entry: object) -> object:
    """``entry`` with every setting that has a default (a declared kind other than none) at its written default: what
    an experiment writes that gives only the required ones."""
    from mime.registry import registration_of

    record = registration_of(entry)
    if record is None:
        return entry
    taken = {name: written(record.cls, name) for name, role in record.roles.items()
             if role.default is not DefaultKind.NONE and written(record.cls, name) is not dataclasses.MISSING}
    return dataclasses.replace(entry, **taken)


def stripped(experiment: Experiment, profile: str) -> Experiment:
    """``experiment`` with every entry minimal, under ``profile``."""
    from mime.build import Attachment, ControllerDeclaration, ObservationDeclaration, ThingDeclaration
    from mime.things import Apparatus, Body, Structure

    def thing(held):
        if isinstance(held, Structure) and held.geometry is not None:
            return dataclasses.replace(held, geometry=minimal(held.geometry))
        if isinstance(held, Body):
            return dataclasses.replace(held, dynamics=minimal(held.dynamics), parts={
                key: dataclasses.replace(part, geometry=minimal(part.geometry)) for key, part in held.parts.items()})
        if isinstance(held, Apparatus):
            return dataclasses.replace(
                held, fields={k: minimal(v) for k, v in held.fields.items()},
                commands={k: minimal(v) for k, v in held.commands.items()},
                chain={k: minimal(v) for k, v in held.chain.items()},
                parts={k: dataclasses.replace(v, geometry=minimal(v.geometry)) for k, v in held.parts.items()})
        return held

    declarations = []
    for declaration in experiment._declarations:
        if isinstance(declaration, ThingDeclaration):
            declaration = ThingDeclaration(thing(declaration.thing))
        elif isinstance(declaration, Attachment):
            declaration = Attachment(minimal(declaration.law), declaration.bindings)
        elif isinstance(declaration, ObservationDeclaration):
            declaration = ObservationDeclaration(minimal(declaration.observation))
        elif isinstance(declaration, ControllerDeclaration):
            declaration = dataclasses.replace(declaration, controller=minimal(declaration.controller))
        declarations.append(declaration)
    return Experiment._declared(dataclasses.replace(experiment._settings, profile=profile), declarations)


def numerical_default_targets(error: BuildRefused) -> set[str]:
    return {r.message.split(" is not given")[0] for r in error.refusals if r.code == "setting.numerical-default"}


#: A determined default refused for having several answers: the bem drag's wall kind, where two or more match.
UNDETERMINED = re.compile(r"determined by the region the medium fills only where one wall kind matches it: .* which "
                          r"([2-9]|\d{2,}) wall kinds match")
#: The cases whose minimal build is refused for what its defaults place, not for a default: each with the codes and
#: why. A listed case that builds fails, naming its line to delete.
PLACED = {"gantry_magnet": (("provider.reader-inside",), "the gantry's placement and travel at their conventions (the "
                                                         "origin, unturned, along +x) put the magnet on the body, "
                                                         "where the dipole's field is singular")}


@pytest.mark.parametrize("case", [case for case in CASES if not case.refused], ids=lambda case: case.name)
def test_minimal_builds_from_the_registry(case):
    """Each case with only its entries' required settings (and the defaults of the rest) builds in exploratory; in
    research it is refused for exactly the numerical defaults exploratory stamps, and for nothing else of that kind."""
    if case.name in PLACED:
        codes, _ = PLACED[case.name]
        with pytest.raises(BuildRefused) as refused:
            stripped(case.make(), "exploratory").plan()
        assert tuple(sorted(set(refused.value.codes()))) == codes
        return
    try:
        exploratory = stripped(case.make(), "exploratory").plan()
    except BuildRefused as error:
        # A determined default with more than one answer in this process (a test registered a second wall kind for
        # the region) is refused with the candidates, which is its rule; nothing else may refuse a minimal build.
        undetermined = [r for r in error.refusals if UNDETERMINED.search(r.message)]
        assert undetermined and len(undetermined) == len(error.refusals), str(error)
        return
    exploratory = stripped(case.make(), "exploratory").plan()
    stamped = {item.target for item in exploratory.profile.pending if item.kind == "a numerical default"}
    declared = {f"{taken.path}" for taken in exploratory.profile.defaults if taken.kind is DefaultKind.NUMERICAL}
    assert stamped == declared
    research = stripped(case.make(), "research")
    try:
        research.plan()
        refused: set[str] = set()
    except BuildRefused as error:
        refused = numerical_default_targets(error)
    assert refused == stamped


def test_the_ratchet_on_required_settings():
    listed = read_list(list_path(ROOT, "required_settings"), LISTS["required_settings"])
    assert not listed.problems, listed.problems
    found = {item(record, name) for record in judged() for name in required(record)}
    assert found, "the scan found no required setting, so it could not fail"
    unlisted = sorted(found - listed.items())
    stale = sorted(listed.items() - found)
    assert unlisted == [], f"a setting is required and not listed (list it, or give it a default): {unlisted}"
    assert stale == [], f"a listed setting is no longer required: delete its line: {stale}"
