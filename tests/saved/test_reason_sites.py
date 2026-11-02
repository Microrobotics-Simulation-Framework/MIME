"""Every quantity that carries a reason has a site (design #191 item 2): a reason the saved form can hold is judged by
the profile wherever it is written, so research refuses a pending one by name; where nothing can judge one, ``reason=``
is an input error naming the slot (``source.no-site``), in every profile.

The test walks the saved form's schema (``ExperimentDocument``) for every place a reason can be written, a quantity's
``reason`` or a reason slot of its own, and for each puts ``{pending: true}`` there in a saved document of the cases
below, then builds the edited document as it is: research must refuse it by its pending code, and exploratory must
stamp it as one more pending item. A slot the cases cannot reach is listed with its reason, and fails when a case
reaches it; a slot neither reached nor listed fails, so a new slot in the schema needs a site, or a reason here."""

import copy
import dataclasses
import re
import shutil
import types
import typing
from pathlib import Path
from typing import Any, TypeAliasType, get_args, get_origin

import pytest
import yaml
from cases import script_a_records
from chain_cases import tracked_rpm
from profile_cases import Paperwork, settling_ball
from typing_extensions import NotRequired, ReadOnly, Required, get_type_hints, is_typeddict

from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.materials import Acceptance, Nominal, Record, packaged_record
from mime.saved import DOCUMENT_FILE
from mime.saved._schema import ExperimentDocument, QuantityDoc, ReasonDoc  # noqa: PLC2701 - the schema is the subject
from mime.scene import Box
from mime.testing.toys import ToyField, full_toy
from mime.things import Part
from mime.units import K, T, Condition, degC, m, mm, ms
from mime.validity import pending

#: N45's remanence read's four reasons, which its acceptance covers.
N45_REASONS = ("stated_without_origin", "condition_unstated:temperature", "condition_unstated:specimen_shape",
               "condition_unstated:test_method")
#: The codes research refuses a pending reason with, by where it is written.
PENDING_CODES = {"source.pending", "value.reason-pending", "material.accept-pending", "material.nominal-pending"}
#: The schema's slots no case reaches, each with why no experiment writes one (a test below holds each claim where the
#: registry can): a group of settings in an entry that takes none, a flow with no flow entry registered.
UNREACHED = {
    ("things", "*", "initial", "*", "settings", "*"): "a body's initial state holds quantities, never a group",
    ("things", "*", "dynamics", "flow", "settings", "*"): "no flow entry is registered in dev1",
    ("things", "*", "dynamics", "flow", "settings", "*", "settings", "*"): "no flow entry is registered in dev1",
    **{(*head, "settings", "*", "settings", "*"): "no registered entry of this kind takes a group of settings"
       for head in (("controller",), ("observations", "[]"), ("things", "*", "chain", "[]"),
                    ("things", "*", "commands", "*"), ("things", "*", "dynamics", "motion"),
                    ("things", "*", "fields", "*"), ("things", "*", "geometry"),
                    ("things", "*", "parts", "*", "solid"))},
}


def slots(held: object, path: tuple[str, ...] = (), stack: frozenset[object] = frozenset()
          ) -> typing.Iterator[tuple[tuple[str, ...], str]]:
    """Every place the schema lets a reason be written: a quantity (whose ``reason`` key is optional) or a reason slot,
    as a path of keys with ``*`` for a mapping's key and ``[]`` for a list's item."""
    if isinstance(held, TypeAliasType):
        if held is ReasonDoc:
            yield path, "reason"
        elif held not in stack:
            yield from slots(held.__value__, path, stack | {held})
        return
    origin = get_origin(held)
    if origin in (ReadOnly, NotRequired, Required):
        yield from slots(get_args(held)[0], path, stack)
    elif held is QuantityDoc:
        yield path, "quantity"
    elif origin in (types.UnionType, typing.Union):
        for arm in get_args(held):
            yield from slots(arm, path, stack)
    elif origin is list:
        yield from slots(get_args(held)[0], (*path, "[]"), stack)
    elif origin is dict:
        yield from slots(get_args(held)[1], (*path, "*"), stack)
    elif is_typeddict(held) and held not in stack:
        for key, inner in get_type_hints(held, include_extras=True).items():
            yield from slots(inner, (*path, key), stack | {held})


SLOTS = dict(slots(ExperimentDocument))


def instances(document: object, pattern: tuple[str, ...], kind: str, path: tuple[object, ...] = ()
              ) -> typing.Iterator[tuple[object, ...]]:
    """Where ``pattern`` is in ``document``: a quantity's mapping, or a reason slot's place."""
    if not pattern:
        if kind == "reason" or (isinstance(document, dict) and {"value", "unit"} <= set(document)
                                and not {"plus_minus", "covers"} & set(document)):
            yield path
        return
    head, *rest = pattern
    if head == "[]" and isinstance(document, list):
        for index, held in enumerate(document):
            yield from instances(held, tuple(rest), kind, (*path, index))
    elif head == "*" and isinstance(document, dict):
        for key, held in document.items():
            yield from instances(held, tuple(rest), kind, (*path, key))
    elif isinstance(document, dict) and head in document:
        yield from instances(document[head], tuple(rest), kind, (*path, head))
    elif kind == "reason" and not rest and isinstance(document, dict) and head == "reason":
        yield path + (head,)


def with_pending(document: dict[str, Any], place: tuple[object, ...], kind: str) -> dict[str, Any]:
    edited = copy.deepcopy(document)
    held: Any = edited
    for key in place[:-1]:
        held = held[key]
    if kind == "quantity":
        held[place[-1]]["reason"] = {"pending": True}
    else:
        held[place[-1]] = {"pending": True}
    return edited


def ball() -> Experiment:
    """The settling ball with a declared value, a source, a condition, a motion setting, a part placed and a nominal:
    sealed but for its numerical defaults, which it writes."""
    exp = settling_ball(Paperwork(profile="research", acceptance="water is Newtonian", density="steel",
                                  spacing=mm(0.2, reason="converged"), blob=0.5, quadrature_refinement=4,
                                  sources={"drag:ball.blob": "usual", "drag:ball.quadrature_refinement": "usual"}))
    body = next(d.thing for d in exp._declarations if getattr(getattr(d, "thing", None), "name", None) == "ball")
    placed = dataclasses.replace(body, dynamics=RigidInertial(rate=ms(1.0)), parts={
        **body.parts, "bead": Part(Box(size=mm((0.1, 0.1, 0.1))), at=mm((0.1, 0.0, 0.0)), inside="shell")})
    exp._declarations = [dataclasses.replace(d, thing=placed) if getattr(d, "thing", None) is body else d
                         for d in exp._declarations]
    return exp


def toy_field() -> Experiment:
    """The full toy, its field's wavenumber written: a field provider's quantity setting."""
    exp = full_toy()
    exp._declarations = [
        dataclasses.replace(d, thing=dataclasses.replace(d.thing, fields={"push": ToyField(wavenumber=2.0 / m(1.0))}))
        if getattr(getattr(d, "thing", None), "name", None) == "source" else d for d in exp._declarations]
    return exp


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The cases, each built and saved once: the settling ball, script A's record arm, A0's tracked apparatus, the full
    toy with its field's setting written."""
    folders = {}
    for name, make in (("ball", ball), ("script_a_records", script_a_records), ("tracked_rpm", tracked_rpm),
                       ("toy_field", toy_field)):
        folder = tmp_path_factory.mktemp(name)
        make().build().save(folder / "saved", steps=1)
        folders[name] = folder / "saved"
    return folders


def plan(document: dict[str, Any], folder: Path, into: Path, profile: str):
    shutil.copytree(folder, into, dirs_exist_ok=True)
    (into / DOCUMENT_FILE).write_text(yaml.safe_dump({**document, "profile": profile}, sort_keys=False))
    try:
        return Experiment.from_document(into / DOCUMENT_FILE).plan(), None
    except BuildRefused as refused:
        return None, refused


def reached(saved: dict[str, Path]) -> dict[tuple[str, ...], tuple[str, tuple[object, ...]]]:
    found: dict[tuple[str, ...], tuple[str, tuple[object, ...]]] = {}
    for name, folder in saved.items():
        document = yaml.safe_load((folder / DOCUMENT_FILE).read_text())
        for pattern, kind in SLOTS.items():
            if pattern not in found:
                place = next(instances(document, pattern, kind), None)
                if place is not None:
                    found[pattern] = (name, place)
    return found


def test_every_slot_the_schema_has_is_reached_or_listed_with_its_reason(saved):
    found = reached(saved)
    assert not set(found) & set(UNREACHED), f"a listed slot is reached by a case now: {set(found) & set(UNREACHED)}"
    missing = sorted(set(SLOTS) - set(found) - set(UNREACHED))
    assert missing == [], f"a place the schema holds a reason that no case reaches and no reason lists: {missing}"
    assert len(SLOTS) >= 30, "the walk found too few slots to judge"


@pytest.mark.parametrize("pattern", sorted(set(SLOTS) - set(UNREACHED)), ids=".".join)
def test_a_pending_reason_in_every_slot_is_refused_in_research_and_stamped_in_exploratory(saved, pattern, tmp_path):
    found = reached(saved)
    assert pattern in found, f"no case reaches {'.'.join(pattern)}"
    name, place = found[pattern]
    folder = saved[name]
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text())
    edited = with_pending(document, place, SLOTS[pattern])
    _, base = plan(document, folder, tmp_path / "base", "research")
    held_base = set(base.codes()) if base is not None else set()
    _, refused = plan(edited, folder, tmp_path / "research", "research")
    assert refused is not None, f"research built with a pending reason at {place}"
    assert set(refused.codes()) & PENDING_CODES - held_base, (place, refused.codes())
    base_plan, _ = plan(document, folder, tmp_path / "base_exploratory", "exploratory")
    explored, refused = plan(edited, folder, tmp_path / "exploratory", "exploratory")
    assert refused is None, (place, refused.codes() if refused else None)
    count = lambda plans: sum("basis_pending" in item.codes for item in plans.profile.pending)  # noqa: E731
    assert count(explored) == count(base_plan) + 1, place


def test_a_nominal_and_an_acceptance_with_a_pending_reason_behave_alike():
    def n45(reason_accept, reason_nominal) -> Record:
        return Record("n45", packaged_record("openmaterica_example_ndfeb_n45"),
                      accept={"magnetic.remanence": Acceptance(N45_REASONS, reason=reason_accept)},
                      nominal={"magnetic.remanence": Nominal(T(1.345), reason=reason_nominal)})
    from script_a import record_arm

    for accept, nominal, code in (("a vendor's limits", pending(), "material.nominal-pending"),
                                  (pending(), "the middle of the limits", "material.accept-pending")):
        exp = record_arm()
        exp._declarations = [_with_material(d, n45(accept, nominal)) for d in exp._declarations]
        plans = exp.plan()
        assert "material 'n45' magnetic.remanence" in [item.target for item in plans.profile.pending]
        exp._settings = dataclasses.replace(exp._settings, profile="research")
        with pytest.raises(BuildRefused) as refused:
            exp.plan()
        assert code in refused.value.codes()


def _with_material(declaration: Any, record: Record) -> Any:
    thing = getattr(declaration, "thing", None)
    parts = getattr(thing, "parts", None)
    if not parts or not any(part.material is not None and part.material.name == "n45" for part in parts.values()):
        return declaration
    held = {key: dataclasses.replace(part, material=record) if part.material is not None and
            part.material.name == "n45" else part for key, part in parts.items()}
    return dataclasses.replace(declaration, thing=dataclasses.replace(thing, parts=held))


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_reason_where_no_site_judges_it_is_an_input_error_naming_the_slot(profile):
    """A nominal's value and a condition's tolerance carry no reason of their own: written there, it is refused, in
    every profile, by where it was written. The control: the same reason where it belongs builds."""
    from script_a import record_arm

    def on_nominal(value) -> Experiment:
        exp = record_arm()
        record = Record("n45", packaged_record("openmaterica_example_ndfeb_n45"),
                        accept={"magnetic.remanence": Acceptance(N45_REASONS, reason="a vendor's limits")},
                        nominal={"magnetic.remanence": Nominal(value, reason="the middle of the limits")})
        exp._declarations = [_with_material(d, record) for d in exp._declarations]
        exp._settings = dataclasses.replace(exp._settings, profile=profile)
        return exp

    with pytest.raises(BuildRefused) as refused:
        on_nominal(T(1.345, reason=pending())).plan()
    (found,) = [r for r in refused.value.refusals if r.code == "source.no-site"]
    assert re.search(r"nominal\['magnetic\.remanence'\]\.value", found.message), found.message
    assert "source.no-site" not in _codes(on_nominal(T(1.345)))
    tolerance = settling_ball(Paperwork(profile=profile, density="steel", acceptance="Newtonian"))
    tolerance._settings = dataclasses.replace(tolerance._settings, conditions={
        "temperature": Condition(K(293.15), K(0.5, reason="the bath's stated stability"))})
    with pytest.raises(BuildRefused) as refused:
        tolerance.plan()
    (found,) = [r for r in refused.value.refusals if r.code == "source.no-site"]
    assert "plus_minus" in found.message
    stated = settling_ball(Paperwork(profile=profile, density="steel", acceptance="Newtonian"))
    stated._settings = dataclasses.replace(stated._settings, conditions={
        "temperature": degC(20.0, plus_minus=0.5, reason="the bath's thermostat")})
    assert "source.no-site" not in _codes(stated)


def _codes(exp: Experiment) -> list[str]:
    try:
        exp.plan()
        return []
    except BuildRefused as refused:
        return refused.codes()


def test_a_condition_s_reason_is_saved_and_judged_after_a_load(tmp_path: Path):
    exp = settling_ball(Paperwork(density="steel", acceptance="Newtonian"))
    exp._settings = dataclasses.replace(exp._settings, conditions={
        "temperature": degC(20.0, plus_minus=0.5, reason=pending())})
    built = exp.build()
    assert "conditions.temperature" in [item.target for item in built.report.profile.pending]
    built.save(tmp_path / "saved", steps=1)
    assert yaml.safe_load((tmp_path / "saved" / DOCUMENT_FILE).read_text())["conditions"]["temperature"]["reason"] \
        == {"pending": True}
    loaded = Experiment.load(tmp_path / "saved")
    assert "conditions.temperature" in [item.target for item in loaded.build().report.profile.pending]
