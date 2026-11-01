"""One dispatcher for every action at every site (design #190 item 3): each class of finding a build gives, under each
action a profile rule can name (``generate``, ``basis``, ``list``, ``refuse``) and under none, does what the rule says,
the same at every site: a read's unknown reason, a pending reason, a numerical default taken, a physical value with no
source, a numerical value with no evidence, a default its entry declares nothing of. A row no site can apply (a
``generate`` on what no waiver covers, a ``list`` of what only a declaration covers, a class with no code) refuses the
build by name, ``profile.tables``. The review's five rows (a1's strict rows and other actions on dev1's classes) are
among the cases, so the seam holds for the rows a1 will add, not for one combination."""

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from profile_cases import Paperwork, settling_ball

from mime.build import BuildRefused
from mime.observations import SwimSpeed
from mime.registry import EntryKind, register_observation, registered
from mime.registry._registry import _by_class, _by_key  # noqa: PLC2701 - the case unregisters what it planted
from mime.scene import Z
from mime.units import mm, s
from mime.validity import (
    FINDING_CLASSES,
    PROFILE_RULES,
    Action,
    Cover,
    Profile,
    ProfileRules,
    Rule,
    rule_problems,
    using_tables,
)
from mime.vocabulary import register_namespace


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabSpeed(SwimSpeed):
    """A lab's observation with a setting whose written default declares no role (design #191 item 1)."""

    name = "labdispatch.speed"
    version = "0.0.1"
    smoothing: float = 3.0


@contextmanager
def lab_speed() -> Iterator[None]:
    """``LabSpeed`` registered while the context is open, and unregistered after, so no other test's registry walk
    meets it."""
    register_namespace("labdispatch", "0.0.1")
    register_observation(LabSpeed)
    try:
        yield
    finally:
        record = next((r for r in registered(EntryKind.OBSERVATION) if r.cls is LabSpeed), None)
        if record is not None:
            del _by_key[(record.kind, record.name, record.version)]
            del _by_class[LabSpeed]


STEEL = Paperwork(density="steel", acceptance="Newtonian")


def labbed():
    experiment = settling_ball(STEEL)
    experiment.observe(LabSpeed("ball", along=-Z, after=s(0.004)))
    return experiment


#: Each class, the experiment that gives one finding of it, the finding's target, the refusal a stopping action gives,
#: and what the site does with no rule (its behaviour before profiles): refuse, or nothing.
SITES = {
    "assumption": (lambda: settling_ball(Paperwork(density="steel")), "material 'water' fluid.dynamic_viscosity",
                   "record.unknown", Action.REFUSE),
    "pending": (lambda: settling_ball(Paperwork(acceptance="Newtonian")), "ball.parts.shell.density",
                "value.reason-pending", Action.REFUSE),
    "numerical_default": (lambda: settling_ball(STEEL), "drag:ball.spacing", "setting.numerical-default",
                          Action.REFUSE),
    "unsourced": (lambda: settling_ball(STEEL), "ball.parts.shell.radius", "setting.source-required", None),
    "unevidenced": (lambda: settling_ball(STEEL), "dt", "setting.evidence-required", None),
    "undeclared_default": (labbed, "labdispatch.speed:ball.smoothing", "setting.undeclared-default", None),
}
ACTIONS = [*Action, None]


def rules_with(profile: Profile, name: str, action: Action | None) -> ProfileRules:
    held = {key: rule for key, rule in PROFILE_RULES.rules.items() if key != (profile, name)}
    if action is not None:
        held[(profile, name)] = Rule(action)
    return ProfileRules(held)


def outcome(make, profile: Profile) -> tuple[str, object]:
    experiment = make()
    experiment._settings = dataclasses.replace(experiment._settings, profile=profile.value)
    try:
        return "built", experiment.plan().profile
    except BuildRefused as refused:
        return "refused", refused


@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: a.value if a is not None else "nothing")
@pytest.mark.parametrize("name", list(SITES))
def test_every_action_on_every_class_does_what_the_rule_says(name, action):
    make, target, code, baseline = SITES[name]
    rules = rules_with(Profile.EXPLORATORY, name, action)
    with lab_speed(), using_tables(FINDING_CLASSES, rules):
        kind, held = outcome(make, Profile.EXPLORATORY)
    if rule_problems(FINDING_CLASSES, rules):
        assert kind == "refused" and "profile.tables" in held.codes(), (name, action)
        return
    effective = action if action is not None else baseline
    if effective in (Action.BASIS, Action.REFUSE):
        assert kind == "refused", (name, action)
        assert code in held.codes(), (name, action, held.codes())
        return
    assert kind == "built", (name, action, held.codes() if kind == "refused" else None)
    pending = [item.target for item in held.pending]
    listed = [item.target for item in held.listed]
    if effective is Action.GENERATE:
        assert target in pending and target not in listed, (name, pending)
        assert any(target in line for line in held.stamp().pending)
    elif effective is Action.LIST:
        assert target in listed and target not in pending, (name, listed)
        assert not any(target in line for line in held.stamp().pending)
    else:
        assert target not in pending and target not in listed, (name, pending, listed)


@pytest.mark.parametrize("row", [
    (Profile.RESEARCH, "unsourced", Action.REFUSE, "setting.source-required"),
    (Profile.RESEARCH, "unevidenced", Action.REFUSE, "setting.evidence-required"),
    (Profile.EXPLORATORY, "numerical_default", Action.LIST, None),
    (Profile.EXPLORATORY, "numerical_default", Action.BASIS, "setting.numerical-default"),
    (Profile.RESEARCH, "assumption", Action.LIST, None),
], ids=lambda row: f"{row[0].value}-{row[1]}-{row[2].value}")
def test_the_review_s_rows_each_act(row):
    """The rows the review found ignored or inverted: research refusing an unsourced or unevidenced value (strict's
    rows), exploratory listing or asking a basis for a numerical default, research listing an assumption."""
    profile, name, action, code = row
    sealed = Paperwork(profile=profile.value, acceptance=None if name == "assumption" else "Newtonian",
                       density="steel", spacing=None if name == "numerical_default" else mm(0.2, reason="converged"),
                       blob=None if name == "numerical_default" else 0.5,
                       quadrature_refinement=None if name == "numerical_default" else 4,
                       sources={} if name == "numerical_default" else {"drag:ball.blob": "usual",
                                                                       "drag:ball.quadrature_refinement": "usual"})
    with using_tables(FINDING_CLASSES, rules_with(profile, name, action)):
        kind, held = outcome(lambda: settling_ball(sealed), profile)
    if code is not None:
        assert kind == "refused" and code in held.codes(), (row, held)
        return
    assert kind == "built", (row, held.codes() if kind == "refused" else None)
    target = "drag:ball.spacing" if name == "numerical_default" else "material 'water' fluid.dynamic_viscosity"
    assert target in [item.target for item in held.listed] and target not in [i.target for i in held.pending]


def test_a_row_no_site_can_apply_is_refused_by_name():
    """K4: tables that class an assumption as covered by a model change, then generate a waiver for it, are refused,
    where before the build waived water's shear rate anyway; a list of what only a declaration covers, and a rule for a
    class with no code, likewise."""
    classes = FINDING_CLASSES.with_class("assumption", Cover.MODEL_CHANGE, ["condition_unstated:*"])
    assert rule_problems(classes, PROFILE_RULES)
    with using_tables(classes, PROFILE_RULES), pytest.raises(BuildRefused) as refused:
        settling_ball(Paperwork(density="steel")).plan()
    assert "profile.tables" in refused.value.codes()
    listing = FINDING_CLASSES.with_class("not_covered", Cover.MODEL_CHANGE, ["outside_range"])
    found = rule_problems(listing, PROFILE_RULES.with_rule(Profile.RESEARCH, "not_covered", Rule(Action.LIST)))
    assert found and "no profile relaxes it" in found[0]
    empty = FINDING_CLASSES.with_class("nothing_of", Cover.WAIVER, [])
    found = rule_problems(empty, PROFILE_RULES.with_rule(Profile.RESEARCH, "nothing_of", Rule(Action.REFUSE)))
    assert found and "holds no code" in found[0]
    assert rule_problems(FINDING_CLASSES, PROFILE_RULES) == []
