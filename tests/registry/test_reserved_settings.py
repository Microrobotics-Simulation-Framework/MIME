"""An entry's setting may not be named like an attribute of its class (the coordinator's G12, found in AP3b): a setting
named ``gradient`` is a dataclass field, which replaces the entry's ``gradient`` record, and the entry's declared
switches went missing (only the switch harness caught it). Registration refuses it, naming the setting and the
attribute it would replace.

The reserved names are read from the entry's base classes (every public attribute a base declares that is not one of
that base's own settings), never listed: so ``rate``, a dynamics' own setting, stays a setting, with the rule of its own
(``run.rate-units``), and a name a plugin's own base class declares is reserved too. The control is MIME's whole
registry, and the fixture lab's, which no entry of is newly refused."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from dataclasses import dataclass

import pytest

from mime.docs._export import load_namespaces
from mime.registry import Dynamics, register_dynamics, register_interaction, registered
from mime.testing.toys import ToyDrag
from mime.vocabulary import register_namespace

register_namespace("reserved", "0.0.1")


def reserved_setting_problems(cls: type) -> list[str]:
    """The registry's own reading of ``cls`` (imported where used, so the tests that register run against a registry that
    has not got it)."""
    from mime.registry._registry import reserved_setting_problems as read

    return read(cls)


def planted(base: type, setting: str, *, name: str = "reserved.planted", value: object = 1.0) -> type:
    """``base`` with one more setting, ``setting``, as a plugin author would write it."""
    return dataclass(frozen=True, kw_only=True)(type("Planted", (base,), {
        "name": name, "__module__": "mime.planted", "__annotations__": {setting: type(value)}, setting: value}))


@pytest.mark.parametrize(("setting", "owner"), [("gradient", "Entry"), ("references", "Entry"), ("validity", "Entry"),
                                                ("builds", "ToyDrag"), ("rungs", "ToyDrag")])
def test_a_setting_named_like_an_attribute_of_its_class_is_refused_naming_both(setting, owner):
    value = ("quadratic",) if setting == "rungs" else 1.0
    cls = planted(ToyDrag, setting, value=value)
    with pytest.raises(TypeError, match="cannot register") as refused:
        register_interaction(cls)
    assert f"setting {setting!r} is named like {owner}.{setting}" in str(refused.value)
    assert reserved_setting_problems(cls) == [
        f"setting {setting!r} is named like {owner}.{setting}, an attribute of the entry's class that the setting would "
        f"replace: rename the setting"]


def test_the_same_rule_holds_for_every_kind_and_a_dynamics_own_setting_is_not_reserved():
    @dataclass(frozen=True, kw_only=True)
    class Inertia(Dynamics):
        name = "reserved.inertia"
        version = "0.0.1"
        gradient: float = 1.0

    assert reserved_setting_problems(Inertia) and "Entry.gradient" in reserved_setting_problems(Inertia)[0]
    assert "rate" in {field.name for field in dataclasses.fields(Dynamics)}
    assert reserved_setting_problems(planted(Dynamics, "stiffness")) == [], "a new name is a setting"
    assert not any("'rate'" in problem for problem in reserved_setting_problems(Dynamics))


def test_control_a_name_that_is_no_attribute_registers_and_rate_keeps_its_own_rule():
    assert reserved_setting_problems(planted(ToyDrag, "stiffness")) == []
    assert "rate" in {field.name for field in dataclasses.fields(Dynamics)}
    assert not any("'rate'" in problem for problem in reserved_setting_problems(planted(Dynamics, "stiffness")))


def test_the_registry_walk_no_mime_entry_and_no_plugin_entry_is_refused():
    """Every entry MIME and the installed lab fixture register (read in this process once every namespace is loaded) has no
    setting named like an attribute of its class: the rule refuses nothing that registered before it."""
    load_namespaces()
    found = [(record.name, record.version, reserved_setting_problems(record.cls)) for record in registered()]
    assert [(name, version, problems) for name, version, problems in found if problems] == []
    assert len(found) >= 70 and any(name.startswith("labx.") for name, _, _ in found), \
        "the walk reached MIME's entries and the lab's"
    assert {each.name for each in importlib.metadata.entry_points(group="mime.entries")} >= {"labx"}
