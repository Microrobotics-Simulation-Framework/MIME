"""The switches' declarations and the statuses they give: each owner declared once, with orders per output group; no
``piecewise`` remains, its four users carry their statuses and the unconfined body is ``exact``; each declared status
equals the one its switches and the listed raw sites that cap it give (``mime docs check`` runs the same check over
every installed entry).

The controls: ``exact`` with a jump switch fails, and so does ``exact`` with a site listed ``cut``; a status named
``piecewise`` fails at the enum and at registration, and so does an entry with no status; an owner declared twice
fails; a ``piecewise_kink`` entry with a jump switch fails."""

from dataclasses import dataclass

import pytest

from mime.core import shape_names, shape_switches, switch
from mime.docs import load_namespaces, read_list
from mime.docs._check import LISTS, Report, list_path, switch_problems
from mime.gradients import GradientStatus, derived_status, status_of_switches, status_problem
from mime.registry import (
    EntryKind,
    GradientRecord,
    Interaction,
    Registered,
    RoleKind,
    declared_switches,
    lookup,
    register_interaction,
)
from mime.testing import installed_entries
from mime.testing.toys._nodes import ToySpringNode

from checkout import ROOT

KINK = switch.Declaration(owner="planted.kink", indicator="a planted gap, m", orders={"twist": 0, "pose": 1},
                          route="none: planted")
JUMP = switch.Declaration(owner="planted.jump", indicator="a planted gap, m", orders={"twist": -1, "pose": 0},
                          route="none: planted")
SMOOTH = switch.Declaration(owner="planted.smooth", indicator="a planted gap, m", orders={"twist": 1},
                            route="none: planted")


@pytest.fixture(scope="module")
def records() -> list[Registered]:
    load_namespaces()
    found = [lookup(EntryKind(kind), name, version) for kind, name, version in installed_entries()]
    return [record for record in found if record is not None]


def raw_switches():
    return read_list(list_path(ROOT, "raw_switches"), LISTS["raw_switches"])


def test_every_installed_entry_declares_the_status_its_switches_and_listed_sites_give(records):
    report = Report()
    switch_problems(records, raw_switches(), report)
    assert report.problems == []
    assert any(line.startswith("switches: 50 owners declared (mime 50)") for line in report.summary), report.summary


def test_no_piecewise_remains_and_its_four_users_carry_their_statuses(records):
    """And the body unconfined is exact: its quaternion's norm is a domain guard, zero at no valid state, so
    ``rigid_inertial`` declares no switch and no gradient record."""
    assert "piecewise" not in {status.value for status in GradientStatus}
    statuses = {record.name: record.cls.differentiability for record in records
                if record.kind is not EntryKind.WALL_KIND}
    assert {name: statuses[name] for name in ("contact", "drag", "swim_speed", "tube", "rigid_inertial")} == {
        "contact": GradientStatus.PIECEWISE_JUMP, "drag": GradientStatus.PIECEWISE_JUMP,
        "swim_speed": GradientStatus.PIECEWISE_JUMP, "tube": GradientStatus.PIECEWISE_JUMP,
        "rigid_inertial": GradientStatus.EXACT}
    assert next(record for record in records if record.name == "rigid_inertial").cls.gradient is None


def test_each_owner_is_declared_once_with_orders_per_output_group(records):
    """A version kept registered beside its successor (the contact 0.1.0) lists the declarations the successor's
    record holds, the same objects: each owner is declared once all the same."""
    declared = {id(declared): declared for record in records if record.cls.gradient is not None
                for declared in record.cls.gradient.switches}
    declared |= {id(declared): declared for name in shape_names() for declared in shape_switches(name)}
    owners = [held.owner for held in declared.values()]
    assert len(owners) == len(set(owners)) == 50
    tube = next(record for record in records if record.name == "tube" and record.kind is EntryKind.GEOMETRY)
    assert [d.owner for d in declared_switches(tube.cls)] == ["cylinder.axis", "cylinder.axis_norm"]
    for record in records:
        for declared in declared_switches(record.cls):
            assert declared.orders and all(order >= -1 for order in declared.orders.values()), declared


def test_the_status_of_switches_is_their_lowest_order():
    assert status_of_switches(()) is GradientStatus.EXACT
    assert status_of_switches((SMOOTH,)) is GradientStatus.EXACT
    assert status_of_switches((SMOOTH, KINK)) is GradientStatus.PIECEWISE_KINK
    assert status_of_switches((KINK, JUMP)) is GradientStatus.PIECEWISE_JUMP
    assert derived_status((KINK,), GradientStatus.UNVERIFIED) is GradientStatus.UNVERIFIED
    assert derived_status((), GradientStatus.CUT) is GradientStatus.CUT
    assert status_problem(GradientStatus.FIXED_KEY, ()) is None
    assert status_problem(GradientStatus.UNVERIFIED, ()) is None


@pytest.mark.parametrize(("declared", "switches", "cap", "complaint"), [
    (GradientStatus.EXACT, (JUMP,), None, "lowest order is -1 (planted.jump): it is piecewise_jump"),
    (GradientStatus.EXACT, (), GradientStatus.CUT, "caps it at cut: it is cut"),
    (GradientStatus.PIECEWISE_KINK, (KINK, JUMP), None, "it is piecewise_jump"),
    (GradientStatus.PIECEWISE_JUMP, (KINK,), None, "it is piecewise_kink"),
    (GradientStatus.PIECEWISE_JUMP, (), None, "it declares no switch: it is exact"),
    (GradientStatus.PIECEWISE_JUMP, (JUMP,), GradientStatus.UNVERIFIED, "it is unverified"),
])
def test_control_a_declared_status_its_switches_or_listed_sites_do_not_give_fails(declared, switches, cap, complaint):
    problem = status_problem(declared, switches, cap)
    assert problem is not None and complaint in problem, problem


@dataclass(frozen=True, kw_only=True)
class Planted(Interaction):
    name = "planted.switched"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    builds = {"": (ToySpringNode,)}


Planted.__module__ = "mime.planted"


def problems_of(tmp_path, listed: str = "", **attributes) -> list[str]:
    cls = type("Planted", (Planted,), {"__module__": "mime.planted", **attributes})
    path = tmp_path / "raw_switches.txt"
    path.write_text(listed, encoding="utf-8")
    report = Report()
    switch_problems([Registered(EntryKind.INTERACTION, cls.name, cls.version, cls)],
                    read_list(path, LISTS["raw_switches"]), report)
    return [problem for problem in report.problems if "planted" in problem]


def test_control_exact_with_a_jump_switch_or_a_site_listed_cut_fails_through_the_check(tmp_path):
    (jumping,) = problems_of(tmp_path, gradient=GradientRecord(switches=(JUMP,)))
    assert jumping.startswith("interaction planted.switched 0.1.0: it is declared exact, and its switches' lowest")
    listed = "core/_geometry.py Planted.step stop_gradient 1 caps interaction planted.switched 0.1.0 cut  # planted\n"
    (capped,) = problems_of(tmp_path, listed)
    assert "a listed raw site caps it at cut: it is cut" in capped
    assert problems_of(tmp_path, listed, differentiability=GradientStatus.CUT) == []
    assert problems_of(tmp_path, differentiability=GradientStatus.PIECEWISE_JUMP,
                       gradient=GradientRecord(switches=(JUMP,))) == []


def test_control_an_owner_declared_twice_fails(tmp_path):
    again = switch.Declaration(owner="contact.band", indicator="planted", orders={"twist": -1}, route="none: planted")
    problems = problems_of(tmp_path, differentiability=GradientStatus.PIECEWISE_JUMP,
                           gradient=GradientRecord(switches=(again,)))
    load_namespaces()
    contact = lookup(EntryKind.INTERACTION, "contact", "0.1.0")
    assert contact is not None and problems == []
    report = Report()
    planted = type("Planted", (Planted,), {"__module__": "mime.planted",
                                           "differentiability": GradientStatus.PIECEWISE_JUMP,
                                           "gradient": GradientRecord(switches=(again,))})
    switch_problems([contact, Registered(EntryKind.INTERACTION, planted.name, planted.version, planted)],
                    read_list(list_path(ROOT, "raw_switches"), LISTS["raw_switches"]), report)
    assert any("the switch 'contact.band' is declared again, first by interaction contact 0.1.0" in p
               for p in report.problems), report.problems


def test_control_a_status_named_piecewise_or_none_fails():
    with pytest.raises(ValueError, match="'piecewise' is not a valid GradientStatus"):
        GradientStatus("piecewise")
    with pytest.raises(AttributeError):
        _ = GradientStatus.PIECEWISE  # type: ignore[attr-defined]
    for status in ("piecewise", None):
        planted = type("Unstatused", (Planted,), {"__module__": "mime.planted", "name": "planted.unstatused",
                                                  "differentiability": status})
        planted = dataclass(frozen=True, kw_only=True)(planted)
        with pytest.raises(TypeError, match="differentiability: declare a mime.gradients.GradientStatus"):
            register_interaction(planted)
