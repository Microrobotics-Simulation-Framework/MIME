"""Every drag rung assumes its medium at rest (it delivers no load from a flow), so until a rung reads a provided flow
each is refused, coded ``drag.medium-flows`` and naming the field, wherever the region its medium fills carries a
field: every rung of every drag version the registry holds, under the prescribed flow (Womersley), and the mrs rung
under the FVM too. The refusal fails closed: it holds for a flow kind that declares it provides a quantity other than
the velocity, and for a flow another medium filling the same lumen carries. Each case's control is the same experiment
with the region at rest, which plans without a refusal.

A medium's near field (drag 0.5.0's mrs rung, two bodies or more in one medium, one joint node) is refused by each
member's own plan, made before the planner joins them: both members are refused, and at rest the same experiment plans
its joint node. Its controls plant the refusal's loss on the joint path, once in 0.5.0's own plan and once in the
planner's join (where the one-body walk still refuses, so only the joint case sees it): each plans the flowing
experiment clean."""

from __future__ import annotations

import dataclasses
import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest import mock

import pytest
from mrs_cases import mrs_tube
from rft_cases import R6_TUBE, SMALL_HELIX, SMALL_SETTINGS, helix_body
from script_a import EPSILON, script_a

from mime.build import BuildRefused
from mime.build._declarations import ValueDeclaration
from mime.build._plan import _Planner
from mime.data import asset
from mime.dynamics import FVM, RigidInertial, Womersley
from mime.experiment import Experiment
from mime.interactions import Lubrication
from mime.registry import EntryKind, register_dynamics, registered
from mime.scene import PeriodicBox, Sphere
from mime.things import Body, Medium, Part, State
from mime.units import Pa_s, kg, kg_m2, kg_per_m3, m_per_s, mm, rad_per_s
from mime.vocabulary import Interface, ReadKind, define, register_namespace

CODE = "drag.medium-flows"
NEAR_FIELD = "near_field:water"
register_namespace("restplant", "0.0.1")
CONCENTRATION = define(Interface("restplant.concentration", ReadKind.FIELD, "1", "world", (), magnitude=1.0,
                                 description="a planted quantity a flow kind declares in place of the velocity"))


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class Elsewise(Womersley):
    """Womersley's flow declaring a concentration as what it provides, not the velocity: the refusal must not rest on
    what a kind declares."""

    name = "restplant.elsewise"
    version = "0.0.1"
    field = dataclasses.replace(Womersley.field, provides=(CONCENTRATION.name,))


def mrs(cls: Any) -> Experiment:
    experiment = mrs_tube()
    experiment.remove("drag:ball")
    experiment.attach(cls.MRS(spacing=mm(0.3), blob=0.5, wall_kind="tube"), body="ball", medium="water")
    return experiment


#: A case for each rung, made with the drag version's own class.
CASES: dict[str, Callable[[Any], Experiment]] = {
    "surrogate": lambda cls: script_a(velocity=m_per_s((1e-3, 0.0, 0.0)), drag=cls.Surrogate(
        weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=EPSILON))),
    "mrs": mrs,
    "rft": lambda cls: helix_body("rft_tube", SMALL_HELIX, SMALL_SETTINGS, cls.RFT(wall_kind="tube"), tube=R6_TUBE,
                                  velocity=m_per_s((1e-3, 0.0, 0.0)), spin=rad_per_s((3.0, -1.0, 2.0))),
}
DRAGS = [pytest.param(record.cls, rung, id=f"{rung}-{record.version}")
         for record in registered(EntryKind.INTERACTION) if record.name == "drag"
         for rung in record.cls.rungs]  # type: ignore[attr-defined]
#: The drag versions whose mrs rung joins a medium's near field.
JOINING = [pytest.param(record.cls, id=record.version) for record in registered(EntryKind.INTERACTION)
           if record.name == "drag" and hasattr(record.cls, "joins_near_field")]


def joint(cls: Any) -> Experiment:
    """``mrs``'s ball and a bead 3 mm along the tube's axis from it, both on the version's mrs rung in one medium."""
    experiment = mrs(cls)
    experiment.add_body(Body("bead", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(0.5)))},
                             initial=State(position=mm((0.0, 0.0, 3.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("bead.mass", kg(1e-7), reason="a millimetre body's order")
    experiment.declare("bead.inertia", kg_m2((1e-14, 2e-14, 3e-14)), reason="a millimetre body's order")
    experiment.attach(cls.MRS(spacing=mm(0.3), blob=0.5, wall_kind="tube"), body="bead", medium="water")
    return experiment


def flowing(make: Callable[[], Experiment], flow: object) -> Experiment:
    """``make``'s experiment with its medium ``water`` carrying ``flow`` (the case modules' ``Medium`` given the field
    for the call), and the water's density, which a flow reads."""
    modules = [sys.modules[name] for name in ("script_a", "mrs_cases", "rft_cases")]
    with pytest.MonkeyPatch.context() as patch:
        for module in modules:
            patch.setattr(module, "Medium", functools.partial(Medium, fields={"flow": flow}))
        experiment = make()
    _, declarations, _ = experiment.declared_as()
    if not any(isinstance(d, ValueDeclaration) and str(d.path) == "water.density" for d in declarations):
        experiment.declare("water.density", kg_per_m3(1000.0), reason="water")
    return experiment


def refused(experiment: Experiment) -> list[BuildRefused]:
    try:
        experiment.plan()
    except BuildRefused as held:
        return [held]
    return []


def assert_refused_naming(experiment: Experiment, field: str) -> None:
    (held,) = refused(experiment)
    assert held.codes() == [CODE], held.codes()
    assert field in held.refusals[0].message and "at rest" in held.refusals[0].message


def test_the_walk_meets_every_rung_of_every_drag_version():
    assert {p.id for p in DRAGS} >= {"surrogate-0.3.0", "mrs-0.3.0", "surrogate-0.4.0", "mrs-0.4.0", "rft-0.4.0",
                                     "surrogate-0.5.0", "mrs-0.5.0", "rft-0.5.0"}
    assert {p.id for p in JOINING} >= {"0.5.0"}


@pytest.mark.parametrize(("cls", "rung"), DRAGS)
def test_every_drag_rung_is_refused_in_a_medium_whose_region_carries_a_flow(cls, rung):
    make = functools.partial(CASES[rung], cls)
    assert refused(make()) == []  # the control: the medium at rest
    assert_refused_naming(flowing(make, Womersley()), "water.flow")


@pytest.mark.parametrize("flow", [FVM(domain=PeriodicBox(length=mm(1.0))), Elsewise()], ids=["fvm", "elsewise"])
def test_the_mrs_rung_is_refused_under_the_fvm_and_under_a_flow_declaring_another_quantity(flow):
    assert_refused_naming(flowing(mrs_tube, flow), "water.flow")


@pytest.mark.parametrize("carries", [True, False], ids=["flowing", "control-at-rest"])
def test_a_flow_carried_by_a_second_medium_filling_the_same_lumen_refuses_the_drag(carries):
    """The drag's own medium is at rest, and a second medium filling the tube's lumen carries the flow: the drag is
    refused naming that medium's flow. The control is the second medium at rest, which plans."""
    experiment = mrs_tube()
    experiment.add_medium(Medium("blood", fills="tube.lumen", fields={"flow": Womersley()} if carries else {}))
    experiment.declare("blood.viscosity", Pa_s(1e-3), reason="a second medium in the lumen")
    experiment.declare("blood.density", kg_per_m3(1000.0), reason="a second medium in the lumen")
    if carries:
        assert_refused_naming(experiment, "blood.flow")
    else:
        assert refused(experiment) == []


def assert_each_member_refused(experiment: Experiment) -> None:
    (held,) = refused(experiment)
    assert held.codes() == [CODE, CODE], held.codes()
    assert {r.message.partition(" (")[0] for r in held.refusals} == {"interaction drag:ball", "interaction drag:bead"}
    assert all("water.flow" in r.message and "at rest" in r.message for r in held.refusals)


@pytest.mark.parametrize("cls", JOINING)
def test_a_near_fields_members_are_each_refused_in_a_medium_whose_region_carries_a_flow(cls):
    make = functools.partial(joint, cls)
    assert NEAR_FIELD in {plan.owner for plan in make().plan().entries}  # the control: at rest, one joint node
    assert_each_member_refused(flowing(make, Womersley()))


def _dropped_in_the_entry(cls: Any) -> Any:
    """The plant in 0.5.0's own plan: on the mrs rung, the plan without the refusal."""
    original = cls.plan

    def plan(self: Any, ctx: Any) -> Any:
        held = original(self, ctx)
        if self.rung != "mrs":
            return held
        return dataclasses.replace(held, problems=tuple(p for p in held.problems if p.code != CODE))

    return mock.patch.object(cls, "plan", plan)


def _dropped_in_the_join(cls: Any) -> Any:
    """The plant in the planner's join: the refusals of the members it joins dropped."""
    original = _Planner.join_near_fields

    def join_near_fields(self: Any) -> None:
        original(self)
        joined = tuple(f"interaction {plan.owner} (" for plan in self.readers if plan.joined)
        self.refusals[:] = [r for r in self.refusals if not (r.code == CODE and r.message.startswith(joined))]

    return mock.patch.object(_Planner, "join_near_fields", join_near_fields)


@pytest.mark.parametrize("cls", JOINING)
@pytest.mark.parametrize("plant", [_dropped_in_the_entry, _dropped_in_the_join], ids=["entry", "join"])
def test_control_a_joint_path_that_loses_the_refusal_plans_the_flowing_near_field_clean(cls, plant):
    with plant(cls):
        assert refused(flowing(functools.partial(joint, cls), Womersley())) == []
        if plant is _dropped_in_the_join:  # the one-body walk does not see this plant
            assert_refused_naming(flowing(functools.partial(mrs, cls), Womersley()), "water.flow")
