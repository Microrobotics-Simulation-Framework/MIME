"""Every drag rung assumes its medium at rest (it delivers no load from a flow), so until a rung reads a provided flow
each is refused, coded ``drag.medium-flows`` and naming the field, wherever the region its medium fills carries a
field: every rung of every drag version the registry holds, under the prescribed flow (Womersley), and the mrs rung
under the FVM too. The refusal fails closed: it holds for a flow kind that declares it provides a quantity other than
the velocity, and for a flow another medium filling the same lumen carries. Each case's control is the same experiment
with the region at rest, which plans without a refusal."""

from __future__ import annotations

import dataclasses
import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from mrs_cases import mrs_tube
from rft_cases import R6_TUBE, SMALL_HELIX, SMALL_SETTINGS, helix_body
from script_a import EPSILON, script_a

from mime.build import BuildRefused
from mime.build._declarations import ValueDeclaration
from mime.data import asset
from mime.dynamics import FVM, Womersley
from mime.experiment import Experiment
from mime.interactions import Lubrication
from mime.registry import EntryKind, register_dynamics, registered
from mime.scene import PeriodicBox
from mime.things import Medium
from mime.units import Pa_s, kg_per_m3, m_per_s, mm, rad_per_s
from mime.vocabulary import Interface, ReadKind, define, register_namespace

CODE = "drag.medium-flows"
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


def test_the_walk_meets_every_rung_of_both_drag_versions():
    assert {p.id for p in DRAGS} >= {"surrogate-0.3.0", "mrs-0.3.0", "surrogate-0.4.0", "mrs-0.4.0", "rft-0.4.0"}


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
