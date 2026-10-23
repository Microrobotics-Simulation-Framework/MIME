"""Geometry reads are static unless declared ``moving``: an entry takes a thing's geometry (a surface, a direction)
once, at construction, so a reference to the geometry of a thing with a dynamics is refused unless the entry declares
``moving``, and the refusal names the entry, the thing and why. With ``moving`` declared, the same reference builds.
Today's reads, of the fixed tube, are unaffected (script A builds)."""

from dataclasses import dataclass, replace
from typing import Any

import pytest
from script_a import script_a

from mime.build import BuildRefused
from mime.gradients import GradientStatus
from mime.interactions import Contact
from mime.observations import SwimSpeed
from mime.registry import Dynamics, register_dynamics, register_observation
from mime.scene import Tube, X
from mime.things import Structure, ThingKind
from mime.units import mm
from mime.vocabulary import register_namespace

register_namespace("moving", "0.0.1")


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class Swaying(Dynamics):
    """A structure that moves: it builds nothing, and is here only to give the tube a dynamics."""

    name = "moving.swaying"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    applies_to = frozenset({ThingKind.STRUCTURE})
    nodes = ()

    def build(self, ctx: Any, b: Any, registrations: Any) -> None:
        return None


@register_observation
@dataclass(frozen=True, kw_only=True)
class FollowingSwimSpeed(SwimSpeed):
    """The swim speed, declaring that it follows moving geometry."""

    name = "moving.following_swim_speed"
    moving = True


def swaying(experiment):
    experiment._declarations[0] = replace(experiment._declarations[0], thing=Structure(
        "tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6), length=mm(100)), dynamics=Swaying()))
    return experiment


def readers_of(experiment) -> dict[str, str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    print(f"\n{refused.value}")
    return {r.message.split(": ")[0]: r.message for r in refused.value.refusals if r.code == "geometry.moving-read"}


def only(experiment, kind: type) -> Any:
    experiment._declarations = [d for d in experiment._declarations
                                if not isinstance(getattr(d, "law", getattr(d, "observation", None)), kind)]
    return experiment


def test_todays_reads_of_the_fixed_tube_are_unaffected():
    assert script_a().build().report.refusals == ()


def test_a_read_of_moving_geometry_is_refused_naming_the_reader_the_thing_and_why():
    found = readers_of(swaying(script_a()))
    assert set(found) == {"field 'magnetic' of apparatus 'field' (uniform_rotating_field 0.1.0)",
                          "interaction contact:umr", "observation swim_speed:umr (swim_speed 0.1.0)"}
    contact = found["interaction contact:umr"]
    assert "contact 0.1.0 reads the geometry of structure 'tube', which moves (it has the dynamics moving.swaying " \
           "0.0.1); contact 0.1.0 takes geometry once, at construction, and would keep it as it was at t = 0" in contact


def test_with_moving_declared_the_same_read_builds():
    experiment = only(only(swaying(script_a()), Contact), SwimSpeed)
    field = next(i for i, d in enumerate(experiment._declarations) if getattr(getattr(d, "thing", None), "name", "")
                 == "field")
    held = experiment._declarations[field].thing
    experiment._declarations[field] = replace(experiment._declarations[field], thing=type(held)(
        "field", fields={"magnetic": replace(held.fields["magnetic"], rotation_axis=(1.0, 0.0, 0.0))},
        commands=held.commands))
    experiment.observe(SwimSpeed("umr", along="tube.axis"))
    assert set(readers_of(experiment)) == {"observation swim_speed:umr (swim_speed 0.1.0)"}
    experiment = only(experiment, SwimSpeed)
    experiment.observe(FollowingSwimSpeed("umr", along="tube.axis"))
    built = experiment.build()
    assert built.report.refusals == () and "moving.following_swim_speed:umr" in [n.node for n in built.report.nodes]
