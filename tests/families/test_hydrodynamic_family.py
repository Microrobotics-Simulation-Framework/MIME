"""The hydrodynamic family's members and controls. Both members conform: the drag's surrogate rung (in the family
walk, ``test_family_walk.py``) and the test-only Stokes sphere, each swapped for the other. And a law planted with one
defect fails the case that exists to catch it:

- the surrogate with its resistance's sign flipped fails the sign case, and only it among the sampled cases (with
  its own oracle, the reference flipped);
- the surrogate with the 0.2 line's frame (the tube along +z whatever the geometry says) fails the frame case on the
  rig's tube along x;
- a rung that ignores the medium's viscosity fails material reach.

The planted laws are registered here, under the ``oracle`` namespace, so the registry walk (a fresh process) does not
hold them to round-trip or node cases.
"""

from dataclasses import dataclass
from typing import override

from family_cases import a_drag, surrogate
from hydro_oracles import PortFrameDrag, StokesSphere, drag_node, sphere_oracle, wire

from mime.build import InteractionBuilder
from mime.interactions import Drag
from mime.nodes.hydro import SurrogateDragNode
from mime.registry import BuildContext, register_interaction
from mime.testing import CaseKind, NodeCase, assert_conforms, conformance, unit_quaternion


def failed(results) -> set[CaseKind]:
    return {case for case, result in results.items() if not result.passed}


def report(results) -> None:
    print("\n" + "\n".join(f"{case.value}: {result}" for case, result in results.items()))


class _FlippedNode(SurrogateDragNode):
    @override
    def advance(self, p, state, boundary_inputs, dt):
        held = super().advance(p, state, boundary_inputs, dt)
        return {**held, "resistance": -held["resistance"], "background_load": -held["background_load"]}


class _InviscidNode(SurrogateDragNode):
    """Water's viscosity, whatever the medium's."""

    @override
    def advance(self, p, state, boundary_inputs, dt):
        return super().advance({**p, "viscosity": 1e-3}, state, boundary_inputs, dt)


class _SquaredNode(SurrogateDragNode):
    """A resistance that grows as the viscosity squared."""

    @override
    def advance(self, p, state, boundary_inputs, dt):
        held = super().advance(p, state, boundary_inputs, dt)
        return {**held, "resistance": held["resistance"] * p["viscosity"] / 1e-3}


def _as(drag: Drag, cls: type[Drag]) -> Drag:
    return cls(rung=drag.rung, weights=drag.weights, lubrication=drag.lubrication)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class FlippedDrag(Drag):
    """The planted sign error: a resistance that drives the body rather than resisting it."""

    name = "oracle.flipped_drag"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        wire(b, drag_node(_FlippedNode, ctx, self))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class InviscidDrag(Drag):
    name = "oracle.inviscid_drag"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        wire(b, drag_node(_InviscidNode, ctx, self))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class SquaredDrag(Drag):
    name = "oracle.squared_drag"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        wire(b, drag_node(_SquaredNode, ctx, self))


def flipped(read, settings):
    held = surrogate(read, settings)
    return {label: -value for label, value in held.items()}


def test_the_stokes_sphere_conforms_and_swaps_with_the_surrogate():
    assert_conforms(StokesSphere(), "hydrodynamic", oracle=sphere_oracle, swap_with=a_drag(),
                    node_case=NodeCase(options={"boundary_bounds": {"pose": (-4e-3, 4e-3)}, "constrain_boundary":
                                                lambda given: {**given, "pose": unit_quaternion(given["pose"])}}))


def test_a_flipped_resistance_fails_the_sign_case_and_only_it():
    results = conformance(_as(a_drag(), FlippedDrag), "hydrodynamic", oracle=flipped,
                          cases=[CaseKind.SIGN, CaseKind.FRAME, CaseKind.ORACLE, CaseKind.GRADIENTS])
    report(results)
    assert failed(results) == {CaseKind.SIGN}


def test_the_02_lines_frame_fails_the_frame_case_in_a_tube_along_x():
    (result,) = conformance(_as(a_drag(), PortFrameDrag), "hydrodynamic", cases=[CaseKind.FRAME]).values()
    print(f"\n{result}")
    assert result.failed


def test_a_rung_that_ignores_the_viscosity_fails_material_reach():
    (result,) = conformance(_as(a_drag(), InviscidDrag), "hydrodynamic", cases=[CaseKind.MATERIAL_REACH]).values()
    print(f"\n{result}")
    assert result.failed and "leaves the law's response unchanged" in result.detail


def test_a_rung_whose_resistance_is_not_proportional_to_the_viscosity_fails_material_reach():
    """The control of the family's proportionality: a resistance growing as the viscosity squared reaches the output,
    and is refused for not scaling with it."""
    (result,) = conformance(_as(a_drag(), SquaredDrag), "hydrodynamic", cases=[CaseKind.MATERIAL_REACH]).values()
    print(f"\n{result}")
    assert result.failed and "by the same factor" in result.detail
