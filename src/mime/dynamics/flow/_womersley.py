"""The prescribed flow of a medium (F0's item 9, 7 (c)): ``Medium("blood", fills=tube.lumen, material=...,
fields={"flow": Womersley()})``, the cheap rung of #216, a CPU path with no grid.

Poiseuille's steady term and Womersley's harmonic in a straight circular tube, responding to the waveform of the body
force bound to the medium (``BodyForce``, the drive of either rung: swapping ``Womersley()`` and ``FVM(...)`` touches
neither the drive's line nor the medium's ``fills=``). It takes nothing back from what reads it, so it is outside (F0-1
M3). It provides the flow's velocity at points (``field.flow.velocity``). **The name is the maintainer's to confirm**:
it is written here and in :data:`PRESCRIBED_FLOW` alone."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from mime.fields._solved import NOT_DECLARED, SolvedField
from mime.gradients import GradientStatus
from mime.nodes.flow._womersley import PrescribedFlowNode
from mime.registry import Dynamics, Port, Problem, Region, register_dynamics
from mime.scene import PeriodicBox
from mime.things import ThingKind
from mime.units import Magnitude
from mime.vocabulary import FLOW_VELOCITY, FLOW_WAVEFORM

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration
    from mime.registry import BuildContext

#: The kind's registered name: the one place it is written.
PRESCRIBED_FLOW = "womersley"


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class Womersley(Dynamics):
    """A medium's flow prescribed in a straight circular tube: Poiseuille's steady term and Womersley's harmonic, the
    response to the body force's waveform (``mime.nodes.flow._womersley`` states the solution and its series)."""

    name = PRESCRIBED_FLOW
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ("Womersley1955",)
    builds = {"": (PrescribedFlowNode,)}
    applies_to = frozenset({ThingKind.MEDIUM})
    accepts = {FLOW_WAVEFORM.name: Port("waveform")}
    values = {"viscosity": "viscosity", "density": "density"}
    unmetered = ("outside (it takes nothing back from what reads it) and it stores nothing: the energy ledger has no "
                 "receiver here")
    setting_roles = {}

    #: The solved-field contract's declarations (SF-1 to SF-11), which SF-12's suite holds it to.
    field: ClassVar[SolvedField] = SolvedField(
        hosting="prescribed", role="flow", provides=(FLOW_VELOCITY.name,), time_level=NOT_DECLARED,
        own_contribution="none", accepts=(FLOW_WAVEFORM.name,), exchange="none", outside=True,
        domain_kinds=("tube_lumen", PeriodicBox.group), ports="none", properties=("viscosity", "density"),
        settings=("domain",), ledger_route=None,
        ledger_reason="outside, it takes nothing back from what reads it, and it stores nothing (R4-C, C-8)",
        steady_solve=False, affine_response=False, stores="nothing", batches=True, dtypes=("float32", "float64"),
        rates=NOT_DECLARED)

    domain: PeriodicBox | None = None

    @override
    def value_problems(self, values: Mapping[str, Magnitude]) -> list[Problem]:
        return []

    @override
    def region_problems(self, region: Region | None) -> list[Problem]:
        """A straight circular tube's lumen, along any axis."""
        if region is None or region.shape != "cylinder":
            fills = f"fills {region.path}, a {region.shape} region" if region is not None else "fills no region"
            return [Problem("flow.prescribed-region",
                            f"the flow's medium {fills}; the prescribed flow solves straight circular tubes",
                            "fill a Tube's lumen, Medium(fills=tube.lumen), or use FVM() for another region")]
        return []

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        region = ctx.region("medium")
        assert region is not None
        axis = [float(c) for c in region.parameters["axis"]]  # type: ignore[attr-defined]
        length = sum(c * c for c in axis) ** 0.5
        centre = [float(c) for c in region.parameters["centre"]]  # type: ignore[attr-defined]
        b.add_node(PrescribedFlowNode(
            ctx.node_name(), ctx.timestep, viscosity=ctx.scalar_value("viscosity"),
            density=ctx.scalar_value("density"), tube_centre=centre, tube_axis=[c / length for c in axis],
            tube_radius=float(region.parameters["radius"])))  # type: ignore[arg-type]


__all__ = ["PRESCRIBED_FLOW", "Womersley"]
