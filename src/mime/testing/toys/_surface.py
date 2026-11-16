"""A toy whose surface deforms by a known field: a square sheet indented at its centre by a Gaussian dimple whose depth
grows at a constant rate, the depth at each vertex and the rate it deepens there the fields it carries. It exercises a
surface's declaration, its stage, its recording and its view (:class:`mime.registry.SurfaceMesh`); it has no
physics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, override

from mime.build import DynamicsBuilder, ResolvedRegistration
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Dynamics, MeshField, Port, SurfaceMesh, register_dynamics
from mime.testing.toys._entries import TOY_UNMETERED, VERSION
from mime.testing.toys._nodes import ToyIndentedSheetNode, sheet
from mime.things import ThingKind
from mime.units import LENGTH, Length, Magnitude, Quantity, Velocity, m, m_per_s


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ToyIndentedSheet(Dynamics):
    """A sheet of ``count`` x ``count`` vertices over a square of ``side``, indented at its centre, the dimple's depth
    growing at ``speed``; it declares its surface with two fields, the depth at each vertex (m) and the rate it deepens
    there (m/s), neither with a declared range (the recording's). ``still`` declares the sheet standing at rest with
    its fields changing (a wall's values on a wall that does not move); ``display`` declares it shown as a section
    would be (``mime:display``), for the stage's readers; ``peak`` reduces the depth by the maximum where the sheet is
    drawn decimated, as a damage-like field declares."""

    name = "toy.indented_sheet"
    unmetered = TOY_UNMETERED
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyIndentedSheetNode,)}
    applies_to = frozenset({ThingKind.BODY})
    initial_state = {"position": LENGTH}
    side: Annotated[Quantity, Length] = m(1e-3)
    count: int = 33
    speed: Annotated[Quantity, Velocity] = m_per_s(1e-3)
    width: Annotated[Quantity, Length] = m(1.5e-4)
    still: bool = False
    display: str | None = None
    peak: bool = False

    @override
    def surface_meshes(self, settings: Mapping[str, Magnitude]) -> tuple[SurfaceMesh, ...]:
        rest, faces = sheet(float(settings["side"]), int(self.count))  # pyright: ignore[reportArgumentType]
        fields = (MeshField("depth", "m", Port("depth"), reduction="max" if self.peak else "mean"),
                  MeshField("rate", "m/s", Port("rate")))
        return (SurfaceMesh("sheet", rest, faces, None if self.still else Port("points"), fields, self.display),)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        b.add_node(ToyIndentedSheetNode(ctx.node_name(), ctx.timestep, side=ctx.scalar("side"), count=self.count,
                                        speed=ctx.scalar("speed"), width=ctx.scalar("width")))


__all__ = ["ToyIndentedSheet"]
