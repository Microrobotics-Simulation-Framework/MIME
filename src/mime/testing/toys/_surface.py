"""A toy whose surface deforms by a known field: a square sheet indented at its centre by a Gaussian dimple whose depth
grows at a constant rate, its depth at each vertex the field it carries. It exercises a surface mesh's declaration, its
stage, its recording and its view (:class:`mime.registry.SurfaceMesh`); it has no physics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy
import numpy as np
from jax import Array

from mime.build import DynamicsBuilder, ResolvedRegistration
from mime.core import NodeFields
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Dynamics, MeshField, Port, SurfaceMesh, register_dynamics
from mime.testing.toys._entries import TOY_UNMETERED, VERSION
from mime.testing.toys._nodes import ToyNode, node_params
from mime.things import ThingKind
from mime.units import LENGTH, Length, Magnitude, Quantity, Velocity, m, m_per_s

jnp: Any = jax.numpy


def sheet(side: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The sheet at rest: ``count`` x ``count`` vertices over a square of ``side`` centred on the origin in z = 0, and
    its triangles, two a cell, counter-clockwise seen from +z."""
    along = np.linspace(-side / 2.0, side / 2.0, count)
    x, y = np.meshgrid(along, along, indexing="ij")
    rest = np.stack([x.reshape(-1), y.reshape(-1), np.zeros(count * count)], axis=1)
    index = np.arange(count * count).reshape(count, count)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], axis=-1).reshape(-1, 3), np.stack([a, c, d], axis=-1).reshape(-1, 3)])
    return rest, faces.astype(np.int64)


def dimple(rest: Any, depth: Any, width: Any) -> Any:
    """The indentation's depth at each rest point: ``depth`` at the centre, a Gaussian of standard deviation ``width``."""
    r2 = rest[:, 0] ** 2 + rest[:, 1] ** 2
    return depth * jnp.exp(-r2 / (2.0 * width * width))


class ToyIndentedSheetNode(ToyNode):
    """The sheet's time, points, depth and slope fields: each step the dimple's depth at the centre is ``speed * t``."""

    def __init__(self, name: str, timestep: float, side: float = 1e-3, count: int = 33, speed: float = 1e-3,
                 width: float = 1e-4) -> None:
        super().__init__(name, timestep, side=float(side), count=int(count), speed=float(speed), width=float(width))

    def _at(self, t: Any, p: Mapping[str, Any]) -> dict[str, Any]:
        unit, _ = sheet(1.0, int(node_params(self)["count"]))
        held = jnp.asarray(unit, dtype=jnp.asarray(t).dtype) * p["side"]
        depth = dimple(held, p["speed"] * t, p["width"])
        radius = jnp.sqrt(held[:, 0] ** 2 + held[:, 1] ** 2)
        slope = depth * radius / (p["width"] * p["width"])  # the dimple's steepness, |grad z|
        return {"t": t, "points": held.at[:, 2].add(-depth), "depth": depth, "slope": slope}

    @override
    def initial_state(self) -> NodeFields:
        return self._at(jnp.zeros(()), node_params(self))

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        del boundary_inputs
        return self._at(state["t"] + dt, node_params(self, params))


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ToyIndentedSheet(Dynamics):
    """A sheet indented at its centre, its depth growing at ``speed``; it declares its surface, the depth at each vertex
    (m) and the dimple's slope there (1), neither with a declared range (the recording's)."""

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
    #: The surface stands at rest and carries the fields alone (a wall's values on a wall that does not move).
    still: bool = False

    @override
    def surface_meshes(self, settings: Mapping[str, Magnitude]) -> tuple[SurfaceMesh, ...]:
        rest, faces = sheet(float(settings["side"]), int(self.count))  # pyright: ignore[reportArgumentType]
        return (SurfaceMesh("sheet", rest, faces, None if self.still else Port("points"),
                            (MeshField("depth", "m", Port("depth")),
                                                                    MeshField("slope", "1", Port("slope")))),)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        b.add_node(ToyIndentedSheetNode(ctx.node_name(), ctx.timestep, side=ctx.scalar("side"), count=self.count,
                                        speed=ctx.scalar("speed"), width=ctx.scalar("width")))


__all__ = ["ToyIndentedSheet", "ToyIndentedSheetNode", "dimple", "sheet"]
