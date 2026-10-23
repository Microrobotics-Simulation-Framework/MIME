"""Test-only entries and closed forms for the hydrodynamic family, registered under the ``oracle`` namespace in this
process only, so the registry walk (a fresh process) holds them to no round-trip case.

- ``StokesSphere`` (``oracle.stokes_sphere``): a sphere of the body's radius ``a`` in an unbounded medium at rest,
  ``R = diag(6 pi mu a I3, 8 pi mu a^3 I3)`` about its centre, with no background load (Stokes 1851; Kim & Karrila
  2005, ch. 2). It is the family's analytic oracle and its second member, so the swap case has something to swap. It
  reads the body's pose, as every drag rung does (a flowing medium's background load is evaluated where the body is),
  and uses none of it. Not public: no public ``Drag.Stokes`` is built yet.
- ``surrogate_reference``: the surrogate's formula evaluated again, in float64 NumPy, from the packaged weights read
  with ``numpy.load``, with the canonical frame built as the node builds it. The oracle case of the surrogate's family
  suite holds the node to it: a second implementation, not a physical oracle; it bounds how far the node departs from
  the formula, not how far the formula departs from the flow.
"""

from __future__ import annotations

import importlib.resources
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import numpy as np
from jax import Array
from maddening.core.node import BoundaryInputSpec

from mime.build import InteractionBuilder
from mime.core import MimeNode, NodeFields, as_run_array, node_fields, node_input
from mime.data import PACKAGED
from mime.gradients import GradientStatus
from mime.interactions import Drag
from mime.interactions.hydrodynamic import UniformTube, uniform_tube
from mime.nodes.hydro import AXIS_ULPS, SurrogateDragNode
from mime.registry import BuildContext, Delivery, Interaction, Read, RoleKind, Value, register_interaction
from mime.surrogates import DEJONGH_MLP_V2, TRIANGLE
from mime.vocabulary import BACKGROUND_LOAD, POSE, RESISTANCE, register_namespace

register_namespace("oracle", "0.0.1")


def sphere_resistance(viscosity: float, radius: float) -> np.ndarray:
    """``diag(6 pi mu a I3, 8 pi mu a^3 I3)``."""
    return np.diag([6 * math.pi * viscosity * radius] * 3 + [8 * math.pi * viscosity * radius ** 3] * 3)


class StokesSphereNode(MimeNode):
    def __init__(self, name: str, timestep: float, *, viscosity: float, radius: float) -> None:
        super().__init__(name, timestep, viscosity=viscosity, radius=radius)

    @override
    def initial_state(self) -> NodeFields:
        return node_fields({"resistance": as_run_array(jnp.zeros((6, 6))), "background_load": as_run_array(jnp.zeros(6))})

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        dtype = state["resistance"].dtype
        translation = 6 * math.pi * p["viscosity"] * p["radius"]
        rotation = 8 * math.pi * p["viscosity"] * p["radius"] ** 3
        diagonal = jnp.concatenate([jnp.full(3, translation, dtype=dtype), jnp.full(3, rotation, dtype=dtype)])
        return node_fields({"resistance": jnp.diag(diagonal), "background_load": jnp.zeros(6, dtype=dtype)})


@register_interaction
@dataclass(frozen=True, kw_only=True)
class StokesSphere(Interaction):
    name = "oracle.stokes_sphere"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    reads = {"pose": Read("body", POSE.name)}
    delivers = {"resistance": Delivery("body", RESISTANCE.name), "background_load": Delivery("body", BACKGROUND_LOAD.name)}
    values = {"viscosity": Value("medium", "viscosity"), "radius": Value("body", "radius")}

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(StokesSphereNode(ctx.node_name(), ctx.timestep,
                                                    viscosity=ctx.scalar_value("viscosity"),
                                                    radius=ctx.scalar_value("radius")))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background_load"], into=b.target("background_load"))


def sphere_oracle(read: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {"resistance": sphere_resistance(settings["viscosity"], settings["radius"]), "background_load": np.zeros(6)}


# -- the surrogate, again ---------------------------------------------------------------------------------------------

def _weights() -> dict[str, np.ndarray]:
    path = importlib.resources.files("mime.data") / PACKAGED["dejongh_mlp_v2"].file
    with path.open("rb") as handle, np.load(handle) as stored:
        return {key: np.asarray(stored[key]) for key in stored.files}


WEIGHTS = _weights()
OUTER = 1.0 + DEJONGH_MLP_V2.modulation


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def canonical_resistance(nu: float, length_nd: float, tube_nd: float, offset_nd: float) -> tuple[np.ndarray, bool]:
    """The non-dimensional resistance in the canonical frame, and whether the offset was clamped."""
    largest = max((tube_nd - OUTER) * 0.95, 0.0)
    clamped = min(offset_nd, largest)
    gap = max(tube_nd - (clamped + OUTER), 1e-3)
    kappa = 1.0 / tube_nd
    features = np.array([nu, length_nd, kappa, clamped, 0.0, math.log(gap), nu ** 2, kappa ** 2])
    x = (features - WEIGHTS["X_mean"].astype(np.float64)) / WEIGHTS["X_std"].astype(np.float64)
    count = int(WEIGHTS["n_layers"])
    for i in range(count):
        x = x @ WEIGHTS[f"W{i}"].astype(np.float64) + WEIGHTS[f"b{i}"].astype(np.float64)
        if i < count - 1:
            x = _silu(x)
    entries = x * WEIGHTS["L_std"] + WEIGHTS["L_mean"]
    factor = np.zeros((6, 6))
    for index, (i, j) in enumerate(TRIANGLE):
        factor[i, j] = np.logaddexp(entries[index], 0.0) + 1e-5 if i == j else entries[index]
    return factor @ factor.T, offset_nd > largest


def rotation_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tube_frame(position: np.ndarray, orientation: np.ndarray, centre: Sequence[float], axis: Sequence[float],
               tube_radius: float, computed_in: Any = np.float64) -> tuple[float, np.ndarray]:
    """The body's offset from the axis and the canonical frame ``[n, a x n, a]``, as the node defines them when it
    computes in ``computed_in`` (which sets the offset below which the body is on the axis)."""
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    relative = np.asarray(position, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    radial = relative - (relative @ a) * a
    offset = float(np.linalg.norm(radial))
    if offset > AXIS_ULPS * float(np.finfo(computed_in).eps) * tube_radius:
        normal = radial / offset
    else:
        offset = 0.0
        body = rotation_matrix(orientation)
        across = [body[:, k] - (body[:, k] @ a) * a for k in (0, 1)]
        normal = across[0] if across[0] @ across[0] > 0.25 else across[1]
        normal = normal / np.linalg.norm(normal)
    return offset, np.stack([normal, np.cross(a, normal), a], axis=1)


def shifted(canonical: np.ndarray, shift: float) -> np.ndarray:
    """``canonical`` about a point ``shift`` further along the canonical frame's axis (z), in its own units:
    ``[F; T - d x F]`` against ``[U - d x Omega; Omega]``, ``d = shift * z``."""
    d = np.array([[0.0, -shift, 0.0], [shift, 0.0, 0.0], [0.0, 0.0, 0.0]])
    eye, zero = np.eye(3), np.zeros((3, 3))
    return np.block([[eye, zero], [-d, eye]]) @ canonical @ np.block([[eye, d], [zero, eye]])


def lubrication(viscosity: float, radius: float, tube_radius: float, offset: float, epsilon: float,
                frame: np.ndarray) -> np.ndarray:
    """The 0.2 line's lubrication correction, as its docstring states it."""
    a = OUTER * radius
    gap = max(tube_radius - (offset + a), 1e-9)
    weight = math.exp(-gap / epsilon)
    logarithm = max(math.log(a / gap), 0.0)
    normal_term = 6 * math.pi * viscosity * a ** 2 / gap
    tangential = 6 * math.pi * viscosity * a * (8 / 15) * logarithm
    rotational = 8 * math.pi * viscosity * a ** 3 * (2 / 5) * logarithm
    n, t1 = frame[:, 0], frame[:, 2]
    t2 = np.cross(n, t1)
    wall = np.outer(t1, t1) + np.outer(t2, t2)
    out = np.zeros((6, 6))
    out[:3, :3] = normal_term * np.outer(n, n) + tangential * wall
    out[3:, 3:] = rotational * wall
    return weight * out


def surrogate_reference(pose: Sequence[float], *, viscosity: float, helix_wavenumber: float, length: float,
                        radius: float, centre: Sequence[float], axis: Sequence[float], tube_radius: float,
                        epsilon: float | None, reference_point: str = "centre") -> tuple[np.ndarray, bool]:
    """The surrogate's resistance (SI, world frame) at ``pose``, and whether the offset was clamped: about the body's
    centre (the frame's axis on the side the body's x axis points to, the resistance moved ``L / 2`` along it from the
    end), or about the end, as drag 0.1.0 delivered it."""
    held = np.asarray(pose, dtype=np.float64)
    offset, frame = tube_frame(held[:3], held[3:], centre, axis, tube_radius, np.asarray(pose).dtype)
    canonical, clamped = canonical_resistance(helix_wavenumber, length / radius, tube_radius / radius,
                                              offset / radius)
    if reference_point == "centre":
        a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        side = 1.0 if float(rotation_matrix(held[3:])[:, 0] @ a) >= 0.0 else -1.0
        frame = frame * np.array([1.0, side, side])
        canonical = shifted(canonical, length / radius / 2)
    turn = np.zeros((6, 6))
    turn[:3, :3] = turn[3:, 3:] = frame
    blocks = np.array([[viscosity * radius, viscosity * radius ** 2], [viscosity * radius ** 2,
                                                                       viscosity * radius ** 3]])
    resistance = np.kron(blocks, np.ones((3, 3))) * (turn @ canonical @ turn.T)
    if epsilon is not None:
        resistance = resistance + lubrication(viscosity, radius, tube_radius, offset, epsilon, frame)
    return resistance, clamped


# -- planted variants -------------------------------------------------------------------------------------------------

class PortFrameNode(SurrogateDragNode):
    """The 0.2 line's frame: the tube along +z through the origin whatever the geometry says, the offset as
    ``sqrt(x^2 + y^2)`` and its direction as ``arctan2(y, x)`` (``mlp_resistance_node.py`` at the port, lines
    ``_compute_R_and_drag``)."""

    @override
    def frame(self, p: Mapping[str, Any], position: Array, orientation: Array) -> tuple[Array, Array]:
        x, y = position[0], position[1]
        offset = jnp.sqrt(x ** 2 + y ** 2)
        theta = jnp.arctan2(y, x)
        c, s = jnp.cos(theta), jnp.sin(theta)
        zero, one = jnp.zeros_like(c), jnp.ones_like(c)
        return offset, jnp.stack([jnp.stack([c, -s, zero]), jnp.stack([s, c, zero]), jnp.stack([zero, zero, one])])


def drag_node(cls: type[SurrogateDragNode], ctx: BuildContext, law: Drag) -> SurrogateDragNode:
    """``law``'s node built as ``Drag`` builds it, of class ``cls``."""
    tube = uniform_tube(ctx.region("medium"), ctx.bindings["medium"])
    assert isinstance(tube, UniformTube) and law.weights is not None
    return cls(ctx.node_name(), ctx.timestep, weights=law.weights.name, weights_sha256=law.weights.sha256,
               viscosity=ctx.scalar_value("viscosity"), helix_wavenumber=ctx.scalar_value("helix_wavenumber"),
               length=ctx.scalar_value("length"), radius=ctx.scalar_value("radius"), vessel_centre=tube.centre,
               vessel_axis=tube.axis, vessel_radius=tube.radius,
               lubrication_epsilon=ctx.scalar("lubrication.epsilon") if law.lubrication is not None else None)


def wire(b: InteractionBuilder, node: MimeNode) -> None:
    ref = b.add_response_node(node)
    b.connect(b.read("pose"), into=ref["pose"])
    b.connect(ref["resistance"], into=b.target("resistance"))
    b.connect(ref["background_load"], into=b.target("background_load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PortFrameDrag(Drag):
    """The surrogate with the 0.2 line's +z frame: the frame fix planted out."""

    name = "oracle.port_frame_drag"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        wire(b, drag_node(PortFrameNode, ctx, self))
