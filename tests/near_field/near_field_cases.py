"""The near field's test cases: bodies on drag 0.5.0's mrs rung in one medium, their joint node, and an independent
dense assembly of the same joint system from dev1's own pieces, to judge the node against.

Lengths in SI; a sphere is MIME's own (its sampling, separated quadrature and blob, as the rung builds them)."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime import Body, Experiment, Medium, State, Structure
from mime.build._maddening import node_state, parameter_leaves
from mime.dynamics import RigidInertial
from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.nodes.hydro.mrs._quadrature import cell_integrated, regularised_stokeslet, turned
from mime.nodes.hydro.near_field import JointDragNode, member_prefix
from mime.registry import Geometry, Interaction
from mime.scene import Plane, Scene, Sphere, Tube, Z
from mime.things import Part
from mime.units import kg, kg_m2, m, m_per_s, mPa_s, ms, rad_per_s

VISCOSITY = 1e-3
RADIUS = 1e-3
IDENTITY = (1.0, 0.0, 0.0, 0.0)
#: The node the build gives a medium named water.
NODE = "near_field:water"


@contextlib.contextmanager
def precision(dtype: str) -> Iterator[None]:
    """JAX's float64 mode on for ``"float64"`` and off for ``"float32"``, restored after."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def bodies(name: str, placed: Mapping[str, tuple[Geometry, Sequence[float]]], law: Interaction | Mapping[str, Interaction],
           *, velocity: Sequence[float] = (0.0, 0.0, 0.0), spin: Sequence[float] = (0.0, 0.0, 0.0),
           mass: float = 4.19e-6, inertia: float = 1.7e-12, dt_ms: float = 1.0, tube: float | None = None,
           plane: float | None = None, orientations: Mapping[str, Sequence[float]] | None = None,
           rate: Any = None) -> Experiment:
    """Each of ``placed`` (a body's hull kind and its centre, m) a rigid body in water (in a tube of radius ``tube`` along
    x, or above a plane wall whose normal is +z at the offset ``plane``, where given, else unbounded), with ``law`` (one
    for all, or one per body), each starting at ``velocity`` and ``spin``, its dynamics stepping at ``rate`` where given
    (else at the base step)."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2_zero(), dt=ms(dt_ms))
    if plane is not None:
        floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=(0.0, 0.0, 1.0), offset=m(plane))))
        experiment.add_medium(Medium("water", fills=floor.half_space))
    elif tube is not None:
        experiment.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=m(tube),
                                                                 wall=m(0.1 * tube), length=m(80.0 * tube))))
        experiment.add_medium(Medium("water", fills="tube.lumen"))
    else:
        experiment.add_medium(Medium("water"))
    experiment.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water")
    for body, (kind, centre) in placed.items():
        orientation = tuple((orientations or {}).get(body, IDENTITY))
        dynamics = RigidInertial() if rate is None else RigidInertial(rate=rate)
        experiment.add_body(Body(body, dynamics=dynamics, parts={"shell": Part(kind)}, initial=State(
            position=m(tuple(centre)), orientation=orientation, velocity=m_per_s(tuple(velocity)),
            angular_velocity=rad_per_s(tuple(spin)))))
        experiment.declare(f"{body}.mass", kg(mass), reason="a millimetre body's order")
        experiment.declare(f"{body}.inertia", kg_m2((inertia,) * 3), reason="a millimetre body's order")
        experiment.attach(law[body] if isinstance(law, Mapping) else law, body=body, medium="water")
    return experiment


def m_per_s2_zero() -> Any:
    from mime.units import m_per_s2

    return m_per_s2(0.0)


def spheres(gap: float, count: int = 2, radius: float = RADIUS) -> dict[str, tuple[Geometry, tuple[float, float, float]]]:
    """``count`` equal spheres of ``radius`` along x (two or more), or for three an equilateral triangle, each pair
    ``gap`` apart (m) where they are nearest."""
    side = 2.0 * radius + gap
    if count == 3:
        centres = [(0.0, 0.0, 0.0), (side, 0.0, 0.0), (0.5 * side, side * math.sqrt(3.0) / 2.0, 0.0)]
    else:
        centres = [(i * side, 0.0, 0.0) for i in range(count)]
    return {f"s{i}": (Sphere(radius=m(radius)), centre) for i, centre in enumerate(centres)}


def line(gap: float, count: int = 3, radius: float = RADIUS) -> dict[str, tuple[Geometry, tuple[float, float, float]]]:
    side = 2.0 * radius + gap
    return {f"s{i}": (Sphere(radius=m(radius)), (i * side, 0.0, 0.0)) for i in range(count)}


def mrs(spacing: float, **settings: Any) -> Interaction:
    """Drag 0.5.0's mrs rung in free space at ``spacing`` (m), the rung's defaults otherwise."""
    wall = settings.pop("wall_kind", "free_space")
    if wall != "free_space":  # the hand-over between bodies sharing a medium comes with the joint node's (W2a's PR 2)
        settings.setdefault("near_contact", False)
    return Drag.MRS(spacing=m(spacing), wall_kind=wall, **settings)


def joint_node(built: Any) -> JointDragNode:
    node = built.graph.get_node(NODE)
    assert isinstance(node, JointDragNode), type(node)
    return node


def start_inputs(built: Any) -> dict[str, Any]:
    """The joint node's inputs at the start: each member's pose, twist and mass matrix from its body's initial
    state."""
    node = joint_node(built)
    found: dict[str, Any] = {}
    for i, body in enumerate(node.members):
        state = node_state(built.graph, body)
        found[f"pose_{i}"] = state["pose"]
        found[f"twist_{i}"] = state["twist"]
        found[f"mass_matrix_{i}"] = state["mass_matrix"]
    return found


def stepped(built: Any, inputs: Mapping[str, Any] | None = None, params: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    """One step of the joint node from its initial state at ``inputs`` (the start's by default)."""
    node = joint_node(built)
    held = inputs if inputs is not None else start_inputs(built)
    out = node.update(node.initial_state(), dict(held), node.delta_t,
                      params=params if params is not None else parameter_leaves(built.graph)[NODE])
    return {key: np.asarray(value) for key, value in out.items()}


def single(node: MRSDragNode, name: str = "near_field:one", **settings: Any) -> JointDragNode:
    """A joint node of one member, built from that member's per-body node's own arguments (and the joint node's own
    ``settings``, its route's): what gate 1 compares with the per-body node."""
    constants = dict(node.constants())
    common = {key: constants.pop(key) for key in ("viscosity", "wall_kind", "wall_kind_version")}
    from mime.nodes.hydro.mrs._walls import wall_correction_named

    walls = {f"wall_{p}" for p in wall_correction_named(common["wall_kind"], common["wall_kind_version"]).parameters}
    held = {f"{member_prefix(0)}{key}": value for key, value in constants.items() if key not in walls}
    return JointDragNode(name, float(node.delta_t), members=("one",), **common,
                         **{key: constants[key] for key in walls}, **settings, **held)


def assembled(nodes: Sequence[MRSDragNode], bodies_named: Sequence[str], name: str = NODE, **settings: Any) -> JointDragNode:
    """The joint node of ``nodes`` (each body's per-body node, as drag 0.4.0 builds it), made from their own arguments:
    what the build makes on 0.5.0, without the build's start check (a configuration the build refuses at its start)."""
    from mime.nodes.hydro.mrs._walls import wall_correction_named

    first = dict(nodes[0].constants())
    common = {key: first[key] for key in ("viscosity", "wall_kind", "wall_kind_version")}
    walls = {f"wall_{p}" for p in wall_correction_named(common["wall_kind"], common["wall_kind_version"]).parameters}
    held: dict[str, Any] = {key: first[key] for key in walls}
    for i, node in enumerate(nodes):
        held.update({f"{member_prefix(i)}{key}": value for key, value in node.constants().items()
                     if key not in walls and key not in common})
    return JointDragNode(name, float(nodes[0].delta_t), members=tuple(bodies_named), **common, **settings, **held)


def uncoupled_inputs(built: Any, bodies_named: Sequence[str]) -> dict[str, Any]:
    """The joint node's inputs at the start from a build on drag 0.4.0: each body's pose and twist, and its mass matrix
    in the world from its own mass and inertia (its dynamics publishes none there)."""
    found: dict[str, Any] = {}
    for i, body in enumerate(bodies_named):
        state = node_state(built.graph, body)
        node = built.graph.get_node(body)
        found[f"pose_{i}"] = state["pose"]
        found[f"twist_{i}"] = state["twist"]
        found[f"mass_matrix_{i}"] = node.world_mass_matrix(
            {key: jnp.asarray(node.constants()[key], dtype=state["pose"].dtype) for key in ("mass", "inertia")},
            state["pose"])
    return found


def reference_grand(node: JointDragNode, poses: np.ndarray, every: bool = True) -> np.ndarray:
    """The joint system's grand resistance assembled independently from dev1's pieces, free space only: each member's
    free part made afresh at its built leaves (``separated_matrix`` through ``MRSDragNode.self_interaction``) and
    turned to its pose, each cross-block the source's regularised Stokeslet integrated over all its cells at the
    target's force points (``cell_integrated`` on the source's whole quadrature, ``every``) or at its force points
    (``every=False``), one ``jnp.linalg.solve`` of the whole, ``W^T A^-1 M`` per body pair: S3's pipeline, none of the
    node's own assembly."""
    from mime.core import quaternion
    from mime.nodes.hydro.mrs._quadrature import Chunks
    from mime.nodes.hydro.mrs._solve import rigid_motions

    members = node.member_nodes
    placed = []
    for member, pose in zip(members, poses, strict=True):
        theta = {key: jnp.asarray(value) for key, value in member.built_theta().items()}
        body = member.body(theta)
        turn = quaternion.rotation_matrix(jnp.asarray(pose[3:]))
        shift = jnp.asarray(pose[:3])
        placed.append(dict(
            self=turned(member.self_interaction(theta), turn) / VISCOSITY,
            force=shift + body.force @ turn.T, centroid=shift + body.centroid @ turn.T, area=body.area,
            points=shift + body.points @ turn.T, weights=body.weights, eps=body.eps_squared, centre=shift,
            owners=np.asarray(member.owners), cells=member.sample_count))
    rows = []
    for i, target in enumerate(placed):
        row = []
        for j, source in enumerate(placed):
            if i == j:
                row.append(target["self"])
                continue
            if every:
                pairs = max(1, math.ceil(16 * target["force"].shape[0] / source["cells"]))
                block, _ = cell_integrated(regularised_stokeslet(source["eps"]), target["force"], source["points"],
                                           source["weights"], Chunks.of(source["owners"], source["cells"],
                                                                        pair_values=pairs))
            else:
                blocks, _ = regularised_stokeslet(source["eps"])(target["force"], source["force"])
                blocks = blocks * source["area"][None, :, None, None]
                t, n = blocks.shape[0], blocks.shape[1]
                block = blocks.transpose(0, 2, 1, 3).reshape(3 * t, 3 * n)
            row.append(block / VISCOSITY)
        rows.append(jnp.concatenate(row, axis=1))
    matrix = jnp.concatenate(rows, axis=0)
    sizes = [3 * p["force"].shape[0] for p in placed]
    at = np.concatenate([[0], np.cumsum(sizes)])
    motions = jnp.zeros((int(at[-1]), 6 * len(placed)), dtype=matrix.dtype)
    for i, p in enumerate(placed):
        motions = motions.at[int(at[i]):int(at[i + 1]), 6 * i:6 * i + 6].set(rigid_motions(p["force"], p["centre"]))
    tractions = jnp.linalg.solve(matrix, motions)
    grand = []
    for i, p in enumerate(placed):
        wrench = rigid_motions(p["centroid"], p["centre"]) * jnp.repeat(p["area"], 3)[:, None]
        grand.append(wrench.T @ tractions[int(at[i]):int(at[i + 1])])
    found = np.asarray(jnp.concatenate(grand, axis=0))
    return 0.5 * (found + found.T)


def rebuilt(node: JointDragNode, **overrides: Any) -> JointDragNode:
    """``node`` constructed again from its own arguments with ``overrides`` (a reach, a budget's share)."""
    return JointDragNode(node.name, float(node.delta_t), **{**node.constants(), **overrides})


def scaled(matrix: np.ndarray) -> np.ndarray:
    """``matrix`` with each row and column scaled by the square root of its diagonal's magnitude: forces and torques on
    one scale."""
    scale = np.sqrt(np.abs(np.diag(matrix)))
    return matrix / np.outer(scale, scale)


def entrywise(found: np.ndarray, reference: np.ndarray) -> float:
    """The largest entry of ``found - reference`` over its row's and column's diagonal scale in ``reference``."""
    scale = np.sqrt(np.outer(np.abs(np.diag(reference)), np.abs(np.diag(reference))))
    return float(np.max(np.abs(found - reference) / scale))
