"""The experiments lubrication's hand-over is gated in (``tests/oracles/test_lubrication.py``): a body above a plane
wall or in a straight tube, with drag 0.5.0's mrs rung and its hand-over on, each surface's roughness declared; the
drag node's state at poses (its own step); and the plants the gates' controls switch on by ``W2A_PLANT`` (read when a
case is built, so that ``.github/probe/detection_rate.py`` can fan a control out over fresh processes)."""

import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.build._maddening import node_state, parameter_leaves
from mime.dynamics import RigidInertial
from mime.experiment import BuiltExperiment, Experiment
from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode, _handover
from mime.nodes.hydro.near_contact import _elliptic, _search, _seam, _sphere
from mime.scene import Plane, Scene, Sphere, Tube, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import kg, kg_m2, kg_per_m3, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

#: The sphere's radius (m), the medium's viscosity (Pa s), the tube's radius (m).
RADIUS, VISCOSITY, TUBE = 1e-3, 1e-3, 2e-3
#: The roughness the oracle gates declare on both surfaces (m): far below every gap they judge.
FINE = 1e-9


def above_plane(spacing: float | None, *, blob: float | None = None, lubrication: bool | None = True,
                roughness: float = FINE, kind: Any = None, name: str = "lubricated_sphere",
                wall_roughness: float | None = None, drag: Any = None) -> Experiment:
    """``kind`` (a sphere of ``RADIUS`` by default) a radius above the plane z = 0, the medium filling the half-space,
    with ``Drag.MRS`` at ``spacing`` (of the radius; the family's where ``None``) and the hand-over ``near_contact``
    as ``lubrication`` (unwritten where ``None``: the build's rule), both surfaces' roughness declared (the wall's
    ``wall_roughness`` where given); ``drag``, where given, the law attached in its place."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=(0.0, 0.0, 1.0), offset=m(0.0))))
    experiment.add_medium(Medium("water", fills=floor.half_space))
    return _with_body(experiment, spacing, blob, lubrication, roughness, kind, (0.0, 0.0, 3.0 * RADIUS), "floor",
                      "plane", wall_roughness, drag)


def in_tube(spacing: float | None, *, blob: float | None = None, lubrication: bool | None = True,
            roughness: float = FINE, kind: Any = None, name: str = "lubricated_tube", flow: Any = None,
            tube: float = TUBE) -> Experiment:
    """As :func:`above_plane`, in a straight tube of radius ``tube`` (``TUBE`` by default) along z, the body on its
    axis; the water carrying ``flow`` where given (a field kind, its density declared)."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=m(tube), wall=mm(0.5),
                                                             length=mm(200.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen") if flow is None else
                          Medium("water", fills="tube.lumen", fields={"flow": flow}))
    if flow is not None:
        experiment.declare("water.density", kg_per_m3(1000.0), reason="water")
    return _with_body(experiment, spacing, blob, lubrication, roughness, kind, (0.0, 0.0, 0.0), "tube", "tube")


def _with_body(experiment: Experiment, spacing: float | None, blob: float | None, lubrication: bool | None,
               roughness: float, kind: Any, at: tuple[float, float, float], wall: str, wall_kind: str,
               wall_roughness: float | None = None, drag: Any = None) -> Experiment:
    experiment.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(),
                             parts={"shell": Part(kind if kind is not None else Sphere(radius=m(RADIUS)))},
                             initial=State(position=m(at), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    experiment.declare("ball.roughness", m(roughness), reason="the gate's declared roughness")
    experiment.declare(f"{wall}.roughness", m(roughness if wall_roughness is None else wall_roughness),
                       reason="the gate's declared roughness")
    settings: dict[str, Any] = {"wall_kind": wall_kind}
    if spacing is not None:
        settings["spacing"] = m(spacing * RADIUS)
    if lubrication is not None:
        settings["near_contact"] = lubrication
    if blob is not None:
        settings["blob"] = blob
    experiment.attach(Drag.MRS(**settings) if drag is None else drag, body="ball", medium="water")
    return experiment


#: The two-contact gate's body and tube (``tests/oracles/test_elliptic_gates.py``, gate 5): an ellipsoid lying across a
#: straight tube of radius 1 mm, its long axis along x, the tube's along z, its two ends 0.08 mm from the wall in all.
ACROSS_AXES, ACROSS_TUBE, ACROSS_BLOB = (0.96e-3, 0.65e-3, 0.65e-3), 1e-3, 0.05


def across(h1: float, h2: float) -> Any:
    """The pose with the body's +x end ``h1`` and its -x end ``h2`` from the wall (m, summing to 0.08 mm): its centre on
    x."""
    return jnp.asarray(((h2 - h1) / 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))


def across_tube(spacing: float, near_contact: bool | None) -> Experiment:
    """The two-contact case at a force ``spacing`` (m), the hand-over on, off or unwritten."""
    from mime.scene import Ellipsoid

    return in_tube(spacing / RADIUS, blob=ACROSS_BLOB, lubrication=near_contact, tube=ACROSS_TUBE,
                   kind=Ellipsoid(semi_axes=m(ACROSS_AXES)), name="ellipsoid_across")


def drag_node(built: BuiltExperiment) -> MRSDragNode:
    node = built.graph.get_node("drag:ball")
    assert isinstance(node, MRSDragNode), type(node)
    return node


def stepper(built: BuiltExperiment) -> tuple[MRSDragNode, Any, dict[str, Any], Any]:
    """The drag node, its state and leaves, and its step at a pose, jitted."""
    node = drag_node(built)
    state = node_state(built.graph, "drag:ball")
    leaves = parameter_leaves(built.graph)["drag:ball"]
    return node, state, leaves, jax.jit(lambda pose, held: node.update(state, {"pose": pose}, 1e-3, params=held))


def plane_pose(gap: float, orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0), lowest: float = RADIUS) -> Any:
    """The pose with the body's lowest point ``gap`` (m) above the plane, ``lowest`` below its centre."""
    return jnp.asarray((0.0, 0.0, lowest + gap, *orientation))


def tube_pose(gap: float, orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0), reach: float = RADIUS,
              tube: float = TUBE) -> Any:
    """The pose with the body ``gap`` (m) from the wall of a tube of radius ``tube`` along -x, ``reach`` its extent
    across the axis."""
    return jnp.asarray((-(tube - reach - gap), 0.0, 0.0, *orientation))


def scaled(resistance: np.ndarray, radius: float = RADIUS) -> np.ndarray:
    """The 6 x 6 in the unit sphere's units (unit radius and viscosity), the method of fundamental solutions'."""
    scale = np.array([1.0, 1.0, 1.0, radius, radius, radius])
    return resistance / (VISCOSITY * radius * np.outer(scale, scale))


def translation_per_rotation(resistance: np.ndarray, radius: float = RADIUS) -> float:
    """A sphere spun about y at a unit rate, force-free: its translation along x over ``radius`` (``U / (a Omega)``)."""
    held = scaled(resistance, radius)
    return float(-np.linalg.solve(held[:3, :3], held[:3, 4])[0])


#: The plants a gate's control switches on by ``W2A_PLANT``: the summed increment scaled, its coupling counted twice,
#: its normal negated, the opening without its derivative or a hard max, the cutoff from the body's roughness alone,
#: the elliptic form's curvatures without their derivative, the wall's curvature added to the body's rather than taken
#: from it, the body's size read from the search's build-time number (a constant of the program); for the sphere's
#: form: the plane's singular terms kept in a tube (the elliptic contact's curvatures all the body's own), the
#: contact's coefficients from the gap's curvature alone (the body's taken as the gap's), the table's continuation below
#: its edge matched in value only (a kink in its slope there), the two rotations' denominators swapped (``k1 (3 k1 + 2
#: k2)`` for ``k2 (2 k1 + 3 k2)``: equal on a plane, so only a tube sees it; the PR's review's mutant) and the sphere's
#: radius without its derivative; and those a test applies itself (the elliptic form's cross term with its sign
#: flipped, the plane's ambient rule in a lumen, every contact classed a point, several contacts composed one at a time,
#: by S4's subtraction or by moving the body).
PLANTS = ("increment_x1.1", "increment_x2", "coupling_twice", "indefinite", "lift_stopped", "hard_max",
          "cutoff_body_only", "curvature_stopped", "wall_sign", "baked_size", "plane_in_tube", "relative_coefficients",
          "edge_kink", "rot_swap", "radius_stopped", "chi_sign", "plane_ambient", "class_point", "sphere_law",
          "one_at_a_time", "subtraction", "per_body_lift")


@contextmanager
def planted(plant: str | None = None) -> Iterator[str | None]:
    """The plant named (or ``W2A_PLANT``'s) applied to the hand-over's functions while a case is built and first
    stepped (its step is traced then); nothing where none is named, or where the test applies it itself."""
    name = plant if plant is not None else os.environ.get("W2A_PLANT") or None
    if name is not None and name not in PLANTS:
        raise ValueError(f"no plant {name!r}; the plants are {PLANTS}")
    held = {"summed": _handover.summed, "opening": _seam.opening, "roughness_cutoff": _handover.roughness_cutoff,
            "relative_curvature": _search.relative_curvature, "near_contacts": _handover.near_contacts,
            "singular": _sphere.singular, "elliptic": _elliptic.singular, "table": _sphere.table,
            "sphere_radius": _sphere.sphere_radius}
    try:
        if name in ("increment_x1.1", "increment_x2", "coupling_twice", "indefinite"):
            def summed(*arguments: Any, **settings: Any) -> Any:
                found = held["summed"](*arguments, **settings)
                value = found.resistance
                if name == "increment_x1.1":
                    value = 1.1 * value
                elif name == "increment_x2":
                    value = 2.0 * value
                elif name == "coupling_twice":
                    value = value.at[:3, 3:].add(value[:3, 3:]).at[3:, :3].add(value[3:, :3])
                else:
                    value = value.at[2, 2].set(-value[2, 2] - 10.0 * jnp.abs(value[2, 2]))
                return found._replace(resistance=value)
            _handover.summed = summed
        elif name == "lift_stopped":
            _seam.opening = lambda gap, resolved: jax.lax.stop_gradient(held["opening"](gap, resolved))
        elif name == "hard_max":
            _seam.opening = lambda gap, resolved: jnp.maximum(gap, resolved) - gap
        elif name == "cutoff_body_only":
            _handover.roughness_cutoff = lambda body, wall: 2.0 * body
        elif name == "curvature_stopped":
            def stopped(k1: Any, k2: Any, b1: Any, b2: Any, b12: Any, *rest: Any) -> Any:
                return held["elliptic"](*jax.lax.stop_gradient((k1, k2, b1, b2, b12)), *rest)
            _elliptic.singular = stopped
        elif name == "wall_sign":
            _search.relative_curvature = lambda body, wall: body + wall
        elif name == "baked_size":
            def baked(search: Any, *arguments: Any) -> Any:
                return held["near_contacts"](search, *arguments[:-2], jnp.asarray(search.size), arguments[-1])
            _handover.near_contacts = baked
        elif name == "plane_in_tube":
            def plane(k1: Any, k2: Any, b1: Any, b2: Any, *rest: Any) -> Any:
                own = 0.5 * (b1 + b2)
                return held["singular"](own, own, own, own, *rest)
            _sphere.singular = plane
        elif name == "relative_coefficients":
            _sphere.singular = lambda k1, k2, b1, b2, *rest: held["singular"](k1, k2, k1, k2, *rest)
        elif name == "rot_swap":
            def swapped(*arguments: Any) -> Any:
                found = held["singular"](*arguments)
                return found.at[3, 3].set(found[4, 4]).at[4, 4].set(found[3, 3])
            _sphere.singular = swapped
        elif name == "radius_stopped":
            _sphere.sphere_radius = lambda b1, b2: jax.lax.stop_gradient(held["sphere_radius"](b1, b2))
        elif name == "edge_kink":
            def kinked() -> Any:
                table = held["table"]()
                edge = table.edge
                value = table.below[0] + table.below[1] * edge * np.log(edge) + table.below[2] * edge
                return table._replace(below=np.stack([value, np.zeros_like(value), np.zeros_like(value)]))
            _sphere.table = kinked
        yield name
    finally:
        _handover.summed = held["summed"]
        _seam.opening = held["opening"]
        _handover.roughness_cutoff = held["roughness_cutoff"]
        _search.relative_curvature = held["relative_curvature"]
        _handover.near_contacts = held["near_contacts"]
        _sphere.singular = held["singular"]
        _elliptic.singular = held["elliptic"]
        _sphere.table = held["table"]
        _sphere.sphere_radius = held["sphere_radius"]


def compared(built, pose_of, gaps, leaf_names, step=1e-6, forward_too=True):
    """Reverse, forward and Richardson's central difference of a weighted sum of the resistance in each leaf, at each
    gap; each leaf stepped by ``step`` of its size. The loss takes the pose and the weights as arguments, so that each
    derivative compiles once for every gap. Without ``forward_too`` the forward column repeats the reverse one (the
    controls judge the reverse alone)."""
    node, state, leaves, _ = stepper(built)

    def loss(values, pose, along, weights):
        held = {**leaves, **{k: v for k, v in zip(leaf_names, values[:-1], strict=True)}}
        moved = pose + values[-1] * along
        return jnp.sum(weights * node.update(state, {"pose": moved}, 1e-3, params=held)["resistance"])

    value, reverse_of = jax.jit(loss), jax.jit(jax.grad(loss))
    forward_of = jax.jit(jax.jacfwd(loss)) if forward_too else reverse_of
    found = []
    for gap in gaps:
        pose = pose_of(gap)
        scale = np.asarray(node.update(state, {"pose": pose}, 1e-3, params=leaves)["resistance"])
        weights = jnp.asarray((1.0 + 0.1 * np.arange(36).reshape(6, 6) / 36.0)
                              / np.sqrt(np.abs(np.outer(np.diag(scale), np.diag(scale)))))
        along = jnp.zeros(7).at[2 if float(pose[2]) != 0.0 else 0].set(1.0)
        at = jnp.asarray([*(leaves[k] for k in leaf_names), 0.0])
        reverse = np.asarray(reverse_of(at, pose, along, weights))
        forward = np.asarray(forward_of(at, pose, along, weights))
        here = []
        for i, name in enumerate((*leaf_names, "position")):
            size = RADIUS if name in ("position", "wall_offset", "wall_radius") else abs(float(at[i]))
            h = step * size
            e = jnp.zeros_like(at).at[i].set(h)
            coarse = (float(value(at + e, pose, along, weights)) - float(value(at - e, pose, along, weights))) / (2 * h)
            fine = (float(value(at + e / 2, pose, along, weights)) - float(value(at - e / 2, pose, along, weights))) / h
            # each derivative on the loss's scale per unit relative move of its leaf, so that a leaf the resistance
            # does not read at this gap (the roughness above the cutoff) is judged against the others
            here.append((gap, name, reverse[i] * size, forward[i] * size, (4 * fine - coarse) / 3 * size))
        scale = max(max(abs(d), abs(r)) for _, _, r, _, d in here)
        found += [(*row, scale) for row in here]
    return found


def turned(angle: float, axis: tuple[float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(c * c for c in axis))
    return (math.cos(angle / 2), *(math.sin(angle / 2) * c / norm for c in axis))


def screw_in_tube(lubrication: bool | None = None, spacing: float | None = None) -> Experiment:
    """FL-9 (``script_a.fl9``) on its tube's axis (script A's 1/4 in tube along x), with ``Drag.MRS`` at its defaults
    but ``lubrication`` and ``spacing`` (m) where written: the record's case of a contact past the film's window."""
    from script_a import fl9

    experiment = Experiment("screw_in_tube", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(3.175),
                                                             wall=mm(0.5), length=mm(100.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen"))
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(fl9())}, initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    settings: dict[str, Any] = {"wall_kind": "tube"}
    if lubrication is not None:
        settings["near_contact"] = lubrication
    if spacing is not None:
        settings["spacing"] = m(spacing)
    experiment.attach(Drag.MRS(**settings), body="ball", medium="water")
    return experiment


def taken_default(experiment: Experiment, setting: str, owner: str = "drag:ball") -> object:
    """The value the plan took for ``owner``'s ``setting`` by its default rule (``None`` where it took none)."""
    return next((d.value for d in experiment.plan().profile.defaults if d.path == f"{owner}.{setting}"), None)
