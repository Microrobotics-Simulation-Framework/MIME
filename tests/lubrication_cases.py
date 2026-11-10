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
from mime.nodes.hydro import _film
from mime.nodes.hydro.mrs import MRSDragNode, _handover
from mime.scene import Plane, Scene, Sphere, Tube, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import kg, kg_m2, kg_per_m3, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

#: The sphere's radius (m), the medium's viscosity (Pa s), the tube's radius (m).
RADIUS, VISCOSITY, TUBE = 1e-3, 1e-3, 2e-3
#: The roughness the oracle gates declare on both surfaces (m): far below every gap they judge.
FINE = 1e-9


def above_plane(spacing: float, *, blob: float | None = None, lubrication: bool | None = True,
                roughness: float = FINE, kind: Any = None, name: str = "lubricated_sphere",
                wall_roughness: float | None = None) -> Experiment:
    """``kind`` (a sphere of ``RADIUS`` by default) a radius above the plane z = 0, the medium filling the half-space,
    with ``Drag.MRS`` at ``spacing`` (of the radius) and the hand-over ``lubrication`` (unwritten where ``None``: the
    build's rule), both surfaces' roughness declared (the wall's ``wall_roughness`` where given)."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=(0.0, 0.0, 1.0), offset=m(0.0))))
    experiment.add_medium(Medium("water", fills=floor.half_space))
    return _with_body(experiment, spacing, blob, lubrication, roughness, kind, (0.0, 0.0, 3.0 * RADIUS), "floor",
                      "plane", wall_roughness)


def in_tube(spacing: float, *, blob: float | None = None, lubrication: bool | None = True, roughness: float = FINE,
            kind: Any = None, name: str = "lubricated_tube", flow: Any = None) -> Experiment:
    """As :func:`above_plane`, in a straight tube of radius ``TUBE`` along z, the body on its axis; the water carrying
    ``flow`` where given (a field kind, its density declared)."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=m(TUBE), wall=mm(0.5),
                                                             length=mm(200.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen") if flow is None else
                          Medium("water", fills="tube.lumen", fields={"flow": flow}))
    if flow is not None:
        experiment.declare("water.density", kg_per_m3(1000.0), reason="water")
    return _with_body(experiment, spacing, blob, lubrication, roughness, kind, (0.0, 0.0, 0.0), "tube", "tube")


def _with_body(experiment: Experiment, spacing: float, blob: float | None, lubrication: bool | None,
               roughness: float, kind: Any, at: tuple[float, float, float], wall: str, wall_kind: str,
               wall_roughness: float | None = None) -> Experiment:
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
    settings: dict[str, Any] = {"spacing": m(spacing * RADIUS), "wall_kind": wall_kind}
    if lubrication is not None:
        settings["lubrication"] = lubrication
    if blob is not None:
        settings["blob"] = blob
    experiment.attach(Drag.MRS(**settings), body="ball", medium="water")
    return experiment


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


def tube_pose(gap: float, orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0), reach: float = RADIUS) -> Any:
    """The pose with the body ``gap`` (m) from the tube's wall along -x, ``reach`` its extent across the axis."""
    return jnp.asarray((-(TUBE - reach - gap), 0.0, 0.0, *orientation))


def scaled(resistance: np.ndarray, radius: float = RADIUS) -> np.ndarray:
    """The 6 x 6 in the unit sphere's units (unit radius and viscosity), the method of fundamental solutions'."""
    scale = np.array([1.0, 1.0, 1.0, radius, radius, radius])
    return resistance / (VISCOSITY * radius * np.outer(scale, scale))


def translation_per_rotation(resistance: np.ndarray, radius: float = RADIUS) -> float:
    """A sphere spun about y at a unit rate, force-free: its translation along x over ``radius`` (``U / (a Omega)``)."""
    held = scaled(resistance, radius)
    return float(-np.linalg.solve(held[:3, :3], held[:3, 4])[0])


#: The plants a gate's control switches on by ``W2A_PLANT``: the increment scaled, its coupling counted twice, its
#: normal negated, the lift's floor without its derivative or a hard max, the film's elimination 1 % off, the cutoff
#: from the body's roughness alone, the film's curvature floored at one over a held size (the floor the first build
#: had), the wall's curvature added to the body's rather than taken from it, the body's size read from the search's
#: build-time number (a constant of the program).
PLANTS = ("increment_x1.1", "increment_x2", "coupling_twice", "indefinite", "lift_stopped", "hard_max", "wrong_solve",
          "cutoff_body_only", "held_floor", "wall_sign", "baked_size")


@contextmanager
def planted(plant: str | None = None) -> Iterator[str | None]:
    """The plant named (or ``W2A_PLANT``'s) applied to the hand-over's functions while a case is built and first
    stepped (its step is traced then); nothing where none is named."""
    name = plant if plant is not None else os.environ.get("W2A_PLANT") or None
    if name is not None and name not in PLANTS:
        raise ValueError(f"no plant {name!r}; the plants are {PLANTS}")
    held = {"increment": _handover.increment, "smooth_floor": _handover.smooth_floor,
            "roughness_cutoff": _handover.roughness_cutoff, "solve_film": _film.solve_film,
            "relative_curvature": _handover.relative_curvature, "near_contact": _handover.near_contact}
    try:
        if name in ("increment_x1.1", "increment_x2", "coupling_twice", "indefinite"):
            def increment(*arguments: Any, **settings: Any) -> Any:
                found = held["increment"](*arguments, **settings)
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
            _handover.increment = increment
        elif name == "lift_stopped":
            _handover.smooth_floor = lambda value, floor, width: jax.lax.stop_gradient(
                held["smooth_floor"](value, floor, width))
        elif name == "hard_max":
            _handover.smooth_floor = lambda value, floor, width: jnp.maximum(value, floor)
        elif name == "wrong_solve":
            _film.solve_film = lambda *arguments: 1.01 * held["solve_film"](*arguments)
        elif name == "cutoff_body_only":
            _handover.roughness_cutoff = lambda body, wall: 2.0 * body
        elif name == "held_floor":
            def floored(contact: Any, delta: Any, cutoff: Any, handover: Any, reach: Any, *rest: Any,
                        **settings: Any) -> Any:
                size = jax.lax.stop_gradient(reach)
                return held["increment"](contact._replace(k1=jnp.maximum(contact.k1, 1.0 / size)), delta, cutoff,
                                         handover, size, *rest, **settings)
            _handover.increment = floored
        elif name == "wall_sign":
            _handover.relative_curvature = lambda body, wall: body + wall
        elif name == "baked_size":
            def baked(search: Any, *arguments: Any) -> Any:
                return held["near_contact"](search, *arguments[:-1], jnp.asarray(search.size))
            _handover.near_contact = baked
        yield name
    finally:
        _handover.increment = held["increment"]
        _handover.smooth_floor = held["smooth_floor"]
        _handover.roughness_cutoff = held["roughness_cutoff"]
        _film.solve_film = held["solve_film"]
        _handover.relative_curvature = held["relative_curvature"]
        _handover.near_contact = held["near_contact"]


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
        settings["lubrication"] = lubrication
    if spacing is not None:
        settings["spacing"] = m(spacing)
    experiment.attach(Drag.MRS(**settings), body="ball", medium="water")
    return experiment


def taken_default(experiment: Experiment, setting: str, owner: str = "drag:ball") -> object:
    """The value the plan took for ``owner``'s ``setting`` by its default rule (``None`` where it took none)."""
    return next((d.value for d in experiment.plan().profile.defaults if d.path == f"{owner}.{setting}"), None)
