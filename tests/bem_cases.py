"""The experiments the boundary-element rung is tested in: rigid bodies in an unbounded medium at rest, each with
``Drag.BEM`` and nothing else, one body per hull kind the rung serves (a sphere, a capsule, de Jongh's FL-9 helix),
each starting at rest at its own place and turned."""

import math
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from script_a import fl9

from mime.build._maddening import node_state, parameter_leaves
from mime.dynamics import RigidInertial
from mime.experiment import BuiltExperiment, Experiment
from mime.interactions import Drag
from mime.nodes.hydro.bem import BEMDragNode
from mime.registry import Geometry, Interaction
from mime.scene import Capsule, Scene, Sphere, Z
from mime.things import Body, Medium, Part, State
from mime.units import Quantity, kg, kg_m2, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

WATER = mPa_s(1.0)
#: The kinds the rung serves, as the cases declare them: a ball, a pill along its part's y, FL-9's screw.
BALL = Sphere(radius=mm(1.0))
PILL = Capsule(radius=mm(0.5), height=mm(2.0), axis=(0.0, 1.0, 0.0))
KINDS: Mapping[str, Geometry] = {"ball": BALL, "pill": PILL, "screw": fl9()}
#: Where each body starts, and turned how: half a radian about a different axis each.
HALF, TURN = math.cos(0.25), math.sin(0.25)
PLACES = {"ball": (mm((0.0, 0.0, 0.0)), (HALF, TURN, 0.0, 0.0)),
          "pill": (mm((10.0, 0.0, 0.0)), (HALF, 0.0, TURN, 0.0)),
          "screw": (mm((0.0, 10.0, 0.0)), (HALF, 0.0, 0.0, TURN))}


def bem(spacing: Quantity, **settings: object) -> Interaction:
    """``Drag.BEM`` in free space with the 0.2 line's blob, half the mean spacing."""
    return Drag.BEM(spacing=spacing, blob=settings.pop("blob", 0.5), wall_kind="free_space", **settings)


def free_space(name: str, kinds: Mapping[str, Geometry], law: Mapping[str, Interaction] | Interaction, *,
               fills: str | None = None, viscosity: Quantity = WATER, mass: Quantity = kg(1e-5),
               inertia: Quantity = kg_m2((1e-11, 1e-11, 1e-11)), velocity: Quantity = m_per_s((0.0, 0.0, 0.0)),
               spin: Quantity = rad_per_s((0.0, 0.0, 0.0))) -> Experiment:
    """Each of ``kinds`` the hull of a body of its name, in water filling no region (or ``fills``), with ``law``
    (one for all, or one per body) and nothing else, each starting at ``velocity`` and ``spin``."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_medium(Medium("water", fills=fills) if fills is not None else Medium("water"))
    experiment.declare("water.viscosity", viscosity, reason="water")
    for body, kind in kinds.items():
        position, orientation = PLACES.get(body, (mm((0.0, 0.0, 0.0)), (1.0, 0.0, 0.0, 0.0)))
        experiment.add_body(Body(body, dynamics=RigidInertial(), parts={"shell": Part(kind)}, initial=State(
            position=position, orientation=orientation, velocity=velocity, angular_velocity=spin)))
        experiment.declare(f"{body}.mass", mass, reason="a millimetre body's order")
        experiment.declare(f"{body}.inertia", inertia, reason="a millimetre body's order")
        held = law[body] if isinstance(law, Mapping) else law
        experiment.attach(held, body=body, medium="water")
    return experiment


def bem_free_space() -> Experiment:
    """The round-trip case: a ball, a pill and a screw, each with the rung at its coarsest spacing, each moving and turning
    from the start, so that every leaf of every node acts within the walk's first steps."""
    return free_space("bem_free_space", KINDS, bem(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


def drag_node(built: BuiltExperiment, body: str) -> BEMDragNode:
    node = built.graph.get_node(f"drag:{body}")
    assert isinstance(node, BEMDragNode), type(node)
    return node


def resistance_at(built: BuiltExperiment, body: str, pose: Any = None) -> np.ndarray:
    """The drag node's resistance with the body at ``pose`` (its starting pose by default), in float64 NumPy."""
    node = drag_node(built, body)
    held = node_state(built.graph, f"drag:{body}")
    if pose is None:
        pose = node_state(built.graph, body)["pose"]
    out = node.update(held, {"pose": jnp.asarray(pose, dtype=held["resistance"].dtype)}, 1e-3, params=parameter_leaves(built.graph)[f"drag:{body}"])
    return np.asarray(out["resistance"], dtype=np.float64)
