"""The experiments the MRS rung is tested in: rigid bodies in an unbounded medium at rest, each with
``Drag.MRS`` and nothing else, one body per hull kind the rung serves (a sphere, a capsule, de Jongh's FL-9 helix),
each starting at rest at its own place and turned."""

import dataclasses
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
from mime.nodes.hydro.mrs import MRSDragNode
from mime.registry import Dynamics, Geometry, Interaction
from mime.scene import KEPT, Box, Capsule, Cylinder, Ellipsoid, Scene, Sphere, Tube, Z
from mime.things import Body, Medium, Part, State, Structure
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


def mrs(spacing: Quantity, **settings: object) -> Interaction:
    """``Drag.MRS`` in free space with the 0.2 line's blob, half the mean spacing."""
    return Drag.MRS(spacing=spacing, blob=settings.pop("blob", 0.5), wall_kind="free_space", **settings)


def free_space(name: str, kinds: Mapping[str, Geometry], law: Mapping[str, Interaction] | Interaction, *,
               fills: str | None = None, viscosity: Quantity = WATER, mass: Quantity = kg(1e-5),
               inertia: Quantity = kg_m2((1e-11, 1e-11, 1e-11)), velocity: Quantity = m_per_s((0.0, 0.0, 0.0)),
               spin: Quantity = rad_per_s((0.0, 0.0, 0.0)), dynamics: Dynamics | None = None) -> Experiment:
    """Each of ``kinds`` the hull of a body of its name, in water filling no region (or ``fills``), with ``law``
    (one for all, or one per body) and nothing else, each starting at ``velocity`` and ``spin``, on ``dynamics`` (the
    rigid body of the current version when none)."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_medium(Medium("water", fills=fills) if fills is not None else Medium("water"))
    experiment.declare("water.viscosity", viscosity, reason="water")
    for body, kind in kinds.items():
        position, orientation = PLACES.get(body, (mm((0.0, 0.0, 0.0)), (1.0, 0.0, 0.0, 0.0)))
        experiment.add_body(Body(body, dynamics=dynamics or RigidInertial(), parts={"shell": Part(kind)}, initial=State(
            position=position, orientation=orientation, velocity=velocity, angular_velocity=spin)))
        experiment.declare(f"{body}.mass", mass, reason="a millimetre body's order")
        experiment.declare(f"{body}.inertia", inertia, reason="a millimetre body's order")
        held = law[body] if isinstance(law, Mapping) else law
        experiment.attach(held, body=body, medium="water")
    return experiment


def mrs_free_space() -> Experiment:
    """The round-trip case: a ball, a pill and a screw, each with the rung at its coarsest spacing, each moving and turning
    from the start, so that every leaf of every node acts within the walk's first steps."""
    return free_space("mrs_free_space", KINDS, mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


def kept_kinds() -> dict[str, Geometry]:
    """The kept versions of the kinds the rung serves, the ball, the pill and FL-9's screw as ``KINDS`` declares them,
    each the version whose sampling is its quadrature (``mime.scene.KEPT``)."""
    found: dict[str, Geometry] = {}
    for body, kind in KINDS.items():
        held = next(cls for (name, _), (cls, _) in KEPT.items() if name == kind.name)
        found[body] = held(**{field.name: getattr(kind, field.name) for field in dataclasses.fields(kind)})
    return found


#: The solids that sample their surface from 7m-a: a cylinder, a box and an ellipsoid, each with distinct lengths.
SOLIDS: Mapping[str, Geometry] = {
    "rod": Cylinder(radius=mm(0.4), length=mm(1.6), axis=(0.0, 0.6, 0.8)),
    "brick": Box(size=mm((0.6, 0.9, 1.3))),
    "egg": Ellipsoid(semi_axes=mm((0.5, 0.7, 1.1))),
}


def mrs_kept() -> Experiment:
    """The round-trip case of the kept versions: ``mrs_free_space`` with each kind the version a folder saved before
    quasi-uniform sampling names, so that a load of one rebuilds what it saved."""
    return free_space("mrs_kept", kept_kinds(), mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


def mrs_solids() -> Experiment:
    """The rung on the solids that sample their surface from 7m-a, at a coarse spacing, each moving and turning."""
    return free_space("mrs_solids", SOLIDS, mrs(mm(0.35)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


def mrs_tube() -> Experiment:
    """The round-trip case of the tube wall kind: a ball a tenth of the tube's radius, moving along the tube's axis and
    turning, with the rung in a straight tube of 5 mm radius along z at its coarsest spacing that keeps every sample
    within the table's readable range. It reads the shipped table, so the asset ``unit_tube_R0.997`` must be fetched."""
    experiment = Experiment("mrs_tube", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(5.0), wall=mm(0.5),
                                                             length=mm(200.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen"))
    experiment.declare("water.viscosity", WATER, reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(0.5)))},
                             initial=State(position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 1e-3)),
                                           angular_velocity=rad_per_s((3.0, -1.0, 2.0)))))
    experiment.declare("ball.mass", kg(1e-7), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-14, 2e-14, 3e-14)), reason="a millimetre body's order")
    experiment.attach(Drag.MRS(spacing=mm(0.3), blob=0.5, wall_kind="tube"), body="ball", medium="water")
    return experiment


def drag_node(built: BuiltExperiment, body: str) -> MRSDragNode:
    node = built.graph.get_node(f"drag:{body}")
    assert isinstance(node, MRSDragNode), type(node)
    return node


def resistance_at(built: BuiltExperiment, body: str, pose: Any = None) -> np.ndarray:
    """The drag node's resistance with the body at ``pose`` (its starting pose by default), in float64 NumPy."""
    node = drag_node(built, body)
    held = node_state(built.graph, f"drag:{body}")
    if pose is None:
        pose = node_state(built.graph, body)["pose"]
    out = node.update(held, {"pose": jnp.asarray(pose, dtype=held["resistance"].dtype)}, 1e-3, params=parameter_leaves(built.graph)[f"drag:{body}"])
    return np.asarray(out["resistance"], dtype=np.float64)
