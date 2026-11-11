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
from mime.interactions.hydrodynamic import PointForceDrag, UncoupledDrag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.registry import Dynamics, Geometry, Interaction
from mime.scene import KEPT, Box, Capsule, Cylinder, Ellipsoid, Plane, Scene, Sphere, Tube, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import Quantity, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

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
    """``Drag.MRS`` in free space with the 0.2 line's blob, half the mean spacing, and the quadrature four times finer
    (its default at that blob, written so that a blob moved by a study's difference keeps the quadrature)."""
    return Drag.MRS(spacing=spacing, blob=settings.pop("blob", 0.5),
                    quadrature_refinement=settings.pop("quadrature_refinement", 4), wall_kind="free_space", **settings)


def uncoupled(spacing: Quantity, **settings: object) -> Interaction:
    """``mrs`` on drag 0.4.0, kept registered: each body's drag alone, its own node, in a medium of several bodies too
    (0.5.0 couples them in the medium's near field)."""
    return UncoupledDrag.MRS(spacing=spacing, blob=settings.pop("blob", 0.5),
                             quadrature_refinement=settings.pop("quadrature_refinement", 4), wall_kind="free_space",
                             **settings)


def point_forces(spacing: Quantity, **settings: object) -> Interaction:
    """``mrs`` on drag 0.3.0, kept registered: a point force at each sample, the samples the quadrature."""
    return PointForceDrag.MRS(spacing=spacing, blob=settings.pop("blob", 0.5), wall_kind="free_space", **settings)


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


def mrs_free_space(law: Interaction | None = None, name: str = "mrs_free_space") -> Experiment:
    """The round-trip case: a ball, a pill and a screw, each with the rung at its coarsest spacing, each moving and turning
    from the start, so that every leaf of every node acts within the walk's first steps; on drag 0.4.0, kept, so that
    each body has its own node, as a folder saved before the near field names (``mrs_near_field`` is 0.5.0's)."""
    return free_space(name, KINDS, law if law is not None else uncoupled(mm(1.5)), mass=kg(1e-7),
                      inertia=kg_m2((1e-14, 2e-14, 3e-14)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
                      spin=rad_per_s((3.0, -1.0, 2.0)))


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
    quasi-uniform sampling names, and the drag the version it names (0.3.0's point forces), so that a load of one
    rebuilds what it saved."""
    return free_space("mrs_kept", kept_kinds(), point_forces(mm(1.5)), mass=kg(1e-7),
                      inertia=kg_m2((1e-14, 2e-14, 3e-14)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
                      spin=rad_per_s((3.0, -1.0, 2.0)))


def mrs_solids() -> Experiment:
    """The rung on the solids that sample their surface from 7m-a, at a coarse spacing, each moving and turning; on drag
    0.4.0, kept, each body its own node."""
    return free_space("mrs_solids", SOLIDS, uncoupled(mm(0.35)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


#: The near field's round-trip case's bodies and where they start: a ball and a pill beside it, a millimetre apart, and a
#: second ball behind them, each turned, so that its joint node couples blocks of different sizes.
NEAR = {"ball": (BALL, mm((0.0, 0.0, 0.0)), (HALF, TURN, 0.0, 0.0)),
        "pill": (PILL, mm((2.5, 0.0, 0.0)), (HALF, 0.0, TURN, 0.0)),
        "bead": (Sphere(radius=mm(0.6)), mm((1.0, 2.2, 0.0)), (HALF, 0.0, 0.0, TURN))}


def mrs_near_field(name: str = "mrs_near_field", spacing: Quantity = mm(1.0), **settings: object) -> Experiment:
    """The round-trip case of drag 0.5.0's near field: three bodies in one unbounded medium on the mrs rung, one joint
    node, each moving and turning from the start (``settings`` the near field's own)."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_medium(Medium("water"))
    experiment.declare("water.viscosity", WATER, reason="water")
    for body, (kind, position, orientation) in NEAR.items():
        experiment.add_body(Body(body, dynamics=RigidInertial(), parts={"shell": Part(kind)}, initial=State(
            position=position, orientation=orientation, velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
            angular_velocity=rad_per_s((3.0, -1.0, 2.0)))))
        experiment.declare(f"{body}.mass", kg(1e-7), reason="a millimetre body's order")
        experiment.declare(f"{body}.inertia", kg_m2((1e-14, 2e-14, 3e-14)), reason="a millimetre body's order")
        experiment.attach(mrs(spacing, **settings), body=body, medium="water")
    return experiment


def mrs_near_field_gmres() -> Experiment:
    """The near field's case past the takeover line (a line of zero): the same bodies, the joint system by GMRES."""
    return mrs_near_field("mrs_near_field_gmres", near_field_dense_unknowns=0)


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
    # the coarse MRS's round trip: the near-contact law's hand-over (a sphere's form, on by default) written off
    experiment.attach(Drag.MRS(spacing=mm(0.3), blob=0.5, wall_kind="tube", near_contact=False), body="ball",
                      medium="water")
    return experiment


def mrs_plane() -> Experiment:
    """The round-trip case of the plane wall kind: a ball of 0.5 mm radius a radius above a plane wall along a turned
    normal, off the world's origin, moving along the plane and turning, with the rung in the plane wall kind at a coarse
    spacing."""
    normal = (0.0, 0.6, 0.8)
    experiment = Experiment("mrs_plane", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=normal, offset=mm(-1.0))))
    experiment.add_medium(Medium("water", fills=floor.half_space))
    experiment.declare("water.viscosity", WATER, reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(0.5)))},
                             initial=State(position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((1e-3, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((3.0, -1.0, 2.0)))))
    experiment.declare("ball.mass", kg(1e-7), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-14, 2e-14, 3e-14)), reason="a millimetre body's order")
    # the coarse MRS's round trip: the near-contact law's hand-over (a sphere's form, on by default) written off
    experiment.attach(Drag.MRS(spacing=mm(0.3), blob=0.5, wall_kind="plane", near_contact=False), body="ball",
                      medium="water")
    return experiment


def mrs_graded() -> Experiment:
    """The round-trip case of graded sampling (design #296): a cylinder tilted above a plane wall, its rims graded
    (``grading="features"``) and its side facing the wall graded as a held contact, the wall's image integrated over the
    cells near it, moving along the plane and turning, at a coarse spacing."""
    experiment = Experiment("mrs_graded", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=(0.0, 0.0, 1.0), offset=mm(0.0))))
    experiment.add_medium(Medium("water", fills=floor.half_space))
    experiment.declare("water.viscosity", WATER, reason="water")
    tilt = math.radians(30.0)
    experiment.add_body(Body("can", dynamics=RigidInertial(),
                             parts={"shell": Part(Cylinder(radius=mm(0.5), length=mm(1.2), axis=(0.0, 0.0, 1.0)))},
                             initial=State(position=mm((0.0, 0.0, 1.0)),
                                           orientation=(math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0),
                                           velocity=m_per_s((1e-3, 0.0, -2e-4)),
                                           angular_velocity=rad_per_s((1.0, -2.0, 0.5)))))
    experiment.declare("can.mass", kg(1e-6), reason="a millimetre body's order")
    experiment.declare("can.inertia", kg_m2((1e-13, 1e-13, 1e-13)), reason="a millimetre body's order")
    experiment.attach(Drag.MRS(spacing=mm(0.4), blob=0.5, wall_kind="plane", wall_refinement=2, grading="features",
                               contact=(1.0, 0.0, -1.0), contact_gap=mm(0.3)), body="can", medium="water")
    return experiment


def prolate(a: float, b: float, mu: float = 1.0) -> np.ndarray:
    """The diagonal of a prolate spheroid's resistance about its centre, its axis along x: Kim & Karrila's ``X^A``,
    ``Y^A``, ``X^C``, ``Y^C`` with ``F = 6 pi mu a X^A U`` and ``T = 8 pi mu a^3 X^C Omega``, ``e`` the eccentricity."""
    e = math.sqrt(1 - b * b / (a * a))
    log = math.log((1 + e) / (1 - e))
    xa = 8 / 3 * e ** 3 / (-2 * e + (1 + e * e) * log)
    ya = 16 / 3 * e ** 3 / (2 * e + (3 * e * e - 1) * log)
    xc = 4 / 3 * e ** 3 * (1 - e * e) / (2 * e - (1 - e * e) * log)
    yc = 4 / 3 * e ** 3 * (2 - e * e) / (-2 * e + (1 + e * e) * log)
    return np.array([6 * math.pi * mu * a * xa] + [6 * math.pi * mu * a * ya] * 2
                    + [8 * math.pi * mu * a ** 3 * xc] + [8 * math.pi * mu * a ** 3 * yc] * 2)


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
