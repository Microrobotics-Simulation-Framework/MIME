"""The experiments the plane wall kind is tested in: a sphere of radius ``a`` held a gap ``h`` above an unbounded plane
wall, the medium filling the half-space above it, with ``Drag.MRS`` in the plane wall kind and nothing else; and the
resistance the drag's node gives at a pose, in the units of the sphere-plane literature (forces in ``6 pi mu a``,
torques in ``8 pi mu a^3``, couplings in ``6 pi mu a^2``)."""

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from mime.build._maddening import node_state, parameter_leaves
from mime.dynamics import RigidInertial
from mime.experiment import BuiltExperiment, Experiment
from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.scene import Plane, Scene, Sphere, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, mPa_s, ms, rad_per_s

#: The sphere's radius (m) and the medium's viscosity (Pa s): a millimetre sphere in water.
RADIUS, VISCOSITY = 1e-3, 1e-3
#: The sphere's MRS settings as S4 ran them: blob 0.1 of the force spacing, its quadrature by the blob's rule.
BLOB = 0.1


def sphere_above_plane(spacing: float, gap: float, *, blob: float = BLOB, wall_refinement: int | None = None,
                       normal: tuple[float, float, float] = (0.0, 0.0, 1.0), offset: float = 0.0,
                       orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
                       name: str = "sphere_above_plane") -> Experiment:
    """The sphere (radius ``RADIUS``) with its surface ``gap`` (in radii) above the plane through ``offset`` (m) along
    ``normal``, its force spacing ``spacing`` (in radii), at rest and held by nothing."""
    unit = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    centre = (offset + RADIUS * (1.0 + gap)) * unit
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=normal, offset=m(offset))))
    experiment.add_medium(Medium("water", fills=floor.half_space))
    experiment.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(RADIUS)))},
                             initial=State(position=m(tuple(float(c) for c in centre)), orientation=orientation,
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    settings: dict[str, Any] = {"spacing": m(spacing * RADIUS), "blob": blob, "wall_kind": "plane"}
    if wall_refinement is not None:
        settings["wall_refinement"] = wall_refinement
    experiment.attach(Drag.MRS(**settings), body="ball", medium="water")
    return experiment


def drag_node(built: BuiltExperiment) -> MRSDragNode:
    node = built.graph.get_node("drag:ball")
    assert isinstance(node, MRSDragNode), type(node)
    return node


def stepped(built: BuiltExperiment, pose: Any = None, params: Any = None) -> dict[str, np.ndarray]:
    """The drag node's state after one step with the ball at ``pose`` (its start by default), in NumPy."""
    node = drag_node(built)
    held = node_state(built.graph, "drag:ball")
    if pose is None:
        pose = node_state(built.graph, "ball")["pose"]
    leaves = parameter_leaves(built.graph)["drag:ball"] if params is None else params
    out = node.update(held, {"pose": jnp.asarray(pose, dtype=held["resistance"].dtype)}, 1e-3, params=leaves)
    return {key: np.asarray(value) for key, value in out.items()}


def scaled(resistance: np.ndarray) -> dict[str, float]:
    """The sphere's entries in the literature's units, the normal along z (the plane's normal +Z): ``normal``
    (``R_zz / 6 pi mu a``), ``tangential`` (``R_xx / 6 pi mu a``), ``rotational`` (``R_(rx rx) / 8 pi mu a^3``), and
    ``coupling`` (``R_(x, ry) / 6 pi mu a^2``, the force along the wall per rotation about the axis across it)."""
    mu, a = VISCOSITY, RADIUS
    return {"normal": float(resistance[2, 2] / (6 * math.pi * mu * a)),
            "tangential": float(resistance[0, 0] / (6 * math.pi * mu * a)),
            "rotational": float(resistance[3, 3] / (8 * math.pi * mu * a ** 3)),
            "coupling": float(resistance[0, 4] / (6 * math.pi * mu * a ** 2))}
