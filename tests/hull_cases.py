"""Bodies held by their hulls' points (the contact 0.2.0): FL-9 in script A's tube, turned and placed so that its hull's
points are at the wall from the first step, which the registry walk, the golden outputs and the switch scan run; and
the placement that puts a hull's nearest point a given gap from the wall less the margin.

Every value is script A's (``script_a.py``), but the inertia, the parts' diagonal (FL-9's axial moment the smallest,
as review 7l-c's anisotropic arms have it), and the contact: ``Contact.Projection``, holding the hull."""

import math

import numpy as np
from script_a import EXTENT, MASS, MOMENT, VOLUME, WATER, fl9

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Contact, Gravity, Magnetic
from mime.scene import Scene, Tube, X, Z
from mime.scene._sampling import quadrature_arrays, quadrature_counts
from mime.things import Part
from mime.units import Hz, kg_m2, m, m_per_s, m_per_s2, mm, ms, mT, rad_per_s

#: FL-9's parts' inertia, its diagonal in the body's frame (``scripts/inertia_arms.py``'s ``DIAGONAL``).
DIAGONAL = kg_m2((1.3043197461318855e-10, 4.631130043564001e-10, 4.570054968582633e-10))
TUBE_RADIUS, MARGIN = 3.175e-3, 1e-4
FL9_SETTINGS = {"radius": 1.56e-3, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2}
#: The case's spacing of the hull's points: coarse, so that a step is cheap (957 points), not the default's.
SPACING = 4e-4
del EXTENT  # read by the contact 0.1.0 alone


def hull_points(spacing: float = SPACING, settings=FL9_SETTINGS) -> np.ndarray:
    """FL-9's hull points in its frame, as the contact holds them: the kind's quadrature at ``spacing`` on every
    patch."""
    kind = fl9()
    patches = tuple(kind.patches)
    points, _, _ = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, spacing, patches))
    return points


def rotation(quaternion) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def least_gap(points: np.ndarray, position, quaternion, radius: float = TUBE_RADIUS, margin: float = MARGIN) -> float:
    """The least gap of the hull's points at the pose to the tube's wall (along x through the origin) less the
    margin, m."""
    placed = np.asarray(position) + points @ rotation(quaternion).T
    return float(np.min(radius - margin - np.hypot(placed[:, 1], placed[:, 2])))


def placed_at(gap: float, quaternion, direction=(0.0, 0.0, -1.0), points: np.ndarray | None = None) -> np.ndarray:
    """The centre's position, from the tube's axis along ``direction`` (across the axis), at which the hull's nearest
    point is ``gap`` from the wall less the margin at the orientation ``quaternion``: by bisection on the offset."""
    held = hull_points() if points is None else points
    along = np.asarray(direction, dtype=np.float64)
    low, high = 0.0, TUBE_RADIUS
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if least_gap(held, middle * along, quaternion) > gap else (low, middle)
    return low * along


def turned(angle: float) -> tuple[float, float, float, float]:
    """The body's x axis turned by ``angle`` about the world's z, across the tube."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def hull_contact(name: str = "hull_contact", *, angle: float = math.radians(12.0), gap: float = 1e-7,
                 spacing: float = SPACING, part=None) -> Experiment:
    """FL-9 turned ``angle`` across script A's tube and lowered until its hull's nearest point is ``gap`` from the
    wall less the margin, under gravity and the field turning at 10 Hz, moving down into the wall at 1 mm/s: held at
    its hull's points from the first steps. ``part``, a kind in FL-9's place (placed as FL-9 would be), for a plan's
    checks."""
    quaternion = turned(angle)
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=m(TUBE_RADIUS),
                                                                    wall=mm(1.6), length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m(tuple(float(c) for c in placed_at(gap, quaternion))), orientation=quaternion,
        velocity=m_per_s((1e-3, 0.0, -1e-3)), angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(fl9() if part is None else part)}))
    field = experiment.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotating(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(10.0), "amplitude": mT(1.2)}))
    for path, value in (("umr.inertia", DIAGONAL), ("umr.mass", MASS), ("umr.buoyant_volume", VOLUME),
                        ("umr.magnetic_moment", MOMENT), ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))),
                        ("water.density", WATER)):
        experiment.declare(path, value, reason="script A's values, FL-9's parts' diagonal inertia")
    experiment.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=m(MARGIN), spacing=m(spacing)), body=umr, surface=tube.inner_surface)
    return experiment


#: A rod, a cylinder 1 mm in radius and 6 mm long along the body's x: a hull the contact holds by the cylinder's closed
#: form, each end's rim.
ROD_PART = {"radius": 1e-3, "length": 6e-3, "axis": (1.0, 0.0, 0.0)}


def rod_contact(name: str = "rod_contact") -> Experiment:
    """The rod in script A's tube, under gravity, turned 30 degrees across it and set 0.2 mm below the axis, so that one
    rim is at the wall less the margin within a few steps: the configuration the switch scan traces a cylinder's
    support in."""
    from mime.scene import Cylinder

    quaternion = turned(math.radians(30.0))
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=m(TUBE_RADIUS),
                                                                    wall=mm(1.6), length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    rod = Cylinder(radius=m(ROD_PART["radius"]), length=m(ROD_PART["length"]), axis=ROD_PART["axis"])
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -0.2)), orientation=quaternion, velocity=m_per_s((0.0, 0.0, -1e-3)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(rod)}))
    for path, value in (("umr.inertia", DIAGONAL), ("umr.mass", MASS), ("umr.buoyant_volume", VOLUME),
                        ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))), ("water.density", WATER)):
        experiment.declare(path, value, reason="script A's values, FL-9's parts' diagonal inertia")
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=m(MARGIN), spacing=m(SPACING)), body=umr, surface=tube.inner_surface)
    return experiment


def hull_node(kind, settings, spacing: float, *, groups=None, mass: float = 1e-5,
              inertia=(1e-11, 1e-11, 1e-11), pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dt: float = 5e-4,
              margin: float = MARGIN, radius: float = TUBE_RADIUS):
    """A body's node held by ``kind``'s hull points (its quadrature at ``spacing``) in a tube along x through the
    origin, as the contact 0.2.0 builds it; ``groups``, rows of the points' indices, in place of the node's own."""
    from mime.nodes.body import HULL, RigidInertialNode

    patches = tuple(kind.patches)
    held = {} if groups is None else {"hull_groups": np.asarray(groups, dtype=np.int32)}
    return RigidInertialNode(
        "umr", dt, mass=mass, inertia=inertia, initial_pose=pose, initial_twist=(0.0,) * 6, surface_shape="cylinder",
        margin=margin, velocity_rule="remove_outward", hull_kind=kind.name, hull_kind_version=kind.version,
        hull_patches=patches, hull_counts=quadrature_counts(kind, settings, spacing, patches),
        surface_centre=(0.0, 0.0, 0.0), surface_axis=(1.0, 0.0, 0.0), surface_radius=radius,
        **held, **{f"{HULL}{name}": value for name, value in settings.items()})


def kind_points(kind, settings, spacing: float) -> np.ndarray:
    """``kind``'s hull points in its frame at ``spacing``, in float64."""
    patches = tuple(kind.patches)
    points, _, _ = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, spacing, patches))
    return points
