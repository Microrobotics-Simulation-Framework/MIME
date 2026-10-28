"""MIME's own families as conformance suites: magnetic, gravity, contact and hydrodynamic, each with its rig (the
smallest experiment holding one of its laws), its sign rule, the values that stand for a material's properties, and the
constraints a drawn reading must meet to be one (a pose's quaternion has unit length; a resistance is positive
semi-definite).

A lab calls them from its own tests: ``assert_conforms(MyTorque(), "magnetic", oracle=...)``. The swap case waits for
each family's second member: a swap with itself could not fail. The hydrodynamic family's has one, which MIME's tests
register and pass as ``swap_with=``: a Stokes sphere, the family's analytic oracle, which is not public."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from mime.core import jnp, quaternion, shape_named
from mime.drives import Constant
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.fields import PointDipoleField, UniformRotatingField
from mime.registry import Interaction
from mime.scene import Scene, SolidCylinder, Tube, Z
from mime.testing._families import CaseKind, Family, Response, SignRule, register_family
from mime.things import Apparatus, Body, Medium, Part, PermanentMagnet, State, Structure
from mime.units import (
    A_m2,
    Hz,
    dimensionless,
    kg,
    kg_m2,
    kg_per_m3,
    m,
    m3,
    m_per_s,
    mm,
    mPa_s,
    ms,
    mT,
    rad_per_s,
    standard_gravity,
)
from mime.vocabulary import APPLIED_WRENCH, FLUX_DENSITY, GRADIENT, POSE, RESISTANCE, TWIST

PROBE = "probe"
#: The rigs' body: a millimetre-scale magnetic swimmer's mass, inertia and extent, round numbers of its order.
MASS, INERTIA, EXTENT = kg(8.5e-5), kg_m2((1.0e-10, 1.2e-10, 0.8e-10)), mm(1.5)


def unit_quaternion(pose: Any) -> Any:
    """A drawn pose with its quaternion scaled to unit length (the identity where it is zero), in its own dtype. The
    quaternion is first divided by its largest component: Hypothesis shrinks towards tiny draws, whose squares
    underflow into the subnormals and leave a norm off by 1e-4."""
    held = np.asarray(pose)
    q = held[3:].astype(np.float64)
    largest = float(np.max(np.abs(q)))
    if largest == 0.0:
        return np.concatenate([held[:3], np.array([1.0, 0.0, 0.0, 0.0])]).astype(held.dtype)
    scaled = q / largest
    return np.concatenate([held[:3], scaled / float(np.linalg.norm(scaled))]).astype(held.dtype)


def positive_semidefinite(matrix: Any) -> Any:
    """A drawn resistance as ``A A^T``, positive semi-definite, as every resistance is."""
    held = np.asarray(matrix, dtype=np.float64)
    return (held @ held.T).astype(np.asarray(matrix).dtype)


def millimetre_lever(wrench: Any) -> Any:
    """A drawn wrench ``[F; tau]`` with its torque scaled by a millimetre, a lever of the rigs' body's size: drawn in
    one box, a torque as large as a force turns a 1e-10 kg m^2 body through 1e8 rad in a step, where float32 has no
    digits left to turn the world with."""
    held = np.asarray(wrench)
    return np.concatenate([held[:3], 1e-3 * held[3:]]).astype(held.dtype)


def _experiment(name: str) -> Experiment:
    return Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(1))


def _body(experiment: Experiment, *, position: Any = m((0.0, 0.0, 0.0))) -> None:
    experiment.add_body(Body(PROBE, dynamics=RigidInertial(), initial=State(
        position=position, orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare(f"{PROBE}.mass", MASS, reason="the rig's body")
    experiment.declare(f"{PROBE}.inertia", INERTIA, reason="the rig's body")


#: The magnetic rig's fixed magnet, for a law that reads a field's gradient: de Jongh's rotor magnet, 18.89 A m^2, 35
#: mm across and 20 mm long, magnetised along z, 10 cm above the body.
MAGNET_MOMENT, MAGNET_HEIGHT = A_m2(18.89), 0.1


def magnetic_rig(law: Interaction) -> Experiment:
    """A body with a permanent moment in a field: for a law that reads a field's gradient, a permanent magnet fixed
    above it (a point dipole); otherwise a uniform field turning about z."""
    experiment = _experiment("magnetic_rig")
    _body(experiment)
    experiment.declare(f"{PROBE}.magnetic_moment", A_m2((0.0, 1.7e-3, 0.4e-3)), reason="the rig's magnet")
    if any(read.interface == GRADIENT.name for read in law.reads.values()):
        experiment.declare(f"{PROBE}.magnet_centre", m((2e-4, -1e-4, 3e-4)), reason="the rig's magnet, off the "
                                                                                     "centre of mass")
        experiment.add_apparatus(Apparatus(
            "stand", parts={"magnet": Part(SolidCylinder(radius=mm(17.5), length=mm(20.0), axis=(0.0, 0.0, 1.0)),
                                           at=m((0.0, 0.0, MAGNET_HEIGHT)))},
            magnets={"magnet": PermanentMagnet(direction=(0.0, 0.0, 1.0))},
            fields={"magnetic": PointDipoleField(part="magnet")}))
        experiment.declare("stand.magnetic_moment", MAGNET_MOMENT, reason="the rig's source magnet")
        experiment.attach(law, body=PROBE, field="stand.magnetic")
        return experiment
    experiment.add_apparatus(Apparatus("coils", fields={
        "magnetic": UniformRotatingField(rotation_axis=(0.0, 0.0, 1.0), initial_direction=(1.0, 0.0, 0.0))},
        commands={"frequency": Constant(Hz(5.0)), "amplitude": Constant(mT(1.0))}))
    experiment.attach(law, body=PROBE, field="coils.magnetic")
    return experiment


def gravity_rig(law: Interaction) -> Experiment:
    """A body denser than the medium it sits in, its centre of buoyancy off its centre of mass, so the law gives
    the couple too."""
    experiment = _experiment("gravity_rig")
    _body(experiment)
    experiment.add_medium(Medium("bath"))
    experiment.declare(f"{PROBE}.buoyant_volume", m3(6.0e-8), reason="the rig's body")
    experiment.declare(f"{PROBE}.centre_of_buoyancy", m((1.0e-4, 0.0, -2.0e-4)), reason="the rig's body")
    experiment.declare("bath.density", kg_per_m3(1000.0), reason="the rig's medium")
    experiment.attach(law, body=PROBE, medium="bath")
    return experiment


def contact_rig(law: Interaction) -> Experiment:
    """A body in a tube along x, near its wall."""
    experiment = _experiment("contact_rig")
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(3.0), wall=mm(1.0),
                                                             length=mm(50.0))))
    _body(experiment, position=mm((0.0, 0.0, -1.2)))
    experiment.declare(f"{PROBE}.extent", EXTENT, reason="the rig's body")
    experiment.attach(law, body=PROBE, surface="tube.inner_surface")
    return experiment


#: The hydrodynamic rig's tube, along x through the origin with script A's 1/4" lumen, and its body's shape: de
#: Jongh's FL-9 screw (nu 2.33, 7.47 mm long, 1.56 mm reference radius), in water at 1 mPa s.
TUBE_AXIS, TUBE_RADIUS = (1.0, 0.0, 0.0), mm(3.175)
SCREW = {"helix_wavenumber": dimensionless(2.33), "length": mm(7.47), "radius": mm(1.56)}
WATER_VISCOSITY = mPa_s(1.0)


#: How far from the rig's tube axis its body is drawn: from 0.2 mm to 1.0 mm, the body's outer radius (2.07 mm) then
#: 0.1 mm or more from the wall, as script A's contact holds it.
NEAREST_THE_AXIS, FURTHEST_FROM_THE_AXIS = 2e-4, 1e-3
_DRAWN_ACROSS = 3e-3 * math.sqrt(2.0)


def along_the_tube(quaternion_drawn: np.ndarray) -> np.ndarray:
    """An orientation with the body's x axis (a screw body's own, along which the surrogate takes it) along the rig's
    tube (x), either way, tilted from it by at most 0.5 rad about z and turned about the tube's axis, each read from
    the drawn quaternion's components: the orientations a body in a tube has, where the surrogate's end, which jumps
    where the body lies across the tube, is constant."""
    w, x, y, z = (float(c) for c in quaternion_drawn)
    turn = 2.0 * math.atan2(x, w)
    tilt = 0.5 * z
    flipped = y < 0.0
    about_x = np.array([math.cos(turn / 2), math.sin(turn / 2), 0.0, 0.0])
    about_z = np.array([math.cos(tilt / 2), 0.0, 0.0, math.sin(tilt / 2)])
    half_turn = np.array([0.0, 0.0, 0.0, 1.0]) if flipped else np.array([1.0, 0.0, 0.0, 0.0])

    def product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = q
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    return product(about_x, product(about_z, half_turn))


def in_the_lumen(pose: Any) -> Any:
    """A drawn pose with its quaternion of unit length, the body along the tube (:func:`along_the_tube`), and its
    position moved across the rig's tube (along x) to between 0.2 and 1.0 mm from the axis, in the direction it was drawn in (along y where it was drawn on the axis).
    These are states a body confined as in script A can have and where the surrogate is smooth. On the axis the
    surrogate is discontinuous (it is not symmetric there), and near the wall its clamp and its lubrication's gap floor
    hold it: both are tested on purpose, in MIME's tests of the drag, not drawn here."""
    held = np.asarray(unit_quaternion(pose))
    held = np.concatenate([held[:3], along_the_tube(held[3:].astype(np.float64)).astype(held.dtype)])
    position = held[:3].astype(np.float64)
    across = position[1:]
    distance = float(np.hypot(across[0], across[1]))
    direction = across / distance if distance > 0.0 else np.array([1.0, 0.0])
    radius = NEAREST_THE_AXIS + (FURTHEST_FROM_THE_AXIS - NEAREST_THE_AXIS) * min(distance / _DRAWN_ACROSS, 1.0)
    moved = np.concatenate([position[:1], radius * direction, held[3:].astype(np.float64)])
    return moved.astype(held.dtype)


def hydrodynamic_rig(law: Interaction) -> Experiment:
    """A screw-shaped body 1 mm off the axis of a water-filled tube along x."""
    experiment = _experiment("hydrodynamic_rig")
    experiment.add_structure(Structure("tube", geometry=Tube(axis=TUBE_AXIS, inner_radius=TUBE_RADIUS, wall=mm(1.6),
                                                             length=mm(100.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen"))
    _body(experiment, position=mm((0.0, 0.0, -1.0)))
    for name, value in SCREW.items():
        experiment.declare(f"{PROBE}.{name}", value, reason="the rig's body: de Jongh's FL-9")
    experiment.declare("water.viscosity", WATER_VISCOSITY, reason="the rig's medium")
    experiment.attach(law, body=PROBE, medium="water")
    return experiment


def _moment_in_world(response: Response) -> np.ndarray:
    pose = np.asarray(response.reading("body", POSE.name))
    return np.asarray(quaternion.rotate(jnp.asarray(pose[3:]), jnp.asarray(response.settings["moment"],
                                                                            dtype=pose.dtype)))


def _magnetic_sign(response: Response) -> bool:
    """A law that reads the field turns the moment toward it, ``tau . (m x B) >= 0``; a law that reads its gradient
    pulls the moment up the gradient of its alignment with the field, ``F . (G^T m) >= 0``, so an aligned moment is
    pulled toward the stronger field. A law that reads neither holds to neither, and fails."""
    reads = {interface for _, interface in response.reads_at}
    wrench = np.asarray(response.delivery("body", APPLIED_WRENCH.name))
    moment = _moment_in_world(response)
    if FLUX_DENSITY.name in reads:
        aligning = np.cross(moment, np.asarray(response.reading("field", FLUX_DENSITY.name)))
        return float(np.dot(wrench[3:], aligning)) >= -1e-6 * float(np.linalg.norm(wrench[3:]) *
                                                                    np.linalg.norm(aligning))
    if GRADIENT.name in reads:
        uphill = np.asarray(response.reading("field", GRADIENT.name)).T @ moment
        return float(np.dot(wrench[:3], uphill)) >= -1e-6 * float(np.linalg.norm(wrench[:3]) * np.linalg.norm(uphill))
    return False


def _weight_follows_the_excess_density(response: Response) -> bool:
    s = response.settings
    excess = float(cast(float, s["mass"])) - float(cast(float, s["fluid_density"])) * float(
        cast(float, s["buoyant_volume"]))
    force = np.asarray(response.delivery("body", APPLIED_WRENCH.name))[:3]
    gravity = np.asarray(response.constants["gravity"], dtype=np.float64)
    along = float(np.dot(force, gravity))
    across = force - (along / float(np.dot(gravity, gravity))) * gravity
    resolution = 8 * float(np.finfo(force.dtype).eps)
    return (along * excess >= -resolution * abs(along) * abs(excess)
            and float(np.linalg.norm(across)) <= resolution * float(np.linalg.norm(force)))


def _nothing_leaves_where_projected(response: Response) -> bool:
    """Where the step would have taken the body past its surface (less its extent and margin), the body it delivers
    moves no further out: its velocity has no outward part along the surface's normal there, beyond the rounding of
    the velocity it reached unconfined.

    That rounding is judged in float64: in float32 the square of a speed below about 3e-23 m/s underflows, and a
    norm taken there is zero. Its floor is the dtype's smallest normal number, below which XLA flushes every result
    to zero: the rule's dot product, its correction and its subtraction can each lose up to that number on each axis,
    under seven of it in all."""
    constants: Mapping[str, Any] = response.constants
    shape = shape_named(str(constants["surface_shape"]))
    parameters = {name: constants[f"surface_{name}"] for name in shape.parameter_names}
    inset = float(constants["extent"]) + float(constants["margin"])
    reached = jnp.asarray(response.reading("body", POSE.name))[:3]
    if float(shape.signed_distance(parameters, reached)) + inset <= 0:
        return True
    pose = np.asarray(response.delivery("body", POSE.name))
    velocity = np.asarray(response.delivery("body", TWIST.name))[:3]
    normal = np.asarray(shape.outward_normal(parameters, jnp.asarray(pose[:3])), dtype=np.float64)
    before = np.asarray(response.reading("body", TWIST.name))[:3].astype(np.float64)
    info = np.finfo(velocity.dtype)
    resolution = 8 * float(info.eps) * float(np.linalg.norm(before)) + 8 * float(info.tiny)
    return float(np.dot(velocity.astype(np.float64), normal)) <= resolution


def _resistance_dissipates(response: Response) -> bool:
    """The resistance is symmetric, block by block to its dtype's rounding, and positive definite: ``V^T R V > 0``
    for every nonzero twist. Definiteness is judged on ``D^-1/2 R D^-1/2``, ``D`` its diagonal, which has the same
    sign as ``R`` and no units, so a rotational block of 1e-10 N m s is not lost beside a translational one of 1e-3."""
    held = np.asarray(response.delivery("body", RESISTANCE.name))
    resistance = held.astype(np.float64)
    resolution = 64 * float(np.finfo(held.dtype).eps)
    halves = (slice(0, 3), slice(3, 6))
    for rows in halves:
        for columns in halves:
            block, mirrored = resistance[rows, columns], resistance[columns, rows].T
            if float(np.max(np.abs(block - mirrored))) > resolution * float(np.max(np.abs(block), initial=0.0)):
                return False
    diagonal = np.diag(resistance)
    if not bool(np.all(diagonal > 0.0)):
        return False
    scaled = resistance / np.sqrt(np.outer(diagonal, diagonal))
    return float(np.min(np.linalg.eigvalsh((scaled + scaled.T) / 2))) > resolution


_POSE_AND_RESISTANCE = {POSE.name: unit_quaternion, RESISTANCE.name: positive_semidefinite}
#: The contact rig's drawn resistance is of a millimetre body's order in water (``A A^T`` up to 6e-10): a box of
#: 1e-2, as for a law's inputs, couples rotation to translation at 1e7 times a real body's rotational resistance, and
#: ``M/dt + R`` is then too ill-conditioned for float32 to turn the world without changing the answer.

MAGNETIC = register_family(Family(
    name="magnetic", rig=magnetic_rig,
    cases=(CaseKind.SIGN, CaseKind.FRAME, CaseKind.UNITS, CaseKind.MATERIAL_REACH, CaseKind.ORACLE,
           CaseKind.GRADIENTS, CaseKind.NODES),
    sign=SignRule("the field turns the body's moment toward it (tau . (m x B) is not negative), and its gradient pulls "
                  "the moment up the gradient of its alignment (F . (G^T m) is not negative): an aligned moment is "
                  "pulled toward the stronger field", _magnetic_sign, reads=(("body", POSE.name),),
                  delivers=(("body", APPLIED_WRENCH.name),)),
    material=("moment",),
    reading_bounds={FLUX_DENSITY.name: (-1e-2, 1e-2), GRADIENT.name: (-1.0, 1.0), POSE.name: (-1.0, 1.0)},
    constrain=_POSE_AND_RESISTANCE,
))

GRAVITY = register_family(Family(
    name="gravity", rig=gravity_rig,
    cases=(CaseKind.SIGN, CaseKind.FRAME, CaseKind.UNITS, CaseKind.MATERIAL_REACH, CaseKind.ORACLE,
           CaseKind.GRADIENTS, CaseKind.NODES),
    sign=SignRule("weight minus buoyancy lies along gravity, pointing with it when the body is heavier than the fluid "
                  "it displaces", _weight_follows_the_excess_density, delivers=(("body", APPLIED_WRENCH.name),)),
    material=("mass", "buoyant_volume", "fluid_density", "centre_of_buoyancy"),
    reading_bounds={POSE.name: (-1.0, 1.0)},
    constrain=_POSE_AND_RESISTANCE,
    frame_constants=("gravity",),
))

CONTACT = register_family(Family(
    name="contact", rig=contact_rig,
    cases=(CaseKind.SIGN, CaseKind.FRAME, CaseKind.UNITS, CaseKind.MATERIAL_REACH, CaseKind.ORACLE, CaseKind.NODES),
    sign=SignRule("where the body is projected back, it moves no further out", _nothing_leaves_where_projected,
                  reads=(("body", POSE.name), ("body", TWIST.name)),
                  delivers=(("body", POSE.name), ("body", TWIST.name))),
    material=("extent",),
    reading_bounds={POSE.name: (-5e-3, 5e-3), TWIST.name: (-1.0, 1.0), APPLIED_WRENCH.name: (-1e-3, 1e-3),
                    RESISTANCE.name: (-1e-5, 1e-5), "body.background_load": (-1e-3, 1e-3)},
    constrain={**_POSE_AND_RESISTANCE, APPLIED_WRENCH.name: millimetre_lever,
               "body.background_load": millimetre_lever},
    frame_constants=("surface_centre", "surface_axis"),
))

HYDRODYNAMIC = register_family(Family(
    name="hydrodynamic", rig=hydrodynamic_rig,
    cases=(CaseKind.SIGN, CaseKind.FRAME, CaseKind.UNITS, CaseKind.MATERIAL_REACH, CaseKind.ORACLE,
           CaseKind.GRADIENTS, CaseKind.SWAP, CaseKind.NODES),
    sign=SignRule("the resistance dissipates: V^T R V > 0 for every nonzero twist, R symmetric positive definite",
                  _resistance_dissipates, delivers=(("body", RESISTANCE.name),)),
    material=("viscosity",),
    linear_in=("viscosity",),
    reading_bounds={POSE.name: (-3e-3, 3e-3)},
    constrain={POSE.name: in_the_lumen},
    frame_constants_held=("vessel_centre", "vessel_axis"),
    uniform_along=((2e-3, 0.0, 0.0), (-7e-3, 0.0, 0.0)),
))
"""The hydrodynamic family: a medium's resistance to a body's motion. Its laws read the body's pose and deliver a
resistance and a background load. Equal and opposite does not apply: the tube is fixed, and a medium at rest takes no
reaction its dynamics would carry. Frame: turning the whole setup, the tube's axis and the body's pose together, turns
the resistance block by block, and moving the body along a straight tube's axis changes nothing."""

__all__ = ["CONTACT", "EXTENT", "FURTHEST_FROM_THE_AXIS", "GRAVITY", "HYDRODYNAMIC", "INERTIA", "MAGNETIC",
           "MAGNET_HEIGHT", "MAGNET_MOMENT", "MASS",
           "NEAREST_THE_AXIS", "PROBE", "SCREW", "TUBE_AXIS", "TUBE_RADIUS", "WATER_VISCOSITY", "contact_rig",
           "gravity_rig", "hydrodynamic_rig", "in_the_lumen", "magnetic_rig", "millimetre_lever",
           "positive_semidefinite", "unit_quaternion"]
