"""The family cases the family walk requires: one per registered interaction, with the law to probe, its closed form
and how its nodes are verified, or why it belongs to no family. The family is the one the entry declares
(``Interaction.family``). An interaction registered without a case fails the walk, as one without a round-trip case
fails the registry walk."""

import os
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from hydro_oracles import StokesSphere, surrogate_reference

from mime.data import asset
from mime.core import quaternion
from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
from mime.interactions.hydrodynamic import PointForceDrag
from mime.registry import Interaction
from mime.testing import NodeCase, Oracle, unit_quaternion
from mime.testing._mime_families import SCREW, TUBE_AXIS, TUBE_RADIUS
from mime.testing.toys import ToyDrag
from mime.units import mm, standard_gravity


@dataclass(frozen=True)
class FamilyCase:
    law: Callable[[], Interaction] | None = None
    oracle: Oracle | None = None
    node_case: NodeCase | None = None
    no_family: str = ""
    swap_with: Callable[[], Interaction] | None = None


def torque(read, settings):
    pose, field = np.asarray(read["pose"]), np.asarray(read["flux_density"])
    moment = np.asarray(quaternion.rotate(jnp.asarray(pose[3:]), jnp.asarray(settings["moment"])))
    return {"wrench": np.concatenate([np.zeros(3), np.cross(moment, field)])}


def force(read, settings):
    """``F = G^T m`` at the magnet's centroid, its lever ``l`` turned with the body: the wrench ``[F; l x F]``."""
    pose, gradient = np.asarray(read["pose"], dtype=np.float64), np.asarray(read["gradient"], dtype=np.float64)
    turn = np.asarray(quaternion.rotation_matrix(jnp.asarray(pose[3:], dtype=jnp.float64)))
    moment, lever = turn @ np.asarray(settings["moment"]), turn @ np.asarray(settings["magnet_centre"])
    pulled = gradient.T @ moment
    return {"wrench": np.concatenate([pulled, np.cross(lever, pulled)])}


#: The gravity rig's: standard gravity, along -z.
G = np.array([0.0, 0.0, -standard_gravity().value])


def weight_and_buoyancy(read, settings):
    buoyancy = -settings["fluid_density"] * settings["buoyant_volume"] * G
    lever = np.asarray(quaternion.rotate(jnp.asarray(np.asarray(read["pose"])[3:]),
                                         jnp.asarray(settings["centre_of_buoyancy"])))
    return {"wrench": np.concatenate([settings["mass"] * G + buoyancy, np.cross(lever, buoyancy)])}


#: The contact rig's tube: along x through the origin, 3 mm inner radius.
AXIS, RADIUS = np.array([1.0, 0.0, 0.0]), 3e-3


def projected(read, settings):
    """The step the body reached unconfined, projected radially back inside ``RADIUS - extent - margin`` with its
    outward velocity removed there: the cylinder's closed form."""
    pose, twist = np.array(read["pose"]), np.array(read["twist"])
    radial = pose[:3] - (pose[:3] @ AXIS) * AXIS
    r = float(np.linalg.norm(radial))
    beyond = r - RADIUS + settings["extent"] + settings["margin"]
    if beyond > 0 and r > 0:
        normal = radial / r
        pose[:3] = pose[:3] - beyond * normal
        twist[:3] = twist[:3] - max(float(twist[:3] @ normal), 0.0) * normal
    return {"pose": pose, "twist": twist}


def surrogate(read, settings):
    """The surrogate's formula evaluated again in float64 NumPy (``hydro_oracles``), in the hydrodynamic rig's tube,
    for the rig's body, whose shape its kind gives."""
    resistance, _ = surrogate_reference(
        np.asarray(read["pose"]), viscosity=settings["viscosity"], helix_wavenumber=SCREW.wavenumber.value,
        length=SCREW.length.value, radius=SCREW.radius.value, centre=(0.0, 0.0, 0.0), axis=TUBE_AXIS,
        tube_radius=TUBE_RADIUS.value, epsilon=settings.get("lubrication.epsilon"))
    return {"resistance": resistance, "background_load": np.zeros(6)}


def a_drag() -> Drag:
    """Script A's drag: the packaged surrogate, with lubrication."""
    return Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15)))


def kept_drag() -> PointForceDrag:
    """Script A's drag on the version kept registered, 0.3.0, whose surrogate rung is 0.4.0's."""
    return PointForceDrag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15)))


#: The drag node's drawn pose is a body's (inside a box a little larger than the tube, its quaternion of unit length):
#: in the default box of 1e4 every body is far beyond the wall, where the clamp and the lubrication's gap floor hold
#: every output still.
DRAG_NODE = NodeCase(options={
    "boundary_bounds": {"pose": (-4e-3, 4e-3)},
    "constrain_boundary": lambda given: {**given, "pose": unit_quaternion(given["pose"])}})


def linear(read, settings):
    return {"load": -settings["c"] * read["v"]}


TOY = "a toy with no physics, which exercises the build: the toy family has one member, the toy drag"
FAMILY_CASES: dict[tuple[str, str, str], FamilyCase] = {
    ("interaction", "magnetic.torque", "0.1.0"): FamilyCase(Magnetic.Torque, torque),
    ("interaction", "magnetic.force", "0.1.0"): FamilyCase(Magnetic.Force, force),
    ("interaction", "gravity", "0.1.0"): FamilyCase(Gravity, weight_and_buoyancy),
    ("interaction", "contact", "0.1.0"): FamilyCase(
        lambda: Contact.Projection(margin=mm(0.1)), projected,
        NodeCase(no_node="a projection, built into the body's node")),
    ("interaction", "drag", "0.3.0"): FamilyCase(kept_drag, surrogate, DRAG_NODE, swap_with=StokesSphere),
    ("interaction", "drag", "0.4.0"): FamilyCase(a_drag, surrogate, DRAG_NODE, swap_with=StokesSphere),
    ("interaction", "toy.drag", "0.2.0"): FamilyCase(ToyDrag.Linear, linear),
    ("interaction", "toy.spring", "0.1.0"): FamilyCase(no_family=TOY),
    ("interaction", "toy.lift", "0.1.0"): FamilyCase(no_family=TOY),
    ("interaction", "toy.hold", "0.1.0"): FamilyCase(no_family=TOY),
    ("interaction", "toy.couple", "0.1.0"): FamilyCase(no_family=TOY),
    ("interaction", "toy.twist", "0.1.0"): FamilyCase(no_family="refused at plan time: it delivers into a reserved "
                                                                  "interface"),
    ("interaction", "labx.nudge", "0.1.0"): FamilyCase(
        no_family="the lab fixture's entry, which exercises registration from outside MIME"),
}

if os.environ.get("MIME_CI_PLANT_FAILURE") == "1":
    # The control that the walk fails CI when an interaction has no family case: the planted run drops gravity's.
    del FAMILY_CASES[("interaction", "gravity", "0.1.0")]
