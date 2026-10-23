"""The family cases the family walk requires: one per registered interaction, naming its family, the law to probe,
its closed form and how its nodes are verified, or why it belongs to no family. An interaction registered without one
fails the walk, as one without a round-trip case fails the registry walk."""

import os
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mime.core import quaternion
from mime.interactions import Contact, Gravity, Magnetic
from mime.registry import Interaction
from mime.testing import NodeCase, Oracle
from mime.testing.toys import ToyDrag
from mime.units import mm, standard_gravity


@dataclass(frozen=True)
class FamilyCase:
    family: str = ""
    law: Callable[[], Interaction] | None = None
    oracle: Oracle | None = None
    node_case: NodeCase | None = None
    no_family: str = ""


def torque(read, settings):
    pose, field = np.asarray(read["pose"]), np.asarray(read["flux_density"])
    moment = np.asarray(quaternion.rotate(jnp.asarray(pose[3:]), jnp.asarray(settings["moment"])))
    return {"wrench": np.concatenate([np.zeros(3), np.cross(moment, field)])}


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


def linear(read, settings):
    return {"load": -settings["c"] * read["v"]}


TOY = "a toy with no physics, which exercises the build: the toy family has one member, the toy drag"
FAMILY_CASES: dict[tuple[str, str, str], FamilyCase] = {
    ("interaction", "magnetic.torque", "0.1.0"): FamilyCase("magnetic", Magnetic.Torque, torque),
    ("interaction", "gravity", "0.1.0"): FamilyCase("gravity", Gravity, weight_and_buoyancy),
    ("interaction", "contact", "0.1.0"): FamilyCase(
        "contact", lambda: Contact.Projection(margin=mm(0.1)), projected,
        NodeCase(no_node="a projection, built into the body's node")),
    ("interaction", "toy.drag", "0.2.0"): FamilyCase("toy", ToyDrag.Linear, linear),
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
