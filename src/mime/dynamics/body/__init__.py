"""The dynamics of bodies: the rigid inertial body, and the version of it kept registered."""

from mime.dynamics.body._previous import KEPT, RigidInertialExplicit
from mime.dynamics.body._rigid_inertial import RigidInertial

__all__ = ["KEPT", "RigidInertial", "RigidInertialExplicit"]
