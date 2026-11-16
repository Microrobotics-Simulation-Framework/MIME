"""The dynamics of bodies: the rigid inertial body, and the versions of it kept registered."""

from mime.dynamics.body._previous import KEPT
from mime.dynamics.body._rigid_inertial import RigidInertial

__all__ = ["KEPT", "RigidInertial"]
