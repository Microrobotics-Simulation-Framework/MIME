"""Contact constraint protocols for rigid body confinement.

Constraints are applied after position integration in RigidBodyNode.
They modify position and velocity to enforce geometric boundaries.

Usage:
    from mime.nodes.robot.constraints import CylindricalVesselConstraint

    constraint = CylindricalVesselConstraint(radius=0.0047, half_length=0.01)
    rigid = RigidBodyNode(..., constraint=constraint)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp


class ContactConstraint(Protocol):
    """Protocol for position/velocity constraints on rigid bodies."""

    def apply(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Constrain position and velocity.

        Called after position integration (pos = old_pos + V * dt).
        Returns corrected (position, velocity).

        Parameters
        ----------
        pos : (3,) float32
            Integrated position (may be outside boundary).
        vel : (3,) float32
            Current velocity.

        Returns
        -------
        (new_pos, new_vel) : tuple of (3,) float32
            Corrected position (inside boundary) and velocity
            (outward component damped on contact).
        """
        ...


@dataclass(frozen=True)
class CylindricalVesselConstraint:
    """Cylindrical vessel confinement: radial + axial clamping.

    The vessel is a cylinder centered at the origin with the given
    radius and half-length along the specified axis.

    Parameters
    ----------
    radius : float
        Vessel inner radius [m].
    half_length : float
        Half-length of the vessel along the axis [m].
    axis : int
        Vessel axis: 0=X, 1=Y, 2=Z. Default 2 (Z).
    """
    radius: float
    half_length: float
    axis: int = 2

    def apply(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Identify radial axes (the two axes perpendicular to vessel axis)
        ax = self.axis
        r_axes = [i for i in range(3) if i != ax]
        a0, a1 = r_axes

        # Radial confinement
        r = jnp.sqrt(pos[a0]**2 + pos[a1]**2 + 1e-30)
        scale = jnp.where(r > self.radius, self.radius / r, 1.0)
        pos = pos.at[a0].set(pos[a0] * scale)
        pos = pos.at[a1].set(pos[a1] * scale)

        # Damp outward radial velocity on contact
        r_hat_0 = pos[a0] / jnp.maximum(r, 1e-30)
        r_hat_1 = pos[a1] / jnp.maximum(r, 1e-30)
        v_radial = vel[a0] * r_hat_0 + vel[a1] * r_hat_1
        v_radial_out = jnp.maximum(v_radial, 0.0)
        vel = jnp.where(
            r > self.radius,
            vel.at[a0].set(vel[a0] - v_radial_out * r_hat_0)
                 .at[a1].set(vel[a1] - v_radial_out * r_hat_1),
            vel,
        )

        # Axial confinement
        pos = pos.at[ax].set(
            jnp.clip(pos[ax], -self.half_length, self.half_length),
        )
        vel = jnp.where(
            jnp.abs(pos[ax]) >= self.half_length,
            vel.at[ax].set(0.0),
            vel,
        )

        return pos, vel
