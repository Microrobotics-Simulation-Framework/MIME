"""A spinner: a rotor symmetric about an axis fixed in its host, carried by its absolute axial spin. The piece any
segment with a spinning body uses (a motor's rotor, a spinning magnet, a reaction wheel), so that each carries it the
same way.

The rotor's angular momentum about its axis is ``J Omega``, ``Omega`` its absolute spin (``e . omega_rotor``), and
``J Omega_dot`` is the sum of the torques on it about the axis, with no term from its host's acceleration: exact for a
rotor symmetric about its axis whose centre of mass is on it. Its mass and its mean inertia across the axis ride on
the host's link, grafted there at build; what crosses between the spinner and its host is velocities, a step late:
the host's angular velocity ``omega`` at the mount, in (the relative turn ``phi_dot = Omega - e . omega``, and the
bearing's ``b (Omega - e . omega)``); the gyroscopic torque ``J Omega (e x omega)`` and the drive's and bearing's
reactions, out. Pure JAX, in the dtype of the spin."""

from __future__ import annotations

from typing import Any

from jax import Array

from mime.core import jnp


def spun(inertia: Any, damping: Any, spin: Array, torque: Array, axis: Array, host: Array, dt: Any) -> Array:
    """The spin at the end of the step: ``J Omega' = J Omega + dt (tau + b e . omega - b Omega')``, the bearing's
    damping on the relative speed taken at the end of the step. ``torque`` is everything else about the axis: the
    drive and the loads on the rotor."""
    return (inertia * spin + dt * (torque + damping * jnp.dot(axis, host))) / (inertia + dt * damping)


def relative_rate(spin: Array, axis: Array, host: Array) -> Array:
    """The rotor's turn relative to its host, ``Omega - e . omega``: what turns its pose on its stator's."""
    return spin - jnp.dot(axis, host)


def rotor_rate(spin: Array, axis: Array, host: Array) -> Array:
    """The rotor's angular velocity in the world: its host's across the axis, and its own spin along it."""
    return host + relative_rate(spin, axis, host) * axis


def gyroscopic(inertia: Any, spin: Array, axis: Array, host: Array) -> Array:
    """The torque the spinning rotor exerts on its host as the host turns its axis: ``J Omega (e x omega)``, the
    reaction of the torque ``J Omega (omega x e)`` that turns the rotor's angular momentum with its axis."""
    return inertia * spin * jnp.cross(axis, host)


__all__ = ["gyroscopic", "relative_rate", "rotor_rate", "spun"]
