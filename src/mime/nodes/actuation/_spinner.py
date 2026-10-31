"""A spinner: a rotor whose centre of mass is on an axis fixed in its host, carried by its absolute axial spin. The
piece any segment with a spinning body uses (a motor's rotor, a spinning magnet, a reaction wheel), so that each carries
it the same way.

The rotor's angular momentum about its axis is ``J Omega``, ``Omega`` its absolute spin (``e . omega_rotor``). Its
mass and its mean inertia across the axis ride on the host's link, grafted there at build. What is left of its inertia
across the axis is ``D``, the half difference of its two moments there, which turns with the rotor at twice its angle
on its stator (``D e = 0``, its trace zero): the host carries it at its instantaneous value, in its mass matrix, and the
spin feels it as ``J Omega_dot = tau + (omega x e) . D omega``, the torque ``1/2 omega^T (dD/dphi) omega`` of the host's
kinetic energy on the rotor's angle. For a rotor symmetric about its axis ``D`` is zero, and ``J Omega_dot`` is the
sum of the axial torques alone, exact with no term from the host's acceleration.

What crosses between the spinner and its host is velocities and inertias. In: the host's angular velocity ``omega`` at
the mount, this step's (the relative turn ``phi_dot = Omega - e . omega``, the bearing's ``b (Omega - e . omega)``, and
the torque of ``D``). Out, read by the host on its next step: the rotor's momentum about its axis, ``h = J Omega e``,
whose gyroscopic torque ``h x omega`` the host applies at its own velocity by a Cayley step (Crank-Nicolson on the
skew joint-space matrix ``G = sum_k J_k^T [h_k]x J_k``, which does no work); ``D`` at the step's start and its end;
and the drive's and the bearing's reactions with the axial reaction of ``D``'s torque. Pure JAX, in the dtype of the
spin."""

from __future__ import annotations

from typing import Any

from jax import Array

from mime.core import jnp


def spun(inertia: Any, damping: Any, spin: Array, torque: Array, axis: Array, host: Array, dt: Any) -> Array:
    """The spin at the end of the step: ``J Omega' = J Omega + dt (tau + b e . omega - b Omega')``, the bearing's
    damping on the relative speed taken at the end of the step. ``torque`` is everything else about the axis: the
    drive, the loads on the rotor and the torque of its inequality across the axis."""
    return (inertia * spin + dt * (torque + damping * jnp.dot(axis, host))) / (inertia + dt * damping)


def relative_rate(spin: Array, axis: Array, host: Array) -> Array:
    """The rotor's turn relative to its host, ``Omega - e . omega``: what turns its pose on its stator's."""
    return spin - jnp.dot(axis, host)


def rotor_rate(spin: Array, axis: Array, host: Array) -> Array:
    """The rotor's angular velocity in the world: its host's across the axis, and its own spin along it."""
    return host + relative_rate(spin, axis, host) * axis


def momentum(inertia: Any, spin: Array, axis: Array) -> Array:
    """The rotor's angular momentum about its axis, ``h = J Omega e``, which its host turns: the gyroscopic torque on
    the host is ``h x omega``, the reaction of ``omega x h``, the torque that turns ``h`` with the axis."""
    return inertia * spin * axis


def deviator_torque(deviator: Array, axis: Array, host: Array) -> Array:
    """The torque about the axis on a rotor unequal across it, ``(omega x e) . D omega``: ``1/2 omega^T (dD/dphi)
    omega``, ``D`` its half difference across the axis in the world at its angle, which turns at ``dD/dphi = [e]x D -
    D [e]x``."""
    return jnp.dot(jnp.cross(host, axis), deviator @ host)


def turned(deviator: Array, rotation: Array) -> Array:
    """A rotor's half difference across its axis, ``deviator`` in its own frame, in the frame ``rotation`` turns it
    into: ``R D R^T``."""
    return rotation @ deviator @ rotation.T


def skew(v: Array) -> Array:
    """``[v]x``, the matrix of ``v x .``, for each row of ``v`` (``(..., 3)`` to ``(..., 3, 3)``)."""
    zero = jnp.zeros_like(v[..., 0])
    return jnp.stack([jnp.stack([zero, -v[..., 2], v[..., 1]], axis=-1),
                      jnp.stack([v[..., 2], zero, -v[..., 0]], axis=-1),
                      jnp.stack([-v[..., 1], v[..., 0], zero], axis=-1)], axis=-2)


def gyroscopic_matrix(jacobians: Array, momenta: Array) -> Array:
    """``G = sum_k J_k^T [h_k]x J_k``, the joint-space matrix of the gyroscopic torques ``h_k x omega_k`` on a tree's
    frames, ``jacobians`` their angular velocity Jacobians ``(frames, 3, dofs)`` and ``momenta`` the spin momenta they
    carry ``(frames, 3)``, in one frame. Skew: the gyroscopic generalised force ``G q_dot`` does no work."""
    return jnp.einsum("kai,kab,kbj->ij", jacobians, skew(momenta), jacobians)


def cayley(mass: Array, gyroscopic: Array, velocity: Array, dt: Any) -> Array:
    """The velocity after the gyroscopic torque over the step, by Crank-Nicolson on ``M v_dot = G v``:
    ``(M - dt G / 2) v' = (M + dt G / 2) v``. With ``G`` skew the map is the Cayley transform of ``M^-1 G``, which
    keeps ``v^T M v`` to rounding at every step size: the term does no work, and the scheme adds none."""
    half = 0.5 * dt * gyroscopic
    return jnp.linalg.solve(mass - half, (mass + half) @ velocity)


__all__ = ["cayley", "deviator_torque", "gyroscopic_matrix", "momentum", "relative_rate", "rotor_rate", "skew", "spun",
           "turned"]
