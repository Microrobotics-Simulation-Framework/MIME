"""A medium's drive as its waveform: the coefficients a body-force law delivers, which a field kind evaluates at the
instants it reads its drive (RK3 at its stage times) or responds to analytically (the prescribed flow), one delivery
for both rungs (F0's parked item 6: the waveform is the drive's, the field kind is what responds).

The vector is ``(along_x, along_y, along_z, steady, amplitude, frequency, phase)``: the force per unit mass at time
``t`` is ``along (steady + amplitude cos(2 pi frequency t + phase))`` (m/s^2, Hz, rad), each entry the law's leaf as
the run holds it, so a study moving one moves the field."""

from __future__ import annotations

from typing import Any

from jax import Array

from mime.core import jnp

#: The waveform's length: the unit direction, the steady part, and one harmonic's amplitude, frequency and phase.
WAVEFORM_SIZE = 7


def waveform(along: Any, steady: Any, amplitude: Any, frequency: Any, phase: Any) -> Array:
    """The coefficients as one vector, in the dtype of ``steady``."""
    a = jnp.asarray(along)
    s = jnp.asarray(steady)
    return jnp.concatenate([a.astype(s.dtype), jnp.stack([s, jnp.asarray(amplitude, dtype=s.dtype),
                                                          jnp.asarray(frequency, dtype=s.dtype),
                                                          jnp.asarray(phase, dtype=s.dtype)])])


def waveform_at(coefficients: Array, time: Any) -> Array:
    """The force per unit mass (m/s^2, shape (3,)) the waveform gives at ``time`` (s)."""
    c = jnp.asarray(coefficients)
    value = c[3] + c[4] * jnp.cos(2.0 * jnp.pi * c[5] * time + c[6])
    return c[:3] * value


__all__ = ["WAVEFORM_SIZE", "waveform", "waveform_at"]
