"""What a magnetic field provider's forward model gives (``mime.registry.ForwardModel``): the flux density and its
gradient at a world point, and the force and torque on a moment stated there. For a fixed moment the force, ``F_j =
m_i dB_i/dx_j``, and the torque, ``m x B``, are linear in the field and its gradient, so a model affine in its commands
stays affine in them."""

from __future__ import annotations

from typing import Any

import jax

from mime.registry import Signal

jnp: Any = jax.numpy

#: Where a magnetic model is read: a world point (m), and the moment its force and torque are stated on (A m^2), which
#: a call may leave out, its force and torque then not given.
POINT, MOMENT = "point", "moment"
INPUTS = {POINT: Signal("m", (3,)), MOMENT: Signal("A m^2", (3,))}
#: What it gives: the flux density (T), its gradient ``dB_i/dx_j`` (T/m, row by row), and the force (N) and torque
#: (N m) on the moment.
QUANTITIES = {"flux_density": Signal("T", (3,)), "gradient": Signal("T/m", (3, 3)), "force": Signal("N", (3,)),
              "torque": Signal("N m", (3,))}


def gradient_rows(gradient: Any) -> Any:
    """A current-free field's gradient, symmetric and trace-free, as its five independent entries ``[Gxx, Gxy, Gxz,
    Gyy, Gyz]`` along its first two axes (mag_manip's packing, ``allocate()``'s rows)."""
    return jnp.stack([gradient[0, 0], gradient[0, 1], gradient[0, 2], gradient[1, 1], gradient[1, 2]])


ROWS = {"gradient": gradient_rows}


def magnetic_outputs(field: Any, gradient: Any, x: Any, dtype: Any) -> dict[str, Any]:
    """The model's quantities from the field and its gradient at the point: with ``x``'s moment, the force on it,
    ``F_j = m_i G_ij``, and the torque, ``m x B``."""
    out = {"flux_density": field, "gradient": gradient}
    if MOMENT in x:
        moment = jnp.asarray(x[MOMENT], dtype=dtype)
        out["force"] = jnp.einsum("i,ij->j", moment, gradient)
        out["torque"] = jnp.cross(moment, field)
    return out


__all__ = ["INPUTS", "MOMENT", "POINT", "QUANTITIES", "ROWS", "gradient_rows", "magnetic_outputs"]
