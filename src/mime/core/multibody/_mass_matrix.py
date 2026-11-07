"""A body's 6 x 6 mass matrix about its centre of mass: the type the multibody library and a delivered added mass share.

Rows ``[linear; angular]``, in the body's frame. A rigid body's is :meth:`MassMatrix.rigid`, built exactly as
:func:`spatial_inertia` builds it with the centre at the origin; an added part is summed in (:meth:`MassMatrix.plus`);
:meth:`MassMatrix.in_world` turns it out by the body's rotation. The rigid body's node keeps its own arithmetic for its
step, so that no bit of a released program moves: this type is not on the step's path.

Definiteness is judged on the symmetric part scaled by its diagonal, so that a translation's kilograms and a rotation's
kilogram square metres compare: its least eigenvalue must clear :data:`DEFINITE_ULPS` units in the last place of the
matrix's dtype, the rounding of a 6 x 6 factorisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np
from jax import Array

from mime.core._node import jnp
from mime.core.multibody._dynamics import spatial_inertia

#: How far from symmetric a mass matrix may be, and how far above zero its scaled least eigenvalue must stand, in units
#: in the last place of its dtype: the rounding of the products that form it and of a 6 x 6 factorisation, not a
#: tolerance on the body.
SYMMETRIC_ULPS = 8
DEFINITE_ULPS = 16
_TRACER: type = cast(Any, jax).core.Tracer


def scaled_least_eigenvalue(matrix: np.ndarray) -> float:
    """The least eigenvalue of ``D^-1/2 sym(A) D^-1/2``, ``D`` the symmetric part's diagonal: one for any diagonal
    matrix with a positive diagonal, below zero for an indefinite one. A diagonal entry that is not above zero makes the
    matrix not definite, and the answer is that entry."""
    symmetric = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(symmetric)
    if np.any(diagonal <= 0.0):
        return float(np.min(diagonal))
    scale = np.sqrt(diagonal)
    return float(np.min(np.linalg.eigvalsh(symmetric / (scale[:, None] * scale[None, :]))))


@dataclass(frozen=True)
class MassMatrix:
    """A symmetric positive definite 6 x 6 mass matrix in the body's frame, rows ``[linear; angular]``. Checked at
    construction where its value is known: finite, symmetric to :data:`SYMMETRIC_ULPS` units in the last place, each
    entry's asymmetry judged against its row's and column's diagonal (a kilogram and a kilogram square metre do not
    compare), and definite, its scaled least eigenvalue above :data:`DEFINITE_ULPS` units in the last place
    (:func:`scaled_least_eigenvalue`). Built inside a traced function, it holds what it was built from unchecked."""

    matrix: Array

    def __post_init__(self) -> None:
        held = self.matrix
        if tuple(held.shape) != (6, 6):
            raise ValueError(f"a mass matrix is 6 x 6, not {tuple(held.shape)}")
        if isinstance(held, _TRACER):
            return
        values = np.asarray(held, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("a mass matrix is finite, and this one is not")
        eps = float(np.finfo(np.dtype(held.dtype)).eps)
        scale = np.sqrt(np.abs(np.diag(values)))
        apart = np.abs(values - values.T)
        asymmetry = float(np.max(np.where(apart > 0.0, apart / np.maximum(scale[:, None] * scale[None, :], 1e-300),
                                          0.0)))
        if asymmetry > SYMMETRIC_ULPS * eps:
            raise ValueError(f"a mass matrix is symmetric: this one's largest asymmetry is {asymmetry:.3g} of its "
                             f"diagonal's scale, above its rounding {SYMMETRIC_ULPS * eps:.3g}")
        least = scaled_least_eigenvalue(values)
        if least <= DEFINITE_ULPS * eps:
            raise ValueError(f"a mass matrix is positive definite: this one's scaled least eigenvalue is {least:.3g}, "
                             f"not above {DEFINITE_ULPS * eps:.3g}")

    @classmethod
    def rigid(cls, mass: Any, inertia: Any) -> MassMatrix:
        """A rigid body's, of ``mass`` and the symmetric 3 x 3 ``inertia`` about its centre of mass in its frame:
        ``[[m 1, 0], [0, J]]``, built as :func:`spatial_inertia` builds it with the centre at the origin."""
        tensor = jnp.asarray(inertia)
        return cls(spatial_inertia(jnp.asarray(mass, dtype=tensor.dtype), jnp.zeros(3, dtype=tensor.dtype), tensor))

    def plus(self, added: Any) -> MassMatrix:
        """This matrix with ``added``, a 6 x 6 in the same frame (an added mass's), summed in."""
        return MassMatrix(self.matrix + jnp.asarray(added, dtype=self.matrix.dtype))

    def in_world(self, rotation: Any) -> Array:
        """The matrix in the world, turned by the body's ``rotation`` (body to world): ``T M T^T``,
        ``T = diag(R, R)``."""
        turn = jnp.asarray(rotation, dtype=self.matrix.dtype)
        both = jnp.zeros((6, 6), dtype=self.matrix.dtype).at[:3, :3].set(turn).at[3:, 3:].set(turn)
        return both @ self.matrix @ both.T

    def kinetic_energy(self, twist: Any) -> Array:
        """``V^T M V / 2`` of a twist in the body's frame."""
        held = jnp.asarray(twist, dtype=self.matrix.dtype)
        return 0.5 * held @ self.matrix @ held


__all__ = ["DEFINITE_ULPS", "SYMMETRIC_ULPS", "MassMatrix", "scaled_least_eigenvalue"]
