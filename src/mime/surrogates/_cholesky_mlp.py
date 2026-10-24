"""The Cholesky MLP: a multilayer perceptron whose 21 outputs are the lower triangle of a Cholesky factor ``L``, its
diagonal through a softplus, so that the 6 x 6 resistance it predicts, ``R = L L^T``, is symmetric positive definite
by construction. Ported from the 0.2 line's ``mime.surrogates.cholesky_mlp`` (the training script stays there)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.core import jnp

#: ``jax.nn``, whose hints name types from jaxlib's compiled core, which ships no stubs.
_nn: Any = cast(Any, jax).nn

#: Added to every diagonal entry of ``L`` after its softplus, as the 0.2 line's training did: ``L``'s diagonal is
#: never below it, so ``R`` stays positive definite even where the network's output saturates.
DIAGONAL_FLOOR = 1e-5

#: The lower triangle's entries in the order the network emits them: row by row, ``(i, j)`` for ``j <= i``.
TRIANGLE = tuple((i, j) for i in range(6) for j in range(i + 1))

#: ``TRIANGLE``, as constants for ``cholesky_resistance``: which of the 21 entries are on the diagonal, each cell of
#: the 6 x 6 factor's position among the 21 (the upper triangle's cells point at entry 0, and are zeroed), and which
#: cells are in the lower triangle. The positions are ``int32`` whatever the mode: as ``int64``, indexing under float32
#: and then under x64 in one process raised an int32/int64 dtype mismatch.
_ROWS, _COLUMNS = np.array(TRIANGLE).T
_ON_DIAGONAL = _ROWS == _COLUMNS
_ENTRY_AT = np.zeros((6, 6), dtype=np.int32)
_ENTRY_AT[_ROWS, _COLUMNS] = np.arange(len(TRIANGLE))
_IN_LOWER_TRIANGLE = np.zeros((6, 6), dtype=bool)
_IN_LOWER_TRIANGLE[_ROWS, _COLUMNS] = True


@dataclass(frozen=True)
class CholeskyMLP:
    """Trained weights and the normalisation of the network's inputs and outputs, as stored: each layer's ``(W, b)``
    (SiLU on every layer but the last), the features' mean and standard deviation, the Cholesky entries' mean and
    standard deviation, whether the features carry ``nu^2`` and ``kappa^2`` after the six base ones, and the
    reference radius (mm) the training non-dimensionalised lengths by."""

    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    entry_mean: np.ndarray
    entry_std: np.ndarray
    squared_features: bool
    reference_radius_mm: float

    def arrays(self) -> dict[str, np.ndarray]:
        """Every array, by the name the stored file gives it."""
        found = {"X_mean": self.feature_mean, "X_std": self.feature_std, "L_mean": self.entry_mean,
                 "L_std": self.entry_std}
        for index, (w, b) in enumerate(self.layers):
            found[f"W{index}"], found[f"b{index}"] = w, b
        return found


def read_cholesky_mlp(data: bytes) -> CholeskyMLP:
    """The network stored in ``data``, an ``.npz`` file as the 0.2 line's training wrote it."""
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        count = int(stored["n_layers"])
        layers = tuple((np.asarray(stored[f"W{i}"]), np.asarray(stored[f"b{i}"])) for i in range(count))
        squared = bool(stored["use_squared_features"]) if "use_squared_features" in stored.files else True
        return CholeskyMLP(layers, np.asarray(stored["X_mean"]), np.asarray(stored["X_std"]),
                           np.asarray(stored["L_mean"]), np.asarray(stored["L_std"]), squared,
                           float(stored["R_CYL_UMR"]))


def mlp_forward(layers: tuple[tuple[Any, Any], ...], x: Any) -> Any:
    """The network's raw outputs for normalised features ``x``: SiLU on every layer but the last, which is linear."""
    for w, b in layers[:-1]:
        x = _nn.silu(jnp.dot(x, w) + b)
    w, b = layers[-1]
    return jnp.dot(x, w) + b


def cholesky_resistance(entries: Any) -> Any:
    """``R = L L^T`` from the 21 entries of ``L``'s lower triangle, its diagonal through ``softplus(x) + 1e-5``. ``L`` is
    built whole-array, one gather and two selects, where one write per entry is one operation each in a compiled step."""
    entries = jnp.asarray(entries)
    diagonal_lifted = jnp.where(_ON_DIAGONAL, _nn.softplus(entries) + DIAGONAL_FLOOR, entries)
    factor = jnp.where(_IN_LOWER_TRIANGLE, diagonal_lifted[_ENTRY_AT], jnp.zeros((), entries.dtype))
    return factor @ factor.T


def predict_resistance(network: CholeskyMLP, features: Any) -> Any:
    """The non-dimensional resistance the network predicts for raw ``features``, in their dtype: normalise, evaluate,
    denormalise the Cholesky entries, rebuild ``R``."""
    dtype = jnp.asarray(features).dtype
    cast = tuple((jnp.asarray(w, dtype=dtype), jnp.asarray(b, dtype=dtype)) for w, b in network.layers)
    normalised = (features - jnp.asarray(network.feature_mean, dtype=dtype)) / jnp.asarray(network.feature_std,
                                                                                           dtype=dtype)
    raw = mlp_forward(cast, normalised)
    entries = raw * jnp.asarray(network.entry_std, dtype=dtype) + jnp.asarray(network.entry_mean, dtype=dtype)
    return cholesky_resistance(entries)
