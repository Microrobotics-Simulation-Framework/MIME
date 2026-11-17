"""The precision a run computes in, as its experiment declares it: ``float32`` or ``float64``.

A build applies its experiment's precision (:func:`apply_precision`): JAX's 64-bit types are then enabled, whatever the
precision, so that a float32 run may hold float64 where it declares it, and :func:`mime.core.run_dtype` reads the
declared precision, not JAX's flag. A node keeps the precision in force when it was made and steps in it
(``MimeNode``), so a run's steps keep their build's precision whatever was built after it. Where no build has applied
one (a node made on its own, outside an experiment), the run's dtype is JAX's flag's, as before precision was declared.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

import jax
import numpy as np

#: ``jax.numpy`` through ``Any``: jaxlib ships no stubs, so a strict checker reads its constructors as partly unknown.
_jnp: Any = jax.numpy

Precision = Literal["float32", "float64"]
PRECISIONS: tuple[Precision, ...] = ("float32", "float64")
#: The precision of an experiment that states none: float64, exact by default; float32 is declared.
DEFAULT_PRECISION: Precision = "float64"

#: The precision the last build applied, for what runs outside a node's step (a controller's code, a sensor).
_applied: list[Precision | None] = [None]
#: The precision of the node whose step or initial state is being traced, over the build's (``MimeNode``).
_NODE: ContextVar[Precision | None] = ContextVar("mime_node_precision", default=None)
#: A TEST-ONLY MIGRATION BRIDGE, removed in precision's P5, and no user's setting. Set to ``1``, an experiment that
#: states no precision is built as JAX's flag says, and the flag is left as it is: how MIME built before precision was
#: declared. ``tests/conftest.py`` sets it for the suite (and the processes its tests start) until its tests declare
#: their precision.
FROM_JAX_FLAG = "MIME_PRECISION_FROM_JAX_FLAG"


def unstated_precision() -> Precision | None:
    """What an experiment that states no precision is built in: :data:`DEFAULT_PRECISION`, or ``None`` where JAX's flag
    decides (:data:`FROM_JAX_FLAG`)."""
    return None if os.environ.get(FROM_JAX_FLAG) == "1" else DEFAULT_PRECISION


def declared_precision() -> Precision | None:
    """The precision in force: the stepping node's, else the last build's; ``None`` where no build applied one."""
    return _NODE.get() or _applied[0]


def apply_precision(precision: Precision | None) -> None:
    """Make ``precision`` the run's, with JAX's 64-bit types enabled; ``None`` applies none and leaves JAX's flag."""
    if precision is not None:
        cast(Any, jax.config).update("jax_enable_x64", True)
    _applied[0] = precision


@contextmanager
def node_precision(precision: Precision | None) -> Generator[None]:
    """For the block, ``precision`` in force over the build's: a node's own, while its step or state is traced."""
    token = _NODE.set(precision)
    try:
        yield
    finally:
        _NODE.reset(token)


@contextmanager
def in_float64() -> Generator[None]:
    """For the block, float64 computing: JAX's 64-bit types enabled and float64 the run's dtype, whatever the
    precision declared, for build-time work made in float64 whatever the run's (a held matrix, a search's points)."""
    with cast(Any, jax).enable_x64(True), node_precision("float64"):
        yield


def precision_dtype() -> np.dtype[Any]:
    """The floating dtype of the precision in force; JAX's flag's where none is declared."""
    declared = declared_precision()
    if declared is not None:
        return np.dtype(declared)
    return np.dtype(np.float64 if cast(Any, jax.config).jax_enable_x64 else np.float32)


def eye(n: int, dtype: Any) -> Any:
    """The ``n x n`` identity in ``dtype``, a constant: ``jax.numpy.eye`` builds it from an ``iota`` in JAX's default
    integer, int64 with 64-bit types enabled, inside a float32 run's step."""
    return _jnp.asarray(np.eye(n), dtype=dtype)


def diagonal_matrix(values: Any) -> Any:
    """The square matrix with ``values`` (a vector) on its diagonal and zeros elsewhere, as ``jax.numpy.diag`` makes it,
    with no ``iota``: the diagonal's indices are constants, int32."""
    index = np.arange(values.shape[0], dtype=np.int32)
    return _jnp.zeros((values.shape[0],) * 2, dtype=values.dtype).at[index, index].set(values)


def diagonal(matrix: Any) -> Any:
    """A square matrix's diagonal, as ``jax.numpy.diag`` takes it, gathered at int32 indices."""
    index = np.arange(matrix.shape[0], dtype=np.int32)
    return matrix[index, index]


__all__ = ["DEFAULT_PRECISION", "FROM_JAX_FLAG", "PRECISIONS", "Precision", "apply_precision", "declared_precision", "diagonal",
           "diagonal_matrix", "eye", "in_float64", "node_precision", "precision_dtype", "unstated_precision"]
