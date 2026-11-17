"""The precision a run computes in, as its experiment declares it: ``float32`` or ``float64``.

A build applies its experiment's precision (:func:`apply_precision`): JAX's 64-bit types are then enabled, whatever the
precision, so that a float32 run may hold float64 where it declares it, and :func:`mime.core.run_dtype` reads the
declared precision, not JAX's flag. A node keeps the precision in force when it was made and steps in it
(``MimeNode``), so a run's steps keep their build's precision whatever was built after it. Where no build has applied
one (a node made on its own, outside an experiment), the run's dtype is JAX's flag's, as before precision was declared.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, NoReturn, cast

import jax
import numpy as np

#: ``jax.numpy`` through ``Any``: jaxlib ships no stubs, so a strict checker reads its constructors as partly unknown.
_jnp: Any = jax.numpy
#: ``jax.tree_util`` through ``Any``, for the same reason.
_tree: Any = jax.tree_util

Precision = Literal["float32", "float64"]
PRECISIONS: tuple[Precision, ...] = ("float32", "float64")
#: The precision of an experiment that states none: float64, exact by default; float32 is declared.
DEFAULT_PRECISION: Precision = "float64"

#: The precision the last build applied, for what runs outside a node's step (a controller's code, a sensor).
_applied: list[Precision | None] = [None]
#: The precision of the node whose step or initial state is being traced, over the build's (``MimeNode``).
_NODE: ContextVar[Precision | None] = ContextVar("mime_node_precision", default=None)
#: A TEST-ONLY MIGRATION BRIDGE, removed in precision's P5, and no user's setting. Set to ``1`` inside a pytest session
#: (pytest sets ``PYTEST_VERSION`` for it, and the processes its tests start inherit both), an experiment that states no
#: precision is built as JAX's flag says, and the flag is left as it is: how MIME built before precision was declared.
#: ``tests/conftest.py`` sets it for the suite until its tests declare their precision. Outside a pytest session it is
#: not read: a user's process always builds an unstated precision in float64.
FROM_JAX_FLAG = "MIME_PRECISION_FROM_JAX_FLAG"


def unstated_precision() -> Precision | None:
    """What an experiment that states no precision is built in: :data:`DEFAULT_PRECISION`, or ``None`` where JAX's flag
    decides (:data:`FROM_JAX_FLAG`, in a pytest session only)."""
    bridged = os.environ.get(FROM_JAX_FLAG) == "1" and "PYTEST_VERSION" in os.environ
    return None if bridged else DEFAULT_PRECISION


def declared_precision() -> Precision | None:
    """The precision in force: the stepping node's, else the last build's; ``None`` where no build applied one."""
    return _NODE.get() or _applied[0]


def apply_precision(precision: Precision | None) -> None:
    """Make ``precision`` the run's, with JAX's 64-bit types enabled; ``None`` applies none and leaves JAX's flag."""
    if precision is not None:
        cast(Any, jax.config).update("jax_enable_x64", True)
    _applied[0] = precision


@contextmanager
def applied(precision: Precision) -> Generator[None]:
    """For the block, ``precision`` the run's, with JAX's 64-bit types enabled (D2's mode: a float32 block is declared
    float32, not JAX's flag turned off); the precision applied before it, and JAX's flag, after it."""
    held = _applied[0]
    with cast(Any, jax).enable_x64(True):
        _applied[0] = precision
        try:
            yield
        finally:
            _applied[0] = held


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


#: A 64-bit JAX dtype and the 32-bit one JAX makes in its place without 64-bit types.
_NARROWER = {np.dtype(np.float64): np.dtype(np.float32), np.dtype(np.complex128): np.dtype(np.complex64)}


def narrowed(tree: Any) -> Any:
    """``tree`` in a declared float32 run as JAX made it before precision was declared (64-bit types disabled): each
    float64 or complex128 JAX array narrowed to the 32-bit one, so that a node's constants, and the derived graph a save
    records, are the ones a float32 run always had. As it is in a float64 run, or where none is declared; Python numbers
    and NumPy arrays pass as they are (a node makes its constants from them in its own precision)."""
    if declared_precision() != "float32":
        return tree
    def narrow(leaf: Any) -> Any:
        return leaf.astype(_NARROWER[leaf.dtype]) if isinstance(leaf, jax.Array) and leaf.dtype in _NARROWER else leaf

    return _tree.tree_map(narrow, tree)


#: What a float64 run does with a JAX array narrower than float64 handed to a node (:func:`declared_values`), the one
#: switch for it: ``"refuse"`` (``run.precision-lossy``) or ``"widen"`` (to float64, with a :class:`LossyValueWarning`).
#: The default follows the maintainer's answer (precision P1); changing it is this line.
NARROW_IN_FLOAT64: Literal["refuse", "widen"] = "refuse"


_lossy_refuser: Callable[[str, str], NoReturn] | None = None


def install_lossy_refuser(refuser: Callable[[str, str], NoReturn]) -> None:
    """Called once by ``mime._refusals``: ``refuser(owner, named)`` raises the coded refusal ``run.precision-lossy``.
    ``mime.core`` imports nothing above itself (``tests/test_import_boundaries.py``), and the refusal's class is
    ``mime._refusals``'s, so it registers itself here, as ``mime.units`` does for labels (``mime.core._labels``)."""
    global _lossy_refuser
    _lossy_refuser = refuser


class LossyValueWarning(UserWarning):
    """A JAX array narrower than float64 widened into a float64 run: computed in float32, it entered rounded."""


def _narrower_than_float64(leaf: Any) -> bool:
    return isinstance(leaf, jax.Array) and bool(jax.numpy.issubdtype(leaf.dtype, jax.numpy.inexact)) and \
        leaf.dtype.itemsize < (16 if jax.numpy.issubdtype(leaf.dtype, jax.numpy.complexfloating) else 8)


def declared_values(tree: Any, owner: str) -> Any:
    """``tree``, the arguments a node is made from, in the run's declared precision: the build's boundary for them.
    In a float32 run it is :func:`narrowed`. In a float64 run a float32 (or narrower) JAX array was computed in float32,
    JAX's default before a build enables 64-bit types, and widening it does not bring back what was rounded away: as
    :data:`NARROW_IN_FLOAT64` says, it is refused (``run.precision-lossy``) or widened with a warning naming it.
    Python numbers and NumPy arrays pass as they are (float64 is exact in both precisions)."""
    if declared_precision() != "float64":
        return narrowed(tree)
    lossy = [(str(_tree.keystr(path)), str(leaf.dtype)) for path, leaf in _tree.tree_leaves_with_path(tree)
             if _narrower_than_float64(leaf)]
    if not lossy:
        return tree
    named = ", ".join(f"{path or 'the value'} ({dtype})" for path, dtype in lossy[:5])
    if NARROW_IN_FLOAT64 == "widen":
        import warnings

        warnings.warn(LossyValueWarning(
            f"{owner} is handed a JAX array narrower than float64 in a float64 run: {named}, widened to float64. It was "
            f"computed in float32 (JAX's default before a build enables 64-bit types), so it enters the run rounded: "
            f"pass a Python float or a NumPy float64 array, or compute it after the build"), stacklevel=3)

        def widen(leaf: Any) -> Any:
            if not _narrower_than_float64(leaf):
                return leaf
            return leaf.astype(np.complex128 if jax.numpy.issubdtype(leaf.dtype, jax.numpy.complexfloating)
                               else np.float64)

        return _tree.tree_map(widen, tree)
    if _lossy_refuser is None:
        raise RuntimeError(f"{owner} is handed a JAX array narrower than float64 in a float64 run: {named}; import mime "
                           f"first, so the refusal (run.precision-lossy) is installed")
    _lossy_refuser(owner, named)


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


__all__ = ["DEFAULT_PRECISION", "FROM_JAX_FLAG", "NARROW_IN_FLOAT64", "PRECISIONS", "LossyValueWarning", "Precision",
           "applied", "apply_precision",
           "declared_precision",
           "declared_values", "diagonal", "install_lossy_refuser", "narrowed",
           "diagonal_matrix", "eye", "in_float64", "node_precision", "precision_dtype", "unstated_precision"]
