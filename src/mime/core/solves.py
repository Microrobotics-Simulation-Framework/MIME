"""Linear solves: the registry of solve sites, the scope a shared solve function runs in, and the one dense helper.

**A solve site** is a place in a node's step that eliminates a linear system: a conjugate gradient, GMRES, a dense LU,
the port elimination, a transform that inverts an operator. Each is registered (:func:`register`) by the module that
performs it, and the shared function that performs it runs inside :func:`solving`, a JAX name scope
(``mime-solve.<site>``). A name scope changes no value and no operation: the traced program is the solve's own, its
operations carrying the site's name in their metadata, so writing a solve through here changes no number.

**What the scope is for.** ``tests/solve_sites`` traces the step of every node kind a site names, in both dtypes, and
walks the program: a linear-solve primitive (``lu``, ``cholesky``, ``triangular_solve``, ``custom_linear_solve``,
``fft``) outside every registered scope is a solve that did not go through a shared function, and fails; and in a
float64 run every operand and result of a registered solve is float64 (no silent downcast inside one). The test reads
the program the step traces, not the source, so it sees a solve wherever it is called from.

**The rule for new code.** A new linear solve goes through the shared solve functions: ``projected_solve`` (the FVM's
conjugate gradient), ``reported_solve`` (GMRES), :func:`dense_solve` (a dense elimination), or registers a site of its
own where none of them fits, with its reason. A solve of a few unknowns that a recipe could never change (a rigid body's
six, a handover's two) registers as exempt, with the reason, and runs inside :func:`solving`.

This module imports JAX and the standard library only; the node kinds a site names are given by class name."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import jax
from jax import Array

jnp: Any = jax.numpy

#: The name scope's prefix: a scope ``mime-solve.<site>`` marks everything traced inside it as that site's.
SCOPE = "mime-solve"

#: What a site eliminates with: ``cg`` (the FVM's guarded conjugate gradient), ``gmres``, ``dense`` (an LU or a
#: Cholesky factor of a matrix held whole), ``transform`` (a transform that inverts an operator that is diagonal in it).
KINDS = ("cg", "gmres", "dense", "transform")

#: The primitives that are a linear solve's own, wherever a program reaches them: a registered scope must hold each.
SOLVE_PRIMITIVES = frozenset({"lu", "cholesky", "triangular_solve", "custom_linear_solve", "fft"})


@dataclass(frozen=True)
class SolveSite:
    """One registered solve site.

    ``name`` is dotted (``fvm_cg``); ``kind`` one of :data:`KINDS`; ``where`` the function that performs the solve
    (``module.function``). ``nodes`` are the class names of the node kinds whose step reaches it, which
    ``tests/solve_sites`` traces: a site no node walks says so in ``unwalked``. ``exempt`` is the reason a site is not
    one of the shared functions (a system too small for a precision recipe to mean anything, a released path left
    untouched), empty for a shared one."""

    name: str
    kind: str
    where: str
    nodes: tuple[str, ...] = ()
    exempt: str = ""
    unwalked: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"solve site {self.name!r}: kind {self.kind!r} is not one of {KINDS}")
        if not self.nodes and not self.unwalked:
            raise ValueError(f"solve site {self.name!r} names no node kind the walk traces and says no reason it is "
                             f"not walked")

    @property
    def scope(self) -> str:
        return f"{SCOPE}.{self.name}"


#: Every registered site by name, filled as the modules that perform solves are imported.
SOLVE_SITES: dict[str, SolveSite] = {}


def register(site: SolveSite) -> SolveSite:
    """Register ``site`` (the same site again is a no-op: a module imported twice); a different site under a name taken
    is a mistake."""
    held = SOLVE_SITES.setdefault(site.name, site)
    if held != site:
        raise ValueError(f"solve site {site.name!r} is registered twice, differently: {held} and {site}")
    return held


@contextlib.contextmanager
def solving(site: SolveSite) -> Generator[None]:
    """Everything traced inside is ``site``'s: a name scope, which changes no value and no operation."""
    with jax.named_scope(site.scope):
        yield


def sites_of(node_class: type) -> tuple[SolveSite, ...]:
    """The sites registered under ``node_class`` or a class it derives from."""
    kinds = {cls.__name__ for cls in node_class.__mro__}
    return tuple(site for site in SOLVE_SITES.values() if kinds & set(site.nodes))


DENSE = register(SolveSite(
    "dense", "dense", "mime.core.solves.dense_solve",
    nodes=("JointDragNode", "OpenFlowNode"),
))


def dense_solve(matrix: Array, rhs: Array) -> Array:
    """``matrix^-1 rhs`` by one dense LU of ``matrix`` (``jnp.linalg.solve``, whose derivative reuses the factors): the
    one place MIME eliminates a dense system whose size a recipe could matter to. The near field's joint matrix and the
    port system go through it, in the run's dtype."""
    with solving(DENSE):
        return jnp.linalg.solve(matrix, rhs)


__all__ = ["DENSE", "KINDS", "SCOPE", "SOLVE_PRIMITIVES", "SOLVE_SITES", "SolveSite", "dense_solve", "register",
           "sites_of", "solving"]
