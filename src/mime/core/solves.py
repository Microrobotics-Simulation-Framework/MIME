"""Linear solves: the registry of solve sites, the scope a shared solve function runs in, and the one dense helper.

**A solve site** is a place in a node's step that eliminates a linear system: a conjugate gradient, GMRES, a dense LU,
the port elimination, a transform that inverts an operator. Each is registered (:func:`register`) by the module that
performs it, and the shared function that performs it runs inside :func:`solving`, a JAX name scope
(``mime-solve.<site>``). A name scope changes no value and no operation: the traced program is the solve's own, its
operations carrying the site's name in their metadata, so writing a solve through here changes no number.

**What the scope is for, and what the test covers.** ``tests/saved/test_solve_walk.py`` traces the step of **every
registered node kind** in the switch scan's configurations (and fails for a kind no configuration traces), and walks
the program each traces. Each solve primitive (:data:`SOLVE_PRIMITIVES`) is judged by its **innermost** scope: a solve
outside every scope went through no shared function; one in a scope whose kind does not own it (:data:`OWNED`) is a raw
solve nested in another solve (an ``lu`` in the conjugate gradient's operator); one in an exempt site over its declared
``bound`` is a raw solve inside a scope that vouched for small systems. In a float64 run no equation of the step holds a
narrower float, inside a solve's scope or outside it, except in the steps of a kind that reaches a site declared
``narrower``. The test reads the program the step traces, not the source, so it sees a solve wherever it is called from.

**What it does not see.** A solve in a node kind none of the configurations traces, or in a branch or an option none of
them takes (the switch scan's own limit); a solve written as a bare loop of matrix products (a Jacobi or Richardson
``fori_loop`` has no primitive of its own); an ``eigh``, which uses a matrix and does not solve one; a solve outside a
node's step (a constructor, a plan, a host-side ``numpy`` call).

**The rule for new code.** A new linear solve goes through the shared solve functions: ``projected_solve`` (the FVM's
conjugate gradient), ``reported_solve`` (GMRES), :func:`dense_solve` (a dense LU; add the node kind to the ``dense``
site's ``nodes``, which the test requires), or ``register()``s a site of its own **in its own module**, runs the solve
inside :func:`solving`, and gives, if it is not one of the shared functions, ``exempt`` (the reason) and ``bound`` (the
largest system it solves). A new node kind with solves is walked once a configuration in
``tests/saved/switch_configurations.py`` builds it.

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

#: The primitives that are a linear solve's own, wherever a program reaches them: each must run inside a registered scope.
#: ``svd`` is what ``lstsq`` and ``pinv`` solve with. Not here, and so not seen: ``eigh`` (a spectral radius or a
#: definiteness check is a use of the matrix, not a solve of it) and a solve written as a bare loop of matrix products
#: (a Jacobi or Richardson ``fori_loop`` has no primitive of its own to see).
SOLVE_PRIMITIVES = frozenset({"lu", "cholesky", "triangular_solve", "custom_linear_solve", "fft", "svd",
                              "tridiagonal_solve"})

#: The solve primitives each kind of site may hold, the walk judging each primitive by its innermost scope: a raw
#: ``jnp.linalg.solve`` written inside the conjugate gradient's operator is an ``lu`` in a ``cg`` scope, which owns none.
OWNED: dict[str, frozenset[str]] = {
    "cg": frozenset({"custom_linear_solve"}),
    "gmres": frozenset({"custom_linear_solve", "triangular_solve"}),
    "dense": frozenset({"lu", "cholesky", "triangular_solve", "custom_linear_solve"}),
    "transform": frozenset({"fft"}),
}

#: The primitives that carry a system's matrix as their first operand, whose last dimension is the system's order: what
#: an exempt site's ``bound`` is checked on.
MATRIX_PRIMITIVES = frozenset({"lu", "cholesky", "triangular_solve", "svd"})


@dataclass(frozen=True)
class SolveSite:
    """One registered solve site.

    ``name`` is dotted (``fvm_cg``); ``kind`` one of :data:`KINDS`; ``where`` the function that performs the solve
    (``module.function``). ``nodes`` are the class names of the node kinds whose step reaches it: the solve walk
    (``tests/saved/test_solve_walk.py``) traces every registered node kind's step and fails if a kind reaches a site it
    does not name, or a site is reached by no kind. ``exempt`` is the reason a site is not one of the shared functions
    (a system too small for a precision recipe to mean anything, a released path left untouched), empty for a shared
    one; an exempt site also gives ``bound``, the largest order of any system solved inside it, which the walk checks
    on the operands it traces. ``narrower`` is the reason a site computes below the run's dtype by design (an exempt
    site only); the float64 dtype walk leaves it, and the steps of the node kinds that reach it, out."""

    name: str
    kind: str
    where: str
    nodes: tuple[str, ...]
    exempt: str = ""
    bound: int = 0
    narrower: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"solve site {self.name!r}: kind {self.kind!r} is not one of {KINDS}")
        if not self.nodes:
            raise ValueError(f"solve site {self.name!r} names no node kind whose step reaches it")
        if bool(self.exempt) != bool(self.bound):
            raise ValueError(f"solve site {self.name!r}: an exempt site gives its reason and a size bound, and a "
                             f"shared one neither")
        if self.narrower and not self.exempt:
            raise ValueError(f"solve site {self.name!r}: a shared solve function computes in the run's dtype")

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
    nodes=("DynamicArmNode", "JointDragNode", "MRSDragNode", "OpenFlowNode", "RFTDragNode", "RigidInertialMidpointNode",
           "RigidInertialNode", "RigidInertialTensorNode"),
))


def dense_solve(matrix: Array, rhs: Array) -> Array:
    """``matrix^-1 rhs`` by one dense LU of ``matrix`` (``jnp.linalg.solve``, whose derivative reuses the factors): the
    one place MIME eliminates a dense system whose size a recipe could matter to. The near field's joint matrix and the
    port system go through it, in the run's dtype."""
    with solving(DENSE):
        return jnp.linalg.solve(matrix, rhs)


__all__ = ["DENSE", "KINDS", "SCOPE", "SOLVE_PRIMITIVES", "SOLVE_SITES", "SolveSite", "dense_solve", "register",
           "sites_of", "solving"]
