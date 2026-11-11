"""Walking the programs a registered node kind's step traces for its linear solves (``mime.core.solves``).

**What a walk finds.** Every equation of a traced step and of every program it holds (a ``jit``, a ``cond`` branch, a
``scan`` or ``while`` body, a ``custom_linear_solve``'s sub-programs), with the name stack it ran under: the stack of
the equation joined to the stacks of the equations that hold it, since a ``jit`` traced once keeps its inner stack
apart from its caller's. A solve primitive (``mime.core.solves.SOLVE_PRIMITIVES``) with no registered ``mime-solve``
scope in that stack is a solve that went through no shared function; a scope no site is registered under is a name
nobody registered. In a float64 run, every equation inside a registered scope is float64 (or complex128) in its
operands and its results.

**The cases** (:func:`cases`): the node kinds the registry names, each as it is built for the gates that already
trace them, a step as ``jax.make_jaxpr`` of the node's own ``advance``.

**The plants** (:data:`PLANTS`): a direct ``jnp.linalg.solve`` added to a node's step; a shared function bypassed for
the raw solve; a downcast to float32 inside a registered solve. ``MIME_PLANT_SOLVE`` names one, and the clean tests then
fail: the probe's detection rate is measured on that."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from mime.core import solves
from mime.core.solves import SCOPE, SOLVE_PRIMITIVES, SOLVE_SITES

PLANT_VAR = "MIME_PLANT_SOLVE"
_SCOPED = re.compile(rf"{re.escape(SCOPE)}\.([A-Za-z0-9_.]+)")


@dataclass(frozen=True)
class Equation:
    primitive: str
    stack: str
    scopes: tuple[str, ...]
    dtypes: frozenset[str]


def _programs(equation: Any) -> Iterator[Any]:
    for value in equation.params.values():
        for held in value if isinstance(value, tuple | list) else (value,):
            inner = getattr(held, "jaxpr", held)
            if hasattr(inner, "eqns"):
                yield inner


def walk(jaxpr: Any, outer: str = "") -> Iterator[Equation]:
    """Every equation of ``jaxpr`` and of the programs it holds, with the stack it ran under."""
    for equation in jaxpr.eqns:
        stack = f"{outer}/{equation.source_info.name_stack}"
        floats = frozenset(
            str(v.aval.dtype) for v in (*equation.invars, *equation.outvars)
            if hasattr(getattr(v, "aval", None), "dtype") and jnp.issubdtype(v.aval.dtype, jnp.inexact))
        yield Equation(equation.primitive.name, stack, tuple(_SCOPED.findall(stack)), floats)
        for inner in _programs(equation):
            yield from walk(inner, stack)


def unregistered(jaxpr: Any) -> list[str]:
    """Each solve primitive outside every registered scope, and each scope nobody registered, as text."""
    found: list[str] = []
    for equation in walk(jaxpr):
        names = [name for name in equation.scopes if name not in SOLVE_SITES]
        if names:
            found.append(f"{equation.primitive} in the scope of {names}, which no module registered")
        if equation.primitive in SOLVE_PRIMITIVES and not equation.scopes:
            found.append(f"{equation.primitive} outside every registered solve scope: {equation.stack!r}")
    return found


def sites_reached(jaxpr: Any) -> set[str]:
    return {name for equation in walk(jaxpr) for name in equation.scopes if name in SOLVE_SITES}


def downcasts(jaxpr: Any, wide: str = "float64") -> list[str]:
    """Each equation inside a registered solve scope that holds a floating value narrower than ``wide``."""
    narrow = {wide, "complex128"}
    return [f"{equation.primitive} in {equation.scopes} holds {sorted(equation.dtypes - narrow)}"
            for equation in walk(jaxpr) if equation.scopes and equation.dtypes - narrow]


# -- the cases ----------------------------------------------------------------------------------------------------------

def _flow_programs() -> dict[str, tuple[str, Callable[[], Any]]]:
    from fvm_cases import step_of, tube_node

    from mime.nodes.flow.fvm import ExplicitFlowNode, periodic_tube_layout

    def implicit(scheme: str, convection: bool) -> Any:
        node = tube_node(8, scheme=scheme, convection=convection)
        return jax.make_jaxpr(step_of(node, node._step_constants(None)))(dict(node.initial_state()))  # noqa: SLF001

    def explicit() -> Any:
        layout = periodic_tube_layout(0.5, 4.0 / 8, cells_across=8)
        node = ExplicitFlowNode("flow", 0.1 * layout.spacing ** 2, cells_x=layout.cells[0], cells_y=layout.cells[1],
                                cells_z=layout.cells[2], spacing=layout.spacing, viscosity=1.0, density=1.0,
                                tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius,
                                scheme="rk3")
        step = step_of(node, node._step_constants(None))  # noqa: SLF001
        return jax.make_jaxpr(step)(dict(node.initial_state()))

    def open_box() -> Any:
        from open_cases import open_node

        node = open_node(8)
        p = node._step_constants(None)  # noqa: SLF001
        return jax.make_jaxpr(lambda s: node.advance(p, s, {}, node.delta_t))(dict(node.initial_state()))

    return {"ImplicitFlowNode/imex": ("ImplicitFlowNode", lambda: implicit("imex_euler", True)),
            "ImplicitFlowNode/steady": ("ImplicitFlowNode", lambda: implicit("steady", False)),
            "ExplicitFlowNode/rk3": ("ExplicitFlowNode", explicit),
            "OpenFlowNode/two openings": ("OpenFlowNode", open_box)}


def _near_field_programs() -> dict[str, tuple[str, Callable[[], Any]]]:
    from near_field_cases import NODE, bodies, joint_node, mrs, spheres, start_inputs

    from mime.build._maddening import parameter_leaves
    from mime.interactions import NearField

    def joint(**settings: Any) -> Any:
        built = bodies("solve_sites", spheres(0.3e-3, 2), mrs(0.6e-3, **settings)).build()
        node = joint_node(built)
        inputs = {key: jnp.asarray(value) for key, value in start_inputs(built).items()}
        p = node._step_constants(parameter_leaves(built.graph)[NODE])  # noqa: SLF001
        return jax.make_jaxpr(lambda q: node.advance(q, node.initial_state(), inputs, 1e-3))(p)

    return {"JointDragNode/dense": ("JointDragNode", lambda: joint()),
            "JointDragNode/gmres": ("JointDragNode", lambda: joint(near_field=NearField(solve="gmres")))}


def cases() -> dict[str, tuple[str, Callable[[], Any]]]:
    """Each case's label: the node kind it traces and the function that traces its step in the run's dtype."""
    return {**_flow_programs(), **_near_field_programs()}


# -- the plants ---------------------------------------------------------------------------------------------------------

def _raw(matrix: Any, rhs: Any) -> Any:
    return jnp.linalg.solve(matrix, rhs)


def _downcast_dense(matrix: Any, rhs: Any) -> Any:
    with solves.solving(solves.DENSE):
        return jnp.linalg.solve(matrix.astype(jnp.float32), rhs.astype(jnp.float32)).astype(matrix.dtype)


def _narrowed(solve: Callable[..., Any]) -> Callable[..., Any]:
    """``solve``'s answer rounded through float32, whatever it returns first (an array, or a pair or triple of them)."""
    def narrow(x: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda a: a.astype(jnp.float32).astype(a.dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, x)

    def planted(*args: Any, **kwargs: Any) -> Any:
        x, report = solve(*args, **kwargs)
        return narrow(x), report

    return planted


def _patch(patch: Any, module: str, name: str, value: Any) -> None:
    import importlib

    patch.setattr(importlib.import_module(module), name, value)


def bypass_dense(patch: Any) -> None:
    """Both dense callers take ``jnp.linalg.solve`` raw, as a new solve written without the shared function would."""
    _patch(patch, "mime.nodes.hydro.near_field._contract", "dense_solve", _raw)
    _patch(patch, "mime.nodes.flow.fvm._ports", "dense_solve", _raw)


def downcast_dense(patch: Any) -> None:
    """The dense helper's elimination run in float32 inside its own scope, whatever the run's dtype."""
    _patch(patch, "mime.nodes.hydro.near_field._contract", "dense_solve", _downcast_dense)
    _patch(patch, "mime.nodes.flow.fvm._ports", "dense_solve", _downcast_dense)


def downcast_cg(patch: Any) -> None:
    from mime.nodes.flow.fvm import _solve

    _patch(patch, "mime.nodes.flow.fvm._solve", "guarded_cg", _narrowed(_solve.guarded_cg))


def downcast_gmres(patch: Any) -> None:
    from mime.nodes.hydro.near_field import _gmres

    _patch(patch, "mime.nodes.hydro.near_field._gmres", "gmres", _narrowed(_gmres.gmres))


def direct_solve_in_step(patch: Any) -> None:
    """A direct ``jnp.linalg.solve`` in a node's step: every node's ``advance`` adds one to what it returns."""
    import importlib

    for module, cls in (("mime.nodes.flow.fvm._node", "ImplicitFlowNode"),
                        ("mime.nodes.hydro.near_field._node", "JointDragNode")):
        node_class = getattr(importlib.import_module(module), cls)
        advance = node_class.advance

        def planted(self: Any, *args: Any, _advance: Any = advance, **kwargs: Any) -> Any:
            out = dict(_advance(self, *args, **kwargs))
            key = next(k for k, v in out.items() if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating))
            first = jnp.asarray(out[key])
            ones = jnp.ones((2,), dtype=first.dtype)
            extra = jnp.linalg.solve(jnp.eye(2, dtype=first.dtype) * (1.0 + 0.0 * jnp.sum(first)), ones)
            out[key] = first + 0.0 * jnp.sum(extra)
            return out

        patch.setattr(node_class, "advance", planted)


DENSE_CASES = ("JointDragNode/dense", "OpenFlowNode/two openings")
#: The plants by name: what applies it, the check that must catch it, and the cases whose programs it touches (each is
#: caught on every one of them).
PLANTS: dict[str, tuple[Callable[[Any], None], str, tuple[str, ...]]] = {
    "bypass_dense": (bypass_dense, "unregistered", DENSE_CASES),
    "direct_solve_in_step": (direct_solve_in_step, "unregistered", ("ImplicitFlowNode/steady", "JointDragNode/dense")),
    "downcast_dense": (downcast_dense, "downcasts", DENSE_CASES),
    "downcast_cg": (downcast_cg, "downcasts", ("ImplicitFlowNode/steady", "OpenFlowNode/two openings")),
    "downcast_gmres": (downcast_gmres, "downcasts", ("JointDragNode/gmres",)),
}


def planted_from_environment() -> str | None:
    return os.environ.get(PLANT_VAR) or None
