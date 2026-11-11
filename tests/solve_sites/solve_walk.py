"""Walking the programs a registered node kind's step traces for its linear solves (``mime.core.solves``).

**What a walk finds.** Every equation of a traced step and of every program it holds (a ``jit``, a ``cond`` branch, a
``scan`` or ``while`` body, a ``custom_linear_solve``'s sub-programs), with the name stack it ran under: the stack of
the equation joined to the stacks of the equations that hold it, since a ``jit`` traced once keeps its inner stack
apart from its caller's. A solve primitive (``mime.core.solves.SOLVE_PRIMITIVES``) is judged by its **innermost**
``mime-solve`` scope: none is a solve that went through no shared function; a site whose kind does not own the primitive
(``OWNED``) is a raw solve nested in another solve; an exempt site is also held to its declared size. A scope no site is
registered under is a name nobody registered. In a float64 run, no equation of the step holds a floating value narrower
than float64 (:func:`narrow_anywhere`), inside a solve's scope or outside it, except in the steps of a node kind that
reaches a site declared ``narrower``.

**What it does not see.** A solve in a node kind no configuration traces; a solve written as a bare loop of matrix
products (a Jacobi or Richardson ``fori_loop`` has no primitive of its own); an ``eigh``, which uses a matrix and does not
solve one. ``tests/saved/test_solve_walk.py`` walks every registered node kind's step in the switch scan's configurations
and fails for a registered kind that none traces.

**The cases** (:func:`cases`): the six steps the plants are applied to and the clean walk is read on in both dtypes, each
as it is built for the gates that already trace them, a step as ``jax.make_jaxpr`` of the node's own ``advance``.

**The plants** (:data:`PLANTS`), each a way a solve could skip the seam: a direct solve added to a node's step (an LU, a
least squares, a pseudo-inverse, a tridiagonal solve); a shared function bypassed for the raw solve; a raw solve nested
in the conjugate gradient's operator, and a raw one written inside an exempt site's scope; a downcast inside a registered
solve, in the Poisson transform among them; a float32 round trip completed in the caller before ``dense_solve`` or on its
answer after it. ``MIME_PLANT_SOLVE`` names one, and the clean tests then fail: the probe's detection rate is measured on
that."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from mime.core import solves
from mime.core.solves import MATRIX_PRIMITIVES, OWNED, SCOPE, SOLVE_PRIMITIVES, SOLVE_SITES, SolveSite

PLANT_VAR = "MIME_PLANT_SOLVE"
_SCOPED = re.compile(rf"{re.escape(SCOPE)}\.([A-Za-z0-9_.]+)")


@dataclass(frozen=True)
class Equation:
    primitive: str
    stack: str
    scopes: tuple[str, ...]
    dtypes: frozenset[str]
    #: The order of the system a matrix primitive holds (its first operand's last dimension), 0 for any other.
    order: int = 0

    @property
    def innermost(self) -> str | None:
        """The registered or unregistered site whose scope this equation sits closest inside."""
        return self.scopes[-1] if self.scopes else None


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
        first = getattr(equation.invars[0], "aval", None) if equation.invars else None
        order = int(first.shape[-1]) if equation.primitive.name in MATRIX_PRIMITIVES and getattr(first, "shape", None) else 0
        yield Equation(equation.primitive.name, stack, tuple(_SCOPED.findall(stack)), floats, order)
        for inner in _programs(equation):
            yield from walk(inner, stack)


def solve_problems(jaxpr: Any, sites: Mapping[str, SolveSite] | None = None) -> list[str]:
    """What a program holds that the rule forbids, as text. Each solve primitive is judged by its **innermost** solve
    scope, so that an enclosing scope is never a pass for what runs inside it:

    - a solve primitive outside every scope: a solve that went through no shared function;
    - a solve primitive its innermost site's kind does not own (a raw ``lu`` inside the conjugate gradient's operator);
    - a solve inside an exempt site whose order is over the site's declared bound (a raw 30 x 30 inside the six-unknown
      blocks' scope);
    - a scope nobody registered."""
    sites = SOLVE_SITES if sites is None else sites
    found: list[str] = []
    for equation in walk(jaxpr):
        for name in equation.scopes:
            if name not in sites:
                found.append(f"{equation.primitive} in the scope of {name!r}, which no module registered")
        if equation.primitive not in SOLVE_PRIMITIVES:
            continue
        site = sites.get(equation.innermost) if equation.innermost else None
        if equation.innermost is None:
            found.append(f"{equation.primitive} outside every registered solve scope: {equation.stack!r}")
        elif site is not None and equation.primitive not in OWNED[site.kind]:
            found.append(f"{equation.primitive} inside {site.name!r} (kind {site.kind!r}), which owns only "
                         f"{sorted(OWNED[site.kind])}: {equation.stack!r}")
        elif site is not None and site.exempt and equation.order > site.bound:
            found.append(f"{equation.primitive} of order {equation.order} inside the exempt site {site.name!r}, whose "
                         f"bound is {site.bound}: {equation.stack!r}")
    return found


def sites_reached(jaxpr: Any) -> set[str]:
    return {name for equation in walk(jaxpr) for name in equation.scopes if name in SOLVE_SITES}


def downcasts(jaxpr: Any, wide: str = "float64") -> list[str]:
    """Each equation inside a registered solve scope that holds a floating value narrower than ``wide``, except in the
    scope of a site declared ``narrower`` by design."""
    narrow = {wide, "complex128"}
    by_design = {name for name, site in SOLVE_SITES.items() if site.narrower}
    return [f"{equation.primitive} in {equation.scopes} holds {sorted(equation.dtypes - narrow)}"
            for equation in walk(jaxpr)
            if equation.scopes and equation.dtypes - narrow and equation.innermost not in by_design]


def narrow_anywhere(jaxpr: Any, wide: str = "float64") -> list[str]:
    """Each equation of a float64 run's step, inside a scope or not, that holds a floating value narrower than
    ``wide``: a narrowing completed outside the solve (a round trip before ``dense_solve``, an answer rounded after it)
    is seen as well as one inside."""
    narrow = {wide, "complex128"}
    return [f"{equation.primitive} holds {sorted(equation.dtypes - narrow)}: {equation.stack!r}"
            for equation in walk(jaxpr) if equation.dtypes - narrow]


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


def _round_trip(x: Any) -> Any:
    return x.astype(jnp.float32).astype(x.dtype)


def _rounded_before(matrix: Any, rhs: Any) -> Any:
    """The caller rounds its operands through float32, and the shared function then sees float64."""
    return solves.dense_solve(_round_trip(matrix), rhs)


def _rounded_after(matrix: Any, rhs: Any) -> Any:
    """The shared function's float64 answer, rounded through float32 by its caller."""
    return _round_trip(solves.dense_solve(matrix, rhs))


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


def _extra_solve(form: str, anchor: Any, order: int = 2) -> Any:
    """A solve of ``form`` of the given order, tied to ``anchor`` (so that its operands are traced values), as a scalar
    to add to something with a zero weight."""
    dtype = anchor.dtype
    matrix = jnp.eye(order, dtype=dtype) * (1.0 + 0.0 * jnp.sum(anchor))
    ones = jnp.ones((order,), dtype=dtype)
    if form == "solve":
        found = jnp.linalg.solve(matrix, ones)
    elif form == "lstsq":
        found = jnp.linalg.lstsq(matrix, ones)[0]
    elif form == "pinv":
        found = jnp.linalg.pinv(matrix) @ ones
    elif form == "tridiagonal":
        zeros = jnp.zeros((order,), dtype=dtype)
        found = jax.lax.linalg.tridiagonal_solve(zeros, jnp.diagonal(matrix), zeros, ones[:, None])[:, 0]
    else:
        raise ValueError(form)
    return jnp.sum(found)


def bypass_dense(patch: Any) -> None:
    """Both dense callers take ``jnp.linalg.solve`` raw, as a new solve written without the shared function would."""
    _patch(patch, "mime.nodes.hydro.near_field._contract", "dense_solve", _raw)
    _patch(patch, "mime.nodes.flow.fvm._ports", "dense_solve", _raw)


def downcast_dense(patch: Any) -> None:
    """The dense helper's elimination run in float32 inside its own scope, whatever the run's dtype."""
    _patch(patch, "mime.nodes.hydro.near_field._contract", "dense_solve", _downcast_dense)
    _patch(patch, "mime.nodes.flow.fvm._ports", "dense_solve", _downcast_dense)


def rounded_before_dense(patch: Any) -> None:
    _patch(patch, "mime.nodes.hydro.near_field._contract", "dense_solve", _rounded_before)
    _patch(patch, "mime.nodes.flow.fvm._ports", "dense_solve", _rounded_before)


def rounded_after_dense(patch: Any) -> None:
    _patch(patch, "mime.nodes.hydro.near_field._contract", "dense_solve", _rounded_after)
    _patch(patch, "mime.nodes.flow.fvm._ports", "dense_solve", _rounded_after)


def downcast_cg(patch: Any) -> None:
    from mime.nodes.flow.fvm import _solve

    _patch(patch, "mime.nodes.flow.fvm._solve", "guarded_cg", _narrowed(_solve.guarded_cg))


def downcast_gmres(patch: Any) -> None:
    from mime.nodes.hydro.near_field import _gmres

    _patch(patch, "mime.nodes.hydro.near_field._gmres", "gmres", _narrowed(_gmres.gmres))


def downcast_poisson(patch: Any) -> None:
    """Both grid kinds' Poisson transform run on a float32 source inside its scope."""
    from mime.nodes.flow.fvm._grid import POISSON_SITE, MACGrid
    from mime.nodes.flow.fvm._open import OpenMACGrid

    for grid in (MACGrid, OpenMACGrid):
        def planted(self: Any, source: Any, _original: Any = grid.poisson) -> Any:
            with solves.solving(POISSON_SITE):
                return _original(self, source.astype(jnp.float32)).astype(source.dtype)

        patch.setattr(grid, "poisson", planted)


def raw_poisson(patch: Any) -> None:
    """The periodic grid's Poisson transform written without its scope: the same transform, as a new one written
    without the seam would be."""
    from mime.nodes.flow.fvm._grid import MACGrid

    def planted(self: Any, source: Any) -> Any:
        symbol = self._poisson_symbol(source.dtype)  # noqa: SLF001
        solved = -jnp.fft.rfftn(source) / symbol
        solved = solved.at[0, 0, 0].set(0.0)
        return jnp.fft.irfftn(solved, s=source.shape).astype(source.dtype)

    patch.setattr(MACGrid, "poisson", planted)


def nested_in_cg(patch: Any) -> None:
    """A raw 8 x 8 ``jnp.linalg.solve`` inside the conjugate gradient's operator, in the scope of the FVM's CG."""
    from mime.nodes.flow.fvm import _solve

    original = _solve.guarded_cg

    def planted(operator: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
        x, report = original(operator, b, *args, **kwargs)
        anchor = jax.tree_util.tree_leaves(x)[0]
        return jax.tree_util.tree_map(lambda a: a + 0.0 * _extra_solve("solve", anchor, 8), x), report

    _patch(patch, "mime.nodes.flow.fvm._solve", "guarded_cg", planted)


def inside_exempt(patch: Any) -> None:
    """A raw 30 x 30 solve written inside the near field's own-blocks scope, an exempt site of six unknowns."""
    original = jnp.linalg.inv

    def planted(a: Any) -> Any:
        found = original(a)
        return found + 0.0 * _extra_solve("solve", found, 30)

    patch.setattr(jnp.linalg, "inv", planted)


def direct_in_step(form: str) -> Callable[[Any], None]:
    """A direct solve of ``form`` in a node's step: every node's ``advance`` adds one to what it returns."""
    def apply(patch: Any) -> None:
        import importlib

        for module, cls in (("mime.nodes.flow.fvm._node", "ImplicitFlowNode"),
                            ("mime.nodes.hydro.near_field._node", "JointDragNode")):
            node_class = getattr(importlib.import_module(module), cls)
            advance = node_class.advance

            def planted(self: Any, *args: Any, _advance: Any = advance, **kwargs: Any) -> Any:
                out = dict(_advance(self, *args, **kwargs))
                key = next(k for k, v in out.items() if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating))
                first = jnp.asarray(out[key])
                out[key] = first + 0.0 * _extra_solve(form, first)
                return out

            patch.setattr(node_class, "advance", planted)

    return apply


type Plant = tuple[Callable[[Any], None], str, tuple[str, ...], str]

DENSE_CASES = ("JointDragNode/dense", "OpenFlowNode/two openings")
STEPS = ("ImplicitFlowNode/steady", "JointDragNode/dense")
POISSON_CASES = ("ImplicitFlowNode/steady", "ExplicitFlowNode/rk3", "OpenFlowNode/two openings")
#: The plants by name: what applies it, the check that must catch it (``problems``, ``downcasts`` or ``narrow``), the
#: cases whose programs it touches (each is caught on every one of them), and the text its finding must hold (so that a
#: plant is caught for the reason it is planted, not by another check).
PLANTS: dict[str, Plant] = {
    "bypass_dense": (bypass_dense, "problems", DENSE_CASES, "outside every registered solve scope"),
    "direct_solve_in_step": (direct_in_step("solve"), "problems", STEPS, "outside every registered solve scope"),
    "direct_lstsq_in_step": (direct_in_step("lstsq"), "problems", STEPS, "svd outside every registered solve scope"),
    "direct_pinv_in_step": (direct_in_step("pinv"), "problems", STEPS, "svd outside every registered solve scope"),
    "direct_tridiagonal_in_step": (direct_in_step("tridiagonal"), "problems", STEPS,
                                   "tridiagonal_solve outside every registered solve scope"),
    "raw_poisson": (raw_poisson, "problems", ("ImplicitFlowNode/steady", "ExplicitFlowNode/rk3"),
                    "fft"),
    "nested_in_cg": (nested_in_cg, "problems", ("ImplicitFlowNode/steady", "OpenFlowNode/two openings"),
                     "inside 'fvm_cg' (kind 'cg'), which owns only"),
    "inside_exempt": (inside_exempt, "problems", ("JointDragNode/dense",), "inside the exempt site 'near_field_own_blocks'"),
    "downcast_dense": (downcast_dense, "downcasts", DENSE_CASES, "float32"),
    "downcast_cg": (downcast_cg, "downcasts", ("ImplicitFlowNode/steady", "OpenFlowNode/two openings"), "float32"),
    "downcast_gmres": (downcast_gmres, "downcasts", ("JointDragNode/gmres",), "float32"),
    "downcast_poisson": (downcast_poisson, "downcasts", POISSON_CASES, "float32"),
    "rounded_before_dense": (rounded_before_dense, "narrow", DENSE_CASES, "float32"),
    "rounded_after_dense": (rounded_after_dense, "narrow", DENSE_CASES, "float32"),
}


def planted_from_environment() -> str | None:
    return os.environ.get(PLANT_VAR) or None
