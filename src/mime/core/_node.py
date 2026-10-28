"""The base of every MIME node: MADDENING's ``SimulationNode``, with every constructor argument kept in ``params``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, ClassVar, cast, override

import jax
import jax.numpy
import numpy as np
from jax import Array
from maddening.core.node import BoundaryInputSpec, SimulationNode

from mime.core._symbols import Symbol

#: ``jax.numpy``'s hints name a device type from jaxlib's compiled core, which ships no stubs, so a strict checker
#: reads its array constructors as partly unknown; nodes call them through this name, and ``jax.jit`` through ``jit``.
jnp: Any = jax.numpy
jit: Any = cast(Any, jax).jit


def run_dtype() -> Any:
    """The floating dtype of the run: float64 under ``jax_enable_x64``, float32 otherwise."""
    return jnp.zeros(()).dtype


def as_run_array(value: Any) -> Any:
    return jnp.asarray(value, dtype=run_dtype())


def node_input(units: str, shape: tuple[int, ...], *, additive: bool = False,
               description: str = "") -> BoundaryInputSpec:
    """An input declared the way every MIME node declares one: its shape, the run's dtype, a zero default, and its
    units as the vocabulary spells them. ``additive`` is documentation on MADDENING 0.4.x: what sums the deliveries is
    each edge's own flag, which MIME's builders set from the target's kind."""
    return BoundaryInputSpec(shape=shape, dtype=run_dtype(), default=as_run_array(np.zeros(shape)),
                             coupling_type="additive" if additive else "replacive", expected_units=units,
                             description=description)


def _numeric(value: object) -> bool:
    """A constant the step may take traced: a float, an array, or a tuple of numbers. An integer, a string or a
    ``None`` is structural, and stays a Python value the step is traced with."""
    if isinstance(value, bool):
        return False
    if isinstance(value, float) or hasattr(value, "dtype"):
        return True
    return isinstance(value, tuple) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                            for v in cast(tuple[object, ...], value))


class MimeNode(SimulationNode):
    """A MIME node. Every constructor argument is kept in ``params``, which is all of a node MADDENING saves, so a node
    rebuilt from its saved form is the node that was built; an initial condition is named ``initial_*``, which
    MADDENING takes as an initial condition rather than a parameter of the step. Its state and inputs are typed as
    ``mime.nodes``'s README sets out.

    A node writes its step as :meth:`advance`, from its numeric constants (traced, with the graph's live ``params``
    written over them) and its state, inputs and timestep; ``update`` runs it compiled, once per node, so that
    evaluated eagerly, as the verification battery does, a step is one call rather than a hundred small ones, and
    traced into the graph's step it is inlined. Structural constants (strings, integers) are read from
    :meth:`constants` inside it.

    A node documented for its pages carries MADDENING's ``meta`` (a ``NodeMeta``, whose ``governing_equations`` are
    the equations' one home) and ``symbols``, the rows joining each symbol of those equations to the code."""

    symbols: ClassVar[tuple[Symbol, ...]] = ()

    def __init__(self, name: str, timestep: float, **params: object) -> None:
        held: Any = super()
        held.__init__(name, timestep, **params)

    def constants(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The constructor's arguments, with the graph's live ``params`` written over them when given: what the step
        reads, so that a study moving a leaf moves the step."""
        own = cast(dict[str, Any], cast(Any, self).params)
        return dict(own) if params is None else {**own, **params}

    @cached_property
    def _compiled(self) -> Callable[..., dict[str, Array]]:
        return cast(Callable[..., dict[str, Array]], jit(self.advance))

    def _step_constants(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        return {key: value for key, value in self.constants(params).items() if _numeric(value)}

    @override
    def update(self, state: dict[str, Array], boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> dict[str, Array]:
        return self._compiled(self._step_constants(params), dict(state), dict(boundary_inputs), dt)

    def fresh_update(self, state: dict[str, Array], boundary_inputs: Mapping[str, Array], dt: float, *,
                     params: Mapping[str, Any] | None = None) -> dict[str, Array]:
        """``update`` with its step traced afresh at every call, never taken from the compiled step's cache: what
        reads a switch context (``mime.core.switch``) traces through it, since a context is read while the step is
        traced, and the compiled step was traced once."""
        return self.advance(self._step_constants(params), dict(state), dict(boundary_inputs), dt)

    def advance(self, p: Mapping[str, Any], state: dict[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> dict[str, Array]:
        """The step: the new state from ``p`` (the numeric constants), the state, the inputs and the timestep."""
        raise NotImplementedError
