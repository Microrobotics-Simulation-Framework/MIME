"""The base of every MIME node: MADDENING's ``SimulationNode``, with every constructor argument kept in ``params``."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

import jax
import jax.numpy
import numpy as np
from jax import Array
from maddening.core.node import BoundaryInputSpec, SimulationNode

from mime.core._symbols import Symbol

if TYPE_CHECKING:
    from mime.core.ledger import LedgerFlag

#: ``jax.numpy``'s hints name a device type from jaxlib's compiled core, which ships no stubs, so a strict checker
#: reads its array constructors as partly unknown; nodes call them through this name, and ``jax.jit`` through ``jit``.
jnp: Any = jax.numpy
jit: Any = cast(Any, jax).jit


def run_dtype() -> Any:
    """The floating dtype of the run: float64 under ``jax_enable_x64``, float32 otherwise."""
    return jnp.zeros(()).dtype


def as_run_array(value: Any) -> Any:
    return jnp.asarray(value, dtype=run_dtype())


#: The most numbers a parameter the save's resolution rerun moves holds (:func:`one_ulp_up_leaf`): a body's or a
#: medium's constants, a frame, an inertia tensor. A table or a precomputed array is kept: moving it would copy it to
#: every call, and the constants already move the rounding of every step that reads the body.
NUDGED_SIZE = 64


def one_ulp_up_leaf(value: Any) -> np.ndarray | None:
    """A parameter leaf as the save's resolution rerun moves it (``mime.build._maddening.params_one_ulp_up``): each
    nonzero, finite element one unit in the last place up, in its own dtype; ``None`` where the rerun keeps it (not
    floating, or more than :data:`NUDGED_SIZE` numbers)."""
    held = np.asarray(value)
    if held.dtype.kind != "f" or held.size > NUDGED_SIZE:
        return None
    up = np.nextafter(held, np.asarray(np.inf, dtype=held.dtype))
    return np.where((held != 0) & np.isfinite(held), up, held).astype(held.dtype)


#: The key a run adds to a node's step constants where it made that node's run-constants
#: (:meth:`MimeNode.run_constants`) for the chunk: an empty tuple, which holds no array and changes the constants'
#: structure, so that the node's compiled step (``MimeNode._compiled``, traced once per structure) keeps the step that
#: reads them as given apart from the one that makes them.
RUN_CONSTANTS_GIVEN = "run_constants_given"

#: The nodes whose run-constants the enclosing chunk made, while its step loop is traced (:func:`run_constants_made`).
_MADE: ContextVar[frozenset[str]] = ContextVar("mime_run_constants_made", default=frozenset())


@contextmanager
def run_constants_made(names: Iterable[str]) -> Generator[None]:
    """For the block, the steps of the nodes ``names`` traced as a chunk's loop traces them, their run-constants made
    once before it and given in their ``params``: each such node's step constants then hold
    :data:`RUN_CONSTANTS_GIVEN`. Read where a node's step is called, never inside its compiled step."""
    token = _MADE.set(frozenset(names))
    try:
        yield
    finally:
        _MADE.reset(token)


def run_constants_given(p: Mapping[str, Any]) -> bool:
    """Whether the step constants ``p`` hold the node's run-constants as its chunk made them, to be read as given."""
    return RUN_CONSTANTS_GIVEN in p


#: The key a differentiated run (``mime.build._maddening.differentiated_run``) adds to every node's step constants: an
#: empty tuple, as :data:`RUN_CONSTANTS_GIVEN` is, so that the node's compiled step traced inside a derivative is kept
#: apart from a forward run's, which never holds it. A guard that stops a forward run by writing NaN writes a finite
#: stand-in where it is set, and raises the same flag (:func:`differentiated`).
DIFFERENTIATED = "differentiated"

#: The nodes whose steps the enclosing differentiated run traces (:func:`differentiated_nodes`).
_DIFFERENTIATED: ContextVar[frozenset[str]] = ContextVar("mime_differentiated", default=frozenset())


@contextmanager
def differentiated_nodes(names: Iterable[str]) -> Generator[None]:
    """For the block, the steps of the nodes ``names`` traced inside a derivative: each such node's step constants then
    hold :data:`DIFFERENTIATED`. Read where a node's step is called, never inside its compiled step."""
    token = _DIFFERENTIATED.set(frozenset(names))
    try:
        yield
    finally:
        _DIFFERENTIATED.reset(token)


def differentiated(p: Mapping[str, Any]) -> bool:
    """Whether the step constants ``p`` are a differentiated run's: a guard that stops a forward run by writing NaN
    writes a finite stand-in here, with its flag, since a NaN in a step leaks into every derivative through the branch
    a ``where`` discards."""
    return DIFFERENTIATED in p


#: The key a recheck (``mime.study``'s gradient, on a member whose solve stopped short) adds to the step constants of the
#: nodes it re-solves: an empty tuple, as :data:`DIFFERENTIATED` is, so that the node's compiled step traced for the
#: recheck is kept apart from the run's. A node with an iterative solve reads it to remove the cause of what stopped it
#: short (:func:`rechecked`): the cap lifted, the stall restarts raised.
RECHECKED = "rechecked"

#: What a recheck names as the cause of a node's stop short (``mime.study``'s gradient, from the recheck's own solves and
#: the member's): ``floor-limited``, its recheck's solves stopped stagnated, and the node's solve stops a recheck's
#: stagnation only at its rounding floor (:attr:`MimeNode.floor_stops`); ``stagnated``, they stopped stagnated where the
#: node's solve does not test its floor, or the member's own had stagnated and the recheck's, past the stall, converged
#: or reached the lifted cap; ``cap-limited``, a cap alone stopped them; ``not-finite``, a recheck's solve had a
#: residual that was not finite.
CAP_LIMITED, FLOOR_LIMITED, STAGNATED, NOT_FINITE = "cap-limited", "floor-limited", "stagnated", "not-finite"

#: The nodes the enclosing recheck re-solves (:func:`rechecked_nodes`).
_RECHECKED: ContextVar[frozenset[str]] = ContextVar("mime_rechecked", default=frozenset())


@contextmanager
def rechecked_nodes(names: Iterable[str]) -> Generator[None]:
    """For the block, the steps of the nodes ``names`` traced as a recheck's: each such node's step constants then hold
    :data:`RECHECKED`. Read where a node's step is called, never inside its compiled step."""
    token = _RECHECKED.set(frozenset(names))
    try:
        yield
    finally:
        _RECHECKED.reset(token)


def rechecked(p: Mapping[str, Any]) -> bool:
    """Whether the step constants ``p`` are a recheck's: the node's iterative solve removes the cause of a stop short
    (its cap, its restarts) rather than keeping the run's settings."""
    return RECHECKED in p


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
    the equations' one home) and ``symbols``, the rows joining each symbol of those equations to the code.

    **The energy ledger's hooks** (``mime.core.ledger``). A node that writes its step identity (a dynamics: its
    storage, and its step's terms) names its own scheme's numerical terms in :attr:`LEDGER_TERMS` and gives
    :meth:`storage` and :meth:`identity_terms`; a potential law gives its family's :meth:`energy`. Any node takes
    ``ledger``, the deliveries it is metered for, which only the build gives it, only where the ledger is attached:
    it is then a constant of the node, read while its step is traced, and without it the node is the node it always
    was, its parameters included."""

    symbols: ClassVar[tuple[Symbol, ...]] = ()
    #: Whether a guard of the node stops a forward run by writing NaN: such a node's step constants hold
    #: :data:`DIFFERENTIATED` inside a differentiated run, where it writes the guard's flag form (:func:`differentiated`).
    nan_stops: ClassVar[bool] = False
    #: Whether the node's iterative solve reads :func:`rechecked`: a recheck of a member whose solve stopped short
    #: re-solves it with the cause removed (the cap lifted, the restarts raised). A solving node that does not is
    #: reported as not rechecked.
    rechecks: ClassVar[bool] = False
    #: Whether, in a recheck, the node's solve stops a stagnation only where its residual is at its rounding floor
    #: (``eps || |A| |x| + |b| ||``), running on past a stall above it: then a recheck that stagnates again is named
    #: ``floor-limited``, and otherwise ``stagnated``, the floor untested.
    floor_stops: ClassVar[bool] = False
    #: The names of the node's own scheme's numerical terms in its step identity, in the order its records keep them;
    #: ``None`` for a node that writes no step identity, which the ledger cannot meter.
    LEDGER_TERMS: ClassVar[tuple[str, ...] | None] = None
    #: The ratios of the node's metered step that flag known bugs (``mime.core.ledger.LedgerFlag``), each with its
    #: line and the planted bugs that prove it: read by ``mime.testing``'s ledger harness, never by the step.
    LEDGER_FLAGS: ClassVar[tuple[LedgerFlag, ...]] = ()
    #: The commands a metered node pairs as loads read at the step's start (a dynamic arm's joint torques): the
    #: ledger moves a command into one of these into the input of its own beside it, and books its work under its
    #: writer, which is outside.
    PAIRED_COMMANDS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, name: str, timestep: float, **params: object) -> None:
        ledger = params.pop("ledger", ())
        if not isinstance(ledger, Sequence) or isinstance(ledger, str):
            raise TypeError(f"ledger is the deliveries a node is metered for, a sequence of strings, not {ledger!r}")
        if ledger and type(self).LEDGER_TERMS is None:
            raise TypeError(f"{type(self).__name__} writes no step identity (LEDGER_TERMS), so the ledger cannot "
                            f"meter it")
        held: Any = super()
        held.__init__(name, timestep, **params,
                      **({"ledger": tuple(str(d) for d in cast(Sequence[object], ledger))} if ledger else {}))

    @property
    def metered(self) -> bool:
        """Whether the node was built with the ledger attached."""
        return bool(self.constants().get("ledger"))

    def storage(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array:
        """The energy the node stores at ``state`` (J), from its constants ``p``: what its step identity's change in
        storage is taken between, the states its step takes and returns."""
        raise NotImplementedError(f"{type(self).__name__} states no storage")

    def hosted_laws(self) -> tuple[tuple[str, str, str], ...]:
        """The laws a metered node holds in its own step beside what is delivered to it (a dynamic arm's weight,
        damping, stops and friction), each ``(role, law, class)``: the ledger books each one's work under its own name
        (``law``) and class. None by default: a body's contact is held in its step by the law that confines it."""
        return ()

    def identity_terms(self, p: Mapping[str, Any], held: Sequence[Any], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: Any,
                       dt: float) -> Any:
        """The terms of the node's step identity (a ``mime.core.ledger.StepAccount``) over one step from ``state`` to
        ``ended``: each of the ``held`` deliveries paired by the one rule from its own input in ``boundary_inputs``,
        what the step left for its books in ``parts``, and its own scheme's terms."""
        raise NotImplementedError(f"{type(self).__name__} writes no step identity")

    def configuration(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array | None:
        """What a potential law's energy is evaluated at on this receiver (a body's pose), from its ``state``: what the
        ledger's records keep at the run's start, so that a potential's change over a run is taken from the run's two
        ends with the law's own parameters. ``None`` for a node no potential law acts on."""
        return None

    def energy(self, p: Mapping[str, Any], configuration: Array) -> Array:
        """A potential law's energy (J) at its receiver's ``configuration`` (a body's pose), from its constants ``p``:
        its family's energy, of which its delivery is the negative gradient."""
        raise NotImplementedError(f"{type(self).__name__} states no energy")

    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Array]:
        """The node's run-constants: quantities made from its leaves (``params``, the node's entry of the tree the run
        is given) that are constant over a run, each by the ``params`` key it replaces, with that leaf's shape and
        dtype. MIME's chunked runs make them once per chunk inside the compiled program, before its step loop, and give
        them to the loop's steps in place of those leaves (:func:`run_constants_given` is then true of the step's
        constants), so that reverse mode derives each once per chunk rather than at every step. A step called
        otherwise (MADDENING's own scan, ``graph.step``) makes them itself. None by default."""
        return {}

    def made_again(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The parameters the node makes from its others and holds among them, made again from ``params`` (its leaves
        as the graph would hold them) where those are not the build's: what the save's resolution rerun gives the
        step (``mime.build._maddening.params_one_ulp_up``), so that the step reads each rather than making it at every
        step. None by default."""
        return {}

    def recheck_route(self, cause: str, dtype: str) -> str:
        """What takes the node's solve past a stop short a recheck named ``cause`` (:data:`FLOOR_LIMITED`,
        :data:`STAGNATED`, ...) in a run in ``dtype``; empty where the node names none."""
        del cause, dtype
        return ""

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

    def _given(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """The step's constants, marked where the enclosing chunk made this node's run-constants."""
        held = self._step_constants(params)
        if params is not None and self.name in _MADE.get():
            held[RUN_CONSTANTS_GIVEN] = ()
        if params is not None and self.nan_stops and self.name in _DIFFERENTIATED.get():
            held[DIFFERENTIATED] = ()
        if params is not None and self.name in _RECHECKED.get():
            held[RECHECKED] = ()
        return held

    @override
    def update(self, state: dict[str, Array], boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> dict[str, Array]:
        return self._compiled(self._given(params), dict(state), dict(boundary_inputs), dt)

    def fresh_update(self, state: dict[str, Array], boundary_inputs: Mapping[str, Array], dt: float, *,
                     params: Mapping[str, Any] | None = None) -> dict[str, Array]:
        """``update`` with its step traced afresh at every call, never taken from the compiled step's cache: what
        reads a switch context (``mime.core.switch``) traces through it, since a context is read while the step is
        traced, and the compiled step was traced once."""
        return self.advance(self._given(params), dict(state), dict(boundary_inputs), dt)

    def advance(self, p: Mapping[str, Any], state: dict[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> dict[str, Array]:
        """The step: the new state from ``p`` (the numeric constants), the state, the inputs and the timestep."""
        raise NotImplementedError
