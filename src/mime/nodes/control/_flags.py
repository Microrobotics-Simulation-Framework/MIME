"""A controller's run flags: declared by name, raised in its step, held by a node and counted by the run as a law's are.

**Declared.** A controller declares the flags its step raises as any entry does, in its class's ``raises`` (each a
``mime.registry.Flag``, its ``field`` the flag's name); what its law calls declares its own when the law is made
(``ctx.allocator`` on the iterative path declares ``mime.control``'s ``UNCONVERGED``). The build report lists them.

**Raised.** The step calls :func:`raise_flag` with a name it declares and a value, in either mode: the flag's value at
that tick is the largest magnitude of the values raised under its name, zero where none was raised, so a flag raised
with a measure (an unconverged solve's stationarity) carries it. A call outside a run's step (a law's step called by
hand) raises nothing. Refused, each where the build first traces the step (``controller.untraceable``) or at a
callback's first tick (``control.law-raised``), its message naming the fix: a name not declared, and a flag raised
below the step's own level, inside a ``jax.lax`` loop's or branch's body (``cond``, ``while_loop``, ``scan``,
``fori_loop``) or a function transformed on its own, whose values do not leave it. A step its law compiled with
``jax.jit`` is traced through the function it compiles (``__wrapped__``); one compiled some other way would hide its
flags, and a controller that declares flags and has such a step is refused (:func:`hidden_step`).

**Held and counted.** Inside the compiled step the controller's node holds each as its state ``flag_<name>``; as a
callback, a node of its own, ``<controller>.flags``, holds what the callback raised at each tick. Either holds it until
the next tick, and the run lays it out under one path in both modes, ``<controller>.<name>``, and counts it as it
counts a law's flags: the steps it was raised at, the first and the last, with a warning after a run that raised one
that ``warns``. A controller that declares none builds nothing, and every program is the one it was."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, NamedTuple, cast

import jax
import jax.extend.core

jnp: Any = jax.numpy
#: The trace a call runs at, as a handle compared for identity (JAX's hint reads as partly unknown).
_trace_state: Callable[[], object] = cast(Any, jax.extend.core).get_opaque_trace_state

#: The prefix of the node field that holds a controller's flag.
PREFIX = "flag_"
#: The node a callback's flags are held by, beside the controller's id: ``<controller>.flags``.
HOLDER = "flags"

class _Gathering(NamedTuple):
    """The flags a step may raise, each with the values raised so far, and the trace the step is called at."""

    held: dict[str, list[Any]]
    level: Any


_raised: ContextVar[_Gathering | None] = ContextVar("mime_controller_flags", default=None)


@contextmanager
def raising(names: Iterable[str]) -> Generator[dict[str, list[Any]]]:
    """Gather the flags ``names`` a step raises while it is called inside, by name, entered at the level the step is
    called at (inside the function a trace compiles, never around it): a name outside them, or a raise below that
    level, is refused."""
    held: dict[str, list[Any]] = {name: [] for name in names}
    token = _raised.set(_Gathering(held, _trace_state()))
    try:
        yield held
    finally:
        _raised.reset(token)


def gathered(step: Callable[..., Any], names: Iterable[str]) -> Callable[..., Any]:
    """``step`` with its flags ``names`` gathered at its own level and dropped: for a trace that checks the step, where
    a refused raise stops the trace with its reason."""
    held = tuple(names)

    def called(*args: Any) -> Any:
        with raising(held):
            return step(*args)

    return called


def raise_flag(name: str, value: Any) -> None:
    """Raise the controller's flag ``name`` at this tick wherever ``value`` is not zero (a bool, or a measure that the
    flag carries): called in a controller's step at its own level, or in what it calls there. Outside a run's step it
    does nothing."""
    gathering = _raised.get()
    if gathering is None:
        return
    if name not in gathering.held:
        raise ValueError(f"raise_flag({name!r}): the controller declares no flag {name!r} (it declares "
                         f"{', '.join(sorted(gathering.held)) or 'none'}): declare `{name}` in the controller's "
                         f"`raises` (a `Flag(field=..., meaning=..., warns=...)`)")
    if _trace_state() != gathering.level:
        raise ValueError(f"raise_flag({name!r}) was called inside a jax.lax.cond, while_loop, scan or fori_loop body "
                         f"(or a function transformed on its own), whose values do not leave it: raise at the step's "
                         f"own level; return the value out of the branch or loop and raise it there")
    gathering.held[name].append(value)


def hidden_step(step: Any) -> bool:
    """Whether a law's step is compiled so that a flag raised inside it cannot be seen: compiled by JAX (it lowers, or
    it is a compiled stage) with no function underneath it (``__wrapped__``) to trace with the flags gathered."""
    if isinstance(step, jax.stages.Compiled):
        return True
    return hasattr(step, "lower") and inspect.unwrap(step) is step


def flag_values(held: Mapping[str, list[Any]], dtype: Any) -> dict[str, Any]:
    """Each flag's value at a tick, by its node field: the largest magnitude raised under its name, zero where none."""
    found: dict[str, Any] = {}
    for name, values in held.items():
        value = jnp.zeros((), dtype=dtype)
        for raised in values:
            value = jnp.maximum(value, jnp.max(jnp.abs(jnp.asarray(raised, dtype=dtype))))
        found[f"{PREFIX}{name}"] = value
    return found


def declared_flags(entry_class: type, made: Iterable[Any] = ()) -> tuple[Any, ...]:
    """A controller's flags (each a ``mime.registry.Flag``): its class's ``raises`` and those its law declared when it
    was made, each once, by name."""
    found: dict[str, Any] = {}
    for flag in (*getattr(entry_class, "raises", ()), *made):
        found.setdefault(str(flag.field), flag)
    return tuple(found[name] for name in sorted(found))


def holder_step(names: Iterable[str]) -> Any:
    """The step of a callback's flag holder: it raises again each flag ``flag_<name>`` it is fed."""
    held = tuple(names)

    def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        for name in held:
            raise_flag(name, reads[f"{PREFIX}{name}"])
        return {}, {}

    return step


__all__ = ["HOLDER", "PREFIX", "declared_flags", "flag_values", "gathered", "hidden_step", "holder_step", "raise_flag",
           "raising"]
