"""A controller's run flags: declared by name, raised in its step, held by a node and counted by the run as a law's are.

**Declared.** A controller declares the flags its step raises as any entry does, in its class's ``raises`` (each a
``mime.registry.Flag``, its ``field`` the flag's name); what its law calls declares its own when the law is made
(``ctx.allocator`` on the iterative path declares ``mime.control``'s ``UNCONVERGED``). The build report lists them.

**Raised.** The step calls :func:`raise_flag` with a name it declares and a value, in either mode: the flag's value at
that tick is the largest magnitude of the values raised under its name, zero where none was raised, so a flag raised
with a measure (an unconverged solve's stationarity) carries it. A call outside a run's step (a law's step called by
hand) raises nothing; a name not declared is refused where the build first traces the step. A flag is raised at the
step's own level of control flow, never inside a ``jax.lax`` loop's or branch's body, whose values do not leave it.

**Held and counted.** Inside the compiled step the controller's node holds each as its state ``flag_<name>``; as a
callback, a node of its own, ``<controller>.flags``, holds what the callback raised at each tick. Either holds it until
the next tick, and the run lays it out and counts it as it counts a law's flags: the steps it was raised at, the first
and the last, with a warning after a run that raised one that ``warns``. A controller that declares none builds
nothing, and every program is the one it was."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import jax

jnp: Any = jax.numpy

#: The prefix of the node field that holds a controller's flag.
PREFIX = "flag_"
#: The node a callback's flags are held by, beside the controller's id: ``<controller>.flags``.
HOLDER = "flags"

_raised: ContextVar[dict[str, list[Any]] | None] = ContextVar("mime_controller_flags", default=None)


@contextmanager
def raising(names: Iterable[str]) -> Generator[dict[str, list[Any]]]:
    """Gather the flags ``names`` a step raises while it is called inside, by name; a name outside them is refused."""
    held: dict[str, list[Any]] = {name: [] for name in names}
    token = _raised.set(held)
    try:
        yield held
    finally:
        _raised.reset(token)


def raise_flag(name: str, value: Any) -> None:
    """Raise the controller's flag ``name`` at this tick wherever ``value`` is not zero (a bool, or a measure that the
    flag carries): called in a controller's step, or in what it calls. Outside a run's step it does nothing."""
    held = _raised.get()
    if held is None:
        return
    if name not in held:
        raise ValueError(f"raise_flag({name!r}): the controller declares no flag {name!r} (it declares "
                         f"{sorted(held) or 'none'}): declare it in its class's raises, Flag({name!r}, meaning=...)")
    held[name].append(value)


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


__all__ = ["HOLDER", "PREFIX", "declared_flags", "flag_values", "holder_step", "raise_flag", "raising"]
