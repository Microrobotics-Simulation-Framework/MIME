"""The forward-model contract (an apparatus entry's capability, ``forward_model``): what a field provider or a chain
segment computes from its commands, as a pure function in ``jax.numpy``, for a controller's model of it
(``ctx.believed``) and for the allocation (``ctx.allocator``).

``fn(u, x, theta) -> {quantity: value}``, each a mapping of arrays in SI by name: ``u`` the commands the entry takes
(its own command names), ``x`` where and on what it is evaluated (a world point, a moment the force and torque are
stated on, the world pose of the frame the entry is mounted on), and ``theta`` its parameters, the numbers its node
holds as leaves, under the node's own names and in its node's shapes. **Every number ``fn`` reads is in ``u``, ``x`` or
``theta``**: what the entry's model is made from (its build context) gives structure alone, names, shapes and what is
mounted where, so that a belief replaces every number of the model by replacing ``theta``. ``affine_in`` names the
commands it is affine in at every ``x`` and ``theta``: an allocation over them is exact. ``misses`` says how a
quantity's miss is measured where it is not the difference of its rows (an orientation's, the rotation between the two),
for the iterative allocation, which takes any model."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import jax

from mime.registry._entries import Signal

jnp: Any = jax.numpy

type Arrays = Mapping[str, Any]


def _none() -> Mapping[str, Callable[[Any], Any]]:
    return MappingProxyType({})


def _no_misses() -> Mapping[str, Callable[[Any, Any, str], Any]]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ForwardModel:
    """An entry's forward model: ``fn(u, x, theta)``, with what it takes and gives. ``commands``, ``inputs`` and
    ``quantities`` hold each of ``u``'s, ``x``'s and the result's names with its units and shape (:class:`Signal`, the
    shapes concrete); ``theta`` the parameters' values in SI as the declaration gives them, and ``theta_units`` their
    units. ``affine_in`` names the commands it is affine in; ``optional`` the inputs a call may leave out, with the
    quantities that need them (a magnetic model's ``moment``: its ``force`` and ``torque`` are then not given).
    ``rows`` gives, for a quantity whose entries are not independent, the independent ones as a vector (a
    current-free field's gradient, symmetric and trace-free: its five), which the allocation's rows are made of; any
    other quantity's rows are its entries, flattened. ``misses`` gives, for a quantity whose miss is not the difference
    of its rows, ``miss(made, wanted, owner) -> rows``: how far what the commands make is from what is wanted, zero
    where they meet, its switches the caller's ``owner`` (an orientation's: the rotation vector from the wanted one to
    the one made). ``owner`` is the id of the entry it models (``<apparatus>.<field>``, ``<apparatus>.<segment>``)."""

    owner: str
    commands: Mapping[str, Signal]
    inputs: Mapping[str, Signal]
    quantities: Mapping[str, Signal]
    theta: Mapping[str, Any]
    theta_units: Mapping[str, str]
    fn: Callable[[Arrays, Arrays, Arrays], Mapping[str, Any]]
    affine_in: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()
    rows: Mapping[str, Callable[[Any], Any]] = field(default_factory=_none)
    misses: Mapping[str, Callable[[Any, Any, str], Any]] = field(default_factory=_no_misses)

    def __call__(self, u: Arrays, x: Arrays, theta: Arrays | None = None) -> dict[str, Any]:
        """The quantities at ``u`` and ``x``, at this model's ``theta`` with any of ``theta`` given in its place."""
        held = dict(self.theta) if theta is None else {**self.theta, **theta}
        return dict(self.fn(u, x, held))

    def bound(self, theta: Arrays) -> ForwardModel:
        """This model with ``theta``'s values in place of its own (a belief's numbers, a controller's leaves)."""
        unknown = sorted(set(theta) - set(self.theta))
        if unknown:
            raise KeyError(f"{self.owner}'s model has no parameter {', '.join(unknown)}; it has {sorted(self.theta)}")
        return dataclasses.replace(self, theta=MappingProxyType({**self.theta, **theta}))

    def affine(self, x: Arrays, *, commands: Sequence[str] | None = None, dtype: Any = None
               ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """The model's coefficients in the commands it is affine in (``commands``, all of ``affine_in`` by default) at
        ``x``, and its offset there at zero command: per quantity, per command, an array of the quantity's shape then
        the command's, and per quantity its value at zero command (the other commands at zero too). Exact for an affine
        model: forward-mode derivatives of a function linear in its argument are its coefficients, to the arithmetic of
        the products that make them. ``dtype`` is the commands' (the default float's unless given)."""
        names = tuple(sorted(self.affine_in) if commands is None else commands)
        beyond = [name for name in names if name not in self.affine_in]
        if beyond:
            raise ValueError(f"{self.owner}'s model is not affine in {', '.join(beyond)}; it is affine in "
                             f"{sorted(self.affine_in) or 'none of its commands'}")
        held = jnp.zeros(()).dtype if dtype is None else dtype
        zero = {name: jnp.zeros(signal.shape, dtype=held) for name, signal in self.commands.items()}

        def made(chosen: Arrays) -> dict[str, Any]:
            return self({**zero, **chosen}, x)

        at_zero = {name: zero[name] for name in names}
        coefficients = cast(Mapping[str, Mapping[str, Any]], cast(Any, jax).jacfwd(made)(at_zero))
        return {q: dict(by) for q, by in coefficients.items()}, made(at_zero)

    def row_values(self, quantity: str, value: Any) -> Any:
        """``value`` of ``quantity`` as the allocation's rows read it, its leading axes the quantity's and any after
        them kept (a coefficient's command axis): its independent entries along one axis."""
        rows = self.rows.get(quantity)
        held = jnp.asarray(value)
        if rows is not None:
            return rows(held)
        size = len(self.quantities[quantity].shape)
        return jnp.reshape(held, (-1, *held.shape[size:]))

    def miss(self, quantity: str, made: Any, wanted: Any, *, owner: str) -> Any:
        """How far ``made`` is from ``wanted``, of ``quantity``, as rows (a vector): its own miss where the model gives
        one (``misses``), else the difference of their rows. Zero where they meet."""
        own = self.misses.get(quantity)
        if own is not None:
            return jnp.reshape(own(made, wanted, owner), (-1,))
        return self.row_values(quantity, made) - self.row_values(quantity, wanted)


__all__ = ["Arrays", "ForwardModel"]
