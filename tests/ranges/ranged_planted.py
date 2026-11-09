"""Planted entries for the declared ranges' controls: a constant wrench on a body whose node holds a table of ``count``
equal shares (summing to one, so the wrench is the same whatever the count), the count chosen from the body's hull by
``mode``:

- ``clean``: over the range, monotone, the count growing with the radius (``choose_over`` with
  ``ChoiceRule.Monotone``);
- ``peak`` and ``peak-sampled``: a count that peaks inside the range, declared monotone (its corners miss the peak) or
  sampled at nine points a setting (which find it);
- ``unruled``: a choice over the range that declares no rule;
- ``own``: from the body's own radius, the range ignored (a rung as dev1's was: it splits a sweep);
- ``unreported``: over the range, plus a share sized from the range's width read past ``choose_over`` (a dependence the
  range forbids);
- ``bake``: over the range, its wrench scaled again by a Python float made from the body's own radius at build (a value
  baked into the step);
- ``array-bake``: the same with a NumPy array made from the radius at build, which the node's step (inside the ``jit``
  a ``MimeNode`` runs its step in) closes over: a constant of that nested sub-program, not of the top-level one;
- ``deaf``: over the range, the node built at a radius of its own, so that a move of the body's changes nothing it
  holds;
- ``unread`` and ``bake-unread``: over the range, a node that never reads its radius leaf (a move of the radius writes a
  leaf the step does not read), and in ``bake-unread`` scales its wrench by a Python float made from the radius at
  build as ``bake`` does: a bake beside an unread leaf.

The node reads the radius as a leaf, so that a sweep of it changes what each member computes.

Registered under the ``rangetoy`` namespace, in this process only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import numpy as np
from jax import Array
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from maddening.core.static_data import StaticArray
from typing_extensions import ReadOnly, TypedDict

from mime.build import InteractionBuilder, entry_id
from mime.core import MimeNode, NodeFields, as_run_array, node_fields
from mime.gradients import GradientStatus
from mime.registry import (
    BuildContext,
    Delivery,
    Hull,
    Interaction,
    ChoiceRule,
    Plan,
    PlanContext,
    RoleKind,
    choose_over,
    register_interaction,
)
from mime.vocabulary import APPLIED_WRENCH, register_namespace

register_namespace("rangetoy", "0.0.1")

#: The radius the planted peak sits at, inside the ranges the tests declare ([1.0, 1.15] mm).
PEAK = 1.075e-3
#: The radius the ``deaf`` plant's node is built at, whatever the body's.
DEAF_RADIUS = 1.0e-3
WRENCH = (1e-9, -2e-9, 0.5e-9, 1e-15, 0.0, -1e-15)


def growing(hull: Hull) -> float:
    return float(int(float(hull.settings["radius"]) * 1e5))


def peaked(hull: Hull) -> float:
    radius = float(hull.settings["radius"])
    return float(50 + int(1e10 * (1.15e-3 - radius) * (radius - 1.0e-3)))


class _Wrench(TypedDict, closed=True):
    value: ReadOnly[Array]


class SharesNode(MimeNode):
    """``value`` times the sum of its ``count`` equal shares (a static table), times the body's radius in mm (a leaf
    it reads), ``kept`` (a Python float, one unless baked) and ``held`` (a NumPy array, none unless baked): the wrench
    it delivers."""

    def __init__(self, name: str, timestep: float, *, value: tuple[float, ...], count: int, radius: float,
                 kept: float = 1.0, held: np.ndarray | None = None) -> None:
        super().__init__(name, timestep, value=value, count=int(count), radius=float(radius))
        self.kept = float(kept)
        self.held = held

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "radius": ParamSpec(units="m")}

    @property
    @override
    def static_data(self) -> dict[str, Any]:
        count = int(self.constants()["count"])
        return {"shares": StaticArray(np.full(count, 1.0 / count))}

    def _value(self, p: Mapping[str, Any]) -> Array:
        shares = jnp.sum(jnp.asarray(self.static_data["shares"].value))
        found = jnp.asarray(p["value"]) * shares * (1e3 * jnp.asarray(p["radius"])) * self.kept
        return as_run_array(found if self.held is None else found * self.held)

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(_Wrench(value=self._value(self.constants())))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {}

    @override
    def advance(self, p: Mapping[str, Any], state: dict[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> dict[str, Array]:
        return node_fields(_Wrench(value=self._value(p)))


class UnreadSharesNode(SharesNode):
    """:class:`SharesNode` whose wrench ignores the radius leaf it declares: ``value`` times its shares and ``kept``."""

    @override
    def _value(self, p: Mapping[str, Any]) -> Array:
        shares = jnp.sum(jnp.asarray(self.static_data["shares"].value))
        return as_run_array(jnp.asarray(p["value"]) * shares * self.kept)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class RangedShares(Interaction):
    name = "rangetoy.shares"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}
    builds = {"": (SharesNode, UnreadSharesNode)}
    references = ()
    mode: str = "clean"

    def _count(self, owner: str, hull: Hull) -> float:
        if self.mode == "own":
            return growing(hull)
        rule: Any = {"peak": ChoiceRule.Monotone("planted: claims monotone, peaks inside"),
                     "peak-sampled": ChoiceRule.Sampled(9, "planted: peaks inside, sampled"),
                     "unruled": None}.get(self.mode, ChoiceRule.Monotone("planted: the count grows with the radius"))
        measure = peaked if self.mode.startswith("peak") else growing
        chosen = choose_over(owner, hull, "shares", rule, measure)
        count = measure(chosen)
        if self.mode == "unreported" and hull.ranges:
            low, high = hull.ranges["radius"]
            count += float(int((high - low) * 1e5))
        return count

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        hull = ctx.geometry(of="body")
        if hull is not None:
            self._count(entry_id(type(self), ctx.bindings), hull)
        return Plan(delivers=dict(self.delivers))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        hull = ctx.geometry(of="body")
        assert hull is not None
        count = self._count(ctx.id, hull)
        radius = DEAF_RADIUS if self.mode == "deaf" else float(hull.settings["radius"])
        kept = 1e3 * radius if self.mode in ("bake", "bake-unread") else 1.0
        held = np.full(6, 1e3 * radius) if self.mode == "array-bake" else None
        node_class = UnreadSharesNode if self.mode in ("unread", "bake-unread") else SharesNode
        node = b.add_response_node(node_class(ctx.node_name(), ctx.timestep, value=WRENCH, count=int(count),
                                              radius=radius, kept=kept, held=held))
        b.connect(node["value"], into=b.target("wrench"))


__all__ = ["DEAF_RADIUS", "PEAK", "WRENCH", "RangedShares", "SharesNode", "UnreadSharesNode", "growing", "peaked"]
