"""The bake-check spike's plants, as entries: a spring on the toy probe whose node, by ``kind``, adds to its stiffness
``k`` (a law setting, so a two-builds pair moves it as a study's axis would) a term made from ``k``, either when the
node is built (a partial bake: ``k`` is still read live in the first term) or in the step (its twin). The kinds:

- ``float`` / ``float_twin``: ``0.25 k^2`` (about 0.3 of the load at ``k`` 1.7), a Python float;
- ``array`` / ``array_twin``: ``0.25 k^2 (1, 1.1, 0.9)``, a NumPy array;
- ``jax``: the same, a JAX array (no twin: the twin is ``array_twin``);
- ``tiny`` / ``tiny_twin``: ``1e-6 k``, a term of share 1e-6;
- ``s7`` / ``s7_twin``: S7's form, ``1e-4 k / eta`` with ``eta = (0.3 h)^2 / k``, ``h`` 1/8;
- ``count``: ``0.1 ceil(4 k) / 4``, from an array of ``ceil(4 k)`` entries: a value-keyed shape (no twin possible).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
import numpy as np

from mime.build import InteractionBuilder
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, Read, RoleKind, register_interaction
from mime.testing.toys._entries import LOAD, POSITION
from mime.testing.toys._experiments import ADD_BATH, ADD_PROBE, toy_experiment
from mime.testing.toys._nodes import ToyLoad, ToySpringInputs, ToySpringNode
from mime.units import Force, Length, N, Quantity, m
from mime.vocabulary import register_namespace

NAMESPACE = "twobuilds"
register_namespace(NAMESPACE, "0.1.0")

W = np.array([1.0, 1.1, 0.9])
#: S7's ``(delta h)^2`` at delta 0.3 and h 1/8.
H2 = (0.3 * 0.125) ** 2
X0 = (0.05, -0.02, 0.01)


class _Planted(ToySpringNode):
    """A toy spring whose stiffness is ``k`` plus :meth:`extra`, made from ``k``: in the step from the leaf, or held
    from the build where :attr:`HELD` is."""

    HELD = False

    def __init__(self, name: str, timestep: float, k: float = 1.7, x0: Sequence[float] = X0) -> None:
        super().__init__(name, timestep, k=k, x0=x0)
        self.built_k = float(k)

    def extra(self, k: Any) -> Any:
        raise NotImplementedError

    @override
    def law(self, p: Mapping[str, Any], given: ToySpringInputs) -> ToyLoad:
        made = self.extra(self.built_k if self.HELD else p["k"])
        return super().law({**p, "k": p["k"] + made}, given)


class FloatTwin(_Planted):
    def extra(self, k: Any) -> Any:
        return 0.25 * k ** 2


class FloatBaked(FloatTwin):
    HELD = True


class ArrayTwin(_Planted):
    def extra(self, k: Any) -> Any:
        return 0.25 * k ** 2 * (W if self.HELD else jnp.asarray(W))


class ArrayBaked(ArrayTwin):
    HELD = True


class JaxBaked(_Planted):
    HELD = True

    def extra(self, k: Any) -> Any:
        return jnp.asarray(0.25 * k ** 2) * jnp.asarray(W)


class TinyTwin(_Planted):
    def extra(self, k: Any) -> Any:
        return 1e-6 * k


class TinyBaked(TinyTwin):
    HELD = True


class S7Twin(_Planted):
    def extra(self, k: Any) -> Any:
        eta = H2 / k
        return 1e-4 * k / eta


class S7Baked(_Planted):
    """``eta`` held from the build, ``k`` in the penalty read live: S7's pattern."""

    def __init__(self, name: str, timestep: float, k: float = 1.7, x0: Sequence[float] = X0) -> None:
        super().__init__(name, timestep, k=k, x0=x0)
        self.eta = H2 / float(k)

    @override
    def law(self, p: Mapping[str, Any], given: ToySpringInputs) -> ToyLoad:
        return ToySpringNode.law(self, {**p, "k": p["k"] + 1e-4 * p["k"] / self.eta}, given)


class CountKeyed(_Planted):
    HELD = True

    def extra(self, k: Any) -> Any:
        return jnp.sum(jnp.asarray(np.full(int(math.ceil(4.0 * k)), 0.1 / 4.0)))


NODES: dict[str, type[_Planted]] = {
    "float": FloatBaked, "float_twin": FloatTwin, "array": ArrayBaked, "array_twin": ArrayTwin, "jax": JaxBaked,
    "tiny": TinyBaked, "tiny_twin": TinyTwin, "s7": S7Baked, "s7_twin": S7Twin, "count": CountKeyed}
BAKED = ("float", "array", "jax", "tiny", "s7")
TWINS = ("float_twin", "array_twin", "tiny_twin", "s7_twin")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PlantedTerm(Interaction):
    """A spring towards ``x0`` whose node adds a term made from ``k``, by ``kind`` (the module's docstring)."""

    name = "twobuilds.spring"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": tuple(NODES.values())}
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    references = ()
    kind: str = "float"
    k: Annotated[Quantity, Force / Length] = N(1.7) / m(1.0)

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        cls = NODES[self.kind]
        node = b.add_response_node(cls(ctx.node_name(), ctx.timestep, k=ctx.scalar("k")))
        b.connect(b.read("x"), into=node[cls.READS])
        b.connect(node[cls.OUTPUT], into=b.target("load"))


def planted(kind: str, k: float = 1.7) -> tuple[Any, Any]:
    """The toy probe in its bath, pulled by a planted spring of ``kind`` at stiffness ``k`` N/m, and the path of the
    spring's ``k`` as a study names it."""
    exp = toy_experiment(f"two_builds_{kind}")
    ADD_BATH.apply(exp)
    ADD_PROBE.apply(exp)
    spring = exp.attach(PlantedTerm(kind=kind, k=N(k) / m(1.0)), body="probe")
    return exp, spring.k


__all__ = ["BAKED", "NODES", "TWINS", "PlantedTerm", "planted"]
