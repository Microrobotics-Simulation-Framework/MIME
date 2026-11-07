"""Planted entries for the study's controls: a spring on the toy body whose node, by ``kind``, either derives a static
array from its stiffness and declares it (``static``), consumes the stiffness when it is built and never reads the
leaf (``baked``), keeps the stiffness in a Python attribute alone, its leaf the default whatever the plan gives
(``hidden``), or stores twice the stiffness the plan gives it (``double``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
import numpy as np
from maddening.core.params import ParamSpec
from maddening.core.static_data import StaticArray

from mime.build import InteractionBuilder
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, Read, RoleKind, register_interaction
from mime.testing.toys import ToyBody
from mime.testing.toys._entries import LOAD, POSITION
from mime.testing.toys._nodes import ToyLoad, ToySpringInputs, ToySpringNode, _zero3
from mime.testing.toys._experiments import ADD_BATH, ADD_PROBE, toy_experiment
from mime.things import Body
from mime.units import Force, Length, N, Quantity, m
from mime.vocabulary import register_namespace

NAMESPACE = "studytoy"
register_namespace(NAMESPACE, "0.1.0")


class StaticSpringNode(ToySpringNode):
    """Its law reads a static array built from ``k``, which it declares (``static_data_deps``): the step reads the
    static, a constant of the compiled program, and never the leaf."""

    @property
    @override
    def static_data(self) -> dict[str, Any]:
        return {"table": StaticArray(np.full(3, float(self.params["k"])))}

    @override
    def static_data_deps(self) -> dict[str, tuple[str, ...]]:
        return {"table": ("k",)}

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "k": ParamSpec(trainable=False)}

    @override
    def law(self, p: Mapping[str, Any], given: ToySpringInputs) -> ToyLoad:
        table = jnp.asarray(self.static_data["table"].value)
        return ToyLoad(load=-table * (given.get("x", _zero3()) - jnp.asarray(p["x0"])))


class BakedSpringNode(ToySpringNode):
    """Consumes ``k`` when it is built, and its law reads the Python number it kept, never the leaf: a value baked into
    the trace that it declares nowhere."""

    def __init__(self, name: str, timestep: float, k: float = 1.7, x0: Sequence[float] = (0.05, -0.02, 0.01)) -> None:
        super().__init__(name, timestep, k=k, x0=x0)
        self.kept = float(k)

    @override
    def law(self, p: Mapping[str, Any], given: ToySpringInputs) -> ToyLoad:
        return ToyLoad(load=-self.kept * (given.get("x", _zero3()) - jnp.asarray(p["x0"])))


class HiddenSpringNode(BakedSpringNode):
    """Keeps ``k`` in a Python attribute its law reads, and gives its leaf the default stiffness whatever it is built
    with: no leaf, static or structure holds the value, so a study's key cannot see it and its tree never changes."""

    def __init__(self, name: str, timestep: float, k: float = 1.7, x0: Sequence[float] = (0.05, -0.02, 0.01)) -> None:
        super().__init__(name, timestep, k=1.7, x0=x0)
        self.kept = float(k)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PlantedSpring(Interaction):
    """A spring towards ``x0``, its node by ``kind``: ``static``, ``baked``, ``hidden`` or ``double`` (the node is given
    ``2 k``)."""

    name = "studytoy.spring"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": (ToySpringNode, StaticSpringNode, BakedSpringNode, HiddenSpringNode)}
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    references = ()
    kind: str = "double"
    k: Annotated[Quantity, Force / Length] = N(1.7) / m(1.0)

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        k = ctx.scalar("k")
        cls = {"static": StaticSpringNode, "baked": BakedSpringNode, "hidden": HiddenSpringNode}.get(self.kind,
                                                                                                  ToySpringNode)
        node = b.add_response_node(cls(ctx.node_name(), ctx.timestep, k=2.0 * k if self.kind == "double" else k))
        b.connect(b.read("x"), into=node[cls.READS])
        b.connect(node[cls.OUTPUT], into=b.target("load"))


def planted(kind: str):
    """The toy probe in its bath, pulled by a planted spring of ``kind``."""
    exp = toy_experiment(f"planted_{kind}")
    ADD_BATH.apply(exp)
    ADD_PROBE.apply(exp)
    spring = exp.attach(PlantedSpring(kind=kind), body="probe")
    return exp, spring


__all__ = ["Body", "PlantedSpring", "ToyBody", "planted"]
