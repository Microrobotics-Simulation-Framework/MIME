"""Planted springs for the gradient path's refusals, on the toy probe in its bath: the node holding the stiffness
squared (a leaf that is not affine in its axis), a copy of it held out of every derivative and read by the step (a
leaf the derivative would drop), or the entry refusing a stiffness above 1.7 N/m (a probe point past a refusal)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

from maddening.core.params import ParamSpec

from mime.build import InteractionBuilder
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, Read, RoleKind, register_interaction
from mime.testing.toys._entries import LOAD, POSITION
from mime.testing.toys._experiments import ADD_BATH, ADD_PROBE, toy_experiment
from mime.testing.toys._nodes import ToyLoad, ToySpringInputs, ToySpringNode, _float, _zero3
from mime.units import Force, Length, Magnitude, N, Quantity, m
from mime.vocabulary import register_namespace

NAMESPACE = "gradtoy"
register_namespace(NAMESPACE, "0.1.0")
#: The bounded spring's largest stiffness, N/m.
BOUND = 1.7


class SquaredSpringNode(ToySpringNode):
    """Holds its stiffness squared, and its law the square root of the leaf: the same spring, its leaf not affine in
    the stiffness."""

    def __init__(self, name: str, timestep: float, k: float = 1.7, x0: Sequence[float] = (0.05, -0.02, 0.01)) -> None:
        super().__init__(name, timestep, k=k * k, x0=x0)

    @override
    def law(self, p: Mapping[str, Any], given: ToySpringInputs) -> ToyLoad:
        return ToyLoad(load=-(p["k"] ** 0.5) * (given.get("x", _zero3()) - _float(p["x0"])))


class HeldSpringNode(ToySpringNode):
    """Holds a copy of its stiffness held out of every derivative (``trainable=False``), which its law reads."""

    def __init__(self, name: str, timestep: float, k: float = 1.7, x0: Sequence[float] = (0.05, -0.02, 0.01)) -> None:
        super(ToySpringNode, self).__init__(name, timestep, k=k, k_held=k, x0=tuple(x0))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "k_held": ParamSpec(trainable=False)}

    @override
    def law(self, p: Mapping[str, Any], given: ToySpringInputs) -> ToyLoad:
        return ToyLoad(load=-(p["k_held"] + 0.0 * p["k"]) * (given.get("x", _zero3()) - _float(p["x0"])))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PlantedGradientSpring(Interaction):
    """A spring towards ``x0``, its node by ``kind``: ``squared``, ``held`` or ``bounded`` (the plain node, a stiffness
    above :data:`BOUND` refused)."""

    name = "gradtoy.spring"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": (ToySpringNode, SquaredSpringNode, HeldSpringNode)}
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    references = ()
    kind: str = "bounded"
    k: Annotated[Quantity, Force / Length] = N(1.7) / m(1.0)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        held = settings["k"]
        if self.kind == "bounded" and isinstance(held, float) and held > BOUND:
            return [f"k is {held!r} N/m, above the planted bound of {BOUND} N/m"]
        return []

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        cls = {"squared": SquaredSpringNode, "held": HeldSpringNode}.get(self.kind, ToySpringNode)
        node = b.add_response_node(cls(ctx.node_name(), ctx.timestep, k=ctx.scalar("k")))
        b.connect(b.read("x"), into=node[cls.READS])
        b.connect(node[cls.OUTPUT], into=b.target("load"))


def planted_gradient(kind: str):
    """The toy probe in its bath, pulled by a planted spring of ``kind``; returns the experiment and the spring's
    handle."""
    exp = toy_experiment(f"gradient_{kind}")
    ADD_BATH.apply(exp)
    ADD_PROBE.apply(exp)
    return exp, exp.attach(PlantedGradientSpring(kind=kind), body="probe")


__all__ = ["BOUND", "PlantedGradientSpring", "planted_gradient"]
