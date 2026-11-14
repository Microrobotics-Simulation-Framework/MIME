"""Responses: the last rung of a command port's delivery stage (``CommandPort(response=...)``), what the receiver gets
from what its amplifier drives. :class:`Ideal` delivers it as it is; :class:`FirstOrderLag` delivers it through a
first-order lag of time constant ``tau``, solved exactly over each step for a value held through it: a coil's L/R
lag (a signal model, with no back-EMF and no mutual inductance) or a servo's. The same declaration is computed where
its receiver says: in the port's stage for a coil array's currents, in the arm's own node for a kinematic arm's
joints (its servo), so that the lag has one implementation per receiver and the arm's programs keep their bits.

A lab registers its own amplifier model with ``mime.registry.register_response``: a ``Response`` whose ``respond``
is a pure function in ``jax.numpy``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

from mime.core import jnp
from mime.gradients import GradientStatus
from mime.registry import PHYSICAL, Response, register_response
from mime.units import Magnitude, Quantity, Time


@register_response
@dataclass(frozen=True, kw_only=True)
class Ideal(Response):
    """What the amplifier drives is what the receiver gets, at once: the response of a port that declares none."""

    name = "response.ideal"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": ()}

    @override
    @classmethod
    def respond(cls, settings: Mapping[str, Any], state: Mapping[str, Any], given: Any,
                dt: float) -> tuple[dict[str, Any], Any]:
        return {}, given


@register_response
@dataclass(frozen=True, kw_only=True)
class FirstOrderLag(Response):
    """A first-order lag of time constant ``tau``: over a step of ``dt`` the delivered value moves from where it was
    towards the value driven, ``y + (y_prev - y) exp(-dt / tau)``, exact for a value held through the step. A coil's
    L/R (``tau = L / R``) and a kinematic arm's servo are each one."""

    name = "response.first_order_lag"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": ()}
    setting_roles = {"tau": PHYSICAL}
    carries = ("lagged",)
    tau: Annotated[Quantity, Time]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        tau = settings["tau"]
        if not (isinstance(tau, float) and math.isfinite(tau) and tau > 0.0):
            return [f"tau is {tau!r} s, not a time constant above zero"]
        return []

    @override
    @classmethod
    def respond(cls, settings: Mapping[str, Any], state: Mapping[str, Any], given: Any,
                dt: float) -> tuple[dict[str, Any], Any]:
        given = jnp.asarray(given)
        lagged = given + (state["lagged"] - given) * jnp.exp(-dt / jnp.asarray(settings["tau"], given.dtype))
        return {"lagged": lagged}, lagged


__all__ = ["FirstOrderLag", "Ideal"]
