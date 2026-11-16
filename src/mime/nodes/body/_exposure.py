"""The field a body is exposed to, each step: the flux density's magnitude, its gradient's Frobenius norm and the rate
of change of the flux density along the body's path, against declared limits, with their maxima, kept in the observing
node's own state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, ParamSpec, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, \
    node_input, switch, typed_view

EQUATIONS = (r"b_{n} = |\mathbf{B}_{n}|, \quad g_{n} = \|G_{n}\|_{F}, \quad "
             r"r_{n} = \frac{|\mathbf{B}_{n} - \mathbf{B}_{n-1}|}{h} \ (r_{0} \text{ not measured, held } 0), \quad "
             r"\hat{x} = \max_{j \le n} x_{j}, \quad a^{x}_{n} = [x_{n} > x_{\max}]")

#: The three quantities, each with its per-step field, its maximum and its flag.
QUANTITIES = ("flux_density", "field_gradient", "rate_of_change")
#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.body._exposure.ExposureNode.advance"


class ExposureState(TypedDict, closed=True):
    flux_density: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    field_gradient: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    rate_of_change: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    max_flux_density: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    max_field_gradient: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    max_rate_of_change: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    above_flux_density: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    above_field_gradient: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    above_rate_of_change: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    last: ReadOnly[Annotated[Float[Array, "3"], Units("T")]]
    measured: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class ExposureInputs(TypedDict, closed=True):
    flux_density: NotRequired[ReadOnly[Float[Array, "3"]]]
    gradient: NotRequired[ReadOnly[Float[Array, "3 3"]]]


class ExposureNode(MimeNode):
    """What a body is exposed to, each step, from the flux density ``B`` and its gradient ``G`` it takes at the body's
    point: ``flux_density``, ``|B|`` (T); ``field_gradient``, the Frobenius norm of ``G`` (T/m),
    ``sqrt(sum_ij G_ij^2)``;
    ``rate_of_change``, the flux density's, ``|B_n - B_{n-1}| / h`` (T/s), a one-step difference along the
    body's path, so it holds the body's motion through the field as well as the field's own change, and is an estimate
    with an error of order ``h`` (the field's second derivative along the path times the step). The first step has no
    earlier reading: its rate is not measured, and is held at zero (``measured`` is zero there, one after). A node built
    with ``uniform`` takes no gradient (a uniform field has none), and its field gradient reads zero.

    Each has its maximum over the steps so far (``max_<quantity>``, zero before the first) and a flag, 1 at a step where
    it is above its declared limit (``<quantity>_limit``, infinite where none is declared, never raised) and 0
    otherwise (``above_<quantity>``): a report against a number the experiment declares, which no edge reads."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-136",
        description="The flux density's magnitude, its gradient's Frobenius norm and its rate of change along a "
                    "body's path at the body's point each step, their maxima and a flag each against a declared "
                    "limit, kept in the node's own state.",
        governing_equations=EQUATIONS,
        assumptions=("The field and its gradient are those a law reads at the body's point, at the end of each step",
                     "The rate is a backward difference over one step: its error is of order the step times the "
                     "field's second derivative along the body's path"),
        limitations=("The rate holds the body's motion through the field (the derivative along its path), not the "
                     "field's change at a point fixed in space",
                     "The first step's rate is not measured: no earlier reading exists, and it is held at zero",
                     "A limit is the experiment's declaration: the flag says a modelled quantity passed a declared "
                     "number, and bounds nothing else"),
    )
    symbols = (
        Symbol("b", SymbolKind.STATE, "flux_density", "T", _ADVANCE),
        Symbol("g", SymbolKind.STATE, "field_gradient", "m^-1 kg s^-2 A^-1", _ADVANCE),
        Symbol("r", SymbolKind.STATE, "rate_of_change", "kg s^-3 A^-1", _ADVANCE),
        Symbol(r"\hat{x}", SymbolKind.STATE, "max_flux_density", "T", _ADVANCE),
        Symbol(r"a^{x}", SymbolKind.STATE, "above_flux_density", "1", _ADVANCE),
        Symbol(r"\mathbf{B}", SymbolKind.INPUT, "flux_density", "T", _ADVANCE),
        Symbol("G", SymbolKind.INPUT, "gradient", "T/m", _ADVANCE),
        Symbol(r"x_{\max}", SymbolKind.PARAMETER, "flux_density_limit", "T", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, flux_density_limit: float = float("inf"),
                 field_gradient_limit: float = float("inf"), rate_of_change_limit: float = float("inf"),
                 uniform: bool = False) -> None:
        super().__init__(name, timestep, flux_density_limit=float(flux_density_limit),
                         field_gradient_limit=float(field_gradient_limit),
                         rate_of_change_limit=float(rate_of_change_limit), uniform=bool(uniform))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(),
                "flux_density_limit": ParamSpec(trainable=False, units="T", description="the declared limit on |B|, "
                                                                                        "infinite where none is"),
                "field_gradient_limit": ParamSpec(trainable=False, units="T/m", description="the declared limit on the "
                                                                                      "gradient's Frobenius norm"),
                "rate_of_change_limit": ParamSpec(trainable=False, units="T/s",
                                                  description="the declared limit on |dB/dt|")}

    @override
    def initial_state(self) -> NodeFields:
        zero = as_run_array(0.0)
        return node_fields(ExposureState(
            flux_density=zero, field_gradient=zero, rate_of_change=zero, max_flux_density=zero, max_field_gradient=zero,
            max_rate_of_change=zero, above_flux_density=zero, above_field_gradient=zero, above_rate_of_change=zero,
            last=as_run_array(jnp.zeros(3)), measured=zero))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        held = {"flux_density": node_input("T", (3,))}
        return held if self.constants()["uniform"] else {**held, "gradient": node_input("T/m", (3, 3))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(ExposureState, state), typed_view(ExposureInputs, boundary_inputs)
        dtype = held["flux_density"].dtype
        field = jnp.asarray(given.get("flux_density", jnp.zeros(3, dtype=dtype)), dtype=dtype)
        gradient = jnp.asarray(given.get("gradient", jnp.zeros((3, 3), dtype=dtype)), dtype=dtype)
        # The first step compares the reading with itself: no earlier one exists, so its rate is not measured.
        last = held["measured"] * held["last"] + (1.0 - held["measured"]) * field
        change = field - last
        values = {
            "flux_density": switch.norm(field, owner="exposure.norm", resolution=0.0,
                                        indicator=lambda: jnp.sqrt(jnp.sum(field * field))),
            "field_gradient": switch.norm(gradient, owner="exposure.norm", resolution=0.0,
                                          indicator=lambda: jnp.sqrt(jnp.sum(gradient * gradient))),
            "rate_of_change": switch.norm(change, owner="exposure.norm", resolution=0.0,
                                indicator=lambda: jnp.sqrt(jnp.sum(change * change))) / dt}
        out: dict[str, Array] = {"last": field, "measured": jnp.ones((), dtype=dtype)}
        for name in QUANTITIES:
            value, peak = values[name], state[f"max_{name}"]
            out[name] = value
            out[f"max_{name}"] = switch.maximum(value, peak, owner="exposure.maximum",
                                                resolution=lambda peak=peak: switch.rounding(peak),
                                                indicator=lambda value=value, peak=peak: value - peak)
            out[f"above_{name}"] = (value > jnp.asarray(p[f"{name}_limit"], dtype=dtype)).astype(dtype)
        return node_fields(out)


__all__ = ["EQUATIONS", "QUANTITIES", "ExposureInputs", "ExposureNode", "ExposureState"]
