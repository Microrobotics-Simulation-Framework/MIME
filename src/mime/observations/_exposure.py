from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as setting
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.core import switch
from mime.gradients import GradientStatus
from mime.nodes.body import ExposureNode
from mime.registry import (
    DESIGN,
    BuildContext,
    Flag,
    GradientRecord,
    Observation,
    Output,
    OutputKind,
    Plan,
    PlanContext,
    Problem,
    Read,
    RoleKind,
    Value,
    convention,
    register_observation,
)
from mime.units import FIELD_GRADIENT, FIELD_RATE, MAGNETIC_FLUX_DENSITY, FieldGradient, FieldRate, \
    MagneticFluxDensity, Quantity
from mime.validity._bases import is_pending
from mime.vocabulary import FLUX_DENSITY, GRADIENT, POSE, Capability

if TYPE_CHECKING:
    from mime.build import ObservationBuilder

#: Each quantity's limit setting, its unit's symbol and what it is.
_LIMITS = {"flux_density": ("T", "|B|, the flux density's magnitude"),
           "field_gradient": ("T/m", "the Frobenius norm of the gradient dB_i/dx_j"),
           "rate_of_change": ("T/s", "|dB/dt|, the flux density's rate of change along the body's path, a one-step "
                                     "difference")}
_NO_LIMIT = convention(DESIGN, "no limit is declared unless one is given: the quantity is reported, with its maximum, "
                               "and flagged against nothing")


@register_observation
@dataclass(frozen=True, kw_only=True)
class Exposure(Observation):
    """The field a body is exposed to, each step, at the point where a law bound to the body reads it (its magnet's
    centre): ``Exposure(umr, field="coils.magnetic", flux_density=mT(40, reason=...),
    rate_of_change=T_per_s(1, reason=...))``.

    It reports three quantities each step: ``flux_density``, ``|B|`` (T); ``field_gradient``, the Frobenius norm of
    the gradient ``dB_i/dx_j`` (T/m), ``sqrt(sum_ij G_ij^2)``; and ``rate_of_change``, ``|B_n - B_{n-1}| / dt`` (T/s), a
    one-step difference of the flux density at the body's point, so it holds the body's motion through the field as
    well as the field's own change (the derivative along the body's path), and is an estimate with an error of order
    ``dt``. Step 0's rate of change is not measured (no earlier reading exists): it is held at 0, so it neither raises
    the flag nor counts in the maximum, and a field there from the start is never seen switching on. A field switched
    on between step 0 and step 1 is seen at step 1. The maximum of each over the run is kept at the end
    (``max_<quantity>``).

    Each quantity has a limit the experiment may declare, a quantity carrying its reason (``mT(40, reason="the
    protocol's ceiling")``), and a flag per quantity (``above_<quantity>``), raised at a step where the quantity is
    above its limit. A limit is a number the experiment declares: the flag says a modelled quantity passed it, nothing more.
    A limit with no reason, or a pending one, or one not above zero, is refused (``exposure.declaration``).

    A field that varies in space is read where a law bound to the body reads it at the same point, adding no slot to
    the provider (its node is what it is without the observation); a uniform field where a law reads its flux density,
    its gradient zero. Where no law reads the field at the body, it is refused (``observation.field-read``): another
    point needs the observation's own instance of the provider, which is later work."""

    name = "exposure"
    version = "0.1.0"
    setting_roles = {"flux_density": _NO_LIMIT, "field_gradient": _NO_LIMIT, "rate_of_change": _NO_LIMIT}
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="exposure.norm",
            indicator="the magnitude of the vector normed (the flux density, the gradient, the change over the step), "
                      "in its units: the norm's kink at zero",
            orders={"flux_density": 0, "field_gradient": 0, "rate_of_change": 0, "max_flux_density": 0,
                    "max_field_gradient": 0,
                    "max_rate_of_change": 0},
            route="none: a magnitude has no derivative at zero, where the norm's is taken as zero (the maxima "
                  "follow it)"),
        switch.Declaration(
            owner="exposure.maximum",
            indicator="the step's value less the maximum before it, in its units: the maximum moves where it is "
                      "positive",
            orders={"max_flux_density": 0, "max_field_gradient": 0, "max_rate_of_change": 0},
            route="none: a maximum over steps takes the derivative of the step that sets it"),
    ))
    references = ()
    builds = {"": (ExposureNode,)}
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    reads = {"flux_density": Read("field", FLUX_DENSITY.name)}
    outputs = (
        Output("flux_density", MAGNETIC_FLUX_DENSITY, "|B|, the flux density's magnitude at the body's point, each "
                                                      "step"),
        Output("field_gradient", FIELD_GRADIENT, "the Frobenius norm of the gradient dB_i/dx_j at the body's point, "
                                                 "each step (zero in a uniform field)"),
        Output("rate_of_change", FIELD_RATE, "|dB/dt| at the body's point, the flux density's change over the step "
                                             "divided by dt: a one-step difference along the body's path (its motion "
                                             "included), an estimate with an error of order dt; the first step's is "
                                             "not measured and is held at zero"),
        Output("max_flux_density", MAGNETIC_FLUX_DENSITY, "the largest |B| over the run", kind=OutputKind.AT_END),
        Output("max_field_gradient", FIELD_GRADIENT, "the largest gradient norm over the run", kind=OutputKind.AT_END),
        Output("max_rate_of_change", FIELD_RATE, "the largest |dB/dt| over the run, the first step's not measured",
               kind=OutputKind.AT_END),
    )
    raises = tuple(Flag(f"above_{name}", f"{what} was above its declared limit", written=(name,))
                   for name, (_, what) in _LIMITS.items())
    body: Any = setting(default=None, kw_only=False)
    field: Any = None
    flux_density: Annotated[Quantity | None, MagneticFluxDensity] = None
    field_gradient: Annotated[Quantity | None, FieldGradient] = None
    rate_of_change: Annotated[Quantity | None, FieldRate] = None

    def declaration_problems(self) -> list[Problem]:
        """Each declared limit that is not one value above zero, or carries no reason or a pending one."""
        found: list[Problem] = []
        for name, (unit, what) in _LIMITS.items():
            limit = getattr(self, name)
            if limit is None:
                continue
            value = limit.value if isinstance(limit, Quantity) else None
            if not (isinstance(value, float) and math.isfinite(value) and value > 0.0):
                found.append(Problem("exposure.declaration",
                                     f"exposure limit {name}={limit!r} is not one value above zero",
                                     f"declare {name}= as one quantity in {unit} above zero: the limit on {what}"))
            elif limit.reason is None or is_pending(limit.reason):
                said = "carries no reason" if limit.reason is None else "carries a pending reason"
                found.append(Problem("exposure.declaration",
                                     f"exposure limit {name}={limit} {said}: a limit is a number the experiment "
                                     f"declares, and says where it comes from",
                                     f"write it with its reason, {name}=<quantity>(..., reason=\"where the number "
                                     f"comes from\")"))
        return found

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        """A uniform field's flux density is read with no points, its gradient zero; any other's flux density and
        gradient at the body's magnet's centre, where the magnetic laws 0.2.0 read it."""
        problems = tuple(self.declaration_problems())
        provider = ctx.capabilities.get("field")
        if provider is not None and Capability.UNIFORM in provider.flags:
            return Plan(reads=dict(self.reads), problems=problems)
        at = Read("body", POSE.name)
        reads = {"flux_density": Read("field", FLUX_DENSITY.name, at=at, offset="magnet_centre"),
                 "gradient": Read("field", GRADIENT.name, at=at, offset="magnet_centre")}
        return Plan(reads=reads, values={"magnet_centre": Value("body", "magnet_centre")}, problems=problems)

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        uniform = "magnet_centre" not in ctx.values  # a uniform field's plan reads no point
        limits = {f"{name}_limit": ctx.scalar(name) for name in _LIMITS if ctx.settings.get(name) is not None}
        node = b.add_node(ExposureNode(ctx.node_name(), ctx.timestep, uniform=uniform, **limits))
        for label in ("flux_density",) if uniform else ("flux_density", "gradient"):
            b.connect(b.read(label), into=node[label])


__all__ = ["Exposure"]
