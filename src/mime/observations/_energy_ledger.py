from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from mime.gradients import GradientStatus
from mime.registry import BuildContext, Observation, Plan, PlanContext, register_observation

if TYPE_CHECKING:
    from mime.build import ObservationBuilder


@register_observation
@dataclass(frozen=True, kw_only=True)
class EnergyLedger(Observation):
    """The energy ledger, in one line: ``exp.observe(EnergyLedger())``. Attached, it has the build meter each body:
    each law then delivers into its own input on the body, and the body records, inside its own step, each law's work
    on it, the energy it stores and its scheme's own numerical terms, and checks its step's energy identity against
    rounding. The build report names, for each law, what kind of energy exchange it is, and what is outside the
    components (``built.report.energy``); a run's result gives the account in plain words, stored, lost, supplied
    (named by what supplies it) and each potential's change (``result.energy``).

    It adds no node and reads nothing of its own: its account is the bodies' records. Without it no program changes,
    bit for bit; with it the bodies' programs are new, judged by their energy identity."""

    name = "energy_ledger"
    version = "0.1.0"
    meters_energy: ClassVar[bool] = True
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": ()}
    roles = {}

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(nodes=())

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        return None
