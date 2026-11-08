"""The build's structural flags: basic mistakes in how laws couple their receivers, found from the graph
in every profile, with or without the energy ledger, since energy cannot see them.

- **A rate divider across a law's two sides is refused** (``law.rate-divided``): a law whose deliveries reach two
  dynamics that step at different rates delivers a value one side reads at every one of its steps and the other only
  on its own, so the slower side takes one sample of a load that changes between, and the pair's momentum drifts
  (measured: 0.13 and 0.58 of its relative momentum's peak at dividers of 2 and 4, with every energy check passing).
- **A damper between two receivers that is neither held in one step nor solved with them in one coupling group is
  flagged**, a line computed on the law: it reads both at a step's start and is paired with each one's own step, so
  its sign is exact on neither and holds to the coupling's order only. In a group, its convergence at run time is
  MADDENING's verdict, which the run reports (``RunResult.groups``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mime.build._builders import DELIVERY, EdgeDecl
from mime.build._plan import EntryPlan, PlanSet, Section
from mime.build._refusals import Refusal, RefusalClass
from mime.registry import Interaction
from mime.vocabulary import EnergyClass


def _receivers(plans: PlanSet, edges: Sequence[EdgeDecl]) -> dict[str, list[str]]:
    """Each law's receivers: the dynamics' nodes its deliveries reach (its reactions on a source's segment
    included), in the order of the nodes."""
    owners: dict[str, EntryPlan] = {name: plan for plan in plans.entries for name in plan.node_names()}
    laws = {plan.owner for plan in plans.entries if plan.section is Section.INTERACTION}
    found: dict[str, list[str]] = {}
    for edge in edges:
        plan = owners.get(edge.target[0])
        if edge.carries != DELIVERY or edge.by not in laws or plan is None or plan.section is not Section.DYNAMICS:
            continue
        held = found.setdefault(edge.by, [])
        if edge.target[0] not in held:
            held.append(edge.target[0])
    return found


def structural_flags(plans: PlanSet, edges: Sequence[EdgeDecl]) -> tuple[list[Refusal], dict[str, tuple[str, ...]]]:
    """The refusals a law's coupling of its receivers draws, and the lines computed on each law it flags."""
    owners: dict[str, EntryPlan] = {name: plan for plan in plans.entries for name in plan.node_names()}
    entries = {plan.owner: plan.entry for plan in plans.entries if plan.section is Section.INTERACTION}
    grouped = [set(group.nodes) for group in plans.groups]
    refusals: list[Refusal] = []
    lines: dict[str, tuple[str, ...]] = {}
    for law, receivers in sorted(_receivers(plans, edges).items()):
        if len(receivers) < 2:
            continue
        rates = {name: owners[name].divider_of(name) for name in receivers}
        if len(set(rates.values())) > 1:
            refusals.append(Refusal(
                "law.rate-divided", RefusalClass.USER,
                f"{law} delivers to receivers that step at different rates: "
                + ", ".join(f"{name} every {rate} steps" for name, rate in rates.items())
                + "; the slower one takes one sample of a load that changes between its steps, and the pair's momentum "
                  "drifts, which no energy check sees",
                "step every receiver a law couples at one rate (rate= on their dynamics), or solve them together in "
                "one coupling group"))
            continue
        entry = entries.get(law)
        if isinstance(entry, Interaction) and entry.energy_class is EnergyClass.DISSIPATIVE and \
                not any(set(receivers) <= group for group in grouped):
            lines[law] = (f"a damper between {' and '.join(receivers)}, neither held in one step nor solved with them "
                          f"in one coupling group: it reads both at the step's start and each pairs it with its own "
                          f"step, so its sign is exact on neither and holds to the coupling's order only",)
    return refusals, lines


def merged(lines: Mapping[str, tuple[str, ...]], more: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Two sets of computed lines by entry, each entry's in order."""
    return {owner: (*lines.get(owner, ()), *more.get(owner, ())) for owner in sorted({*lines, *more})}


__all__ = ["merged", "structural_flags"]
