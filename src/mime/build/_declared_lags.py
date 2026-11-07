"""The lag an observation is net of, read from the build's own lag accounting rather than assumed.

An observation reads its body's pose and its field at the end of the step. The laws bound to the same body and field
read them where the build's lag accounting puts them (:mod:`mime.build._lags`): a law outside a coupling group reads
the body's published state as it stood at the start of the step, and one inside a group with the body reads the
current iterate. An observation with :attr:`~mime.registry.Observation.net_of_lag` is handed the difference, in steps,
between the pose its laws read and the one it reads, from those same edges, so a build that declares another lag (a
law in a coupling group) gives another correction, and a lag it cannot net out is refused."""

from __future__ import annotations

from collections.abc import Sequence

from maddening.core.node import SimulationNode

from mime.build._lags import EdgeLag, Lag
from mime.build._plan import EntryPlan, PlanSet, ResolvedRead, Section
from mime.build._refusals import Refusal, RefusalClass
from mime.registry import Observation

#: The steps behind the end of a step that an edge's value is read from.
STEPS_BEHIND = {Lag.THIS_STEP: 0, Lag.CURRENT_ITERATE: 0, Lag.PREVIOUS_STEP: 1}


def _lag_of(lags: Sequence[EdgeLag], plan: EntryPlan, read: ResolvedRead) -> EdgeLag | None:
    """The edge by which ``plan`` reads ``read``."""
    return next((e for e in lags if e.by == plan.owner and e.source == (read.node, read.field)), None)


def declared_lag(plans: PlanSet, observation: EntryPlan,
                 lags: Sequence[EdgeLag]) -> tuple[int, str] | tuple[None, str]:
    """The steps the pose its laws read lags behind the observation's, and the words for the build report; or ``None``
    and why it cannot be netted. The laws are the interactions bound to the observation's body and field that read the
    field's flux density: those that turn the body through that field."""
    cls = observation.registered.cls
    assert issubclass(cls, Observation) and cls.net_of_lag is not None
    body_role, field_role = cls.net_of_lag
    body, field = observation.bindings.get(body_role), observation.bindings.get(field_role)
    pose, flux = observation.reads.get("pose"), observation.reads.get("flux_density")
    if pose is None or flux is None:
        return None, "it reads no pose or no flux density to compare the laws' reads with"
    own_pose, own_flux = _lag_of(lags, observation, pose), _lag_of(lags, observation, flux)
    if own_pose is None or own_flux is None:
        return None, "the build declares no edge for its own read of the pose or the field"
    laws = [p for p in plans.entries if p.section is Section.INTERACTION and p.bindings.get(body_role) == body
            and p.bindings.get(field_role) == field
            and any(r.interface.endswith("flux_density") for r in p.reads.values())]
    if not laws:
        return 0, (f"no law bound to {body} and {field} reads the field's flux density, so none turns the body "
                   f"through it and there is no lag to remove")
    found: dict[int, list[str]] = {}
    words: list[str] = []
    for law in laws:
        law_pose = next((r for r in law.reads.values() if r.interface.endswith("pose")), None)
        law_flux = next((r for r in law.reads.values() if r.interface.endswith("flux_density")), None)
        pose_edge = _lag_of(lags, law, law_pose) if law_pose is not None else None
        flux_edge = _lag_of(lags, law, law_flux) if law_flux is not None else None
        if pose_edge is None or flux_edge is None:
            return None, f"{law.owner} reads the body's pose or the field by no edge the build declares"
        field_behind = STEPS_BEHIND[flux_edge.lag] - STEPS_BEHIND[own_flux.lag]
        if field_behind != 0:
            return None, (f"{law.owner} reads the field {field_behind} step(s) behind the observation's read "
                          f"({flux_edge.text}, {flux_edge.lag.value}), and the field's turn over those steps is not "
                          f"the body's to remove")
        behind = STEPS_BEHIND[pose_edge.lag] - STEPS_BEHIND[own_pose.lag]
        found.setdefault(behind, []).append(law.owner)
        words.append(f"{law.owner} reads the body's pose by {pose_edge.text} ({pose_edge.lag.value}) and the field by "
                     f"{flux_edge.text} ({flux_edge.lag.value}), the observation by {own_pose.text} "
                     f"({own_pose.lag.value})")
    if len(found) > 1:
        return None, ("the laws read the body's pose different numbers of steps behind the observation's ("
                      + "; ".join(f"{', '.join(owners)}: {steps}" for steps, owners in sorted(found.items())) + ")")
    (steps,) = found
    return steps, f"net of {steps} step(s) of the body's turn: " + "; ".join(words)


def net_of_declared_lags(plans: PlanSet, nodes: Sequence[SimulationNode],
                         lags: Sequence[EdgeLag]) -> tuple[list[SimulationNode], list[Refusal]]:
    """``nodes`` with each net-of-lag observation's node made net of the lag the build declares (its node reports why
    in the build report), and the refusals for those that cannot be."""
    replaced = {node.name: node for node in nodes}
    refusals: list[Refusal] = []
    for plan in plans.entries:
        cls = plan.registered.cls
        if plan.section is not Section.OBSERVATION or not (isinstance(plan.entry, Observation)
                                                          and getattr(cls, "net_of_lag", None) is not None):
            continue
        steps, why = declared_lag(plans, plan, lags)
        if steps is None:
            refusals.append(Refusal(
                "observation.lag-ambiguous", RefusalClass.USER,
                f"{plan.owner} is net of the lag the laws that turn its body read the pose by, and the build declares "
                f"no single lag it can remove: {why}",
                "observe through a field one law acts through on the body, exp.attach(Magnetic.Torque(), body=..., "
                "field=...) then exp.observe(PhaseLag(..., field=...)), so that the build declares one lag"))
            continue
        for name in plan.node_names():
            replaced[name] = plan.entry.net_of(replaced[name], steps, why)
    return [replaced[node.name] for node in nodes], refusals
