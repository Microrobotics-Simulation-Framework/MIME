"""Phase 2: construct every node, add nodes and edges in the canonical order, add the coupling groups, run the
post-build checks, compile. The only code that builds a MADDENING graph."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from maddening import GraphManager

from mime.build import _testing
from mime.build._builders import (
    DynamicsBuilder,
    EdgeDecl,
    EntryBuildError,
    InteractionBuilder,
    ObservationBuilder,
    ProviderBuilder,
    _Builder,
)
from mime.build._checks import post_build_checks
from mime.build._plan import EntryPlan, PlanSet, Section
from mime.build._refusals import MIME_DEFECT, BuildRefused, Refusal, RefusalClass
from mime.build._report import BuildReport, GroupLine, NodeLine, graph_config
from mime.registry import BuildContext

_BUILDERS: dict[Section, type[_Builder]] = {
    Section.PROVIDER: ProviderBuilder,
    Section.INTERACTION: InteractionBuilder,
    Section.DYNAMICS: DynamicsBuilder,
    Section.OBSERVATION: ObservationBuilder,
}


def report_of(plans: PlanSet, order: list[EntryPlan], refusals: Sequence[Refusal] = (),
              external_inputs: tuple[str, ...] = ()) -> BuildReport:
    lines, position = [], 0
    for plan in order:
        for name in plan.node_names():
            position += 1
            lines.append(NodeLine(position, name, plan.owner, plan.section.name.lower(), plan.describe(),
                                  plan.divider, plans.dt * plan.divider))
    return BuildReport(
        plans.settings.experiment, plans.dt, tuple(lines),
        tuple(GroupLine(g.nodes, g.tolerance, g.max_iterations, g.requested_by) for g in plans.groups),
        external_inputs,
        {p.owner: dict(p.seeds) for p in plans.entries if p.seeds},
        {p.owner: tuple(p.registered.cls.validity) for p in plans.entries if p.registered.cls.validity},
        dict(plans.vocabulary), tuple(refusals))


def _construct_entry(plan: EntryPlan, plans: PlanSet) -> tuple[_Builder, list[str]]:
    timestep = plans.dt * plan.divider
    builder = _BUILDERS[plan.section](plan, timestep)
    context = BuildContext(plan.owner, plans.dt, timestep, plan.owner, dict(plan.bindings), dict(plan.seeds),
                           dict(plan.initial))
    try:
        if plan.section is Section.DYNAMICS:
            plan.entry.build(context, builder, list(plan.registrations))  # type: ignore[call-arg]
        elif plan.section is Section.PROVIDER:
            plan.entry.build(context, builder, list(plan.slots))  # type: ignore[call-arg]
        else:
            plan.entry.build(context, builder)  # type: ignore[call-arg]
    except EntryBuildError as error:
        return builder, [str(error)]
    except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
        return builder, [f"build() raised {error!r}"]
    return builder, builder.finish()


def construct(plans: PlanSet) -> tuple[Any, BuildReport]:
    canonical = not _testing.skipping_canonical_order()
    order = list(plans.entries) if canonical else sorted(plans.declared, key=lambda p: p.declared)
    builders: dict[str, _Builder] = {}
    refusals: list[Refusal] = []
    for plan in plans.entries:
        builder, problems = _construct_entry(plan, plans)
        builders[plan.owner] = builder
        cls = plan.registered.cls
        refusals += [Refusal("entry.defect", RefusalClass.INTERNAL, f"{cls.label()} ({cls.__module__}), "
                                                                    f"building {plan.owner}: {problem}",
                             f"a defect in {cls.label()}, defined in {cls.__module__}: report it to that package. "
                             f"If that package is MIME: {MIME_DEFECT}") for problem in problems]
    if refusals:
        raise BuildRefused(refusals, report_of(plans, order, refusals))

    nodes = [node for plan in order
             for _, node in sorted(builders[plan.owner].nodes, key=lambda held: plan.nodes.index(held[0]))]
    position = {node.name: i for i, node in enumerate(nodes)}
    edges: list[EdgeDecl] = [edge for plan in order for edge in builders[plan.owner].edges]
    if canonical:
        # Edges into one input are summed in the order they are added, so within each target they go in the order
        # of the delivering entry's id.
        edges.sort(key=lambda e: (position[e.target[0]], e.target[1], e.by, e.seq))

    graph = GraphManager()
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge.source[0], edge.target[0], edge.source[1], edge.target[1], additive=edge.additive)
    for group in sorted(plans.groups, key=lambda g: min(position[n] for n in g.nodes)):
        # Every setting that decides what the group returns is written here, defaults included: "ift" is exact only
        # at a converged fixed point, and atol above zero can accept a group that has not converged.
        graph.add_coupling_group(sorted(group.nodes, key=position.__getitem__), max_iterations=group.max_iterations,
                                 tolerance=group.tolerance, solver="ift", atol=0.0)

    _testing.run_hook(graph, plans)
    external = tuple(f"{e['target_node']}.{e['target_field']}" for e in graph_config(graph)["external_inputs"])
    refusals = post_build_checks(graph, plans)
    if refusals:
        raise BuildRefused(refusals, report_of(plans, order, refusals, external))
    graph.compile()
    if not math.isclose(graph.timestep, plans.dt, rel_tol=1e-12):
        refusals = [Refusal("post-build.base-step", RefusalClass.INTERNAL,
                            f"MADDENING's base step is {graph.timestep!r} s, not dt = {plans.dt!r} s", MIME_DEFECT)]
        raise BuildRefused(refusals, report_of(plans, order, refusals, external))
    return graph, report_of(plans, order, [], external)
