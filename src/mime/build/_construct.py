"""Phase 2: construct every node, add nodes and edges in the canonical order, add the coupling groups, run the
post-build checks, compile, probe the compiled step, and derive the hash. The only code that builds a MADDENING
graph."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any, cast

from maddening import GraphManager
from maddening.core.node import SimulationNode
from maddening.warnings import UnitMismatchWarning

from mime.build import _testing
from mime.build._builders import (
    DriveBuilder,
    DynamicsBuilder,
    EdgeDecl,
    EntryBuildError,
    InteractionBuilder,
    ObservationBuilder,
    ProviderBuilder,
    RestrictedBuilder,
)
from mime.build._checks import post_build_checks
from mime.build._control import Mode
from mime.build._derived import DerivedGraph, derive
from mime.build._in_graph import controller_nodes, in_graph_order, order_refusals
from mime.build._lags import EdgeLag, classify, cross_check, unplanned_refusals
from mime.build._maddening import (
    add_coupling_group,
    add_edge,
    add_external_input,
    graph_edges,
    graph_schedule,
    initial_fields,
)
from mime.build._plan import EntryPlan, PlanSet, Section
from mime.build._probes import node_host_callbacks
from mime.build._refusals import MIME_DEFECT, BuildRefused, Explained, Refusal, RefusalClass, entry_defect
from mime.core.multibody import ElementGroup
from mime.build._report import BuildReport, GroupLine, NodeLine, ReportedWarning, graph_config
from mime.registry import (
    BuildContext,
    ControlLaw,
    Drive,
    Dynamics,
    FieldProvider,
    Interaction,
    Observation,
)

_BUILDERS: dict[Section, type[RestrictedBuilder]] = {
    Section.PROVIDER: ProviderBuilder,
    Section.INTERACTION: InteractionBuilder,
    Section.DYNAMICS: DynamicsBuilder,
    Section.OBSERVATION: ObservationBuilder,
    Section.DRIVE: DriveBuilder,
}

#: MADDENING's ``validate()`` marks each issue with its kind. These four stop ``compile()``, so each is a refusal.
_REFUSED_ISSUES = {"ERROR": "maddening.error", "WARNING[units]": "maddening.units",
                   "WARNING[shape]": "maddening.shape", "WARNING[dtype]": "maddening.dtype"}


def report_of(plans: PlanSet, order: Sequence[str], refusals: Sequence[Refusal] = (),
              external_inputs: tuple[str, ...] = (), reported: Sequence[ReportedWarning] = (),
              derived_hash: str | None = None, lags: Sequence[EdgeLag] = (),
              models: tuple[str, ...] | None = None) -> BuildReport:
    owners = {name: plan for plan in plans.entries for name in plan.node_names()}
    lines = [NodeLine(position, name, owners[name].owner, owners[name].section.name.lower(), owners[name].describe(),
                      owners[name].divider_of(name), plans.dt * owners[name].divider_of(name))
             for position, name in enumerate(order, 1)]
    return BuildReport(
        plans.settings.experiment, plans.dt, tuple(lines),
        tuple(GroupLine(g.nodes, g.tolerance, g.max_iterations, g.requested_by) for g in plans.groups),
        external_inputs,
        {p.owner: dict(p.seeds) for p in plans.entries if p.seeds},
        {p.owner: attached_validity(p) for p in plans.entries if attached_validity(p)},
        dict(plans.vocabulary), tuple(refusals), plans.run, tuple(reported), derived_hash, plans.values, tuple(lags),
        plans.scene_text(), plans.orientation, materials=plans.materials, descriptions=description_lines(plans),
        limits={p.owner: p.record.limits for p in plans.entries if p.record is not None and p.record.limits},
        controller=(*plans.controller.lines(plans.dt), *model_lines(plans, models))
        if plans.controller is not None else (),
        masses=plans.masses)


def model_lines(plans: PlanSet, models: tuple[str, ...] | None) -> tuple[str, ...]:
    """Which built trees the controller's law read while it was made (``ctx.models``), where that is known: a
    model-based controller reads the tree it controls with its mounted mass grafted on, and one that reads none holds a
    model of its own, a declared choice."""
    if models is None:
        return ()
    offered = ", ".join(sorted(plans.models)) or "none"
    if not models:
        return (f"  models: its law read none of the built trees (ctx.models offers {offered}): any model it holds "
                f"is its own, a declared choice",)
    return (f"  models: its law read the built tree of {', '.join(models)}, its mounted mass grafted on (ctx.models "
            f"offers {offered})",)


def attached_validity(plan: EntryPlan) -> tuple[str, ...]:
    """An entry's declared validity lines, less those of its rungs other than the one attached: a line that starts
    with another rung's name and a colon (``bem: ...`` beside an attached ``surrogate``) does not hold for this
    build."""
    cls = plan.registered.cls
    rung = getattr(plan.entry, "rung", None)
    others = {name for name in (*getattr(cls, "rungs", ()), *getattr(cls, "builds", {})) if name and name != rung}
    return tuple(line for line in cls.validity if line.split(": ", 1)[0] not in others)


def description_lines(plans: PlanSet) -> tuple[str, ...]:
    """Each chain segment's description as read: the file and its pin, how many elements each group holds, and one
    line per kind of element not used, read past or read as massless, naming every one."""
    found: list[str] = []
    for plan in plans.entries:
        held = plan.segment.description if plan.segment is not None else None
        if held is None:
            continue
        groups = ", ".join(f"{len(held.group(group))} {group.value}" for group in ElementGroup)
        found.append(f"{plan.owner}: {held.source.format} {held.source.file_name}, sha256 {held.source.sha256}: "
                     f"{groups}")
        found += [f"  {line}" for line in held.warnings()]
    return tuple(found)


def _entry_defect(plan: EntryPlan, problem: str) -> Refusal:
    return entry_defect(plan.registered.cls, problem, context=f", building {plan.owner}")


def declared_field_problems(plan: EntryPlan, built: dict[str, SimulationNode]) -> list[str]:
    """Every output an observation declares, and every flag an entry raises on the rung attached, that its node does
    not hold in its state as declared: a declaration a run's result could not read."""
    cls = plan.registered.cls
    rung = getattr(plan.entry, "rung", None)
    held: list[tuple[str, str, str, tuple[int, ...] | None]] = [
        ("flag", flag.node, flag.field, None) for flag in cls.raises if flag.rung is None or flag.rung == rung]
    if issubclass(cls, Observation):
        held += [("output", output.node, output.field, output.shape) for output in cls.outputs]
    found: list[str] = []
    for what, local, field, shape in held:
        node = built.get(plan.node_name(local))
        state = initial_fields(node) if node is not None else {}
        if field not in state:
            found.append(f"it declares the {what} {field!r} of its node {plan.node_name(local)!r}, which holds no "
                         f"such state field ({', '.join(sorted(state)) or 'none'})")
        elif shape is not None and tuple(state[field].shape) != shape:
            found.append(f"it declares the {what} {field!r} with shape {shape}, and its node holds it with shape "
                         f"{tuple(state[field].shape)}")
    return found


def _construct_entry(plan: EntryPlan, plans: PlanSet) -> tuple[RestrictedBuilder, list[str]]:
    timestep = plans.dt * plan.divider
    builder = _BUILDERS[plan.section](plan, timestep)
    context = BuildContext(plan.owner, plans.dt, timestep, plan.owner, dict(plan.bindings), dict(plan.seeds),
                           dict(plan.initial), dict(plan.settings),
                           {label: value.value for label, value in plan.values.items()}, tuple(plan.confinements),
                           plans.frame.up, plans.frame.gravity, dict(plan.regions), dict(plan.hulls), plan.mount,
                           dict(plan.parts), plan.mounted, plan.spin_inertia, spin_asymmetry=plan.spin_asymmetry,
                           rates_read=plan.rates_read)
    entry = plan.entry
    try:
        if isinstance(entry, Drive) and isinstance(builder, DriveBuilder):
            entry.build(context, builder)
        elif isinstance(entry, Dynamics) and isinstance(builder, DynamicsBuilder):
            entry.build(context, builder, list(plan.registrations))
        elif isinstance(entry, FieldProvider) and isinstance(builder, ProviderBuilder):
            entry.build(context, builder, list(plan.slots))
        elif isinstance(entry, Interaction) and isinstance(builder, InteractionBuilder):
            entry.build(context, builder)
        elif isinstance(entry, Observation) and isinstance(builder, ObservationBuilder):
            entry.build(context, builder)
        else:
            return builder, [f"a {type(entry).__name__} planned in the {plan.section.name.lower()} section"]
    except EntryBuildError as error:
        return builder, [str(error)]
    except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
        from mime.errors import raised_at

        return builder, [Explained(f"build() raised {error!r}", where=raised_at(error))]
    return builder, builder.finish()


def _validate(graph: GraphManager) -> tuple[list[ReportedWarning], list[Refusal], list[str]]:
    """MADDENING's own checks of the graph: each issue that would stop ``compile()`` is a refusal, a units warning
    among them (MIME labels every edge from the vocabulary, so one means a node and the vocabulary disagree); every
    other warning is reported. Its ``INFO`` lines describe the schedule: the report lists it from MIME's own lag
    accounting, which they are checked against, and they are returned for that."""
    reported: list[ReportedWarning] = []
    refusals: list[Refusal] = []
    issues = list(graph.validate())
    for issue in issues:
        code = next((code for prefix, code in _REFUSED_ISSUES.items() if issue.startswith(prefix)), None)
        if code is not None:
            refusals.append(Refusal(code, RefusalClass.INTERNAL, f"MADDENING's validate(): {issue}", MIME_DEFECT))
        elif issue.startswith("WARNING"):
            reported.append(ReportedWarning("MADDENING GraphManager.validate()", issue))
    return reported, refusals, [issue for issue in issues if issue.startswith("INFO")]


def _compile(graph: GraphManager, reported: list[ReportedWarning]) -> list[Refusal]:
    """``compile()``, with what it warns caught into the report rather than left on stderr: a
    ``UnitMismatchWarning`` is a refusal, and any other warning is reported with its class, once."""
    seen = {w.message for w in reported}
    refusals: list[Refusal] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            graph.compile()
        except Exception as error:  # noqa: BLE001 - MADDENING refused the graph; say so in the experiment's report
            refusals.append(Refusal("maddening.compile", RefusalClass.INTERNAL, f"MADDENING's compile() raised "
                                                                                f"{error!r}", MIME_DEFECT))
    for warning in caught:
        message = str(warning.message)
        if issubclass(warning.category, UnitMismatchWarning):
            refusals.append(Refusal("maddening.units", RefusalClass.INTERNAL,
                                    f"MADDENING's compile(): {message}", MIME_DEFECT))
        elif message not in seen:
            seen.add(message)
            reported.append(ReportedWarning(f"MADDENING GraphManager.compile(): {warning.category.__name__}", message))
    return refusals


def _host_callback_refusals(graph: GraphManager, plans: PlanSet) -> list[Refusal]:
    refusals: list[Refusal] = []
    for name in graph.node_names:
        calls = node_host_callbacks(graph, name)
        if not calls:
            continue
        plan = next((p for p in plans.entries if name in p.node_names()), None)
        built = f"{name} (built by {plan.owner}, {plan.registered.cls.label()})" if plan else name
        defined = f"{plan.registered.cls.__module__}" if plan else "an unknown module"
        refusals.append(Refusal(
            "post-build.host-callback", RefusalClass.INTERNAL,
            f"node {built} calls the host from its step code ({', '.join(sorted(calls))}): a run is one compiled "
            f"step, scanned, with no Python inside a step, and nothing on the host runs between compiled calls "
            f"unless the run's structure names it",
            f"remove the callback from the node's update or compute_boundary_fluxes (jax.debug.print included); it "
            f"is a defect of the entry defined in {defined}. If that is MIME: {MIME_DEFECT}"))
    return refusals


def derived_inputs(plans: PlanSet) -> list[tuple[str, str]]:
    """Every external input of the graph, derived from the writes of a controller called between compiled chunks: the
    node input of each command it writes, in place of that command's drive. Nothing else declares one; a controller
    inside the compiled step feeds its commands by edges."""
    controller = plans.controller
    if controller is None or controller.mode is not Mode.CALLBACK:
        return []
    return [(write.node, write.field) for write in controller.writes]


def construct(plans: PlanSet, law: ControlLaw | None = None,
              models: tuple[str, ...] | None = None) -> tuple[GraphManager, BuildReport, DerivedGraph]:
    """Phase 2. ``law`` is the controller's, made from its settings: a controller inside the compiled step is built
    from it; ``models`` the segments whose built trees its law read, which the build report names."""
    canonical = not _testing.skipping_canonical_order()
    building = [plan for plan in plans.entries if plan.section is not Section.CONTROLLER]
    entry_order = building if canonical else sorted((p for p in plans.declared if p.section is not Section.CONTROLLER),
                                                    key=lambda p: p.declared)
    order = list(plans.order) if canonical else [name for plan in sorted(plans.declared, key=lambda p: p.declared)
                                                 for name in plan.node_names()]
    builders: dict[str, RestrictedBuilder] = {}
    refusals: list[Refusal] = []
    for plan in building:
        builder, problems = _construct_entry(plan, plans)
        builders[plan.owner] = builder
        refusals += [_entry_defect(plan, problem) for problem in problems]
    if refusals:
        raise BuildRefused(refusals, report_of(plans, order, refusals))

    built = {node.name: node for plan in building for _, node in builders[plan.owner].nodes}
    refusals = [_entry_defect(plan, problem) for plan in building for problem in declared_field_problems(plan, built)]
    if refusals:
        raise BuildRefused(refusals, report_of(plans, order, refusals))
    in_graph: list[SimulationNode] = []
    controller_edges: list[EdgeDecl] = []
    if plans.controller is not None and plans.controller.mode is Mode.GRAPH:
        if law is None:
            raise ValueError("a controller inside the compiled step is built from its law: pass construct its law")
        in_graph, controller_edges, refusals = controller_nodes(plans.controller, law, plans.dt, built)
        if refusals:
            raise BuildRefused(refusals, report_of(plans, order, refusals))
        built.update({node.name: node for node in in_graph})
    nodes: list[SimulationNode] = [built[name] for name in order]
    if not nodes:
        refusals = [Refusal("run.nothing-to-step", RefusalClass.USER,
                            "the experiment builds no node: nothing it declares has dynamics, a field or a law that "
                            "steps, so a run would have no step to take",
                            "declare what moves or acts: a body with its dynamics, exp.add_body(Body(..., "
                            "dynamics=...)), a "
                            "medium's flow, an apparatus's field, or a law bound to them")]
        raise BuildRefused(refusals, report_of(plans, order, refusals))
    position = {name: i for i, name in enumerate(order)}
    edges: list[EdgeDecl] = [edge for plan in entry_order for edge in builders[plan.owner].edges]
    groups = [sorted(group.nodes, key=position.__getitem__) for group in plans.groups]
    physics_lags: tuple[EdgeLag, ...] = ()
    if in_graph:
        mine = [node.name for node in in_graph]
        physics = [name for name in order if name not in mine]
        physics_lags = classify(physics, edges, groups)
        order = in_graph_order(mine, physics, edges, groups)
        position = {name: i for i, name in enumerate(order)}
        groups = [sorted(group.nodes, key=position.__getitem__) for group in plans.groups]
        nodes = [built[name] for name in order]
        edges += controller_edges
    if canonical:
        # Edges into one input are summed in the order they are added, so within each target they go in the order
        # of the delivering entry's id.
        edges.sort(key=lambda e: (position[e.target[0]], e.target[1], e.by, e.seq))
    lags = classify(order, edges, groups)
    owners = {name: plan.owner for plan in plans.entries for name in plan.node_names()}

    graph = GraphManager()
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        add_edge(graph, edge.source, edge.target, additive=edge.additive, units=edge.units)
    for group in sorted(groups, key=lambda members: position[members[0]]):
        held = next(g for g in plans.groups if set(g.nodes) == set(group))
        add_coupling_group(graph, group, max_iterations=held.max_iterations, tolerance=held.tolerance)
    for node, field in derived_inputs(plans):
        add_external_input(graph, node, field)

    _testing.run_hook(graph, plans)
    external = tuple(f"{e['target_node']}.{e['target_field']}"
                     for e in cast(list[dict[str, Any]], graph_config(graph)["external_inputs"]))
    refusals = post_build_checks(graph, plans)
    if _testing.refusing_unplanned_loops():
        refusals += unplanned_refusals(lags, owners)
    refusals += order_refusals(plans.controller, lags, physics_lags)
    reported, refused_by_maddening, schedule_lines = _validate(graph)
    refusals += refused_by_maddening
    if refusals:
        raise BuildRefused(refusals, report_of(plans, order, refusals, external, reported, lags=lags))
    refusals = _compile(graph, reported)
    if not refusals and not math.isclose(graph.timestep, plans.dt, rel_tol=1e-12):
        refusals = [Refusal("post-build.base-step", RefusalClass.INTERNAL,
                            f"MADDENING's base step is {graph.timestep!r} s, not dt = {plans.dt!r} s", MIME_DEFECT)]
    if not refusals:
        refusals = [Refusal("lag.cross-check", RefusalClass.INTERNAL, f"lag accounting: {problem}", MIME_DEFECT)
                    for problem in cross_check(lags, graph_schedule(graph), graph_edges(graph), groups,
                                               schedule_lines)]
    if not refusals:
        refusals = _host_callback_refusals(graph, plans)
    if refusals:
        raise BuildRefused(refusals, report_of(plans, order, refusals, external, reported, lags=lags))
    derived = derive(graph, (n for p in plans.entries if p.section is Section.OBSERVATION for n in p.node_names()))
    return graph, report_of(plans, order, [], external, reported, derived.physics_hash, lags, models), derived
