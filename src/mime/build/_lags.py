"""Lag accounting: which step's value each edge reads, computed from MIME's own plan (the nodes in the order phase 2
adds them, the edges its builders declared, the coupling groups), and checked against what MADDENING compiled.

On MADDENING 0.4.x, an edge inside a coupling group reads the current iterate. Outside one, an edge on a feedback loop
whose source is scheduled at or after its target reads the previous step (a back edge), and every other edge reads
this step. The schedule keeps the nodes of a loop in the order they were added, with each coupling group's members
moved together to its first member's place; an edge between two nodes on no common loop always points forward.

A loop closed by a back edge is **planned** when every edge on it that reads the previous step carries a thing's
published state to the law, provider, dynamics or observation that reads it: each reader takes the state at the start
of the step, and each dynamics advances with this step's loads. That is what the canonical order builds. One load is
planned a step late: a chain segment's reaction on the segment it is mounted on, which is built after it (a staggered
coupling, stated in the report). Any other previous-step read (a delivered load, a field, a command, read a step
late) is refused unless a coupling group closes the loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from mime.build._builders import COMMAND, REACTION, STATE, EdgeDecl
from mime.build._maddening import topological_order
from mime.build._refusals import MIME_DEFECT, Refusal, RefusalClass

#: The rule the build report states for a planned previous-step read.
PLANNED = ("a law, provider, dynamics or observation reads the published state of a thing bound to it as it stood at "
           "the start of the step")
#: The note the build report gives a chain segment's reaction, read the previous step.
STAGGERED = ("planned: a chain segment's reaction loads the segment it is mounted on a step late, what its stator "
             "passed over the step before (a staggered coupling)")


class Lag(Enum):
    THIS_STEP = "this step"
    PREVIOUS_STEP = "previous step"
    CURRENT_ITERATE = "current iterate"


@dataclass(frozen=True)
class EdgeLag:
    """An edge MIME planned, with the step its value is read from. ``planned`` holds for every edge but one that reads
    the previous step and carries neither a thing's state nor a segment's reaction."""

    source: tuple[str, str]
    target: tuple[str, str]
    lag: Lag
    carries: str
    by: str

    @property
    def planned(self) -> bool:
        return self.lag is not Lag.PREVIOUS_STEP or self.carries in (STATE, REACTION)

    @property
    def text(self) -> str:
        return f"{self.source[0]}.{self.source[1]} -> {self.target[0]}.{self.target[1]}"


def _components(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, int]:
    """Each node's strongly connected component, by Tarjan's algorithm, iteratively; a node on no loop is a
    component of its own."""
    successors: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        if source in successors and target in successors:
            successors[source].append(target)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    component: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child = work.pop()
            if child == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            if child < len(successors[node]):
                work.append((node, child + 1))
                nxt = successors[node][child]
                if nxt not in index:
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            if low[node] == index[node]:
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component[member] = index[node]
                    if member == node:
                        break
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return component


def schedule(order: Sequence[str], groups: Sequence[Sequence[str]]) -> list[str]:
    """``order`` with each coupling group's members moved together to its first member's place."""
    member_of = {node: i for i, group in enumerate(groups) for node in group}
    placed: set[int] = set()
    out: list[str] = []
    for node in order:
        group = member_of.get(node)
        if group is None:
            out.append(node)
        elif group not in placed:
            placed.add(group)
            out.extend(n for n in order if member_of.get(n) == group)
    return out


def scheduled(order: Sequence[str], edges: Sequence[EdgeDecl], groups: Sequence[Sequence[str]]) -> list[str]:
    """The order MADDENING runs nodes added in ``order`` in: its topological sort, which keeps the nodes of a loop in
    the order they were added, then each coupling group's members moved together to its first member's place."""
    return schedule(topological_order(order, [(e.source[0], e.target[0]) for e in edges]), groups)


def classify(order: Sequence[str], edges: Sequence[EdgeDecl], groups: Sequence[Sequence[str]]) -> tuple[EdgeLag, ...]:
    """Every edge's lag, in the order the edges are added."""
    position = {node: i for i, node in enumerate(schedule(order, groups))}
    component = _components(list(order), [(e.source[0], e.target[0]) for e in edges])
    group_of = {node: i for i, group in enumerate(groups) for node in group}
    lags: list[EdgeLag] = []
    for edge in edges:
        source, target = edge.source[0], edge.target[0]
        if source in group_of and group_of.get(source) == group_of.get(target):
            lag = Lag.CURRENT_ITERATE
        elif source == target or (component[source] == component[target] and position[source] >= position[target]):
            lag = Lag.PREVIOUS_STEP
        else:
            lag = Lag.THIS_STEP
        lags.append(EdgeLag(edge.source, edge.target, lag, edge.carries, edge.by))
    return tuple(lags)


def unplanned_refusals(lags: Sequence[EdgeLag], owner_of: dict[str, str]) -> list[Refusal]:
    refusals: list[Refusal] = []
    for edge in lags:
        if edge.planned:
            continue
        refusals.append(Refusal(
            "lag.unplanned", RefusalClass.INTERNAL,
            f"the edge {edge.text}, built by {edge.by}, carries a {edge.carries} and reads it from the previous step: "
            f"it closes a feedback loop through {owner_of.get(edge.target[0], edge.target[0])} that no coupling "
            f"group solves and that no start-of-step read of a thing's state plans",
            f"a defect in how {edge.by} or the dynamics it reaches is built: solve the loop together (a coupling "
            f"group requested in a plan), or have the reader read the thing's state rather than a delivered value. "
            f"If it is MIME's: {MIME_DEFECT}"))
    return refusals


def maddening_back_edges(schedule_order: Sequence[str], edges: Sequence[tuple[str, str, str, str]],
                         groups: Sequence[Sequence[str]]) -> set[tuple[str, str, str, str]]:
    """The edges MADDENING's compiled schedule reads from the previous step: those whose source is scheduled at or
    after their target, except between members of one coupling group, which read the current iterate."""
    position = {node: i for i, node in enumerate(schedule_order)}
    group_of = {node: i for i, group in enumerate(groups) for node in group}
    return {edge for edge in edges if position.get(edge[0], -1) >= position.get(edge[2], -1)
            and not (edge[0] in group_of and group_of.get(edge[0]) == group_of.get(edge[2]))}


def cross_check(lags: Sequence[EdgeLag], schedule_order: Sequence[str],
                graph_edges: Sequence[tuple[str, str, str, str]], groups: Sequence[Sequence[str]],
                validate_lines: Sequence[str]) -> list[str]:
    """Every way MIME's classification disagrees with what MADDENING compiled: its schedule's back edges among the
    edges MIME planned, and ``validate()``'s line for each loop (staggered: some edge on the loop reads the
    previous step; iterated: all of the loop lies in one coupling group)."""
    planned = {(e.source[0], e.source[1], e.target[0], e.target[1]): e for e in lags}
    back = maddening_back_edges(schedule_order, [e for e in graph_edges if e in planned], groups)
    mine = {key for key, e in planned.items() if e.lag is Lag.PREVIOUS_STEP}
    problems = [f"MADDENING reads {a}.{b} -> {c}.{d} from the previous step; MIME classified it "
                f"{planned[(a, b, c, d)].lag.value}" for a, b, c, d in sorted(back - mine)]
    problems += [f"MIME classified {a}.{b} -> {c}.{d} as reading the previous step; MADDENING's schedule reads it "
                 f"this step or as an iterate" for a, b, c, d in sorted(mine - back)]
    by_pair: dict[tuple[str, str], set[Lag]] = {}
    for e in lags:
        by_pair.setdefault((e.source[0], e.target[0]), set()).add(e.lag)
    group_sets = [set(group) for group in groups]
    for line in validate_lines:
        staggered = line.startswith("INFO: cycle detected: ")
        iterated = line.startswith("INFO: cycle ") and "handled by iterative coupling" in line
        if not (staggered or iterated):
            continue
        body = line.removeprefix("INFO: cycle detected: ").removeprefix("INFO: cycle ")
        cycle = body.split(". Back-edges")[0].split(" handled by iterative coupling")[0].split(" -> ")
        steps = list(zip(cycle, cycle[1:] + cycle[:1], strict=True))
        if staggered and not any(Lag.PREVIOUS_STEP in by_pair.get(step, set()) for step in steps):
            problems.append(f"MADDENING staggers the loop {' -> '.join(cycle)}, and MIME classified no edge on it as "
                            f"reading the previous step")
        if iterated and not any(set(cycle) <= group for group in group_sets):
            problems.append(f"MADDENING iterates the loop {' -> '.join(cycle)} in a coupling group, and no group MIME "
                            f"planned holds it")
    return problems


def latency_in_steps(path: Sequence[EdgeLag]) -> int:
    """How many base steps a value takes along ``path`` (consecutive edges, from a thing's state to a command): one
    for each edge that reads the previous step. A controller's declared latency counts from the state as it stands
    at the start of a step (``mime.build.LATENCY_RULE``), which a read at the start of the step already takes one
    previous-step edge to reach: inside the compiled step, a controller's path is that one read and its delay line,
    which holds its writes for the declared latency's steps (``mime.build._in_graph``), so its edges count one."""
    for first, second in zip(path, path[1:], strict=False):
        if first.target[0] != second.source[0]:
            raise ValueError(f"{first.text} and {second.text} are not consecutive")
    return sum(1 for edge in path if edge.lag is Lag.PREVIOUS_STEP)


def commands(lags: Sequence[EdgeLag]) -> list[EdgeLag]:
    return [edge for edge in lags if edge.carries == COMMAND]
