"""A study's fit (``Study.fit``): the experiment's values of a few axes fitted so that a residual of its run's final
state, over its declared noise, is least, by damped Gauss-Newton steps on its derivative (the gradient path's, forward
mode) under **the fit-loop rule**: each iterate is planned again through ``build_point`` (every check the plan makes,
judged at the iterate) and run on the compiled program the start built; every validity line is judged there and
given to the optimiser as a constraint with its margin; the iterates are clipped to each axis's declared range
intersected with its leaves' bounds; and the Fisher information of the residual with the declared noise is taken at
the start (MADDENING's ``fim``), so an axis the residual cannot see is named before the first step.

**The lines.** A refusal of the iterate's plan; a flag its run raised; a node's stop; each post-run Reynolds watch,
read from the run's end, its number against the rung's declared bound (a line with a number, whose margin the step is
limited by: the step's prediction of the body's twist, from its derivative, is read by the watch's own code); and the
box. An iterate past any line is not taken: the step is halved toward the last iterate taken, and where a line's
margin limits it to nothing the fit ends ``at a line``, naming it. A line that cannot be judged (the Reynolds number
needs the medium's density, and none is declared) is never taken as inside: the start refuses (``fit.line-unjudged``),
and an iterate where one cannot be judged ends the fit ``unjudged``."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._checks import stopped_refusals
from mime.build._control import Mode
from mime.build._maddening import META, fisher_information, full_state, parameter_specs, state_put_back
from mime.build._refusals import ArgumentRefused
from mime.build._watch import WatchPlan
from mime.gradients import composed, unacknowledged
from mime.results._fit import FitIterate, FitResult, LineMargin
from mime.results._watch import read_watch
from mime.saved._pins import keeping_pins
from mime.study._axes import STUDY_REASON, Axis, range_of, resolve
from mime.study._gradient import REPORT_LEAF, Written, final_state, magnitude, objective_nodes, path_nodes, \
    path_statuses, prepare, stacked_members, written
from mime.study._point import Point, build_point, precision
from mime.study._study import stacked_state
from mime.units import Quantity

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment, Experiment

_jax: Any = jax
_jnp: Any = jax.numpy

#: How many times a step is backed off toward the last iterate taken before the fit stops where it is.
HALVINGS = 30
#: An iterate within this share of a numbered line's limit, inside it, is on the line: where the line limits the
#: step, the fit ends there.
LINE_REACHED = 1e-4
#: Where a step crossed a numbered line, the next try is this share of the way to the secant's crossing.
INSIDE = 0.999
#: The default stop of a step, relative to the values, is this many times the run's machine epsilon where that is above
#: :data:`TOLERANCE`: float32's 1.2e-7 makes 1.2e-6, float64's 2.2e-16 leaves :data:`TOLERANCE`. Measured on the gently
#: pulled sphere in float32 (one rig): the leaf's own spacing is one epsilon, the steps fall through 5.7e-7 of the value
#: while the loss still moves by 4e-4 of itself, and under 1.2e-7 they move it by 2e-4 and go no nearer the target, so a
#: stop below a few epsilons is a stop at the arithmetic's noise (the default 1e-9 spun to the iteration limit).
FLOOR_EPSILONS = 10
#: The default stop in float64, where the floor is far below it.
TOLERANCE = 1e-9
#: How many iterates in a row the loss may fail to fall, short of any line, before the fit ends ``stagnated``.
STAGNATION = 3
#: The damping's start and its factors on a step taken and on one that does not lower the loss.
DAMPING, EASED, STIFFENED = 1e-3, 1.0 / 3.0, 4.0


class _Keyed(Exception):  # noqa: N818 - control flow inside the module: an iterate that needs another program
    def __init__(self, values: Mapping[Axis, np.ndarray]) -> None:
        super().__init__()
        self.values = values


class FitRefused(BuildRefused):
    """A fit refused after it took iterates: ``partial`` is the fit as it stood (``ended`` ``refused``)."""

    partial: FitResult | None = None


def _si(value: object) -> np.ndarray:
    held = magnitude(value)
    if held is None:
        raise ArgumentRefused([Refusal(
            "fit.start", RefusalClass.USER, f"a fit starts from a value of numbers, not {value!r}",
            "give each axis its starting value as a quantity: `Study(exp).fit(residual, start={'umr.mass': kg(1e-5)}, "
            "...)`")])
    return held


def _quantity(like: object, value: np.ndarray) -> object:
    if isinstance(like, Quantity):
        return dataclasses.replace(like, value=float(value) if value.ndim == 0 else tuple(float(c) for c in value))
    return float(value)


def _box(axis: Axis, write: Written, point: Point, start: np.ndarray, declarations: Sequence[Any], built: Any,
         given: tuple[object, object] | None) -> tuple[np.ndarray, np.ndarray]:
    """The axis's box: its declared range, each written leaf's bounds read back through its factor, and ``given``,
    intersected."""
    low = np.full(start.shape, -np.inf)
    high = np.full(start.shape, np.inf)
    held = range_of(axis, declarations)
    if held is not None:
        low, high = np.maximum(low, _si(held.low)), np.minimum(high, _si(held.high))
    if given is not None:
        low, high = np.maximum(low, _si(given[0])), np.minimum(high, _si(given[1]))
    specs = parameter_specs(built.graph)
    for node, leaf in write.leaves:
        spec = specs.get(node, {}).get(leaf)
        bounds = getattr(spec, "bounds", None)
        if bounds is None:
            continue
        factor = np.asarray(write.factors[f"leaf:{node}.{leaf}"], dtype=np.float64).reshape(-1)
        at = np.asarray(point.tree[node][leaf], dtype=np.float64).reshape(-1)
        if factor.size != start.size and start.size != 1:
            raise ArgumentRefused([Refusal(
                "fit.box", RefusalClass.USER,
                f"{axis.path} writes {node}.{leaf} through a factor of {factor.size} numbers, and the axis holds "
                f"{start.size}: the leaf's bounds cannot be read back to the axis's values the iterates are kept inside",
                "fit an axis that writes the leaf one for one or by one scalar, `Study(exp).fit(..., start={'<axis>': ...})`, or "
                "one whose leaf has no bounds",
                why="an iterate past a leaf's bounds is past a line the fit would not see")])
        for edge, side in ((bounds[0], "low"), (bounds[1], "high")):
            if edge is None:
                continue
            # leaf = at + factor (theta - start): the value of theta at which the leaf reaches its bound. A negative
            # factor turns the side round; a zero one leaves the leaf where it is, whatever theta.
            moving = factor != 0.0
            reached = start.reshape(-1) + (float(edge) - at) / np.where(moving, factor, 1.0)
            floor = np.asarray(factor > 0.0) == (side == "low")
            floors = np.where(moving & floor, reached, -np.inf)
            ceilings = np.where(moving & np.logical_not(floor), reached, np.inf)
            if factor.size == start.size:
                low, high = np.maximum(low, floors.reshape(start.shape)), np.minimum(high, ceilings.reshape(start.shape))
            else:  # one number written to a leaf of several: every one of them inside
                low, high = np.maximum(low, np.max(floors)), np.minimum(high, np.min(ceilings))
    return low, high


class _Fit:
    """The fit's evaluation of an iterate on the program the start built."""

    def __init__(self, experiment: Experiment, axes: Sequence[Axis], start: Mapping[Axis, object], residual: Any,
                 duration: Quantity, noise: np.ndarray, acknowledged: Mapping[str, str],
                 taken: frozenset[str] = frozenset()) -> None:
        from mime.study._study import Study, along

        self.axes = tuple(axes)
        self.taken = taken
        self.settings, self.declarations, self.saved = experiment.declared_as()
        study = Study(experiment).sweep(along({axis.path: [start[axis]] for axis in self.axes}), duration=duration)
        plan = study.plan()
        if plan.refused:
            raise BuildRefused(list(plan.refused[0].refusals))
        group = plan.groups[0]
        built = group.built
        assert built is not None
        self.built: BuiltExperiment = built
        self.group = group
        self.leader = plan.points[0]
        self.like = dict(start)
        reference = {self.leader.key: self.leader} if self.leader.key is not None else {}
        self.reference = reference.get
        with keeping_pins(self.saved), precision(self.leader.declared.x64):
            self.writes = [written(axis, self.leader, self.settings, self.declarations, built, self.reference, None,
                                   None) for axis in self.axes]
        self.watches = tuple(plan_ for plan_ in built.plans.watches if isinstance(plan_, WatchPlan))
        graph = built.graph
        with state_put_back(graph):
            graph.reset_state()
            self.meta = dict(full_state(graph).get(META) or {})
        template = {node: {name: value[0] for name, value in fields.items()}
                    for node, fields in final_state(stacked_state([self.leader], [0], self.meta)).items()}
        self.size = int(np.prod(_jax.eval_shape(residual, template).shape, dtype=np.int64))
        watches = self.watches

        def of(final: Any) -> Any:
            parts = [_jnp.reshape(residual(final), (-1,))]
            parts += [_jnp.reshape(final[plan_.node][plan_.field], (-1,)) for plan_ in watches]
            return _jnp.concatenate(parts)

        self.of = of
        callback = built.plans.controller is not None and built.plans.controller.mode is Mode.CALLBACK
        if callback:
            raise BuildRefused([Refusal(
                "fit.callback", RefusalClass.USER,
                "a fit's steps are taken from its derivative, and this experiment's controller is called on the host, "
                "where no derivative passes (its path is cut)",
                "run the controller inside the compiled step, `exp.set_controller(..., mode='graph')`",
                why="a fit by a cut derivative would step as if the controller did not answer")])
        sources = sorted({node for write in self.writes for node, _ in (*write.leaves, *write.states)})
        self.path = path_statuses(built, path_nodes(graph, sources, objective_nodes(of, template)), False)
        refused = unacknowledged(self.path, acknowledged)
        if refused:
            raise BuildRefused([Refusal(
                "gradient.path-status", RefusalClass.USER,
                "the fit's derivative passes through " + "; ".join(
                    f"{owner}, whose status is {self.path[owner].value}" for owner in refused),
                "acknowledge each with its reason, `Study(exp).fit(..., acknowledged={" +
                ", ".join(f"{owner!r}: '<why>'" for owner in refused) + "})`",
                why="a cut or unverified path is refused unless acknowledged (mime.gradients)")])
        params = {"nodes": self.leader.tree, "mappings": {}}
        marks = tuple((node, leaf, None) for node, leaves in self.leader.tree.items() for leaf in leaves)
        with precision(self.leader.declared.x64):
            self.prepared = prepare(built, self.writes, of, "forward", params, marks, False, None, group.steps)
        self.noise = np.broadcast_to(noise, (self.size,)).copy()

    def point(self, values: Mapping[Axis, np.ndarray]) -> Point:
        declared = {axis: _quantity(self.like[axis], value) for axis, value in values.items()}
        with keeping_pins(self.saved), precision(self.leader.declared.x64):
            return build_point(-1, self.settings, self.declarations, declared, STUDY_REASON, self.reference)

    def evaluate(self, point: Point) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[LineMargin, ...], Any]:
        """The scaled residual, its derivative in each axis (scaled, the residual's rows first), the lines judged
        from the run, and the watched twists with their derivative (for the margins' prediction)."""
        with precision(self.leader.declared.x64):
            theta0, factors = stacked_members([point], [0], self.writes, self.settings, self.declarations,
                                       np.float64 if _jnp.zeros(()).dtype == np.float64 else np.float32)
            theta0 = {path: held[0] for path, held in theta0.items()}
            factors = {key_: held[0] for key_, held in factors.items()}
            state = {node: {name: value[0] for name, value in fields.items()}
                     for node, fields in stacked_state([point], [0], self.meta).items()}
            params = {"nodes": point.tree, "mappings": {}}
            taps = {node: params["nodes"][node][REPORT_LEAF] for node in self.prepared.solving}
            value, jacobian, _, aux = self.prepared.function(theta0, theta0, factors, params, state, taps)
        value = np.asarray(value, dtype=np.float64)
        tally, final, _ = aux
        final = {node: {name: np.asarray(held) for name, held in fields.items()} for node, fields in final.items()}
        scaled = value[:self.size] / self.noise
        jac = {path: np.asarray(held, dtype=np.float64) for path, held in jacobian.items()}
        rows = {path: np.reshape(held, (-1, value.size))[:, :self.size] / self.noise for path, held in jac.items()}
        watches = self.watches_of(point)
        twists = (value[self.size:], {path: np.reshape(held, (-1, value.size))[:, self.size:]
                                      for path, held in jac.items()}, watches)
        lines = list(self.run_lines(tally, final, watches))
        return scaled, rows, tuple(lines), twists

    def watches_of(self, point: Point) -> tuple[WatchPlan, ...]:
        """The iterate's own watches (its plan's: the hull's diameter and the medium's values at its values), in the
        order the objective reads their twists."""
        plans = point.plans
        own = {(held.node, held.field): held for held in (plans.watches if plans is not None else ())
               if isinstance(held, WatchPlan)}
        return tuple(own.get((held.node, held.field), held) for held in self.watches)

    def run_lines(self, tally: Any, final: Mapping[str, Any],
                  watches: Sequence[WatchPlan]) -> list[LineMargin]:
        counted, _, _ = (np.asarray(each) for each in tally)
        meanings = dict(self.prepared.meanings)
        states = {i for i, flag in enumerate(self.built.layout.flags) if flag.name in self.taken}
        found = [LineMargin(f"{test.node}.{test.field}", "run", float(counted[i]), 0.0, -float(counted[i]),
                            meanings.get(f"{test.node}.{test.field}", ""))
                 for i, test in enumerate(self.prepared.tests) if int(counted[i]) > 0 and i not in states]
        found += [LineMargin(refusal.code, "run", 1.0, 0.0, -1.0, refusal.message)
                  for refusal in stopped_refusals(self.built.graph, final)]
        found += self.watch_lines(final, watches)
        return found

    def watch_lines(self, final: Mapping[str, Any], watches: Sequence[WatchPlan]) -> list[LineMargin]:
        found: list[LineMargin] = []
        for plan_ in watches:
            read = read_watch(plan_, cast(Any, final), None)
            if read.upper is None:
                continue
            if read.reynolds is None:
                found.append(LineMargin(f"reynolds:{plan_.owner}", "run", float("nan"), float(read.upper),
                                        float("nan"), "not computed: " + "; ".join(read.missing)))
                continue
            found.append(LineMargin(f"reynolds:{plan_.owner}", "run", float(read.reynolds), float(read.upper),
                                    float(read.upper - read.reynolds),
                                    f"the Reynolds number the body moved at, against the {plan_.entry} rung's "
                                    f"declared bound"))
        return found

    def predicted_lines(self, twists: Any, step: Mapping[str, np.ndarray]) -> list[LineMargin]:
        """The watches' lines at ``theta + step``, from the twists' derivative (a linear prediction of the twist,
        read by the watch's own code)."""
        values, rows, watches = twists
        moved = np.asarray(values, dtype=np.float64).copy()
        for path, held in rows.items():
            moved = moved + np.reshape(step[path], (-1,)) @ held
        final: dict[str, dict[str, np.ndarray]] = {}
        offset = 0
        for plan_ in watches:
            final.setdefault(plan_.node, {})[plan_.field] = moved[offset:offset + 6]
            offset += 6
        return self.watch_lines(final, watches)


def fit(experiment: Experiment, residual: Callable[[Any], Any], start: Mapping[object, object], noise: object,
        duration: Quantity, iterations: int, bounds: Mapping[object, tuple[object, object]] | None,
        tolerance: float | None, acknowledged: Mapping[str, str] | None,
        taken: frozenset[str] = frozenset()) -> FitResult:
    """``Study.fit``; ``taken``, the flags a caller's fit takes as a state the run is in rather than a line (none for
    ``Study.fit``: ``mime.study.fit_friction`` takes ``touching``, its bodies touching by design)."""
    if not callable(residual):
        raise ArgumentRefused([Refusal("call.wrong-type", RefusalClass.USER,
                                       f"the residual is a function of the final state, not {type(residual).__name__}",
                                       "write residual=lambda final: final['ball']['pose'][:3] - target")])
    if not start:
        raise ArgumentRefused([Refusal("fit.start", RefusalClass.USER, "a fit is given no axis to start from",
                                       "name each axis with its starting value: `Study(exp).fit(residual, "
                                       "start={'umr.mass': kg(1e-5)}, noise=..., duration=...)`")])
    _, declarations, _ = experiment.declared_as()
    axes = [resolve(path, declarations) for path in start]
    starts = {axis: value for axis, value in zip(axes, start.values(), strict=True)}
    sigma = np.asarray(_si(noise) if isinstance(noise, Quantity) else noise, np.float64) if noise is not None else None
    if sigma is None or not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ArgumentRefused([Refusal("fit.noise", RefusalClass.USER,
                                       f"a fit is given the residual's declared noise, a positive number (or one a "
                                       f"residual entry), not {noise!r}: the Fisher information and the loss are "
                                       f"over it, and a unit noise would make them a statement about nothing",
                                       "declare the residual's noise, `Study(exp).fit(..., noise=1e-3)`")])
    run = _Fit(experiment, axes, starts, residual, duration, sigma, dict(acknowledged or {}), taken)
    given = {resolve(path, declarations): pair for path, pair in (bounds or {}).items()}
    theta = {axis: _si(starts[axis]) for axis in axes}
    boxes = {axis: _box(axis, write, run.leader, theta[axis], declarations, run.built, given.get(axis))
             for axis, write in zip(axes, run.writes, strict=True)}
    outside = [axis.path for axis, value in theta.items() if not np.array_equal(np.clip(value, *boxes[axis]), value)]
    if outside:
        raise ArgumentRefused([Refusal(
            "fit.start", RefusalClass.USER,
            f"the fit starts {', '.join(outside)} outside its box (its declared range, its leaves' bounds and the "
            f"bounds given), which its iterates are kept inside",
            "start inside the box: `Study(exp).fit(..., start={...})` within the range `exp.declare_range(...)` "
            "declares, or widen the range")])
    fim, insensitive = _fisher(run, theta)
    with precision(run.leader.declared.x64):  # the run's own dtype, which a process default can set
        x64 = bool(_jnp.zeros(()).dtype == np.float64)
    stop = tolerance if tolerance is not None else default_tolerance(x64)
    return _descend(run, axes, theta, boxes, iterations, stop, fim, insensitive)


def default_tolerance(x64: bool) -> float:
    """The step's stop, relative to the values, where none is given: :data:`TOLERANCE`, or :data:`FLOOR_EPSILONS` times
    the run's machine epsilon where that is higher (float32)."""
    return max(TOLERANCE, FLOOR_EPSILONS * float(np.finfo(np.float64 if x64 else np.float32).eps))


def _fisher(run: _Fit, theta: Mapping[Axis, np.ndarray]) -> tuple[Any, tuple[str, ...]]:
    """MADDENING's Fisher information of the residual (its rows, not the watched twists) with the declared noise, at
    the start, and every axis it cannot resolve."""
    point = run.point(theta)
    theta0, factors = stacked_members([point], [0], run.writes, run.settings, run.declarations, np.float64)
    theta0 = {path: held[0] for path, held in theta0.items()}
    factors = {key_: held[0] for key_, held in factors.items()}
    state = {node: {name: value[0] for name, value in fields.items()}
             for node, fields in stacked_state([point], [0], run.meta).items()}
    params = {"nodes": point.tree, "mappings": {}}
    taps = {node: params["nodes"][node][REPORT_LEAF] for node in run.prepared.solving}
    objective = run.prepared.objective

    def residual(given: Mapping[str, Any]) -> Any:
        value, _ = objective(dict(given), theta0, factors, params, state, taps)
        return value[:run.size]

    with precision(run.leader.declared.x64):
        report = fisher_information(residual, {path: _jnp.asarray(value) for path, value in theta0.items()},
                                    run.noise)
    return report, _insensitive(report, tuple(theta0))


def _insensitive(report: Any, paths: Sequence[str]) -> tuple[str, ...]:
    """Every axis the information cannot resolve: those with support in its null space (a Cramer-Rao bound of
    infinity), by axis path; not one at exactly 0.0, whose relative scale is none (``zero_scaled``: the column
    vanishes whatever the data say, which the report holds)."""
    bound = np.reshape(np.asarray(report.crb), (-1,))
    unscaled = set(report.zero_scaled)
    found: list[str] = []
    for name, held in zip(report.param_names, bound, strict=True):
        if np.isfinite(held) or name in unscaled:
            continue
        # MADDENING names a leaf by its key path, ``['<path>']`` for a flat dict, ``['<path>'][i]`` for an entry.
        path = next((path for path in paths if name == path or name.startswith(f"[{path!r}]")), name)
        if path not in found:
            found.append(path)
    return tuple(found)


def _flat(values: Mapping[Axis, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.reshape(value, (-1,)) for value in values.values()])


def _descend(run: _Fit, axes: Sequence[Axis], theta: dict[Axis, np.ndarray],
             boxes: Mapping[Axis, tuple[np.ndarray, np.ndarray]], iterations: int, tolerance: float, fim: Any,
             insensitive: tuple[str, ...]) -> FitResult:
    paths = [axis.path for axis in axes]
    start = {axis.path: value.copy() for axis, value in theta.items()}
    iterates: list[FitIterate] = []

    def judge(values: Mapping[Axis, np.ndarray], taken_if: Callable[[float, tuple[LineMargin, ...]], tuple[bool, str]]
              ) -> tuple[FitIterate, Any, np.ndarray]:
        point = run.point(values)
        box = tuple(LineMargin(f"box:{axis.path}", "box", float(np.max(values[axis])), float(np.max(high)),
                               float(min(np.min(values[axis] - low), np.min(high - values[axis]))))
                    for axis, (low, high) in boxes.items())
        if point.refused:
            lines = tuple(LineMargin(refusal.code, "plan", 1.0, 0.0, -1.0, refusal.message)
                          for refusal in point.refusals) + box
            iterate = FitIterate(len(iterates), {a.path: v.copy() for a, v in values.items()}, float("inf"),
                                 np.zeros(0), {}, lines, False, "its plan is refused")
            return iterate, None, np.zeros((0, 0))
        if point.key != run.leader.key:
            raise _Keyed(values)
        scaled, rows, lines, twists = run.evaluate(point)
        loss = 0.5 * float(scaled @ scaled)
        taken, why = taken_if(loss, lines)
        jacobian = np.concatenate([rows[path] for path in sorted(rows)], axis=0).T  # m x n
        iterate = FitIterate(len(iterates), {a.path: v.copy() for a, v in values.items()}, loss, scaled,
                             {path: rows[path] for path in paths}, (*lines, *box), taken, why,
                             dict(point.computed or {}))
        return iterate, (scaled, twists), jacobian

    current: FitIterate
    current, held, jacobian = judge(theta, lambda loss, lines: (not any(line.past for line in lines),
                                                                  "the start" if not any(line.past for line in lines)
                                                                  else "the start is past a line"))
    iterates.append(current)
    if current.unjudged:
        raise BuildRefused([Refusal(
            "fit.line-unjudged", RefusalClass.USER,
            "the start has a validity line that cannot be judged, and a fit does not take a line it did not judge as "
            "inside: " + "; ".join(line.text() for line in current.unjudged),
            "declare what the line needs (the Reynolds number needs the medium's density, `exp.declare("
            "'water.density', ...)`), or drop the rung the line belongs to",
            why="an iterate past an unjudged line would be taken, and the fit would end converged past it")])
    if not current.taken:
        return _result(run, paths, start, theta, current, "at a line", _past_names(current), iterates, fim,
                       insensitive)
    damping, gain = DAMPING, 1.0
    ended, why = "iterations", f"{iterations} iterates judged"
    order = sorted(range(len(axes)), key=lambda i: axes[i].path)
    stop, rejected = False, 0
    while len(iterates) < iterations and not stop:
        scaled, twists = held
        gradient = jacobian.T @ scaled
        normal = jacobian.T @ jacobian
        system: np.ndarray = normal + damping * np.diag(np.diag(normal)) + np.finfo(np.float64).tiny * np.eye(len(normal))
        step_flat: np.ndarray = -cast(np.ndarray, np.linalg.solve(system, gradient))
        wanted = _unflat(step_flat, theta, axes, order)
        trial = {axis: np.clip(theta[axis] + wanted[axis], *boxes[axis]) for axis in axes}
        step = {axis: trial[axis] - theta[axis] for axis in axes}
        scale, line, predicted = _line_limit(run, twists, step, current.lines, gain)
        reached = _reached(current.lines, line)
        size = float(np.max(np.abs(_flat(step))) * scale)
        floor = tolerance * max(float(np.max(np.abs(_flat(theta)))), np.finfo(np.float64).tiny)
        if scale <= 0.0 or (line and reached):
            ended, why = "at a line", line
            break
        if size <= floor:
            pinned = [axis.path for axis in axes if np.max(np.abs(step[axis] - wanted[axis])) > floor]
            if line:
                ended, why = "at a line", line
            elif pinned:
                ended, why = "at a line", ", ".join(f"box:{path}" for path in pinned)
            else:
                ended, why = "converged", f"the step fell below {tolerance:g} of the values"
            break
        taken = False
        try:
            for _ in range(HALVINGS):
                values = {axis: theta[axis] + scale * step[axis] for axis in axes}
                iterate, candidate, candidate_jacobian = judge(values, lambda loss, lines: _rule(loss, lines, current))
                iterates.append(iterate)
                if iterate.taken:
                    if line:
                        gain = _measured_gain(current.lines, iterate.lines, line, predicted, gain)
                    theta, current, held, jacobian = values, iterate, candidate, candidate_jacobian
                    damping *= EASED
                    taken, rejected = True, 0
                    break
                if iterate.unjudged and not iterate.past:
                    ended, why, stop = "unjudged", "; ".join(line_.text() for line_ in iterate.unjudged), True
                    break
                if not iterate.past or len(iterates) >= iterations:
                    damping *= STIFFENED
                    rejected += 1
                    break
                scale = _back_off(scale, current.lines, iterate.lines)
        except _Keyed as keyed:
            at = ", ".join(f"{axis.path} = {np.array2string(value, precision=10)}" for axis, value in
                           keyed.values.items())
            refused = FitRefused([Refusal(
                "fit.value-keyed", RefusalClass.USER,
                f"the iterate at {at} (judged after {sum(1 for each in iterates if each.taken)} taken) needs another "
                f"compiled program than the start's: an entry makes a choice from the value when it is built, so the "
                f"fit's objective changes under it",
                "declare a range of each fitted axis, `exp.declare_range(path, low, high, reason=...)`, which fixes "
                "the choice over it and bounds the fit; the iterates taken are in the error's `partial`",
                why="a fit whose iterates are built for their own values converges on a moving objective")])
            refused.partial = _result(run, paths, start, theta, current, "refused", f"the iterate at {at} needs "
                                      f"another compiled program", iterates, fim, insensitive)
            raise refused from None
        if stop:
            break
        if rejected >= STAGNATION and not taken:
            ended, why = "stagnated", (f"the loss did not fall in {rejected} iterates in a row (damping {damping:.3g}), "
                                       f"short of any line: the step is below what the arithmetic resolves")
            break
        if not taken and iterates[-1].past and len(iterates) < iterations:
            ended, why = "at a line", _past_names(iterates[-1])
            break
    return _result(run, paths, start, theta, current, ended, why, iterates, fim, insensitive)


def _rule(loss: float, lines: tuple[LineMargin, ...], current: FitIterate) -> tuple[bool, str]:
    """Whether an iterate is taken: past no line, every line judged (a NaN margin is not a pass), and its loss below the
    last taken iterate's."""
    past = [line.name for line in lines if line.past]
    if past:
        return False, f"past {', '.join(past)}"
    unjudged = [line.name for line in lines if not line.judged]
    if unjudged:
        return False, f"cannot judge {', '.join(unjudged)}"
    if loss < current.loss:
        return True, "the loss fell"
    return False, "the loss did not fall"


def _measured_gain(before: Sequence[LineMargin], after: Sequence[LineMargin], name: str, predicted: float,
                   gain: float) -> float:
    """The share of its predicted change a line's margin made over the last limited step, applied to the next
    prediction: where the line's measure depends on the values beyond the twist (a hull's diameter), the linear
    prediction of the twist alone misjudges it, and this measures by how much."""
    old = next((line.margin for line in before if line.name == name), None)
    new = next((line.margin for line in after if line.name == name), None)
    if old is None or new is None or not np.isfinite(predicted) or predicted == old:
        return gain
    return float(np.clip(gain * (new - old) / (predicted - old), 0.05, 20.0))


def _reached(lines: Sequence[LineMargin], name: str) -> bool:
    """Whether the iterate lies on the line ``name``: inside it by under :data:`LINE_REACHED` of its limit."""
    held = next((line for line in lines if line.name == name), None)
    return held is not None and np.isfinite(held.limit) and 0.0 <= held.margin <= LINE_REACHED * abs(held.limit)


def _back_off(scale: float, inside: Sequence[LineMargin], past: Sequence[LineMargin]) -> float:
    """The fraction of the step to try after one past a line: where a numbered line was crossed, the secant's
    crossing between the two iterates' margins, a little inside it (:data:`INSIDE`); half the step otherwise."""
    before = {line.name: line.margin for line in inside if line.judged and np.isfinite(line.limit)}
    crossings = [scale * before[line.name] / (before[line.name] - line.margin) for line in past
                 if line.past and line.name in before and before[line.name] > 0.0 and np.isfinite(line.limit)]
    if crossings and len(crossings) == len([line for line in past if line.past]):
        return min(crossings) * INSIDE
    return 0.5 * scale


def _past_names(iterate: FitIterate) -> str:
    return ", ".join(line.name for line in iterate.lines if line.past) or "none"


def _unflat(flat: np.ndarray, like: Mapping[Axis, np.ndarray], axes: Sequence[Axis],
            order: Sequence[int]) -> dict[Axis, np.ndarray]:
    """A flat step (the axes in their paths' sorted order) as each axis's own shape."""
    found: dict[Axis, np.ndarray] = {}
    offset = 0
    for index in order:
        axis = axes[index]
        size = int(np.size(like[axis]))
        found[axis] = np.reshape(flat[offset:offset + size], np.shape(like[axis]))
        offset += size
    return found


def _line_limit(run: _Fit, twists: Any, step: Mapping[Axis, np.ndarray], lines: Sequence[LineMargin],
                gain: float = 1.0) -> tuple[float, str, float]:
    """The largest fraction of ``step`` at which every numbered watch line inside at the iterate stays inside by its
    prediction (bisected), the line that limits it ("" where none does), and that line's predicted margin there. The
    prediction is linear in the body's twist, read by the watch's own code at the iterate's plan; ``gain`` scales its
    change from the iterate's margin by what the last limited step measured of it (the hull's own size moves too)."""
    by_path = {axis.path: value for axis, value in step.items()}
    inside = {line.name: line.margin for line in lines if line.name.startswith("reynolds:") and line.judged and
              not line.past}

    def worst(fraction: float) -> tuple[float, str]:
        predicted = run.predicted_lines(twists, {path: fraction * value for path, value in by_path.items()})
        held = [(inside[line.name] + gain * (line.margin - inside[line.name]), line.name) for line in predicted
                if line.name in inside and line.judged]
        return min(held) if held else (np.inf, "")

    margin, name = worst(1.0)
    if margin >= 0.0:
        return 1.0, "", margin
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if worst(middle)[0] >= 0.0:
            low = middle
        else:
            high = middle
    return low, name, worst(low)[0]


def _result(run: _Fit, paths: Sequence[str], start: Mapping[str, np.ndarray], theta: Mapping[Axis, np.ndarray],
            current: FitIterate, ended: str, why: str, iterates: Sequence[FitIterate], fim: Any,
            insensitive: tuple[str, ...]) -> FitResult:
    return FitResult(tuple(paths), dict(start), {axis.path: value.copy() for axis, value in theta.items()},
                     current.loss, ended, why, tuple(iterates), fim, insensitive, composed(run.path.values()),
                     dict(run.path))


__all__ = ["DAMPING", "FLOOR_EPSILONS", "HALVINGS", "INSIDE", "LINE_REACHED", "STAGNATION", "TOLERANCE",
           "default_tolerance", "fit"]
