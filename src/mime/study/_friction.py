"""Identifying a contact's friction (decided Fr5): ``mu`` fitted through ``Study.fit`` on the data a run gives (the
gradient path's damped Gauss-Newton under the fit-loop rule), with a gradient-free search over the register's range
on the same loss as its control, and an **identifiability verdict** made explicit before anyone reads the estimate:

- ``"identified"``: the loss is curved in ``mu`` on both sides of the estimate, one standard error read on each side
  apart (from the residual's change over its noise) within :data:`IDENTIFIED_SHARE` of it, and the two searches
  agree: a verdict claims no more than its standard error supports;
- ``"bounded below only"``: the loss is flat above the estimate: a larger coefficient moves no datum, as for data that
  never slide (they stick, or roll without slipping), whose trajectory does not depend on ``mu`` above its threshold;
  the data give only a lower bound, the largest ratio of the body's whole traction to its whole normal force at a held
  step of the run at the estimate (which reproduces the data);
- ``"confounded"``: the loss is curved above the estimate, but not within that share on both sides, or the two
  searches disagree (another axis, a torque amplitude declared and not measured, carrying what ``mu`` would).

The gradient path runs ``Study.fit``'s, with ``touching`` taken as the state its bodies are in (:data:`TAKEN`; every
other fit keeps it a line) and its iterates kept in the register's range; where its start leaves it stuck in the
region where the model sticks (its loss above the control's), it starts again from the control.

A fitted value never becomes a default: it is shipped as a named record pinned by the hash of its content and of the
data it was estimated from (:meth:`FrictionFit.record`), labelled ``estimated from <data>``. Rolling resistance is
identified only from data past step-out (its derivative is zero below it); its fit is the same call on its own
axis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mime.results._fit import FitResult
from mime.units import Quantity

if TYPE_CHECKING:
    from mime.experiment import Experiment

#: The register's range of a friction coefficient, where the gradient-free control searches and the gradient path's
#: iterates are kept.
FRICTION_RANGE = (0.01, 1.2)
#: The gradient-free control's two sweeps: log-spaced coefficients over the register's range, then over the two cells
#: about the least loss, the control the vertex of the parabola through the least and its neighbours there (the two
#: cells span a twelfth of the range's logarithm, the second sweep's cells a 72nd of it, about 7 % of the coefficient,
#: within which the parabola's vertex is the least loss's to the loss's own third derivative).
CONTROL_POINTS = (25, 13)
#: How far the gradient path's estimate and the control's may differ, as a share of the control's, before the verdict
#: is ``confounded``: the control's own resolution, the parabola's vertex in cells of about 7 % of the coefficient (on
#: two of twenty sliding draws the control stood 6e-4 of it from an estimate equal to the truth to 1e-6, the fix round's
#: probe), with room.
AGREEMENT = 2e-3
#: The verdict claims ``identified`` only where one standard error, read from the loss's curvature on each side of
#: the estimate apart, is within this share of it: a verdict claims no more than its standard error supports (data that
#: never slide leave the loss flat above the threshold, a standard error of about 1e11 from that side).
IDENTIFIED_SHARE = 0.1
#: The step either side of the estimate the curvature is read over, a share of the estimate.
CURVATURE_STEP = 1e-3
#: A held point slides where its traction stands on its limit (``mu`` times its normal force) within this share, the
#: sliding pair's update reaching its disk's boundary to rounding, and its slip is above zero: a test on forces, never
#: on a speed relative to the run's own (a body at rest moves at 1e-18 m/s, and its rounding slipped past any share
#: of that).
ON_LIMIT = 1e-6
#: A gradient path whose loss stands this far above the control's (half a unit of chi-square, one standard error of
#: one coefficient) is started again from the control: a start in the region where the model sticks leaves the path
#: there, its derivative zero.
RESTART_LOSS = 0.5
#: The flags a friction fit takes as a state its run is in, never a line (``mime.study._fit.fit``'s ``taken``): its
#: bodies touch the wall by design, the contact working. Every other fit keeps ``touching`` a line.
TAKEN = frozenset({"touching"})
VERDICTS = ("identified", "bounded below only", "confounded")


@dataclass(frozen=True)
class FrictionFit:
    """A fit of a friction coefficient: the axis, the gradient path's estimate and the gradient-free control's, one
    standard error at the estimate (``None`` where the loss is flat on either side of it), the verdict and why, the
    lower bound the data hold where that is all they hold (``bounded below only``: the largest traction over normal
    force at a held step of the run at the estimate, which reproduces the data; ``None`` otherwise), the held steps and
    those that slid in the run at the estimate, whether the gradient path was started again from the control, and the
    gradient path's own result."""

    axis: str
    estimate: float
    control: float
    uncertainty: float | None
    verdict: str
    why: str
    lower_bound: float | None
    held_steps: int
    sliding_steps: int
    restarted: bool
    fit: FitResult

    def record(self, name: str, data: bytes) -> dict[str, Any]:
        """The estimate as a named record pinned by its content's hash and its data's (``data``, the bytes the residual
        was made from): what ships, never a default. Its ``label`` says what it was estimated from."""
        held = {"name": name, "axis": self.axis, "value": self.estimate, "uncertainty": self.uncertainty,
                "verdict": self.verdict, "lower_bound": self.lower_bound, "control": self.control,
                "data_sha256": hashlib.sha256(data).hexdigest(),
                "label": f"estimated from {name}'s data ({self.verdict}), never a default"}
        return {**held, "sha256": hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest()}


def _owner(axis: str) -> str:
    """The contact's id an axis ``<contact id>.friction.<coefficient>`` belongs to."""
    from mime.build import Refusal, RefusalClass
    from mime.build._refusals import ArgumentRefused

    head, found, _ = axis.rpartition(".friction.")
    if not found:
        raise ArgumentRefused([Refusal("fit.start", RefusalClass.USER,
                                       f"fit_friction starts from a friction coefficient's axis, "
                                       f"'<contact id>.friction.<coefficient>', not {axis!r}",
                                       "name the contact's coefficient: `fit_friction(..., "
                                       "axis='contact:ball.friction.mu', ...)`")])
    return head


def _loss(residual: Callable[[Any], Any], noise: float, final: Mapping[str, Any]) -> float:
    scaled = np.asarray(residual(final), dtype=np.float64) / noise
    return 0.5 * float(np.sum(scaled * scaled))


def fit_friction(experiment: Experiment, residual: Callable[[Any], Any], *, axis: str, start: float, noise: float,
                 duration: Quantity, body: str, iterations: int = 20,
                 search: tuple[float, float] = FRICTION_RANGE) -> FrictionFit:
    """``axis`` (``"contact:<body>.friction.mu"``) fitted from ``start`` so that ``residual`` (of the run's final
    state) over ``noise`` is least, through ``Study.fit``'s path (``touching`` taken, :data:`TAKEN`), its iterates kept
    in ``search``; the gradient-free control a search of the same loss over ``search`` (log-spaced grids narrowed round
    by round, each a study's sweep, every member from the start), from which the gradient path starts again where its
    own start left it stuck (:data:`RESTART_LOSS`); the verdict from the loss's curvature on each side of the estimate,
    the two searches' agreement, and the run at the estimate (its ``body``'s held points' traction and slip)."""
    from mime.study._fit import fit

    owner = _owner(axis)
    if not search[0] <= start <= search[1]:
        from mime.build import Refusal, RefusalClass
        from mime.build._refusals import ArgumentRefused

        raise ArgumentRefused([Refusal("fit.start", RefusalClass.USER,
                                       f"fit_friction starts {axis} at {start!r}, outside its search {search} (the "
                                       f"register's range unless written), where its iterates are kept",
                                       "start inside it, or widen it: `fit_friction(..., search=(0.01, 2.5))`")])

    def finals(values: list[float]) -> list[Mapping[str, Any]]:
        """Each value's run from the start, its final state: one study's sweep (one build, its members batched)."""
        from mime.study import Study, along

        return [member.final for member in Study(experiment).sweep(along({axis: values}), duration=duration).run()]

    def path(begin: float) -> tuple[FitResult, float]:
        found = fit(experiment, residual, {axis: begin}, noise, duration, iterations, {axis: search}, None, None,
                    taken=TAKEN)
        return found, float(np.asarray(found.values[axis]).reshape(()))

    low, high = math.log(search[0]), math.log(search[1])
    grid = list(np.linspace(low, high, CONTROL_POINTS[0]))
    held = [_loss(residual, noise, final) for final in finals([math.exp(x) for x in grid])]
    best = int(np.argmin(held))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    grid = list(np.linspace(low, high, CONTROL_POINTS[1]))
    held = [_loss(residual, noise, final) for final in finals([math.exp(x) for x in grid])]
    best = min(max(int(np.argmin(held)), 1), len(grid) - 2)
    # the parabola through the least loss and its neighbours, its vertex the control (inside their cells)
    x0, x1, x2 = grid[best - 1:best + 2]
    y0, y1, y2 = held[best - 1:best + 2]
    bend = y0 - 2 * y1 + y2
    vertex = x1 - 0.5 * (x1 - x0) * (y2 - y0) / bend if bend > 0.0 else x1
    control = math.exp(min(max(vertex, x0), x2))

    fitted, estimate = path(start)
    lost, against = (_loss(residual, noise, final) for final in finals([estimate, control]))
    restarted = lost > against + RESTART_LOSS
    if restarted:
        fitted, estimate = path(control)
    h = CURVATURE_STEP * estimate
    below, at, above = (np.asarray(residual(final), dtype=np.float64)
                        for final in finals([estimate - h, estimate, estimate + h]))
    errors: list[float | None] = []
    for side in (below, above):
        curvature = float(np.sum(((side - at) / (h * noise)) ** 2))
        errors.append(1.0 / math.sqrt(curvature) if curvature > 0.0 else None)
    flat = [error is None or error > IDENTIFIED_SHARE * estimate for error in errors]
    uncertainty = None if any(flat) else max(cast(float, error) for error in errors)

    history = experiment.build().run(duration, override={axis: estimate}, history=True).history
    data: Mapping[str, Any] = {} if history is None else cast(Mapping[str, Any], history[body])
    forces = np.asarray(data["contact_forces"], dtype=np.float64)
    slips = np.asarray(data["contact_slip_speed"], dtype=np.float64)
    traction = np.asarray(data["contact_friction_forces"], dtype=np.float64)
    mu = estimate if axis.endswith(".friction.mu") else float(cast(Any, experiment.handle(owner)).law.friction.mu)
    held_points = forces > 0.0
    on_limit = np.linalg.norm(traction, axis=-1) >= (1.0 - ON_LIMIT) * mu * forces
    held_steps = int(np.sum(np.any(held_points, axis=-1)))
    sliding_steps = int(np.sum(np.any(held_points & on_limit & (slips > 0.0), axis=-1)))
    shown = ", ".join(f"{name} {'flat' if error is None or error > IDENTIFIED_SHARE * estimate else f'{error:.3g}'}"
                      for name, error in zip(("below", "above"), errors, strict=True))
    lower: float | None = None
    if uncertainty is not None and abs(estimate - control) <= AGREEMENT * control + uncertainty:
        verdict, why = "identified", (
            f"one standard error {uncertainty:.3g} (read on each side of the estimate: {shown}), within "
            f"{IDENTIFIED_SHARE:g} of it; the gradient-free control {control:.9g} agrees; the run at the estimate "
            f"slides on {sliding_steps} of {held_steps} held steps")
    elif flat[1]:
        # the body's whole traction over its whole normal force at each held step: |sum t_k| <= mu sum n_k whatever
        # the points' split (one point held by several groups shares its forces in no unique way), so its largest is
        # a bound; the run at the estimate reproduces the data, which raising mu does not move
        total = np.sum(forces, axis=-1)
        pulled = np.linalg.norm(np.sum(traction, axis=-2), axis=-1)
        lower = float(np.max(np.where(total > 0.0, pulled / np.where(total > 0.0, total, 1.0), 0.0))) \
            if held_steps else 0.0
        verdict, why = "bounded below only", (
            f"the loss is flat above the estimate (one standard error read on each side: {shown}): a larger "
            f"coefficient moves no datum, so the data hold only mu >= {lower:.6g} (the run at the estimate slides on "
            f"{sliding_steps} of {held_steps} held steps)")
    else:
        verdict, why = "confounded", (
            f"the loss is curved above the estimate but its standard error ({shown}) is not within "
            f"{IDENTIFIED_SHARE:g} of it on both sides, or the gradient path's estimate {estimate:.9g} and the "
            f"gradient-free control's {control:.9g} disagree: another axis may carry what mu would")
    return FrictionFit(axis, estimate, control, uncertainty, verdict, why, lower, held_steps, sliding_steps, restarted,
                       fitted)


__all__ = ["AGREEMENT", "CONTROL_POINTS", "CURVATURE_STEP", "FRICTION_RANGE", "IDENTIFIED_SHARE", "ON_LIMIT",
           "RESTART_LOSS", "TAKEN", "VERDICTS", "FrictionFit", "fit_friction"]
