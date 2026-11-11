"""Identifying a contact's friction (decided Fr5): ``mu`` fitted through ``Study.fit`` on the data a run gives (the
gradient path's damped Gauss-Newton under the fit-loop rule), with a gradient-free search over the register's range
on the same loss as its control, and an **identifiability verdict** made explicit before anyone reads the estimate:

- ``"identified"``: the data's run slides at its held points, the loss is curved in ``mu`` there (one standard error
  from the residual's derivative over its noise), and the two searches agree;
- ``"bounded below only"``: the data's run never slides (it sticks, or rolls without slipping), so its trajectory does
  not depend on ``mu`` above the threshold and the data give only a lower bound, the largest ratio of a held point's
  traction to its normal force (a test on such data cannot discriminate, and says so);
- ``"confounded"``: the data slide, but the two searches disagree or the loss is flat in ``mu`` (another axis, a torque
  amplitude declared and not measured, carrying what ``mu`` would).

A fitted value never becomes a default: it is shipped as a named record pinned by the hash of its content and of the
data it was estimated from (:meth:`FrictionFit.record`), labelled ``estimated from <data>``. Rolling resistance is
identified only from data past step-out (its derivative is zero below it, and one-sided at zero, so its search starts
above zero); its fit is the same call on its own axis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mime.results._fit import FitResult
from mime.units import Quantity

if TYPE_CHECKING:
    from mime.experiment import Experiment

#: The register's range of a friction coefficient, where the gradient-free control searches.
FRICTION_RANGE = (0.01, 1.2)
#: The golden-section search's evaluations of the loss, in the logarithm of the coefficient: its bracket then spans a
#: share 0.618^n of the range's, 1e-6 of it at 30.
CONTROL_EVALUATIONS = 30
#: How far the gradient path's estimate and the control's may differ, as a share of the control's, before the verdict
#: is ``confounded``: the control's bracket at its end, with room.
AGREEMENT = 1e-4
#: A held point slides where its slip at the solved twist is above this share of the run's largest point speed.
SLIDING = 1e-9
VERDICTS = ("identified", "bounded below only", "confounded")


@dataclass(frozen=True)
class FrictionFit:
    """A fit of a friction coefficient: the axis, the gradient path's estimate and the gradient-free control's, one
    standard error at the estimate (``None`` where the loss is flat there), the verdict and why, the lower bound the
    data hold (the largest traction over normal force at a held point), the held steps and those that slid in the
    data's run at the estimate, and the gradient path's own result."""

    axis: str
    estimate: float
    control: float
    uncertainty: float | None
    verdict: str
    why: str
    lower_bound: float
    held_steps: int
    sliding_steps: int
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
    head, found, _ = axis.rpartition(".friction.")
    if not found:
        raise ValueError(f"a friction axis is '<contact id>.friction.<coefficient>', not {axis!r}")
    return head


def _loss(residual: Callable[[Any], Any], noise: float, final: Mapping[str, Any]) -> float:
    scaled = np.asarray(residual(final), dtype=np.float64) / noise
    return 0.5 * float(np.sum(scaled * scaled))


def fit_friction(experiment: Experiment, residual: Callable[[Any], Any], *, axis: str, start: float, noise: float,
                 duration: Quantity, body: str, iterations: int = 20,
                 search: tuple[float, float] = FRICTION_RANGE) -> FrictionFit:
    """``axis`` (``"contact:<body>.friction.mu"``) fitted from ``start`` so that ``residual`` (of the run's final
    state) over ``noise`` is least, through ``Study.fit``; the gradient-free control a golden-section search of the same
    loss over ``search`` (log-spaced), each run on the built program with the axis overridden; the verdict from the
    data's run at the estimate (its ``body``'s held points' slip and traction) and the two searches' agreement."""
    from mime.study import Study

    _owner(axis)
    fitted = Study(experiment).fit(residual, start={axis: start}, noise=noise, duration=duration,
                                   iterations=iterations)
    estimate = float(np.asarray(fitted.values[axis]).reshape(()))
    built = experiment.build()

    def run(value: float, history: bool = False) -> Any:
        return built.run(duration, override={axis: value}, history=history)

    def loss(value: float) -> float:
        return _loss(residual, noise, run(value).final)

    low, high = math.log(search[0]), math.log(search[1])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low + (1 - ratio) * (high - low), low + ratio * (high - low)
    fa, fb = loss(math.exp(a)), loss(math.exp(b))
    for _ in range(CONTROL_EVALUATIONS - 2):
        if fa <= fb:
            high, b, fb = b, a, fa
            a = low + (1 - ratio) * (high - low)
            fa = loss(math.exp(a))
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = loss(math.exp(b))
    control = math.exp(0.5 * (low + high))

    data = run(estimate, history=True).history[body]
    forces = np.asarray(data["contact_forces"], dtype=np.float64)
    slips = np.asarray(data["contact_slip"], dtype=np.float64)
    traction = np.linalg.norm(np.asarray(data["contact_traction"], dtype=np.float64), axis=-1)
    twist = np.asarray(data["twist"], dtype=np.float64)
    speed = float(np.max(np.linalg.norm(twist[:, :3], axis=-1))) + float(np.finfo(np.float64).tiny)
    held = forces > 0.0
    held_steps = int(np.sum(np.any(held, axis=-1)))
    sliding_steps = int(np.sum(np.any(held & (slips > SLIDING * speed), axis=-1)))
    lower = float(np.max(np.where(held, traction / np.where(held, forces, 1.0), 0.0))) if held_steps else 0.0

    h = 1e-3 * estimate
    slope = (np.asarray(residual(run(estimate + h).final), dtype=np.float64)
             - np.asarray(residual(run(estimate - h).final), dtype=np.float64)) / (2 * h * noise)
    curvature = float(np.sum(slope * slope))
    uncertainty = 1.0 / math.sqrt(curvature) if curvature > 0.0 else None
    if sliding_steps == 0:
        verdict, why = "bounded below only", (
            f"the data's run never slides at a held point ({held_steps} held steps): its trajectory does not depend on "
            f"the coefficient above its threshold, so the data hold only mu >= {lower:.6g}")
    elif uncertainty is None or abs(estimate - control) > AGREEMENT * control + (uncertainty or 0.0):
        verdict, why = "confounded", (
            f"the data slide ({sliding_steps} of {held_steps} held steps), but the gradient path's estimate "
            f"{estimate:.9g} and the gradient-free control's {control:.9g} disagree, or the loss is flat in it")
    else:
        verdict, why = "identified", (
            f"the data slide ({sliding_steps} of {held_steps} held steps); one standard error {uncertainty:.3g}; the "
            f"gradient-free control {control:.9g} agrees")
    return FrictionFit(axis, estimate, control, uncertainty, verdict, why, lower, held_steps, sliding_steps, fitted)


__all__ = ["AGREEMENT", "CONTROL_EVALUATIONS", "FRICTION_RANGE", "SLIDING", "VERDICTS", "FrictionFit", "fit_friction"]
