"""The post-run Reynolds-number watch: after a run, the Reynolds number the body moved at, against the number its rung
is declared up to, and the size of the inertia the rung neglects where that is known.

A watch is read from the run's result and the plan's inputs (``mime.build._watch``), after the graph has run: it adds
no node, so the graph and its derived hash are what they would be without it, and it raises a flag on the result,
never a refusal. Two numbers are read, both by the hull's diameter ``D``, the medium's density ``rho`` and viscosity
``mu``: the translation's, ``rho |v| D / mu`` from the body's speed, and the turning's, ``rho |w| r D / mu`` from the
speed of the hull's surface at its radial reach ``r = D / 2`` (the speed a turning screw's surface moves at, which its
translation does not show, and the number that is also the quasi-steady parameter of a body turning at ``|w|``). The
larger is compared with the bound. From the whole history of a run that kept it, the peak of each over the run is
read; from a run that kept none, the last step's, which says nothing of the steps before it.

The size of the neglected inertia is known for a sphere's translation: Oseen's first correction raises its drag by
``3 Re / 16``, the leading term of the expansion in the Reynolds number. It holds as Re tends to zero, is within a few
per cent of the measured drag to Re of order one, and overstates the rise beyond (an empirical correlation of a sphere's
drag, Schiller and Naumann's, gives about half Oseen's rise at Re of four: quoted from the literature, not checked here). For a turning sphere the correction is of
second order (the torque is odd in the rate), and for another body none is known: the watch says so."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mime.build._watch import WatchPlan

#: A sphere's neglected inertia in the drag, Oseen's first correction, by the Reynolds number of its translation.
OSEEN = 3.0 / 16.0
#: The hull kinds whose translation has a known first correction, with its coefficient.
FIRST_CORRECTION: Mapping[str, float] = {"sphere": OSEEN}

WITHIN, ABOVE, UNKNOWN, NO_RANGE = "within", "above", "not_computed", "no_range"

type Final = Mapping[str, Mapping[str, NDArray[Any]]]


@dataclass(frozen=True)
class WatchRead:
    """One watched entry in a run: its Reynolds numbers, what they were read from, the bound and the neglected
    inertia's size. ``status`` is ``above`` (raised), ``within``, ``no_range`` (the entry declares none to compare
    with) or ``not_computed`` (an input is missing: ``missing`` says which). ``peak`` is whether the speeds are the
    peaks over the whole run (a run that kept its history) or the last step's alone."""

    owner: str
    entry: str
    rung: str | None
    body: str
    medium: str
    status: str
    translation: float | None
    turning: float | None
    speed: float | None
    surface_speed: float | None
    diameter: float | None
    density: float | None
    viscosity: float | None
    upper: float | None
    basis: str | None
    peak: bool
    inertia: float | None
    inertia_text: str
    missing: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return f"reynolds:{self.owner}"

    @property
    def reynolds(self) -> float | None:
        """The larger of the two, which the bound is compared with."""
        held = [n for n in (self.translation, self.turning) if n is not None]
        return max(held) if held else None

    @property
    def raised(self) -> bool:
        return self.status == ABOVE

    def text(self) -> str:
        what = f"{self.path} ({self.entry}{f', rung {self.rung}' if self.rung else ''})"
        if self.status == UNKNOWN:
            return f"{what}: not computed, {'; '.join(self.missing)}"
        scope = "the peak over the run" if self.peak else "the last step only, which says nothing of the steps before it"
        numbers = (f"Reynolds number {self.reynolds:.3g} (translation {self.translation:.3g}, turning "
                   f"{self.turning:.3g}), from {scope}")
        if self.status == NO_RANGE:
            return f"{what}: {numbers}; the entry declares no range to compare it with. {self.inertia_text}"
        verdict = (f"ABOVE the {self.upper:.3g} the entry is declared up to: the answer leaves out inertia of the fluid that "
                   f"this flow has"
                   if self.status == ABOVE else f"within the {self.upper:.3g} the entry is declared up to")
        return f"{what}: {numbers}; {verdict}. {self.inertia_text}"

    def manifest(self) -> dict[str, Any]:
        return {"watch": self.path, "entry": self.entry, "owner": self.owner, "rung": self.rung, "body": self.body,
                "medium": self.medium, "status": self.status, "raised": self.raised, "reynolds": self.reynolds,
                "reynolds_translation": self.translation, "reynolds_turning": self.turning, "speed": self.speed,
                "surface_speed": self.surface_speed, "diameter": self.diameter, "density": self.density,
                "viscosity": self.viscosity, "upper": self.upper, "basis": self.basis, "peak_over_run": self.peak,
                "neglected_inertia": self.inertia, "neglected_inertia_text": self.inertia_text,
                "missing": list(self.missing), "text": self.text()}

    @classmethod
    def from_manifest(cls, held: Mapping[str, Any]) -> WatchRead:
        def number(name: str) -> float | None:
            value = held.get(name)
            return float(value) if value is not None else None

        return cls(str(held["owner"]), str(held["entry"]), cast(str | None, held.get("rung")), str(held["body"]),
                   str(held["medium"]), str(held["status"]), number("reynolds_translation"),
                   number("reynolds_turning"), number("speed"), number("surface_speed"), number("diameter"),
                   number("density"), number("viscosity"), number("upper"), cast(str | None, held.get("basis")),
                   bool(held.get("peak_over_run")), number("neglected_inertia"),
                   str(held.get("neglected_inertia_text", "")),
                   tuple(str(m) for m in cast(Sequence[object], held.get("missing", ()))))


def inertia_of(plan: WatchPlan, translation: float, turning: float) -> tuple[float | None, str]:
    """The size of the inertia the rung neglects, as a fraction of the drag, and what it rests on."""
    coefficient = FIRST_CORRECTION.get(plan.hull_kind or "")
    if coefficient is None:
        return None, (f"The size of the neglected inertia is not estimated: no correction is known for a "
                      f"{plan.hull_kind or 'body'} hull.")
    size = coefficient * translation
    text = (f"For a sphere's translation, Oseen's first correction raises the drag by 3 Re / 16 = {100.0 * size:.3g}% "
            f"of the Stokes drag at the translation's {translation:.3g}: the leading term of the expansion in Re, "
            f"good to a few per cent below Re of one and an overestimate above it.")
    if turning > 0.0:
        text += (f" The turning's correction is of second order in its {turning:.3g} (the torque is odd in the "
                 f"rate) and is not estimated.")
    return size, text


def speeds(node: str, field: str, final: Final, history: Final | None) -> tuple[float, float, bool]:
    """The body's speed and its angular speed (rad/s) from its twist: the peaks over the run's history, or the last
    step's where the run kept none."""
    if history is not None and node in history and field in history[node]:
        twist = np.asarray(history[node][field], dtype=np.float64).reshape(-1, 6)
        peak = True
    else:
        twist = np.asarray(final[node][field], dtype=np.float64).reshape(1, 6)
        peak = False
    return (float(np.linalg.norm(twist[:, :3], axis=1).max()), float(np.linalg.norm(twist[:, 3:], axis=1).max()),
            peak)


def read_watch(plan: WatchPlan, final: Final, history: Final | None) -> WatchRead:
    """``plan``'s watch in a run that ended in ``final`` and kept ``history`` (or none)."""
    try:
        speed, turning_rate, peak = speeds(plan.node, plan.field, final, history)
    except (KeyError, ValueError):
        return _unknown(plan, (f"the body's twist ({plan.node}.{plan.field}) is not in the run's state",))
    missing = list(plan.missing)
    if plan.viscosity is None:
        missing.append(f"{plan.medium}'s viscosity is not among the values the entry read")
    elif not plan.viscosity > 0.0:
        missing.append(f"{plan.medium}'s viscosity is not above zero")
    if not (math.isfinite(speed) and math.isfinite(turning_rate)):
        missing.append("the body's speed is not finite")
    if missing or plan.diameter is None or plan.density is None or plan.viscosity is None:
        return _unknown(plan, tuple(missing))
    scale = plan.density * plan.diameter / plan.viscosity
    surface = turning_rate * plan.diameter / 2.0
    translation, turning = scale * speed, scale * surface
    size, inertia_text = inertia_of(plan, translation, turning)
    status = NO_RANGE if plan.upper is None else ABOVE if max(translation, turning) > plan.upper else WITHIN
    return WatchRead(plan.owner, plan.entry, plan.rung, plan.body, plan.medium, status, translation, turning, speed,
                     surface, plan.diameter, plan.density, plan.viscosity, plan.upper, plan.basis, peak, size,
                     inertia_text)


def _unknown(plan: WatchPlan, missing: tuple[str, ...]) -> WatchRead:
    return WatchRead(plan.owner, plan.entry, plan.rung, plan.body, plan.medium, UNKNOWN, None, None, None, None,
                     plan.diameter, plan.density, plan.viscosity, plan.upper, plan.basis, False, None, "", missing)


def read_watches(plans: Sequence[WatchPlan], final: Final, history: Final | None) -> tuple[WatchRead, ...]:
    """Every watch of a run."""
    return tuple(read_watch(plan, final, history) for plan in plans)


def watches_text(reads: Sequence[WatchRead]) -> str:
    """The watches as the result prints them."""
    if not reads:
        return "post-run watches: none (no entry of this experiment neglects the fluid's inertia)"
    return "post-run watches:\n" + "\n".join(f"  {read.text()}" for read in reads)


__all__ = ["ABOVE", "FIRST_CORRECTION", "NO_RANGE", "OSEEN", "UNKNOWN", "WITHIN", "WatchRead", "inertia_of",
           "read_watch", "read_watches", "speeds", "watches_text"]
