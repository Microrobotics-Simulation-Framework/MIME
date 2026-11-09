"""The flow's step rule (F0's item 1, decided 2026-11-08): the scheme from the viscous limit (RK3 first), the CFL
number predicted at build from what the build knows, a refusal before any step past the chosen scheme's line, and
``dt="auto"``, the largest step that is stable.

**The prediction** (the record's inputs): the body force's steady part as Poiseuille's centreline, ``g0 R^2 / (4 nu)``
(twice the mean), and each harmonic's peak from Womersley's profile (the prescribed flow's series), their magnitudes
summed, along the tube's axis (only that component drives a straight tube's flow); the bodies' speeds join it once
bodies couple to the flow. It is the empty tube's: a body in the lumen slows the flow it blocks (the tableau's sphere:
the face peak 2.85 against the empty tube's prediction of about 4) and can speed it locally where it does not.

**The harmonic's peak** is summed on the host in float64 whatever the run's dtype (build-time arithmetic, before any
step), from the series to its float64 line (``SERIES_LINE``, the prescribed flow's); past the line, or where the series
is not finite, the line's peak scaled by ``(line / alpha)^2``. That stays a bound: from rest the harmonic's speed is at
most ``2 A / omega`` at any alpha [derived: with ``c = (A / omega) sin(omega t + phi)`` the core's swing, ``u - c`` is
a diffusion whose wall value is ``-c``, so the maximum principle holds it to ``A / omega``], and twice the line's
peak scaled is ``2.14 A / omega``; the periodic peak's ``alpha^2 max|D|`` falls from 1.0719 at alpha 50 to 1.0704 at
120 [measured: SciPy], so the scaled value is above it too. A prediction that is not finite is refused at any step.

**"Safe" is stability only** (the maintainer's decision): the step is the largest the scheme's lines allow;
the accuracy it buys is reported beside it, never acted on. Where two schemes are possible the larger safe step wins,
RK3 on a tie (B1's order); the alternative's step is shown as the choice's price (round 3, item 5)."""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mime.nodes.flow._womersley import VALIDITY, host_profile, series_terms
from mime.registry import DriveQuery

#: The law whose waveform the prediction reads.
BODY_FORCE = "body_force"
#: How far past the signed distance's zero the band's effective wall may sit, in cells: the prediction takes the tube
#: this much wider, so a lumen the band widens does not outrun it (F1a's gate 3 holds the band's shift to 0.15 cells
#: against its planar calibration; dt="auto" at the line, on a tube whose wall sat 0.03 cells out, read its CFL 0.7 %
#: over the line and stopped, before this was added).
WALL_REACH_CELLS = 0.25
#: The harmonic's peak from rest over its periodic state's. The axial flow in a straight tube is a diffusion with the
#: wall at zero, so the run from rest is the periodic flow plus a transient that starts as minus the periodic flow at
#: the start and, by the maximum principle, never exceeds it: the speed stays under the steady centreline plus twice
#: the harmonic's periodic peak, at any start. Measured at D/h 16 from rest at the worst phase (the drive starting at
#: zero), the periodic peak alone predicted 0.54 to 0.56 of the largest speed (alpha 6 to 20); doubled, 1.08 to 1.13.
START_FROM_REST = 2.0
#: The peak of ``|D(rho)|`` over the cross-section is taken on this many radii (the profile is smooth; at alpha 50 its
#: Stokes layer holds about 30 of them).
_RADII = 1001
#: The largest Womersley number the prediction reads the series at: float64's line (the series summed on the host in
#: float64 whatever the run's dtype; in float32 it read -1.8 % at alpha 60, +166 % at 65, zero at 70 and NaN from 75).
SERIES_LINE = VALIDITY["float64"][0]


@dataclass(frozen=True)
class Drive:
    """The medium's drive along the tube's axis: its steady part, one harmonic's amplitude and frequency (Hz)."""

    steady: float
    amplitude: float
    frequency: float

    @property
    def period(self) -> float | None:
        return 1.0 / self.frequency if self.frequency > 0.0 and self.amplitude != 0.0 else None


def axial_drive(drives: Sequence[DriveQuery], axis: Sequence[float]) -> Drive:
    """The body forces' parts along ``axis`` (a unit vector), summed; the harmonic's frequency the first one's."""
    steady = amplitude = frequency = 0.0
    for drive in drives:
        if drive.law != BODY_FORCE:
            continue
        s = drive.settings
        along = np.asarray(s["along"], float)
        component = float(np.dot(along / np.linalg.norm(along), np.asarray(axis, float)))
        steady += component * float(s["steady"])  # type: ignore[arg-type]
        harmonic = float(s.get("amplitude") or 0.0)  # type: ignore[arg-type]
        if harmonic:
            amplitude += abs(component * harmonic)
            frequency = frequency or float(s.get("frequency") or 0.0)  # type: ignore[arg-type]
    return Drive(steady, amplitude, frequency)


def series_peak(alpha: float) -> float:
    """``max_rho |D(rho)|`` from the series, summed on the host in float64."""
    rho = np.linspace(0.0, 1.0, _RADII)
    return float(np.max(np.abs(host_profile(rho * rho, alpha * alpha, series_terms(np.float64)))))


@functools.cache
def _line_peak() -> float:
    return series_peak(SERIES_LINE)


def series_holds(alpha: float) -> bool:
    """Whether the harmonic's peak at ``alpha`` is the series' own: within its line and finite (a NaN is past it)."""
    return bool(alpha <= SERIES_LINE) and math.isfinite(series_peak(alpha))


def harmonic_peak(alpha: float) -> float:
    """``max_rho |D(rho)|``, the harmonic's periodic peak speed over ``A R^2 / nu``: 1/4 at alpha 0 (quasi-steady
    Poiseuille), falling as the Stokes layer thins; past the series' line (or where it is not finite) the line's peak
    times ``(line / alpha)^2``, which the module shows stays a bound."""
    return series_peak(alpha) if series_holds(alpha) else _line_peak() * (SERIES_LINE / alpha) ** 2


def womersley_number(drive: Drive, radius: float, nu: float) -> float:
    return radius * math.sqrt(2.0 * math.pi * abs(drive.frequency) / nu)


def predicted_peak(drive: Drive, radius: float, nu: float) -> float:
    """The largest axial speed the drive gives an empty straight tube of ``radius`` (m/s) over a run from rest: the
    steady centreline (approached from below) plus the harmonic's periodic peak times :data:`START_FROM_REST`, in
    magnitude. The step rule passes the radius widened by :data:`WALL_REACH_CELLS`."""
    steady = abs(drive.steady) * radius ** 2 / (4.0 * nu)
    if not drive.amplitude:
        return steady
    return steady + START_FROM_REST * abs(drive.amplitude) * radius ** 2 / nu * harmonic_peak(
        womersley_number(drive, radius, nu))


def steps_a_period(dt: float, period: float | None) -> float | None:
    return None if period is None else period / dt


def report_lines(scheme: str, step: float, every: int, viscous_number: float, viscous_line: float | None,
                 cfl: float | None, line: float | None, period: float | None, price: Mapping[str, float],
                 alpha: float | None = None) -> tuple[str, ...]:
    """The build report's lines for the flow's step (``every`` base steps, ``rate=``, where it is more than one): the
    scheme and why, the predicted CFL and its margin, the series' line where the drive is past it, the steps a period,
    and what the other scheme's step would be."""
    lines = [f"the flow's step: {step:.6g} s" + (f", every {every} base steps (its rate=)" if every > 1 else ""),
             f"the flow's scheme: {scheme}, at nu dt / h^2 = {viscous_number:.3g} at the flow's step"
             + (f" (RK3's viscous number {viscous_line:g})" if viscous_line is not None else "")]
    if cfl is not None and line is not None:
        lines.append(f"the predicted CFL max|u| dt / h = {cfl:.3g} at the flow's step against the {scheme} line of "
                     f"{line:.3g} along the vessel's axis (margin {100.0 * (1.0 - cfl / line):.0f} %); the empty "
                     f"tube's peak, a body's own speed not included")
    if alpha is not None and not series_holds(alpha):
        lines.append(f"the drive's Womersley number {alpha:.4g} is past the series' line of {SERIES_LINE:g}: the "
                     f"harmonic's peak is the line's scaled by (line / alpha)^2, above the maximum principle's bound "
                     f"from rest (2 A / omega)")
    held = steps_a_period(step, period)
    if held is not None:
        lines.append(f"{held:.0f} steps a period of the drive (the step is chosen for stability only, its accuracy "
                     f"not bounded: a penalty sphere's drag history in a pulsatile tube flow, measured with RK3, was "
                     f"within 1 % from about 400 steps a period; IMEX Euler's phase error is about pi / steps)")
    for name, other in sorted(price.items()):
        if name != scheme:
            lines.append(f"the largest stable step with {name} would be {other:.4g} s")
    return tuple(lines)


__all__ = ["BODY_FORCE", "SERIES_LINE", "START_FROM_REST", "WALL_REACH_CELLS", "Drive", "axial_drive", "harmonic_peak",
           "predicted_peak", "report_lines", "series_holds", "series_peak", "steps_a_period", "womersley_number"]
