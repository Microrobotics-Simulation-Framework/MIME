"""The flow's step rule (F0's item 1, decided 2026-11-08): the scheme from the viscous limit (RK3 first), the CFL
number predicted at build from what the build knows, a refusal before any step past the chosen scheme's line, and
``dt="auto"``, the largest step that is stable.

**The prediction** (the record's inputs): the body force's steady part as Poiseuille's centreline, ``g0 R^2 / (4 nu)``
(twice the mean), and each harmonic's peak from Womersley's profile (the prescribed flow's series), their magnitudes
summed, along the tube's axis (only that component drives a straight tube's flow); the bodies' speeds join it once
bodies couple to the flow. It is the empty tube's: a body in the lumen slows the flow it blocks (the tableau's sphere:
the face peak 2.85 against the empty tube's prediction of about 4) and can speed it locally where it does not.

**"Safe" is stability only** (the maintainer's question 1 is open): the step is the largest the scheme's lines allow;
the accuracy it buys is reported beside it, never acted on. Where two schemes are possible the larger safe step wins,
RK3 on a tie (B1's order); the alternative's step is shown as the choice's price (round 3, item 5)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mime.core import jnp
from mime.nodes.flow._womersley import profile, series_terms
from mime.registry import DriveQuery

#: The law whose waveform the prediction reads.
BODY_FORCE = "body_force"
#: How far past the signed distance's zero the band's effective wall may sit, in cells: the prediction takes the tube
#: this much wider, so a lumen the band widens does not outrun it (F1a's gate 3 holds the band's shift to 0.15 cells
#: against its planar calibration; dt="auto" at the line, on a tube whose wall sat 0.03 cells out, read its CFL 0.7 %
#: over the line and stopped, before this was added).
WALL_REACH_CELLS = 0.25
#: The harmonic's peak from rest over its periodic state's: a run starts from rest, and an inviscid core driven by
#: ``A cos(omega t + phi)`` from rest moves as ``(A / omega)(sin(omega t + phi) - sin phi)``, up to twice its periodic
#: peak (measured: at alpha 6 the start overshot the periodic peak by 8 %, and dt="auto" on the periodic peak alone
#: predicted 0.92 of the measured largest CFL).
START_FROM_REST = 2.0
#: The peak of ``|D(rho)|`` over the cross-section is taken on this many radii (the profile is smooth; at alpha 50 its
#: Stokes layer holds about 30 of them).
_RADII = 1001


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


def harmonic_peak(alpha: float) -> float:
    """``max_rho |D(rho)|``, the harmonic's peak speed over ``A R^2 / nu``: 1/4 at alpha 0 (quasi-steady Poiseuille),
    falling as the Stokes layer thins."""
    rho = np.linspace(0.0, 1.0, _RADII)
    re, im = profile(jnp.asarray(rho * rho), jnp.asarray(alpha * alpha), series_terms(np.float64))
    return float(np.max(np.hypot(np.asarray(re, float), np.asarray(im, float))))


def predicted_peak(drive: Drive, radius: float, nu: float) -> float:
    """The largest axial speed the drive gives an empty straight tube of ``radius`` (m/s) over a run from rest: the
    steady centreline (approached from below) plus the harmonic's periodic peak times :data:`START_FROM_REST`, in
    magnitude. The step rule passes the radius widened by :data:`WALL_REACH_CELLS`."""
    steady = abs(drive.steady) * radius ** 2 / (4.0 * nu)
    if not drive.amplitude:
        return steady
    alpha = radius * math.sqrt(2.0 * math.pi * drive.frequency / nu)
    return steady + START_FROM_REST * drive.amplitude * radius ** 2 / nu * harmonic_peak(alpha)


def steps_a_period(dt: float, period: float | None) -> float | None:
    return None if period is None else period / dt


def report_lines(scheme: str, viscous_number: float, viscous_line: float | None, cfl: float | None,
                 line: float | None, period: float | None, dt: float, price: Mapping[str, float]) -> tuple[str, ...]:
    """The build report's lines for the flow's step: the scheme and why, the predicted CFL and its margin, the steps a
    period against the measured 1 % lines, and what the other scheme's step would be."""
    lines = [f"the flow's scheme: {scheme}, at nu dt / h^2 = {viscous_number:.3g}"
             + (f" (RK3's viscous number {viscous_line:g})" if viscous_line is not None else "")]
    if cfl is not None and line is not None:
        lines.append(f"the predicted CFL max|u| dt / h = {cfl:.3g} against the {scheme} line of {line:.3g} along the "
                     f"vessel's axis (margin {100.0 * (1.0 - cfl / line):.0f} %); the empty tube's peak, a body's "
                     f"own speed not included")
    held = steps_a_period(dt, period)
    if held is not None:
        lines.append(f"{held:.0f} steps a period of the drive (stability only: RK3's drag history at M1's point was "
                     f"within 1 % from about 400 steps a period and 3.5 % at its line; IMEX Euler's phase error is "
                     f"about pi / steps)")
    for name, step in sorted(price.items()):
        if name != scheme:
            lines.append(f"the largest stable step with {name} would be {step:.4g} s")
    return tuple(lines)


__all__ = ["BODY_FORCE", "START_FROM_REST", "WALL_REACH_CELLS", "Drive", "axial_drive", "harmonic_peak", "predicted_peak", "report_lines", "steps_a_period"]
