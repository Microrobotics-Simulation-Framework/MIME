"""What the post-run watches need from the plan: for each entry of the hydrodynamic family, the body it moves, the
medium it moves in, the hull's diameter, the medium's density and viscosity where the experiment gives them, the
Reynolds number its rung is declared up to, and where in the graph the body's twist is held. Plain data read after a
run (``mime.results.watch``); it adds no node and no edge, so the graph and its hash are what they would be without it.

It also holds what the build can say before any run: where the experiment declares the rate a field turns the body at
(a constant drive on the command the field turns at), the turning Reynolds number that rate gives, computed as the
watch computes it, and a warning when that lies past the number the rung is declared up to. The warning is in every
profile and is never a refusal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mime.units import Magnitude

if TYPE_CHECKING:
    from mime.registry import Solid

#: The family whose entries are watched: those whose model neglects the fluid's inertia.
WATCHED_FAMILY = "hydrodynamic"


def turning_reynolds(density: float, diameter: float, viscosity: float, rate: float) -> float:
    """The Reynolds number of a hull of ``diameter`` turning at ``rate`` (rad/s) in a medium: by the speed of its surface
    at its radial reach, ``rho * rate * (D / 2) * D / mu``. The post-run watch and the warning before the run both
    compute it here."""
    scale = density * diameter / viscosity
    surface = rate * diameter / 2.0
    return scale * surface


@dataclass(frozen=True)
class WatchPlan:
    """One watched entry: its id, label and rung, the body and the medium, the node and field that hold the body's
    twist (linear velocity, then angular velocity), the hull's kind and diameter (twice its radial reach), the
    medium's density and viscosity (SI), the Reynolds number the rung is declared up to with its basis, and what is
    missing for a number to be computed (``missing``, one clause each). A number that is not computed says why.

    ``turning_rate`` is the rate (rad/s) at which the experiment's field turns a body that follows it, where a constant
    drive on the command the field turns at declares it, and ``turning_from`` says what it was computed from."""

    owner: str
    entry: str
    rung: str | None
    body: str
    medium: str
    node: str
    field: str
    hull_kind: str | None
    diameter: float | None
    density: float | None
    viscosity: float | None
    upper: float | None
    basis: str | None
    missing: tuple[str, ...] = ()
    turning_rate: float | None = None
    turning_from: str = ""

    def turning_reynolds(self) -> float | None:
        """The turning Reynolds number the declared rate gives, or ``None`` where the rate or an input is not known."""
        if self.turning_rate is None or self.diameter is None or self.density is None or self.viscosity is None or \
                not self.viscosity > 0.0:
            return None
        return turning_reynolds(self.density, self.diameter, self.viscosity, self.turning_rate)

    def range_warning(self) -> str | None:
        """The warning before the run: the operating point the experiment declares lies past the Reynolds number the
        rung is declared up to, with what it was computed from. ``None`` inside the range or where it cannot be computed."""
        number = self.turning_reynolds()
        if number is None or self.upper is None or not number > self.upper:
            return None
        assert self.density is not None and self.diameter is not None and self.viscosity is not None
        assert self.turning_rate is not None
        what = f"{self.entry}{f' (rung {self.rung})' if self.rung else ''} of {self.body} in {self.medium}"
        return (f"{self.owner}: {what} is declared up to Reynolds number {self.upper:g}, and the drive in this "
                f"experiment turns {self.body} at {self.turning_rate:.4g} rad/s ({self.turning_from}), which gives a "
                f"turning Reynolds number of {number:.3g}, {number / self.upper:.3g} times that. Computed at the "
                f"drive's rate as density {self.density:.6g} kg/m^3 x rate x (D / 2) x D / viscosity "
                f"{self.viscosity:.6g} Pa s, with D = {self.diameter:.6g} m the hull's diameter; a body held by contact "
                f"can turn faster than its drive, and the post-run watch reads the number the run reached. A warning, "
                f"not a refusal: the answer would leave out inertia of the fluid that this flow has")

    def line(self) -> str:
        """The build report's line for it."""
        what = f"{self.entry}{f' (rung {self.rung})' if self.rung else ''} of {self.body} in {self.medium}"
        if self.upper is None:
            held = "no Reynolds number declared for it, so a run reports the number and compares it with nothing"
        else:
            held = f"compared with the {self.upper!r} its rung is declared up to"
        gaps = f"; no number can be computed until {'; '.join(self.missing)}" if self.missing else ""
        before = ""
        if self.turning_rate is not None:
            number = self.turning_reynolds()
            before = (f"; before the run, the declared rate ({self.turning_from}) gives a turning Reynolds number of "
                      f"{number:.3g}" if number is not None else
                      f"; the declared rate ({self.turning_from}) gives no number before the run: an input is missing")
        return f"{self.owner}: the Reynolds number after each run for {what}, {held}{gaps}{before}"


#: The spacing of the points the containment watch compares with the surface, as a share of the hull's enclosing radius
#: (about 0.2 mm for FL-9, whose samples then fall 20 um short of the body and whose convex hull has about 500
#: vertices): coarser than the hull contact's, which must hold the body to its margin, since the watch reads a run
#: after it, at every step it kept, and its cost is the vertices times the steps; it quotes the shortfall.
WATCH_SPACING = 0.05
#: The hulls made in this process (``ContainmentPlan.made``), and the most kept.
_MADE: dict[tuple[object, ...], tuple[np.ndarray, float]] = {}
_MADE_MOST = 8


@dataclass(frozen=True)
class ContainmentPlan:
    """What the containment watch needs for a body the centre-held contact (0.1.0) holds, which has a hull it does not
    read: the contact's id and label, the body and the surface it is held against, the node and field that hold its
    pose, the surface's shape with its parameters in SI, the contact's margin, and the hull (its kind, settings and
    placement in the body, the spacing of its sample points). The body's hull is what the watch compares with the
    surface after the run; ``vertices`` are the hull's points the surface's signed distance is read at, in the body's
    frame: the vertices of the convex hull of the samples where the surface's region is convex (the largest signed
    distance over a set is at a vertex of its hull), else every sample. They are made when first read, not at the build.

    ``shortfall`` is how far the samples fall short of the body (m, ``mime.scene.samples_shortfall``), made when first
    read: the depth the watch reports is of the samples, so the body may pass the wall by up to that more."""

    owner: str
    entry: str
    body: str
    surface: str
    node: str
    field: str
    shape: str
    parameters: Mapping[str, Magnitude]
    margin: float
    hull_label: str
    kind: Solid
    settings: Mapping[str, Magnitude]
    centre: tuple[float, float, float]
    rotation: tuple[tuple[float, ...], ...]
    spacing: float
    convex: bool

    @property
    def made(self) -> tuple[np.ndarray, float]:
        """The hull's points to compare with the surface (``vertices``) and how far the samples fall short of the body
        (``shortfall``, m), made once per process for the same hull: a study builds the same body many times."""
        key = (self.kind.name, self.kind.version, tuple(sorted((n, repr(v)) for n, v in self.settings.items())),
               self.centre, self.rotation, self.spacing, self.convex)
        if key not in _MADE:
            from mime.nodes.body._rigid_inertial import hull_vertices
            from mime.scene._sampling import quadrature_arrays, quadrature_counts, samples_shortfall

            patches = tuple(self.kind.patches)
            counts = quadrature_counts(self.kind, self.settings, self.spacing, patches)
            points = quadrature_arrays(self.kind, self.settings, patches, counts)[0]
            turn = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
            placed = points @ turn.T + np.asarray(self.centre, dtype=np.float64)
            _MADE[key] = (placed[hull_vertices(placed)] if self.convex else placed,
                          float(samples_shortfall(self.kind, self.settings, self.spacing)))
            while len(_MADE) > _MADE_MOST:
                del _MADE[next(iter(_MADE))]
        return _MADE[key]

    @property
    def vertices(self) -> np.ndarray:
        """The hull's points to compare with the surface, in the body's frame, ``K x 3`` in float64."""
        return self.made[0]

    @property
    def shortfall(self) -> float:
        """How far the samples fall short of the body in any direction (m)."""
        return self.made[1]

    def range_warning(self) -> str | None:
        """None: a run's pose is what the watch reads, so there is nothing to warn before one."""
        return None

    def line(self) -> str:
        """The build report's line for it."""
        return (f"{self.owner}: after each run, {self.body}'s hull against {self.surface} ({self.entry} holds only its "
                f"centre by its extent, never reading its orientation): the deepest point of the hull past the "
                f"surface, at each step the run kept or at its last step; flagged, never a refusal")


__all__ = ["WATCH_SPACING", "WATCHED_FAMILY", "ContainmentPlan", "WatchPlan", "turning_reynolds"]
