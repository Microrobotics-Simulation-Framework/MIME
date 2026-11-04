"""What the post-run watch needs from the plan: for each entry of the hydrodynamic family, the body it moves, the
medium it moves in, the hull's diameter, the medium's density and viscosity where the experiment gives them, the
Reynolds number its rung is declared up to, and where in the graph the body's twist is held. Plain data read after a
run (``mime.results.watch``); it adds no node and no edge, so the graph and its hash are what they would be without it.

It also holds what the build can say before any run: where the experiment declares the rate a field turns the body at
(a constant drive on the command the field turns at), the turning Reynolds number that rate gives, computed as the
watch computes it, and a warning when that lies past the number the rung is declared up to. The warning is in every
profile and is never a refusal."""

from __future__ import annotations

from dataclasses import dataclass

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
