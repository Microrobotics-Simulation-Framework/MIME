"""What the post-run watch needs from the plan: for each entry of the hydrodynamic family, the body it moves, the
medium it moves in, the hull's diameter, the medium's density and viscosity where the experiment gives them, the
Reynolds number its rung is declared up to, and where in the graph the body's twist is held. Plain data read after a
run (``mime.results.watch``); it adds no node and no edge, so the graph and its hash are what they would be without it."""

from __future__ import annotations

from dataclasses import dataclass

#: The family whose entries are watched: those whose model neglects the fluid's inertia.
WATCHED_FAMILY = "hydrodynamic"


@dataclass(frozen=True)
class WatchPlan:
    """One watched entry: its id, label and rung, the body and the medium, the node and field that hold the body's
    twist (linear velocity, then angular velocity), the hull's kind and diameter (twice its radial reach), the
    medium's density and viscosity (SI), the Reynolds number the rung is declared up to with its basis, and what is
    missing for a number to be computed (``missing``, one clause each). A number that is not computed says why."""

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

    def line(self) -> str:
        """The build report's line for it."""
        what = f"{self.entry}{f' (rung {self.rung})' if self.rung else ''} of {self.body} in {self.medium}"
        if self.upper is None:
            held = "no Reynolds number declared for it, so a run reports the number and compares it with nothing"
        else:
            held = f"compared with the {self.upper!r} its rung is declared up to"
        gaps = f"; no number can be computed until {'; '.join(self.missing)}" if self.missing else ""
        return f"{self.owner}: the Reynolds number after each run for {what}, {held}{gaps}"
