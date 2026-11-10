"""The resolution report, its walls' half: how many cells a grid of a given spacing puts across each feature of a wall
that limits what a smooth band resolves, each beside its line where one is measured or estimated. Below a line the
route is a refinement patch for the wall (its fine cell from the report), never a silent result; an entry with no
line is reported as unmeasured. The bodies' half comes with the phase that resolves bodies on a grid.

Entries, each in cells of the spacing:

- **the smallest wall radius**: the tightest curvature of the wall in the band, a sharp seam's zero (a fillet's radius
  where the wall is filleted);
- **the narrowest lumen**: the smallest diameter, against the band's thin-feature estimate (a band two cells wide on
  each side over-conducts a gap narrower than about five cells);
- **the narrowest solid**: the least solid between two tubes that share no point, and the sharpest wedge of solid
  between two tubes leaving a junction, as the length from its apex over which it is thinner than the band (a feature
  thinner than the band does not become fully solid inside it);
- **the solid the fillet fills** between tubes that share no point, where there is one: a wall thinner than the
  fillet's diameter, which its closing removes, joining the tubes (``WallFeatures.fused``);
- **the wall layer**, given a flow: the Stokes layer ``R sqrt(2) / alpha`` in pulsatile flow, ``D / sqrt(Re)`` in steady
  flow, against the lines a Womersley sweep measured for the wall shear read along the normal from the calibrated wall
  (alpha 6 to 20 at 16 to 64 cells a diameter, two sub-cell offsets): within 5 % from about 2.7 cells of layer,
  interpolated on the worst arm between its rows at 2.26 cells (6.5 %) and 3.23 (3.2 %); within 2 % from about 4.5,
  between 3.77 (2.7 %) and 4.53 (1.95 %). The rows are kept with the tests that hold the lines to them."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mime.scene._tube_tree import WallFeatures

#: The wall layer's lines, in cells of layer: the wall shear within 5 % and within 2 % (interpolated on a sweep's worst
#: arm, as the docstring says).
LAYER_LINES = {"wall shear within 5 %": 2.7, "wall shear within 2 %": 4.5}
#: The narrowest lumen's line, in cells: the band's thin-feature estimate.
LUMEN_LINE = 5.0
#: The registered band's width in cells (the default band, two cells), which the wedge's entry is read against.
BAND_WIDTH = 2.0


@dataclass(frozen=True)
class ResolutionEntry:
    """One feature: its ``name``, its size in ``cells``, its ``line`` in cells (``None``: unmeasured) and the line's
    ``basis``."""

    name: str
    cells: float
    line: float | None
    basis: str

    @property
    def below(self) -> bool | None:
        """Whether it lies below its line (``None`` where it has none)."""
        return None if self.line is None else self.cells < self.line

    def text(self) -> str:
        if self.line is None:
            return f"{self.name}: {self.cells:.3g} cells ({self.basis or 'no line measured yet'})"
        verdict = "below its line: refine the wall here" if self.below else "at or above its line"
        return f"{self.name}: {self.cells:.3g} cells, {verdict} ({self.line:.3g}; {self.basis})"


@dataclass(frozen=True)
class ResolutionReport:
    """The walls' resolution at a grid spacing: its entries, and their lines of text for a build's report."""

    spacing: float
    entries: tuple[ResolutionEntry, ...]

    def lines(self) -> tuple[str, ...]:
        return tuple(entry.text() for entry in self.entries)

    def entry(self, name: str) -> ResolutionEntry:
        for held in self.entries:
            if held.name == name:
                return held
        raise KeyError(f"no entry {name!r}; the report's entries are {[e.name for e in self.entries]}")


def stokes_layer(radius: float, womersley: float) -> float:
    """The Stokes layer's thickness in pulsatile flow, ``R sqrt(2) / alpha`` (m), ``alpha`` the Womersley number at
    the radius ``R``."""
    return radius * math.sqrt(2.0) / womersley


def steady_layer(diameter: float, reynolds: float) -> float:
    """The wall layer's thickness in steady flow, ``D / sqrt(Re)`` (m), ``Re`` the Reynolds number on the diameter."""
    return diameter / math.sqrt(reynolds)


def wall_resolution(features: WallFeatures, spacing: float, *, layer: float | None = None,
                    band_width: float = BAND_WIDTH) -> ResolutionReport:
    """The report for a wall with ``features`` on a grid of ``spacing`` (m), with the wall ``layer`` (m) where a flow
    gives one (:func:`stokes_layer`, :func:`steady_layer`)."""
    if not (math.isfinite(spacing) and spacing > 0.0):
        raise ValueError(f"a grid's spacing is a length above zero, not {spacing!r}")
    half_angle = math.radians(features.sharpest_wedge) / 2.0
    wedge_cells = band_width / (2.0 * math.tan(half_angle)) if half_angle < math.pi / 2.0 else 0.0
    entries = [
        ResolutionEntry("smallest wall radius", features.smallest_radius / spacing, None, ""),
        ResolutionEntry("narrowest lumen", features.narrowest_lumen / spacing, LUMEN_LINE,
                        "an estimate: a band of two cells each side over-conducts a gap narrower than about five"),
        ResolutionEntry("narrowest solid between tubes that share no point", features.narrowest_solid / spacing, None,
                        ""),
        ResolutionEntry(f"sharpest wedge ({features.sharpest_wedge:.3g} degrees), thinner than the band from its apex",
                        wedge_cells, None, ""),
    ]
    if features.fused:
        entries.append(ResolutionEntry(
            f"solid the fillet fills between tubes that share no point {list(features.fused)}",
            features.narrowest_solid / spacing, None,
            "the fillet's closing joins these tubes; within its radius of this wall the solid's distance may read too "
            "near the fluid"))
    if layer is not None:
        cells = layer / spacing
        for name, line in LAYER_LINES.items():
            entries.append(ResolutionEntry(f"wall layer, for {name}", cells, line,
                                           "a Womersley sweep's worst arm, alpha 6 to 20, 16 to 64 cells a diameter"))
    return ResolutionReport(spacing, tuple(entries))


__all__ = ["BAND_WIDTH", "LAYER_LINES", "LUMEN_LINE", "ResolutionEntry", "ResolutionReport", "steady_layer",
           "stokes_layer", "wall_resolution"]
