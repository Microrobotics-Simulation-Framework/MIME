"""What a packaged surrogate was trained on: the declared data a build checks a body against, and that validity lines
will read."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class DeclaredRange:
    """A range of one quantity a surrogate is declared for, inclusive, with where its bounds come from."""

    quantity: str
    symbol: str
    low: float
    high: float
    basis: str

    def holds(self, value: float) -> bool:
        return self.low <= value <= self.high

    def describe(self) -> str:
        return f"{self.quantity} {self.symbol} from {self.low:g} to {self.high:g}"


@dataclass(frozen=True)
class ScrewSurrogate:
    """A resistance surrogate for de Jongh's screw-shaped bodies (de Jongh et al. 2025, Eq. 1): a cylinder of reference
    radius ``R``, its cross-section modulated as ``R (1 + modulation sin(starts theta))`` and turned along its axis by
    the normalised wavenumber ``nu``: the geometry kind ``family``, whose parameters a build reads. ``modulation`` and
    ``starts`` were fixed in training, so the body's outer radius is ``(1 + modulation) R``. ``shape`` holds the ranges
    of the body and tube a build checks before it builds:
    ``helix_wavenumber`` (``nu``), ``confinement`` (``kappa = R / R_tube``) and ``aspect`` (``L / R``). ``offset_fraction``, the body's distance
    from the tube's axis over the tube's radius, moves during a run and is not checked; the node keeps the 0.2 line's
    clamp of the offset to the wall instead, and reports when it fires."""

    asset: str
    shape: tuple[DeclaredRange, ...]
    offset_fraction: DeclaredRange
    modulation: float
    starts: int
    reference: str
    family: str = "dejongh_helix"

    def shape_range(self, quantity: str) -> DeclaredRange:
        (found,) = (r for r in self.shape if r.quantity == quantity)
        return found


#: Read from the training data of these weights (the 0.2 line's ``swimming_speeds_*.json`` under
#: ``data/dejongh_benchmark``, all but the held-out ``lhs`` set, which reproduce the stored feature standard deviations
#: to float32 rounding): nu from 0.25 to 3.5, kappa 0 (free space) and from 0.12 to 0.655 at seven values, offset
#: fractions up to 0.35, lengths from 4.13 to 9.73 mm (L / R from 2.65 to 6.24). The weights file itself records only
#: the features' means and standard deviations. What the ranges bound is the training data, not the surrogate's
#: accuracy inside them, which MIME-ANO-114 records against the BEM it was checked against.
DEJONGH_MLP_V2 = ScrewSurrogate(
    asset="dejongh_mlp_v2",
    shape=(
        DeclaredRange("helix_wavenumber", "nu", 0.25, 3.5,
                      "the training data's range (de Jongh's FL designs 1 to 11 and the fills between them); the 0.2 "
                      "line declared 2.33 to 3.7, which is not the training data's"),
        DeclaredRange("confinement", "kappa", 0.25, 0.655,
                      "the training data hold kappa at 0 (free space), 0.12, 0.246, 0.327, 0.4, 0.491 and 0.655: the "
                      "upper bound is the largest trained value (the 0.2 line declared 0.66, above it)"),
        DeclaredRange("aspect", "L / R", 2.65, 6.24,
                      "the training data's lengths, 4.13 to 9.73 mm, over the reference radius: the surrogate reads "
                      "the length only as L / R, its second feature (the 0.2 line declared no range)"),
    ),
    offset_fraction=DeclaredRange(
        "offset_fraction", "offset / R_tube", 0.0, 0.3,
        "as the 0.2 line declares it, with its reason now: the training data hold the operating geometries only at "
        "offsets 0, 0.024, 0.1, 0.2 and 0.3 of the tube's radius, and past 0.3 the BEM itself fails near the wall "
        "(MIME-ANO-114); some designs reach 0.35"),
    modulation=0.33,
    starts=2,
    reference="de Jongh et al. (2025), Nonlinear Dyn. 113:29197-29213, Eq. 1 and Tables 1 and 2",
)

#: Every packaged surrogate, by the name of the asset holding its weights.
SURROGATES: Mapping[str, ScrewSurrogate] = MappingProxyType({DEJONGH_MLP_V2.asset: DEJONGH_MLP_V2})
