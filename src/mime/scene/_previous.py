"""The versions of the solid kinds kept registered after quasi-uniform sampling (7m-a) changed what they give the MRS
rung, so that a folder saved with one still loads and reproduces: the sphere 0.1.0 (the saved-form fixtures of
versions 1 and 2, ``tests/saved/previous/``, name it), the capsule 0.1.0 and the screw 0.2.0 (7j's saved experiments
of its frozen model name it). Each is its successor in every respect but one: its sampling at a spacing is its own
quadrature at that spacing, the tensor grid it gave before. A script that writes ``Sphere(...)`` gets the current
version; a load of a document that names an old one gets it from here. None is exported for a new experiment: their
samples crowd, which is why they were superseded, and the sampling conformance suite holds them to that (each fails
the spacing check, the control that the check can fail on a real kind)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import override

from mime.registry import Density, Geometry, Sampling, register_geometry, uniform
from mime.scene._sampling import quadrature
from mime.scene._screw import Screw
from mime.scene._solids import Capsule, Sphere
from mime.units import Magnitude


@register_geometry
@dataclass(frozen=True, kw_only=True)
class SphereGrid(Sphere):
    """The sphere 0.1.0: its sampling at a spacing is its quadrature, Gauss-Legendre in the cosine of the polar angle
    and even in azimuth, a quarter of whose samples near the poles lie closer than half the mean spacing
    (MIME-ANO-123)."""

    version = "0.1.0"

    @override
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quadrature(self, settings, spacing, patches)


@register_geometry
@dataclass(frozen=True, kw_only=True)
class CapsuleGrid(Capsule):
    """The capsule 0.1.0: its sampling at a spacing is its quadrature, its caps' polar grids crowded at their poles
    (MIME-ANO-123)."""

    version = "0.1.0"

    @override
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quadrature(self, settings, spacing, patches)


@register_geometry
@dataclass(frozen=True, kw_only=True)
class ScrewGrid(Screw):
    """The screw 0.2.0, 7j's frozen model: its sampling at a spacing is its quadrature, the lateral tensor grid and the
    caps in rings with no two samples closer than 0.3 spacings (MIME-ANO-119's fix)."""

    version = "0.2.0"

    @override
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quadrature(self, settings, spacing, patches)


#: Each kept version by its kind's name and version, with the version that superseded it.
KEPT: Mapping[tuple[str, str], tuple[type[Geometry], str]] = MappingProxyType({
    ("sphere", "0.1.0"): (SphereGrid, Sphere.version),
    ("capsule", "0.1.0"): (CapsuleGrid, Capsule.version),
    ("screw", "0.2.0"): (ScrewGrid, Screw.version),
})

__all__ = ["KEPT", "CapsuleGrid", "ScrewGrid", "SphereGrid"]
