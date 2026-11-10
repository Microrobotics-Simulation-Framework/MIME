"""Feature-graded sampling: one policy that turns a kind's declared features (``Geometry.features``: its edges and
crests) and, where a script opts in, a contact held facing a wall into a spacing over the body's surface, and that
spacing into the density a kind's sampling takes (``Geometry.sampling``'s ``density``; ``mime.scene.quasi_uniform``).

**The policy.** The local spacing at a point ``x`` of the surface is the least of each target's

    edge:     EDGE_SHARE s + GROWTH d(x)                 (s the sampling's spacing, d the distance to the edge)
    crest:    CREST_SHARE r + GROWTH d(x)                (r the crest's least radius across it)
    contact:  (gap + h(x)) / CONTACT_SPACINGS            (h the height above the contact's plane)

held between ``FINEST`` s and s, and the density is ``(s / spacing)^2``: neighbouring samples lie about the local
spacing apart. The contact is the body's farthest point along a direction the script names in the body's frame, the
side held facing the wall, and ``gap`` the gap to the wall it is graded for (the experiment's closest approach by
default); its spacing there keeps the contact ``CONTACT_SPACINGS`` local spacings from the wall at that gap.

**What it is for, and what it is not.** A graded sampling is fixed on the body, as every sampling is, so it serves a
feature or a contact that stays where it was graded: a crest or a rim, which is the contact whichever way the body
rolls about its axis, or a body whose orientation to the wall is held. A body that turns carries a graded contact away
from the wall within a fraction of a turn; the MRS drag's step flags it (``off_patch``) rather than resampling, which
would break the held matrix and the derivatives.

**The figures** are declarations from the measurements made before the build (a sphere at a plane wall, a cylinder's rim
at a plane and in free space, blobs 0.1 and 0.025), where graded sampling with each force point
at its cell's area centre held the sphere's normal resistance at half a force spacing from the wall to +5.2 % (blob
0.1) and +1.8 % (0.025) against +12.2 % and +7.2 % uniform, and the rim's to +1.4 % and -0.6 % against +2.6 % and
-0.5 %; the gates in ``tests/oracles/test_graded_sampling.py`` measure them on the build."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from functools import cached_property

import numpy as np

from mime.registry import FEATURE_FORMS, Feature, Geometry, implements
from mime.scene._sampling import FINEST, REFINEMENT, nearest, quadrature_arrays, quadrature_counts
from mime.units import Magnitude

#: An edge's spacing, as a share of the sampling's: the rim grading measured before the build, a third.
EDGE_SHARE = 1.0 / 3.0
#: A crest's spacing, as a share of its least radius across it: about 1.2 times the force points on FL-9 at script A's
#: 0.181 mm (1.6 times at 0.15).
CREST_SHARE = 0.3
#: How fast the spacing grows away from a feature, per unit of distance: the rim grading measured before the build, one.
GROWTH = 1.0
#: The contact's local spacings from the wall at the gap it is graded for: the contact grading measured before the build,
#: 1.5.
CONTACT_SPACINGS = 1.5
#: The features are traced at the finest spacing over this, so that a point's distance to one is read to a small share
#: of the spacing there.
TRACE = 8.0
#: The significant figures a graded sampling is chosen at (:func:`chosen_at`): the body's settings, the spacing and the
#: contact's gap and direction rounded to four, so that every body whose numbers round alike takes one sampling, moved
#: with its own shape, as a declared range's bodies take the range's. A graded choice is discrete (the farthest points,
#: Lloyd's cells, each cell's centred point) and a graded density is no scale of the body's, so a move of a setting far
#: below the spacing can tip a near-tie: measured before this rounding, under a relative 1e-6 move of one setting, two
#: of a box's 299 contact-graded points chosen afresh, four and eight of a helix's 770, and twenty of FL-9's 2227
#: feature-graded points at 0.4 mm. A move of a relative 1e-6 of a round value (three figures) stays in its rounding.
FIGURES = 4


@dataclass(frozen=True)
class Contact:
    """A contact held facing a wall: ``direction``, the side of the body that faces it, three numbers in the kind's own
    frame (its farthest point that way is the contact), and ``gap``, the gap to the wall it is graded for (m)."""

    direction: tuple[float, float, float]
    gap: float


@dataclass(frozen=True)
class GradedDensity:
    """The policy's density over ``kind``'s surface (the module's docstring): hashable, so that a sampling made with it
    is made once per process (``mime.scene.quasi_uniform``'s cache), and callable on points in the kind's frame."""

    kind: Geometry
    settings: tuple[tuple[str, Magnitude], ...]
    spacing: float
    features: bool
    contact: Contact | None

    def spacing_at(self, points: np.ndarray) -> np.ndarray:
        """The local spacing at ``points`` ``(n, 3)`` in the kind's frame (m)."""
        spacing = self.spacing
        held = np.full(len(points), spacing, dtype=np.float64)
        for feature in self.traced:
            own = EDGE_SHARE * spacing if feature.form == "edge" else CREST_SHARE * feature.size
            away = feature.distance(points) if feature.distance is not None else nearest(feature.points, points)[0]
            held = np.minimum(held, own + GROWTH * away)
        if self.contact is not None:
            held = np.minimum(held, (self.contact.gap + self._height(points)) / CONTACT_SPACINGS)
        return np.clip(held, FINEST * spacing, spacing)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return (self.spacing / self.spacing_at(np.asarray(points, dtype=np.float64))) ** 2

    @cached_property
    def traced(self) -> tuple[Feature, ...]:
        """The kind's features traced at the finest spacing over :data:`TRACE`; none where features are not graded."""
        if not self.features:
            return ()
        found = self.kind.features(dict(self.settings), FINEST * self.spacing / TRACE)
        for feature in found:
            if feature.form not in FEATURE_FORMS:
                raise ValueError(f"{self.kind.label()} declares a feature {feature.name!r} of form {feature.form!r}, "
                                 f"not one of {FEATURE_FORMS}")
        return found

    def _height(self, points: np.ndarray) -> np.ndarray:
        """Each point's height above the contact's plane, across it (m)."""
        direction, reach = self._plane
        return np.maximum(reach - points @ direction, 0.0)

    @cached_property
    def _plane(self) -> tuple[np.ndarray, float]:
        """The contact's plane: its unit direction and its reach along it, the body's farthest point that way over its
        quadrature at :data:`REFINEMENT` times finer than the spacing."""
        assert self.contact is not None
        direction = np.asarray(self.contact.direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        patches = tuple(self.kind.patches)
        dense = quadrature_arrays(self.kind, dict(self.settings), patches,
                                  quadrature_counts(self.kind, dict(self.settings), self.spacing / REFINEMENT,
                                                    patches))[0]
        return direction, float(np.max(dense @ direction))


def figures(value: float) -> float:
    """``value`` rounded to :data:`FIGURES` significant figures."""
    return float(f"{value:.{FIGURES}g}")


def chosen_at(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings a graded sampling is chosen at (:data:`FIGURES`): each number, or tuple of numbers, rounded; an
    integer (a count), a flag or a name as it is."""
    held: dict[str, Any] = {}
    for name, value in settings.items():
        if isinstance(value, float):
            held[name] = figures(value)
        elif isinstance(value, tuple) and all(isinstance(c, float) for c in cast(tuple[object, ...], value)):
            held[name] = tuple(figures(c) for c in cast(tuple[float, ...], value))
        else:
            held[name] = value
    return held


def graded_density(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, *, features: bool,
                   contact: Contact | None = None) -> GradedDensity:
    """The policy's density for ``kind`` at ``settings`` (SI) sampled at ``spacing`` (m): towards its declared features
    where ``features``, and towards ``contact`` where one is given."""
    if features and not implements(kind, "features"):
        raise ValueError(f"{kind.label()} declares no features to grade a sampling towards")
    return GradedDensity(kind, tuple(sorted(settings.items())), float(spacing), features, contact)


def finest_share(density: GradedDensity, points: np.ndarray) -> float:
    """The finest local spacing over ``points`` as a share of the sampling's."""
    return float(np.min(density.spacing_at(points))) / density.spacing


def graded_patch(density: GradedDensity, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Which of ``points`` (in the kind's frame) the grading refined: where the local spacing is below the sampling's.
    The patch a graded run's step judges its nearest point to the wall against (``off_patch``)."""
    held = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return density.spacing_at(held) < density.spacing * (1.0 - 1e-9)


__all__ = ["CONTACT_SPACINGS", "CREST_SHARE", "EDGE_SHARE", "FIGURES", "GROWTH", "TRACE", "Contact", "GradedDensity",
           "chosen_at", "figures", "finest_share", "graded_density", "graded_patch"]
