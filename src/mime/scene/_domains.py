"""Domain kinds: where a medium's solved field lives, as scene geometry, field-agnostic (F0's B6 and B7, decision 1).
Each field kind declares the domain kinds it supports, and the build refuses the others with the route.

Two kinds. The **periodic box** is the first: a medium's vessel held in a box periodic on every axis, its period along the vessel's
axis declared (``length``) and its other extents chosen by the field kind to hold the vessel with its images clear of
each other. A periodic box has no openings: its flow is driven by a body force on the fluid, a law bound to the medium
(``mime.interactions.BodyForce``), which stands in for the pressure drop a period of the vessel would carry.

The **vessel's region** is the second: the region of the vessel the medium fills, cut at the openings its geometry
kind declares for it (``Geometry.openings``: a straight tube's ``start`` and ``end`` along its declared axis; a tree's
root and branch ends), each opening taking the law attached to it (``mime.interactions.Inflow``, ``Windkessel``). It
reads the kind's geometry, so a vessel of another kind needs no domain class of its own. F2b builds its first form,
a straight vessel cut at its own two ends; the cut planes the region builder places come later."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from mime.registry import NUMERICAL, PHYSICAL, SettingGroup, convention, register_setting_group
from mime.units import Length, Quantity


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class PeriodicBox(SettingGroup):
    """A box periodic on every axis around the vessel the medium fills: ``length``, the period along the vessel's
    axis; ``margin``, the cells of solid kept between the vessel's band and its images, across the box."""

    group = "periodic_box"
    setting_roles = {"length": PHYSICAL,
                     "margin": convention(NUMERICAL, "eight cells of solid across the box between the vessel's images")}
    length: Annotated[Quantity, Length]
    margin: int = 8


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class VesselRegion(SettingGroup):
    """The region of the vessel the medium fills, cut at the openings its geometry kind declares for it (a straight
    tube's ``start`` and ``end``), each an opening whose law is attached there; ``margin``, the cells of solid kept
    across the box between the vessel's band and its images."""

    group = "vessel_region"
    setting_roles = {"margin": convention(NUMERICAL, "eight cells of solid across the box between the vessel's images")}
    margin: int = 8


__all__ = ["PeriodicBox", "VesselRegion"]
