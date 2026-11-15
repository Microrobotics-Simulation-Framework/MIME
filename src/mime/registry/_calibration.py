"""Calibration types and instances (an apparatus entry's capability, ``calibration_instance``).

A **type** is a registered reader: the format it reads, by name and version (``mpem/1``, ``urdf/1``), the model family
its numbers belong to, the reader with its checks (the file's bytes in, the calibration out, or a ``ValueError`` saying
why), which of the format's fields a fit produces and which are declared, and how the file names the unit it describes.
It holds no settings and builds no node, so it is not an entry kind: a lab adds a reader for a format of its own with
:func:`register_calibration_type`, from its package.

An **instance** is one asset a type reads: its name and sha256, its tier (the verdict of its licence: ``open``,
``conditional``, ``referenced``, or ``carried`` for the author's own file under none), the unit it was fitted on as the file names it, and its fitted and declared fields.
An entry that reads one gives it (``calibration_instance``), and the build report prints it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mime.terms import Tier


def _no_unit(_: Any) -> str | None:
    return None


@dataclass(frozen=True)
class CalibrationType:
    """A registered reader of one calibration format: ``name`` and ``version`` (its label, ``mpem/1``), the model
    ``family`` its numbers are the parameters of, ``read`` (bytes to the calibration, raising a ``ValueError`` that says
    why a file is refused), the format's ``fitted`` and ``declared`` fields, and ``unit``, the unit a read calibration
    names as the one it describes (``None`` where it names none)."""

    name: str
    version: int
    family: str
    read: Callable[[bytes], Any]
    fitted: tuple[str, ...]
    declared: tuple[str, ...]
    unit: Callable[[Any], str | None] = _no_unit

    @property
    def label(self) -> str:
        return f"{self.name}/{self.version}"


@dataclass(frozen=True)
class CalibrationInstance:
    """One calibration asset as a type reads it: the type's label, the asset's name and sha256, its licence and tier,
    the unit it was fitted on as its file names it (``None``: the file names none), and the type's fitted and declared
    fields."""

    type: str
    asset: str
    sha256: str
    licence: str
    tier: Tier
    unit: str | None
    fitted: tuple[str, ...]
    declared: tuple[str, ...]

    def text(self) -> str:
        unit = f"the unit it describes, as the file names it: {self.unit}" if self.unit else \
            "the file names no unit it describes"
        fitted = ", ".join(self.fitted) or "none"
        declared = ", ".join(self.declared) or "none"
        return (f"calibration {self.asset} ({self.type}, sha256 {self.sha256[:12]}, {self.licence}, tier {self.tier}): "
                f"{unit}; fitted: {fitted}; declared, never fitted: {declared}")


_TYPES: dict[str, CalibrationType] = {}


def register_calibration_type(held: CalibrationType) -> CalibrationType:
    """Register ``held`` under its label; registering a label again with another reader raises ``ValueError``."""
    found = _TYPES.get(held.label)
    if found is not None and found != held:
        raise ValueError(f"the calibration type {held.label} is registered already, with another reader")
    _TYPES[held.label] = held
    return held


__all__ = ["CalibrationInstance", "CalibrationType", "register_calibration_type"]
