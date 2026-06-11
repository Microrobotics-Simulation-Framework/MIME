"""Typed errors for the EffectModel contract (ADR-2026-EFFECT-MODEL §5).

Each ``Experiment.build()`` validation pass raises a specific error so the
failure mode is specifiable as an acceptance test rather than aspirational
prose. Version-compatibility validation (§7) raises
:class:`IncompatibleMimeVersionError` at load time.
"""

from __future__ import annotations


class EffectModelError(Exception):
    """Base class for all EffectModel-contract errors."""


class CouplingError(EffectModelError):
    """A CouplingSpec references an unknown effect or a missing port (pass 1/2)."""


class PortTypeMismatchError(EffectModelError):
    """A coupling's source EdgeSpec is structurally incompatible with the
    target port — shape / dtype / units (pass 2)."""


class BodyPropertyMissing(EffectModelError):
    """The body lacks a per-physics property section an attached effect
    declares as required (pass 3)."""


class MediumPropertyMissing(EffectModelError):
    """The medium lacks a property an attached effect declares as required
    (pass 3)."""


class EffectBuildError(EffectModelError):
    """An EffectModel's ``build()`` raised while materialising its subgraph
    (pass 5). Wraps the underlying exception."""


class IncompatibleMimeVersionError(EffectModelError):
    """The installed MIME version is outside an experiment's declared
    ``[mime_version_min, mime_version_max]`` window (load-time, before
    ``build()``)."""
