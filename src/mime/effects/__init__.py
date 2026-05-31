"""EffectModel contract — the v0.2 pilot (ADR-2026-EFFECT-MODEL).

A common builder pattern for environment effects that deliver force/torque
to a body. v0.2 ships the Protocol surface + registry + Experiment composition
with the **HydrodynamicModel** family piloted (LBM / FVM / Stokeslet /
DefectCorrection). The MagneticModel family, SourceInputProvider, and
cross-effect coupling implementation land in v0.3.

Public surface::

    from mime.effects import (
        EffectModel, SourcedEffectModel, EffectHandle, Source, CouplingSpec,
        Regime, HydrodynamicRegime, RegimeWarning,
        Body, Medium,
        Experiment, BenchmarkRef, CitationInfo,
        register_effect, list_registered_effects, get_effect,
        HydrodynamicModel,
    )
"""

from mime.effects.body_medium import Body, Medium
from mime.effects.errors import (
    BodyPropertyMissing,
    CouplingError,
    EffectBuildError,
    EffectModelError,
    IncompatibleMimeVersionError,
    MediumPropertyMissing,
    PortTypeMismatchError,
)
from mime.effects.experiment import (
    BenchmarkRef,
    CitationInfo,
    Experiment,
)
from mime.effects.hydrodynamic import HydrodynamicModel
from mime.effects.protocol import (
    BaseEffectModel,
    CouplingSpec,
    EffectHandle,
    EffectModel,
    HydrodynamicRegime,
    Regime,
    RegimeWarning,
    Source,
    SourcedEffectModel,
)
from mime.effects.registry import (
    get_effect,
    list_registered_effects,
    register_effect,
)

__all__ = [
    "Body",
    "Medium",
    "BodyPropertyMissing",
    "CouplingError",
    "EffectBuildError",
    "EffectModelError",
    "IncompatibleMimeVersionError",
    "MediumPropertyMissing",
    "PortTypeMismatchError",
    "BenchmarkRef",
    "CitationInfo",
    "Experiment",
    "HydrodynamicModel",
    "BaseEffectModel",
    "CouplingSpec",
    "EffectHandle",
    "EffectModel",
    "HydrodynamicRegime",
    "Regime",
    "RegimeWarning",
    "Source",
    "SourcedEffectModel",
    "get_effect",
    "list_registered_effects",
    "register_effect",
]
