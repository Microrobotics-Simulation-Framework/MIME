"""Domain metadata dataclasses for MIME.

All types in this module are pure Python dataclasses/enums with no JAX
dependency, so they can be imported without installing JAX or MADDENING's
simulation stack.

These metadata types compose into MimeNodeMeta, which every MimeNode
carries alongside MADDENING's NodeMeta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeRole(Enum):
    """Physical subsystem a MimeNode belongs to."""
    EXTERNAL_APPARATUS = "external_apparatus"
    ROBOT_BODY = "robot_body"
    ENVIRONMENT = "environment"
    SENSING = "sensing"
    THERAPEUTIC = "therapeutic"


class AnatomicalCompartment(Enum):
    """Physiological compartment the node operates in."""
    CSF = "csf"
    BLOOD = "blood"
    INTERSTITIAL = "interstitial"
    TISSUE = "tissue"


class FlowRegime(Enum):
    """Dominant flow regime."""
    PULSATILE_CSF = "pulsatile_csf"
    POISEUILLE = "poiseuille"
    STAGNANT = "stagnant"
    OSCILLATORY = "oscillatory"


class ActuationPrinciple(Enum):
    """Actuation mechanism."""
    ROTATING_MAGNETIC_FIELD = "rotating_magnetic_field"
    GRADIENT_MAGNETIC_FIELD = "gradient_magnetic_field"
    ACOUSTIC_STREAMING = "acoustic_streaming"
    OPTICAL_TRAPPING = "optical_trapping"


class ImagingModality(Enum):
    """Sensing/imaging modality."""
    MRI = "mri"
    FLUORESCENCE = "fluorescence"
    ULTRASOUND = "ultrasound"
    PHOTOACOUSTIC = "photoacoustic"
    MPI = "mpi"


class ReleaseKinetics(Enum):
    """Drug release mechanism."""
    PH_TRIGGERED = "ph_triggered"
    MAGNETIC = "magnetic"
    ENZYMATIC = "enzymatic"
    PASSIVE_DIFFUSION = "passive_diffusion"
    ACOUSTIC = "acoustic"


class BiocompatibilityClass(Enum):
    """ISO 10993 biocompatibility level."""
    NOT_ASSESSED = "not_assessed"
    SURFACE_CONTACT = "surface_contact"
    EXTERNALLY_COMMUNICATING = "externally_communicating"
    IMPLANT = "implant"


# ---------------------------------------------------------------------------
# Meta dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnatomicalRegimeMeta:
    """Physiological operating context for a MimeNode.

    Extends (does not replace) MADDENING's ValidatedRegime with
    anatomically grounded bounds.
    """
    compartment: AnatomicalCompartment
    anatomy: str = ""
    flow_regime: FlowRegime = FlowRegime.STAGNANT
    re_min: Optional[float] = None
    re_max: Optional[float] = None
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    viscosity_min_pa_s: Optional[float] = None
    viscosity_max_pa_s: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class BiocompatibilityMeta:
    """Materials and biocompatibility descriptor.

    Technical descriptor only — NOT a safety claim. The presence of
    this metadata does not constitute a biocompatibility assessment.
    The manufacturer must perform their own ISO 10993 evaluation.
    """
    materials: tuple[str, ...] = ()
    iso_10993_class: BiocompatibilityClass = BiocompatibilityClass.NOT_ASSESSED
    cytotoxicity_tested: bool = False
    haemocompatibility_tested: bool = False
    genotoxicity_tested: bool = False
    implantation_tested: bool = False
    biocompatibility_hazard_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActuationMeta:
    """Actuation system descriptor."""
    principle: ActuationPrinciple = ActuationPrinciple.ROTATING_MAGNETIC_FIELD
    is_onboard: bool = False
    max_field_strength_mt: Optional[float] = None
    max_frequency_hz: Optional[float] = None
    max_gradient_t_per_m: Optional[float] = None
    commandable_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensingMeta:
    """Sensing/imaging system descriptor."""
    modality: ImagingModality = ImagingModality.MRI
    spatial_resolution_mm: Optional[float] = None
    temporal_resolution_ms: Optional[float] = None
    snr: Optional[float] = None
    position_noise_std_mm: Optional[float] = None
    dropout_probability: Optional[float] = None
    imaging_artifact_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class TherapeuticMeta:
    """Drug delivery descriptor."""
    payload_type: str = ""
    payload_name: str = ""
    release_kinetics: ReleaseKinetics = ReleaseKinetics.PASSIVE_DIFFUSION
    target_anatomy: str = ""
    target_pathway: str = ""
    payload_capacity_ng: Optional[float] = None
    release_half_life_s: Optional[float] = None
    therapeutic_window_ratio: Optional[float] = None


@dataclass(frozen=True)
class MimeNodeMeta:
    """Top-level MIME domain metadata container.

    Composes all domain-specific metadata via composition (not inheritance).
    Every MimeNode carries both a NodeMeta (MADDENING-level) and a
    MimeNodeMeta (MIME-level). Compliance tooling consumes either layer
    independently.
    """
    role: NodeRole = NodeRole.ROBOT_BODY
    anatomical_regimes: tuple[AnatomicalRegimeMeta, ...] = ()
    biocompatibility: Optional[BiocompatibilityMeta] = None
    actuation: Optional[ActuationMeta] = None
    sensing: Optional[SensingMeta] = None
    therapeutic: Optional[TherapeuticMeta] = None
