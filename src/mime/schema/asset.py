"""MimeAssetSchema — the registry model card for a microrobot simulation.

Serves dual roles mirroring MICROBOTICA's dual nature:
1. Simulator artifact: everything MICROBOTICA needs to instantiate and run
2. Registry artifact: everything needed for community comparison

On disk: USD (Phase 4). In Phase 0-3: Python dataclass + JSON serialisation.

The compliance gate (mime_compliant property) determines whether an asset
can be published to the MICROBOTICA registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from mime.core.metadata import (
    ActuationMeta,
    AnatomicalRegimeMeta,
    BiocompatibilityMeta,
    SensingMeta,
    TherapeuticMeta,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark execution.

    Attached to MimeAssetSchema.benchmark_results. Carries full
    version information for reproducibility.
    """
    benchmark_id: str = ""           # B0, B1, B2, B3, B4-T1, B4-T2, B4-T3, B5
    passed: bool = False
    metric_value: float = 0.0        # The actual measured value
    metric_threshold: float = 0.0    # The pass criterion
    n_ensemble: int = 1              # For B4/B5 ensemble benchmarks
    mime_version: str = ""
    maddening_version: str = ""
    microbotica_version: str = ""
    execution_timestamp: str = ""    # ISO 8601
    hardware_description: str = ""   # GPU model, etc.

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "passed": self.passed,
            "metric_value": self.metric_value,
            "metric_threshold": self.metric_threshold,
            "n_ensemble": self.n_ensemble,
            "mime_version": self.mime_version,
            "maddening_version": self.maddening_version,
            "microbotica_version": self.microbotica_version,
            "execution_timestamp": self.execution_timestamp,
            "hardware_description": self.hardware_description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MimeAssetSchema:
    """Registry model card for a complete microrobot simulation configuration.

    The mime_compliant property and compliance_report() method form the
    quality gate for registry publication.
    """

    # -- Identity ----------------------------------------------------------
    asset_id: str = ""
    asset_version: str = "0.1.0"
    mime_schema_version: str = "0.1.0"
    maddening_version_pin: str = ""
    robot_morphology: str = ""           # e.g., "helical", "spherical", "flagellar"
    characteristic_length_um: float = 0.0

    # -- Functional composition --------------------------------------------
    onboard_node_classes: tuple[str, ...] = ()
    external_apparatus_node_classes: tuple[str, ...] = ()
    environment_node_classes: tuple[str, ...] = ()
    sensing_node_classes: tuple[str, ...] = ()
    therapeutic_node_classes: tuple[str, ...] = ()
    compatible_control_policies: tuple[str, ...] = ()
    asset_usd_path: str = ""             # Root USD file (Phase 4; JSON path in Phase 0-3)

    # -- Domain metadata ---------------------------------------------------
    biocompatibility: Optional[BiocompatibilityMeta] = None
    actuation: Optional[ActuationMeta] = None
    sensing: Optional[SensingMeta] = None
    therapeutic: Optional[TherapeuticMeta] = None

    # -- Validated context -------------------------------------------------
    anatomical_regimes: tuple[AnatomicalRegimeMeta, ...] = ()

    # -- Regulatory --------------------------------------------------------
    regulatory_class: str = ""           # EU MDR Class I-III
    soup_classification: str = ""        # IEC 62304 Class A/B/C

    # -- Benchmarks --------------------------------------------------------
    benchmark_results: tuple[BenchmarkResult, ...] = ()

    # -- Provenance --------------------------------------------------------
    authors: tuple[str, ...] = ()
    orcid_ids: tuple[str, ...] = ()
    zenodo_doi: str = ""
    license: str = "LGPL-3.0-or-later"
    description: str = ""

    # -- Verification mode per node ----------------------------------------
    verification_modes: dict[str, str] = field(default_factory=dict)
    # e.g., {"CSFFlowNode": "Mode 1 (Wrapping)", "RigidBodyNode": "Mode 2 (Independent)"}

    # -- Compliance gate ---------------------------------------------------

    @property
    def mime_compliant(self) -> bool:
        """Whether this asset meets minimum registry requirements."""
        return all([
            self.asset_id,
            self.robot_morphology,
            self.characteristic_length_um > 0,
            len(self.onboard_node_classes) > 0,
            len(self.anatomical_regimes) > 0,
            self.biocompatibility is not None,
            self.maddening_version_pin,
        ])

    def compliance_report(self) -> list[str]:
        """Return a list of compliance issues (empty = compliant)."""
        issues = []
        if not self.asset_id:
            issues.append("asset_id is empty")
        if not self.robot_morphology:
            issues.append("robot_morphology is empty")
        if self.characteristic_length_um <= 0:
            issues.append("characteristic_length_um must be > 0")
        if len(self.onboard_node_classes) == 0:
            issues.append("at least one onboard_node_classes entry required")
        if len(self.anatomical_regimes) == 0:
            issues.append("at least one anatomical_regimes entry required")
        if self.biocompatibility is None:
            issues.append(
                "biocompatibility must be set (describes materials, "
                "not a safety certification)"
            )
        if not self.maddening_version_pin:
            issues.append("maddening_version_pin is empty")
        return issues

    # -- Benchmark queries -------------------------------------------------

    def benchmark_passed(self, benchmark_id: str) -> bool | None:
        """Check if a specific benchmark passed. None if not run."""
        for r in self.benchmark_results:
            if r.benchmark_id == benchmark_id:
                return r.passed
        return None

    def passed_benchmarks(self) -> list[str]:
        """List of benchmark IDs that passed."""
        return [r.benchmark_id for r in self.benchmark_results if r.passed]

    # -- Serialisation (Phase 0-3: JSON) -----------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-compatible)."""
        d: dict[str, Any] = {}
        for f_name, f_def in self.__dataclass_fields__.items():
            val = getattr(self, f_name)
            if val is None:
                d[f_name] = None
            elif isinstance(val, tuple) and val and hasattr(val[0], '__dataclass_fields__'):
                d[f_name] = [_dataclass_to_dict(item) for item in val]
            elif isinstance(val, tuple):
                d[f_name] = list(val)
            elif hasattr(val, '__dataclass_fields__'):
                d[f_name] = _dataclass_to_dict(val)
            else:
                d[f_name] = val
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> MimeAssetSchema:
        """Reconstruct from a plain dict."""
        kwargs: dict[str, Any] = {}
        for f_name, f_def in cls.__dataclass_fields__.items():
            if f_name not in d:
                continue
            val = d[f_name]
            if val is None:
                kwargs[f_name] = None
            elif f_name == "benchmark_results" and isinstance(val, list):
                kwargs[f_name] = tuple(BenchmarkResult.from_dict(v) for v in val)
            elif f_name == "anatomical_regimes" and isinstance(val, list):
                kwargs[f_name] = tuple(
                    AnatomicalRegimeMeta(**_enum_restore(v, AnatomicalRegimeMeta))
                    for v in val
                )
            elif f_name == "biocompatibility" and isinstance(val, dict):
                kwargs[f_name] = BiocompatibilityMeta(
                    **_enum_restore(val, BiocompatibilityMeta)
                )
            elif f_name == "actuation" and isinstance(val, dict):
                kwargs[f_name] = ActuationMeta(**_enum_restore(val, ActuationMeta))
            elif f_name == "sensing" and isinstance(val, dict):
                kwargs[f_name] = SensingMeta(**_enum_restore(val, SensingMeta))
            elif f_name == "therapeutic" and isinstance(val, dict):
                kwargs[f_name] = TherapeuticMeta(**_enum_restore(val, TherapeuticMeta))
            elif isinstance(val, list):
                kwargs[f_name] = tuple(val)
            else:
                kwargs[f_name] = val
        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> MimeAssetSchema:
        """Reconstruct from a JSON string."""
        return cls.from_dict(json.loads(json_str))


# -- Helpers ---------------------------------------------------------------

def _dataclass_to_dict(obj: Any) -> dict:
    """Convert a dataclass (possibly with enums) to a plain dict."""
    from dataclasses import fields
    d = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if hasattr(val, 'value'):  # Enum
            d[f.name] = val.value
        elif isinstance(val, tuple):
            d[f.name] = list(val)
        else:
            d[f.name] = val
    return d


def _enum_restore(d: dict, cls: type) -> dict:
    """Restore enum values from strings for a frozen dataclass constructor.

    Only handles fields that exist on the dataclass and are tuples
    (converted to tuples from lists).
    """
    from dataclasses import fields as dc_fields
    result = {}
    field_map = {f.name: f for f in dc_fields(cls)}
    for k, v in d.items():
        if k not in field_map:
            continue
        f = field_map[k]
        # Convert lists back to tuples
        if isinstance(v, list):
            result[k] = tuple(v)
        # Try to restore enums
        elif isinstance(v, str) and hasattr(f.type, '__args__'):
            result[k] = v
        else:
            result[k] = v
    return result
