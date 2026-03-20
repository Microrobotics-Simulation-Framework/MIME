"""Tests for MimeAssetSchema and BenchmarkResult."""

import json
import pytest

from mime.schema.asset import BenchmarkResult, MimeAssetSchema
from mime.core.metadata import (
    ActuationMeta,
    ActuationPrinciple,
    AnatomicalCompartment,
    AnatomicalRegimeMeta,
    BiocompatibilityClass,
    BiocompatibilityMeta,
    FlowRegime,
)


# -- Fixtures --------------------------------------------------------------

def make_minimal_compliant() -> MimeAssetSchema:
    """Create a minimal asset that passes the compliance gate."""
    return MimeAssetSchema(
        asset_id="test-helical-001",
        robot_morphology="helical",
        characteristic_length_um=300.0,
        maddening_version_pin="0.1.0",
        onboard_node_classes=("MagneticResponseNode",),
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.CSF,
                anatomy="lateral_ventricle",
                flow_regime=FlowRegime.PULSATILE_CSF,
            ),
        ),
        biocompatibility=BiocompatibilityMeta(
            materials=("NdFeB", "SU-8"),
            iso_10993_class=BiocompatibilityClass.IMPLANT,
        ),
    )


def make_non_compliant() -> MimeAssetSchema:
    """Create an asset that fails the compliance gate."""
    return MimeAssetSchema()


# -- BenchmarkResult -------------------------------------------------------

class TestBenchmarkResult:
    def test_to_dict_roundtrip(self):
        br = BenchmarkResult(
            benchmark_id="B1",
            passed=True,
            metric_value=0.03,
            metric_threshold=0.05,
            mime_version="0.1.0",
        )
        d = br.to_dict()
        br2 = BenchmarkResult.from_dict(d)
        assert br2.benchmark_id == "B1"
        assert br2.passed is True
        assert br2.metric_value == 0.03
        assert br2.mime_version == "0.1.0"

    def test_tiered_benchmark_ids(self):
        """B4-T1, B4-T2, B4-T3 are valid benchmark IDs."""
        for bid in ["B0", "B1", "B2", "B3", "B4-T1", "B4-T2", "B4-T3", "B5"]:
            br = BenchmarkResult(benchmark_id=bid, passed=True)
            assert br.benchmark_id == bid


# -- Compliance gate -------------------------------------------------------

class TestComplianceGate:
    def test_minimal_compliant(self):
        asset = make_minimal_compliant()
        assert asset.mime_compliant is True
        assert asset.compliance_report() == []

    def test_empty_asset_not_compliant(self):
        asset = make_non_compliant()
        assert asset.mime_compliant is False
        issues = asset.compliance_report()
        assert len(issues) >= 5

    def test_missing_asset_id(self):
        asset = make_minimal_compliant()
        asset.asset_id = ""
        assert asset.mime_compliant is False
        assert any("asset_id" in i for i in asset.compliance_report())

    def test_missing_morphology(self):
        asset = make_minimal_compliant()
        asset.robot_morphology = ""
        assert asset.mime_compliant is False

    def test_zero_length(self):
        asset = make_minimal_compliant()
        asset.characteristic_length_um = 0.0
        assert asset.mime_compliant is False

    def test_negative_length(self):
        asset = make_minimal_compliant()
        asset.characteristic_length_um = -1.0
        assert asset.mime_compliant is False

    def test_no_onboard_nodes(self):
        asset = make_minimal_compliant()
        asset.onboard_node_classes = ()
        assert asset.mime_compliant is False

    def test_no_anatomical_regimes(self):
        asset = make_minimal_compliant()
        asset.anatomical_regimes = ()
        assert asset.mime_compliant is False

    def test_no_biocompatibility(self):
        asset = make_minimal_compliant()
        asset.biocompatibility = None
        assert asset.mime_compliant is False
        assert any("biocompatibility" in i for i in asset.compliance_report())

    def test_no_maddening_pin(self):
        asset = make_minimal_compliant()
        asset.maddening_version_pin = ""
        assert asset.mime_compliant is False


# -- Benchmark queries -----------------------------------------------------

class TestBenchmarkQueries:
    def test_benchmark_passed_true(self):
        asset = make_minimal_compliant()
        asset.benchmark_results = (
            BenchmarkResult(benchmark_id="B1", passed=True),
        )
        assert asset.benchmark_passed("B1") is True

    def test_benchmark_passed_false(self):
        asset = make_minimal_compliant()
        asset.benchmark_results = (
            BenchmarkResult(benchmark_id="B1", passed=False),
        )
        assert asset.benchmark_passed("B1") is False

    def test_benchmark_not_run(self):
        asset = make_minimal_compliant()
        assert asset.benchmark_passed("B1") is None

    def test_passed_benchmarks(self):
        asset = make_minimal_compliant()
        asset.benchmark_results = (
            BenchmarkResult(benchmark_id="B0", passed=True),
            BenchmarkResult(benchmark_id="B1", passed=True),
            BenchmarkResult(benchmark_id="B2", passed=False),
            BenchmarkResult(benchmark_id="B4-T1", passed=True),
        )
        assert asset.passed_benchmarks() == ["B0", "B1", "B4-T1"]

    def test_tiered_b4_queries(self):
        asset = make_minimal_compliant()
        asset.benchmark_results = (
            BenchmarkResult(benchmark_id="B4-T1", passed=True),
            BenchmarkResult(benchmark_id="B4-T2", passed=True),
            BenchmarkResult(benchmark_id="B4-T3", passed=False),
        )
        assert asset.benchmark_passed("B4-T1") is True
        assert asset.benchmark_passed("B4-T2") is True
        assert asset.benchmark_passed("B4-T3") is False


# -- JSON serialisation ----------------------------------------------------

class TestAssetSerialisation:
    def test_to_json_returns_string(self):
        asset = make_minimal_compliant()
        j = asset.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["asset_id"] == "test-helical-001"

    def test_roundtrip_minimal(self):
        asset = make_minimal_compliant()
        j = asset.to_json()
        asset2 = MimeAssetSchema.from_json(j)
        assert asset2.asset_id == asset.asset_id
        assert asset2.robot_morphology == asset.robot_morphology
        assert asset2.characteristic_length_um == asset.characteristic_length_um
        assert asset2.mime_compliant is True

    def test_roundtrip_with_benchmarks(self):
        asset = make_minimal_compliant()
        asset.benchmark_results = (
            BenchmarkResult(benchmark_id="B1", passed=True, metric_value=0.03),
            BenchmarkResult(benchmark_id="B2", passed=False, metric_value=0.08),
        )
        j = asset.to_json()
        asset2 = MimeAssetSchema.from_json(j)
        assert len(asset2.benchmark_results) == 2
        assert asset2.benchmark_results[0].benchmark_id == "B1"
        assert asset2.benchmark_results[0].passed is True
        assert asset2.benchmark_results[1].passed is False

    def test_roundtrip_with_metadata(self):
        asset = make_minimal_compliant()
        asset.actuation = ActuationMeta(
            principle=ActuationPrinciple.ROTATING_MAGNETIC_FIELD,
            is_onboard=False,
            max_frequency_hz=50.0,
            commandable_fields=("frequency_hz", "field_strength_mt"),
        )
        j = asset.to_json()
        asset2 = MimeAssetSchema.from_json(j)
        assert asset2.actuation is not None
        assert asset2.actuation.max_frequency_hz == 50.0

    def test_roundtrip_preserves_compliance(self):
        """Compliance status survives serialisation roundtrip."""
        asset = make_minimal_compliant()
        assert asset.mime_compliant is True
        j = asset.to_json()
        asset2 = MimeAssetSchema.from_json(j)
        assert asset2.mime_compliant is True

    def test_from_dict_ignores_unknown_fields(self):
        d = make_minimal_compliant().to_dict()
        d["unknown_future_field"] = "some_value"
        asset = MimeAssetSchema.from_dict(d)
        assert asset.mime_compliant is True

    def test_empty_roundtrip(self):
        asset = MimeAssetSchema()
        j = asset.to_json()
        asset2 = MimeAssetSchema.from_json(j)
        assert asset2.asset_id == ""
        assert asset2.mime_compliant is False

    def test_verification_modes_roundtrip(self):
        asset = make_minimal_compliant()
        asset.verification_modes = {
            "CSFFlowNode": "Mode 1 (Wrapping)",
            "RigidBodyNode": "Mode 2 (Independent)",
        }
        j = asset.to_json()
        asset2 = MimeAssetSchema.from_json(j)
        assert asset2.verification_modes["CSFFlowNode"] == "Mode 1 (Wrapping)"
