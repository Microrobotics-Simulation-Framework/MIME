"""B0–B5 benchmark definitions.

Each benchmark is a dataclass describing:
- What it tests
- What the pass criterion is
- What nodes/infrastructure it depends on
- Its current implementation status

The BenchmarkSuite class discovers all registered benchmarks and
can run them (once they are implemented beyond stubs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class BenchmarkStatus(Enum):
    NOT_IMPLEMENTED = "not_implemented"
    IMPLEMENTED = "implemented"
    PASSING = "passing"
    FAILING = "failing"


@dataclass(frozen=True)
class BenchmarkDefinition:
    """Definition of a single MIME benchmark."""
    benchmark_id: str
    name: str
    pass_criterion: str
    phase: str
    dependencies: tuple[str, ...] = ()
    status: BenchmarkStatus = BenchmarkStatus.NOT_IMPLEMENTED
    run_fn: Optional[Callable] = None


# -- Benchmark registry ----------------------------------------------------

_REGISTRY: dict[str, BenchmarkDefinition] = {}


def register_benchmark(defn: BenchmarkDefinition) -> BenchmarkDefinition:
    """Register a benchmark definition."""
    _REGISTRY[defn.benchmark_id] = defn
    return defn


def get_benchmark(benchmark_id: str) -> Optional[BenchmarkDefinition]:
    """Look up a benchmark by ID."""
    return _REGISTRY.get(benchmark_id)


def all_benchmarks() -> dict[str, BenchmarkDefinition]:
    """Return all registered benchmarks."""
    return dict(_REGISTRY)


# -- B0–B5 definitions ----------------------------------------------------

B0 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B0",
    name="Experimental validation",
    pass_criterion=(
        "Simulated helical robot trajectory in a straight cylindrical channel "
        "matches a published experimental dataset (position RMSE < 15% of "
        "channel diameter, velocity RMSE < 20% of mean velocity)"
    ),
    phase="Phase 1",
    dependencies=(
        "ExternalMagneticFieldNode", "MagneticResponseNode",
        "RigidBodyNode", "CSFFlowNode",
        "published experimental dataset (Rodenborn et al. 2013)",
    ),
))

B1 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B1",
    name="Step-out frequency detection",
    pass_criterion=(
        "Simulated step-out frequency within +/-5% of a regularised Stokeslet "
        "reference solution computed for the specific robot geometry, validated "
        "against Rodenborn et al. (2013, PNAS) experimental data"
    ),
    phase="Phase 1",
    dependencies=(
        "ExternalMagneticFieldNode", "MagneticResponseNode",
        "RigidBodyNode", "CSFFlowNode", "PhaseTrackingNode",
        "regularised Stokeslet reference solution",
    ),
))

B2 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B2",
    name="Stokes drag in CSF",
    pass_criterion=(
        "Drag force < 5% relative error vs. Stokes law F=6*pi*eta*r*v "
        "at Re < 0.1"
    ),
    phase="Phase 1",
    dependencies=("RigidBodyNode", "CSFFlowNode"),
))

B3 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B3",
    name="Drug release kinetics",
    pass_criterion=(
        "L2 error < 10% vs. analytical diffusion equation solution"
    ),
    phase="Phase 2",
    dependencies=("DrugReleaseNode", "ConcentrationDiffusionNode"),
))

B4_T1 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B4-T1",
    name="Closed-loop navigation — simple geometry",
    pass_criterion=(
        ">= 80% of N=32 ensemble runs reach within 2mm of target "
        "in a straight cylindrical channel (D=2mm, L=50mm)"
    ),
    phase="Phase 3",
    dependencies=(
        "PolicyRunner", "LocalisationUncertainty",
        "parametric cylindrical channel geometry",
        "full magnetic actuation chain",
    ),
))

B4_T2 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B4-T2",
    name="Closed-loop navigation — realistic anatomy",
    pass_criterion=(
        ">= 80% of N=32 ensemble runs reach within 2mm of target "
        "in a Neurobotika-derived ventricular mesh"
    ),
    phase="Phase 3",
    dependencies=("B4-T1 passing", "Neurobotika-derived ventricular mesh"),
))

B4_T3 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B4-T3",
    name="Closed-loop navigation — pathological anatomy",
    pass_criterion=(
        ">= 70% of N=32 ensemble runs reach within 2mm of target "
        "in a pathological anatomy variant (stenosed aqueduct, hydrocephalus)"
    ),
    phase="Advanced",
    dependencies=(
        "B4-T2 passing",
        "pathological anatomy mesh variants",
    ),
))

B5 = register_benchmark(BenchmarkDefinition(
    benchmark_id="B5",
    name="Step-out recovery under actuation uncertainty",
    pass_criterion=(
        "Recovery within 5s for >= 90% of N=32 ensemble runs"
    ),
    phase="Phase 3",
    dependencies=(
        "PhaseTrackingNode", "ActuationUncertainty",
        "StepOutDetector feedback policy",
    ),
))


# -- BenchmarkSuite --------------------------------------------------------

class BenchmarkSuite:
    """Discovers and runs registered benchmarks."""

    def list_benchmarks(self) -> list[BenchmarkDefinition]:
        """Return all registered benchmark definitions, ordered by ID."""
        return sorted(_REGISTRY.values(), key=lambda b: b.benchmark_id)

    def run(self, benchmark_id: str) -> dict:
        """Run a single benchmark. Returns result dict.

        Raises NotImplementedError if the benchmark has no run_fn.
        """
        defn = _REGISTRY.get(benchmark_id)
        if defn is None:
            raise KeyError(f"Unknown benchmark: {benchmark_id}")
        if defn.run_fn is None:
            raise NotImplementedError(
                f"Benchmark {benchmark_id} ({defn.name}) is not yet implemented. "
                f"Dependencies: {', '.join(defn.dependencies)}. "
                f"Target phase: {defn.phase}."
            )
        return defn.run_fn()

    def run_all(self) -> dict[str, dict | str]:
        """Run all implemented benchmarks. Returns {id: result_or_error}."""
        results = {}
        for defn in self.list_benchmarks():
            try:
                results[defn.benchmark_id] = self.run(defn.benchmark_id)
            except NotImplementedError as e:
                results[defn.benchmark_id] = f"NOT_IMPLEMENTED: {e}"
        return results
