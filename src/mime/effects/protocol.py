"""EffectModel Protocol surface (ADR-2026-EFFECT-MODEL §1-§3).

The load-bearing abstraction: an :class:`EffectModel` is a *builder* for a
"subgraph that delivers force/torque to a body". The hydrodynamic family
(this pilot) and the magnetic family (v0.3) are concrete instances.

This module is deliberately framework-light — an EffectModel's ``build()``
receives a MADDENING ``GraphManager`` and materialises nodes/edges onto it,
returning an introspection-only :class:`EffectHandle`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Callable, Optional, Protocol, Sequence, TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from maddening.core.edge import EdgeSpec
    from maddening.core.graph_manager import GraphManager

    from mime.effects.body_medium import Body, Medium


# ── Regime metadata (§2) ─────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeWarning:
    """A single advisory that a model is applied outside its validated
    regime. Surfaced via ``warnings.warn`` and collected on
    ``Experiment.warnings``; never gates execution."""

    effect: str
    message: str


class Regime(ABC):
    """Per-physics-domain regime metadata.

    Polymorphic: each physics family ships its own subclass with its own
    ``check`` logic. ``regime_check`` is family-agnostic — it just calls
    ``model.applicable_regime().check(...)`` (no central isinstance dispatch).
    """

    @abstractmethod
    def check(
        self,
        *,
        body: "Body",
        medium: "Medium",
        sources: Sequence["Source"] = (),
    ) -> list[RegimeWarning]:
        """Return advisories (possibly empty). Never raises for regime."""


# ── Sources (§1 — for SourcedEffectModel families, e.g. magnetic) ─────────

@dataclass(frozen=True)
class Source:
    """Base for an external driver of a SourcedEffectModel (magnetic dipole,
    coil, acoustic transducer, ...). The hydrodynamic pilot has no sources;
    this exists so the magnetic family (v0.3) slots in without a refactor."""

    name: str


S = TypeVar("S", bound=Source, covariant=True)


# ── Coupling (§4) ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CouplingPort:
    """A named cross-effect port — a binding to one (node, field) on the graph
    the effect builds, plus the metadata ``Experiment.build`` needs to validate
    and materialise a coupling edge.

    ``shape`` is used by pass-2 type compatibility; ``node`` / ``field`` are the
    graph binding pass-5 wires (``add_edge(out.node, in.node, out.field,
    in.field)``). Computable from constructor arguments alone (per the
    EffectModel contract), since an effect knows its node names up front.
    """

    node: str
    field: str
    shape: tuple = ()
    additive: bool = False
    transform: Optional[Callable] = None


@dataclass(frozen=True)
class CouplingSpec:
    """String-based cross-effect coupling. Hashable and serialisation-clean;
    instances are resolved to registered names by ``Experiment.couple``."""

    source: str
    source_port: str
    target: str
    target_port: str


# ── Handle (§1) ────────────────────────────────────────────────────────────

@dataclass
class EffectHandle:
    """Pure-introspection handle returned by ``build()`` — the names of the
    nodes the effect added to the graph."""

    node_names: tuple[str, ...]
    coupling_ports: dict = field(default_factory=dict)


# ── The Protocol (§1, §3) ───────────────────────────────────────────────────

@runtime_checkable
class EffectModel(Protocol):
    """Builder for an environment effect that delivers force/torque to a body.

    Concrete implementations: ``HydrodynamicModel.LBM``,
    ``MagneticModel.PointDipole`` (v0.3), ``AcousticModel.LBMAcoustic``
    (post-1.0), ...
    """

    @property
    def coupling_ports(self) -> dict[str, "CouplingPort"]:
        """Named **output** ports for cross-effect coupling (this effect
        *produces* these). Each is a :class:`CouplingPort` binding to one
        (node, field) on the subgraph this effect builds.

        Invariant: computable from constructor arguments alone (no graph
        state) — pass 2 of ``build()`` validation needs these before any
        subgraph is materialised. Empty for effects with no cross-coupling.
        The complementary **input** ports an effect *consumes* are declared on
        the optional ``input_ports`` property (default empty).
        """
        ...

    def applicable_regime(self) -> Regime:
        """Per-physics-domain regime metadata (advisory only)."""
        ...

    def required_body_properties(self) -> set[str]:
        """Per-physics body-property sections this effect needs (pass 3)."""
        ...

    def required_medium_properties(self) -> set[str]:
        """Medium properties this effect needs (pass 3)."""
        ...

    def build(
        self,
        gm: "GraphManager",
        *,
        body: "Body",
        medium: "Medium",
    ) -> EffectHandle:
        """Materialise the subgraph onto ``gm``. Called once per
        ``Experiment.build()``; the graph is static for the run."""
        ...


class SourcedEffectModel(EffectModel, Protocol[S]):
    """An EffectModel with external sources (magnetic, acoustic, electric).

    ``S`` is covariant — it appears only in return position (``sources``),
    so ``SourcedEffectModel[MagneticSource]`` is usable where
    ``SourcedEffectModel[Source]`` is expected (narrow → wide).
    """

    @property
    def sources(self) -> Sequence[S]:
        ...


class HydrodynamicRegime(Regime):
    """Reynolds-number regime for hydrodynamic effects.

    The validated microrobot regime is creeping flow (Re ≪ 1). A model
    applied far outside its ``re_range`` gets an advisory, not a failure.
    """

    def __init__(self, re_range: tuple[float, float] = (0.0, 1.0)):
        self.re_range = re_range

    def check(self, *, body, medium, sources=()) -> list[RegimeWarning]:
        warnings_out: list[RegimeWarning] = []
        re = medium.properties.get("reynolds_number")
        if re is not None and not (self.re_range[0] <= re <= self.re_range[1]):
            warnings_out.append(
                RegimeWarning(
                    effect="hydrodynamic",
                    message=(
                        f"Reynolds number {re:g} is outside the validated "
                        f"range {self.re_range}; results are extrapolative."
                    ),
                )
            )
        return warnings_out


# ── Base class with sensible defaults ───────────────────────────────────────

class BaseEffectModel:
    """Optional convenience base supplying the no-op declarations so concrete
    models override only what they use. Implements the :class:`EffectModel`
    Protocol structurally."""

    @property
    def coupling_ports(self) -> dict[str, "CouplingPort"]:
        return {}

    # NB: ``input_ports`` (the consumed cross-coupling ports) is intentionally
    # *not* a property here — concrete effects (and test stubs) set it as a
    # plain attribute. Experiment.build reads it via ``getattr(effect,
    # "input_ports", {})`` so effects without inputs need declare nothing.

    def applicable_regime(self) -> Regime:
        raise NotImplementedError

    def required_body_properties(self) -> set[str]:
        return set()

    def required_medium_properties(self) -> set[str]:
        return set()

    def build(self, gm, *, body, medium) -> EffectHandle:
        raise NotImplementedError
