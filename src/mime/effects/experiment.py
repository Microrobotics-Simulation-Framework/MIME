"""Experiment composition + the six-pass build() validation contract.

ADR-2026-EFFECT-MODEL §4-§5, §7. An ``Experiment`` is a set of attached
:class:`EffectModel`s (and optional cross-effect couplings) over one body in
one medium. ``build()`` materialises the MADDENING graph through six ordered
validation passes, each with a specific typed error so the validation
promise is testable rather than aspirational.

Version-and-provenance metadata (§7) lives on the object; version
compatibility is validated at *construction/load* time — before any graph
is built — so a clean-install reproducer gets a clear "your MIME is too
old / too new" message immediately.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from mime.effects.body_medium import Body, Medium
from mime.effects.errors import (
    BodyPropertyMissing,
    CouplingError,
    EffectBuildError,
    IncompatibleMimeVersionError,
    MediumPropertyMissing,
    PortTypeMismatchError,
)
from mime.effects.protocol import CouplingSpec, EffectHandle, RegimeWarning

if TYPE_CHECKING:  # pragma: no cover
    from maddening.core.graph_manager import GraphManager

    from mime.effects.protocol import EffectModel

    # A coupling-group member: a node-name string, an attached EffectModel
    # (expands to its node names), or the experiment Body (→ body node name).
    CouplingGroupMember = Union[str, "EffectModel", Body]


@dataclass(frozen=True)
class _CouplingGroupSpec:
    """A declared implicit-coupling group, materialised in ``build()``."""

    members: tuple
    max_iterations: int
    tolerance: float
    kwargs: dict


def _installed_mime_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mime-engine")
    except PackageNotFoundError:  # pragma: no cover - editable/source tree
        return "0.0.0"


def _version_tuple(v: str) -> tuple[int, ...]:
    """Minimal PEP 440-ish parse (major.minor.patch); ignores pre/post tags."""
    core = v.split("+")[0].split("-")[0]
    parts: list[int] = []
    for chunk in core.split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts) or (0,)


def _version_lt(a: str, b: str) -> bool:
    return _version_tuple(a) < _version_tuple(b)


@dataclass
class BenchmarkRef:
    """A validated experimental benchmark this experiment reproduces."""

    key: str
    citation: str = ""


@dataclass
class CitationInfo:
    doi: str = ""
    authors: str = ""
    reference: str = ""


class Experiment:
    """A directory + a set of attached EffectModels over one body/medium.

    Two construction paths (ADR §4): ``Experiment(name=..., directory=...)``
    programmatically (used in ``setup.py``'s ``make_experiment``), or
    ``Experiment.from_directory(path)`` declaratively.
    """

    def __init__(
        self,
        name: str,
        directory: Optional[Path] = None,
        *,
        description: str = "",
        mime_version_min: str = "0.0.0",
        mime_version_max: Optional[str] = None,
        asset_paths: Optional[dict[str, Path]] = None,
        asset_hashes: Optional[dict[str, str]] = None,
        benchmark_refs: tuple[BenchmarkRef, ...] = (),
        citation: Optional[CitationInfo] = None,
    ):
        self.name = name
        self.directory = directory
        self.description = description
        self.mime_version_min = mime_version_min
        self.mime_version_max = mime_version_max
        self.asset_paths = asset_paths or {}
        self.asset_hashes = asset_hashes or {}
        self.benchmark_refs = benchmark_refs
        self.citation = citation

        self.warnings: list[RegimeWarning] = []
        self._effects: dict[str, "EffectModel"] = {}
        self._effect_by_id: dict[int, str] = {}
        self._auto_counter: dict[str, int] = {}
        self._couplings: list[CouplingSpec] = []
        self._coupling_groups: list[_CouplingGroupSpec] = []
        self._external_inputs: list[tuple[str, str, tuple, Any]] = []
        self.body: Optional[Body] = None
        self.medium: Optional[Medium] = None

        self._validate_version()

    # ── version compatibility (load-time, §7) ────────────────────────────

    def _validate_version(self) -> None:
        installed = _installed_mime_version()
        if _version_lt(installed, self.mime_version_min):
            raise IncompatibleMimeVersionError(
                f"experiment {self.name!r} requires MIME >= "
                f"{self.mime_version_min}, but {installed} is installed"
            )
        if self.mime_version_max is not None and _version_lt(
            self.mime_version_max, installed
        ):
            raise IncompatibleMimeVersionError(
                f"experiment {self.name!r} requires MIME <= "
                f"{self.mime_version_max}, but {installed} is installed"
            )

    # ── composition surface (§4) ─────────────────────────────────────────

    def set_body(self, body: Body) -> None:
        self.body = body

    def set_medium(self, medium: Medium) -> None:
        self.medium = medium

    def add_external_input(
        self, node: str, field: str, shape: tuple = (), dtype: Any = None,
    ) -> None:
        """Declare a graph-external input injected at ``step()`` time.

        Registers a `BoundaryInputSpec` that no edge drives (e.g. a prescribed
        body velocity for a kinematic drive, or a commanded field frequency /
        motor velocity), so a caller can feed it via
        ``gm.step({node: {field: value}})``. Registered on the GraphManager in
        `build()` before `compile()`. ``dtype`` defaults to MADDENING's default
        (``float32``) when None.
        """
        self._external_inputs.append((node, field, tuple(shape), dtype))

    def add_coupling_group(
        self,
        members: "Sequence[CouplingGroupMember]",
        *,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Declare an implicit-coupling group over a set of nodes (E6a).

        Generalises the hand-built coupling groups the directory experiments
        wire today (``umr_confinement`` Schwarz IQN-ILS + subcycling;
        ``dejongh_new_chain`` Gauss-Seidel body↔drag). Materialised in
        ``build()`` after every effect's subgraph exists, by calling
        ``GraphManager.add_coupling_group``.

        ``members`` entries may be:

        * a node-name ``str`` (e.g. ``"body"``),
        * an attached :class:`EffectModel` (expands to *all* node names the
          effect added to the graph), or
        * the experiment's :class:`Body` (expands to the body node name).

        Extra ``kwargs`` (``acceleration="iqn-ils"``, ``subcycling=True``,
        ``convergence_norm``, ``boundary_interpolation`` ...) pass straight
        through to MADDENING.
        """
        self._coupling_groups.append(
            _CouplingGroupSpec(
                members=tuple(members),
                max_iterations=max_iterations,
                tolerance=tolerance,
                kwargs=dict(kwargs),
            )
        )

    def attach(self, model: "EffectModel", *, name: Optional[str] = None) -> str:
        """Register an EffectModel. Auto-names from its registered effect name
        (``"HydrodynamicModel.LBM_0"``, ``..._1``) when ``name`` is None.
        Returns the resolved name."""
        if name is None:
            base = getattr(model, "_effect_name", type(model).__name__)
            idx = self._auto_counter.get(base, 0)
            self._auto_counter[base] = idx + 1
            name = f"{base}_{idx}"
        if name in self._effects:
            raise ValueError(f"effect name {name!r} already attached")
        self._effects[name] = model
        self._effect_by_id[id(model)] = name
        return name

    def _resolve(self, ref: "EffectModel | str") -> str:
        if isinstance(ref, str):
            return ref
        return self._effect_by_id.get(id(ref), "<unattached>")

    def couple(
        self,
        target: "EffectModel | str",
        *,
        target_port: str,
        source: "EffectModel | str",
        source_port: str,
    ) -> None:
        """Declare a cross-effect coupling (resolved to registered names)."""
        self._couplings.append(
            CouplingSpec(
                source=self._resolve(source),
                source_port=source_port,
                target=self._resolve(target),
                target_port=target_port,
            )
        )

    # ── build() — six ordered validation passes (§5) ─────────────────────

    def build(self) -> tuple["GraphManager", list[EffectHandle]]:
        from maddening.core.graph_manager import GraphManager

        if self.body is None or self.medium is None:
            raise EffectModelError_body_medium()

        # Pass 1 — effect-reference resolution.
        for c in self._couplings:
            for ref in (c.source, c.target):
                if ref not in self._effects:
                    raise CouplingError(f"unknown effect reference: {ref!r}")

        # Pass 2 — port existence + type compatibility.
        for c in self._couplings:
            src = self._effects[c.source]
            tgt = self._effects[c.target]
            src_ports = src.coupling_ports
            tgt_ports = getattr(tgt, "input_ports", {})
            if c.source_port not in src_ports:
                raise CouplingError(
                    f"effect {c.source!r} has no coupling port "
                    f"{c.source_port!r}"
                )
            if c.target_port not in tgt_ports:
                raise CouplingError(
                    f"effect {c.target!r} has no input port {c.target_port!r}"
                )
            if not _edge_specs_compatible(
                src_ports[c.source_port], tgt_ports[c.target_port]
            ):
                raise PortTypeMismatchError(
                    f"port {c.source}.{c.source_port} is incompatible with "
                    f"{c.target}.{c.target_port}"
                )

        # Pass 3 — body/medium property presence.
        for name, model in self._effects.items():
            for section in model.required_body_properties():
                if not self.body.has_section(section):
                    raise BodyPropertyMissing(
                        f"effect {name!r} requires body section "
                        f"{section!r}; body has {sorted(self.body.properties)}"
                    )
            for prop in model.required_medium_properties():
                if not self.medium.has(prop):
                    raise MediumPropertyMissing(
                        f"effect {name!r} requires medium property "
                        f"{prop!r}; medium has {sorted(self.medium.properties)}"
                    )

        # Pass 4 — regime checks (advisories, never failures).
        self.warnings = []
        for name, model in self._effects.items():
            for w in model.applicable_regime().check(
                body=self.body, medium=self.medium
            ):
                self.warnings.append(w)
                warnings.warn(w.message, _RegimeUserWarning, stacklevel=2)

        # Pass 5 — per-EffectModel build() in attachment order.
        gm = GraphManager()
        if self.body.node is not None:
            gm.add_node(self.body.node)
        handles: list[EffectHandle] = []
        handle_by_effect: dict[str, EffectHandle] = {}
        for name, model in self._effects.items():
            try:
                handle = model.build(gm, body=self.body, medium=self.medium)
            except Exception as exc:  # noqa: BLE001 - re-wrapped with context
                raise EffectBuildError(
                    f"effect {name!r} build() failed: {exc}"
                ) from exc
            handles.append(handle)
            handle_by_effect[name] = handle

        # Pass 5.5 — materialise cross-effect couplings (E3) now that both
        # effects' subgraphs exist. couple() recorded source/target ports;
        # here we wire the bound (node, field) edges (additive operator-split
        # by default; the coupling port may request additive / a transform).
        for c in self._couplings:
            src_port = self._effects[c.source].coupling_ports[c.source_port]
            tgt_port = getattr(
                self._effects[c.target], "input_ports", {})[c.target_port]
            for port, role in ((src_port, "output"), (tgt_port, "input")):
                if not (hasattr(port, "node") and hasattr(port, "field")):
                    raise CouplingError(
                        f"coupling {role} port is not materialisable (no "
                        f"node/field binding — use a CouplingPort)"
                    )
            gm.add_edge(
                src_port.node, tgt_port.node,
                src_port.field, tgt_port.field,
                transform=src_port.transform,
                additive=src_port.additive or tgt_port.additive,
            )

        # Materialise implicit-coupling groups (E6a) now that every effect's
        # nodes exist. Must precede compile().
        for group in self._coupling_groups:
            nodes = self._resolve_group_members(group.members, handle_by_effect)
            gm.add_coupling_group(
                nodes,
                max_iterations=group.max_iterations,
                tolerance=group.tolerance,
                **group.kwargs,
            )

        # Register declared graph-external inputs (must precede compile()).
        for node, field, shape, dtype in self._external_inputs:
            if dtype is None:
                gm.add_external_input(node, field, shape=shape)
            else:
                gm.add_external_input(node, field, shape=shape, dtype=dtype)

        # Pass 6 — MADDENING edge validation / compile.
        gm.compile()

        return gm, handles

    def _resolve_group_members(
        self,
        members: "Sequence[CouplingGroupMember]",
        handle_by_effect: dict[str, EffectHandle],
    ) -> list[str]:
        """Expand coupling-group members (str / EffectModel / Body) to a
        de-duplicated, order-preserving list of graph node names."""
        out: list[str] = []

        def _add(name: str) -> None:
            if name not in out:
                out.append(name)

        for m in members:
            if isinstance(m, str):
                _add(m)
            elif isinstance(m, Body):
                _add(m.name)
            else:
                # Treat as an attached EffectModel — expand to its node names.
                eff_name = self._effect_by_id.get(id(m))
                if eff_name is None or eff_name not in handle_by_effect:
                    raise CouplingError(
                        f"coupling-group member {m!r} is not an attached "
                        f"effect, a node name, or the body"
                    )
                for n in handle_by_effect[eff_name].node_names:
                    _add(n)
        return out

    @classmethod
    def from_directory(cls, path: Path) -> "Experiment":  # pragma: no cover
        raise NotImplementedError(
            "Experiment.from_directory (YAML loader) lands with the v0.3 "
            "magnetic family; the v0.2 pilot uses make_experiment(params)."
        )


class _RegimeUserWarning(UserWarning):
    """Category for regime advisories (so callers can filter them)."""


def EffectModelError_body_medium() -> Exception:
    from mime.effects.errors import EffectModelError

    return EffectModelError(
        "Experiment.build() needs both a body (set_body) and a medium "
        "(set_medium) before building."
    )


def _edge_specs_compatible(src: Any, tgt: Any) -> bool:
    """Structural EdgeSpec compatibility (shape; dtype/units if present)."""
    s_shape = getattr(src, "shape", None)
    t_shape = getattr(tgt, "shape", None)
    if s_shape is not None and t_shape is not None and s_shape != t_shape:
        return False
    return True
