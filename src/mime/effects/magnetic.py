"""MagneticModel family — the v0.3 EffectModel for magnetic actuation.

ADR-2026-EFFECT-MODEL §7 item 1 / MAGNETIC_NODE_AUDIT.md: the magnetic family
is the second EffectModel family (after the v0.2 ``HydrodynamicModel`` pilot).
Like the pilot, each backend is an *adapter* over pre-built nodes — it
materialises the field-production chain plus the body-coupling edges onto a
``GraphManager``; **no node is rewritten**.

Three backends:

* ``MagneticModel.UniformField`` — a far-field uniform rotating field
  (``ExternalMagneticFieldNode``) driving a magnetic body
  (``PermanentMagnetResponseNode``). The field gradient is zero, so this
  delivers torque only (Abbott "how should microrobots swim").
* ``MagneticModel.PointDipole`` — a positioned permanent-magnet dipole
  (``PermanentMagnetNode``) whose field *and gradient* act on the body, giving
  both steering torque and a gradient force ``F = ∇(m·B)``.
* ``MagneticModel.DualDipole`` — two (or more) dipoles superposed through a
  ``FieldSuperpositionNode`` before the body response. This is the dual-RPM
  controller primitive: synchronised dipoles can cancel the net gradient
  (force-free rotation) or hold a net lateral force while the net field still
  rotates (Hosney 2016 / Mahoney 2014).

The body always receives ``magnetic_force`` / ``magnetic_torque`` on additive
edges, so a magnetic effect composes with a hydrodynamic effect on the same
body without either overwriting the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from mime.effects.protocol import (
    BaseEffectModel,
    EffectHandle,
    RegimeWarning,
    Regime,
)
from mime.effects.registry import register_effect
from mime.effects.sources import MagneticSource

if TYPE_CHECKING:  # pragma: no cover
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import SimulationNode

    from mime.effects.body_medium import Body, Medium


# ── Regime metadata ─────────────────────────────────────────────────────────

class MagneticRegime(Regime):
    """Drive-frequency / field-strength regime for magnetic actuation.

    Validated microrobot actuation sits below soft-magnetic saturation and
    in the sub-200 Hz rotating-field band (``ExternalMagneticFieldNode``'s
    ``ValidatedRegime``). A drive outside the range gets an advisory, never a
    failure. The drive parameters are read from the body's ``"magnetic"``
    property section (keys ``drive_frequency_hz`` / ``drive_field_mt``); if
    absent, no advisory is emitted.
    """

    def __init__(
        self,
        freq_range_hz: tuple[float, float] = (0.0, 200.0),
        field_range_mt: tuple[float, float] = (0.0, 100.0),
    ):
        self.freq_range_hz = freq_range_hz
        self.field_range_mt = field_range_mt

    def check(self, *, body, medium, sources=()) -> list[RegimeWarning]:
        out: list[RegimeWarning] = []
        mag = body.properties.get("magnetic", {}) if body is not None else {}
        f = mag.get("drive_frequency_hz")
        if f is not None and not (self.freq_range_hz[0] <= f <= self.freq_range_hz[1]):
            out.append(RegimeWarning(
                effect="magnetic",
                message=(
                    f"drive frequency {f:g} Hz is outside the validated range "
                    f"{self.freq_range_hz}; actuation is extrapolative."
                ),
            ))
        b = mag.get("drive_field_mt")
        if b is not None and not (self.field_range_mt[0] <= b <= self.field_range_mt[1]):
            out.append(RegimeWarning(
                effect="magnetic",
                message=(
                    f"drive field {b:g} mT is outside the validated range "
                    f"{self.field_range_mt}; soft-magnetic saturation likely."
                ),
            ))
        return out


# ── Common adapter ──────────────────────────────────────────────────────────

class _MagneticEffect(BaseEffectModel):
    """Common magnetic adapter.

    Subclasses implement :meth:`_wire_field_production`, which adds the
    field-producing nodes (uniform field, one dipole, or N superposed dipoles)
    and returns ``(emitter_name, added_node_names)`` where ``emitter_name`` is
    the node exposing the net ``field_vector`` / ``field_gradient``. The base
    then wires that emitter into the shared ``PermanentMagnetResponseNode`` and
    the response into the body.
    """

    def __init__(
        self,
        response_node: "SimulationNode",
        *,
        freq_range_hz: tuple[float, float] = (0.0, 200.0),
        field_range_mt: tuple[float, float] = (0.0, 100.0),
        torque_only: bool = False,
    ):
        self._response = response_node
        # Torque-only (uniform rotating field / Helmholtz idealization): skip the
        # ∇B edge so the response force = ∇B·m = 0. The field still ROTATES (the
        # dipole direction is driven by the motor) → pure rotating torque, no
        # gradient attraction. This is the standard rotating-field microswimmer
        # actuation and keeps the screw on the vessel axis (where the confined
        # wall table is valid); a positioned permanent magnet's gradient pull
        # would otherwise eject the unconfined body from the lumen.
        self._torque_only = bool(torque_only)
        self._freq_range_hz = freq_range_hz
        self._field_range_mt = field_range_mt

    # -- Protocol surface --------------------------------------------------

    def applicable_regime(self) -> MagneticRegime:
        return MagneticRegime(self._freq_range_hz, self._field_range_mt)

    def required_body_properties(self) -> set[str]:
        return {"magnetic"}

    def required_medium_properties(self) -> set[str]:
        # Magnetic actuation does not read the surrounding fluid medium.
        return set()

    # -- Build -------------------------------------------------------------

    def _wire_field_production(
        self, gm: "GraphManager", body: "Body",
    ) -> tuple[str, tuple[str, ...]]:
        raise NotImplementedError

    def build(self, gm: "GraphManager", *, body: "Body", medium: "Medium") -> EffectHandle:
        emitter, added = self._wire_field_production(gm, body)
        gm.add_node(self._response)
        resp = self._response.name
        # Net field → the magnetic-response node. The ∇B edge is skipped in
        # torque-only mode (uniform rotating field) so the response force = 0.
        gm.add_edge(emitter, resp, "field_vector", "field_vector")
        if not self._torque_only:
            gm.add_edge(emitter, resp, "field_gradient", "field_gradient")
        # Back-edge: the body's orientation rotates the body-frame moment into
        # the lab frame before the torque/force evaluation.
        gm.add_edge(body.name, resp, "orientation", "orientation")
        # Deliver the magnetic load to the body on additive edges so the
        # magnetic effect composes with other effects on the same body.
        gm.add_edge(resp, body.name, "magnetic_force", "magnetic_force",
                    additive=True)
        gm.add_edge(resp, body.name, "magnetic_torque", "magnetic_torque",
                    additive=True)
        return EffectHandle(node_names=(*added, resp))


# ── Backends ────────────────────────────────────────────────────────────────

class MagneticModel:
    """Namespace of magnetic-actuation backends (ADR §7 item 1)."""

    @register_effect("MagneticModel.UniformField")
    class UniformField(_MagneticEffect):
        """Far-field uniform rotating field (torque-only; ∇B = 0).

        Parameters
        ----------
        field_node : SimulationNode
            An ``ExternalMagneticFieldNode`` emitting ``field_vector`` /
            ``field_gradient`` (the latter zero for a uniform field).
        response_node : SimulationNode
            A ``PermanentMagnetResponseNode`` mapping (B, ∇B, orientation) →
            (magnetic_force, magnetic_torque).
        """

        def __init__(
            self,
            field_node: "SimulationNode",
            response_node: "SimulationNode",
            *,
            freq_range_hz: tuple[float, float] = (0.0, 200.0),
            field_range_mt: tuple[float, float] = (0.0, 100.0),
        ):
            super().__init__(response_node, freq_range_hz=freq_range_hz,
                             field_range_mt=field_range_mt)
            self._field_node = field_node

        def _wire_field_production(self, gm, body):
            gm.add_node(self._field_node)
            return self._field_node.name, (self._field_node.name,)

    @register_effect("MagneticModel.PointDipole")
    class PointDipole(_MagneticEffect):
        """A positioned permanent-magnet dipole (field + gradient).

        Field-model choice (set on the wrapped ``PermanentMagnetNode`` via its
        ``field_model``; this effect only wraps the node):

        * ``"point_dipole"`` — faithful at standoff **z ≳ 5·R_magnet** (the
          node's documented near-field envelope). Correct for the **de Jongh /
          confined-swimming envelope** (R_magnet≈1 mm, cm-scale standoff →
          z/R ≫ 5). The off-axis field tilt (the rotating-magnet step-out
          mechanism) is present at leading order.
        * ``"coulombian_poles"`` — adds the true two-point-monopole **off-axis
          tilt** correction (finite magnet length); use for **close-standoff**
          work where z/R < 5 (e.g. ar4 at the 7 cm corner, z/R≈4). Captures
          finite length off-axis; the finite-**radius** off-axis (disk) term is
          a documented further step (matters when R ≳ L/2 at very close range).
        * ``"current_loop"`` — finite-radius on-axis magnitude; its off-axis
          correction is a *scalar* on the dipole (magnitude only, **not** tilt).

        Parameters
        ----------
        magnet_node : SimulationNode
            A ``PermanentMagnetNode`` emitting ``field_vector`` /
            ``field_gradient`` at the body, reading ``magnet_pose_world`` and
            ``target_position_world``.
        response_node : SimulationNode
            A ``PermanentMagnetResponseNode``.
        pose_source : SimulationNode | None
            Convenience: an upstream node the effect adds-and-wires to provide
            the magnet's world pose (e.g. a ``MotorNode`` emitting
            ``rotor_pose_world``). When None, the magnet's ``magnet_pose_world``
            input is left to its default unless ``source`` provides it.
        pose_source_field : str
            The output field on ``pose_source`` carrying the 7-vector pose.
        source : MagneticSource | None
            General drive (E2): resolves ``SourceInputProvider``s for any magnet
            input (``magnet_pose_world`` from an external RPM command / a
            constant / an existing node; ``target_position_world`` to override
            the default body-position back-edge). Mutually exclusive with
            ``pose_source`` for the *same* field.
        """

        def __init__(
            self,
            magnet_node: "SimulationNode",
            response_node: "SimulationNode",
            *,
            pose_source: Optional["SimulationNode"] = None,
            pose_source_field: str = "rotor_pose_world",
            source: Optional["MagneticSource"] = None,
            freq_range_hz: tuple[float, float] = (0.0, 200.0),
            field_range_mt: tuple[float, float] = (0.0, 100.0),
            torque_only: bool = False,
        ):
            super().__init__(response_node, freq_range_hz=freq_range_hz,
                             field_range_mt=field_range_mt,
                             torque_only=torque_only)
            self._magnet = magnet_node
            self._pose_source = pose_source
            self._pose_source_field = pose_source_field
            self._source = source
            if (source is not None and pose_source is not None
                    and "magnet_pose_world" in source.inputs):
                raise ValueError(
                    "PointDipole: magnet_pose_world is driven by both "
                    "pose_source and source — pick one"
                )

        def _wire_field_production(self, gm, body):
            gm.add_node(self._magnet)
            added = [self._magnet.name]
            if self._pose_source is not None:
                gm.add_node(self._pose_source)
                added.append(self._pose_source.name)
                gm.add_edge(self._pose_source.name, self._magnet.name,
                            self._pose_source_field, "magnet_pose_world")
            src_fields: set[str] = set()
            if self._source is not None:
                src_fields = set(self._source.inputs)
                added.extend(self._source.resolve_all(
                    gm, self._magnet.name, timestep=self._magnet.delta_t))
            # The dipole field is evaluated at the body's location — unless the
            # source drives the target explicitly.
            if "target_position_world" not in src_fields:
                gm.add_edge(body.name, self._magnet.name,
                            "position", "target_position_world")
            return self._magnet.name, tuple(added)

    @register_effect("MagneticModel.DualDipole")
    class DualDipole(_MagneticEffect):
        """N superposed dipoles → one net field/gradient → the body.

        The dual-RPM controller primitive: each dipole's field and gradient
        are summed by a ``FieldSuperpositionNode`` before the body response, so
        synchronised dipoles can cancel the net gradient (force-free rotation)
        or hold a net lateral force while the net field rotates.

        Parameters
        ----------
        magnet_nodes : Sequence[SimulationNode]
            The ``PermanentMagnetNode`` dipoles (length must equal
            ``superposition_node.n_sources``).
        superposition_node : SimulationNode
            A ``FieldSuperpositionNode`` summing the dipoles.
        response_node : SimulationNode
            A ``PermanentMagnetResponseNode``.
        pose_sources : Sequence[SimulationNode | None] | None
            Convenience per-dipole pose-source nodes (e.g. motors) the effect
            adds-and-wires. ``None`` (or a ``None`` entry) leaves that dipole's
            pose to its default unless ``sources`` provides it.
        pose_source_field : str
            Output field on each pose source carrying the 7-vector pose.
        sources : Sequence[MagneticSource | None] | None
            General per-dipole drive (E2): resolves SourceInputProviders for
            each dipole's inputs (external RPM commands, constants, existing
            nodes). Mutually exclusive with ``pose_sources`` for the same field.
        """

        def __init__(
            self,
            magnet_nodes: Sequence["SimulationNode"],
            superposition_node: "SimulationNode",
            response_node: "SimulationNode",
            *,
            pose_sources: Optional[Sequence[Optional["SimulationNode"]]] = None,
            pose_source_field: str = "rotor_pose_world",
            sources: Optional[Sequence[Optional["MagneticSource"]]] = None,
            freq_range_hz: tuple[float, float] = (0.0, 200.0),
            field_range_mt: tuple[float, float] = (0.0, 100.0),
        ):
            super().__init__(response_node, freq_range_hz=freq_range_hz,
                             field_range_mt=field_range_mt)
            self._magnets = list(magnet_nodes)
            self._superpose = superposition_node
            self._pose_sources = list(pose_sources) if pose_sources else None
            self._pose_source_field = pose_source_field
            self._sources = list(sources) if sources else None
            n = getattr(superposition_node, "n_sources", len(self._magnets))
            if n != len(self._magnets):
                raise ValueError(
                    f"DualDipole: {len(self._magnets)} dipoles but the "
                    f"superposition node sums {n} sources"
                )
            if self._pose_sources is not None and \
                    len(self._pose_sources) != len(self._magnets):
                raise ValueError(
                    "DualDipole: pose_sources length must match magnet_nodes"
                )
            if self._sources is not None and \
                    len(self._sources) != len(self._magnets):
                raise ValueError(
                    "DualDipole: sources length must match magnet_nodes"
                )
            # Reject double-driving the same field.
            for i in range(len(self._magnets)):
                ps = self._pose_sources[i] if self._pose_sources else None
                sr = self._sources[i] if self._sources else None
                if ps is not None and sr is not None and \
                        "magnet_pose_world" in sr.inputs:
                    raise ValueError(
                        f"DualDipole dipole {i}: magnet_pose_world driven by "
                        f"both pose_sources and sources — pick one"
                    )

        def _wire_field_production(self, gm, body):
            gm.add_node(self._superpose)
            added = [self._superpose.name]
            for i, magnet in enumerate(self._magnets):
                gm.add_node(magnet)
                added.append(magnet.name)
                gm.add_edge(magnet.name, self._superpose.name,
                            "field_vector", f"field_vector_{i}")
                gm.add_edge(magnet.name, self._superpose.name,
                            "field_gradient", f"field_gradient_{i}")
                src_fields: set[str] = set()
                if self._sources is not None and self._sources[i] is not None:
                    src_fields = set(self._sources[i].inputs)
                    added.extend(self._sources[i].resolve_all(
                        gm, magnet.name, timestep=magnet.delta_t))
                if "target_position_world" not in src_fields:
                    gm.add_edge(body.name, magnet.name,
                                "position", "target_position_world")
                if self._pose_sources is not None and \
                        self._pose_sources[i] is not None:
                    ps = self._pose_sources[i]
                    gm.add_node(ps)
                    added.append(ps.name)
                    gm.add_edge(ps.name, magnet.name,
                                self._pose_source_field, "magnet_pose_world")
            return self._superpose.name, tuple(added)
