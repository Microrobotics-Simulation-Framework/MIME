"""FieldSuperpositionNode — linear superposition of N magnetic field sources.

Magnetostatics is linear: the field and its spatial gradient from several
independent sources (permanent-magnet dipoles, coils, a uniform background)
add at any point::

    B_net(p)   = Σ_i B_i(p)
    ∇B_net(p)  = Σ_i ∇B_i(p)

This node combines ``n_sources`` ``(field_vector_i, field_gradient_i)`` input
pairs into a single ``(field_vector, field_gradient)`` output, so a single
``PermanentMagnetResponseNode`` downstream sees the *net* field and the net
gradient — the physics the dual-RPM controller relies on (gradient
cancellation between two synchronised dipoles while the net rotating field is
preserved; net lateral holding force ``F = ∇(m·B_net)``).

It is a stateless combiner: ``update`` recomputes the sum each step from its
inputs; the cached state exists only so ``compute_boundary_fluxes`` can expose
the result on the node's output edges.

Output: ``field_vector`` (3,) [T], ``field_gradient`` (3,3) [T/m].
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryFluxSpec, BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole, ActuationMeta, ActuationPrinciple,
)


@stability(StabilityLevel.EXPERIMENTAL)
class FieldSuperpositionNode(MimeNode):
    """Sum the magnetic field + gradient of ``n_sources`` independent sources.

    Parameters
    ----------
    name : str
    timestep : float
    n_sources : int
        Number of ``(field_vector_i, field_gradient_i)`` input pairs to sum,
        for ``i`` in ``0 .. n_sources - 1``. Must be >= 1.

    Notes
    -----
    Each input defaults to zero, so an unwired source contributes nothing —
    a graph may legitimately leave a slot unconnected (e.g. a 3-source node
    driven by only two active dipoles during a transient).
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-015",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Linear superposition of N magnetic field / field-gradient "
            "sources into a single net field and net gradient."
        ),
        governing_equations=(
            "B_net = Σ_i B_i ;  ∇B_net = Σ_i ∇B_i  (linearity of "
            "magnetostatics)"
        ),
        discretization="Pointwise algebraic sum; no integration.",
        assumptions=(
            "Sources are independent (no mutual magnetisation / saturation "
            "coupling between them).",
            "All inputs are expressed in the same world frame and SI units "
            "(Tesla, Tesla/metre).",
        ),
        limitations=(
            "Does not model nonlinear material response (saturation, "
            "hysteresis) — superposition holds only in the linear regime.",
        ),
        validated_regimes=(
            ValidatedRegime("n_sources", 1.0, 8.0, "count",
                            "Small fixed source counts for multi-magnet "
                            "steering / gradient-cancellation controllers."),
        ),
        references=(
            Reference("Hosney2016",
                      "Dual-magnet gradient cancellation for 5-DOF control"),
            Reference("Mahoney2014",
                      "Single-magnet rotating-field manipulation"),
        ),
        hazard_hints=(
            "Mixing frames (one source in body frame, another in world) "
            "silently produces a meaningless net field — ensure every "
            "upstream source emits in the shared world frame.",
        ),
        implementation_map={
            "B_net = Σ_i B_i":
                "mime.nodes.actuation.field_superposition."
                "FieldSuperpositionNode.update",
        },
    )

    # Models the aggregate field/gradient delivered by the external magnetic
    # apparatus (the combined dipoles), so it carries the apparatus role. Its
    # dual-RPM purpose is gradient-based steering; it is a passive combiner, so
    # nothing is directly commandable on it (the dipoles' poses are).
    mime_meta = MimeNodeMeta(
        role=NodeRole.EXTERNAL_APPARATUS,
        actuation=ActuationMeta(
            principle=ActuationPrinciple.GRADIENT_MAGNETIC_FIELD,
            is_onboard=False,
            commandable_fields=(),
        ),
    )

    def __init__(self, name: str, timestep: float, n_sources: int = 2, **kwargs):
        super().__init__(name, timestep, **kwargs)
        if int(n_sources) < 1:
            raise ValueError(
                f"FieldSuperpositionNode needs n_sources >= 1, got {n_sources}"
            )
        self._n_sources = int(n_sources)

    @property
    def n_sources(self) -> int:
        return self._n_sources

    def initial_state(self) -> dict:
        return {
            "field_vector": jnp.zeros(3),
            "field_gradient": jnp.zeros((3, 3)),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        spec: dict[str, BoundaryInputSpec] = {}
        for i in range(self._n_sources):
            spec[f"field_vector_{i}"] = BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description=f"Field vector [T] from source {i}",
            )
            spec[f"field_gradient_{i}"] = BoundaryInputSpec(
                shape=(3, 3), default=jnp.zeros((3, 3)),
                description=f"Field gradient dB/dx [T/m] from source {i}",
            )
        return spec

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "field_vector": BoundaryFluxSpec(
                shape=(3,), description="Net field vector [T]",
                output_units="T",
            ),
            "field_gradient": BoundaryFluxSpec(
                shape=(3, 3), description="Net field gradient dB/dx [T/m]",
                output_units="T/m",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        b_net = jnp.zeros(3)
        grad_net = jnp.zeros((3, 3))
        for i in range(self._n_sources):
            b_net = b_net + boundary_inputs.get(f"field_vector_{i}", jnp.zeros(3))
            grad_net = grad_net + boundary_inputs.get(
                f"field_gradient_{i}", jnp.zeros((3, 3))
            )
        return {"field_vector": b_net, "field_gradient": grad_net}

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "field_vector": state["field_vector"],
            "field_gradient": state["field_gradient"],
        }
