"""ExternalMagneticFieldNode — magnetic field from coils or rotating permanent magnet.

Two modes:
1. Helmholtz/coil array: B(p) = B_map(p) @ I, where B_map is the
   field-current mapping matrix. Uniform field approximation at workspace
   center.
2. Rotating uniform field: B(t) = B_0 * [cos(omega*t), sin(omega*t), 0],
   the simplest model for a rotating permanent magnet or Helmholtz pair
   driven in quadrature.

The node's state is the current field vector B and its spatial gradient.
The ControlPolicy commands frequency_hz and field_strength_mt via
ExternalInputSpec boundary inputs.

When to use this node vs. the Motor + PermanentMagnet chain
-----------------------------------------------------------
``ExternalMagneticFieldNode`` is **not deprecated** — it remains a
first-class peer of the new actuation chain (``MotorNode`` +
``PermanentMagnetNode``). Pick this node when:

* ``field_gradient = 0`` is acceptable (it is hard-wired to zero —
  see ``update`` below), AND
* the uniform-field assumption holds in the workspace (e.g., the
  workspace centre of a Helmholtz coil pair).

Pick the new chain instead when any of (a) field gradient matters,
(b) the magnet has a finite or tracked physical pose,
(c) misalignment / wobble effects are under study, or (d) the demo
scene needs a rendered apparatus. The two paths are wired
identically downstream — both produce ``field_vector`` and
``field_gradient`` consumed by ``PermanentMagnetResponseNode`` —
so the experiment's graph builder picks one or the other based on
what is being studied.

Reference: Appendix C of "Mathematical Modelling of Swimming Soft Microrobots"
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole, ActuationMeta, ActuationPrinciple,
)


@stability(StabilityLevel.EXPERIMENTAL)
class ExternalMagneticFieldNode(MimeNode):
    """Generates a rotating uniform magnetic field.

    The field rotates in the xy-plane at the commanded frequency:
    B(t) = B_0 * [cos(2*pi*f*t), sin(2*pi*f*t), 0]

    This is the simplest model — a uniform field over the workspace.
    Suitable for Helmholtz coil pairs driven in quadrature or a
    distant rotating permanent magnet (far-field dipole approximation).

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.

    Boundary Inputs (from ControlPolicy)
    ------------------------------------
    frequency_hz : scalar
        Rotation frequency in Hz. Commandable by ControlPolicy.
    field_strength_mt : scalar
        Field magnitude in milliTesla. Commandable by ControlPolicy.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-001",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Rotating uniform magnetic field from Helmholtz coils or "
            "distant permanent magnet"
        ),
        governing_equations=(
            r"B(t) = B_0 [cos(2\pi f t), sin(2\pi f t), 0]; "
            r"B_map: B(p) = \mathcal{B}(p) I (coil array mode)"
        ),
        discretization="Analytical — no discretisation needed",
        assumptions=(
            "Uniform field over the workspace (valid near Helmholtz coil center)",
            "No eddy currents or shielding from biological tissue",
            "Coil inductance delay is negligible (quasi-static field)",
            "Field rotation is in the xy-plane",
        ),
        limitations=(
            "Uniform field approximation invalid far from workspace center",
            "No spatial gradient computation in uniform mode (gradient = 0)",
            "2D rotation only — no out-of-plane field components",
        ),
        validated_regimes=(
            ValidatedRegime("frequency_hz", 0.0, 200.0, "Hz",
                            "Typical microrobot actuation range"),
            ValidatedRegime("field_strength_mt", 0.0, 100.0, "mT",
                            "Below saturation for most soft-magnetic materials"),
        ),
        references=(
            Reference("Abbott2009", "How Should Microrobots Swim?"),
        ),
        hazard_hints=(
            "Uniform field approximation breaks down far from coil center; "
            "position-dependent B_map(p) required for off-center operation",
            "No gradient force possible in uniform field mode — only torque "
            "actuation is physically meaningful",
        ),
        implementation_map={
            "B(t) = B_0 * [cos, sin, 0]": (
                "mime.nodes.actuation.external_magnetic_field."
                "ExternalMagneticFieldNode.update"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.EXTERNAL_APPARATUS,
        actuation=ActuationMeta(
            principle=ActuationPrinciple.ROTATING_MAGNETIC_FIELD,
            is_onboard=False,
            max_frequency_hz=200.0,
            max_field_strength_mt=100.0,
            commandable_fields=("frequency_hz", "field_strength_mt"),
        ),
    )

    def __init__(self, name: str, timestep: float, **kwargs):
        super().__init__(name, timestep, **kwargs)

    def initial_state(self) -> dict:
        return {
            "field_vector": jnp.zeros(3),       # B(t) in Tesla
            "field_gradient": jnp.zeros((3, 3)), # dB/dx (zero for uniform)
            "sim_time": jnp.array(0.0),          # accumulated time
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "frequency_hz": BoundaryInputSpec(
                shape=(), default=10.0,
                description="Rotation frequency in Hz",
            ),
            "field_strength_mt": BoundaryInputSpec(
                shape=(), default=10.0,
                description="Field magnitude in milliTesla",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        freq = boundary_inputs.get("frequency_hz", 10.0)
        strength_mt = boundary_inputs.get("field_strength_mt", 10.0)

        t = state["sim_time"] + dt
        omega = 2.0 * jnp.pi * freq
        B_0 = strength_mt * 1e-3  # mT -> T

        field_vector = B_0 * jnp.array([
            jnp.cos(omega * t),
            jnp.sin(omega * t),
            0.0,
        ])

        return {
            "field_vector": field_vector,
            "field_gradient": jnp.zeros((3, 3)),  # uniform field
            "sim_time": t,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "field_vector": state["field_vector"],
            "field_gradient": state["field_gradient"],
        }
