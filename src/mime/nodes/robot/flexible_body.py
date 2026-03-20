"""FlexibleBodyNode — Euler-Bernoulli beam dynamics for flagellar robots.

Models transverse bending waves along a flexible filament:
  K * d^4y/dx^4 + d^2M_act/dx^2 = -xi_perp * dy/dt

State: transverse deflection y(x,t) and velocity dy/dt at N discrete nodes
along the filament.

Uses finite-difference discretisation for the 4th-order spatial derivative
and implicit Euler integration for stability (the beam equation is stiff).

Reference: Ch 5 and Ch 7 of "Mathematical Modelling of Swimming Soft Microrobots"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole, BiocompatibilityMeta, BiocompatibilityClass,
)


def build_beam_stiffness_matrix(n_nodes: int, dx: float, K: float) -> jnp.ndarray:
    """Build the 4th-order FD stiffness matrix for an Euler-Bernoulli beam.

    Uses the stencil: [1, -4, 6, -4, 1] for d^4y/dx^4.
    Boundary conditions: clamped at x=0 (y=0, dy/dx=0), free at x=L.

    Returns S such that K * d^4y/dx^4 ≈ (K/dx^4) * S @ y.
    """
    coeff = K / dx**4
    S = jnp.zeros((n_nodes, n_nodes))

    # Interior nodes: standard 5-point stencil
    for i in range(2, n_nodes - 2):
        S = S.at[i, i-2].set(1.0 * coeff)
        S = S.at[i, i-1].set(-4.0 * coeff)
        S = S.at[i, i].set(6.0 * coeff)
        S = S.at[i, i+1].set(-4.0 * coeff)
        S = S.at[i, i+2].set(1.0 * coeff)

    # Near-boundary nodes: simplified (clamped at 0, free at L)
    if n_nodes > 2:
        # Node 0: clamped (y=0 enforced externally)
        S = S.at[0, 0].set(1.0 * coeff)

        # Node 1
        if n_nodes > 3:
            S = S.at[1, 0].set(-4.0 * coeff)
            S = S.at[1, 1].set(6.0 * coeff)
            S = S.at[1, 2].set(-4.0 * coeff)
            if n_nodes > 3:
                S = S.at[1, 3].set(1.0 * coeff)

        # Node n-2 (near free end)
        if n_nodes > 3:
            i = n_nodes - 2
            S = S.at[i, i-2].set(1.0 * coeff)
            S = S.at[i, i-1].set(-4.0 * coeff)
            S = S.at[i, i].set(5.0 * coeff)  # modified for free BC
            S = S.at[i, i+1].set(-2.0 * coeff)

        # Node n-1 (free end: M=0, dM/dx=0)
        i = n_nodes - 1
        if n_nodes > 2:
            S = S.at[i, i-2].set(1.0 * coeff) if n_nodes > 2 else None
            S = S.at[i, i-1].set(-2.0 * coeff)
            S = S.at[i, i].set(1.0 * coeff)

    return S


@stability(StabilityLevel.EXPERIMENTAL)
class FlexibleBodyNode(MimeNode):
    """Euler-Bernoulli beam for flagellar/compliant microrobots.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    n_nodes : int
        Number of discretisation nodes along the filament.
    length_m : float
        Filament length [m].
    bending_stiffness_nm2 : float
        Bending stiffness K = EI [N.m^2].
    drag_coeff_perp : float
        Perpendicular RFT drag coefficient xi_perp [Pa.s].
        Used as analytical fallback; replaced by IB-LBM in Phase 2+.

    Boundary Inputs
    ---------------
    actuation_moment : scalar
        Bending moment applied at the proximal end (x=0) [N.m].
    fluid_load : (n_nodes,)
        Distributed fluid force per unit length [N/m] from IB-LBM.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-006",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Euler-Bernoulli beam dynamics for flexible flagellar microrobots",
        governing_equations=(
            r"K * d^4y/dx^4 = -xi_perp * dy/dt + f_fluid(x,t); "
            r"K = EI (bending stiffness)"
        ),
        discretization=(
            "4th-order central FD stencil [1,-4,6,-4,1] for d^4y/dx^4; "
            "implicit Euler for time integration (stiff system)"
        ),
        assumptions=(
            "Small-amplitude transverse deflections (linearised beam theory)",
            "Inertia negligible (overdamped, low Re)",
            "Uniform material properties along filament",
            "Clamped-free boundary conditions (clamped at head, free at tail)",
        ),
        limitations=(
            "Small-deformation only — fails for large curvatures (Sp >> 2.1)",
            "1D transverse deflection — no torsion or 3D shape",
            "RFT drag coefficient is a scalar approximation",
        ),
        validated_regimes=(
            ValidatedRegime("sperm_number", 0.5, 3.0, "",
                            "Optimal propulsion near Sp~2.1"),
        ),
        references=(
            Reference("Lighthill1976", "Flagellar Hydrodynamics"),
        ),
        hazard_hints=(
            "Sp >> 2.1 causes exponential wave decay — tail becomes stationary",
            "Explicit Euler with 4th-order stiffness requires dt < dx^4/(K/xi) — "
            "use implicit integration",
        ),
        implementation_map={
            "K*d^4y/dx^4": "mime.nodes.robot.flexible_body.build_beam_stiffness_matrix",
            "Implicit Euler solve": "mime.nodes.robot.flexible_body.FlexibleBodyNode.update",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ROBOT_BODY,
        biocompatibility=BiocompatibilityMeta(
            materials=("SU-8",),
            iso_10993_class=BiocompatibilityClass.NOT_ASSESSED,
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        n_nodes: int = 20,
        length_m: float = 100e-6,
        bending_stiffness_nm2: float = 4e-21,
        drag_coeff_perp: float = 0.0,
        **kwargs,
    ):
        # Compute default RFT drag if not provided
        if drag_coeff_perp <= 0:
            # Approximate: xi_perp ~ 4*pi*mu / (ln(2*L/d) + 0.5)
            # For a 100um filament of 5um diameter in CSF:
            drag_coeff_perp = 4 * jnp.pi * 8.5e-4 / (jnp.log(2 * length_m / 5e-6) + 0.5)

        super().__init__(
            name, timestep,
            n_nodes=n_nodes,
            length_m=length_m,
            bending_stiffness_nm2=bending_stiffness_nm2,
            drag_coeff_perp=float(drag_coeff_perp),
            **kwargs,
        )

    def initial_state(self) -> dict:
        n = self.params["n_nodes"]
        return {
            "deflection": jnp.zeros(n),      # y(x) [m]
            "velocity": jnp.zeros(n),         # dy/dt [m/s]
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        n = self.params["n_nodes"]
        return {
            "actuation_moment": BoundaryInputSpec(
                shape=(), default=0.0,
                description="Bending moment at proximal end [N.m]",
            ),
            "fluid_load": BoundaryInputSpec(
                shape=(n,), coupling_type="additive",
                description="Distributed fluid force [N/m] from IB-LBM",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        n = self.params["n_nodes"]
        L = self.params["length_m"]
        K = self.params["bending_stiffness_nm2"]
        xi = self.params["drag_coeff_perp"]
        dx = L / max(n - 1, 1)

        y = state["deflection"]
        M_act = boundary_inputs.get("actuation_moment", 0.0)
        f_fluid = boundary_inputs.get("fluid_load", jnp.zeros(n))

        # Build stiffness matrix
        S = build_beam_stiffness_matrix(n, dx, K)

        # Implicit Euler: (xi*I + dt*S) @ y_new = xi * y_old + dt * (f_fluid + M_forcing)
        # M_forcing: applied moment at x=0 creates a curvature source
        M_forcing = jnp.zeros(n)
        # Moment at x=0: second derivative source term
        if n > 2:
            M_forcing = M_forcing.at[1].set(M_act / dx**2)

        A = xi * jnp.eye(n) + dt * S
        rhs = xi * y + dt * (f_fluid + M_forcing)

        # Clamp node 0 (y=0 at anchor)
        A = A.at[0, :].set(0.0)
        A = A.at[0, 0].set(1.0)
        rhs = rhs.at[0].set(0.0)

        y_new = jnp.linalg.solve(A, rhs)
        v_new = (y_new - y) / dt

        return {
            "deflection": y_new,
            "velocity": v_new,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "deflection": state["deflection"],
            "velocity": state["velocity"],
        }
