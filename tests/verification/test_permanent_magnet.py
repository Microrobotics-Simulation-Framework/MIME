"""Verification benchmarks for ``PermanentMagnetNode``.

Three benchmarks:

- **MIME-VER-110**: dipole field vs analytical at $r \\in \\{10, 20, 50\\}\\,R_{\\text{magnet}}$,
  for all three field models. The ``point_dipole`` model must match the
  closed-form $\\mathbf{B}_{\\text{dipole}}$ to $<10^{-4}$ relative error in
  the far field. ``current_loop`` must match its own on-axis closed form
  $B_z = \\mu_0 I R^2 / (2 (R^2+z^2)^{3/2})$. ``coulombian_poles`` must
  match its own two-disc closed form.
- **MIME-VER-111**: dipole gradient (``jax.jacrev``) vs the analytical
  $\\nabla\\mathbf{B}$ formula at the same configurations.
- **MIME-VER-112**: Earth-field superposition: total field equals
  magnet-only $\\mathbf{B}$ plus ``earth_field_world_t`` exactly; flipping
  the Earth field gives a difference of exactly $2\\,\\mathbf{B}_{\\text{earth}}$.

Plus standard JAX-traceability tests (``jit``, ``grad``, ``vmap``) and a
coupling test that runs the node inside a ``GraphManager`` for one step.

The Earth-field background is set to **zero** for benchmarks 110 and 111
(the analytical references in those benchmarks are magnet-only). 112
explicitly exercises the superposition.
"""

from __future__ import annotations

import jax
# Enable double precision for analytical-tolerance verification.
# This affects only this test module's JAX traces.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)
from maddening.core.graph_manager import GraphManager

from mime.nodes.actuation.permanent_magnet import (
    PermanentMagnetNode, MU_0,
)


# ---------------------------------------------------------------------------
# Closed-form references (numpy, double precision)
# ---------------------------------------------------------------------------

MU_0_NP = 4.0 * np.pi * 1e-7  # T·m/A


def _b_dipole_np(r_xyz, m_source):
    """Point-dipole field at displacement r from source moment m."""
    r_xyz = np.asarray(r_xyz, dtype=np.float64)
    m_source = np.asarray(m_source, dtype=np.float64)
    r = np.linalg.norm(r_xyz)
    rhat = r_xyz / r
    return (MU_0_NP / (4.0 * np.pi)) * (
        3.0 * np.dot(m_source, rhat) * rhat - m_source
    ) / (r ** 3)


def _grad_b_dipole_np(r_xyz, m_source):
    """Analytical gradient of the point-dipole field.

    For B(r) = (μ₀/4π)·[3(m·r̂)r̂ − m] / r³, the gradient
    ∂B_i/∂x_j is

        (μ₀/4π) · 1/r⁵ · [
            3 (m_i r_j + m_j r_i + (m·r) δ_ij)
          − 15 (m·r) r_i r_j / r²
        ].
    """
    r_xyz = np.asarray(r_xyz, dtype=np.float64)
    m = np.asarray(m_source, dtype=np.float64)
    r = np.linalg.norm(r_xyz)
    mr = np.dot(m, r_xyz)
    pref = MU_0_NP / (4.0 * np.pi) / r ** 5
    grad = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            term1 = 3.0 * (m[i] * r_xyz[j] + m[j] * r_xyz[i])
            term2 = 3.0 * mr * (1.0 if i == j else 0.0)
            term3 = -15.0 * mr * r_xyz[i] * r_xyz[j] / r ** 2
            grad[i, j] = pref * (term1 + term2 + term3)
    return grad


def _b_loop_on_axis_np(z, m_norm, R_loop):
    """Closed-form on-axis field of a circular current loop, |m| = I·πR²."""
    Bz = (MU_0_NP * m_norm) / (2.0 * np.pi * (R_loop ** 2 + z ** 2) ** 1.5)
    return np.array([0.0, 0.0, Bz])


def _b_coulomb_on_axis_np(z, m_norm, R, L):
    """Closed-form on-axis field of a uniformly magnetised cylinder."""
    M_bulk = m_norm / (np.pi * R ** 2 * L)
    zp = z + 0.5 * L
    zm = z - 0.5 * L
    Bz = 0.5 * MU_0_NP * M_bulk * (
        zp / np.sqrt(R ** 2 + zp ** 2)
        - zm / np.sqrt(R ** 2 + zm ** 2)
    )
    return np.array([0.0, 0.0, Bz])


# ---------------------------------------------------------------------------
# Test fixtures: magnet sized for clean far-field arithmetic
# ---------------------------------------------------------------------------

R_MAGNET = 1e-3      # 1 mm radius
L_MAGNET = 2e-3      # 2 mm length
DIPOLE_M = 1.0       # 1 A·m²

# Magnet pose: origin, identity quaternion (=> moment along +z)
IDENTITY_POSE = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# Far-field distances measured in multiples of R_magnet
FAR_FIELD_DISTANCES_M = [
    10 * R_MAGNET,   # z = 10 mm — dipole-loop disagreement < 1 %
    20 * R_MAGNET,   # z = 20 mm
    50 * R_MAGNET,   # z = 50 mm
]


def _make_node(field_model: str, earth_field=(0.0, 0.0, 0.0)):
    """Construct a PermanentMagnetNode with the supplied model + Earth field."""
    return PermanentMagnetNode(
        f"pm_{field_model}", timestep=1e-3,
        dipole_moment_a_m2=DIPOLE_M,
        magnet_radius_m=R_MAGNET,
        magnet_length_m=L_MAGNET,
        magnetization_axis_in_body=(0.0, 0.0, 1.0),
        field_model=field_model,
        earth_field_world_t=tuple(earth_field),
    )


def _eval_node(node, target_xyz):
    """Run one ``update`` and return (B, grad_B) as numpy arrays."""
    state = node.initial_state()
    bi = {
        "magnet_pose_world": IDENTITY_POSE,
        "target_position_world": jnp.asarray(target_xyz),
        "amplitude_scale": jnp.asarray(1.0),
    }
    new_state = node.update(state, bi, 1e-3)
    B = np.asarray(new_state["field_vector"], dtype=np.float64)
    G = np.asarray(new_state["field_gradient"], dtype=np.float64)
    return B, G


# ---------------------------------------------------------------------------
# MIME-VER-110: dipole field vs analytical, all three models
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-110",
    description=(
        "PermanentMagnetNode field vs analytical at r ∈ {10, 20, 50}·R_magnet "
        "for all three field models (point_dipole, current_loop, "
        "coulombian_poles). Each model must match its own closed-form "
        "reference to <1e-4 relative error in the far field."
    ),
    node_type="PermanentMagnetNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "|B_node - B_analytic| / |B_analytic| < 1e-4 at r = 10, 20, 50 R_magnet, "
        "all three field models, on-axis."
    ),
    references=("Jackson1998",),
)
def test_ver110_dipole_field_far_field():
    m_source = np.array([0.0, 0.0, DIPOLE_M])

    for model in ("point_dipole", "current_loop", "coulombian_poles"):
        node = _make_node(model)
        for z in FAR_FIELD_DISTANCES_M:
            target = np.array([0.0, 0.0, z])
            B_node, _ = _eval_node(node, target)

            if model == "point_dipole":
                B_ref = _b_dipole_np(target, m_source)
            elif model == "current_loop":
                B_ref = _b_loop_on_axis_np(z, DIPOLE_M, R_MAGNET)
            else:  # coulombian_poles
                B_ref = _b_coulomb_on_axis_np(z, DIPOLE_M, R_MAGNET, L_MAGNET)

            rel = np.linalg.norm(B_node - B_ref) / np.linalg.norm(B_ref)
            assert rel < 1e-4, (
                f"{model} at z={z*1e3:.1f} mm: rel={rel:.3e}, "
                f"B_node={B_node}, B_ref={B_ref}"
            )


# Independent off-axis check for the point-dipole model: the dipole
# formula does not depend on whether the target is on-axis, so we
# verify a fully 3D off-axis configuration here as well.

def test_ver110_dipole_field_off_axis():
    """Off-axis sanity for point_dipole — magnitude of test grid spans
    several decades to catch sign / coordinate-frame errors."""
    node = _make_node("point_dipole")
    m_source = np.array([0.0, 0.0, DIPOLE_M])
    targets = [
        np.array([3e-3, 0.0, 5e-3]),
        np.array([3e-3, 4e-3, 5e-3]),
        np.array([0.0, 50e-3, 0.0]),
    ]
    for t in targets:
        B_node, _ = _eval_node(node, t)
        B_ref = _b_dipole_np(t, m_source)
        rel = np.linalg.norm(B_node - B_ref) / np.linalg.norm(B_ref)
        assert rel < 1e-4, f"Off-axis dipole rel={rel:.3e} at t={t}"


# ---------------------------------------------------------------------------
# MIME-VER-111: dipole gradient (jax.jacrev) vs analytical ∇B
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-111",
    description=(
        "PermanentMagnetNode gradient via jax.jacrev vs analytical ∇B "
        "formula at r = 10, 20, 50 R_magnet on-axis and at three off-axis "
        "configurations. <1e-4 relative error per component (or absolute "
        "< 1e-12 T/m for near-zero components)."
    ),
    node_type="PermanentMagnetNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "|grad_B_node - grad_B_analytic| / |grad_B_analytic| < 1e-4 "
        "(or abs < 1e-12 for near-zero components) at all 6 test points."
    ),
    references=("Jackson1998",),
)
def test_ver111_dipole_gradient():
    node = _make_node("point_dipole")
    m_source = np.array([0.0, 0.0, DIPOLE_M])

    # Combined on-axis + off-axis configurations
    targets = []
    for z in FAR_FIELD_DISTANCES_M:
        targets.append(np.array([0.0, 0.0, z]))
    targets.append(np.array([3e-3, 0.0, 5e-3]))
    targets.append(np.array([3e-3, 4e-3, 5e-3]))
    targets.append(np.array([0.0, 50e-3, 0.0]))

    for t in targets:
        _, G_node = _eval_node(node, t)
        G_ref = _grad_b_dipole_np(t, m_source)

        abs_err = np.abs(G_node - G_ref)
        denom = np.maximum(np.abs(G_ref), 1e-15)
        rel_err = abs_err / denom

        ok = (abs_err < 1e-12) | (rel_err < 1e-4)
        assert ok.all(), (
            f"at t={t}\n"
            f"G_node=\n{G_node}\nG_ref=\n{G_ref}\n"
            f"abs_err=\n{abs_err}\nrel_err=\n{rel_err}"
        )


# ---------------------------------------------------------------------------
# MIME-VER-112: Earth-field superposition exactness
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-112",
    description=(
        "Total field = α·B_magnet + B_earth exactly; flipping B_earth "
        "changes the total by exactly 2·B_earth (differential check that "
        "the magnet contribution is otherwise unchanged)."
    ),
    node_type="PermanentMagnetNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "B_total_with_earth - B_total_without_earth == B_earth (atol 1e-12); "
        "B_total(+earth) - B_total(-earth) == 2·B_earth (atol 1e-12)."
    ),
    references=(),
)
def test_ver112_earth_field_superposition():
    earth = np.array([2e-5, 1e-5, -4.5e-5])  # asymmetric, all three components nonzero

    target = np.array([0.0, 0.0, 10 * R_MAGNET])

    node_no_earth   = _make_node("point_dipole", earth_field=(0.0, 0.0, 0.0))
    node_with_earth = _make_node("point_dipole", earth_field=tuple(earth))
    node_neg_earth  = _make_node("point_dipole", earth_field=tuple(-earth))

    B_no, _   = _eval_node(node_no_earth,   target)
    B_pos, _  = _eval_node(node_with_earth, target)
    B_neg, _  = _eval_node(node_neg_earth,  target)

    # Additive superposition exactly:
    diff_pos = B_pos - B_no
    np.testing.assert_allclose(diff_pos, earth, atol=1e-12, rtol=0.0)

    # Flipping Earth flips the residual by exactly 2·B_earth:
    diff_flip = B_pos - B_neg
    np.testing.assert_allclose(diff_flip, 2.0 * earth, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# JAX-traceability tests: jit, grad, vmap
# ---------------------------------------------------------------------------

class TestPermanentMagnetJAX:
    def test_jit_traceable(self):
        node = _make_node("point_dipole")
        state = node.initial_state()
        bi = {
            "magnet_pose_world":     IDENTITY_POSE,
            "target_position_world": jnp.array([0.0, 0.0, 10 * R_MAGNET]),
            "amplitude_scale":       jnp.array(1.0),
        }
        jitted = jax.jit(node.update)
        new_state = jitted(state, bi, 1e-3)
        assert new_state["field_vector"].shape == (3,)
        assert new_state["field_gradient"].shape == (3, 3)
        assert jnp.isfinite(new_state["field_vector"]).all()
        assert jnp.isfinite(new_state["field_gradient"]).all()

    def test_grad_wrt_amplitude(self):
        """d|B - B_earth|/d(amplitude) at α = 1 must equal |B_magnet| —
        i.e. amplitude scales the magnet contribution linearly."""
        node = _make_node("point_dipole")
        state = node.initial_state()
        target_z = 10 * R_MAGNET

        def field_magnitude_minus_earth(alpha):
            bi = {
                "magnet_pose_world":     IDENTITY_POSE,
                "target_position_world": jnp.array([0.0, 0.0, target_z]),
                "amplitude_scale":       alpha,
            }
            B = node.update(state, bi, 1e-3)["field_vector"]
            # subtract default Earth field then take magnitude — this gives
            # alpha · |B_magnet|, whose derivative wrt alpha is |B_magnet|.
            B_earth = jnp.asarray(node.params["earth_field_world_t"])
            return jnp.linalg.norm(B - B_earth)

        g = float(jax.grad(field_magnitude_minus_earth)(jnp.array(1.0)))
        # At α=1 along +z axis with m=1 A·m², |B_magnet| = (μ₀/4π)·2/r³
        B_mag_expected = (MU_0_NP / (4 * np.pi)) * 2.0 / target_z ** 3
        assert abs(g - B_mag_expected) < 1e-7, (
            f"grad={g}, expected {B_mag_expected}"
        )

    def test_vmap_over_target_positions(self):
        """vmap-batched evaluation over an array of target points."""
        node = _make_node("point_dipole")
        state = node.initial_state()
        targets = jnp.array([
            [0.0, 0.0, 10 * R_MAGNET],
            [0.0, 0.0, 20 * R_MAGNET],
            [0.0, 0.0, 50 * R_MAGNET],
            [3e-3, 4e-3, 5e-3],
        ])

        def run_at_target(t):
            bi = {
                "magnet_pose_world":     IDENTITY_POSE,
                "target_position_world": t,
                "amplitude_scale":       jnp.array(1.0),
            }
            return node.update(state, bi, 1e-3)["field_vector"]

        results = jax.vmap(run_at_target)(targets)
        assert results.shape == (4, 3)
        assert jnp.isfinite(results).all()

        # Cross-check the first sample against the closed form.
        m_source = np.array([0.0, 0.0, DIPOLE_M])
        B_ref0 = _b_dipole_np(np.asarray(targets[0]), m_source)
        # Account for Earth field added by the node:
        B_earth = np.asarray(node.params["earth_field_world_t"])
        B_node0 = np.asarray(results[0]) - B_earth
        rel = np.linalg.norm(B_node0 - B_ref0) / np.linalg.norm(B_ref0)
        assert rel < 1e-4


# ---------------------------------------------------------------------------
# Coupling test: run the node inside a GraphManager for one step
# ---------------------------------------------------------------------------

def test_graph_manager_one_step():
    """Standalone GraphManager wiring: PermanentMagnetNode is added,
    fed a non-degenerate target via ``add_external_input``, and produces
    finite B and ∇B after one step.

    The default ``target_position_world`` is the origin, which sits at
    the dipole singularity (target == magnet_pos). Driving the target
    away from origin via an external input checks that the boundary-
    input plumbing reaches the node correctly."""
    gm = GraphManager()
    node = _make_node("point_dipole")
    gm.add_node(node)

    # Register external inputs so the node is not "disconnected".
    gm.add_external_input(
        node.name, "target_position_world",
        shape=(3,), dtype=jnp.float64,
    )

    target = jnp.array([0.0, 0.0, 10 * R_MAGNET], dtype=jnp.float64)
    gm.step(external_inputs={
        node.name: {"target_position_world": target},
    })

    state = gm._state[node.name]
    B = np.asarray(state["field_vector"])
    G = np.asarray(state["field_gradient"])

    assert B.shape == (3,)
    assert G.shape == (3, 3)
    assert np.isfinite(B).all()
    assert np.isfinite(G).all()

    # Cross-check the field magnitude against the closed form.
    m_source = np.array([0.0, 0.0, DIPOLE_M])
    B_ref = _b_dipole_np(np.asarray(target), m_source)
    B_earth = np.asarray(node.params["earth_field_world_t"])
    rel = np.linalg.norm(B - (B_ref + B_earth)) / np.linalg.norm(B_ref + B_earth)
    assert rel < 1e-4, f"Graph-manager B mismatch: rel={rel:.3e}"
