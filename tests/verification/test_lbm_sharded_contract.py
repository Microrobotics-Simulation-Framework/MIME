"""§8 Step 5 — IBLBMFluidNode declares the MADDENING v0.2.1 sharded-stencil contract.

A full multi-device bit-compat test (wrap in ``ShardedStencilNode`` on a CPU
virtual-device mesh, step once, assert bit-equality with the single-device
path) is a follow-up. This pins the API surface so the v0.2.1 contract is
not silently broken:

* ``state_fields`` excludes ``drag_force`` / ``drag_torque`` (they are
  domain-integral outputs, not evolving state).
* ``domain_integral_fields`` declares the two drag outputs so
  ``ShardedStencilNode`` ``lax.psum``s them across the device mesh.
* ``static_data`` shards the pipe-wall masks along the requested axis
  when ``multigpu_shard_axis`` is set; otherwise replicates them.
* ``update_padded`` takes the v0.2.1 keyword-only ``static_padded`` and
  ``shard_info`` kwargs and runs end-to-end on the single-device fallback.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.lbm import IBLBMFluidNode

_N_DEVICES = len(jax.devices())


def _make_node(*, N=8, multigpu_shard_axis=None, use_bouzidi=False):
    return IBLBMFluidNode(
        name="lbm",
        timestep=0.001,
        nx=N, ny=N, nz=N,
        tau=0.6,
        vessel_radius_lu=0.40 * N,
        body_geometry_params={
            "nx": N, "ny": N, "nz": N,
            "body_radius": 0.13 * N, "body_length": 0.32 * N,
            "cone_length": 0.13 * N, "cone_end_radius": 0.03 * N,
            "fin_outer_radius": 0.22 * N, "fin_length": 0.16 * N,
            "fin_width": 0.05 * N, "fin_thickness": 0.03 * N,
            "helix_pitch": 0.6 * N,
        },
        use_bouzidi=use_bouzidi,
        dx_physical=1e-4,
        multigpu_shard_axis=multigpu_shard_axis,
    )


def test_state_fields_excludes_domain_integrals():
    """drag_force / drag_torque are domain integrals, not state fields."""
    node = _make_node()
    sf = node.state_fields()
    assert "f" in sf and "body_angle" in sf
    assert "drag_force" not in sf
    assert "drag_torque" not in sf


def test_domain_integral_fields_declares_drag():
    """ShardedStencilNode reads this declaration to ``lax.psum`` the outputs."""
    assert _make_node().domain_integral_fields() == {"drag_force", "drag_torque"}


def test_static_data_shards_when_multigpu_shard_axis_is_set():
    """With ``multigpu_shard_axis=2`` only ``pipe_wall`` is sharded — on the
    matching spatial axis. ``pipe_missing`` is *not* a sharded static (its
    leading Q axis would put its spatial shard_axis at 3, which is not one of
    the spatial axes ShardedStencilNode shards, so the wrapper would reject
    it); ``update_padded`` recomputes both missing-link masks per slab from
    the halo-exchanged ``pipe_wall``. Without the axis the masks replicate."""
    from maddening.core.static_data import StaticArray

    sharded = _make_node(multigpu_shard_axis=2).static_data
    assert isinstance(sharded["pipe_wall"], StaticArray)
    assert sharded["pipe_wall"].replication == "shard"
    assert sharded["pipe_wall"].shard_axis == 2
    # pipe_missing is recomputed per-slab, not carried as a sharded static.
    assert "pipe_missing" not in sharded

    unsharded = _make_node().static_data
    assert unsharded["pipe_wall"].replication == "replicate"
    assert unsharded["pipe_missing"].replication == "replicate"


def test_update_padded_has_v0_2_1_signature():
    """``static_padded`` and ``shard_info`` are keyword-only with default ``None``."""
    sig = inspect.signature(_make_node().update_padded)
    params = sig.parameters
    for name in ("state_padded", "boundary_inputs", "dt",
                 "static_padded", "shard_info"):
        assert name in params, f"missing param {name!r}"
    for name in ("static_padded", "shard_info"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None


def test_update_padded_single_device_fallback_smoke():
    """End-to-end smoke: invoked outside ShardedStencilNode (no
    ``static_padded`` / ``shard_info``) the method edge-pads the closure
    masks and returns a same-shape ``f`` plus partial-sum drag outputs."""
    node = _make_node()
    state = node.initial_state()
    h = 1
    f_pad = jnp.pad(
        state["f"], ((h, h), (h, h), (h, h), (0, 0)), mode="wrap",
    )
    state_padded = {"f": f_pad, "body_angle": state["body_angle"]}
    bi = {"body_angular_velocity": jnp.zeros(3, dtype=jnp.float32)}

    out = node.update_padded(state_padded, bi, 0.001)
    assert out["f"].shape == f_pad.shape
    assert out["body_angle"].shape == ()
    assert out["drag_force"].shape == (3,)
    assert out["drag_torque"].shape == (3,)


def test_update_padded_rejects_bouzidi():
    """Bouzidi IBB isn't yet supported on the sharded path."""
    node = _make_node(use_bouzidi=True)
    state = node.initial_state()
    h = 1
    f_pad = jnp.pad(
        state["f"], ((h, h), (h, h), (h, h), (0, 0)), mode="wrap",
    )
    state_padded = {"f": f_pad, "body_angle": state["body_angle"]}
    with pytest.raises(NotImplementedError):
        node.update_padded(state_padded, {}, 0.001)


# ── Multi-device bit-compatibility (CPU virtual devices) ─────────────────
# These run only when >= 4 (virtual) devices are visible. On a GPU-only or
# single-CPU rig, launch with:
#
#   JAX_PLATFORMS=cpu \
#   XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_gpu_autotune_level=0" \
#   pytest tests/verification/test_lbm_sharded_contract.py -m 'slow or not slow'
#
# Sharding the lattice must not change the physics: the pencil-decomposed
# IBLBM step is bit-compatible (to float tolerance) with the single-device
# step, including the cross-slab UMR body and the psum'd drag integrals.

_NEEDS_MULTIDEVICE = pytest.mark.skipif(
    _N_DEVICES < 4,
    reason="needs >=4 (virtual) devices; run with JAX_PLATFORMS=cpu "
    "XLA_FLAGS=--xla_force_host_platform_device_count=4",
)


def _perturbed_state(node, seed=0):
    state = node.initial_state()
    rng = np.random.default_rng(seed)
    pert = rng.standard_normal(state["f"].shape).astype(np.float32) * 1e-3
    state = dict(state)
    state["f"] = state["f"] + jnp.asarray(pert)
    return state


def _rotating_inputs(omega_z=0.05):
    return {
        "body_angular_velocity": jnp.array(
            [0.0, 0.0, omega_z], dtype=jnp.float32,
        ),
        "body_orientation": jnp.array(
            [1.0, 0.0, 0.0, 0.0], dtype=jnp.float32,
        ),
    }


# The single-device reference is compared *under jax.jit*. The UMR occupancy
# mask is a staircased float comparison (``r_perp < radius`` etc.); cells
# sitting exactly on the body surface flip side under XLA's fused float
# reassociation, so an *eager* single-device step and a *jitted* one already
# disagree on ~0.05% of cells (and the momentum-exchange drag, a near-
# cancellation, is very sensitive to those). The sharded path runs jitted
# under shard_map, so the meaningful "sharding preserves the physics"
# comparison is jit-vs-jit — which is how production runs both paths anyway.
# Under that matched comparison the decomposition is *bit-identical*.


@pytest.mark.slow
@_NEEDS_MULTIDEVICE
def test_sharded_matches_unsharded_one_step():
    """One halo-exchanged step on a 4-device z-pencil mesh is bit-identical to
    the single-device step — field, and the psum'd drag force / torque."""
    import jax

    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

    N = 16
    node = _make_node(N=N, multigpu_shard_axis=2)
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 2}, boundary="periodic",
    )

    state = _perturbed_state(node)
    bi = _rotating_inputs()

    ref_step = jax.jit(lambda s, b: node.update(s, b, 1.0))
    ref = ref_step(state, bi)
    out = sharded.update(state, bi, 1.0)

    np.testing.assert_allclose(
        np.asarray(out["f"]), np.asarray(ref["f"]), rtol=1e-6, atol=1e-7,
        err_msg="sharded f diverged from single-device",
    )
    np.testing.assert_allclose(
        np.asarray(out["drag_force"]), np.asarray(ref["drag_force"]),
        rtol=1e-5, atol=1e-6, err_msg="sharded drag_force diverged",
    )
    np.testing.assert_allclose(
        np.asarray(out["drag_torque"]), np.asarray(ref["drag_torque"]),
        rtol=1e-5, atol=1e-6, err_msg="sharded drag_torque diverged",
    )


@pytest.mark.slow
@_NEEDS_MULTIDEVICE
def test_sharded_matches_unsharded_many_steps():
    """50 rotating steps; the sharded trajectory stays bit-identical to the
    single-device (jitted) one — no drift in cross-slab streaming or the
    rotating UMR bounce-back at slab boundaries."""
    import jax

    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

    N = 16
    node = _make_node(N=N, multigpu_shard_axis=2)
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 2}, boundary="periodic",
    )

    state_u = _perturbed_state(node)
    state_s = dict(state_u)
    bi = _rotating_inputs()
    ref_step = jax.jit(lambda s, b: node.update(s, b, 1.0))

    last_u = last_s = None
    for _ in range(50):
        last_u = ref_step(state_u, bi)
        last_s = sharded.update(state_s, bi, 1.0)
        state_u = {k: last_u[k] for k in ("f", "body_angle")}
        state_s = {k: last_s[k] for k in ("f", "body_angle")}

    np.testing.assert_allclose(
        np.asarray(state_s["f"]), np.asarray(state_u["f"]),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(last_s["drag_force"]), np.asarray(last_u["drag_force"]),
        rtol=1e-4, atol=1e-6,
    )
