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

import jax.numpy as jnp
import pytest

from mime.nodes.environment.lbm import IBLBMFluidNode


def _make_node(*, multigpu_shard_axis=None, use_bouzidi=False):
    return IBLBMFluidNode(
        name="lbm",
        timestep=0.001,
        nx=8, ny=8, nz=8,
        tau=0.6,
        vessel_radius_lu=3.0,
        body_geometry_params={
            "nx": 8, "ny": 8, "nz": 8,
            "body_radius": 1.0, "body_length": 4.0,
            "cone_length": 1.5, "cone_end_radius": 0.3,
            "fin_outer_radius": 1.5, "fin_length": 2.0,
            "fin_width": 0.5, "fin_thickness": 0.1,
            "helix_pitch": 6.0,
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
    """With ``multigpu_shard_axis=2`` the masks declare ``replication='shard'``
    on the matching array axis; without it they replicate (single-device path)."""
    from maddening.core.static_data import StaticArray

    sharded = _make_node(multigpu_shard_axis=2).static_data
    assert isinstance(sharded["pipe_wall"], StaticArray)
    assert sharded["pipe_wall"].replication == "shard"
    assert sharded["pipe_wall"].shard_axis == 2
    assert sharded["pipe_missing"].replication == "shard"
    assert sharded["pipe_missing"].shard_axis == 3  # +1 for the leading Q axis

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
