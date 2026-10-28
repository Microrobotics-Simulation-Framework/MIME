"""The drag's surrogate on its tube's axis in float32, the precision CI's default lane runs in (``tests/oracles``
checks the same in float64): the resistance and its derivative with respect to the body's position are finite on the
axis and 1e-30 m from it. The control: the 0.2 line's ``sqrt(x^2 + y^2)`` and ``arctan2(y, x)`` give NaN there."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydro_oracles import PortFrameNode

from mime.data import asset
from mime.nodes.hydro import SurrogateDragNode


def node(axis, cls=SurrogateDragNode):
    return cls("drag", 5e-4, weights="dejongh_mlp_v2", weights_sha256=asset("dejongh_mlp_v2").sha256, viscosity=1e-3,
               helix_wavenumber=2.33, length=7.47e-3, radius=1.56e-3, outer_radius=1.33 * 1.56e-3,
               vessel_shape="cylinder", vessel_centre=(0.0, 0.0, 0.0), vessel_axis=axis,
               vessel_radius=3.175e-3, lubrication_epsilon=1.5e-4)


def value_and_gradient(n, position):
    def scalar(at):
        out = n.update(n.initial_state(), {"pose": jnp.concatenate([at, jnp.asarray([1.0, 0.0, 0.0, 0.0])])}, 5e-4)
        return jnp.sum(out["resistance"] * jnp.arange(36.0).reshape(6, 6))

    held = jnp.asarray(position, dtype=jnp.float32)
    return np.asarray(scalar(held)), np.asarray(jax.grad(scalar)(held))


@pytest.mark.parametrize("offset", [0.0, 1e-30], ids=["on the axis", "1e-30 m off it"])
def test_on_the_axis_the_value_and_its_derivative_are_finite_in_float32(offset):
    assert jnp.zeros(()).dtype == jnp.float32
    value, gradient = value_and_gradient(node((1.0, 0.0, 0.0)), (0.012, offset, 0.0))
    assert np.isfinite(value) and np.all(np.isfinite(gradient))


def test_the_02_lines_formula_gives_nan_on_the_axis_in_float32():
    _, gradient = value_and_gradient(node((0.0, 0.0, 1.0), PortFrameNode), (0.0, 0.0, 0.012))
    assert np.any(np.isnan(gradient))
