"""The contact's conformance suites (``mime.testing.friction_law_suite``, ``normal_law_suite``, ``row_kind_suite``) on
the default implementations (``Coulomb``, ``Hold()``, the sliding rows), in both dtypes, with the controls that fail them
on every draw where friction acts (decided Fr6: every implementation of a seam passes its suite and its own oracle):

- a sign flip of friction (the pair's force negated) is caught by the power guard in every such draw, on the sphere
  and on the cylinder held at its rims, both dtypes (the spike's control, which the kinetic energy missed in most
  draws); ``W2B_PLANT=flip`` plants it for ``.github/probe/detection_rate.py``;
- the reserved row kinds (rolling resistance, spin) are refused by the contact with their reasons, and an adhesion
  written into ``Hold`` is refused as reserved.

And the extremes the suites' draws leave out, on purpose: a bound at rounding (a coefficient of 1e-30) and a normal
force of zero (a point free of the wall) give a zero traction and no non-finite value, either dtype, in value and in
reverse mode (the spike's float32 defect, ``0 x inf``)."""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.interactions import Contact
from mime.interactions.friction import ADHESION_RESERVED, ROW_KINDS, Coulomb, Hold
from mime.nodes.body._friction import disk_update
from mime.testing import friction_law_suite, normal_law_suite, row_kind_suite
from mime.testing._friction import held_node


@pytest.fixture(params=[jnp.float64, jnp.float32], ids=["float64", "float32"])
def dtype(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == jnp.float64)
    yield request.param
    jax.config.update("jax_enable_x64", held)


def flipped(force):
    return -force


def test_coulomb_passes_the_friction_law_suite(dtype):
    planted = os.environ.get("W2B_PLANT") == "flip"
    result = friction_law_suite(Coulomb(mu=0.3), plant=flipped if planted else None)
    print(f"\n{result.detail}")
    assert result.status == "PASS", result.detail


def test_control_a_sign_flip_of_friction_fails_every_draw_where_friction_acts(dtype):
    result = friction_law_suite(Coulomb(mu=0.3), plant=flipped, max_examples=100)
    print(f"\n{result.detail}")
    assert result.status == "FAIL"
    failed, judged, acted = (int(word) for word in _counts(result.detail))
    assert acted > 0 and failed == acted, result.detail


def _counts(detail: str) -> tuple[str, str, str]:
    """The failed, judged and with-friction counts of a suite's failing detail."""
    import re

    found = re.search(r"failed on (\d+) of (\d+) draws \((\d+) with friction\)", detail)
    assert found is not None, detail
    return found.group(1), found.group(2), found.group(3)


def test_hold_passes_the_normal_law_suite(dtype):
    result = normal_law_suite(Hold())
    print(f"\n{result.detail}")
    assert result.status == "PASS", result.detail


@pytest.mark.parametrize("kind", sorted(ROW_KINDS))
def test_each_row_kind_passes_its_suite(dtype, kind):
    result = row_kind_suite(kind)
    print(f"\n{result.detail}")
    assert result.status == "PASS", result.detail


def test_an_adhesion_is_refused_as_reserved():
    from mime.units import N

    contact = Contact.Projection(margin=1e-4, normal=Hold(adhesion=N(1e-6)), friction=Coulomb(mu=0.2))
    assert ADHESION_RESERVED in contact.setting_problems({"margin": 1e-4})


def test_a_coefficient_out_of_range_is_refused():
    assert Contact.Projection(margin=1e-4, friction=Coulomb(mu=-0.1)).setting_problems({"margin": 1e-4})
    assert Contact.Projection(margin=1e-4, friction=Coulomb(mu=math.inf)).setting_problems({"margin": 1e-4})


@pytest.mark.parametrize("bound", [0.0, 1e-30])
def test_a_bound_at_rounding_or_zero_gives_no_traction_and_nothing_non_finite(dtype, bound):
    """The extremes the draws leave out: a disk of radius zero (a point with no normal force) or at rounding; the
    force is zero to the bound's size, finite, and so is its derivative in the load and in the bound."""
    block = jnp.asarray([[2.0, 0.3], [0.3, 1.5]], dtype=dtype) * 1e3
    load = jnp.asarray([1e-3, -2e-3], dtype=dtype)
    held = jnp.asarray(bound, dtype=dtype)
    force = disk_update(block, load, held)
    assert np.all(np.isfinite(np.asarray(force))) and float(jnp.max(jnp.abs(force))) <= 2 * bound
    for derivative in jax.grad(lambda l, b: jnp.sum(disk_update(block, l, b)), argnums=(0, 1))(load, held):
        assert np.all(np.isfinite(np.asarray(derivative)))


def test_a_free_point_gives_no_traction_and_a_finite_derivative(dtype):
    """A sphere a micrometre above the plane, sliding: no normal force, no friction, and reverse mode through its step in
    the coefficient finite (zero)."""
    node = held_node("sphere")
    leaves = node.params_pytree()
    state = {**node.initial_state(), "pose": jnp.asarray([0.0, 0.0, 1e-3 + 1e-6, 1.0, 0.0, 0.0, 0.0], dtype),
             "twist": jnp.asarray([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype)}
    inputs = {"applied_wrench": jnp.zeros(6, dtype), "resistance": jnp.zeros((6, 6), dtype),
              "background_load": jnp.zeros(6, dtype)}

    def speed(mu):
        return node.update(state, inputs, 1e-3, params={**leaves, "friction_mu": mu})["twist"][0]

    derivative = jax.grad(speed)(jnp.asarray(0.3, dtype))
    assert float(derivative) == 0.0
    assert float(jnp.max(jnp.abs(node.update(state, inputs, 1e-3)["contact_traction"]))) == 0.0
