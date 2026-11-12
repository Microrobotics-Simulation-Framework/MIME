"""The contact's conformance suites (``mime.testing.friction_law_suite``, ``normal_law_suite``, ``row_kind_suite``) on
the default implementations (``Coulomb``, ``Hold()``, the sliding rows), in both dtypes, with the controls that fail them
on every draw where friction acts (decided Fr6: every implementation of a seam passes its suite and its own oracle):

- a sign flip of friction (the pair's force negated) is flagged in every such draw of every body, the sphere, the
  cylinder held at its rims and a box held by its samples, both dtypes: by the power guard, or by ``unconverged``
  where the flip lifts the body off and leaves no friction to read (the spike's control, which the kinetic energy
  missed in most draws); the clean suite raises neither on any draw; ``W2B_PLANT=flip`` plants it for
  ``.github/probe/detection_rate.py``. Friction acts where the clean solve's friction is above the guard's line of the
  load; the draws under it (a load within about 1e-8 rad of the normal and no motion, where the solve's friction is
  1e-8 of the load and a flip of it moves the step below any check's resolution) are counted apart, never dropped;
- the reserved row kinds (rolling resistance, spin) are refused by the contact with their reasons, and an adhesion
  written into ``Hold`` is refused as reserved;
- a law the contact's solve does not hold (a lab's restitution, a hold that pulls, a friction limit of its own) is
  refused by the contact by name and fails its suite unrun, never judged as the default in its place.

And the extremes the suites' draws leave out, on purpose: a bound at rounding (a coefficient of 1e-30) and a normal
force of zero (a point free of the wall) give a zero traction and no non-finite value, either dtype, in value and in
reverse mode (the spike's float32 defect, ``0 x inf``)."""

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.interactions import Contact
from mime.interactions.friction import ADHESION_RESERVED, ROW_KINDS, Coulomb, Hold, NormalLaw
from mime.nodes.body._friction import disk_update
from mime.testing import friction_law_suite, normal_law_suite, rolling_resistance_suite, row_kind_suite
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
    assert all(flagged == 0 for flagged, _ in _by_body(result.detail).values()), result.detail


def test_control_a_sign_flip_of_friction_is_flagged_in_every_draw_of_every_body_where_friction_acts(dtype):
    """The planted sign flip raises the guard, or ``unconverged`` where it lifts the body off, in every draw of every
    body where friction acts (the clean solve's friction above the guard's line of the load); the clean suite (above)
    raises neither on any draw."""
    result = friction_law_suite(Coulomb(mu=0.3), plant=flipped, max_examples=100)
    print(f"\n{result.detail}")
    assert result.status == "FAIL"
    counts = _by_body(result.detail)
    assert set(counts) == {"sphere", "cylinder", "box"}, result.detail
    assert all(acted > 0 and flagged == acted for flagged, acted in counts.values()), result.detail


def _by_body(detail: str) -> dict[str, tuple[int, int]]:
    """The flagged and with-friction draws of each body, from a suite's detail."""
    import re

    found = re.search(r"flagged/with friction by body: ([^;]*)", detail)
    assert found is not None, detail
    return {name: (int(flagged), int(acted)) for name, flagged, acted in
            re.findall(r"(\w+) (\d+)/(\d+)", found.group(1))}


def test_rolling_resistance_is_passive_on_every_draw(dtype):
    """Rolling resistance dissipates, never injects (the review of #261, M2): the guard, the rolling pair's power
    included, and the residual quiet on every draw of the sphere and the cylinder on its side, a rolling resistance
    drawn with each; ``W2B_PLANT=rollflip`` plants the control here for ``.github/probe/detection_rate.py``."""
    planted = os.environ.get("W2B_PLANT") == "rollflip"
    result = rolling_resistance_suite(planted=planted)
    print(f"\n{result.detail}")
    assert result.status == "PASS", result.detail


def test_control_rolling_resistances_sign_flipped_is_flagged_on_every_draw_where_it_acts(dtype):
    """The control: the rolling pair's bound negated in the solve (its torque along the rotation it acts on) raises the
    guard or ``unconverged`` on every draw where rolling resistance acts, the clean step settled, on both bodies."""
    result = rolling_resistance_suite(planted=True, max_examples=100)
    print(f"\n{result.detail}")
    assert result.status == "FAIL", result.detail
    import re

    acted = dict(re.findall(r"(\w+) (\d+)", re.search(r"acted by body: ([^;]*)", result.detail).group(1)))
    flagged = dict(re.findall(r"(\w+) (\d+)", re.search(r"flagged by body: ([^;]*)", result.detail).group(1)))
    assert set(acted) == {"sphere", "cylinder"} and all(int(acted[b]) > 0 for b in acted), result.detail
    assert acted == flagged, result.detail


def test_hold_passes_the_normal_law_suite(dtype):
    result = normal_law_suite(Hold())
    print(f"\n{result.detail}")
    assert result.status == "PASS", result.detail


@pytest.mark.parametrize("kind", sorted(ROW_KINDS))
def test_each_row_kind_passes_its_suite(dtype, kind):
    result = row_kind_suite(kind)
    print(f"\n{result.detail}")
    assert result.status == "PASS", result.detail


@dataclass(frozen=True, kw_only=True)
class Bouncy(NormalLaw):
    """A lab's restitution law (the review's plant): a normal law the contact's solve does not hold."""

    group: ClassVar[str] = "bouncy"
    restitution: float = 0.5

    def least(self) -> float:
        return 0.0

    def load(self, normal: float) -> float:
        return normal


@dataclass(frozen=True, kw_only=True)
class Pulling(Hold):
    """A hold that may pull (its least below zero): Hold's class extended, which the solve does not run either."""

    def least(self) -> float:
        return -1e-6


@dataclass(frozen=True, kw_only=True)
class Faster(Coulomb):
    """A friction law with Coulomb's name and another limit: the solve holds Coulomb's own."""

    def limit(self, load: float, speed: float) -> float:
        return self.mu * load * (1.0 + speed)


@pytest.mark.parametrize("law", [Bouncy(), Pulling()], ids=["restitution", "hold-subclass"])
def test_a_normal_law_the_solve_does_not_hold_is_refused_by_name_and_fails_its_suite_unrun(law):
    """The plant: a lab's normal law is refused by the contact, by its name, and ``normal_law_suite`` fails it unrun,
    where it once judged ``Hold()`` in its place and passed; ``Hold()`` itself passes (above)."""
    found = Contact.Projection(margin=1e-4, normal=law).setting_problems({"margin": 1e-4})
    name = type(law).__name__
    assert any(f"normal={name}(...)" in problem for problem in found), found
    result = normal_law_suite(law)
    assert result.status == "FAIL" and f"{name}: not run" in result.detail, result.detail


def test_a_friction_law_the_solve_does_not_hold_is_refused_by_name_and_fails_its_suite_unrun():
    law = Faster(mu=0.3)
    found = Contact.Projection(margin=1e-4, friction=law).setting_problems({"margin": 1e-4})
    assert any("friction=Faster(...)" in problem for problem in found), found
    result = friction_law_suite(law)
    assert result.status == "FAIL" and "Faster: not run" in result.detail, result.detail
    assert not Contact.Projection(margin=1e-4, friction=Coulomb(mu=0.3), normal=Hold()).setting_problems(
        {"margin": 1e-4})


def test_the_built_laws_are_the_ones_the_body_solve_holds():
    from mime.interactions.friction import BUILT_FRICTION_LAWS
    from mime.nodes.body._rigid_inertial import FRICTION_LAWS

    assert all(name in FRICTION_LAWS and law.law == name for name, law in BUILT_FRICTION_LAWS.items())


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


def test_a_free_point_at_rest_steps_with_the_frictionless_derivative(dtype):
    """A sphere a micrometre above the plane, at rest: its pair's bound and its load are both zero, a tie the pair
    takes on its boundary, so the step's derivative in the twist is the frictionless hold's, bit for bit (the free
    minimiser's derivative there read friction at a point with no normal force)."""
    pose = jnp.asarray([0.0, 0.0, 1e-3 + 1e-6, 1.0, 0.0, 0.0, 0.0], dtype)
    inputs = {"applied_wrench": jnp.zeros(6, dtype), "resistance": jnp.zeros((6, 6), dtype),
              "background_load": jnp.zeros(6, dtype)}

    def twist_after(node):
        return lambda twist: node.update({**node.initial_state(), "pose": pose, "twist": twist}, inputs, 1e-3)["twist"]

    rest = jnp.zeros(6, dtype)
    held = jax.jacfwd(twist_after(held_node("sphere")))(rest)
    free = jax.jacfwd(twist_after(held_node("sphere", friction=False)))(rest)
    assert np.array_equal(np.asarray(held), np.asarray(free)), np.asarray(held) - np.asarray(free)


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
    assert float(jnp.max(jnp.abs(node.update(state, inputs, 1e-3)["contact_friction_forces"]))) == 0.0
