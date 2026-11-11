"""Rolling resistance (``Coulomb(mu, rolling=mu_r)``, W2b's PR 2) and the spun sphere, against closed forms, in both
dtypes, each with a control that fails on every run (fixed cases, nothing drawn):

- **A sphere on a slope with rolling resistance**: held still for ``tan(theta) <= mu_r / a``; rolling without slipping
  at ``(5/7) g (sin(theta) - (mu_r / a) cos(theta))`` for ``tan(theta) <= (7/2) mu - (5/2) mu_r / a``; slipping past it,
  its spin growing at ``(mu a - mu_r) g cos(theta) / (2/5 a^2)``; each threshold either side at 0.9995 and 1.0005.
- **Rolling on a flat slows at ``(5/7) mu_r g / a``** and then stays still. The control: rolling resistance taken about
  the normal (a spin row, ``[0; n]``) never slows the rolling.
- **A sphere spun on a flat** sticks after ``2 a omega0 / (7 mu g)`` within a step and ends at ``(2/7) a omega0``,
  whatever ``mu``: its angular momentum about the contact point is conserved. The control: friction taken at the
  centre (``[t; 0]``) never turns the spin into rolling.
- **The derivative in mu_r** while rolling, ``-(5/7) g cos(theta) T / a``, reverse mode against the closed form and
  central differences, at ``mu_r = 0`` too, where it is the slope into the admissible side (the right-hand difference's;
  the spike's prototype read zero there)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from friction_cases import BALL, MASS, RADIUS, plane_node, rolling_oracle, slope_inputs

import mime.nodes.body._friction as friction_module

G, DT, STEPS = 9.81, 1e-3, 100
EXACT = {jnp.float64: 1e-11, jnp.float32: 5e-5}


@pytest.fixture(params=[jnp.float64, jnp.float32], ids=["float64", "float32"])
def dtype(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == jnp.float64)
    yield request.param
    jax.config.update("jax_enable_x64", held)


def stepped(node, inputs, steps=STEPS, state=None, params=None):
    start = node.initial_state() if state is None else state

    def scan(held):
        return jax.lax.scan(lambda s, _: (node.update(s, inputs, DT, params=params), None), held, None,
                            length=steps)[0]
    return jax.jit(scan)(start)


THRESHOLDS = [("still", 0.9995), ("still", 1.0005), ("roll", 0.9995), ("roll", 1.0005)]


@pytest.mark.parametrize("which, ratio", THRESHOLDS)
def test_a_sphere_with_rolling_resistance_is_held_rolls_or_slips_either_side_of_each_threshold(dtype, which, ratio):
    mu, mu_r = 0.2, 0.05 * RADIUS
    tangent = (mu_r / RADIUS if which == "still" else 3.5 * mu - 2.5 * mu_r / RADIUS) * ratio
    theta = math.atan(tangent)
    out = stepped(plane_node(BALL, mu=mu, rolling=mu_r), slope_inputs(MASS, theta, dtype))
    state, acceleration, spin = rolling_oracle(theta, mu, mu_r)
    T = STEPS * DT
    v, w = float(out["twist"][0]), float(out["twist"][4])
    print(f"\n{state}: {v / T:.10g} against {acceleration:.10g} m/s^2, spin rate {w / T:.10g} against {spin:.10g}")
    assert abs(v / T - acceleration) <= EXACT[dtype] * G
    assert abs(w / T - spin) <= EXACT[dtype] * max(abs(spin), G / RADIUS)


def rolling_on_the_flat(dtype, *, about_the_normal: bool = False):
    """A sphere rolling without slip at 2 mm/s on the flat with rolling resistance 0.05 a, mu 0.3: its speed over
    steps; ``about_the_normal``, the control's planted rows, the rolling pair turned about the normal."""
    mu, mu_r, v0 = 0.3, 0.05 * RADIUS, 2e-3
    node = plane_node(BALL, mu=mu, rolling=mu_r)
    original = friction_module.traction_rows
    if about_the_normal:
        def planted(rows, points, centre, along, *, rolling=False):
            held = original(rows, points, centre, along, rolling=rolling)
            normal = jnp.concatenate([jnp.zeros_like(rows[:, :3]), rows[:, :3]], axis=-1)
            return held.at[:, 3].set(normal).at[:, 4].set(normal)
        friction_module.traction_rows = planted
        import mime.nodes.body._rigid_inertial as rigid
        rigid.traction_rows = planted
    try:
        state = {**node.initial_state(), "twist": jnp.asarray([v0, 0.0, 0.0, 0.0, v0 / RADIUS, 0.0], dtype)}
        steps = int(1.5 * 7 * v0 * RADIUS / (5 * mu_r * G) / DT)

        def scan(held):
            def one(s, _):
                ended = node.update(s, slope_inputs(MASS, 0.0, dtype), DT)
                return ended, ended["twist"][0]
            return jax.lax.scan(one, held, None, length=steps)[1]
        speeds = np.asarray(jax.jit(scan)(state), dtype=np.float64)
    finally:
        friction_module.traction_rows = original
        import mime.nodes.body._rigid_inertial as rigid
        rigid.traction_rows = original
    return speeds, mu_r, v0


def test_rolling_on_a_flat_slows_at_five_sevenths_mu_r_g_over_a_and_then_stays(dtype):
    speeds, mu_r, v0 = rolling_on_the_flat(dtype)
    rate = 5.0 / 7.0 * mu_r * G / RADIUS
    expected = np.maximum(v0 - rate * DT * (np.arange(len(speeds)) + 1), 0.0)
    print(f"\nworst {np.max(np.abs(speeds - expected)):.3g} m/s against v0 {v0:g}; last {speeds[-1]:.3g}")
    assert np.max(np.abs(speeds - expected)) <= 10 * EXACT[dtype] * v0 + rate * DT
    assert abs(speeds[-1]) <= EXACT[dtype] * v0


def test_control_rolling_resistance_about_the_normal_never_slows_the_rolling():
    speeds, _, v0 = rolling_on_the_flat(jnp.float64, about_the_normal=True)
    assert abs(speeds[-1] - v0) <= 1e-9 * v0


def spun(dtype, mu: float, *, at_the_centre: bool = False):
    """A sphere spun at 100 rad/s about the plane's y on the flat, dry: its speed and spin over the steps."""
    omega0, dt = 100.0, 1e-4
    stick = 2 * RADIUS * omega0 / (7 * mu * G)
    steps = int(2 * stick / dt)
    node = plane_node(BALL, mu=mu, dt=dt)
    original = friction_module.traction_rows
    if at_the_centre:
        def planted(rows, points, centre, along, *, rolling=False):
            held = original(rows, points, centre, along, rolling=rolling)
            return held.at[:, 1:3, 3:].set(0.0)
        friction_module.traction_rows = planted
        import mime.nodes.body._rigid_inertial as rigid
        rigid.traction_rows = planted
    try:
        state = {**node.initial_state(), "twist": jnp.asarray([0.0, 0.0, 0.0, 0.0, omega0, 0.0], dtype)}

        def scan(held):
            def one(s, _):
                ended = node.update(s, slope_inputs(MASS, 0.0, dtype), dt)
                return ended, (ended["twist"][0], ended["twist"][4])
            return jax.lax.scan(one, held, None, length=steps)[1]
        speeds, spins = (np.asarray(x, dtype=np.float64) for x in jax.jit(scan)(state))
    finally:
        friction_module.traction_rows = original
        import mime.nodes.body._rigid_inertial as rigid
        rigid.traction_rows = original
    return speeds, spins, stick, omega0, dt


@pytest.mark.parametrize("mu", [0.1, 0.5])
def test_a_sphere_spun_on_a_flat_sticks_on_time_and_rolls_at_two_sevenths_a_omega0(dtype, mu):
    speeds, spins, stick, omega0, dt = spun(dtype, mu)
    rolling = np.abs(speeds - RADIUS * spins) <= 1e-6 * RADIUS * omega0
    first = int(np.argmax(rolling)) + 1
    print(f"\nstuck at step {first} ({first * dt:.6g} s) against {stick:.6g} s; final {speeds[-1]:.10g} against "
          f"{2 / 7 * RADIUS * omega0:.10g}")
    assert abs(first * dt - stick) <= dt
    assert abs(speeds[-1] - 2.0 / 7.0 * RADIUS * omega0) <= 10 * EXACT[dtype] * RADIUS * omega0


def test_control_friction_at_the_centre_never_turns_the_spin_into_rolling():
    speeds, spins, _, omega0, _ = spun(jnp.float64, 0.5, at_the_centre=True)
    assert abs(spins[-1] - omega0) <= 1e-9 * omega0 and abs(speeds[-1]) <= 1e-12


@pytest.mark.parametrize("start", [0.05, 0.0])
def test_the_derivative_in_mu_r_rolling_is_the_closed_form_even_at_zero(dtype, start):
    """Rolling at ``tan(theta) = mu`` (well within ``(7/2) mu``), ``dv/dmu_r = -(5/7) g cos(theta) T / a``. At
    ``mu_r = 0`` too: the pair's bound is zero there and its force is the bound along its direction (the boundary
    branch, taken wherever the free force is outside a disk of radius zero), so reverse mode reads the slope into the
    admissible side, ``mu_r >= 0``, the right-hand difference's. (The friction spike's prototype set the force to zero
    by a selection at a zero bound and read 0 there, which is why it advised starting a fit above zero: MIME's need
    not.)"""
    mu, steps = 0.2, 50
    theta = math.atan(mu)
    node = plane_node(BALL, mu=mu, rolling=start * RADIUS)
    leaves = node.params_pytree()
    inputs = slope_inputs(MASS, theta, dtype)

    def speed(coefficient):
        return stepped(node, inputs, steps, params={**leaves, "friction_rolling": coefficient})["twist"][0]

    held = jnp.asarray(start * RADIUS, dtype=dtype)
    derivative = float(jax.grad(speed)(held))
    closed = -5.0 / 7.0 * G * math.cos(theta) * steps * DT / RADIUS
    print(f"\nat mu_r = {start} a: AD {derivative:.10g} against the closed form {closed:.10g}")
    if start > 0.0:
        assert abs(derivative - closed) <= (1e-9 if dtype == jnp.float64 else 1e-4) * abs(closed)
        if dtype == jnp.float64:
            h = 1e-6 * RADIUS
            central = (float(speed(held + h)) - float(speed(held - h))) / (2 * h)
            assert abs(derivative - central) <= 1e-6 * abs(closed)
    else:
        right = (float(speed(held + 1e-6 * RADIUS)) - float(speed(held))) / (1e-6 * RADIUS)
        assert abs(derivative - closed) <= (1e-9 if dtype == jnp.float64 else 1e-4) * abs(closed)
        assert abs(right - closed) <= (1e-6 if dtype == jnp.float64 else 2e-1) * abs(closed)


def test_rolling_resistance_is_refused_negative_and_builds_at_zero():
    from mime.interactions import Contact
    from mime.interactions.friction import Coulomb
    from mime.units import m, um

    assert Contact.Projection(margin=m(1e-4), friction=Coulomb(mu=0.2, rolling=um(-1.0))).setting_problems(
        {"margin": 1e-4})
    law = Coulomb(mu=0.2, rolling=um(0.0))
    assert law.held_rows() == ("sliding", "rolling") and law.solve_name() == "coulomb_rolling"
    assert Coulomb(mu=0.2).held_rows() == ("sliding",)
