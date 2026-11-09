"""The prescribed flow (F1b's gate 5): the in-program series for Womersley's profile and its mean against SciPy's
Bessel functions across alpha, per dtype, with the validity line it sets; Poiseuille exact; the series' controls; the
node's reading, its integer clock and its refusal past the line; its derivatives against central differences.

**Measured (F1b's first step):** float64 under 2e-13 to alpha 30, 3.6e-12 at 40, 1.3e-9 at 60; float32 7.5e-6 at alpha
20, 3.2e-5 at 25, 1.1e-4 at 28. A planted sign on the k = 3 ratio is caught above 1e-3 only for alpha in [1, 25] (a low
term is negligible where q is small, and dwarfed by |J0| ~ e^{alpha/sqrt 2} where it is large); a halved term count only
from alpha 35: the controls draw alpha where each bites."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from fvm_cases import in_dtype

from mime.nodes.flow import _womersley as wom
from mime.nodes.flow._waveform import waveform

ROT = np.exp(-1j * np.pi / 4)


@pytest.fixture
def float32():
    with in_dtype("float32"):
        yield "float32"


def scipy_profile(rho, alpha):
    """``(1 - J0(b r) / J0(b R)) / alpha^2`` from SciPy (complex)."""
    return (1.0 - scipy.special.jv(0, ROT * alpha * rho) / scipy.special.jv(0, ROT * alpha)) / alpha ** 2


def scipy_mean(alpha):
    z = ROT * alpha
    return (1.0 - 2.0 * scipy.special.jv(1, z) / (z * scipy.special.jv(0, z))) / alpha ** 2


def _series(rho, alpha, terms, dtype):
    re, im = wom.profile(jnp.asarray(rho, dtype=dtype) ** 2, jnp.asarray(alpha, dtype=dtype) ** 2, terms)
    return np.asarray(re, np.float64) + 1j * np.asarray(im, np.float64)


RHO = np.linspace(0.0, 1.0, 400)


def profile_error(alpha, dtype, terms=None):
    terms = wom.series_terms(dtype) if terms is None else terms
    exact = scipy_profile(RHO, alpha)
    return float(np.max(np.abs(_series(RHO, alpha, terms, dtype) - exact)) / np.max(np.abs(exact)))


#: The validity line each dtype's series is held to (the profile's error over its peak), with the alphas it is checked
#: at; SciPy's own ``1 - ratio`` form loses eps / alpha^2 below about alpha 0.5, so the oracle starts there.
LINES = {"float64": ((0.5, 1, 2, 6, 10, 20, 30), 1e-12, (40, 50), 1e-10), "float32": ((0.5, 1, 6, 15, 20, 25), 4e-5,
                                                                                    (), 0.0)}


def test_the_series_matches_scipy_within_its_dtypes_validity_line(dtype):
    low, line, high, high_line = LINES[dtype]
    for alpha in low:
        assert profile_error(alpha, dtype) < line, alpha
    for alpha in high:
        assert profile_error(alpha, dtype) < high_line, alpha
    assert wom.VALIDITY[dtype][0] == (50.0 if dtype == "float64" else 25.0)


def test_the_mean_profile_matches_scipy(dtype):
    for alpha in (0.5, 6.0, 20.0):
        re, im = wom.mean_profile(jnp.asarray(alpha) ** 2, wom.series_terms(dtype))
        found = float(re) + 1j * float(im)
        exact = scipy_mean(alpha)
        assert abs(found - exact) / abs(exact) < (1e-11 if dtype == "float64" else 1e-5)


def test_the_series_has_the_quasi_steady_limit_at_alpha_zero(dtype):
    """At alpha 0 the profile is ``i (1 - rho^2) / 4`` and its mean ``i / 8``: no division by zero, no NaN."""
    found = _series(RHO, 0.0, wom.series_terms(dtype), dtype)
    assert np.max(np.abs(found - 1j * (1 - RHO ** 2) / 4)) < 1e-6
    re, im = wom.mean_profile(jnp.asarray(0.0), wom.series_terms(dtype))
    assert abs(float(re)) < 1e-7 and abs(float(im) - 0.125) < 1e-7


def _planted_sign(rho, alpha, terms, dtype):
    original = wom._nested

    def planted(x, count, ratio):
        return original(x, count, lambda k: -ratio(k) if k == 2 else ratio(k))  # the ratio 1 / 3^2 of J0's terms

    wom._nested = planted
    try:
        return _series(rho, alpha, terms, dtype)
    finally:
        wom._nested = original


@pytest.mark.parametrize("alpha", [1.0, 4.0, 10.0, 20.0])
def test_control_a_planted_term_sign_fails_the_comparison(float64, alpha):
    exact = scipy_profile(RHO, alpha)
    found = _planted_sign(RHO, alpha, wom.series_terms("float64"), "float64")
    assert np.max(np.abs(found - exact)) / np.max(np.abs(exact)) > 1e-3


@pytest.mark.parametrize("alpha", [40.0, 50.0])
def test_control_a_halved_term_count_fails_the_comparison(float64, alpha):
    assert profile_error(alpha, "float64", terms=wom.series_terms("float64") // 2) > 1e-3


def host_error(alpha, terms=None):
    terms = wom.series_terms("float64") if terms is None else terms
    exact = scipy_profile(RHO, alpha)
    return float(np.max(np.abs(wom.host_profile(RHO ** 2, alpha ** 2, terms) - exact)) / np.max(np.abs(exact)))


def test_the_host_series_holds_float64s_line_in_either_run_dtype(dtype):
    """The build's step rule sums the series on the host in float64, whatever the run's dtype."""
    low, line, high, high_line = LINES["float64"]
    assert all(host_error(alpha) < line for alpha in low)
    assert all(host_error(alpha) < high_line for alpha in high)


@pytest.mark.parametrize("alpha", [40.0, 50.0])
def test_control_the_host_series_with_half_its_terms_fails(alpha):
    assert host_error(alpha, wom.series_terms("float64") // 2) > 1e-3


# -- the node --------------------------------------------------------------------------------------------------------

def _node(dt=1e-3, **kw):
    settings = dict(viscosity=1e-3, density=1e3, tube_centre=(0.0, 0.0, 0.0), tube_axis=(0.0, 0.0, 1.0),
                    tube_radius=0.5e-3)
    settings.update(kw)
    return wom.PrescribedFlowNode("flow", dt, **settings)


def _run(node, w, steps):
    p = node._step_constants(None)  # noqa: SLF001
    w = jnp.asarray(w, dtype=jnp.asarray(node.initial_state()["centreline"]).dtype)

    def body(s, _):
        return node.advance(p, s, {"waveform": w}, node.delta_t), None

    return dict(jax.lax.scan(body, dict(node.initial_state()), None, length=steps)[0]), p


def test_a_steady_drive_gives_poiseuille_exactly(dtype):
    """``g0 (R^2 - r^2) / (4 nu)`` at points across the lumen, its mean half the centreline, nothing beyond the wall."""
    with in_dtype(dtype):
        node = _node()
        state, p = _run(node, waveform((0.0, 0.0, 1.0), 2.0, 0.0, 0.0, 0.0), 3)
        r = np.linspace(0.0, 0.7e-3, 15)
        points = jnp.stack([jnp.asarray(r), jnp.zeros(15), jnp.zeros(15)], axis=-1)
        found = np.asarray(node.velocity_at(p, state, points), np.float64)
    nu, radius = 1e-6, 0.5e-3
    exact = np.where(r < radius, 2.0 * (radius ** 2 - r ** 2) / (4 * nu), 0.0)
    tol = 1e-12 if dtype == "float64" else 1e-6
    assert found.shape == (15, 3)
    assert np.max(np.abs(found[:, 2] - exact)) <= tol * np.max(exact)
    assert np.max(np.abs(found[:, :2])) == 0.0
    assert float(state["lumen_flux"]) == pytest.approx(np.pi * radius ** 2 * 2.0 * radius ** 2 / (8 * nu),
                                                       rel=tol * 10)


def test_the_reading_follows_the_drives_axial_component_on_a_tilted_tube(float64):
    axis = np.asarray([1.0, 0.0, 3.0]) / math.sqrt(10.0)
    node = _node(tube_axis=tuple(axis))
    state, p = _run(node, waveform((0.0, 0.0, 1.0), 2.0, 0.0, 0.0, 0.0), 1)
    found = np.asarray(node.velocity_at(p, state, jnp.zeros((1, 3))))[0]
    expected = 2.0 * axis[2] * (0.5e-3) ** 2 / (4e-6) * axis
    assert np.allclose(found, expected, rtol=1e-12)


def test_the_harmonic_term_matches_scipys_womersley_at_its_time(dtype):
    """alpha 6 at points and at the node's own time, the step count times dt (the integer clock)."""
    with in_dtype(dtype):
        radius, nu = 0.5e-3, 1e-6
        omega = (6.0 / radius) ** 2 * nu
        node = _node(dt=2 * math.pi / omega / 300)
        state, p = _run(node, waveform((0.0, 0.0, 1.0), 0.0, 3.0, omega / (2 * math.pi), 0.4), 1234)
        r = np.linspace(0.0, 0.99 * radius, 12)
        points = jnp.stack([jnp.zeros(12), jnp.asarray(r), jnp.zeros(12)], axis=-1)
        found = np.asarray(node.velocity_at(p, state, points), np.float64)[:, 2]
    t = 1234 * node.delta_t
    exact = np.real((3.0 / (1j * omega)) * (1 - scipy.special.jv(0, ROT * 6.0 * r / radius) / scipy.special.jv(0, ROT * 6.0))
                    * np.exp(1j * (omega * t + 0.4)))
    tol = 1e-10 if dtype == "float64" else 1e-4
    assert np.max(np.abs(found - exact)) <= tol * np.max(np.abs(exact))


def test_control_a_summed_float32_clock_moves_the_phase(float32):
    """The node's time is its count times dt; a float32 sum of dt over the same count is off by more than the
    reading's float32 tolerance at alpha 6 (S1: 2.0e-3 rad at 4,190 steps)."""
    radius, nu = 0.5e-3, 1e-6
    omega = (6.0 / radius) ** 2 * nu
    dt = np.float32(2 * math.pi / omega / 300)
    count = 20000
    summed = np.float32(0.0)
    for _ in range(count):
        summed = np.float32(summed + dt)
    exact = count * float(dt)
    assert abs(float(summed) - exact) * omega > 1e-4
    assert abs(float(np.float32(count) * dt) - exact) * omega < 1e-5


def test_past_the_validity_line_the_run_is_refused(dtype):
    with in_dtype(dtype):
        radius, nu = 0.5e-3, 1e-6
        line = wom.VALIDITY[dtype][0]
        omega = (1.2 * line / radius) ** 2 * nu
        node = _node()
        state, _ = _run(node, waveform((0.0, 0.0, 1.0), 0.0, 1.0, omega / (2 * math.pi), 0.0), 2)
        ok, _ = _run(node, waveform((0.0, 0.0, 1.0), 0.0, 1.0, omega / (2 * math.pi) / 4.0, 0.0), 2)
    assert node.stop_problem(state) is not None
    assert node.stop_problem(state).code == "flow.prescribed-alpha"
    assert node.stop_problem(ok) is None


def test_the_readings_derivatives_in_viscosity_and_radius_equal_central_differences(float64):
    """Route 4 (the series recomputed in the program): reverse mode against a central difference at alpha 6."""
    radius, nu = 0.5e-3, 1e-6
    omega = (6.0 / radius) ** 2 * nu
    node = _node(dt=2 * math.pi / omega / 300)
    w = waveform((0.0, 0.0, 1.0), 1.0, 3.0, omega / (2 * math.pi), 0.4)
    points = jnp.stack([jnp.zeros(5), jnp.linspace(0.0, 0.4e-3, 5), jnp.zeros(5)], axis=-1)

    def loss(mu, r):
        p = {**node._step_constants(None), "viscosity": mu, "tube_radius": r}  # noqa: SLF001
        s = node.advance(p, dict(node.initial_state()), {"waveform": w}, node.delta_t)
        return jnp.sum(node.velocity_at(p, s, points) ** 2)

    grads = jax.grad(loss, argnums=(0, 1))(1e-3, radius)
    for k, (x0, h) in enumerate(((1e-3, 1e-9), (radius, 1e-10))):
        args = [1e-3, radius]
        args[k] = x0 + h
        up = float(loss(*args))
        args[k] = x0 - h
        down = float(loss(*args))
        assert float(grads[k]) == pytest.approx((up - down) / (2 * h), rel=1e-6)


def test_the_readings_derivative_is_finite_at_a_steady_drive(float64):
    """The series takes alpha squared, never its root, so a steady drive (frequency zero) has finite derivatives in
    the viscosity and the radius, Poiseuille's own."""
    node = _node()
    w = waveform((0.0, 0.0, 1.0), 2.0, 0.0, 0.0, 0.0)
    points = jnp.stack([jnp.zeros(3), jnp.linspace(0.0, 0.4e-3, 3), jnp.zeros(3)], axis=-1)

    def loss(mu, r):
        p = {**node._step_constants(None), "viscosity": mu, "tube_radius": r}  # noqa: SLF001
        s = node.advance(p, dict(node.initial_state()), {"waveform": w}, node.delta_t)
        return jnp.sum(node.velocity_at(p, s, points)[:, 2])

    grads = jax.grad(loss, argnums=(0, 1))(1e-3, 0.5e-3)
    assert all(np.isfinite(float(g)) for g in grads)
    exact = -float(loss(1e-3, 0.5e-3)) / 1e-3  # u is proportional to 1 / mu
    assert float(grads[0]) == pytest.approx(exact, rel=1e-9)
