"""The rft rung's gradient gate (phase R's gate 5), in float64: R6's swimming speed U / Omega in free space, differentiated
in the helix's pitch, radius and filament radius in reverse and in forward mode, against a central difference whose two
arms each rebuild the node at the moved value (so a value baked at build moves with them and not with the derivative),
the count of points held (no count switch within the step); and the resistance's viscosity derivative against its
exact one (the resistance is linear in the viscosity).

The line is ten times the difference's own resolution, measured on each run: the spread of the central difference
between its step and a step ten times larger and ten times smaller. At the scale test, the derivative of the speed times
1e-12 and times 1e6 is the derivative times each, to rounding.

**Controls**, each a fresh node (a node's step is traced once, so a patch reaches only a node not yet traced):
``stop_gradient`` on the coefficients, and the coefficients baked as Python floats from the build's values. Both leave
the central difference (which rebuilds) where it was and move the derivative off it by far more than the line."""

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from rft_cases import R6_SETTINGS, r6_node

import mime.nodes.hydro._rft_drag as rft_module

NAMES = ("pitch", "radius", "filament_radius")
#: The count of points the probe holds: R6's helix at one point a filament radius (its default).
COUNT = (94,)
STEP = 1e-5
SCALES = (1e-12, 1e6)


def speed(settings: Mapping[str, float], theta: Mapping[str, Any] | None = None, viscosity: Any = None) -> tuple[
        Any, Any]:
    """U / Omega and R[0, 0] of a node built at ``settings``, its shape leaves ``theta`` (the built values where none)."""
    node = r6_node(tube=False, settings=settings, counts=COUNT)
    params = {**node.constants(), **node.params_pytree()}
    p = {key: jnp.asarray(value) if isinstance(value, float) else value for key, value in params.items()}
    p = {**node._step_constants(p), **p}
    if theta is not None:
        p.update({f"shape_{key}": value for key, value in theta.items()})
    if viscosity is not None:
        p["viscosity"] = viscosity
    found, _ = node.resistance_at(p, jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    return -found[0, 3] / found[0, 0], found[0, 0]


def central(name: str, step: float) -> float:
    h = step * R6_SETTINGS[name]
    up, down = ({**R6_SETTINGS, name: R6_SETTINGS[name] + d} for d in (h, -h))
    return float((speed(up)[0] - speed(down)[0]) / (2.0 * h))


def derivatives() -> tuple[dict[str, float], dict[str, float]]:
    theta = {key: jnp.asarray(value) for key, value in R6_SETTINGS.items()}
    reverse = jax.grad(lambda t: speed(R6_SETTINGS, t)[0])(theta)
    forward = {name: jax.jvp(lambda x, name=name: speed(R6_SETTINGS, {**theta, name: x})[0], (theta[name],),
                             (jnp.asarray(1.0),))[1] for name in NAMES}
    return {k: float(reverse[k]) for k in NAMES}, {k: float(v) for k, v in forward.items()}


def line(name: str) -> tuple[float, float]:
    """The central difference at ``STEP`` and ten times its measured resolution."""
    held = central(name, STEP)
    spread = max(abs(central(name, 10 * STEP) - held), abs(central(name, STEP / 10) - held))
    return held, 10.0 * spread


@pytest.fixture(scope="module")
def measured() -> dict[str, tuple[float, float]]:
    return {name: line(name) for name in NAMES}


def test_reverse_and_forward_agree_with_a_rebuilding_central_difference(measured):
    reverse, forward = derivatives()
    for name in NAMES:
        held, allowed = measured[name]
        assert abs(reverse[name] - held) <= allowed, (name, reverse[name], held, allowed)
        assert abs(forward[name] - held) <= allowed, (name, forward[name], held, allowed)


def test_the_resistance_is_linear_in_the_viscosity():
    viscosity = jnp.asarray(1e-3)
    held = jax.grad(lambda v: speed(R6_SETTINGS, viscosity=v)[1])(viscosity)
    assert float(held) == pytest.approx(float(speed(R6_SETTINGS)[1]) / 1e-3, rel=1e-12)


@pytest.mark.parametrize("scale", SCALES)
def test_the_derivative_scales_with_the_loss(scale):
    theta = {key: jnp.asarray(value) for key, value in R6_SETTINGS.items()}
    plain = jax.grad(lambda t: speed(R6_SETTINGS, t)[0])(theta)
    scaled = jax.grad(lambda t: scale * speed(R6_SETTINGS, t)[0])(theta)
    for name in NAMES:
        assert float(scaled[name]) == pytest.approx(scale * float(plain[name]), rel=1e-12)


def _patched(make: Callable[[Callable[..., tuple[Any, Any]]], Callable[..., tuple[Any, Any]]]) -> dict[str, float]:
    original = rft_module.drag_coefficients
    rft_module.drag_coefficients = make(original)
    try:
        return derivatives()[0]
    finally:
        rft_module.drag_coefficients = original


def _stopped(original: Callable[..., tuple[Any, Any]]) -> Callable[..., tuple[Any, Any]]:
    def coefficients(*args: Any) -> tuple[Any, Any]:
        par, perp = original(*args)
        return jax.lax.stop_gradient(par), jax.lax.stop_gradient(perp)

    return coefficients


def _baked(original: Callable[..., tuple[Any, Any]]) -> Callable[..., tuple[Any, Any]]:
    from mime.scene._helix import HelixTurn

    turn = HelixTurn(R6_SETTINGS)
    wavelength, length = float(2.0 * jnp.pi * turn.w), float(turn.length)
    held = {name: tuple(float(v) for v in original(name, 1e-3, R6_SETTINGS["filament_radius"], wavelength, length))
            for name in rft_module.COEFFICIENT_SETS}

    def coefficients(name: str, viscosity: Any, filament: Any, wavelength: Any, length: Any) -> tuple[Any, Any]:
        par, perp = held[name]
        return par * viscosity / 1e-3, perp * viscosity / 1e-3

    return coefficients


@pytest.mark.parametrize("plant", [_stopped, _baked], ids=["stop_gradient", "baked"])
def test_control_a_coefficient_held_out_of_the_derivative_is_caught(plant, measured):
    planted = _patched(plant)
    caught = [name for name in NAMES if abs(planted[name] - measured[name][0]) > measured[name][1]]
    assert caught == list(NAMES), caught
