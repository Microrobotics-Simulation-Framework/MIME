"""G16, the cheap sign functions (``Geometry.sign_function``): on drawn points, each kind's sign function has its
region's sign (its registered shape's signed distance, or the conformance harness's level set) and its gradient's
length is the length of its value's gradient by autodiff within half its smallest length of the boundary, off its
medial set (``mime.testing.sign_function_problems``).

The points are drawn at the kinds' own scale: uniformly through a box 1.3 times the enclosing radius each way, and
near the surface, a sample of each patch moved along its normal by up to a fifth of the kind's smallest length. A
sign is judged only where the exact distance is beyond the comparison's rounding (``KIND_ROUNDING_ULPS`` units in the
last place of the points' size).

**The controls**, each failing on every draw: a sign function whose sign is flipped within a band a tenth of the
smallest length wide about the boundary, and one whose gradient's length is 10 % long. ``SIGN_PLANT`` switches them
on for ``.github/probe/detection_rate.py``; here each is its own test.

**The helix's held angle** (its nearest centreline angle found under ``stop_gradient``): the derivative of its sign
function in the point and in its settings equals the central difference of the converged distance, which the switch
scan's list cites (``cleared_by``)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from screw_grid import FL9

from mime.registry import SignValue, implements
from mime.scene import Capsule, Cylinder, Ellipsoid, Helix, Screw, Sphere
from mime.testing import sign_function_problems
from mime.units import um

PLANT = os.environ.get("SIGN_PLANT", "")

#: Each kind at a setting of its own scale, written as a script declares it.
KINDS = {
    "sphere": Sphere(radius=um(600.0)),
    "capsule": Capsule(radius=um(300.0), height=um(1000.0), axis=(1.0, 0.2, 0.3)),
    "cylinder": Cylinder(radius=um(400.0), length=um(1200.0), axis=(0.3, 1.0, -0.2)),
    "ellipsoid": Ellipsoid(semi_axes=um((900.0, 400.0, 300.0))),
    "screw": Screw(radius=um(FL9["radius"] * 1e6), length=um(FL9["length"] * 1e6), wavenumber=2.33, modulation=0.33,
                   starts=2),
    "helix": Helix(radius=um(400.0), pitch=um(600.0), turns=2.5, filament_radius=um(80.0)),
}


def settings_of(kind) -> dict:
    import dataclasses

    from mime.units import Quantity

    held = {}
    for item in dataclasses.fields(kind):
        value = getattr(kind, item.name)
        held[item.name] = value.value if isinstance(value, Quantity) else value
    return held


def smallest(kind) -> float:
    found = []
    for value in settings_of(kind).values():
        found += [abs(float(v)) for v in (value if isinstance(value, tuple) else (value,))
                  if isinstance(v, float) and v != 0.0 and abs(v) < 1.0]
    return min(found)


def band(kind) -> float:
    """Within this of the boundary the gradient's length is checked: half the smallest length, off every medial set
    (a centre, an axis, a segment, a centreline lies a radius or more inside)."""
    return 0.5 * smallest(kind)


def drawn_points(kind, seed: int, count: int = 96) -> np.ndarray:
    """Uniform points through the enclosing box and points near the surface, from ``seed``."""
    held = settings_of(kind)
    rng = np.random.default_rng(seed)
    reach = 1.3 * kind.enclosing_radius(held)
    uniform = rng.uniform(-reach, reach, (count, 3))
    near = []
    for patch in kind.patches:
        counts = kind.sample_counts(held, smallest(kind) / 2.0, patch)
        samples = kind.samples({k: (jnp.asarray(v) if isinstance(v, (float, tuple)) else v) for k, v in held.items()},
                               patch, counts)
        points, normals = np.asarray(samples.points), np.asarray(samples.normals)
        chosen = rng.choice(len(points), size=min(count, len(points)), replace=False)
        offsets = rng.uniform(-0.2, 0.2, (len(chosen), 1)) * smallest(kind)
        near.append(points[chosen] + offsets * normals[chosen])
    return np.concatenate([uniform, *near])


class Flipped:
    """A planted sign function: the kind's own with its sign flipped within ``band`` of the boundary (a tenth of the
    smallest length), or its gradient's length 10 % long."""

    def __init__(self, kind, how: str):
        self.kind, self.how = kind, how
        self.band = 0.1 * smallest(kind)

    def __getattr__(self, name):
        return getattr(self.kind, name)

    def sign_function(self, parameters, region, point):
        found = self.kind.sign_function(parameters, region, point)
        if self.how == "slope":
            return SignValue(found.value, 1.1 * found.slope)
        distance = found.value / found.slope
        return SignValue(jnp.where(jnp.abs(distance) < self.band, -found.value, found.value), found.slope)


def planted(kind):
    return Flipped(kind, PLANT) if PLANT in ("sign", "slope") else kind


@pytest.mark.parametrize("name", sorted(KINDS))
@settings(deadline=None)
@given(seed=st.integers(0, 2 ** 31 - 1))
def test_each_kinds_sign_function_has_its_regions_sign_and_its_gradients_length(name, seed):
    kind = KINDS[name]
    assert implements(kind, "sign_function")
    problems = sign_function_problems(planted(kind), settings_of(kind), drawn_points(kind, seed), band(kind))
    assert problems == [], "\n".join(problems)


@pytest.mark.parametrize("how", ["sign", "slope"])
@pytest.mark.parametrize("name", sorted(KINDS))
def test_control_a_flipped_sign_or_a_long_gradient_fails_the_check(name, how):
    kind = KINDS[name]
    problems = sign_function_problems(Flipped(kind, how), settings_of(kind), drawn_points(kind, 7), band(kind))
    assert problems, f"{name}: the planted {how} passed"


def test_the_helixs_held_angle_gives_the_converged_distances_derivative():
    """At points near the filament and its caps, the sign function's derivative in the point and in the helix's settings
    (its held angle clamped to the arc with the sweep's own derivative) equals the central difference of the distance
    itself: the envelope's rule, which holding the angle relies on."""
    kind = KINDS["helix"]
    held = settings_of(kind)
    points = drawn_points(kind, 11, 24)
    names = ("radius", "pitch", "turns", "filament_radius")

    def value(parameters, point):
        return kind.sign_function(dict(zip(names, parameters, strict=True)), "solid",
                                  (point[0], point[1], point[2])).value

    base = jnp.asarray([held[n] for n in names], dtype=jnp.float64)
    for point in points[len(points) // 2:]:
        p = jnp.asarray(point)
        grad_point = np.asarray(jax.grad(value, argnums=1)(base, p))
        grad_settings = np.asarray(jax.grad(value, argnums=0)(base, p))
        for i in range(3):
            step = np.zeros(3)
            step[i] = 1e-7
            difference = (float(value(base, p + step)) - float(value(base, p - step))) / 2e-7
            assert abs(difference - grad_point[i]) <= 1e-5 * (1.0 + abs(difference)), (point, i)
        for i in range(4):
            step = np.zeros(4)
            step[i] = 1e-6 * float(base[i])
            difference = (float(value(base + step, p)) - float(value(base - step, p))) / (2 * step[i])
            scale = abs(difference) + float(np.max(np.abs(grad_settings)))
            assert abs(difference - grad_settings[i]) <= 1e-4 * scale, (point, names[i])
