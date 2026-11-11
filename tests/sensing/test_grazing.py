"""G6, grazing rays and misses: for rays built exactly tangent to a sphere, a capsule's side and a cylinder's side,
and rays just past them, every chord, edge value and coverage, and their derivatives in the body's position and radius,
are finite, in both dtypes.

**The control** (``CAMERA_PLANT=hard-clamp``) takes the grazing square root of ``max(h2, 0)`` in place of the declared
softplus: its derivative at the tangent ray is infinite, and the gate fails every run (the tangent ray is built
exactly)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from camera_cases import planted

import mime.sensing._forms as forms
from mime.sensing._coverage import box

RADIUS = 1e-3


def hard_clamp(h2, radius):
    return jnp.maximum(h2, 0.0)


@pytest.mark.parametrize("form", ["sphere", "capsule", "cylinder"])
def test_tangent_and_missing_rays_are_finite(dtype, form, monkeypatch):
    if planted("hard-clamp"):
        monkeypatch.setattr(forms, "_soft", hard_clamp)
    held = {"sphere": forms.SPHERE, "capsule": forms.CAPSULE, "cylinder": forms.CYLINDER}[form]
    # Rays along +y, x = R from the axis exactly (tangent: the squared half-chord is zero), and just beyond and inside
    # it; at z = 0.3 R, within a capsule's or a cylinder's span (whose axis is z), or z = 0 for the ball.
    across = jnp.asarray([RADIUS, RADIUS * (1 + 1e-6), RADIUS * 1.5, RADIUS * (1 - 1e-6)])
    height = 0.0 if form == "sphere" else 0.3 * RADIUS

    def values(radius, offset):
        p = {"radius": radius, "height": 4 * RADIUS, "length": 4 * RADIUS}
        o = (across + offset, jnp.full_like(across, -0.1), jnp.full_like(across, height))
        d = (jnp.zeros_like(across), jnp.ones_like(across), jnp.zeros_like(across))
        interval = held.interval(p, o, d)
        g = held.edge(p, o, d)
        return jnp.stack([forms.chord(interval), g, box(g, jnp.full_like(g, 1.0), jnp.zeros_like(g))])

    value = values(jnp.asarray(RADIUS), jnp.asarray(0.0))
    gradient = jax.jacfwd(values, argnums=(0, 1))(jnp.asarray(RADIUS), jnp.asarray(0.0))
    assert abs(float(value[1, 0])) <= 4 * float(jnp.finfo(value.dtype).eps) * RADIUS
    assert bool(jnp.all(jnp.isfinite(value))) and all(bool(jnp.all(jnp.isfinite(g))) for g in gradient)
