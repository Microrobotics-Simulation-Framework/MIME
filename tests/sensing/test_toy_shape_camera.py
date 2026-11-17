"""The camera on a shape with only a distance: the toy kind ``toy.capsule`` (``mime.testing.toys``), whose one region is
the toy plugin shape, gives no closed form and no sign function of its own, so the camera chooses the sampled route and
marches its derived sign function, the shape's signed distance (``mime.core.Shape.sign_function``). Its analytic
reference is MIME's ``capsule`` of the same settings imaged by route 1, the closed form: within G18's swap lines
(``tests/sensing/test_sampled.py``, the same lines route 2 holds MIME's capsule to), per pixel and in the silhouette's
motion, in both dtypes. Beside it, the toy's image against MIME's capsule by route 2 (its sign function its own closed
form) shows how much of the difference is the route's and how much the shape's.

**Control** (``SHAPE_PLANT=sign``): the derived sign function's value offset by a tenth of a pixel's footprint, its
slope right, so the outline moves by that; it must fail on every run (G18's own control, ``sampled-offset``, is the same
size on the march)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import across

import mime.core._geometry as geometry
import mime.testing.toys  # noqa: F401 - registers the toy kind and its shape
from mime.core.shapes import SignValue
from mime.sensing import Appearance, Subject, render
from mime.sensing._subjects import route_of

PLANT = os.environ.get("SHAPE_PLANT", "")
#: G18's swap lines (``tests/sensing/test_sampled.py``): route 2 against route 1 on a body whose sign function is its
#: exact distance, per pixel and in the silhouette's motion.
SWAP_PIXEL_LINE = 2e-3
SWAP_MOTION_LINE = 2e-3
LENS, POSE = across(128, field=3e-3)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
SETTINGS = {"radius": 0.25e-3, "height": 0.8e-3, "axis": (1.0, 0.3, 0.4)}
POSITION = (0.11e-3, 0.0, -0.07e-3)
#: The plant's offset: a tenth of the pixel's footprint at the body's distance (3 mm over 128 pixels), in metres.
OFFSET = 0.1 * 3e-3 / 128


@pytest.fixture(autouse=True)
def planted(monkeypatch: pytest.MonkeyPatch) -> None:
    if PLANT == "sign":
        found = geometry.Shape.sign_function

        def offset(self, parameters, point):
            held = found(self, parameters, point)
            return SignValue(held.value + OFFSET, held.slope)

        monkeypatch.setattr(geometry.Shape, "sign_function", offset)


def subject(kind: str, route: str | None = None) -> Subject:
    return Subject("body", kind, SETTINGS, appearance=Appearance.opaque(), moving=True, route=route)


def frame(body: Subject) -> np.ndarray:
    found = render([body], LENS, POSE, {"body": (jnp.asarray(POSITION), jnp.asarray(IDENTITY))})
    return np.asarray(found.transmission, dtype=np.float64)


def motion(body: Subject) -> np.ndarray:
    """The darkness's total and centroid (u, v), and their derivatives in the body's x and z: ``(3, 3)``."""
    v, u = np.indices((LENS.height, LENS.width))

    def aggregates(at):
        dark = 1.0 - render([body], LENS, POSE, {"body": (at, jnp.asarray(IDENTITY))}).transmission
        total = jnp.sum(dark)
        return jnp.stack([total, jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])

    at = jnp.asarray(POSITION)
    jacobian = jax.jacfwd(aggregates)(at)
    return np.asarray(jnp.stack([aggregates(at), jacobian[:, 0], jacobian[:, 2]]), dtype=np.float64)


def test_a_kind_whose_region_has_only_a_distance_takes_the_sampled_route():
    assert route_of("toy.capsule") == "sampled"
    assert route_of("capsule") == "closed form"


def test_the_toy_capsule_is_imaged_as_route_one_images_mimes_capsule(dtype):
    toy, exact, marched = subject("toy.capsule"), subject("capsule"), subject("capsule", route="sampled")
    toy_frame, exact_frame = frame(toy), frame(exact)
    pixel = float(np.max(np.abs(toy_frame - exact_frame)))
    shapes = float(np.max(np.abs(toy_frame - frame(marched))))
    a, b = motion(exact), motion(toy)
    moved = max(float(np.linalg.norm(b[i] - a[i]) / np.linalg.norm(a[i])) for i in (1, 2))
    print(f"\ntoy.capsule {dtype}: against route 1 per pixel {pixel:.3g}, motion {moved:.3g}; against MIME's capsule "
          f"by route 2 per pixel {shapes:.3g}")
    assert pixel <= SWAP_PIXEL_LINE
    assert moved <= SWAP_MOTION_LINE
