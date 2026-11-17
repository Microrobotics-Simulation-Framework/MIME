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
from mime.testing.toys import registered_toy_capsule
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


@pytest.fixture(autouse=True, scope="module")
def toy_kind():
    """The toy kind, registered for this module's tests only: never at import, so that collecting every file leaves the
    registry walk's registry as a fresh process has it."""
    with registered_toy_capsule() as kind:
        yield kind


@pytest.fixture(autouse=True)
def planted(monkeypatch: pytest.MonkeyPatch) -> None:
    if PLANT == "sign":
        found = geometry.Shape.sign_function

        def offset(self, parameters, point):
            held = found(self, parameters, point)
            return SignValue(held.value + OFFSET, held.slope)

        monkeypatch.setattr(geometry.Shape, "sign_function", offset)


def subject(kind: str, route: str | None = None, settings: dict | None = None) -> Subject:
    return Subject("body", kind, settings or SETTINGS, appearance=Appearance.opaque(), moving=True, route=route)


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


def compared(settings: dict, dtype: str) -> tuple[float, float, float, float]:
    """The toy against MIME's capsule by route 1, per pixel and in the silhouette's motion; and against MIME's capsule
    by route 2 (the same march, its sign function its own closed form), per pixel and in the motion."""
    toy, exact, marched = (subject("toy.capsule", settings=settings), subject("capsule", settings=settings),
                           subject("capsule", route="sampled", settings=settings))
    toy_frame, exact_frame = frame(toy), frame(exact)
    pixel = float(np.max(np.abs(toy_frame - exact_frame)))
    shapes = float(np.max(np.abs(toy_frame - frame(marched))))
    a, b, c = motion(exact), motion(toy), motion(marched)
    moved = max(float(np.linalg.norm(b[i] - a[i]) / np.linalg.norm(a[i])) for i in (1, 2))
    shape_moved = max(float(np.linalg.norm(b[i] - c[i]) / np.linalg.norm(c[i])) for i in (1, 2))
    print(f"\ntoy.capsule {dtype} {settings}: against route 1 per pixel {pixel:.3g}, motion {moved:.3g}; against "
          f"MIME's capsule by route 2 per pixel {shapes:.3g}, motion {shape_moved:.3g}")
    return pixel, moved, shapes, shape_moved


def test_the_toy_capsule_is_imaged_as_route_one_images_mimes_capsule(dtype):
    pixel, moved, _, _ = compared(SETTINGS, dtype)
    assert pixel <= SWAP_PIXEL_LINE
    assert moved <= SWAP_MOTION_LINE


#: The derived route's hard cases against the closed form: a rod a pixel's footprint thick (the radius of 25 um is the
#: pixel's 23 um at the body's distance; ``smallest_length`` sets its samples, ``SPACING`` of the radius apart), and the
#: body seen nearly along its axis, so that the rays at its silhouette run beside its side for its whole length (a
#: grazing incidence of about three degrees), and both. Per pixel every case holds G18's swap line. In the silhouette's
#: motion the grazing cases hold it, and the thin rod does not: 2.1 % against route 1, **the sampled route's own limit
#: on a rod one pixel across, not the shape's**: MIME's capsule by the same route (its closed-form sign function) is as
#: far from route 1 (2.1 %), and the toy is 2e-6 from it. A thin feature is therefore held to the same-route line
#: (``SHAPE_MOTION_LINE``: the derivation adds nothing), and to the route's measured limit against route 1, stated.
HARD = {"thin": {"radius": 25e-6, "height": 0.8e-3, "axis": (1.0, 0.3, 0.4)},
        "grazing": {"radius": 0.25e-3, "height": 0.8e-3, "axis": (0.03, 0.0, 1.0)},
        "thin and grazing": {"radius": 25e-6, "height": 0.8e-3, "axis": (0.03, 0.0, 1.0)}}
SHAPE_MOTION_LINE = 1e-4
#: The motion against route 1 where the route itself limits it: a rod about one pixel across, measured at 2.1 %.
THIN_ROUTE_LIMIT = 3e-2


@pytest.mark.parametrize("case", HARD)
def test_the_toy_capsule_thin_or_seen_at_grazing_incidence_is_imaged_as_route_one_images_mimes_capsule(dtype, case):
    pixel, moved, _, shape_moved = compared(HARD[case], dtype)
    assert pixel <= SWAP_PIXEL_LINE
    assert shape_moved <= SHAPE_MOTION_LINE
    assert moved <= (THIN_ROUTE_LIMIT if case == "thin" else SWAP_MOTION_LINE)
