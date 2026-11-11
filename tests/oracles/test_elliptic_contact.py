"""The near-contact seam's general point form and its several contacts, as functions (W2a PR 1b-ii-a; predictions
committed first, ``tests/oracles/predictions_w2a_1b2.json``):

- **The elliptic form is the compilation's equations term for term:** Janoschek, Harting and Toschi's Appendix A (Eqs.
  A.1, A.2, A.4 to A.6, A.8 to A.10, A.13 to A.16: Cox's and Claeys and Brady's terms the curvatures alone give),
  transcribed here as printed, in the body's own principal frame at an angle chi from the gap's, with the compilation's
  ``ln h`` and ``1 / h``, turned to the gap's frame and given MIME's sign convention (the couplings of a force with a
  rotation the other sign), equal the covariant form ``_elliptic.singular`` to rounding over drawn curvatures and angles;
  with the frames aligned it is the sphere's form's ``singular``. Control: the cross term ``b12`` with its sign flipped.
- **One surface opened for several contacts:** a single contact's opening is its own bit for bit; contacts at one point
  count once (their opening and shares are one contact's); the opening is continuous where two contacts' gaps cross;
  a contact's share is one in the band and none past twice it.
- **The ambient's zero on a dilated lumen:** the dilated wall, the far side included, reads the true wall. Control: the
  plane's translation in a lumen leaves the far side twice the opening off.
- **m8:** ``lubrication.twice`` sees a hand-over on by default (the entry after its defaults); written off, it plans.
- **The classes:** a sphere's contact on a plane is a point; a cylinder's side a line; a sphere nearly filling a lumen a
  ring; the gated sphere-to-tube ratios points. Control: a classifier that calls every contact a point.

Float64 (``tests/oracles/conftest.py``). The controls honour ``W2A_PLANT`` (``lubrication_cases.planted``)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lubrication_cases import planted

from mime.nodes.hydro.near_contact import (
    LIFT_WIDTH,
    NearContact,
    ambient_point,
    classified,
    in_band,
    merged,
    surface_opening,
)
from mime.nodes.hydro.near_contact import _elliptic, _seam, _sphere

#: Curvatures drawn at a body's scales, per unit length of its size (a contact's radius from a fifth of the size to five
#: times it), and the two gaps an increment spans.
CURVATURE = st.floats(0.2, 5.0)
ANGLE = st.floats(0.0, math.pi)
INVERSE, LOG = 37.0, 1.3


def compilation(k1: float, k2: float, c1: float, c2: float, chi: float) -> np.ndarray:
    """Appendix A's terms as printed, in the body's principal frame (translations along its ``x1``, ``x2``, the normal;
    rotations about them), the gap's half-curvatures ``lambda_i = k_i / 2``, the body's radii ``S_i = 1 / c_i``, ``chi``
    the angle from the body's ``x1`` to the gap's (``l2 = (sin chi, cos chi)``), ``ln h`` as ``-LOG`` and ``1 / h`` as
    ``INVERSE``; the terms that need the cubic coefficients (A.3, A.7, A.11, A.12) zero."""
    l1, l2, s1, s2 = k1 / 2.0, k2 / 2.0, 1.0 / c1, 1.0 / c2
    root, a1, a2 = math.sqrt(l1 * l2), (3 * l1 + 2 * l2) * l1, (2 * l1 + 3 * l2) * l2
    c, s = math.cos(chi), math.sin(chi)
    ln_h = -LOG
    k = np.zeros((6, 6))
    k[0, 0] = -math.pi * ln_h / (root * s1 ** 2) * (3 * c * c * (1 - l1 * s1) ** 2 / a1
                                                    + 3 * s * s * (1 - l2 * s1) ** 2 / a2 + s1 ** 2)  # A.1
    k[0, 1] = 3 * math.pi * ln_h / (root * s1 * s2) * s * c * ((1 - l1 * s1) * (1 - l1 * s2) / a1
                                                               - (1 - l2 * s1) * (1 - l2 * s2) / a2)  # A.2
    k[0, 3] = 3 * math.pi * ln_h / (root * s1) * s * c * ((1 - l1 * s1) / a1 - (1 - l2 * s1) / a2)  # A.4
    k[0, 4] = 3 * math.pi * ln_h / (root * s1) * (c * c * (1 - l1 * s1) / a1 + s * s * (1 - l2 * s1) / a2)  # A.5
    k[1, 1] = -math.pi * ln_h / (root * s2 ** 2) * (3 * s * s * (1 - l1 * s2) ** 2 / a1
                                                    + 3 * c * c * (1 - l2 * s2) ** 2 / a2 + s2 ** 2)  # A.6
    k[1, 3] = -3 * math.pi * ln_h / (root * s2) * (s * s * (1 - l1 * s2) / a1 + c * c * (1 - l2 * s2) / a2)  # A.8
    k[1, 4] = 3 * math.pi * ln_h / (root * s2) * s * c * (-(1 - l1 * s2) / a1 + (1 - l2 * s2) / a2)  # A.9
    k[2, 2] = 3 * math.pi * INVERSE / (root * (l1 + l2))  # A.10
    k[2, 5] = -3 * math.pi * ln_h / (2 * root * (l1 + l2)) * s * c * (1 / s1 - 1 / s2) * (1 / l1 - 1 / l2)  # A.13
    k[3, 3] = -3 * math.pi * ln_h / root * (s * s / a1 + c * c / a2)  # A.14
    k[3, 4] = 3 * math.pi * ln_h / root * s * c * (-1 / a1 + 1 / a2)  # A.15
    k[4, 4] = -3 * math.pi * ln_h / root * (c * c / a1 + s * s / a2)  # A.16
    return np.triu(k) + np.triu(k, 1).T


def in_the_gaps_frame(k1: float, k2: float, c1: float, c2: float, chi: float) -> tuple[np.ndarray, tuple]:
    """The compilation's terms with MIME's sign convention, turned to the gap's frame, and the body's curvature there
    (``b1``, ``b2``, ``b12``)."""
    convention = np.diag([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    c, s = math.cos(chi), math.sin(chi)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # rows: the gap's axes in the body's frame
    both = np.block([[turn, np.zeros((3, 3))], [np.zeros((3, 3)), turn]])
    gap = both @ convention @ compilation(k1, k2, c1, c2, chi) @ convention @ both.T
    body = turn[:2, :2] @ np.diag([c1, c2]) @ turn[:2, :2].T
    return gap, (body[0, 0], body[1, 1], body[0, 1])


def covariant(k1: float, k2: float, b1: float, b2: float, b12: float) -> np.ndarray:
    return np.asarray(_elliptic.singular(*(jnp.asarray(v) for v in (k1, k2, b1, b2, b12, INVERSE, LOG))))


@settings(max_examples=200, deadline=None)
@given(CURVATURE, CURVATURE, CURVATURE, CURVATURE, ANGLE)
def test_the_elliptic_form_is_the_compilations_equations_term_for_term(k1, k2, c1, c2, chi):
    k1, k2 = max(k1, k2), min(k1, k2)
    expected, body = in_the_gaps_frame(k1, k2, c1, c2, chi)
    with planted() as plant:
        found = covariant(k1, k2, body[0], body[1], -body[2] if plant == "chi_sign" else body[2])
    # each entry on its row's and column's diagonal scale, the spin's (which has no singular term) the normal's
    size = np.sqrt(np.abs(np.diag(expected)))
    size[5] = size[2]
    assert np.max(np.abs(found - expected) / np.outer(size, size)) < 1e-12


@settings(max_examples=100, deadline=None)
@given(CURVATURE, CURVATURE, CURVATURE, CURVATURE)
def test_with_its_frames_aligned_the_elliptic_form_is_the_spheres(k1, k2, b1, b2):
    k1, k2 = max(k1, k2), min(k1, k2)
    sphere = np.asarray(_sphere.singular(*(jnp.asarray(v) for v in (k1, k2, b1, b2, INVERSE, LOG))))
    found = covariant(k1, k2, b1, b2, 0.0)
    assert np.max(np.abs(found - sphere)) <= 1e-12 * np.max(np.abs(sphere))


def test_one_contacts_opening_is_its_own_and_contacts_at_one_point_count_once():
    resolved = jnp.asarray(1e-4)
    for gap in (2e-5, 9e-5, 1.4e-4):
        alone = _seam.opening(jnp.asarray(gap), resolved)
        assert surface_opening(jnp.asarray([gap]), jnp.ones(1), resolved) == alone
        points = jnp.zeros((3, 3)).at[2].set(jnp.asarray([1.0, 0.0, 0.0]))
        shares = merged(points, jnp.asarray(1.0))
        assert np.allclose(np.asarray(shares), [0.5, 0.5, 1.0])
        together = surface_opening(jnp.asarray([gap, gap, 5e-4]), shares, resolved)
        assert abs(float(together) - float(alone)) <= 1e-15


def test_the_opening_is_continuous_where_two_contacts_gaps_cross():
    resolved = jnp.asarray(1e-4)
    shift = np.linspace(-2e-6, 2e-6, 401)
    found = np.array([float(surface_opening(jnp.asarray([5e-5 + d, 5e-5 - d]), jnp.ones(2), resolved))
                      for d in shift])
    assert np.max(np.abs(np.diff(found))) < 2.0 * np.max(np.abs(np.diff(5e-5 - np.abs(shift))))


def test_a_contacts_share_is_one_in_the_band_and_none_past_twice_it():
    resolved = jnp.asarray(1e-4)
    band = (1.0 + LIFT_WIDTH) * 1e-4
    assert float(in_band(jnp.asarray(0.999 * band), resolved)) == 1.0
    assert float(in_band(jnp.asarray(2.0 * band), resolved)) == 0.0
    assert 0.0 < float(in_band(jnp.asarray(1.5 * band), resolved)) < 1.0


def test_a_dilated_lumens_wall_reads_the_true_walls_zero_all_round():
    radius, delta = 2e-3, 1e-4
    lumen = {"centre": jnp.zeros(3), "axis": jnp.asarray([0.0, 0.0, 1.0]), "radius": radius}
    contact = NearContact(jnp.asarray([-radius - delta, 0.0, 0.0]), jnp.eye(3)[jnp.asarray([1, 2, 0])],
                          jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0))
    for angle in np.linspace(0.0, 2.0 * math.pi, 13):
        on_opened = jnp.asarray([(radius + delta) * math.cos(angle), (radius + delta) * math.sin(angle), 0.7e-3])
        with planted() as plant:
            read = ambient_point(on_opened, contact, jnp.asarray(delta), None if plant == "plane_ambient" else lumen)
        assert abs(float(jnp.hypot(read[0], read[1])) - radius) < 1e-12 * radius
        assert float(read[2]) == pytest.approx(0.7e-3, abs=1e-15)


@pytest.mark.parametrize("case, expected", [
    ("sphere on a plane", "point"), ("cylinder's side on a plane", "line"),
    ("sphere filling 0.9 of a lumen", "ring"), ("sphere at 0.34 of a lumen", "point")])
def test_each_contact_is_classed_by_its_curvatures_and_extent(case, expected):
    radius, resolved, spacing = 1e-3, 0.188e-3, 0.251e-3
    band = (1.0 + LIFT_WIDTH) * resolved
    lumen = {"sphere filling 0.9 of a lumen": radius / 0.9, "sphere at 0.34 of a lumen": radius / 0.34}.get(case)
    # a cylinder's side as a support shows it: curved round its axis alone, its measured footprint its length
    body = 1.0 / radius
    k2 = body - (1.0 / lumen if lumen else 0.0)
    footprint = 3e-3 if case.startswith("cylinder") else math.sqrt(2.0 * band * radius) + spacing
    wall = 1.0 / lumen if lumen else 0.0
    with planted() as plant:
        found = (lambda *a: jnp.asarray(0)) if plant == "class_point" else classified
        index = int(found(*(jnp.asarray(v) for v in (k2, body, footprint, wall, resolved, spacing))))
    assert _seam.CLASSES[index] == expected


@pytest.mark.parametrize("written, refused", [(None, True), (True, True), (False, False)])
def test_lubrication_twice_sees_the_hand_over_on_by_default(monkeypatch, written, refused):
    """#245's review's m8: the plan asks ``includes_lubrication`` of the entry after its defaults, so a hand-over on by
    default counts as one written on; a contact patched to add lubrication beside it is refused ``lubrication.twice``.
    Control: ``near_contact=False`` written plans."""
    from lubrication_cases import in_tube

    from mime.build import BuildRefused
    from mime.interactions import Contact
    from mime.units import mm

    monkeypatch.setattr(Contact, "includes_lubrication", lambda self: True)
    experiment = in_tube(0.2, blob=0.25, lubrication=written)
    experiment.attach(Contact.Projection(margin=mm(0.01)), body="ball", surface="tube.inner_surface")
    if refused:
        with pytest.raises(BuildRefused) as found:
            experiment.plan()
        assert "lubrication.twice" in found.value.codes()
    else:
        assert experiment.plan() is not None
