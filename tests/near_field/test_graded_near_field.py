"""Graded bodies in a medium's near field (``drag.grading-near-field``): refused first, before the near field's other
refusals of the same medium (its members at different rates, ``drag.near-field-rates``; a kept version beside it,
``drag.near-field-mixed``), so that a user sees it on the first round; and its reason fits the medium: in free space
the cross-kernel's reach measured on uniform samplings only, with a wall also the wall image and ``off_patch``. Its
route says that a medium of the graded body's own drops its coupling to the others. The controls: the same media with
no grading are refused by the other codes alone, so the graded refusal is the grading's."""

import pytest
from near_field_cases import RADIUS, bodies, mrs, spheres

from mime.build import BuildRefused
from mime.interactions import UncoupledDrag
from mime.scene import Cylinder
from mime.units import m, ms

SPACING = 0.6e-3


def refused(experiment) -> BuildRefused:
    with pytest.raises(BuildRefused) as raised:
        experiment.plan()
    return raised.value


def placed() -> dict:
    """Three bodies 0.4 mm apart, the first a cylinder (a kind with features to grade), the others spheres."""
    found = spheres(0.4e-3, 3)
    found["s0"] = (Cylinder(radius=m(RADIUS), length=m(2.0 * RADIUS), axis=(0.0, 0.0, 1.0)), found["s0"][1])
    return found


def laws(graded: bool, *, kept: bool = False, wall: str = "free_space") -> dict:
    grading = ({"contact": (0.0, 0.0, -1.0), "contact_gap": m(0.3e-3)} if wall == "plane" else
               {"grading": "features"}) if graded else {}
    third = UncoupledDrag.MRS(spacing=m(SPACING), wall_kind=wall) if kept else mrs(SPACING, wall_kind=wall)
    return {"s0": mrs(SPACING, wall_kind=wall, **grading), "s1": mrs(SPACING, wall_kind=wall), "s2": third}


@pytest.mark.parametrize("graded", [True, False], ids=["graded", "control"])
def test_members_at_different_rates_see_the_graded_refusal_first(graded):
    codes = refused(bodies("rates", placed(), laws(graded), rate=ms(10.0))).codes()
    assert codes == (["drag.grading-near-field", "drag.near-field-rates"] if graded else ["drag.near-field-rates"])


@pytest.mark.parametrize("graded", [True, False], ids=["graded", "control"])
def test_a_medium_of_mixed_versions_sees_the_graded_refusal_first(graded):
    codes = refused(bodies("mixed", placed(), laws(graded, kept=True))).codes()
    assert codes == (["drag.grading-near-field", "drag.near-field-mixed"] if graded else ["drag.near-field-mixed"])


def test_in_free_space_the_reason_is_the_cross_kernels_and_the_route_names_the_lost_coupling():
    (refusal,) = refused(bodies("free", placed(), laws(True))).refusals
    assert refusal.code == "drag.grading-near-field"
    assert "cross-kernel's reach" in refusal.message and "off_patch" not in refusal.message
    assert "drops its hydrodynamic coupling" in refusal.action


def test_with_a_wall_the_reason_adds_the_wall_image_and_the_flag():
    (refusal,) = refused(bodies("walled", placed(), laws(True, wall="plane"), plane=-2e-3)).refusals
    assert refusal.code == "drag.grading-near-field"
    assert "cross-kernel's reach" in refusal.message and "off_patch" in refusal.message
