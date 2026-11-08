"""The rft rung's plan: the refusals it adds, on every profile (each is a model's limit, not an evidence standard).

- A body whose hull kind declares no centreline (a sphere; the screw, whose straight axis would carry no chirality) is
  refused, ``drag.no-centreline``, its message naming the guide's page on adding a centreline to a kind, and that page
  is in the documentation.
- Lighthill's coefficients on a straight centreline (no wavelength), ``drag.rft-coefficients``.
- A centreline point outside the wall's table at the start, ``drag.rft-wall-range``.
- Settings of other rungs, and of the rft rung on another rung, refused by name.
- The defaults: Lighthill's on a helix, the slender spheroid's on a capsule; one point a filament radius."""

from pathlib import Path

import pytest
from rft_cases import R6_HELIX, R6_SETTINGS, R6_TUBE, helix_body, rft

from mime.build import ArgumentRefused, BuildRefused
from mime.interactions import Drag
from mime.interactions.hydrodynamic import CENTRELINE_GUIDE, centreline_guide
from mime.scene import Capsule, Sphere
from mime.units import mm

PROFILES = ("exploratory", "research", "strict")
GUIDE = Path(__file__).resolve().parents[2] / "docs" / f"{CENTRELINE_GUIDE}.md"


def codes(experiment):
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    return refused.value.refusals


@pytest.mark.parametrize("profile", PROFILES)
def test_a_kind_without_a_centreline_is_refused_naming_the_guide(profile):
    experiment = helix_body("rft_plan", Sphere(radius=mm(0.1)), {}, rft(wall_kind="free_space"), profile=profile)
    found = [r for r in codes(experiment) if r.code == "drag.no-centreline"]
    assert len(found) == 1
    assert centreline_guide() in found[0].message
    assert GUIDE.is_file()


def test_the_screw_declares_no_centreline():
    from script_a import fl9

    from mime.registry import implements

    assert not implements(fl9(), "centreline")


@pytest.mark.parametrize("profile", PROFILES)
def test_lighthills_coefficients_on_a_straight_centreline_are_refused(profile):
    experiment = helix_body("rft_plan", Capsule(radius=mm(0.03), height=mm(1.0), axis=(1.0, 0.0, 0.0)), {},
                            rft(coefficients="lighthill", wall_kind="free_space"), profile=profile)
    assert "drag.rft-coefficients" in [r.code for r in codes(experiment)]


def test_a_centreline_outside_the_walls_table_is_refused():
    experiment = helix_body("rft_out", R6_HELIX, R6_SETTINGS, rft(wall_kind="tube"), tube=R6_TUBE,
                            offset=(0.0, 0.36e-3, 0.0))
    assert "drag.rft-wall-range" in [r.code for r in codes(experiment)]


@pytest.mark.parametrize("written", [dict(blob=0.5), dict(quadrature_refinement=4), dict(patches=("lateral",))])
def test_settings_of_other_rungs_are_refused(written):
    experiment = helix_body("rft_plan", R6_HELIX, R6_SETTINGS, rft(wall_kind="free_space", **written))
    with pytest.raises((ArgumentRefused, BuildRefused)) as refused:
        experiment.build()
    assert "rft rung reads none" in str(refused.value)


def test_coefficients_on_another_rung_are_refused():
    experiment = helix_body("rft_plan", Sphere(radius=mm(0.1)), {},
                            Drag.MRS(spacing=mm(0.05), coefficients="lighthill", wall_kind="free_space"))
    with pytest.raises((ArgumentRefused, BuildRefused)) as refused:
        experiment.build()
    assert "coefficients= is a setting of the rft rung" in str(refused.value)


def test_the_defaults():
    from mrs_cases import free_space

    built = free_space("rft_defaults", {"coil": R6_HELIX, "rod": Capsule(radius=mm(0.03), height=mm(1.0),
                                                                         axis=(1.0, 0.0, 0.0))},
                       rft(wall_kind="free_space")).build()
    coil, rod = (built.graph.get_node(f"drag:{name}") for name in ("coil", "rod"))
    assert coil.constants()["coefficients"] == "lighthill"
    assert rod.constants()["coefficients"] == "spheroid"
    assert coil.point_count == 94  # 2.79 mm of centreline at one point each 0.03 mm, rounded up
    assert rod.point_count == 36  # 1.06 mm tip to tip at one point each 0.03 mm
