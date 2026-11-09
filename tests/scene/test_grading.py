"""Feature-graded sampling (design #296): the kinds' declared features, the one grading policy, and the graded
sampling every kind takes through ``Geometry.sampling``'s density (``mime.scene.quasi_uniform``).

- **The features lie on their kinds' surfaces**, and the screw's crest's declared radius across it is the surface's own
  largest principal curvature's inverse, by finite differences of its parameterisation.
- **The policy** is the module's formula at the features, away from them, at a contact, and at the floor.
- **A graded sampling** keeps its generators apart from its samples: each sample is the dense point of its generator's
  cell nearest that cell's area centre, the cells the separated quadrature takes are the generators', the count is the
  uniform one times the density's mean, and the cells near a feature are smaller. **Its control:** the generators
  themselves, where the relaxation leaves them (the arm design #296 measured, whose error near a wall was larger than
  uniform's), lie at least one and a half times as far from their cells' area centres near the feature, every run
  (nothing is drawn).
- **The uniform sampling takes none of it**: no generators, its method the uniform one."""

import math

import numpy as np
import pytest

from mime.registry import implements
from mime.scene import Box, Cylinder, Screw, Sphere, quadrature_owners
from mime.scene import _sampling
from mime.scene._grading import CONTACT_SPACINGS, CREST_SHARE, EDGE_SHARE, Contact, graded_density, graded_patch
from mime.scene._sampling import FINEST, GRADED, QUASI_UNIFORM, quadrature_arrays, quadrature_counts, sampled_points
from mime.units import dimensionless, mm

FL9 = {"radius": 1.56e-3, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33}
SCREW = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
              starts=2)
CYLINDER = (Cylinder(radius=mm(1.0), length=mm(2.0), axis=(0.0, 0.6, 0.8)),
            {"radius": 1e-3, "length": 2e-3, "axis": (0.0, 0.6, 0.8)})
BOX = (Box(size=mm((0.6, 0.9, 1.3))), {"size": (6e-4, 9e-4, 1.3e-3)})


def test_only_the_kinds_with_edges_or_crests_declare_features():
    assert all(implements(kind, "features") for kind in (Cylinder, Box, Screw))
    assert not implements(Sphere, "features")


def test_the_cylinders_rims_lie_on_its_rims():
    kind, settings = CYLINDER
    axis = np.asarray(settings["axis"]) / np.linalg.norm(settings["axis"])
    rims = kind.features(settings, 1e-5)
    assert [(f.name, f.form, f.size) for f in rims] == [("rim", "edge", 0.0)] * 2
    for rim, end in zip(rims, (-1.0, 1.0), strict=True):
        along = rim.points @ axis
        across = np.linalg.norm(rim.points - along[:, None] * axis, axis=1)
        np.testing.assert_allclose(along, end * 1e-3, rtol=0, atol=1e-15)
        np.testing.assert_allclose(across, 1e-3, rtol=1e-12)
        assert np.max(np.linalg.norm(np.diff(rim.points, axis=0), axis=1)) <= 1e-5


def test_the_boxs_edges_join_its_corners():
    kind, settings = BOX
    size = np.asarray(settings["size"])
    edges = kind.features(settings, 2e-5)
    assert len(edges) == 12 and {f.form for f in edges} == {"edge"}
    for edge in edges:
        on_faces = np.isclose(np.abs(edge.points), size / 2, rtol=0, atol=1e-15)
        assert np.all(on_faces.sum(axis=1) >= 2), "every point of an edge lies on two faces"
        assert np.max(np.linalg.norm(np.diff(edge.points, axis=0), axis=1)) <= 2e-5 * (1 + 1e-12)


def surface(theta, x, settings=FL9, starts=2):
    """The screw's surface at the section's angle ``theta`` and axial position ``x``, as its quadrature places it."""
    radius, length, nu, epsilon = (settings[k] for k in ("radius", "length", "wavenumber", "modulation"))
    rho = radius * (1.0 + epsilon * np.sin(starts * theta))
    turned = theta + nu / radius * (x + length / 2.0)
    return np.stack([x, rho * np.cos(turned), rho * np.sin(turned)], axis=-1)


def test_the_screws_crests_lie_on_its_crests_and_their_radius_is_its_surfaces():
    """Each crest is the surface's line at the section's farthest angle, and its declared radius across it is the
    inverse of the surface's largest principal curvature there (finite differences of the parameterisation, to 1 %)."""
    found = SCREW.features(FL9, 1e-5)
    crests = [f for f in found if f.form == "crest"]
    rims = [f for f in found if f.form == "edge"]
    assert len(crests) == 2 and len(rims) == 2
    for j, crest in enumerate(crests):
        x = crest.points[:, 0]
        expected = surface(np.full_like(x, (math.pi / 2 + 2 * math.pi * j) / 2), x)
        np.testing.assert_allclose(crest.points, expected, rtol=0, atol=1e-15)
    for rim, end in zip(rims, (-1.0, 1.0), strict=True):
        np.testing.assert_allclose(rim.points[:, 0], end * FL9["length"] / 2, rtol=0, atol=1e-15)
    theta, x, h = math.pi / 4, 0.0, 1e-5
    d_theta = (surface(theta + h, x) - surface(theta - h, x)) / (2 * h)
    d_x = (surface(theta, x + h * 1e-3) - surface(theta, x - h * 1e-3)) / (2 * h * 1e-3)
    tt = (surface(theta + h, x) - 2 * surface(theta, x) + surface(theta - h, x)) / h ** 2
    xx = (surface(theta, x + h * 1e-3) - 2 * surface(theta, x) + surface(theta, x - h * 1e-3)) / (h * 1e-3) ** 2
    tx = (surface(theta + h, x + h * 1e-3) - surface(theta + h, x - h * 1e-3) - surface(theta - h, x + h * 1e-3)
          + surface(theta - h, x - h * 1e-3)) / (4 * h * h * 1e-3)
    normal = np.cross(d_theta, d_x)
    normal /= np.linalg.norm(normal)
    first = np.array([[d_theta @ d_theta, d_theta @ d_x], [d_theta @ d_x, d_x @ d_x]])
    second = np.array([[tt @ normal, tx @ normal], [tx @ normal, xx @ normal]])
    curvatures = np.linalg.eigvals(np.linalg.solve(first, second)).real
    assert crests[0].size == pytest.approx(1.0 / np.max(np.abs(curvatures)), rel=0.01)


def test_the_policy_is_the_formula_at_a_rim_away_from_it_and_at_the_floor():
    kind, settings = CYLINDER
    spacing = 0.3e-3
    density = graded_density(kind, settings, spacing, features=True)
    rim = density.traced[1].points[:1]
    axis = np.asarray(settings["axis"]) / np.linalg.norm(settings["axis"])
    inward = -(rim[0] - (rim[0] @ axis) * axis) / np.linalg.norm(rim[0] - (rim[0] @ axis) * axis)
    on_cap = rim + 0.1e-3 * inward
    middle = np.zeros((1, 3)) + (rim[0] - (rim[0] @ axis) * axis)
    assert density.spacing_at(rim)[0] == pytest.approx(EDGE_SHARE * spacing, rel=1e-3)
    assert density.spacing_at(on_cap)[0] == pytest.approx(EDGE_SHARE * spacing + 0.1e-3, rel=1e-3)
    assert density.spacing_at(middle)[0] == spacing
    assert density(middle)[0] == 1.0 and density(rim)[0] == pytest.approx((1 / EDGE_SHARE) ** 2, rel=2e-3)
    coarse = graded_density(SCREW, FL9, 0.4e-3, features=True)
    crest = coarse.traced[2]
    assert coarse.spacing_at(crest.points[:1])[0] == FINEST * 0.4e-3, "0.3 R_c is below the floor at 0.4 mm"
    fine = graded_density(SCREW, FL9, 0.15e-3, features=True)
    middle = fine.traced[2].points[len(fine.traced[2].points) // 2:][:1]
    assert fine.spacing_at(middle)[0] == pytest.approx(CREST_SHARE * crest.size, rel=1e-9), "mid-length, off the rims"


def test_the_contact_is_graded_by_its_height_above_the_contacts_plane():
    kind, settings = Sphere(radius=mm(1.0)), {"radius": 1e-3}
    spacing, gap = 0.3e-3, 0.15e-3
    density = graded_density(kind, settings, spacing, features=False, contact=Contact((0.0, 0.0, -1.0), gap))
    angle = np.radians([0.0, 20.0, 40.0, 90.0])
    points = 1e-3 * np.stack([np.sin(angle), np.zeros_like(angle), -np.cos(angle)], axis=1)
    height = 1e-3 * (1 - np.cos(angle))
    expected = np.clip((gap + height) / CONTACT_SPACINGS, FINEST * spacing, spacing)
    np.testing.assert_allclose(density.spacing_at(points), expected, rtol=2e-3)
    assert list(graded_patch(density, points)) == [True, True, True, False]
    with pytest.raises(ValueError, match="declares no features"):
        graded_density(kind, settings, spacing, features=True)


@pytest.fixture(scope="module")
def graded_cylinder():
    kind, settings = CYLINDER
    spacing = 0.4e-3
    density = graded_density(kind, settings, spacing, features=True)
    return kind, settings, spacing, density, kind.sampling(settings, spacing, tuple(kind.patches), density)


def offsets(kind, settings, sampling, indices):
    """Each cell's ``indices`` point's distance from the cell's area centroid, over the square root of its area."""
    points, weights, _ = quadrature_arrays(kind, settings, sampling.patches, sampling.counts)
    pad = len(points)
    found = []
    for i, row in enumerate(sampling.members):
        cell = row[row < pad]
        centre = np.sum(weights[cell, None] * points[cell], axis=0) / np.sum(weights[cell])
        found.append(np.linalg.norm(points[indices[i]] - centre) / math.sqrt(np.sum(weights[cell])))
    return np.asarray(found)


def test_a_graded_sampling_samples_each_cell_at_its_area_centre(graded_cylinder):
    kind, settings, spacing, density, sampling = graded_cylinder
    assert sampling.method == GRADED and sampling.generators is not None and sampling.chosen is not None
    points, weights, _ = quadrature_arrays(kind, settings, sampling.patches, sampling.counts)
    pad = len(points)
    quantum = _sampling.tie_quantum(points)
    for i, row in enumerate(sampling.members):
        cell = row[row < pad]
        assert sampling.chosen[i] in cell and sampling.generators[i] in cell
        centre = np.sum(weights[cell, None] * points[cell], axis=0) / np.sum(weights[cell])
        away = np.floor(np.sum((points[cell] - centre) ** 2, axis=1) / quantum)
        assert sampling.chosen[i] == cell[np.lexsort((cell, away))[0]]
    own = len(quadrature_arrays(kind, settings, sampling.patches, quadrature_counts(kind, settings, spacing,
                                                                                    sampling.patches))[0])
    assert len(sampling.chosen) == round(own * float(np.sum(weights * density(points)) / np.sum(weights)))
    assert float(np.sum(sampling.samples(kind, settings).weights)) == pytest.approx(float(np.sum(weights)), rel=1e-12)


def test_the_cells_near_a_rim_are_smaller(graded_cylinder):
    kind, settings, spacing, density, sampling = graded_cylinder
    held = sampling.samples(kind, settings)
    size = np.sqrt(np.asarray(held.weights))
    near = graded_patch(density, np.asarray(held.points))
    assert near.any() and (~near).any()
    assert np.mean(size[near]) < 0.7 * np.mean(size[~near])
    ratio = np.asarray(_sampling.nearest(np.asarray(held.points), np.asarray(held.points), k=2)[0][:, 1]) / \
        density.spacing_at(np.asarray(held.points))
    assert 0.35 < np.percentile(ratio, 5) and 0.6 < np.median(ratio) < 1.0 and np.percentile(ratio, 95) < 1.25


def test_the_separated_quadratures_cells_are_the_generators(graded_cylinder):
    """The owner map is each quadrature point's nearest generator; the nearest sample's would differ (the control that
    the generators are read)."""
    kind, settings, spacing, _, sampling = graded_cylinder
    counts = quadrature_counts(kind, settings, spacing / 6, sampling.patches)
    owners = quadrature_owners(kind, settings, sampling, counts)
    points = quadrature_arrays(kind, settings, sampling.patches, counts)[0]
    generators = quadrature_arrays(kind, settings, sampling.patches, sampling.counts)[0][sampling.generators]
    nearest = _sampling.nearest(generators, points)[1]
    assert np.mean(owners == nearest) > 0.999
    samples = sampled_points(kind, settings, sampling)
    assert np.mean(_sampling.nearest(samples, points)[1] != owners) > 0.01


def test_the_generators_lie_farther_from_their_cells_area_centres(graded_cylinder):
    """The control: near the rims the relaxed generators lie at least one and a half times as far from their cells'
    area centres as the graded samples do (measured 1.97 times here; 2.04 on design #296's cylinder)."""
    kind, settings, _, density, sampling = graded_cylinder
    near = graded_patch(density, sampled_points(kind, settings, sampling))
    placed = offsets(kind, settings, sampling, sampling.chosen)
    relaxed = offsets(kind, settings, sampling, sampling.generators)
    assert np.mean(relaxed[near]) > 1.5 * np.mean(placed[near])


def test_the_uniform_sampling_takes_none_of_it():
    kind, settings = CYLINDER
    held = kind.sampling(settings, 0.4e-3, tuple(kind.patches))
    assert held.method == QUASI_UNIFORM and held.generators is None


def test_a_graded_density_is_at_most_the_floors():
    kind, settings = CYLINDER
    with pytest.raises(ValueError, match="at most 16"):
        kind.sampling(settings, 0.4e-3, tuple(kind.patches), lambda points: np.full(len(points), 17.0))
