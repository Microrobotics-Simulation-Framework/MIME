"""The geometry protocol's sampling (``Geometry.sampling``, ``mime.scene.quasi_uniform``) and hull query
(``Geometry.reach``), as a kind's code gives them; the conformance suite over every registered kind is
``tests/saved/test_sampling_walk.py``.

- **A choice among the quadrature's samples.** Every sample is one of the kind's quadrature samples at the dense counts,
  bit for bit, so it lies on the surface and carries the surface's normal; the weights are the cells' areas and sum
  to the patches' area; the choice is a JAX function of the parameters through the quadrature (a radius scaled moves
  every sample with it, and the weights by its square).
- **Made once, the same everywhere.** Two builds choose the same samples, in float32 and float64 alike (the choice is
  made in float64 whatever the run's precision), and a cache hit is the same object.
- **A density grades it.** Twice the density on one half of a sphere puts about twice the samples per area there, and
  raises the count by the density's mean, with no change to the protocol.
- **The kept versions** sample by their quadrature, exactly what the MRS rung took before 7m-a.
- **The reach's closed forms** for each kind against an independent computation of the same closed form (the brute
  force over samples is the walk's), and its gradient the farthest point.
- **The MRS rung refuses a kind that does not implement the sampling, by name.**"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.build import BuildRefused
from mime.registry import Geometry, register_geometry
from mime.scene import KEPT, Box, Capsule, Cylinder, Ellipsoid, Screw, Sphere, Tube, quasi_uniform
from mime.scene import _sampling
from mime.scene._sampling import QUASI_UNIFORM, quadrature, quadrature_arrays
from mime.units import mm
from mime.vocabulary import register_namespace
from mrs_cases import free_space, mrs
from script_a import fl9

FL9 = {"radius": 1.56e-3, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33}
SPHERE = (Sphere(radius=mm(1.0)), {"radius": 1e-3})
KINDS = [
    SPHERE,
    (Capsule(radius=mm(0.5), height=mm(2.0), axis=(0.0, 0.6, 0.8)), {"radius": 5e-4, "height": 2e-3,
                                                                      "axis": (0.0, 0.6, 0.8)}),
    (fl9(), FL9),
    (Cylinder(radius=mm(0.4), length=mm(1.6), axis=(0.0, 0.6, 0.8)), {"radius": 4e-4, "length": 1.6e-3,
                                                                       "axis": (0.0, 0.6, 0.8)}),
    (Box(size=mm((0.6, 0.9, 1.3))), {"size": (6e-4, 9e-4, 1.3e-3)}),
    (Ellipsoid(semi_axes=mm((0.5, 0.7, 1.1))), {"semi_axes": (5e-4, 7e-4, 1.1e-3)}),
]
IDS = [kind.name for kind, _ in KINDS]


def spacing_for(kind, settings, samples=300):
    size = kind.enclosing_radius(settings)
    return size * math.sqrt(4.0 * math.pi / samples)


def evaluated(kind, settings, sampling, scale=1.0):
    parameters = {name: (jnp.asarray(value, dtype=jnp.float64) * (scale if name != "axis" else 1.0)
                         if isinstance(value, (float, tuple)) else value) for name, value in settings.items()}
    if "wavenumber" in parameters:  # a screw scaled whole keeps its twist per radius
        parameters["wavenumber"] = jnp.asarray(settings["wavenumber"], dtype=jnp.float64)
        parameters["modulation"] = jnp.asarray(settings["modulation"], dtype=jnp.float64)
    held = sampling.samples(kind, parameters)
    return (np.asarray(held.points, dtype=np.float64), np.asarray(held.normals, dtype=np.float64),
            np.asarray(held.weights, dtype=np.float64))


@pytest.mark.parametrize(("kind", "settings"), KINDS, ids=IDS)
def test_each_sample_is_a_quadrature_sample_and_the_weights_are_the_cells_areas(kind, settings):
    spacing = spacing_for(kind, settings)
    sampling = kind.sampling(settings, spacing, tuple(kind.patches))
    assert sampling.method == QUASI_UNIFORM and sampling.chosen is not None and sampling.members is not None
    dense, weights, _ = quadrature_arrays(kind, settings, sampling.patches, sampling.counts)
    points, normals, cell_weights = evaluated(kind, settings, sampling)
    assert np.array_equal(points, dense[sampling.chosen])
    assert len(np.unique(sampling.chosen)) == len(sampling.chosen)
    members = sampling.members[sampling.members < len(dense)]
    assert np.array_equal(np.sort(members), np.arange(len(dense))), "the cells partition the quadrature"
    area = math.fsum(kind.area(settings, patch) for patch in kind.patches)
    assert abs(float(np.sum(cell_weights)) - area) <= 1e-13 * area
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0.0, atol=1e-14)
    own = sum(len(quadrature_arrays(kind, settings, (patch,), (kind.sample_counts(settings, spacing, patch),))[0])
              for patch in kind.patches)
    assert len(points) == own, "as many samples as its quadrature has at the spacing"


@pytest.mark.parametrize(("kind", "settings"), KINDS[:3], ids=IDS[:3])
def test_the_samples_move_with_the_kinds_parameters(kind, settings):
    """A study's derivative flows through the samples: the same choice, the shape scaled by 1.1, scales every sample's
    point by 1.1 and its weight by 1.21 (the screw at its own twist per radius)."""
    sampling = kind.sampling(settings, spacing_for(kind, settings), tuple(kind.patches))
    points, normals, weights = evaluated(kind, settings, sampling)
    scaled_points, scaled_normals, scaled_weights = evaluated(kind, settings, sampling, scale=1.1)
    assert np.allclose(scaled_points, 1.1 * points, rtol=1e-13, atol=1e-18)
    assert np.allclose(scaled_normals, normals, rtol=0.0, atol=1e-13)
    assert np.allclose(scaled_weights, 1.21 * weights, rtol=1e-12, atol=0.0)


def test_the_choice_is_the_same_twice_and_in_either_precision():
    kind, settings = KINDS[2]
    spacing = spacing_for(kind, settings)
    first = kind.sampling(settings, spacing, tuple(kind.patches))
    _sampling._MADE.clear()
    again = kind.sampling(settings, spacing, tuple(kind.patches))
    assert again is not first
    assert np.array_equal(first.chosen, again.chosen) and np.array_equal(first.members, again.members)
    assert kind.sampling(settings, spacing, tuple(kind.patches)) is again, "a cache hit is the same choice"
    _sampling._MADE.clear()
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        single = kind.sampling(settings, spacing, tuple(kind.patches))
    finally:
        jax.config.update("jax_enable_x64", held)
    assert np.array_equal(first.chosen, single.chosen) and np.array_equal(first.members, single.members)


def test_a_density_grades_the_sampling_with_no_change_to_the_protocol():
    """The density 2 on the upper half of a sphere: the count rises by the density's mean (1.5) and the upper half
    holds about two thirds of the samples, twice the lower's per area."""
    kind, settings = SPHERE
    spacing = spacing_for(kind, settings, 400)

    def upper(points):
        return np.where(points[:, 2] > 0.0, 2.0, 1.0)

    even = kind.sampling(settings, spacing, ("surface",))
    graded = kind.sampling(settings, spacing, ("surface",), upper)
    assert len(graded.chosen) == round(1.5 * len(even.chosen))
    points, _, _ = evaluated(kind, settings, graded)
    share = float(np.mean(points[:, 2] > 0.0))
    assert share == pytest.approx(2.0 / 3.0, abs=0.03)
    with pytest.raises(ValueError, match="above zero"):
        kind.sampling(settings, spacing, ("surface",), lambda points: np.zeros(len(points)))


@pytest.mark.parametrize("key", sorted(KEPT))
def test_each_kept_version_samples_by_its_quadrature(key):
    held, current = KEPT[key]
    successor = next(kind for kind, _ in KINDS if kind.name == key[0])
    assert successor.version == current and held.version == key[1] and issubclass(held, type(successor))
    kept = held(**{field.name: getattr(successor, field.name) for field in dataclasses.fields(successor)})
    settings = next(s for kind, s in KINDS if kind.name == key[0])
    spacing = spacing_for(kept, settings)
    sampling = kept.sampling(settings, spacing, tuple(kept.patches))
    assert sampling.chosen is None and sampling.members is None
    assert sampling.counts == quadrature(kept, settings, spacing, tuple(kept.patches)).counts == tuple(
        tuple(kept.sample_counts(settings, spacing, patch)) for patch in kept.patches)


def _reach(kind, settings, direction):
    parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                  for name, value in settings.items()}
    value, point = jax.value_and_grad(lambda d: kind.reach(parameters, d))(jnp.asarray(direction, dtype=jnp.float64))
    return float(value), np.asarray(point, dtype=np.float64)


def _closed(name, settings, d):
    """Each kind's farthest point along the unit ``d``, written from its definition here."""
    if name == "sphere":
        return settings["radius"] * d
    if name in ("capsule", "cylinder"):
        a = np.asarray(settings["axis"]) / np.linalg.norm(settings["axis"])
        along = float(d @ a)
        across = d - along * a
        half = (settings["height"] if name == "capsule" else settings["length"]) / 2.0
        if name == "capsule":
            return np.sign(along) * half * a + settings["radius"] * d
        return np.sign(along) * half * a + settings["radius"] * across / np.linalg.norm(across)
    if name == "box":
        return np.sign(d) * np.asarray(settings["size"]) / 2.0
    if name == "ellipsoid":
        axes = np.asarray(settings["semi_axes"])
        return axes * axes * d / np.linalg.norm(axes * d)
    if name == "screw":  # the envelope: the cylinder of its crests
        crest = settings["radius"] * (1.0 + settings["modulation"])
        across = d * np.array([0.0, 1.0, 1.0])
        return np.array([np.sign(d[0]) * settings["length"] / 2.0, 0.0, 0.0]) + crest * across / np.linalg.norm(across)
    raise KeyError(name)


@pytest.mark.parametrize(("kind", "settings"), KINDS, ids=IDS)
def test_the_reach_is_its_closed_form_and_its_gradient_the_farthest_point(kind, settings):
    rng = np.random.default_rng(7)
    for d in rng.normal(size=(20, 3)):
        d = d / np.linalg.norm(d)
        value, point = _reach(kind, settings, d)
        farthest = _closed(kind.name, settings, d)
        assert value == pytest.approx(float(farthest @ d), rel=1e-13)
        assert np.allclose(point, farthest, rtol=0.0, atol=1e-13 * kind.enclosing_radius(settings))
        doubled, _ = _reach(kind, settings, 2.0 * d)
        assert doubled == pytest.approx(2.0 * value, rel=1e-14), "homogeneous of degree one"


def test_along_an_axis_the_gradient_is_the_middle_of_the_farthest_face():
    box, box_settings = KINDS[4]
    value, point = _reach(box, box_settings, (1.0, 0.0, 0.0))
    assert value == pytest.approx(3e-4) and np.array_equal(point, np.array([3e-4, 0.0, 0.0]))
    screw = KINDS[2][0]
    value, point = _reach(screw, FL9, (-1.0, 0.0, 0.0))
    assert value == pytest.approx(FL9["length"] / 2.0) and np.array_equal(point, np.array([-FL9["length"] / 2.0,
                                                                                          0.0, 0.0]))


def test_the_tubes_reach_is_the_finite_tubes_from_its_centre():
    tube = Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(3.0), wall=mm(1.0), length=mm(20.0), centre=mm((1.0, 2.0, 3.0)))
    settings = {"axis": (0.0, 0.0, 1.0), "inner_radius": 3e-3, "wall": 1e-3, "length": 2e-2,
                "centre": (1e-3, 2e-3, 3e-3)}
    for d, expected in (((0.0, 0.0, 1.0), 3e-3 + 1e-2), ((1.0, 0.0, 0.0), 1e-3 + 4e-3),
                        ((0.6, 0.0, -0.8), 0.6 * 1e-3 - 0.8 * 3e-3 + 0.6 * 4e-3 + 0.8 * 1e-2)):
        value, _ = _reach(tube, settings, d)
        assert value == pytest.approx(expected, rel=1e-13)


register_namespace("unsampled", "0.0.1")


@register_geometry
@dataclasses.dataclass(frozen=True, kw_only=True)
class _Unsampled(Sphere):
    """A sphere that gives its quadrature and not the protocol's sampling: what a lab's kind written before 7m-a is."""

    name = "unsampled.ball"
    version = "0.0.1"
    sampling = Geometry.sampling  # type: ignore[assignment]


def test_the_mrs_rung_refuses_a_kind_that_does_not_implement_the_sampling_by_name():
    experiment = free_space("unsampled", {"ball": _Unsampled(radius=mm(1.0))}, mrs(mm(0.5)))
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    (refusal,) = [r for r in refused.value.refusals if r.code == "drag.no-samples"]
    assert "Geometry.sampling" in refusal.message and "unsampled.ball 0.0.1" in refusal.message


def test_quasi_uniform_refuses_a_count_its_dense_set_cannot_hold():
    kind, settings = SPHERE
    with pytest.raises(ValueError, match="dense set"):
        quasi_uniform(kind, settings, spacing_for(kind, settings), ("surface",), lambda p: np.full(len(p), 1e6))
