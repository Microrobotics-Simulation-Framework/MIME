"""``Drag.BEM`` in free space against closed forms, and the rung's refusals.

**The oracles.**

- The sphere's Stokes drag and torque, ``6 pi mu a`` and ``8 pi mu a^3`` (Stokes 1851), through the rung on the
  sphere kind: the regularised method's error is of the order of the blob over the radius, so each resolution is off by
  more than a stated tolerance and the error falls as the spacing falls; the first-order extrapolation of two
  resolutions, the spacing halved (``2 R(h/2) - R(h)``), is held within 0.5% in translation and rotation, with every
  coupling zero to rounding.
- The prolate spheroid's resistance (semi-axes 2, 1, 1), Kim & Karrila (2005), Table 3.4 (Chwang & Wu's 1975 forms), at
  the library: no 0.3 kind gives a spheroid's samples, so the test samples its surface itself (Gauss-Legendre in the
  axial coordinate, the trapezoid round the axis); the extrapolation of two resolutions within 1%, and the 0.2 line's
  own spheroid mesh (frozen, ``references/port_bem.npz``) within 1.5%, which is how close the 0.2 BEM came.

**Stated before the tests were written, from a probe of the convergence (in the pull request): ** at the two
resolutions here the sphere's extrapolation sits 0.32% (translation) and 0.34% (rotation) below Stokes, the
spheroid's 0.28% to 0.65%; the 0.2 spheroid 0.03% to 1.2% above.

**Generality** (the rule that the same BEM serves another microrobot): the round-trip case's ball, pill and screw each
give a symmetric positive definite resistance; the pill moves most easily along its axis; the screw couples turning
about its axis to moving along it, with the sign of its handedness.

**The controls**, each failing on every run: the free-space wall replaced by a non-zero correction (a Stokeslet image
of the body five radii below it); the sphere's radius read 1% too large, which moves the rotation 3%.

**The refusals:** a body whose hull kind gives no samples, or with no hull; a patch its kind does not have; a medium
filling a tube's lumen with the free-space wall; and settings of the other rung, a missing spacing or blob, and a wall
kind not built."""

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from bem_cases import BALL, KINDS, bem, bem_free_space, drag_node, free_space, resistance_at
from script_a import fl9

import mime.interactions.hydrodynamic as hydrodynamic
import mime.nodes.hydro.bem._walls as walls
from mime.build import BuildRefused
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.interactions import Drag
from mime.nodes.hydro.bem import BEMDragNode, WallCorrection, blob_squared, resistance
from mime.scene import SolidCylinder, Sphere, Tube
from mime.things import Body, State, Structure
from mime.units import dimensionless, mm

MU = 1e-3
#: The sphere's two spacings: its samples at (10, 20) and (20, 40), polar by round.
COARSE, FINE = mm(math.pi / 10 + 1e-5), mm(math.pi / 20 + 5e-6)
SPHERE_TOLERANCE, SPHEROID_TOLERANCE, SPHEROID_02_TOLERANCE = 5e-3, 1e-2, 1.5e-2


def stokes(radius: float) -> np.ndarray:
    return np.array([6 * math.pi * MU * radius] * 3 + [8 * math.pi * MU * radius ** 3] * 3)


def sphere_resistances(radius=BALL.radius) -> list[np.ndarray]:
    found = []
    for spacing in (COARSE, FINE):
        built = free_space("sphere_oracle", {"ball": Sphere(radius=radius)}, bem(spacing)).build()
        found.append(resistance_at(built, "ball"))
    return found


def sphere_misses(found: list[np.ndarray], radius: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float]:
    """Each resolution's diagonal against Stokes, the extrapolation's, and the largest coupling of translation and
    rotation, which the samples' symmetry through the centre makes zero, scaled by the diagonal's geometric mean."""
    expected = stokes(radius)
    each = np.stack([np.diag(r) / expected - 1 for r in found])
    extrapolated = np.diag(2 * found[1] - found[0]) / expected - 1
    coupling = max(float(np.abs(r[:3, 3:]).max() / math.sqrt(r[0, 0] * r[3, 3])) for r in found)
    return each, extrapolated, coupling


def test_the_sphere_converges_on_stokes_drag_and_torque():
    each, extrapolated, coupling = sphere_misses(sphere_resistances())
    print(f"\nagainst Stokes: (10, 20) {np.round(each[0], 5)}; (20, 40) {np.round(each[1], 5)}; extrapolated "
          f"{np.round(extrapolated, 5)}; largest coupling of translation and rotation {coupling:.1e}")
    assert np.all(np.abs(each[1]) < np.abs(each[0])) and np.all(np.abs(each[1]) > SPHERE_TOLERANCE)
    assert np.all(np.abs(extrapolated) < SPHERE_TOLERANCE)
    assert coupling < 1e-10


def spheroid_samples(a: float, b: float, along: int, around: int) -> tuple[np.ndarray, np.ndarray]:
    """The spheroid ``x^2/a^2 + (y^2 + z^2)/b^2 = 1`` by Gauss-Legendre in ``u = x / a`` and the trapezoid round x:
    the surface element ``b sqrt(b^2 u^2 + a^2 (1 - u^2)) du dphi``."""
    u, weight = np.polynomial.legendre.leggauss(along)
    phi = 2 * np.pi * (np.arange(around) + 0.5) / around
    uu, pp = np.meshgrid(u, phi, indexing="ij")
    s = np.sqrt(1 - uu ** 2)
    points = np.stack([a * uu, b * s * np.cos(pp), b * s * np.sin(pp)], axis=-1).reshape(-1, 3)
    element = b * np.sqrt(b ** 2 * uu ** 2 + a ** 2 * s ** 2) * np.outer(weight, np.full(around, 2 * np.pi / around))
    return points, element.ravel()


def prolate(a: float, b: float, mu: float = 1.0) -> np.ndarray:
    """The diagonal of a prolate spheroid's resistance about its centre, its axis along x: Kim & Karrila's ``X^A``,
    ``Y^A``, ``X^C``, ``Y^C`` with ``F = 6 pi mu a X^A U`` and ``T = 8 pi mu a^3 X^C Omega``, ``e`` the eccentricity."""
    e = math.sqrt(1 - b * b / (a * a))
    log = math.log((1 + e) / (1 - e))
    xa = 8 / 3 * e ** 3 / (-2 * e + (1 + e * e) * log)
    ya = 16 / 3 * e ** 3 / (2 * e + (3 * e * e - 1) * log)
    xc = 4 / 3 * e ** 3 * (1 - e * e) / (2 * e - (1 - e * e) * log)
    yc = 4 / 3 * e ** 3 * (2 - e * e) / (-2 * e + (1 + e * e) * log)
    return np.array([6 * math.pi * mu * a * xa] + [6 * math.pi * mu * a * ya] * 2
                    + [8 * math.pi * mu * a ** 3 * xc] + [8 * math.pi * mu * a ** 3 * yc] * 2)


def library(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(resistance(jnp.asarray(points), blob_squared(jnp.asarray(weights), 0.5), 1.0, jnp.zeros(3)))


def test_the_closed_form_spheroid_has_the_spheres_limit():
    assert np.allclose(prolate(1.0 + 1e-6, 1.0), stokes(1.0) / MU, rtol=1e-4)


def test_the_prolate_spheroid_converges_on_its_closed_form_and_the_02_mesh_comes_within_its_own_error():
    expected = prolate(2.0, 1.0)
    found = [library(*spheroid_samples(2.0, 1.0, along, 2 * along)) for along in (10, 20)]
    extrapolated = np.diag(2 * found[1] - found[0]) / expected - 1
    held = np.load(Path(__file__).parents[1] / "differential" / "references" / "port_bem.npz")
    old = np.diag(held["spheroid_coarse/R_library"]) / expected - 1
    print(f"\nagainst the closed form: extrapolated {np.round(extrapolated, 5)}; the 0.2 mesh {np.round(old, 5)}")
    assert np.all(np.abs(extrapolated) < SPHEROID_TOLERANCE)
    assert np.all(np.abs(old) < SPHEROID_02_TOLERANCE)


@pytest.fixture(scope="module")
def free_bodies():
    return bem_free_space().build()


@pytest.mark.parametrize("body", sorted(KINDS))
def test_each_kind_gives_a_symmetric_positive_definite_resistance(free_bodies, body):
    found = resistance_at(free_bodies, body)
    scaled = found / np.sqrt(np.outer(np.diag(found), np.diag(found)))
    assert np.array_equal(found, found.T) and float(np.linalg.eigvalsh(scaled).min()) > 0.0
    assert type(drag_node(free_bodies, body)) is BEMDragNode


def test_the_pill_moves_most_easily_along_its_axis_and_the_screw_couples_turning_to_moving_by_its_handedness():
    pill = resistance_at(free_space("pill", {"pill": KINDS["pill"]}, bem(mm(0.4))).build(), "pill",
                         (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    assert pill[1, 1] < min(pill[0, 0], pill[2, 2]) and pill[4, 4] < min(pill[3, 3], pill[5, 5])
    identity = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    right = resistance_at(free_space("screw", {"screw": fl9()}, bem(mm(0.9))).build(), "screw", identity)
    left = resistance_at(free_space("screw", {"screw": fl9(nu=-2.33)}, bem(mm(0.9))).build(), "screw", identity)
    assert abs(right[0, 3]) > 1e-3 * math.sqrt(right[0, 0] * right[3, 3])
    print(f"\nthe screw's axial coupling: right-handed {right[0, 3]:.6e}, left-handed {left[0, 3]:.6e}")
    assert math.isclose(left[0, 3], -right[0, 3], rel_tol=1e-9)


def image_of_a_plane_below(points, parameters, viscosity):
    """A Stokeslet image of each sample through the plane five radii below the ball: a wall's correction of a few
    percent, planted in the free space's place."""
    below = points * jnp.asarray((1.0, 1.0, -1.0)) - jnp.asarray((0.0, 0.0, 10e-3))
    r = points[:, None, :] - below[None, :, :]
    r2 = jnp.sum(r * r, axis=-1)
    blocks = (jnp.eye(3) * r2[..., None, None] + r[..., :, None] * r[..., None, :]) / (8 * math.pi * viscosity
                                                                                      * r2[..., None, None] ** 1.5)
    count = 3 * points.shape[0]
    return blocks.transpose(0, 2, 1, 3).reshape(count, count)


def test_control_the_free_space_wall_replaced_by_a_non_zero_correction_fails_the_sphere(monkeypatch):
    monkeypatch.setitem(walls._corrections, "free_space", WallCorrection("free_space", (), image_of_a_plane_below))
    each, extrapolated, _ = sphere_misses(sphere_resistances())
    print(f"\na wall planted: extrapolated {np.round(extrapolated, 5)}")
    assert np.any(np.abs(extrapolated) > 2 * SPHERE_TOLERANCE)


class _WiderNode(BEMDragNode):
    """The sphere's radius read 1% too large."""

    def samples(self, p):
        return super().samples({**p, "shape_radius": 1.01 * p["shape_radius"]})


def test_control_a_sphere_radius_off_by_one_percent_fails_the_sphere(monkeypatch):
    monkeypatch.setattr(hydrodynamic, "BEMDragNode", _WiderNode)
    each, extrapolated, _ = sphere_misses(sphere_resistances())
    print(f"\nthe radius 1% too large: extrapolated {np.round(extrapolated, 5)}")
    assert np.any(np.abs(extrapolated) > 2 * SPHERE_TOLERANCE)


def codes(experiment) -> set[str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    return set(refused.value.codes())


def test_a_body_with_no_hull_or_a_hull_kind_without_samples_is_refused():
    cylinder = SolidCylinder(radius=mm(1.0), length=mm(3.0), axis=(1.0, 0.0, 0.0))
    assert "drag.no-samples" in codes(free_space("no_samples", {"rod": cylinder}, bem(mm(0.5))))
    experiment = free_space("no_hull", {}, bem(mm(0.5)))
    experiment.add_body(Body("bare", dynamics=RigidInertial(), initial=State(position=mm((0.0, 0.0, 0.0)))))
    experiment.declare("bare.extent", mm(1.0), reason="no parts")
    experiment.attach(bem(mm(0.5)), body="bare", medium="water")
    assert "drag.no-samples" in codes(experiment)


def test_a_patch_the_kind_does_not_have_is_refused():
    assert "drag.patches" in codes(free_space("patches", {"ball": BALL}, bem(mm(0.5), patches=("lateral",))))


def test_a_medium_filling_a_tube_with_the_free_space_wall_is_refused():
    experiment = free_space("in_a_tube", {"ball": BALL}, bem(mm(0.5)), fills="tube.lumen")
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(3.0),
                                                             wall=mm(1.0), length=mm(50.0))))
    assert codes(experiment) == {"drag.wall-kind"}


@pytest.mark.parametrize(("law", "said"), [
    (lambda: Drag.BEM(blob=0.5, wall_kind="free_space"), "spacing is None m"),
    (lambda: Drag.BEM(spacing=mm(0.5), blob=0.0, wall_kind="free_space"), "blob is 0.0"),
    (lambda: Drag.BEM(spacing=mm(0.5), blob=0.5, wall_kind="tube"), "wall_kind='tube' is not a wall kind"),
    (lambda: Drag.BEM(spacing=mm(0.5), blob=0.5, wall_kind="free_space", weights=asset("dejongh_mlp_v2")),
     "weights= is a setting of the surrogate rung"),
    (lambda: Drag.Surrogate(weights=asset("dejongh_mlp_v2"), spacing=mm(0.5)), "spacing= is a setting of the bem rung"),
])
def test_a_rungs_settings_are_its_own_and_the_bems_are_written(law, said):
    with pytest.raises(BuildRefused) as refused:
        free_space("settings", {"ball": BALL}, law()).plan()
    assert "setting.value" in refused.value.codes() and said in str(refused.value)


def test_the_screws_kind_numbers_are_leaves_and_its_starts_held_as_built(free_bodies):
    node = drag_node(free_bodies, "screw")
    leaves = {key for key in node.param_specs() if key.startswith("shape_") and node.param_specs()[key].trainable}
    assert leaves == {"shape_radius", "shape_length", "shape_wavenumber", "shape_modulation"}
    assert node.constants()["shape_starts"] == 2 and dimensionless(0.33).value == node.constants()["shape_modulation"]
