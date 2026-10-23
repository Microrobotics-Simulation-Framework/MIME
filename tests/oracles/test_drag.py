"""The drag's surrogate rung against what can check it, each with a control that fails (float64, per module).

- **Accuracy against the BEM values the surrogate was checked against** (the 0.2 line's six offsets in the 1/4"
  tube, ``references/dejongh_bem_offsets.*``): the swim speed the surrogate gives, against the BEM's. This bounds the
  surrogate against that BEM (float32, with the 2026-05-21 wall table, which was never checked against Fourier-Bessel
  series), not against the flow. At 0.30 of the radius it is within the 0.2 line's own assertion, 5%; at 0.20, 0.25
  and 0.28 it misses the 1% that line's docstring and metadata state (2.6%, 16.2%, 10.2%), as the 0.2 line measured,
  and the test holds those numbers so a change shows. The control: the coupling blocks scaled by ``mu R`` instead of
  ``mu R^2`` (the 0.2 node's own stated hazard) misses 5% at 0.30.
- **The three fixes.** nu: 2.33 builds and 3.6 is refused naming the entry, the value, the range and where it comes
  from; with the 0.2 line's declared range, 3.6 builds. kappa likewise. The axis: a tube along x and the same tube along
  z give resistances related by the rotation between them, on and off the axis; moving along the axis changes nothing;
  the 0.2 line's +z frame in the x tube does not. On the axis, and 1e-30 m from it, the value and the derivative with
  respect to the position are finite, and match finite differences where those can tell (along the axis, and turning
  the body about it); the 0.2 line's formula gives NaN there.
- **Lubrication**: the correction is the 0.2 line's formula, it is finite as the gap closes (the 1 nm floor), and the
  clamp holds the surrogate's offset beyond 0.95 of the room to the wall, reporting it.
- **The Stokes sphere**: its resistance is the closed form, and a body under a constant force and torque reaches
  ``F / (6 pi mu a)`` and ``T / (8 pi mu a^3)``.
- **The entry**: a medium filling no region, or a region with no axis, is refused with that reason; weights whose
  sha256 is not the packaged file's are refused; ``Drag.Surrogate`` saves as ``Drag(rung="surrogate")``; the same
  experiment with the Stokes sphere in place builds, steps and delivers into the same targets, and a rung delivering
  into a target the body does not accept is refused.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rigid_oracles
from hydro_oracles import PortFrameNode, StokesSphere, lubrication, sphere_resistance
from script_a import script_a

from mime import Body, Experiment, Medium, Scene, State, Structure
from mime import asset as scene_asset
from mime.build import BuildRefused
from mime.core import register_shape
from mime.data import Asset, AssetError, asset
from mime.dynamics import RigidInertial
from mime.gradients import GradientStatus
from mime.interactions import Drag, Lubrication
from mime.nodes.hydro import GAP_FLOOR, SurrogateDragNode
from mime.registry import Delivery, Geometry, register_geometry, register_interaction
from mime.surrogates import DEJONGH_MLP_V2, DeclaredRange
from mime.units import Length, Quantity, dimensionless, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, rad_per_s, s
from mime.vocabulary import Interface, TargetKind, define, register_namespace

REFERENCES = Path(__file__).parent / "references"
DT, MU, NU, LENGTH, RADIUS, TUBE, EPSILON = 5e-4, 1e-3, 2.33, 7.47e-3, 1.56e-3, 3.175e-3, 1.5e-4
SHA = asset("dejongh_mlp_v2").sha256
Z, X = (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def node(axis=Z, *, epsilon: float | None = EPSILON, cls=SurrogateDragNode, tube=TUBE, viscosity=MU,
         reference_point="centre"):
    return cls("drag", DT, weights="dejongh_mlp_v2", weights_sha256=SHA, viscosity=viscosity, helix_wavenumber=NU,
               length=LENGTH, radius=RADIUS, vessel_centre=(0.0, 0.0, 0.0), vessel_axis=axis, vessel_radius=tube,
               lubrication_epsilon=epsilon, reference_point=reference_point)


def step(n: SurrogateDragNode, pose) -> dict[str, np.ndarray]:
    out = n.update(n.initial_state(), {"pose": jnp.asarray(pose, dtype=jnp.float64)}, DT)
    return {key: np.asarray(value) for key, value in out.items()}


def resistance(n: SurrogateDragNode, pose) -> np.ndarray:
    return step(n, pose)["resistance"]


def blockwise(found: np.ndarray, expected: np.ndarray) -> float:
    """The largest difference in each 3 x 3 block relative to that block's largest magnitude, the worst block's."""
    worst = 0.0
    for rows in (slice(0, 3), slice(3, 6)):
        for columns in (slice(0, 3), slice(3, 6)):
            scale = float(np.max(np.abs(expected[rows, columns])))
            worst = max(worst, float(np.max(np.abs(found[rows, columns] - expected[rows, columns]))) / scale)
    return worst


# -- accuracy against the BEM ---------------------------------------------------------------------------------------

BEM = np.load(REFERENCES / "dejongh_bem_offsets.npz")
RECORD = json.loads((REFERENCES / "dejongh_bem_offsets.json").read_text())
OMEGA = 2 * math.pi * 10.0
#: The 0.2 test's: the 1/4" tube's confinement, by which it turned an offset fraction into the MLP's offset.
KAPPA = 0.491


def to_si(r_nd: np.ndarray, coupling=RADIUS ** 2) -> np.ndarray:
    out = np.zeros_like(r_nd)
    out[:3, :3] = MU * RADIUS * r_nd[:3, :3]
    out[:3, 3:] = MU * coupling * r_nd[:3, 3:]
    out[3:, :3] = MU * coupling * r_nd[3:, :3]
    out[3:, 3:] = MU * RADIUS ** 3 * r_nd[3:, 3:]
    return out


def swim_speed(r: np.ndarray) -> float:
    """The force-free speed along the tube turning at 10 Hz about it: ``U = -R_FU^-1 R_FOmega [0, 0, omega]``."""
    return float(-np.linalg.solve(r[:3, :3], r[:3, 3:] @ np.array([0.0, 0.0, OMEGA]))[2])


def surrogate_speed(fraction: float, scaled=lambda r: r) -> float:
    offset = fraction / KAPPA * RADIUS
    return swim_speed(scaled(resistance(node(epsilon=None, reference_point="end"), (offset, 0.0, 0.0, *IDENTITY))))


def test_the_surrogate_against_the_bem_values_it_was_checked_against():
    points = RECORD["envelope_0_2"]["points"]
    errors = {}
    for fraction, r_bem, point in zip(BEM["offsets"], BEM["R_BEMs"], points, strict=True):
        assert point["offset_frac"] == pytest.approx(float(fraction))
        bem, mlp = swim_speed(to_si(r_bem)), surrogate_speed(float(fraction))
        assert bem * 1e3 == pytest.approx(point["v_BEM_mm_s"], rel=1e-12)
        # The 0.2 line evaluated the MLP at a float32 position: its speeds agree with 0.3's to that rounding.
        assert mlp * 1e3 == pytest.approx(point["v_MLP_mm_s"], rel=1e-5)
        errors[round(float(fraction), 2)] = abs(mlp - bem) / abs(bem)
    print(f"\nrelative swim-speed error against the BEM: {errors}")
    assert errors[0.3] < 0.05
    assert errors == pytest.approx({0.2: 0.0260, 0.25: 0.1620, 0.28: 0.1016, 0.3: 0.0028, 0.32: 0.0826, 0.35: 0.8346},
                                   abs=1e-3)


def about(r_nd: np.ndarray, shift: float) -> np.ndarray:
    """``r_nd`` about a point ``shift`` (in units of R) further along the body's axis (z): ``[F; T - d x F]`` against
    ``[U - d x Omega; Omega]``."""
    d = np.array([[0.0, -shift, 0.0], [shift, 0.0, 0.0], [0.0, 0.0, 0.0]])
    eye, zero = np.eye(3), np.zeros((3, 3))
    return np.block([[eye, zero], [-d, eye]]) @ r_nd @ np.block([[eye, d], [zero, eye]])


def test_the_surrogates_resistance_is_about_the_end_of_the_bodys_axis():
    """MIME-ANO-112: moving the reference by L/2 along the axis, towards the body's centre, takes the coupling across
    the axis from thousands (non-dimensional) to under 1% of that in the BEM values and under 10% in the surrogate's;
    moving it the other way (the control) doubles it. So the training's
    resistances, and the surrogate's, are about the end of the axis, and the port applies them about the centre."""
    half = LENGTH / RADIUS / 2
    for fraction, r_bem in zip(BEM["offsets"][:4], BEM["R_BEMs"][:4], strict=True):
        surrogate = resistance(node(epsilon=None, reference_point="end"),
                               (fraction / KAPPA * RADIUS, 0.0, 0.0, *IDENTITY))
        r_nd = surrogate / np.kron(np.array([[MU * RADIUS, MU * RADIUS ** 2], [MU * RADIUS ** 2, MU * RADIUS ** 3]]),
                                   np.ones((3, 3)))
        # The surrogate is held to the BEM's pattern within its own accuracy against it (up to 16% at these points).
        for held, within in ((r_bem, 0.01), (r_nd, 0.1)):
            assert held[0, 4] / held[0, 0] == pytest.approx(half, rel=within)
            centred, away = about(held, half), about(held, -half)
            assert abs(centred[0, 4]) < within * abs(held[0, 4]) and abs(centred[1, 3]) < within * abs(held[1, 3])
            assert abs(away[0, 4]) > 1.9 * abs(held[0, 4])


#: A body along the tube (z) with its x axis, the helix's, pointing along +z, and the same body turned end for end.
ALONG = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)
END_FOR_END = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
NODE_SCALE = np.kron(np.array([[MU * RADIUS, MU * RADIUS ** 2], [MU * RADIUS ** 2, MU * RADIUS ** 3]]), np.ones((3, 3)))


def test_about_the_centre_the_cross_axis_coupling_collapses_and_the_opposite_shift_doubles_it():
    """drag 0.2.0 (MIME-ANO-112 resolved): the node's own resistance, body along the tube. About the end (the
    test-only switch) the force across the axis from a rotation across it is about L/(2R) times the translational
    resistance; about the centre, under 10% of that (the surrogate's own asymmetry, as 4c1 measured it). The control:
    the shift made the other way doubles it."""
    for fraction in BEM["offsets"][:4]:
        pose = (fraction / KAPPA * RADIUS, 0.0, 0.0, *ALONG)
        end = resistance(node(epsilon=None, reference_point="end"), pose) / NODE_SCALE
        centre = resistance(node(epsilon=None), pose) / NODE_SCALE
        assert end[0, 4] / end[0, 0] == pytest.approx(LENGTH / RADIUS / 2, rel=0.1)
        assert abs(centre[0, 4]) < 0.1 * abs(end[0, 4]) and abs(centre[1, 3]) < 0.1 * abs(end[1, 3])
        assert np.allclose(centre, about(end, LENGTH / RADIUS / 2), rtol=1e-12, atol=1e-12 * np.abs(end).max())
        away = about(end, -LENGTH / RADIUS / 2)
        assert abs(away[0, 4]) > 1.9 * abs(end[0, 4])


def test_about_the_centre_the_resistance_stays_spd_and_its_axial_block_is_unchanged():
    for fraction in (0.0, 0.1, 0.2, 0.3):
        pose = (fraction / KAPPA * RADIUS + 1e-6, 0.0, 0.0, *ALONG)
        end = resistance(node(reference_point="end"), pose)
        centre = resistance(node(), pose)
        scaled = centre / np.sqrt(np.outer(np.diag(centre), np.diag(centre)))
        assert np.allclose(centre, centre.T, rtol=0.0, atol=1e-13 * np.abs(centre).max())
        assert float(np.min(np.linalg.eigvalsh((scaled + scaled.T) / 2))) > 0.0
        axial = np.ix_([2, 5], [2, 5])
        assert np.allclose(centre[axial], end[axial], rtol=1e-12, atol=0.0)


def test_a_body_turned_end_for_end_gets_the_mirror_image_shift():
    """Turned end for end, the body's zeta runs the other way: its resistance is the half turn about the offset's
    direction of the one it had. The control: with the end fixed whatever the orientation (the 0.2 line's), it is
    not."""
    pose, turned_pose = (6e-4, 0.0, 0.0, *ALONG), (6e-4, 0.0, 0.0, *END_FOR_END)
    half_turn = np.zeros((6, 6))
    half_turn[:3, :3] = half_turn[3:, 3:] = np.diag([1.0, -1.0, -1.0])
    for reference_point, mirrored in (("centre", True), ("end", False)):
        n = node(reference_point=reference_point)
        found, expected = resistance(n, turned_pose), half_turn @ resistance(n, pose) @ half_turn.T
        assert blockwise(found, expected) < 1e-12 if mirrored else blockwise(found, expected) > 1e-2


def test_a_planted_unit_scaling_misses_the_bem_at_030():
    bem = swim_speed(to_si(BEM["R_BEMs"][3]))
    planted = surrogate_speed(0.30, scaled=lambda r: np.block([[r[:3, :3], r[:3, 3:] / RADIUS],
                                                               [r[3:, :3] / RADIUS, r[3:, 3:]]]))
    assert abs(planted - bem) / abs(bem) > 0.05


# -- the training domain --------------------------------------------------------------------------------------------

def refusal(experiment: Experiment, code: str) -> str:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    found = [r for r in refused.value.refusals if r.code == code]
    assert found, refused.value.codes()
    return f"{found[0].message} | {found[0].action}"


def test_a_wavenumber_inside_the_training_data_plans_and_one_outside_it_is_refused():
    script_a(nu=dimensionless(2.33)).plan()
    text = refusal(script_a(nu=dimensionless(3.6)), "drag.outside-training")
    assert "interaction drag:umr (drag 0.2.0)" in text
    assert "helix_wavenumber nu from 0.25 to 3.5" in text and "umr.helix_wavenumber is 3.6" in text
    assert "the 0.2 line declared 2.33 to 3.7" in text


def test_with_the_02_lines_declared_range_the_same_wavenumber_would_plan(monkeypatch):
    """The control: 3.6 lies inside the 0.2 line's declared range, which is not its training data's."""
    old = replace(DEJONGH_MLP_V2, shape=(DeclaredRange("helix_wavenumber", "nu", 2.33, 3.7, "the 0.2 line's"),
                                         DEJONGH_MLP_V2.shape_range("confinement")))
    monkeypatch.setattr(Drag, "surrogates", {old.asset: old})
    script_a(nu=dimensionless(3.6)).plan()


def test_a_tube_too_tight_for_the_training_data_is_refused():
    text = refusal(script_a(inner_radius=mm(2.3)), "drag.outside-training")
    assert "confinement kappa from 0.25 to 0.66" in text and "umr.radius / the tube's radius" in text


def test_the_training_domain_is_declared_data_on_the_entry():
    surrogate = Drag.surrogates["dejongh_mlp_v2"]
    assert (surrogate.shape_range("helix_wavenumber").low, surrogate.shape_range("helix_wavenumber").high) == (0.25,
                                                                                                              3.5)
    assert (surrogate.shape_range("confinement").low, surrogate.shape_range("confinement").high) == (0.25, 0.66)
    assert surrogate.offset_fraction.high == 0.3 and (surrogate.modulation, surrogate.starts) == (0.33, 2)


# -- the axis -------------------------------------------------------------------------------------------------------

#: A quarter turn about y, taking z to x: the tube along z becomes the tube along x.
TURN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
TURN_Q = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])


def product(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def poses(count: int, seed: int) -> list[np.ndarray]:
    """Bodies in the tube along z: off the axis (inside the clamp and beyond it), and on it."""
    rng = np.random.default_rng(seed)
    found = []
    for k in range(count):
        q = rng.normal(size=4)
        radial = [0.0, 1e-30, 3e-4, 1e-3, 1.3e-3][k % 5]
        angle = rng.uniform(0, 2 * math.pi)
        found.append(np.array([radial * math.cos(angle), radial * math.sin(angle), rng.uniform(-0.02, 0.02),
                               *(q / np.linalg.norm(q))]))
    return found


def turned(pose: np.ndarray) -> np.ndarray:
    return np.concatenate([TURN @ pose[:3], product(TURN_Q, pose[3:])])


def turned_operator(r: np.ndarray) -> np.ndarray:
    pair = np.zeros((6, 6))
    pair[:3, :3] = pair[3:, 3:] = TURN
    return pair @ r @ pair.T


def test_a_tube_along_x_and_the_same_tube_along_z_give_resistances_related_by_the_turn():
    along_z, along_x = node(Z), node(X)
    worst = max(blockwise(resistance(along_x, turned(pose)), turned_operator(resistance(along_z, pose)))
                for pose in poses(25, 1))
    print(f"\nworst blockwise difference {worst:.2e}")
    assert worst < 1e-12


def test_moving_along_the_tubes_axis_changes_nothing():
    along_x = node(X)
    for pose in poses(10, 2):
        moved = turned(pose)
        shifted = moved + np.array([0.037, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(resistance(along_x, moved), resistance(along_x, shifted))


def test_the_02_lines_frame_in_a_tube_along_x_is_not_the_turned_resistance():
    """The control: the 0.2 line's +z frame reads the body's position along an x tube as an offset."""
    along_z, planted = node(Z), node(X, cls=PortFrameNode)
    pose = poses(5, 3)[2]
    assert blockwise(resistance(planted, turned(pose)), turned_operator(resistance(along_z, pose))) > 1e-2
    assert step(planted, turned(pose))["clamp_fired"] == 1.0


# -- the axis: finite values and derivatives ----------------------------------------------------------------------

def grad_in_position(n: SurrogateDragNode, at, orientation=IDENTITY) -> np.ndarray:
    weights = jnp.asarray(np.random.default_rng(4).normal(size=(6, 6)))

    def scalar(position):
        out = n.update(n.initial_state(), {"pose": jnp.concatenate([position, jnp.asarray(orientation)])}, DT)
        return jnp.sum(weights * out["resistance"])

    return np.asarray(jax.grad(scalar)(jnp.asarray(at, dtype=jnp.float64)))


@pytest.mark.parametrize("offset", [0.0, 1e-30], ids=["on the axis", "1e-30 m off it"])
def test_on_the_axis_the_value_and_its_derivative_are_finite(offset):
    along_x = node(X)
    at = (0.012, offset, 0.0)
    assert np.all(np.isfinite(resistance(along_x, (*at, *IDENTITY))))
    assert np.all(np.isfinite(grad_in_position(along_x, at)))


def test_the_02_lines_formula_differentiates_to_nan_on_the_axis():
    """The control: ``sqrt(x^2 + y^2)`` and ``arctan2(y, x)`` at zero."""
    assert np.any(np.isnan(grad_in_position(node(Z, cls=PortFrameNode), (0.0, 0.0, 0.012))))


def central(f, h: float) -> np.ndarray:
    return (np.asarray(f(h)) - np.asarray(f(-h))) / (2 * h)


def turning(n: SurrogateDragNode, at: Any, angle: Any) -> Any:
    """The resistance of a body on the axis at ``at``, tilted, then turned by ``angle`` about the tube's axis (x)."""
    half = angle / 2
    turn = jnp.stack([jnp.cos(half), jnp.sin(half), jnp.zeros_like(half), jnp.zeros_like(half)])
    tilt = jnp.asarray([math.cos(0.3), 0.0, math.sin(0.3), 0.0])
    w1, x1, y1, z1 = turn[0], turn[1], turn[2], turn[3]
    w2, x2, y2, z2 = tilt[0], tilt[1], tilt[2], tilt[3]
    body = jnp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return n.update(n.initial_state(), {"pose": jnp.concatenate([jnp.asarray(at), body])}, DT)["resistance"]


def test_on_the_axis_the_derivatives_match_finite_differences_where_they_can_tell():
    """Along the axis (zero both ways), and turning the body about the axis, which turns the frame the surrogate takes
    there; across the axis the resistance is discontinuous (the surrogate is not symmetric about it), so no finite
    difference can tell."""
    along_x = node(X)
    on_axis = np.array([0.012, 0.0, 0.0])
    tangent = grad_in_position(along_x, on_axis, orientation=(math.cos(0.3), 0.0, math.sin(0.3), 0.0))
    assert np.all(np.isfinite(tangent)) and tangent[0] == 0.0
    _, forward = jax.jvp(lambda a: turning(along_x, on_axis, a), (jnp.asarray(0.2),), (jnp.asarray(1.0),))
    difference = central(lambda h: turning(along_x, on_axis, jnp.asarray(0.2 + h)), 1e-6)
    print(f"\nlargest derivative turning the body on the axis: {float(np.max(np.abs(difference))):.3e}")
    assert float(np.max(np.abs(difference))) > 0.0, "turning the body on the axis turns the frame the surrogate takes"
    assert np.allclose(np.asarray(forward), difference, rtol=1e-6, atol=1e-6 * float(np.max(np.abs(difference))))


@pytest.mark.parametrize("at", [(0.01, 3e-4, -2e-4), (-0.02, -7e-4, 6e-4), (0.0, 1e-6, 0.0)])
def test_off_the_axis_the_derivative_in_position_matches_finite_differences(at):
    along_x = node(X)
    tangent = grad_in_position(along_x, at)
    weights = jnp.asarray(np.random.default_rng(4).normal(size=(6, 6)))
    difference = np.array([central(lambda h, k=k: float(jnp.sum(weights * jnp.asarray(resistance(
        along_x, (*(np.asarray(at) + h * np.eye(3)[k]), *IDENTITY))))), 1e-9 if at[1] > 1e-5 else 1e-10)
        for k in range(3)])
    print(f"\n{tangent}\n{difference}")
    assert np.allclose(tangent, difference, rtol=1e-5, atol=1e-6 * float(np.max(np.abs(difference))))


# -- lubrication and the clamp ------------------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [2e-4, 9e-4, 1.09e-3])
def test_the_lubrication_correction_is_the_02_lines_formula(offset):
    along_x = node(X)
    pose = (0.004, 0.0, -offset, *IDENTITY)
    correction = resistance(along_x, pose) - resistance(node(X, epsilon=None), pose)
    frame = np.stack([np.array([0.0, 0.0, -1.0]), np.cross(X, [0.0, 0.0, -1.0]), np.array(X)], axis=1)
    expected = lubrication(MU, RADIUS, TUBE, offset, EPSILON, frame)
    assert np.allclose(correction, expected, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(expected))))


def test_as_the_gap_closes_the_correction_stays_finite_at_its_floor():
    along_x = node(X)
    outer = 1.33 * RADIUS
    at_wall = TUBE - outer
    floored = [resistance(along_x, (0.0, 0.0, -(at_wall + extra), *IDENTITY)) for extra in (0.0, 5e-10, 2e-3)]
    assert all(np.all(np.isfinite(r)) for r in floored)
    # At the wall and past it the gap is floored and the offset clamped: the same resistance to the clamp's rounding.
    assert np.allclose(floored[1], floored[0], rtol=1e-12, atol=0.0)
    assert np.allclose(floored[1], resistance(along_x, (0.0, 0.0, -(at_wall + 1e-3), *IDENTITY)), rtol=1e-12, atol=0.0)
    normal = 6 * math.pi * MU * outer ** 2 / GAP_FLOOR * math.exp(-GAP_FLOOR / EPSILON)
    bare = resistance(node(X, epsilon=None), (0.0, 0.0, -(at_wall + 2e-3), *IDENTITY))
    assert floored[2][2, 2] - bare[2, 2] == pytest.approx(normal, rel=1e-12)
    assert np.all(np.isfinite(grad_in_position(along_x, (0.0, 0.0, -(at_wall + 1e-3)))))


def test_beyond_the_clamp_the_surrogate_holds_its_offset_and_says_so():
    along_x = node(X, epsilon=None)
    largest = (TUBE / RADIUS - 1.33) * 0.95 * RADIUS
    inside, at_clamp = step(along_x, (0.0, 1.0e-3, 0.0, *IDENTITY)), step(along_x, (0.0, largest, 0.0, *IDENTITY))
    beyond = step(along_x, (0.0, 1.3e-3, 0.0, *IDENTITY))
    assert inside["clamp_fired"] == 0.0 and beyond["clamp_fired"] == 1.0
    assert np.allclose(beyond["resistance"], at_clamp["resistance"], rtol=1e-13, atol=0.0)


# -- the Stokes sphere --------------------------------------------------------------------------------------------

def test_the_stokes_sphere_is_its_closed_form():
    from hydro_oracles import StokesSphereNode

    sphere = StokesSphereNode("sphere", DT, viscosity=MU, radius=RADIUS)
    out = sphere.update(sphere.initial_state(), {"pose": jnp.zeros(7)}, DT)
    assert np.allclose(np.asarray(out["resistance"]), sphere_resistance(MU, RADIUS), rtol=1e-15, atol=0.0)
    assert np.array_equal(np.asarray(out["background_load"]), np.zeros(6))


def test_a_body_under_a_constant_force_and_torque_reaches_the_stokes_terminal_twist():
    """``F / (6 pi mu a)`` and ``T / (8 pi mu a^3)``, the approach by the implicit scheme's own closed form (as
    ``test_integrator.py`` checks it): after 40 relaxation times the gap is under 1e-12 of the terminal value."""
    mass, inertia, a, mu = 1e-6, 1e-12, 1e-3, 1e-3
    linear, rotational = 6 * math.pi * mu * a, 8 * math.pi * mu * a ** 3
    tau = max(mass / linear, inertia / rotational)
    dt = tau / 20
    experiment = Experiment("terminal_sphere", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    experiment.add_medium(Medium("bath"))
    experiment.add_body(Body("probe", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=IDENTITY, velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("probe.mass", kg(mass)), ("probe.inertia", kg_m2((inertia,) * 3)), ("probe.radius", m(a)),
                        ("bath.viscosity", mPa_s(1.0))):
        experiment.declare(path, value, reason="the oracle's sphere")
    wrench = np.concatenate([rigid_oracles.FORCE, rigid_oracles.TORQUE])
    experiment.attach(rigid_oracles.ConstantWrench(value=tuple(wrench)), body="probe")
    experiment.attach(StokesSphere(), body="probe", medium="bath")
    history = experiment.build().run(s(800 * dt), history=True).history
    assert history is not None
    twist = np.asarray(history["probe"]["twist"])[-1]
    terminal = np.concatenate([rigid_oracles.FORCE / linear, rigid_oracles.TORQUE / rotational])
    assert np.allclose(twist, terminal, rtol=1e-12, atol=0.0)


# -- the entry ------------------------------------------------------------------------------------------------------

register_namespace("oracle", "0.0.1")


class _Ball:
    """A region with no axis: the inside of a sphere."""

    parameter_names = ("centre", "radius")

    def signed_distance(self, parameters, point):
        return jnp.linalg.norm(point - jnp.asarray(parameters["centre"])) - parameters["radius"]

    def outward_normal(self, parameters, point):
        offset = point - jnp.asarray(parameters["centre"])
        return offset / jnp.linalg.norm(offset)

    def axis(self, parameters):
        return None

    def axis_point(self, parameters):
        return None


register_shape("oracle_ball", _Ball())


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Flask(Geometry):
    name = "oracle.flask"
    version = "0.0.1"
    differentiability = GradientStatus.PIECEWISE
    regions = ("inside",)
    region_shapes = {"inside": "oracle_ball"}
    surfaces = {}
    directions = ()
    radius: Annotated[Quantity, Length] = mm(5.0)

    @override
    def region_parameters(self, region, settings):
        return {"centre": (0.0, 0.0, 0.0), "radius": settings["radius"]}


def experiment_in(medium: Medium, structure: Structure | None = None) -> Experiment:
    experiment = Experiment("drag_medium", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=s(DT))
    if structure is not None:
        experiment.add_structure(structure)
    experiment.add_medium(medium)
    experiment.add_body(Body("probe", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=IDENTITY, velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("probe.mass", kg(1e-4)), ("probe.inertia", kg_m2((1e-10,) * 3)),
                        ("probe.helix_wavenumber", dimensionless(NU)), ("probe.length", m(LENGTH)),
                        ("probe.radius", m(RADIUS)), (f"{medium.name}.viscosity", mPa_s(1.0))):
        experiment.declare(path, value, reason="the test's")
    experiment.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2")), body="probe", medium=medium.name)
    return experiment


def test_a_medium_that_fills_no_region_is_refused_naming_what_to_write():
    text = refusal(experiment_in(Medium("bath")), "drag.no-region")
    assert "medium 'bath' fills no region with a shape" in text and "fills=tube.lumen" in text


def test_a_medium_whose_region_has_no_axis_is_refused_with_that_reason():
    text = refusal(experiment_in(Medium("bath", fills="flask.inside"), Structure("flask", geometry=Flask())),
                   "drag.no-axis")
    assert "fills flask.inside, a oracle_ball region with no axis" in text


def test_weights_whose_hash_is_not_the_packaged_files_are_refused():
    text = refusal(script_a(drag=Drag.Surrogate(weights=Asset("dejongh_mlp_v2", "0" * 64),
                                                lubrication=Lubrication(epsilon=mm(0.15)))), "setting.value")
    assert f"has sha256 {SHA}, and the experiment asks for {'0' * 64}" in text
    assert "the surrogate rung reads its weights" in refusal(script_a(drag=Drag.Surrogate()), "setting.value")
    named_by_path = Drag.Surrogate(weights=scene_asset("dejongh_mlp_v2"))
    assert "is not a packaged asset" in refusal(script_a(drag=named_by_path), "setting.value")
    with pytest.raises(AssetError, match="no packaged asset is named 'dejongh_mlp_v3'"):
        asset("dejongh_mlp_v3")


def test_the_alias_saves_as_the_written_rung():
    written = Drag(rung="surrogate", weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15)))
    assert script_a(drag=written).document() == script_a().document()
    (drag,) = [d for d in script_a().document()["interactions"] if d["name"] == "drag"]
    assert drag["settings"] == {"rung": "surrogate", "weights": {"asset": "dejongh_mlp_v2", "sha256": SHA},
                                "lubrication": {"group": "lubrication",
                                                "settings": {"epsilon": {"value": 0.00015, "unit": "m"}}}}


def test_the_stokes_sphere_in_the_surrogates_place_builds_steps_and_delivers_into_the_same_targets():
    surrogate, sphere = script_a().build(), script_a(drag=StokesSphere()).build()
    into = {name: sorted((e.source[0].split(":")[0], e.source[1], e.target[1]) for e in built.report.lags
                         if e.target[0] == "umr") for name, built in (("surrogate", surrogate), ("sphere", sphere))}
    renamed = [(source.replace("oracle.stokes_sphere", "drag"), out, target) for source, out, target in into["sphere"]]
    assert sorted(renamed) == into["surrogate"]
    for built in (surrogate, sphere):
        history = built.run(s(10 * DT), history=True).history
        assert history is not None and np.all(np.isfinite(np.asarray(history["umr"]["pose"])))


ADDED_DAMPING = define(Interface(
    "oracle.body.added_damping", TargetKind.OPERATOR, "N s/m; N s; N m s", "world", (6, 6), magnitude=1e-6,
    description="a target no body's dynamics accepts"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MisdirectedSphere(StokesSphere):
    """The control: a rung delivering into a target the body does not accept."""

    name = "oracle.misdirected_sphere"
    delivers = {"resistance": Delivery("body", ADDED_DAMPING.name),
                "background_load": StokesSphere.delivers["background_load"]}



def test_a_rung_delivering_into_a_target_the_body_lacks_is_refused():
    text = refusal(script_a(drag=MisdirectedSphere()), "target.not-accepted")
    assert "oracle.body.added_damping" in text
