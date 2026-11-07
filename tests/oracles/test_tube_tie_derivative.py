"""The tube wall read's derivative where samples tie (MIME-ANO-143), in the tube demo's tube
(``examples/sphere_settling_tube.py``: 60 um in radius, along x): pairs of samples at exactly equal height or at
exactly equal azimuth, and pairs apart, each sample also read against itself (``tube_reads``, what the correction
reads, where a sample's pair with itself is a tie in both); the derivative of the read along a rotation about each
axis across the tube against the central difference.

An odd component of the read, the sign of the axial offset (or of the angle's sine) times the table at its absolute
value, has the table's slope as its derivative through zero only where its sign test and its absolute value's
derivative see the same number. Jitted they did not always: XLA computes the offset inside each fusion that reads it,
and a difference of products near zero is not the same number in every fusion, so a derivative came out of the
opposite sign: 2.0 of the column's largest entry at a pair tied in height, and in reverse mode at a sample's pair with
itself where no two samples tied (the cotangent of its offset and its sine enters the sample twice, with opposite
signs, and did not cancel), on one of eight pairs apart (with twelve samples, three clouds, on none: the fusion's). ``pair_block`` holds the sine and the offset once (``held_once``) wherever a derivative is taken through
them; a program that takes none is unchanged.

- **The gate**: five pairs tied in height, four in azimuth (one across the axis) and eight apart, rotated about the
  two axes across the tube, jitted, in forward mode (``jvp``) and in reverse mode (``jacrev``, a cotangent for each
  entry), in float32 and float64, within the central difference's floor; eager too, on a pair tied in height.
- **The controls**, each failing on every run, the same harness on the read before the fix (``held_once`` the
  identity, so the program traced is the one before it): a height tie of the wrong sign; in reverse mode a pair apart
  wrong through a sample's pair with itself; and with the offset held but not the sine, the angle's fold still wrong,
  which is why the sine is held too.
- **Under a barrier on the sine**, another fusion of the same read (before the fix, float32 took a tie right only by
  the fusion XLA chose), the gate holds, and the read before the fix fails.
- **The whole body**: the tube demo's 338 force points 0.4 of the way to its floor, 7 pairs of them tied in height,
  the correction's derivative along the two rotations across the tube (``jit(jacfwd)``, float64) within the floor;
  before the fix it missed by 1.0 to 1.7 of the column's largest entry.
- **Unchanged**: the read jitted is the program before the fix to the last bit, at the ties and off them; and off the
  ties (the pairs nudged 1 nm off, and the pairs apart) the forward derivative is the one before the fix to rounding.

The floors are the central difference's at its step, measured on a laptop's CPU with the fix: at most 5.2e-9
(float64) and 2.6e-4 (float32) of the column's largest entry on the pairs, 2.3e-7 on the whole body.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tube_asset

import mime.nodes.hydro.mrs._tube as tube_module
from mime.core import quaternion
from mime.nodes.hydro.mrs import WALL, frame_across, tube_correction, tube_reads, tube_table_from

ROOT = Path(__file__).resolve().parents[2]
DTYPES = {"float32": False, "float64": True}
#: The tube demo's tube: 60 um in radius, its axis along x through the origin.
RADIUS = 60e-6
FRAME = np.asarray(frame_across((1.0, 0.0, 0.0))).reshape(3, 3)
#: Pairs of samples in the tube's frame, in its radius (across the axis, then along it): at exactly equal height (the
#: same last coordinate); at exactly equal azimuth, the parts across the axis a power of two apart (so that their
#: directions are the same numbers), one pair on opposite sides of the axis, their heights distinct; and apart, tied
#: in neither.
HEIGHT_TIES = {
    "height a": ((0.30, 0.10, 0.0), (-0.20, 0.40, 0.0)),
    "height b": ((0.45, -0.25, 0.1), (0.05, 0.35, 0.1)),
    "height c": ((-0.5, 0.2, -0.2), (0.3, -0.3, -0.2)),
    "height d": ((0.12, -0.61, 0.33), (-0.44, 0.27, 0.33)),
    "height e": ((0.6, 0.1, -0.05), (-0.1, -0.55, -0.05)),
}
AZIMUTH_TIES = {
    "azimuth a": ((0.3, 0.1, 0.2), (0.6, 0.2, -0.1)),
    "azimuth b": ((-0.2, 0.35, 0.0), (-0.1, 0.175, 0.3)),
    "azimuth c": ((0.15, -0.3, -0.1), (-0.3, 0.6, 0.25)),
    "azimuth d": ((0.4, 0.2, 0.05), (0.1, 0.05, -0.2)),
}
APART = {
    "apart a": ((0.3, 0.1, 0.2), (-0.25, 0.4, -0.15)),
    "apart b": ((-0.5, -0.2, 0.05), (0.1, 0.6, 0.3)),
    "apart c": ((0.05, 0.55, -0.3), (0.42, -0.31, 0.12)),
    "apart d": ((-0.36, -0.44, 0.27), (0.21, 0.08, -0.09)),
    "apart e": ((0.25, -0.5, 0.18), (-0.33, -0.12, -0.27)),
    "apart f": ((0.6, 0.05, -0.12), (0.0, -0.45, 0.22)),
    "apart g": ((-0.15, 0.62, 0.31), (0.38, 0.29, -0.04)),
    "apart h": ((-0.58, 0.13, -0.21), (0.27, -0.52, 0.16)),
}
TIES = {**HEIGHT_TIES, **AZIMUTH_TIES}
#: The tied pairs with one sample moved 1 nm off the tie (along the axis for a height tie, across it otherwise).
NUDGE = 1e-9 / RADIUS
NUDGED = {f"{name}, nudged": (first, tuple(c + NUDGE * (k == (2 if name.startswith("height") else 0))
                                           for k, c in enumerate(second)))
          for name, (first, second) in TIES.items()}
#: The rotation's step for the central difference and the floor there, relative to the column's largest entry.
STEP = {"float32": 3e-3, "float64": 1e-6}
FLOOR = {"float32": 3e-3, "float64": 1e-6}
#: A miss the controls must reach: an odd component of the opposite sign is 2.0 of its entry.
WRONG = 1.0
#: Off the ties, the forward derivative against the one before the fix, in units in the last place of the column's
#: largest entry: the rule's barrier changes how XLA fuses the derivative, not what it computes.
ROUNDING_ULPS = 64

def dtype_of(name: str):
    return jnp.float64 if DTYPES[name] else jnp.float32


def parameters(name: str) -> dict:
    return {"table": tube_table_from(str(tube_asset.path())).arrays(dtype_of(name)), "frame": tuple(FRAME.ravel()),
            "radius": RADIUS, "centre": (0.0, 0.0, 0.0)}


def placed(local, dtype) -> jax.Array:
    return jnp.asarray(RADIUS * np.asarray(local, dtype=np.float64) @ FRAME, dtype=dtype)


def turn_about(axis: int, angle, dtype) -> jax.Array:
    """The rotation by ``angle`` about the tube frame's ``axis`` (0 or 1, across the tube)."""
    a = jnp.asarray(FRAME[axis], dtype=dtype)
    cross = jnp.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=dtype)
    return jnp.eye(3, dtype=dtype) + jnp.sin(angle) * cross + (1.0 - jnp.cos(angle)) * (cross @ cross)


def turned(points: jax.Array, angles: jax.Array) -> jax.Array:
    """``points`` turned about their mean by ``angles[0]`` about the tube frame's first axis across it, then by
    ``angles[1]`` about its second."""
    middle = jnp.mean(points, axis=0)
    turn = turn_about(1, angles[1], points.dtype) @ turn_about(0, angles[0], points.dtype)
    return middle + (points - middle) @ turn.T


MODES = ("forward", "reverse")
EAGER = ("eager forward", "eager reverse")


def derivatives(name: str, pairs, modes=MODES) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each pair's derivative of the read (every pair of its two samples, each with itself too) along each of the two
    rotations, in each of ``modes``, and its central difference, in float64: forward mode a ``jvp`` along each rotation,
    reverse mode one ``jacrev`` in both. The programs are traced afresh at each call, so a rule patched in before it
    is the one traced."""
    with jax.enable_x64(DTYPES[name]):
        dtype, held = dtype_of(name), parameters(name)
        zero, step = jnp.zeros(2, dtype), STEP[name]
        along = [jnp.zeros(2, dtype).at[axis].set(1.0) for axis in (0, 1)]

        def read(points, angles):
            return tube_reads(turned(points, angles), held).world

        def forward(axis):
            return lambda points, angles: jax.jvp(lambda at: read(points, at), (angles,), (along[axis],))[1]

        def reverse(points, angles):
            return jax.jacrev(lambda at: read(points, at))(angles)

        value = jax.jit(read)
        arms = {"forward": [jax.jit(forward(axis)) for axis in (0, 1)], "reverse": jax.jit(reverse),
                "eager forward": [forward(axis) for axis in (0, 1)], "eager reverse": reverse}
        found: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for pair, local in pairs.items():
            points = placed(local, dtype)
            taken = {mode: (np.stack([np.asarray(arm(points, zero), np.float64) for arm in arms[mode]], axis=-1)
                            if isinstance(arms[mode], list) else np.asarray(arms[mode](points, zero), np.float64))
                     for mode in modes}
            for axis in (0, 1):
                shift = jnp.zeros(2, dtype).at[axis].set(step)
                difference = (np.asarray(value(points, shift), np.float64)
                              - np.asarray(value(points, -shift), np.float64)) / (2.0 * step)
                assert np.max(np.abs(difference)) > 0.0, f"{pair}: the rotation moves nothing, so nothing is compared"
                for mode in modes:
                    found[f"{pair}, about axis {axis}, {mode}"] = (taken[mode][..., axis], difference)
        return found


def misses(name: str, pairs, modes=MODES) -> dict[str, float]:
    """Each derivative's largest miss of its central difference, relative to the difference's largest entry."""
    return {case: float(np.max(np.abs(taken - difference)) / np.max(np.abs(difference)))
            for case, (taken, difference) in derivatives(name, pairs, modes).items()}


def beyond(found: dict[str, float], bound: float) -> dict[str, float]:
    return {case: miss for case, miss in found.items() if not miss <= bound}


def wrong_before(name: str, pairs, modes=MODES) -> list[str]:
    wrong = sorted(beyond(misses(name, pairs, modes), WRONG))
    print(f"\n{name}: {len(wrong)} of {len(pairs) * 2 * len(modes)} wrong: {wrong}")
    return wrong


def without_the_rule(monkeypatch) -> None:
    """The read as it was before the fix: ``held_once`` the identity, so the program traced is the one before it."""
    monkeypatch.setattr(tube_module, "held_once", lambda x: x)


def with_the_offset_held_alone(monkeypatch) -> None:
    """The rule on the axial offset and not on the sine."""
    rule, read = tube_module.held_once, tube_module.pair_block
    without_the_rule(monkeypatch)
    monkeypatch.setattr(tube_module, "pair_block", lambda table, rho_t, rho_s, sine, cosine, offset: read(
        table, rho_t, rho_s, sine, cosine, rule(offset)))


def with_the_sine_held(monkeypatch) -> None:
    """``pair_block`` with an optimization barrier on its sine in every program: another fusion of the same read."""
    read = tube_module.pair_block
    monkeypatch.setattr(tube_module, "pair_block", lambda table, rho_t, rho_s, sine, cosine, offset: read(
        table, rho_t, rho_s, jax.lax.optimization_barrier(sine), cosine, offset))


@pytest.fixture(autouse=True)
def fresh_programs():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_derivative_through_ties_is_the_central_difference_jitted_in_both_modes(dtype):
    found = misses(dtype, {**TIES, **APART})
    print(f"\n{dtype}: largest miss {max(found.values()):.2e} of {len(found)} (floor {FLOOR[dtype]:g})")
    assert beyond(found, FLOOR[dtype]) == {}


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_derivative_through_ties_is_the_central_difference_eager(dtype):
    pairs = {"height c": HEIGHT_TIES["height c"]}
    assert beyond(misses(dtype, pairs, EAGER), FLOOR[dtype]) == {}


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_read_before_the_fix_takes_a_height_tie_of_the_wrong_sign(dtype, monkeypatch):
    without_the_rule(monkeypatch)
    assert wrong_before(dtype, HEIGHT_TIES), "every height tie was right: the gate cannot discriminate here"


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_read_before_the_fix_in_reverse_mode_takes_a_sample_with_itself_of_the_wrong_sign(dtype,
                                                                                                    monkeypatch):
    without_the_rule(monkeypatch)
    assert wrong_before(dtype, APART, ("reverse",)), "every pair apart was right in reverse mode"


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_with_the_offset_held_alone_the_angles_fold_takes_the_wrong_sign(dtype, monkeypatch):
    with_the_offset_held_alone(monkeypatch)
    assert wrong_before(dtype, {**AZIMUTH_TIES, **APART}), "holding the sine has no evidence here"


@pytest.mark.parametrize("dtype", DTYPES)
def test_under_a_barrier_on_the_sine_the_derivative_through_ties_holds(dtype, monkeypatch):
    with_the_sine_held(monkeypatch)
    assert beyond(misses(dtype, {**TIES, **APART}), FLOOR[dtype]) == {}


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_under_a_barrier_on_the_sine_the_read_before_the_fix_takes_a_tie_of_the_wrong_sign(dtype,
                                                                                                    monkeypatch):
    with_the_sine_held(monkeypatch)
    without_the_rule(monkeypatch)
    assert wrong_before(dtype, TIES)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_read_itself_is_the_program_before_the_fix_to_the_last_bit(dtype, monkeypatch):
    """The read jitted, at the ties and off them: in a program that takes no derivative the rule is the identity."""

    def reads() -> list[np.ndarray]:
        with jax.enable_x64(DTYPES[dtype]):
            held = parameters(dtype)
            read = jax.jit(lambda points: tube_reads(points, held).world)
            return [np.asarray(read(placed(local, dtype_of(dtype))))
                    for local in [*TIES.values(), *NUDGED.values(), *APART.values()]]

    found = reads()
    without_the_rule(monkeypatch)
    jax.clear_caches()
    assert all(np.array_equal(a, b) for a, b in zip(found, reads(), strict=True))


@pytest.mark.parametrize("dtype", DTYPES)
def test_off_the_ties_the_forward_derivative_is_the_one_before_the_fix_to_rounding(dtype, monkeypatch):
    """Forward mode only: in reverse mode the read before the fix is wrong off the ties too, through each sample's
    pair with itself (the control above), so there the fix is held to the central difference (the gate)."""
    pairs = {**NUDGED, **APART}
    found = derivatives(dtype, pairs, ("forward",))
    without_the_rule(monkeypatch)
    jax.clear_caches()
    before = derivatives(dtype, pairs, ("forward",))
    epsilon = float(jnp.finfo(dtype_of(dtype)).eps)
    apart = {case: float(np.max(np.abs(taken - before[case][0])) / (epsilon * np.max(np.abs(before[case][0]))))
             for case, (taken, _) in found.items()}
    print(f"\n{dtype}: off the ties, the largest difference {max(apart.values()):.3g} units in the last place")
    assert beyond(apart, ROUNDING_ULPS) == {}


# -- the whole body ------------------------------------------------------------------------------------------------------

#: The tube demo's ball 0.4 of the way from the tube's centre to its floor (it falls 47.5 um to one spacing off it),
#: unturned; the rotation's step for the central difference, its floor there, and the miss the control must reach.
BODY_POSE = (0.0, 0.0, -0.4 * 47.5e-6, 1.0, 0.0, 0.0, 0.0)
BODY_STEP, BODY_FLOOR, BODY_WRONG = 1e-5, 1e-5, 0.5


@pytest.fixture(scope="module")
def demo():
    """The tube demo's drag node's wall, its force points at ``BODY_POSE`` and its viscosity, in float64."""
    spec = importlib.util.spec_from_file_location("example_sphere_settling_tube",
                                                  ROOT / "examples" / "sphere_settling_tube.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with jax.enable_x64(True):
        node = module.sphere_settling_tube().build().graph.get_node("drag:ball")
        held = node._step_constants(node.params_pytree())
        wall = {name: held[f"{WALL}{name}"] for name in tube_module.TUBE_CORRECTION.parameters}
        body = node.body({key: jnp.asarray(node.constants()[key]) for key in node.body_leaf_names()})
        pose = np.asarray(BODY_POSE)
        turn = np.asarray(quaternion.rotation_matrix(jnp.asarray(pose[3:])), dtype=np.float64)
        return wall, pose[:3] + np.asarray(body.force, dtype=np.float64) @ turn.T, float(held["viscosity"])


def height_ties(points: np.ndarray) -> int:
    """The pairs of distinct points at exactly equal height along the tube's axis."""
    height = (points @ FRAME.T)[:, 2]
    return (int(np.sum(height[:, None] == height[None, :])) - len(height)) // 2


def body_misses(wall, points: np.ndarray, viscosity: float) -> dict[str, float]:
    """The correction's derivative along the rotations about the two axes across the tube, about the points' mean,
    jitted (``jacfwd``), against the central difference: each rotation's largest miss relative to the difference's
    largest entry."""
    with jax.enable_x64(True):
        held = jnp.asarray(points)

        def matrix(angles):
            return tube_correction(turned(held, angles), wall, viscosity).matrix

        value = jax.jit(matrix)
        derivative = np.asarray(jax.jit(jax.jacfwd(matrix))(jnp.zeros(2)))
        found = {}
        for axis in (0, 1):
            step = jnp.zeros(2).at[axis].set(BODY_STEP)
            difference = (np.asarray(value(step)) - np.asarray(value(-step))) / (2.0 * BODY_STEP)
            found[f"about axis {axis}"] = float(np.max(np.abs(derivative[..., axis] - difference))
                                                / np.max(np.abs(difference)))
        return found


def test_the_whole_bodys_derivative_through_its_ties_is_the_central_difference(demo):
    wall, points, viscosity = demo
    tied = height_ties(points)
    assert tied > 0, "no two force points tie: the case cannot discriminate"
    found = body_misses(wall, points, viscosity)
    print(f"\n{tied} pairs tied in height; misses {found}")
    assert beyond(found, BODY_FLOOR) == {}


def test_control_the_read_before_the_fix_misses_the_whole_bodys_derivative_at_its_ties(demo, monkeypatch):
    without_the_rule(monkeypatch)
    found = body_misses(*demo)
    print(f"\nthe read before the fix: misses {found}")
    assert max(found.values()) > BODY_WRONG
