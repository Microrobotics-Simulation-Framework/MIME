"""The surrogate drag's two tilt flags (7l-c fix round, friction 4c; design #196). The surrogate cannot see a tilt (its
resistance is built in the tube's frame from the offset), so it raises ``tilt_fired`` where the body's axis has left the
tube's by more than the tilt it is declared valid to (``TILT_VALIDITY``, a node constant with its basis on the entry's
validity line), and ``bore_fired`` where the body has left it by so much that it no longer fits the bore (the
geometric limit, beside the first so that a user sees how far the body is from passing through the wall). The second
is the first half of this file; the first is the end of it.

**What these bound.** That the node raises the flag where a body, taken as the cylinder of its length and outer radius,
no longer fits the bore (``L sin(a) + 2 r cos(a) > 2 R_tube``, ``a`` the angle of the body's x axis to the tube's, either
way round), and nowhere else: at planted tilts a factor of 1.05 either side of the bound, which a closed form gives
(``asin(2 R / sqrt(L^2 + 4 r^2)) - atan(2 r / L)``), over bodies longer than the bore and no wider than 0.9 of it
(a body shorter than the bore has no bound on the far side of the peak, and one wider has none). That the resistance
the surrogate delivers does not change with the tilt (so that a flag is all there is to say of it), and that no
derivative passes through the flag. They do not bound the surrogate's accuracy at a tilt, which is not measured, nor
the body's centre being off the axis, which the flag does not read (a body off the axis meets the wall at a smaller
tilt, so the flag rises no earlier than the body does, and later if the centre is off).

**Controls.** Two wrong flags, each a plausible mistake: the bore read as a radius where it is a diameter, and the
body's axis read as a row of the rotation, not a column (the world's x axis in the body's frame). A planted draw is a
pair, one tilt below the bound and one above, and a wrong flag is caught if it answers either wrongly. The first is
caught on every draw (measured at 1.0 over 3000 draws of the strategy, and asserted draw by draw); the second only on
about 94% of them, because a row and a column have the same tilt where the rotation is symmetric, so its control is a
search for a draw it misreads, with two poses fixed beforehand. The CI stress workflow runs this file twenty times with
fresh seeds.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mime.core import quaternion
from mime.data import asset
from mime.nodes.hydro import SurrogateDragNode

DT, MU, NU, LENGTH, RADIUS, TUBE = 5e-4, 1e-3, 2.33, 7.47e-3, 1.56e-3, 3.175e-3
OUTER = 1.33 * RADIUS
SHA = asset("dejongh_mlp_v2").sha256
X, Z = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
LOG = dict(allow_nan=False, allow_infinity=False)


def node(axis=X, *, cls=SurrogateDragNode):
    return cls("drag", DT, weights="dejongh_mlp_v2", weights_sha256=SHA, viscosity=MU, helix_wavenumber=NU,
               length=LENGTH, radius=RADIUS, outer_radius=OUTER, vessel_shape="cylinder",
               vessel_centre=(0.0, 0.0, 0.0), vessel_axis=axis, vessel_radius=TUBE, lubrication_epsilon=None)


class BoreAsRadius(SurrogateDragNode):
    """The plant: the bore read as the tube's radius where the flag compares the body's extent with its diameter."""

    def beyond_the_bore(self, p, orientation, axis):
        along = quaternion.rotation_matrix(orientation)[:, 0]
        cosine = jnp.abs(jnp.dot(along, axis))
        sine = jnp.sqrt(jnp.sum(jnp.cross(along, axis) ** 2))
        return (p["length"] * sine + 2.0 * p["outer_radius"] * cosine > p["vessel_radius"]).astype(orientation.dtype)


class AxisAsRow(SurrogateDragNode):
    """The plant: the body's axis read as the first row of the rotation, the world's x axis in the body's frame."""

    def beyond_the_bore(self, p, orientation, axis):
        along = quaternion.rotation_matrix(orientation)[0, :]
        cosine = jnp.abs(jnp.dot(along, axis))
        sine = jnp.sqrt(jnp.sum(jnp.cross(along, axis) ** 2))
        return (p["length"] * sine + 2.0 * p["outer_radius"] * cosine > 2.0 * p["vessel_radius"]).astype(
            orientation.dtype)


REAL, PLANTS = node(), (node(cls=BoreAsRadius), node(cls=AxisAsRow))


def flag_of(n: SurrogateDragNode, geometry: dict[str, float], axis: np.ndarray, orientation: np.ndarray) -> float:
    p = {**n.constants(), **geometry}
    return float(n.beyond_the_bore(p, jnp.asarray(orientation), jnp.asarray(axis)))


@pytest.fixture(autouse=True, scope="module")
def warmed(x64):
    """Each flag traced once in float64 before the property tests draw, so that an example's time is its own and not
    the first call's compile."""
    for n in (REAL, *PLANTS):
        flag_of(n, {}, np.array(X), np.array((1.0, 0.0, 0.0, 0.0)))


def quaternion_of(matrix: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix, by Shepperd's method (the largest of the four, for stability)."""
    trace = np.trace(matrix)
    diagonal = np.diag(matrix)
    if trace >= diagonal.max():
        s = 2.0 * math.sqrt(1.0 + trace)
        w, x, y, z = (0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s,
                      (matrix[1, 0] - matrix[0, 1]) / s)
    else:
        i = int(np.argmax(diagonal))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k])
        v = np.zeros(3)
        v[i] = 0.25 * s
        v[j] = (matrix[j, i] + matrix[i, j]) / s
        v[k] = (matrix[k, i] + matrix[i, k]) / s
        w = (matrix[k, j] - matrix[j, k]) / s
        x, y, z = v
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def orientation_at(axis: np.ndarray, tilt: float, azimuth: float, roll: float, reverse: bool) -> np.ndarray:
    """A body whose x axis makes ``tilt`` with the tube's axis (pointing the other way where ``reverse``), at
    ``azimuth`` round the tube and ``roll`` about its own axis."""
    first = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    across = math.cos(azimuth) * first + math.sin(azimuth) * second
    direction = math.cos(tilt) * axis + math.sin(tilt) * across
    if reverse:
        direction = -direction
    u = np.cross(direction, axis if abs(math.sin(tilt)) > 1e-3 else first)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    columns = np.stack([direction, math.cos(roll) * u + math.sin(roll) * v, -math.sin(roll) * u + math.cos(roll) * v],
                       axis=1)
    assert np.linalg.det(columns) == pytest.approx(1.0)
    return quaternion_of(columns)


def first_tilt(length: float, outer: float, tube: float) -> float:
    """The tilt at which the body's extent across the tube reaches the bore, in closed form."""
    return math.asin(2.0 * tube / math.hypot(length, 2.0 * outer)) - math.atan2(2.0 * outer, length)


@st.composite
def draws(draw):
    """A bore, a body longer than it and no wider than 0.9 of it, a tube axis, and the angles of a pair of tilts."""
    tube = draw(st.floats(1e-3, 1e-2, **LOG))
    length = 2.0 * tube * draw(st.floats(1.05, 3.0, **LOG))
    outer = tube * draw(st.floats(0.55, 0.9, **LOG))
    axis = np.array(draw(st.lists(st.floats(-1.0, 1.0, **LOG), min_size=3, max_size=3)))
    if np.linalg.norm(axis) < 0.2:
        axis = np.array(X)
    axis = axis / np.linalg.norm(axis)
    geometry = {"length": length, "outer_radius": outer, "vessel_radius": tube}
    angles = (draw(st.floats(0.0, 2 * math.pi, **LOG)), draw(st.floats(0.0, 2 * math.pi, **LOG)), draw(st.booleans()))
    below = draw(st.floats(0.05, 0.5, **LOG))
    above = draw(st.floats(1.05, 3.0, **LOG))
    return geometry, axis, angles, below, above


def planted_tilts(geometry, below, above):
    bound = first_tilt(geometry["length"], geometry["outer_radius"], geometry["vessel_radius"])
    return bound * below, min(bound * above, math.pi / 2)


def disagreements(n, geometry, axis, angles, below, above) -> list[str]:
    low, high = planted_tilts(geometry, below, above)
    found = []
    for name, tilt, expected in (("below", low, 0.0), ("above", high, 1.0)):
        got = flag_of(n, geometry, axis, orientation_at(axis, tilt, *angles))
        if got != expected:
            found.append(f"{name} the bound ({math.degrees(tilt):.2f} degrees): flag {got}, expected {expected}")
    return found


@given(draws())
def test_the_flag_is_raised_above_the_bound_and_never_below_it_for_any_bore_body_axis_and_pointing(held):
    assert disagreements(REAL, *held) == []


@given(draws())
def test_the_bore_read_as_a_radius_is_caught_on_every_draw(held):
    """The control for the bore: a body no narrower than 0.55 of the bore has an extent of at least 1.1 of the radius at
    no tilt at all, so the plant raises the flag at the smallest tilt of every pair. Measured at 1.0 over 3000 draws of
    this strategy before it was fixed."""
    assert disagreements(PLANTS[0], *held) != []


def test_the_axis_read_as_a_row_is_found_by_search_and_caught_at_two_planted_poses():
    """The control for the frame. A row of the rotation is the world's x axis in the body's frame; it has the body
    axis's tilt only where the rotation is symmetric (a tilt about one axis, or a tube along x), so this plant is
    caught on about 94% of the draws (3000 measured), not every one, and the control is a search that finds a draw it
    misreads (about 6% of draws do, so the chance of none in a thousand is of the order of 1e-27), with two poses fixed beforehand: FL-9 in its
    tube along z, tilted 11.5 degrees (below the bound, the plant raises it) and 51.6 degrees with a quarter-turn roll
    (above it, the plant does not)."""
    from hypothesis import find

    find(draws(), lambda held: disagreements(PLANTS[1], *held) != [])
    axis = np.array(Z)
    low = orientation_at(axis, 0.2, 0.0, 0.0, False)
    high = orientation_at(axis, 0.9, 0.0, math.pi / 2, False)
    assert (flag_of(REAL, {}, axis, low), flag_of(PLANTS[1], {}, axis, low)) == (0.0, 1.0)
    assert (flag_of(REAL, {}, axis, high), flag_of(PLANTS[1], {}, axis, high)) == (1.0, 0.0)


def test_the_bound_for_the_fl9_in_the_quarter_inch_tube_is_about_19_degrees():
    bound = math.degrees(first_tilt(LENGTH, OUTER, TUBE))
    assert bound == pytest.approx(18.9, abs=0.1)
    axis = np.array(X)
    for degrees, expected in ((18.0, 0.0), (19.8, 1.0), (45.0, 1.0), (90.0, 1.0)):
        found = flag_of(REAL, {}, axis, orientation_at(axis, math.radians(degrees), 0.7, 0.3, False))
        assert found == expected, degrees


def test_a_body_along_the_tube_either_way_round_is_clear_and_one_across_it_is_raised_for_a_bore_it_is_longer_than():
    for axis in (np.array(X), np.array(Z), np.array([1.0, 2.0, -0.5]) / math.sqrt(5.25)):
        for reverse in (False, True):
            assert flag_of(REAL, {}, axis, orientation_at(axis, 0.0, 0.0, 0.0, reverse)) == 0.0
            assert flag_of(REAL, {}, axis, orientation_at(axis, math.pi / 2, 0.3, 1.1, reverse)) == 1.0


def test_the_flag_is_in_the_nodes_state_and_a_step_reports_what_the_method_does():
    n = node()
    across = jnp.asarray((1e-4, 0.0, 0.0, *orientation_at(np.array(X), math.pi / 2, 0.0, 0.0, False)))
    along = jnp.asarray((1e-4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    for pose, expected in ((across, 1.0), (along, 0.0)):
        out = n.update(n.initial_state(), {"pose": pose}, DT)
        assert float(out["bore_fired"]) == expected and float(out["clamp_fired"]) == 0.0
    assert float(n.initial_state()["bore_fired"]) == 0.0


def test_the_delivered_resistance_does_not_change_with_the_tilt_so_a_flag_is_all_there_is_to_say_of_it():
    """The surrogate is built in the tube's frame from the offset, so the orientation enters only through which way the
    body's axis points along the tube. Aligned, tilted and across, at one offset and the same pointing: the same
    resistance. The control: the body turned end for end, which the surrogate does read, differs."""
    n = node()
    centre = (0.0, 6e-4, 3e-4)

    def resistance(orientation):
        pose = jnp.asarray((*centre, *orientation))
        return np.asarray(n.update(n.initial_state(), {"pose": pose}, DT)["resistance"])

    axis = np.array(X)
    aligned = resistance(orientation_at(axis, 0.0, 0.0, 0.0, False))
    for tilt in (math.radians(25.0), math.radians(60.0)):
        tilted = resistance(orientation_at(axis, tilt, 0.4, 0.9, False))
        assert np.max(np.abs(tilted - aligned)) <= 1e-12 * np.max(np.abs(aligned))
    reversed_ = resistance(orientation_at(axis, 0.0, 0.0, 0.0, True))
    assert np.max(np.abs(reversed_ - aligned)) > 1e-3 * np.max(np.abs(aligned))


@pytest.mark.parametrize("degrees", [0.0, 90.0, 30.0])
def test_no_derivative_passes_through_the_flag_and_none_is_not_a_number_where_the_roots_are_singular(degrees):
    """Parallel and perpendicular axes are where the cosine's and the sine's roots have infinite slope: the flag's
    derivative is zero and finite there, in both modes, and the state's other fields differentiate as they did."""
    n = node()
    q = jnp.asarray(orientation_at(np.array(X), math.radians(degrees), 0.2, 0.5, False))
    base = jnp.asarray((0.0, 4e-4, 2e-4, *np.asarray(q)))

    def flag(pose):
        return n.update(n.initial_state(), {"pose": pose}, DT)["bore_fired"]

    assert np.all(np.asarray(jax.grad(flag)(base)) == 0.0)
    value, tangent = jax.jvp(flag, (base,), (jnp.ones(7),))
    assert float(tangent) == 0.0 and float(value) in (0.0, 1.0)

    def resistance_sum(pose):
        return n.update(n.initial_state(), {"pose": pose}, DT)["resistance"].sum()

    assert np.all(np.isfinite(np.asarray(jax.grad(resistance_sum)(base))))


# -- tilt_fired: the declared validity range -----------------------------------------------------------------------

def range_flag_of(n, limit: float, axis: np.ndarray, orientation: np.ndarray) -> float:
    return float(n.beyond_the_range(jnp.asarray(orientation), jnp.asarray(axis), limit))


class NoAbsolute(SurrogateDragNode):
    """The plant: the cosine compared signed, so that a body pointing against the tube's axis is always beyond."""

    def beyond_the_range(self, orientation, axis, limit):
        cosine = jnp.dot(quaternion.rotation_matrix(orientation)[:, 0], axis)
        return (cosine < jnp.cos(limit)).astype(orientation.dtype)


class LimitAsHalf(SurrogateDragNode):
    """The plant: the declared limit taken as a half angle."""

    def beyond_the_range(self, orientation, axis, limit):
        cosine = jnp.dot(quaternion.rotation_matrix(orientation)[:, 0], axis)
        return (cosine ** 2 < jnp.cos(limit / 2.0) ** 2).astype(orientation.dtype)


RANGE_PLANTS = (node(cls=NoAbsolute), node(cls=LimitAsHalf))


@st.composite
def range_draws(draw):
    limit = math.radians(draw(st.floats(1.0, 40.0, **LOG)))
    axis = np.array(draw(st.lists(st.floats(-1.0, 1.0, **LOG), min_size=3, max_size=3)))
    axis = np.array(X) if np.linalg.norm(axis) < 0.2 else axis / np.linalg.norm(axis)
    angles = (draw(st.floats(0.0, 2 * math.pi, **LOG)), draw(st.floats(0.0, 2 * math.pi, **LOG)), draw(st.booleans()))
    return limit, axis, angles, draw(st.floats(0.05, 0.95, **LOG)), draw(st.floats(1.05, 3.0, **LOG))


def range_disagreements(n, limit, axis, angles, below, above) -> list[str]:
    found = []
    for name, tilt, expected in (("below", limit * below, 0.0), ("above", min(limit * above, math.pi / 2), 1.0)):
        got = range_flag_of(n, limit, axis, orientation_at(axis, tilt, *angles))
        if got != expected:
            found.append(f"{name} the limit ({math.degrees(tilt):.2f} degrees): flag {got}, expected {expected}")
    return found


@given(range_draws())
def test_tilt_fired_is_raised_beyond_the_declared_limit_and_never_within_it_either_way_round(held):
    assert range_disagreements(REAL, *held) == []


def test_the_half_angle_limit_is_found_by_search_and_the_unsigned_cosine_plant_by_a_fixed_pose():
    """The controls. The limit taken as a half angle is caught wherever the pair's lower tilt lies beyond half the limit
    (about half the draws), so it is a search that must find one; the signed cosine is caught wherever the body points
    against the tube's axis (half the draws), with a fixed pose besides."""
    from hypothesis import find

    find(range_draws(), lambda draw: range_disagreements(RANGE_PLANTS[1], *draw) != [])
    axis, limit = np.array(X), math.radians(10.0)
    against = orientation_at(axis, 0.0, 0.0, 0.0, True)
    assert (range_flag_of(REAL, limit, axis, against), range_flag_of(RANGE_PLANTS[0], limit, axis, against)) == (0.0, 1.0)


def test_the_flag_is_raised_by_the_limit_not_by_the_bore_so_both_are_seen_beside_each_other():
    """A tilt between the declared limit and the bore's (about 19 degrees for FL-9 in the quarter-inch tube) raises
    ``tilt_fired`` and not ``bore_fired``; one past the bore's raises both; one within the limit raises neither."""
    from mime.nodes.hydro import TILT_VALIDITY

    limit = math.degrees(TILT_VALIDITY)
    bore = math.degrees(first_tilt(LENGTH, OUTER, TUBE))
    assert limit + 1.0 < bore - 1.0, "the limit lies well inside the bore's"
    n = node(X)
    axis = np.array(X)
    for degrees, tilt_expected, bore_expected in ((limit - 1.0, 0.0, 0.0), ((limit + bore) / 2.0, 1.0, 0.0),
                                                  (bore + 6.0, 1.0, 1.0)):
        pose = jnp.asarray((1e-4, 0.0, 0.0, *orientation_at(axis, math.radians(degrees), 0.3, 0.2, False)))
        out = n.update(n.initial_state(), {"pose": pose}, DT)
        assert (float(out["tilt_fired"]), float(out["bore_fired"])) == (tilt_expected, bore_expected), degrees


# -- the declared limit and the measurement it rests on ----------------------------------------------------------------

#: The error this entry states for the surrogate against the MRS rung (its validity line): up to 16% in swim speed.
STATED_ERROR = 0.16


def largest_tilt_inside(error: float, rows) -> float:
    """The largest measured tilt (degrees) at which the swim speed moves by less than ``error`` in every direction
    measured, with every smaller tilt inside as well."""
    inside = 0.0
    for degrees in sorted({row["tilt_degrees"] for row in rows}):
        if max(abs(row["change_vs_aligned"]) for row in rows if row["tilt_degrees"] == degrees) >= error:
            break
        inside = degrees
    return inside


def test_the_declared_limit_is_the_largest_measured_tilt_inside_the_stated_error_and_no_more():
    """What the limit rests on: ``docs/data/tilt_validity.json``, the MRS rung's swim speed of FL-9 on the
    axis tilted, from ``scripts/tilt_validity.py``. The declaration is not the fit of a curve: it is the largest
    measured tilt at which the speed is inside the error the entry states, one spacing, one offset, one body. The
    controls: a limit one measured step further is outside it in a direction measured, and the same rule over a
    tenth of the error gives a smaller one."""
    import json
    from pathlib import Path

    from mime.nodes.hydro import TILT_VALIDITY

    rows = json.loads((Path(__file__).resolve().parents[2] / "docs" / "data" / "tilt_validity.json").read_text())["rows"]
    tilts = sorted({row["tilt_degrees"] for row in rows})
    declared = math.degrees(TILT_VALIDITY)
    assert declared == pytest.approx(largest_tilt_inside(STATED_ERROR, rows))
    further = tilts[tilts.index(declared) + 1]
    assert max(abs(row["change_vs_aligned"]) for row in rows if row["tilt_degrees"] == further) >= STATED_ERROR
    assert largest_tilt_inside(STATED_ERROR / 10.0, rows) < declared
    assert declared + 1.0 < math.degrees(first_tilt(LENGTH, OUTER, TUBE)), "inside the bore's limit, as the flags order"
