"""The contact 0.5.0, whose hold's sweeps and search's Newton steps are XLA loops' iterations, against the contact
0.4.0, kept registered, which unrolls them: the same updates in the same order, a program that rounds otherwise.
0.4.0's program is today's, to the last bit, so this file holds 0.5.0 to it within tolerances stated in units in the
last place of the scale each result came from (CLAUDE.md: no absolute part), committed before the cases first ran:

- **The sweeps** (``_contact.sweeps``), drawn: a millimetre body's rows ``J = [n; r x n]`` (ties among them, as groups
  holding one point give), its system ``A = M/dt + R``, the coupling ``J A^-1 J^T`` and the free speeds at the
  points, 7, 8 and 16 rows, 10 and 20 sweeps, jitted, in float32 and float64: within ``SWEEPS_ULPS`` of the inputs'
  scale, the largest of the forces and of each row's speed over its own coupling.
- **The Newton steps** (``_contact.climbed``), on every kind with a support (the sphere, the capsule, the cylinder,
  the ellipsoid and FL-9's screw; the box has none and is held by its samples, no search), at drawn poses within the
  wall's reach, from each group's kept start, judged against 0.4.0's own resolution there: each group's end (the
  steps' point and the support there, where the search converged, and the converged flag outside a band of
  ``NEWTON_ULPS``) within ``NEWTON_ULPS`` of the hull's size of an end 0.4.0's search reaches from the pose or from a
  pose one ulp away. A group seeded far off the wall may end anywhere on the body (``_contact.shallower``), and there
  0.4.0's own end moves by millimetres under a change of one ulp; where it does not, this is the first head's
  comparison with 0.4.0 at the pose. Two draws on which the versions ended 1.7 mm apart on CI's runners are checked on
  every run, with a control: the looped search from each group's neighbour's start ends where 0.4.0's does not.
- **The whole step**, the contact active: script A's FL-9 held from its first step (``hull_cases.hull_contact``, 20
  steps) and the tube demo (``examples/sphere_settling_tube.py``, 40), each built fresh on each version and run, in
  float32 and float64, by the history scan and by one-step chunks (the tube demo's own run): the pose's position and
  orientation, the twist's two parts and the held points (on the rows that take a force) each on its own scale over
  the run. The forces are judged by their net wrench on the body (each row's force times its row ``[n; r x n]``,
  summed), never row by row: the rows of one body share their forces in no unique way, and rounding redistributes
  them. It is judged on the scale it came from, the momentum rate that stops the body's motion in a step
  (``m |v| / dt``, ``I |omega| / dt``); the per-row forces are recorded beside it, not judged. Each is judged
  against the case's own resolution, measured where the test runs: 0.4.0's divergence from itself when its own
  program is run again from its start one ulp up, and with its mass and moments one ulp up. The bound is the larger
  of the committed tolerance (``STEP_ULPS``, ``WRENCH_ULPS``) and ``RESOLUTIONS`` times that resolution, since FL-9's
  velocity moves by 60 to 100 units of its scale under a change of one ulp, on a laptop's CPU, past the committed 64.
  The control: the looped sweeps' carry lost (no iteration, every force zero) misses FL-9's bound
  ``PLANT_FACTOR`` times over.
- **The derivatives**, float64: forward mode of the couple's net wrench in its torque (the long rod pressed against both
  walls, ``test_hull_contact.py``'s statics) and of the held points in the pose (a rod and FL-9 1 um from the wall),
  within ``SLOPE_ULPS`` of each derivative's largest entry: they pass through the one flowing step at the stopped
  steps' point, which moves by a few units, and through the sweeps' forward derivative, which rounds as they do; and
  reverse mode through the looped sweeps, which a fixed bound lets through, within ``SWEEPS_ULPS``.
- **Controls**, each failing on every run, on fixed hand-built coupled cases: the looped sweeps one short and with the
  rows' order reversed each miss the unrolled ones by ``PLANT_FACTOR`` times ``SWEEPS_ULPS`` or more. A third,
  ``tests/saved/test_switch_harness.py``'s, removes the collector's fallback to the unrolled loops and finds its row
  failing.

What this bounds: 0.5.0 against 0.4.0 at these operating points (a CPU; the tube demo and FL-9 in script A's tube),
not either against the physics, which ``test_hull_contact.py`` and its neighbours hold 0.4.0 to. The whole step on
FL-9 resolves no finer than its own rounding allows: a sweep fewer is below it in float32, so the sweeps' own property
holds their count and order."""

import importlib.util
import json
import math
import platform
from functools import cache, partial
from pathlib import Path
from typing import Any, NamedTuple

import _cache_ceiling
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import event, given, settings, target
from hypothesis import strategies as st

import mime.nodes.body._contact as contact
from hull_cases import FL9_SETTINGS, MARGIN, TUBE_RADIUS, hull_contact, hull_node, kind_points, rotation
from mime.build import Attachment
from mime.interactions import KEPT
from mime.nodes.body._rigid_inertial import RigidInertialNode
from mime.scene import Capsule, Cylinder, Ellipsoid, Screw, Sphere
from mime.testing import first_steps
from mime.units import dimensionless, mm, s

UnrolledContact, _ = KEPT["contact", "0.4.0"]
ROOT = Path(__file__).resolve().parents[2]
#: How far above where a worker stands after giving its compiled programs back this file's tests let its resident memory
#: grow before the next gives them back again (``tests/_cache_ceiling.py``, as ``test_hull_contact.py`` does): each case
#: compiles both versions' programs, and the file run in one process held 3.6 GB at its end.
HEADROOM_MIB = 1024
LIMIT_MIB: list[float] = []  # one worker's limit: where it stood after the last give-back, plus the headroom


@pytest.fixture(autouse=True)
def programs_given_back():
    yield
    now = _cache_ceiling.resident_mib()
    if now is None:
        return
    if not LIMIT_MIB or now > LIMIT_MIB[0]:
        _cache_ceiling.give_back_programs()
        LIMIT_MIB[:] = [(_cache_ceiling.resident_mib() or now) + HEADROOM_MIB]

#: The tolerances, in units in the last place of each result's scale: committed with their predictions before the cases
#: first ran (largest misses expected 16, 8, 12, 12 and 8), from the sizes measured on the pull request's first head:
#: the held points 2.8 units, the state 10, the net wrench 10 of its momentum rate.
SWEEPS_ULPS, NEWTON_ULPS, STEP_ULPS, WRENCH_ULPS, SLOPE_ULPS = 128, 128, 64, 64, 64
#: A whole step's bound in units of its case's own resolution, where that is coarser than the committed tolerance: the
#: resolution is 0.4.0's divergence from itself under a one-ulp change of its inputs, measured where the test runs, so
#: a miss past this many of it is a difference of 0.5.0's, not rounding (the coordinator's decision, logged on the PR).
RESOLUTIONS = 100
#: How many of its bound a planted control misses by, at least, on every run.
PLANT_FACTOR = 16
#: Each dtype, and whether x64 is on for it: a case runs in the dtype the node computes in.
DTYPES = {"float32": False, "float64": True}
#: A millimetre body's mass and principal moments (FL-9's, as ``test_hull_contact.py`` has them).
MASS, INERTIA = 8.5e-5, (1.3e-10, 4.6e-10, 4.6e-10)
#: The hull's spacing in the Newton steps' cases: coarse, so that a node is cheap, at the 0.1 mm margin.
SPACING = 4e-4


def ulps(found: Any, reference: Any, scale: float, dtype: Any) -> float:
    """The largest difference of ``found`` from ``reference`` in units in the last place of ``scale`` in ``dtype``."""
    difference = np.abs(np.asarray(found, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    largest = float(np.max(difference, initial=0.0))
    if largest == 0.0:
        return 0.0
    return largest / (float(np.finfo(dtype).eps) * scale) if scale > 0.0 else math.inf


# -- the sweeps ------------------------------------------------------------------------------------------------------

def direction(draw: Any) -> np.ndarray:
    """A unit vector of three components each 0.1 to 1 in size with its own sign: none underflowing, none on an axis
    plane."""
    held = np.array([draw(st.floats(0.1, 1.0)) * (1.0 if draw(st.booleans()) else -1.0) for _ in range(3)])
    return held / np.linalg.norm(held)


@st.composite
def solves(draw: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """A held body's sweeps: its coupling ``J A^-1 J^T`` and the free speeds at its points (m/s), in float64, and the
    sweeps' count. Each row a point's normal ``n`` and its arm ``r`` (0.3 to 4 mm) from the centre of mass, or a tie
    with an earlier row (a quarter of them: two groups holding one point); ``A`` a millimetre body's ``M / dt``, with a
    drag's resistance or none; each speed 1e-5 to 1e-2 m/s, into the wall or out of it."""
    rows = draw(st.sampled_from((7, 8, 16)))
    count = draw(st.sampled_from((10, 20)))
    dt = draw(st.floats(1e-4, 1e-3))
    mass = draw(st.floats(2e-5, 2e-4))
    moments = [draw(st.floats(3e-11, 1e-9)) for _ in range(3)]
    system = np.diag([mass / dt] * 3 + [m / dt for m in moments])
    if draw(st.booleans()):
        system += np.diag([1e-2] * 3 + [3e-11] * 3)
    held: list[np.ndarray] = []
    for k in range(rows):
        if k and draw(st.integers(0, 3)) == 0:
            held.append(held[draw(st.integers(0, k - 1))])
            continue
        normal = direction(draw)
        arm = direction(draw) * draw(st.floats(3e-4, 4e-3))
        held.append(np.concatenate([normal, np.cross(arm, normal)]))
    jacobian = np.stack(held)
    coupling = jacobian @ np.linalg.solve(system, jacobian.T)
    start = np.array([draw(st.floats(1e-5, 1e-2)) * (-1.0 if draw(st.booleans()) else 1.0) for _ in range(rows)])
    return coupling, start, count


@cache
def swept(count: int, loops: bool):
    return jax.jit(partial(contact.sweeps, count=count, owner="contact.impulse", loops=loops))


def sweeps_scale(coupling: Any, start: Any, forces: Any) -> float:
    """The scale the sweeps' forces came from: the largest of them and of each row's speed over its own coupling."""
    own = np.diagonal(np.asarray(coupling, dtype=np.float64))
    speeds = np.abs(np.asarray(start, dtype=np.float64))[own > 0] / own[own > 0]
    return max(float(np.max(np.abs(np.asarray(forces, dtype=np.float64)), initial=0.0)),
               float(np.max(speeds, initial=0.0)))


def sweeps_miss(coupling: np.ndarray, start: np.ndarray, count: int, dtype: str, looped=None) -> float:
    """The looped sweeps' miss of the unrolled ones in ``dtype``, in units of the inputs' scale; ``looped`` a planted
    looped arm in their place."""
    with jax.enable_x64(DTYPES[dtype]):
        held, speeds = jnp.asarray(coupling, dtype=dtype), jnp.asarray(start, dtype=dtype)
        unrolled = np.asarray(swept(count, False)(held, speeds))
        found = np.asarray((looped or swept(count, True))(held, speeds))
    return ulps(found, unrolled, sweeps_scale(held, speeds, unrolled), np.dtype(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@settings(deadline=None)
@given(solve=solves())
def test_the_looped_sweeps_are_the_unrolled_ones_to_rounding(dtype, solve):
    coupling, start, count = solve
    found = sweeps_miss(coupling, start, count, dtype)
    event("bit for bit" if found == 0.0 else "rounding apart")
    target(found, label="units of the inputs' scale apart")
    assert found <= SWEEPS_ULPS, f"{found:.3g} units of the inputs' scale apart, rows {len(start)}, count {count}"


#: Two rows coupled at 0.9, both pushing: the sweeps converge by 0.81 a sweep, so one fewer leaves about 1e-3 of the
#: scale (1e4 units in float32, 1e12 in float64).
COUPLED = (50.0 * np.array([[1.0, 0.9], [0.9, 1.0]]), np.array([-1e-3, -1e-3]))
#: Two groups holding one point and a third row coupled to both: which of the two ties takes the force is the rows'
#: order, so the order reversed moves the whole force from one to the other.
TIED = (50.0 * (lambda rows: rows @ rows.T)(np.array([[1.0, 0.0], [1.0, 0.0], [0.6, 0.8]])),
        np.array([-1e-3, -1e-3, -5e-4]))


@pytest.mark.parametrize("count", (10, 20))
@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_looped_sweeps_one_short_miss_the_unrolled_ones(dtype, count):
    coupling, start = COUPLED
    short = jax.jit(partial(contact.sweeps, count=count - 1, owner="contact.impulse", loops=True))
    assert sweeps_miss(coupling, start, count, dtype, looped=short) > PLANT_FACTOR * SWEEPS_ULPS


@pytest.mark.parametrize("count", (10, 20))
@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_looped_sweeps_with_their_rows_reversed_miss_the_unrolled_ones(dtype, count):
    coupling, start = TIED

    def reversed_rows(held, speeds):
        return swept(count, True)(held[::-1, ::-1], speeds[::-1])[::-1]

    assert sweeps_miss(coupling, start, count, dtype, looped=reversed_rows) > PLANT_FACTOR * SWEEPS_ULPS


def test_reverse_mode_passes_through_the_looped_sweeps_as_through_the_unrolled_ones():
    coupling, start = COUPLED

    def gradient(loops: bool):
        return np.asarray(jax.grad(lambda speeds: jnp.sum(contact.sweeps(
            jnp.asarray(coupling), speeds, 20, owner="contact.impulse", loops=loops)))(jnp.asarray(start)))

    unrolled, looped = gradient(False), gradient(True)
    assert np.all(np.isfinite(looped)) and np.any(looped != 0.0)
    assert ulps(looped, unrolled, float(np.max(np.abs(unrolled))), np.float64) <= SWEEPS_ULPS


# -- the Newton steps --------------------------------------------------------------------------------------------------

#: Every kind with a support, at a millimetre body's sizes in script A's tube.
SUPPORTED = {
    "sphere": (Sphere(radius=mm(1.0)), {"radius": 1e-3}),
    "capsule": (Capsule(radius=mm(0.8), height=mm(3.0), axis=(1.0, 0.0, 0.0)),
                {"radius": 8e-4, "height": 3e-3, "axis": (1.0, 0.0, 0.0)}),
    "cylinder": (Cylinder(radius=mm(1.0), length=mm(6.0), axis=(1.0, 0.0, 0.0)),
                 {"radius": 1e-3, "length": 6e-3, "axis": (1.0, 0.0, 0.0)}),
    "ellipsoid": (Ellipsoid(semi_axes=mm((2.0, 1.5, 1.2))), {"semi_axes": (2e-3, 1.5e-3, 1.2e-3)}),
    "screw": (Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
                    starts=2), FL9_SETTINGS),
}


@cache
def points_of(name: str) -> np.ndarray:
    kind, held = SUPPORTED[name]
    return kind_points(kind, held, SPACING)


def least_gap(name: str, pose: np.ndarray) -> float:
    placed = pose[:3] + points_of(name) @ rotation(pose[3:]).T
    return float(np.min(TUBE_RADIUS - MARGIN - np.hypot(placed[:, 1], placed[:, 2])))


def bisected(fits, high: float) -> float:
    """The largest value in ``[0, high]`` that ``fits`` (``fits(0)`` holds), by bisection."""
    if fits(high):
        return high
    low = 0.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low


def placed(name: str, spin: float, tilt: float, about: float, towards: float, gap: float) -> np.ndarray:
    """The body spun about its own x and tilted about the direction across the tube at ``about``, the tilt cut to
    nine tenths of what the bore admits with the centre on the axis, its centre moved across the axis along
    ``towards`` until its samples' least gap to the shrunk wall is ``gap``."""
    def turn(angle: float) -> np.ndarray:
        spun = np.array([math.cos(spin / 2), math.sin(spin / 2), 0.0, 0.0])
        axis = np.array([0.0, math.cos(about), math.sin(about)]) * math.sin(angle / 2)
        w1, x1, y1, z1 = math.cos(angle / 2), *axis
        w2, x2, y2, z2 = spun
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    orientation = turn(0.9 * bisected(lambda a: least_gap(name, np.concatenate([np.zeros(3), turn(a)])) > 0, tilt))
    across = np.array([0.0, math.cos(towards), math.sin(towards)])
    low, high = 0.0, TUBE_RADIUS
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if least_gap(name, np.concatenate([middle * across, orientation])) > gap else \
            (low, middle)
    return np.concatenate([low * across, orientation])


def traced(p: dict[str, Any], dtype: Any) -> dict[str, Any]:
    """The node's numeric constants as the run's graph passes them, arrays of its dtype (integers as they are)."""
    return {key: jnp.asarray(value, dtype=dtype) if np.issubdtype(np.asarray(value).dtype, np.floating)
            else jnp.asarray(value) for key, value in p.items()}


@cache
def searches(name: str, dtype: str):
    """For ``name``'s node in ``dtype``: the kept starts at a pose (by the unrolled screening, the same for both arms),
    and each arm's search from them, jitted: the steps' point, the support there and how far within the tolerance
    it stands (``_contact.searched``'s indicator)."""
    kind, held = SUPPORTED[name]
    node = hull_node(kind, held, SPACING, mass=MASS, inertia=INERTIA)
    p = node._step_constants(None)

    def pair(q: dict[str, Any], pose: Any, loops: bool):
        return node.contact_pairs(q, node.hull_points(q, pose.dtype), pose)[0]._replace(loops=loops)

    def seeds(q, pose):
        found = pair(q, pose, False)
        return contact.kept(found, *contact.starts(found), q["margin"])

    def search(q, pose, seeds_, loops):
        x, reached = contact.climbed(pair(q, pose, loops), seeds_, contact.CONTACT_NEWTON)
        within = contact.search_tolerance(x.dtype, node._size) - jnp.sqrt(jnp.sum((x - reached) ** 2, axis=-1))
        return x, reached, within

    return (p, node._size, jax.jit(seeds), jax.jit(partial(search, loops=False)),
            jax.jit(partial(search, loops=True)))


def one_ulp_away(values: np.ndarray) -> list[np.ndarray]:
    """``values`` with one nonzero entry at a time moved one unit in the last place of its dtype, up and then down.
    Zeros are kept: the next number from zero is subnormal, a change of no rounding's size."""
    moved = []
    for index in np.flatnonzero(values):
        for towards in (np.inf, -np.inf):
            held = values.copy()
            held[index] = np.nextafter(held[index], np.asarray(towards, dtype=values.dtype))
            moved.append(held)
    return moved


class Searched(NamedTuple):
    """The looped search at a pose judged against 0.4.0's own resolution there, per group, in units of the hull's
    size: ``missed``, how far its point and its support stand from the nearest of the points 0.4.0's search ends at
    from the pose or from a pose one ulp away (``one_ulp_away``) whose converged flag it shares off the flag's band
    (zero where neither converged; infinite where no flag agrees); ``alone``, how far it stands from 0.4.0's at the pose
    alone (the first head's measure); ``spread``, how far 0.4.0's own ends one ulp away stand from its end at the pose;
    and ``value``, how far the wall's gap at its support stands from the gap at 0.4.0's (the support's value there)."""

    missed: np.ndarray
    alone: np.ndarray
    spread: np.ndarray
    value: np.ndarray


def searched_against_0_4_0(name: str, dtype: str, pose: np.ndarray, *, planted: Any = None) -> Searched:
    """``Searched`` at ``pose``; ``planted``, a looped arm's search in place of 0.5.0's, given the node's constants, the
    pose and the kept starts."""
    with jax.enable_x64(DTYPES[dtype]):
        p, size, seeds, unrolled, looped = searches(name, dtype)
        q, at = traced(p, dtype), np.asarray(jnp.asarray(pose, dtype=dtype))
        kept = seeds(q, jnp.asarray(at))
        ends = [[np.asarray(v, dtype=np.float64) for v in unrolled(q, jnp.asarray(moved), kept)]
                for moved in (at, *one_ulp_away(at))]
        x_l, s_l, w_l = (np.asarray(v, dtype=np.float64) for v in (planted or looped)(q, jnp.asarray(at), kept))
    xs, ss, ws = (np.stack(parts) for parts in zip(*ends, strict=True))
    unit = float(np.finfo(dtype).eps) * size
    band = NEWTON_ULPS * unit

    def apart(x, s, w, x_k, s_k, w_k):
        """Per end ``k`` and group: the points' distance in units where both converged, else zero; infinite where
        the flags differ off the band."""
        both = (w >= 0) & (w_k >= 0)
        distance = np.maximum(np.max(np.abs(x - x_k), axis=-1), np.max(np.abs(s - s_k), axis=-1)) / unit
        agrees = ((w >= 0) == (w_k >= 0)) | (np.abs(w_k) <= band)
        return np.where(agrees, np.where(both, distance, 0.0), np.inf)

    def gaps(local):
        world = at[:3].astype(np.float64) + local @ rotation(at[3:].astype(np.float64)).T
        return TUBE_RADIUS - MARGIN - np.hypot(world[..., 1], world[..., 2])

    around = apart(x_l, s_l, w_l, xs, ss, ws)
    return Searched(np.min(around, axis=0), around[0], np.max(apart(xs[0], ss[0], ws[0], xs, ss, ws), axis=0),
                    np.abs(gaps(s_l) - gaps(ss[0])) / unit)


@pytest.mark.parametrize("dtype", DTYPES)
@settings(deadline=None)
@given(name=st.sampled_from(sorted(SUPPORTED)), spin=st.floats(0.0, 2 * math.pi), tilt=st.floats(0.0, 1.5),
       about=st.floats(0.0, 2 * math.pi), towards=st.floats(0.0, 2 * math.pi), share=st.floats(-0.5, 1.0))
def test_the_looped_newton_steps_end_where_the_unrolled_ones_do_from_a_pose_one_ulp_away(dtype, name, spin, tilt, about,
                                                                                         towards, share):
    found = searched_against_0_4_0(name, dtype, placed(name, spin, tilt, about, towards, share * MARGIN))
    missed = float(np.max(found.missed))
    event("bit for bit" if not np.any(found.alone) else "rounding apart")
    event("0.4.0 ends elsewhere one ulp away" if np.any(found.spread > NEWTON_ULPS) else "0.4.0 settled one ulp away")
    target(min(missed, 1e30), label="units of the hull's size from 0.4.0's nearest end")
    assert missed <= NEWTON_ULPS, (f"{name}: groups {np.flatnonzero(found.missed > NEWTON_ULPS)} end {missed:.3g} units "
                                   f"of the hull's size from every end of 0.4.0's within one ulp of the pose")


#: The two draws on which the looped and unrolled searches ended about 1.7 mm apart on CI's runners in the pull request's
#: first round (stress run 14, and the default 3.14 lane on an AMD EPYC 7763), FL-9 in float32, the spin, tilt, about,
#: towards and share of the property above. At both one group's kept start stands about 0.9 mm (nine margins) off the
#: shrunk wall, where the search may end anywhere on the body (``_contact.shallower``), and 0.4.0's alone ends on two to
#: four points one ulp of the pose away (measured on a laptop's CPU): a property of the search both versions share.
ENDED_ELSEWHERE = {"stress run 14": (1.0, 1.0, 2.0, 1.03, 0.0),
                   "default 3.14": (5.796875, 1.0, 0.3984375, 5.78125, 0.0)}


def ended_elsewhere_pose(draw: str) -> np.ndarray:
    spin, tilt, about, towards, share = ENDED_ELSEWHERE[draw]
    return placed("screw", spin, tilt, about, towards, share * MARGIN)


@pytest.mark.parametrize("draw", ENDED_ELSEWHERE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_looped_newton_steps_at_the_draws_that_ended_elsewhere_end_where_0_4_0_does_one_ulp_away(dtype, draw,
                                                                                                    record_property):
    found = searched_against_0_4_0("screw", dtype, ended_elsewhere_pose(draw))
    measured = {"cpu": cpu_model(), "missed": found.missed.tolist(), "alone": found.alone.tolist(),
                "spread": found.spread.tolist(), "value": found.value.tolist()}
    record_property("measured", json.dumps(measured))
    print(f"\n{draw} {dtype}: {measured}")
    assert float(np.max(found.missed)) <= NEWTON_ULPS


@pytest.mark.parametrize("draw", ENDED_ELSEWHERE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_looped_search_from_the_next_groups_start_ends_where_0_4_0_does_not(dtype, draw):
    """The looped search started from each group's neighbour's kept start, at the draws above: it ends where 0.4.0's
    does not within one ulp of the pose, so the comparison with 0.4.0's ends one ulp away still discriminates where
    0.4.0's own end moves."""
    *_, looped = searches("screw", dtype)

    def from_the_next(q, at, kept):
        return looped(q, at, jnp.roll(kept, 1, axis=0))

    found = searched_against_0_4_0("screw", dtype, ended_elsewhere_pose(draw), planted=from_the_next)
    assert float(np.max(found.missed)) > PLANT_FACTOR * NEWTON_ULPS


# -- the whole step ----------------------------------------------------------------------------------------------------

def on_the_kept_contact(experiment):
    """``experiment`` with its contact attached at the version kept registered, 0.4.0, its settings and bindings as
    written, nothing else changed."""
    declarations = [Attachment(UnrolledContact(**{f: getattr(d.law, f) for f in ("rung", "margin", "rule", "spacing")}),
                               d.bindings) if isinstance(d, Attachment) and d.law.name == "contact" else d
                    for d in experiment._declarations]
    return experiment._declared(experiment._settings, declarations)


def tube_demo():
    import tube_asset

    tube_asset.path()  # stored by CI; a missing table fails here by name, with the command that stores it
    spec = importlib.util.spec_from_file_location("example_sphere_settling_tube",
                                                  ROOT / "examples" / "sphere_settling_tube.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sphere_settling_tube()


CASES = {"hull_contact": (hull_contact, 20), "tube_demo": (tube_demo, 40)}
#: The programs a run compiles: the history scan of a run asked for its steps (``first_steps``, what the goldens and a
#: save's history take), and the one-step program of a run chunked by one step (``run_to``'s ``stage_every``, the tube
#: demo's own run), each compared within itself: bit for bit is per compiled program.
SHAPES = ("history scan", "one-step chunks")


def steps_of(built: Any, steps: int, shape: str) -> dict[str, dict[str, Any]]:
    if shape == "history scan":
        return first_steps(built, steps)
    dt = float(built.report.dt)
    history = built.run(s(steps * dt), history=True, hook=lambda chunk: None, chunk=s(dt)).history
    assert history is not None
    return history


def net_wrench(node: Any, history: dict[str, np.ndarray], initial: np.ndarray) -> np.ndarray:
    """Each step's net wrench of the held points' forces on the body, ``sum_k f_k [n_k; r_k x n_k]``: each point's
    normal into the region, its arm from the centre of mass where the step started."""
    points, forces = history["contact_points"].astype(np.float64), history["contact_forces"].astype(np.float64)
    starts = np.concatenate([initial[None, :3], history["pose"][:-1, :3].astype(np.float64)])
    with jax.enable_x64(True):
        p = traced(node._step_constants(None), np.float64)
        normals = np.asarray(node.normals(p, jnp.asarray(points)))
    arms = points - starts[:, None, :]
    rows = np.concatenate([normals, np.cross(arms, normals)], axis=-1)
    return np.einsum("tk,tki->ti", forces, rows)


#: The parts of the body's state and its held points each judged on its own scale, by field and slice of it.
PARTS = {"position": ("pose", slice(0, 3)), "orientation": ("pose", slice(3, 7)), "velocity": ("twist", slice(0, 3)),
         "angular velocity": ("twist", slice(3, 6)), "held points": ("contact_points", slice(None))}
#: Each judged quantity's committed tolerance: the state's and the held points', and the net wrench's.
COMMITTED = {**{label: STEP_ULPS for label in PARTS}, "net force": WRENCH_ULPS, "net torque": WRENCH_ULPS}


def apart(node: Any, reference: dict[str, Any], found: dict[str, Any], dtype: Any) -> dict[str, float]:
    """``found``'s difference from ``reference`` (two runs' histories of the body), each part in units in the last place
    of the scale it came from: the pose's and the twist's parts and the held points of their own scale over the
    reference run, the held points judged on the rows that take a force in either run (a row far from the wall takes
    none, and its search may end anywhere on the body: ``_contact.shallower``); the net wrench of the momentum rate
    that stops the body's motion in a step (``m |v| / dt``, ``I |omega| / dt``), or of its own size where that is
    larger."""
    reference, found = ({k: np.asarray(v) for k, v in h.items()} for h in (reference, found))
    pushing = (reference["contact_forces"] > 0) | (found["contact_forces"] > 0)
    measured = {}
    for label, (field, part) in PARTS.items():
        want, got = reference[field][..., part], found[field][..., part]
        scale = float(np.max(np.abs(want)))
        if field == "contact_points":
            want, got = want[pushing], got[pushing]
        measured[label] = ulps(got, want, scale, dtype)
    constants = node.constants()
    initial = np.asarray(node.initial_state()["pose"], dtype=np.float64)
    wrenches = [net_wrench(node, history, initial) for history in (reference, found)]
    twist = reference["twist"].astype(np.float64)
    dt = float(node.delta_t)
    rate = {"force": float(constants["mass"]) * float(np.max(np.abs(twist[:, :3]))) / dt,
            "torque": float(np.max(constants["inertia"])) * float(np.max(np.abs(twist[:, 3:]))) / dt}
    for label, part in (("force", slice(0, 3)), ("torque", slice(3, 6))):
        want, got = wrenches[0][:, part], wrenches[1][:, part]
        measured[f"net {label}"] = ulps(got, want, max(float(np.max(np.abs(want))), rate[label]), dtype)
    return measured


def one_ulp_up(values: Any) -> np.ndarray:
    """``values`` with each nonzero entry one unit in the last place of its dtype larger. Zeros are kept: the next
    number from zero is subnormal, a change of no rounding's size."""
    held = np.asarray(values)
    return np.where(held != 0, np.nextafter(held, np.asarray(np.inf, dtype=held.dtype)), held)


def start_moved(graph: Any, body: str) -> None:
    state = graph.get_node_state(body)
    graph.set_node_state(body, {**state, **{field: one_ulp_up(state[field]) for field in ("pose", "twist")}})


def inertia_moved(graph: Any, body: str) -> None:
    leaves = graph.params["nodes"][body]
    leaves.update({name: one_ulp_up(leaves[name]) for name in ("mass", "inertia")})


def rerun(built: Any, steps: int, shape: str, body: str, change=None) -> dict[str, Any]:
    """``built``'s own compiled program run again from its start, with ``change(graph, body)`` made to its start state
    or its parameters first, which the compiled step reads, so nothing is traced again; the graph put back after."""
    graph = built.graph
    graph.reset_state()
    if change is not None:
        change(graph, body)
    try:
        return steps_of(built, steps, shape)[body]
    finally:
        graph.reset_params()
        graph.reset_state()


def sweeps_carry_lost(node: Any, coupling: Any, start: Any, count: int, *, owner: str) -> Any:
    """``RigidInertialNode.sweeps`` whose loop runs no iteration, so its carry, every force, stays at its start, zero:
    a planted defect of the loops. A sweep fewer, or half of them, is no control here: FL-9's sweeps converge to about
    5e-7 of their forces in one, which float32's one-ulp resolution of this case does not tell from rounding (measured
    on a laptop's CPU: one sweep and half of them 0.012 of the bound in float32, one sweep 4.5e5 of it in float64);
    the sweeps' count and order are held by the sweeps' own property and its controls."""
    return contact.sweeps(coupling, start, 0, owner=owner, loops=node.loops)


def cpu_model() -> str:
    """The processor's model, which a runner's rounding can depend on (an AVX-512 machine's program is not a laptop's)."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_the_whole_step_on_the_looped_contact_is_the_unrolled_ones_within_its_own_resolution(case, dtype, shape,
                                                                                            record_property):
    make, steps = CASES[case]
    with jax.enable_x64(DTYPES[dtype]):
        built = {"0.4.0": on_the_kept_contact(make()).build(), "0.5.0": make().build()}
        histories = {version: steps_of(arm, steps, shape) for version, arm in built.items()}
    (body,) = [name for name, fields in histories["0.4.0"].items() if "contact_forces" in fields]
    nodes = {version: arm.graph.get_node(body) for version, arm in built.items()}
    assert (nodes["0.4.0"].loops, nodes["0.5.0"].loops) == (False, True), "the arms are one program"
    reference, found = (histories[version][body] for version in ("0.4.0", "0.5.0"))
    node, eps_dtype = nodes["0.4.0"], np.dtype(dtype)
    pushed = int(np.sum(np.any(np.asarray(reference["contact_forces"]) > 0, axis=-1)))
    assert pushed > 0, "no step took a contact force: the case cannot discriminate"

    with jax.enable_x64(DTYPES[dtype]):
        again = rerun(built["0.4.0"], steps, shape, body)
        moved = {why: rerun(built["0.4.0"], steps, shape, body, change)
                 for why, change in (("start", start_moved), ("mass and moments", inertia_moved))}
    assert all(np.array_equal(np.asarray(again[k]), np.asarray(reference[k])) for k in reference), \
        "0.4.0 run again from its start is not itself to the last bit: the reruns are not its program"
    # A change that moves nothing measures a resolution of zero, and the bound stays the committed tolerance: the
    # resolution only widens a bound where a change moved the run.
    reached = {why: any(not np.array_equal(np.asarray(arm[k]), np.asarray(reference[k])) for k in reference)
               for why, arm in moved.items()}
    resolution = {label: max(apart(node, reference, arm, eps_dtype)[label] for arm in moved.values())
                  for label in COMMITTED}
    bound = {label: max(COMMITTED[label], RESOLUTIONS * resolution[label]) for label in COMMITTED}
    missed = apart(node, reference, found, eps_dtype)
    rows = ulps(found["contact_forces"], reference["contact_forces"],
                float(np.max(np.abs(np.asarray(reference["contact_forces"])))), eps_dtype)
    same = sorted(k for k in reference if np.array_equal(np.asarray(reference[k]), np.asarray(found[k])))
    measured = {"cpu": cpu_model(), "pushed": pushed, "bit for bit": same, "one ulp moved the run": reached,
                "resolution": resolution, "bound": bound,
                "missed": missed, "per-row forces, not judged": rows}

    if case == "hull_contact":  # FL-9's bound is its resolution's; the tube demo's stays the committed tolerance
        with pytest.MonkeyPatch.context() as patch, jax.enable_x64(DTYPES[dtype]):
            patch.setattr(RigidInertialNode, "sweeps", sweeps_carry_lost)
            planted = steps_of(make().build(), steps, shape)[body]
        jax.clear_caches()
        measured["planted, the sweeps' carry lost, of the bound"] = max(
            value / bound[label] for label, value in apart(node, reference, planted, eps_dtype).items())
    record_property("measured", json.dumps(measured))
    print(f"\n{case} {dtype} {shape}: {measured}")
    assert {k: v for k, v in missed.items() if v > bound[k]} == {}
    if case == "hull_contact":
        assert measured["planted, the sweeps' carry lost, of the bound"] > PLANT_FACTOR


# -- the derivatives ---------------------------------------------------------------------------------------------------

#: The statics' long rod and its torque about z (``test_hull_contact.py``'s couple).
LONG_ROD = (Cylinder(radius=mm(0.5), length=mm(8.0), axis=(1.0, 0.0, 0.0)),
            {"radius": 5e-4, "length": 8e-3, "axis": (1.0, 0.0, 0.0)})
TORQUE = 2e-9


@cache
def rod_node(name: str, loops: bool):
    kind, held = {"long_rod": LONG_ROD, "rod": SUPPORTED["cylinder"], "screw": SUPPORTED["screw"]}[name]
    return hull_node(kind, held, SPACING, mass=MASS, inertia=INERTIA, loops=loops)


def couple_pose() -> np.ndarray:
    """The long rod tilted about z until its rims' points ``(L/2, r)`` and ``(-L/2, -r)`` touch the shrunk wall."""
    x, y = LONG_ROD[1]["length"] / 2.0, LONG_ROD[1]["radius"]
    tilt = math.asin((TUBE_RADIUS - MARGIN) / math.hypot(x, y)) - math.atan2(y, x)
    return np.array([0.0, 0.0, 0.0, math.cos(tilt / 2), 0.0, 0.0, math.sin(tilt / 2)])


def couple_slope(loops: bool) -> np.ndarray:
    """The derivative of the couple's net wrench in the torque about z, by forward mode through the step."""
    node = rod_node("long_rod", loops)
    p = traced(node._step_constants(None), np.float64)
    state = {**node.initial_state(), "pose": jnp.asarray(couple_pose()), "twist": jnp.zeros(6)}

    def wrench(torque):
        out = node.update(state, {"applied_wrench": jnp.zeros(6).at[5].set(torque), "resistance": jnp.zeros((6, 6)),
                                  "background_load": jnp.zeros(6)}, 5e-4)
        normals = node.normals(p, out["contact_points"])
        rows = jnp.concatenate([normals, jnp.cross(out["contact_points"], normals)], axis=-1)
        return out["contact_forces"] @ rows

    return np.asarray(jax.jacfwd(wrench)(jnp.asarray(TORQUE)))


def point_slope(name: str, loops: bool) -> np.ndarray:
    """The held points' derivative in the pose's position and a turn about each axis, by forward mode through the
    contact's search, the body 1 um from the shrunk wall."""
    node = rod_node(name, loops)
    p = traced(node._step_constants(None), np.float64)
    body = node.hull_points(p, jnp.float64)
    label = {"rod": "cylinder"}.get(name, name)
    pose = placed(label, 0.7, 0.6, 1.1, 4.4, 1e-6)

    def held(shift):
        from mime.core import quaternion

        moved = jnp.concatenate([jnp.asarray(pose[:3]) + shift[:3],
                                 quaternion.multiply(quaternion.exponential(shift[3:]), jnp.asarray(pose[3:]))])
        return node.contacts(p, body, moved).points

    return np.asarray(jax.jacfwd(held)(jnp.zeros(6)))


@pytest.mark.parametrize("quantity", ["couple's net wrench in the torque", "rod's held points in the pose",
                                      "screw's held points in the pose"])
def test_the_derivatives_through_the_looped_contact_are_the_unrolled_ones_to_rounding(quantity):
    slope = {"couple's net wrench in the torque": couple_slope,
             "rod's held points in the pose": partial(point_slope, "rod"),
             "screw's held points in the pose": partial(point_slope, "screw")}[quantity]
    unrolled, looped = slope(False), slope(True)
    scale = float(np.max(np.abs(unrolled)))
    assert scale > 0.0, "the derivative is zero: the check cannot discriminate"
    found = ulps(looped, unrolled, scale, np.float64)
    print(f"\n{quantity}: {found:.3g} units of its largest entry apart")
    assert found <= SLOPE_ULPS
