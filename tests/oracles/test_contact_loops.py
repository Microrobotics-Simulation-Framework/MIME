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
  wall's reach, from each group's kept start, judged against 0.4.0's own resolution there: on each group whose end
  0.4.0 resolves (its own ends from the pose's entries one ulp up and down, and from ``RATTLES`` copies of its search
  whose every step's point is moved by up to the rounding that step's solve may leave, its matrix's condition number
  times the dtype's resolution times the point's size, stay within ``NEWTON_ULPS`` of the hull's size of its end at
  the pose, and its converged flag outside a band of that width), the steps' point and the support there within
  ``NEWTON_ULPS`` of 0.4.0's where the search converged, and the converged flag the same off the band. The rattled
  copies are there because a pose's last place does not reach the rounding a differently compiled program makes inside
  the iteration: at ``SHALLOW_BASIN`` the first step overshoots to 50 hull sizes, its rounding is amplified a
  thousandfold by the second, and 0.4.0 ends at the deepest point from every pose one ulp away and at a point 1 mm
  shallower from one in ten of its rattled copies. A group 0.4.0 does not resolve is **not judged**, and is counted:
  nothing of its end is the law's against the rounding of another program. The converged flag admits any residual up
  to the search's tolerance (the square root of the dtype's resolution, of the hull's size: 2,896 units of
  ``eps * size`` in float32), and a search whose rounding takes it into another basin late ends one or two steps short
  of that basin's floor, where none of 0.4.0's 79 arms ends: on CI's runners the clean looped end stood converged one
  step short of a stationary point 4.1 mm from 0.4.0's, at 23 units (an AMD EPYC 7763, run 43862787891; one more step
  takes it to 0.7, its floor ``cond (|x| / size + 1)`` being 3.4), and unconverged two steps short of another, at
  210,100 units, where all 79 arms converged (an Intel Xeon 8573C, run 43871832051). Every judgement of such a group
  measured flagged the clean search on run 43871832051 (1,498,465 draws, uniform, at the strategy's bounds and simple
  values, and from the property's own strategy, on five CPU models): the converged flag against the arms' where they
  all agree (1 of 695,744 float32 draws, and 1 run of this property in 600, run 43872190302), the residual against 8
  times the arms' largest (the same draw), and the residual against the end's own rounding floor, ``c cond (|x| / size
  + 1)`` (c = 8: 1,280 float32 draws and 3,070 float64; c = 64: 638 and 2,665, at groups where 0.4.0's own end stands
  one step short). Judged only where 0.4.0 resolves the group, none of the 1,498,465 was flagged. A group seeded far
  off the wall may end anywhere on the body (``_contact.shallower``), and there 0.4.0's own end moves by millimetres
  under a change of one ulp: neither its point nor its value is resolved, and it is reported, not judged. Two draws on
  which the versions ended 1.7 mm apart on CI's runners are checked on every run, with a control: the looped search
  from each group's neighbour's start. A third (``SHALLOW_BASIN``) is checked with its own control: the poses alone
  read its group as resolved, the rattled searches do not. The property counts the groups it leaves unjudged and all
  it drew, per dtype, reports both (``record_property``, printed) and fails above ``UNJUDGED_CEILING`` of them; its
  control sets the rule's width to zero, so that most groups read unresolved. It does not hold the Newton count to the
  last step: a search one step short of ``CONTACT_NEWTON`` (11 of 12) passes, as searches converge in 9 to 11.

  **The first steps**, in the same draws: the looped search after one to four steps (``EARLY_STEPS``) against the
  unrolled one from the same kept starts, on every group of the kinds whose support is a closed form of its direction
  (``EARLY_KINDS``: the sphere, the capsule, the cylinder and the ellipsoid), the groups 0.4.0 does not resolve at its
  end included: it needs no end state. Its bound is derived from the step's rounding, never fitted (``early_bound``:
  each step's ``eps cond(A) (|x| + |x'| + size)`` in either program, carried to the count by the later steps'
  derivatives along 0.4.0's own steps), and a group past it fails the draw. On CI's runners the two programs' points
  of these kinds were the same to the bit on every draw (run 43927692071: 3,509,271 draws in float32 and 4,123,040 in
  float64, uniform and at the strategy's bounds and simple values, on AMD EPYC 7763, 9V45 and 9V74 and Intel Xeon
  8370C, 8573C and 6973P), so that run measured no margin of the bound, only that the comparison raised no false
  alarm. FL-9's screw is left out: its support is refined on its surface from the search's point, held as a value,
  which the step's derivative does not carry. Its two programs' points differ, by 0.23 to 0.25 of the bound in the
  median there, and stood past it on 5.7 % of its draws in each dtype (49,894 of 877,089 in float32, up to 4.4e7
  times); on a laptop's CPU each of five such draws came after a step that landed 2 to 27 hull sizes off the body. The
  end-state judgement holds the screw as before. The first steps' controls (``EARLY_PINNED``): a loop five steps
  short, a stale carry and a stop at 1e-5 of the size each fail the comparison at two pinned cylinder draws on every
  run, in both dtypes; a loop capped at seven steps is the clean loop at four, which it cannot see. A second target
  (``target``) leads Hypothesis's draws towards the groups whose search needs the most of 0.4.0's steps (each moving
  the point past the end-state rule's width).

  **What it bounds and what it samples.** It bounds the looped search to 0.4.0's on the groups 0.4.0 resolves at the
  poses it draws, and its first steps on every group of ``EARLY_KINDS``; a defect elsewhere is sampled, and how often
  a run sees one is measured, not bounded. The property as committed (100 examples a dtype) failed, of 200 runs a row
  (run 43927793890, seeds paired; the file as merged before, this one after), with a defect of ``planted_search`` on
  the cylinder alone, float32 before / after, float64 before / after: a loop five steps short (seven of twelve) 2 /
  194, 2 / 198; one capped at seven, 2 / 1, 2 / 4 (the same plant at twelve steps: before is the short loop's); a
  carry from two steps back, 1 / 196, 118 / 194; a stop at 1e-5 of the hull's size, 55 / 195, 194 / 195. On every
  kind, after: 200 of 200 in each row but the capped loop's float32 196 (before 195 to 197 in float32, 200 in
  float64). Clean, 0 of 600 runs in each dtype. A defect of one kind is seen only by the draws of that kind, about a
  sixth of a run's, and every cylinder run the first steps' comparison passed with a plant in drew no cylinder (2 to 6
  of 200), so no defect of one kind fails every run. The capped loop, which only the end state sees, changes the end
  only where a cylinder group's search needs more than seven steps: 2 and 5 cylinder draws in 1,000 in float32 and
  float64 (run 43927692071), against 24 and 58 % of the screw's and 14 and 20 % of the ellipsoid's. The steering's
  region is theirs, and it did not raise the capped loop's rate; the cylinder's draws fell to 17 a run from 19 to 20.
  Each defect is caught on every run by its pinned controls (``PLANT_DRAWS`` at the end, ``EARLY_PINNED`` in the first
  steps), in both dtypes.
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
  The resolution samples two directions only (every nonzero start entry up together, then the mass and moments up
  together, the larger taken), where the Newton rule samples each entry up and down; the factor of 100 is the margin
  for what two directions can miss. The controls: the looped sweeps' carry lost (no iteration, every force zero) misses
  FL-9's bound ``PLANT_FACTOR`` times over; a second plant, a mass scaled in the looped arm only, sized to land at
  about twice the bound (``NEAR_BOUND``), must fail it too, and its size over the bound is recorded.
- **The derivatives**, float64: forward mode of the couple's net wrench in its torque (the long rod pressed against both
  walls, ``test_hull_contact.py``'s statics) and of the held points in the pose (a rod and FL-9 1 um from the wall),
  within ``SLOPE_ULPS`` of each derivative's largest entry: they pass through the one flowing step at the stopped
  steps' point, which moves by a few units, and through the sweeps' forward derivative, which rounds as they do; and
  reverse mode through the looped sweeps, which a fixed bound lets through, within ``SWEEPS_ULPS``, and through
  FL-9's whole step at the wall (``jacrev`` against ``jacfwd`` of the pose, twist, held points and forces).
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
from hull_cases import (FL9_SETTINGS, MARGIN, TUBE_RADIUS, hull_contact, hull_node, kind_points, placed_at, rotation,
                        turned)
from mime.build import Attachment
from mime.interactions import KEPT
from mime.nodes.body._rigid_inertial import RigidInertialNode, RigidInertialTensorNode
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
#: The near-bound plant of the whole step: a mass error that lands at ``NEAR_BOUND`` times the bound, sized from a probe
#: at a relative mass error of ``MASS_PROBE`` (large against the bound, small enough that the response is linear).
NEAR_BOUND, MASS_PROBE = 2.0, 1e-3
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

    def rattle(q, pose, seeds_, kicks):
        """The unrolled search with each step's point moved by ``kicks`` (``CONTACT_NEWTON x groups x 3``, each entry
        drawn from -1 to 1) times the rounding the step's solve may leave: the dtype's resolution times the condition
        number of the step's matrix (``I - S R^T N R``) times the point's size. A differently compiled program rounds
        inside the iteration, which a change of the pose's last place does not reach."""
        found = pair(q, pose, False)
        x = seeds_
        for kick in kicks:
            _, slope = contact._supported(found, x, found.body.pose, found.body.parameters, found.other)
            x = contact.climbed(found, x, 1)[0]
            x = x + kick * jnp.finfo(x.dtype).eps * jnp.linalg.cond(slope)[:, None] * jnp.linalg.norm(
                x, axis=-1, keepdims=True)
        reached = contact._supported(found, x, found.body.pose, found.body.parameters, found.other)[0]
        within = contact.search_tolerance(x.dtype, node._size) - jnp.sqrt(jnp.sum((x - reached) ** 2, axis=-1))
        return x, reached, within

    return (p, node._size, jax.jit(seeds), jax.jit(partial(search, loops=False)),
            jax.jit(jax.vmap(rattle, in_axes=(None, None, None, 0))), jax.jit(partial(search, loops=True)))


#: How many rattled copies of 0.4.0's search (``searches``) stand beside the one-ulp poses in judging whether 0.4.0
#: resolves a group. One copy moves a group that sits at a basin's edge about one time in ten (FL-9's group 1 at the draw
#: of ``SHALLOW_BASIN``: 43 of 400 copies), so 64 leave it unmoved about one time in a thousand.
RATTLES = 64


@cache
def kicks_for(groups: int, dtype: str) -> np.ndarray:
    """``RATTLES`` tables of kicks for a search of ``groups`` groups, each entry from -1 to 1: fixed numbers, so that a
    pose's judgement is the same on every run and every replay."""
    return np.random.default_rng(groups).uniform(-1.0, 1.0, (RATTLES, contact.CONTACT_NEWTON, groups, 3)).astype(dtype)


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
    size: ``alone``, how far its point and its support stand from 0.4.0's at the pose (zero where neither converged;
    infinite where their converged flags differ off the flag's band); ``spread``, how far 0.4.0's own ends from the
    poses one ulp away (``one_ulp_away``) and from its rattled searches (``RATTLES``) stand from its end at the pose, on
    the same measure; ``resolved``, the groups whose ``spread`` is within the rule's width (``NEWTON_ULPS``);
    ``missed``, ``alone`` on the groups 0.4.0 resolves and zero on the rest, which are not judged; and ``value``, how
    far the wall's gap at its support stands from the gap at 0.4.0's (the support's value there)."""

    missed: np.ndarray
    alone: np.ndarray
    spread: np.ndarray
    value: np.ndarray
    resolved: np.ndarray


def searched_against_0_4_0(name: str, dtype: str, pose: np.ndarray, *, planted: Any = None,
                           width: float = NEWTON_ULPS, rattles: int = RATTLES) -> Searched:
    """``Searched`` at ``pose``; ``planted``, a looped arm's search in place of 0.5.0's, given the node's constants, the
    pose and the kept starts; ``width``, how far 0.4.0's own ends one ulp away, and from its rattled searches
    (``rattles`` of them, ``RATTLES``; none to judge by the poses alone), may stand from its end at the pose for a
    group to be judged (``NEWTON_ULPS``; the rule's widening is the control of ``UNJUDGED_CEILING``)."""
    with jax.enable_x64(DTYPES[dtype]):
        p, size, seeds, unrolled, rattled, looped = searches(name, dtype)
        q, at = traced(p, dtype), np.asarray(jnp.asarray(pose, dtype=dtype))
        kept = seeds(q, jnp.asarray(at))
        ends = [[np.asarray(v, dtype=np.float64) for v in unrolled(q, jnp.asarray(moved), kept)]
                for moved in (at, *one_ulp_away(at))]
        if rattles:
            kicks = jnp.asarray(kicks_for(kept.shape[0], dtype)[:rattles])
            rattled_ends = [np.asarray(v, dtype=np.float64) for v in rattled(q, jnp.asarray(at), kept, kicks)]
            ends += [[v[k] for v in rattled_ends] for k in range(rattles)]
        found = (planted or looped)(q, jnp.asarray(at), kept)
        x_l, s_l, w_l = (np.asarray(v, dtype=np.float64) for v in found)
    (x_u, *x_moved), (s_u, *s_moved), (w_u, *w_moved) = zip(*ends, strict=True)
    unit = float(np.finfo(dtype).eps) * size
    band = NEWTON_ULPS * unit

    def apart(x, s, w, x_k, s_k, w_k):
        both = (w >= 0) & (w_k >= 0)
        distance = np.maximum(np.max(np.abs(x - x_k), axis=-1), np.max(np.abs(s - s_k), axis=-1)) / unit
        agrees = ((w >= 0) == (w_k >= 0)) | (np.abs(w_k) <= band)
        return np.where(agrees, np.where(both, distance, 0.0), np.inf)

    def gaps(local):
        world = at[:3].astype(np.float64) + local @ rotation(at[3:].astype(np.float64)).T
        return TUBE_RADIUS - MARGIN - np.hypot(world[..., 1], world[..., 2])

    alone = apart(x_l, s_l, w_l, x_u, s_u, w_u)
    spread = np.max([apart(x_k, s_k, w_k, x_u, s_u, w_u) for x_k, s_k, w_k in zip(x_moved, s_moved, w_moved)], axis=0)
    resolved = spread <= width
    return Searched(np.where(resolved, alone, 0.0), alone, spread, np.abs(gaps(s_l) - gaps(s_u)) / unit, resolved)


#: The counts of the search's Newton steps after which the early comparison holds the looped search to the unrolled one,
#: both from the same kept starts.
EARLY_STEPS = (1, 2, 3, 4)
#: The kinds the early comparison judges: those whose support is a closed form of its direction (``near`` choosing an
#: end at most). FL-9's screw refines its support on its own surface from the search's point, held as a value
#: (``_screw.Screw.support``), so a step's point depends on the last through that start, which the step's derivative
#: does not carry: where a step lands hull sizes off the body, the refinement's start moves the next point by what the
#: bound (``early_bound``) leaves out. Measured: 5.7 % of screw draws past the bound in each dtype on CI's runners (run
#: 43927692071), and on a laptop's CPU each of five such draws after a step that landed 2 to 27 hull sizes off.
EARLY_KINDS = ("capsule", "cylinder", "ellipsoid", "sphere")


@cache
def early_searches(name: str, dtype: str, plant: str | None = None):
    """For ``name``'s node in ``dtype``, jitted, from the node's constants, the pose and the kept starts: the unrolled
    and the looped search's points after each of ``EARLY_STEPS`` steps (``plant``, a defect of ``planted_search``, in
    the looped one's place), and 0.4.0's steps one at a time to ``CONTACT_NEWTON``: each step's point, the residual
    ``|x - s(x)|`` at it, and at the first ``EARLY_STEPS`` the condition number of the step's matrix and the step's
    derivative in its point."""
    kind, held = SUPPORTED[name]
    node = hull_node(kind, held, SPACING, mass=MASS, inertia=INERTIA)
    planted = {count: planted_search(name, dtype, plant, count) for count in EARLY_STEPS} if plant else {}

    def run(q, pose, seeds_):
        found = node.contact_pairs(q, node.hull_points(q, pose.dtype), pose)[0]
        unrolled = jnp.stack([contact.climbed(found._replace(loops=False), seeds_, k)[0] for k in EARLY_STEPS])
        looped = jnp.stack([planted[k](q, pose, seeds_)[0] if plant else
                            contact.climbed(found._replace(loops=True), seeds_, k)[0] for k in EARLY_STEPS])
        hull, wall = found.body, found.other
        values = jax.lax.stop_gradient((hull.pose, contact._numbers(hull.parameters),
                                        contact._numbers(wall.parameters), wall.margin))
        parameters = {**hull.parameters, **values[1]}
        surface = contact.Wall(wall.shape, {**wall.parameters, **values[2]}, values[3])

        def step(x):
            return contact._step(found, x, values[0], parameters, surface)

        x, points, residuals, conditions, slopes = seeds_, [seeds_], [], [], []
        for k in range(contact.CONTACT_NEWTON + 1):
            point, matrix = contact._supported(found, x, values[0], parameters, surface)
            residuals.append(jnp.linalg.norm(x - point, axis=-1))
            if k == contact.CONTACT_NEWTON:
                break
            if k < max(EARLY_STEPS):
                conditions.append(jnp.linalg.cond(matrix))
                slopes.append(jnp.stack([jax.jvp(step, (x,), (jnp.broadcast_to(e, x.shape),))[1]
                                         for e in jnp.eye(3, dtype=x.dtype)], axis=-1))
            x = step(x)
            points.append(x)
        return (unrolled, looped, jnp.stack(points), jnp.stack(residuals), jnp.stack(conditions),
                jnp.stack(slopes))

    return jax.jit(run)


def early_bound(points: np.ndarray, conditions: np.ndarray, slopes: np.ndarray, eps: float, size: float) -> np.ndarray:
    """The bound of the looped and the unrolled points' distance after each of ``EARLY_STEPS`` steps (``steps x
    groups``, m), derived from the step's rounding, never fitted. The step ``x' = x - A^-1 (x - s(x))`` rounds its data,
    the point it starts from, the point it lands on (the solve's ``|A^-1 (x - s)|`` is at most their sum) and the
    body's point ``s`` (within the hull's size), each to ``eps`` of its size, and carries them through ``A^-1``, whose
    norm is at most ``A``'s condition number (``A = I - S B`` keeps a direction fixed, the support's derivative ``S``
    having rank two at most, so its norm is at least one): ``eps cond(A) (|x| + |x'| + size)`` in either program, twice
    between two. A difference made at step ``j`` reaches step ``k`` through the later steps' derivatives along 0.4.0's
    own steps (``G'(x_{k-1}) ... G'(x_j)``, first order), by at most their product's largest singular value."""
    groups = points.shape[1]
    lengths = np.linalg.norm(points, axis=-1)
    kicks = [eps * conditions[j] * (lengths[j] + lengths[j + 1] + size) for j in range(max(EARLY_STEPS))]
    carried: list[np.ndarray] = []
    bounds = []
    for k in range(1, max(EARLY_STEPS) + 1):
        carried = [slopes[k - 1] @ p for p in carried] + [np.broadcast_to(np.eye(3), (groups, 3, 3))]
        bounds.append(2.0 * sum(np.linalg.norm(p, ord=2, axis=(-2, -1)) * kick
                                for p, kick in zip(carried, kicks, strict=False)))
    return np.stack(bounds)[[k - 1 for k in EARLY_STEPS]]


def steps_needed(early: "Early") -> np.ndarray:
    """Per group, how many of 0.4.0's ``CONTACT_NEWTON`` steps move its point past the end-state rule's width
    (``NEWTON_ULPS`` of the hull's size's resolution): the steps its search needs."""
    return np.sum(np.linalg.norm(np.diff(early.points, axis=0), axis=-1) > NEWTON_ULPS * early.unit, axis=0)


class Early(NamedTuple):
    """The looped search's first steps against the unrolled ones', per count of ``EARLY_STEPS`` and group: ``missed``,
    their points' distance over its bound (``early_bound``); ``apart`` and ``bound``, the distance and the bound in
    units of the hull's size's resolution (``unit``, m); and 0.4.0's steps to ``CONTACT_NEWTON``, their ``points`` (m)
    and the ``residuals`` ``|x - s(x)|`` at them (in units of ``unit``)."""

    missed: np.ndarray
    apart: np.ndarray
    bound: np.ndarray
    points: np.ndarray
    residuals: np.ndarray
    unit: float


def early_against_0_4_0(name: str, dtype: str, pose: np.ndarray, *, plant: str | None = None) -> Early:
    """``Early`` at ``pose``, from the kept starts the end-state judgement takes; ``plant``, a defect of
    ``planted_search`` in the looped search's place."""
    with jax.enable_x64(DTYPES[dtype]):
        p, size, seeds = searches(name, dtype)[:3]
        q, at = traced(p, dtype), jnp.asarray(pose, dtype=dtype)
        found = early_searches(name, dtype, plant)(q, at, seeds(q, at))
        unrolled, looped, points, residuals, conditions, slopes = (np.asarray(v, dtype=np.float64) for v in found)
    eps = float(np.finfo(dtype).eps)
    unit = eps * size
    apart = np.linalg.norm(looped - unrolled, axis=-1)
    bound = early_bound(points, conditions, slopes, eps, size)
    with np.errstate(divide="ignore", invalid="ignore"):
        missed = np.where(apart == 0.0, 0.0, apart / bound)
    return Early(missed, apart / unit, bound / unit, points, residuals / unit, unit)


#: The most groups of a run, in a dtype, that the Newton steps' property may leave unjudged against 0.4.0's end (its groups
#: where 0.4.0's own ends one ulp away, or from its rattled searches, stand past ``NEWTON_ULPS``): a share of all the groups
#: its draws held. These groups are not judged (the module's docstring says why). The ceiling is a monitor of the rule's
#: own premise, how often 0.4.0 does not resolve a group: it is computed from 0.4.0's arms alone, so that a defect of the
#: looped search cannot raise it. Measured: the committed property under twelve Hypothesis seeds of 100 draws (800
#: groups a dtype), the first the seed of a stress run that read 22 and 19 past a first ceiling of 2 %: float32 0.5 to
#: 3.0 %, float64 0.25 to 2.4 %; over 600 runs a dtype on CI's runners (run 43877350461), at most 4.6 % in float32 and
#: 3.9 % in float64. The rule judging by the poses alone (the pull request's base) left, under four of those seeds,
#: float32 0.1 to 0.5 % and float64 none: the rattled searches move 0.4.0's end where a pose's last place does not,
#: which Hypothesis's targeting finds. The ceiling is about twice the worst of those, so that the set can grow a little
#: before anything shows and cannot grow unseen; each run reports its own count (``record_property``, and printed).
UNJUDGED_CEILING = 0.10


def over_the_ceiling(unjudged: int, groups: int) -> bool:
    return unjudged > UNJUDGED_CEILING * groups


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_looped_newton_steps_are_the_unrolled_ones_to_rounding_where_0_4_0_resolves_them(dtype, record_property):
    unjudged: dict[str, list[int]] = {}  # by kind: the groups left unjudged, the groups drawn
    worst = [0.0]  # the early comparison's largest distance over its bound

    @settings(deadline=None)
    @given(name=st.sampled_from(sorted(SUPPORTED)), spin=st.floats(0.0, 2 * math.pi), tilt=st.floats(0.0, 1.5),
           about=st.floats(0.0, 2 * math.pi), towards=st.floats(0.0, 2 * math.pi), share=st.floats(-0.5, 1.0))
    def draws(name, spin, tilt, about, towards, share):
        pose = placed(name, spin, tilt, about, towards, share * MARGIN)
        found = searched_against_0_4_0(name, dtype, pose)
        early = early_against_0_4_0(name, dtype, pose)
        missed = float(np.max(found.missed))
        counts = unjudged.setdefault(name, [0, 0])
        counts[0] += int(np.sum(~found.resolved))
        counts[1] += len(found.resolved)
        event("bit for bit" if not np.any(found.alone) else "rounding apart")
        event("0.4.0 ends elsewhere one ulp away" if np.any(~found.resolved) else "0.4.0 settled one ulp away")
        target(min(missed, 1e30), label="units of the hull's size apart where 0.4.0 resolves them")
        target(float(np.max(steps_needed(early))), label="the most of 0.4.0's steps a group's search needs")
        first = float(np.max(early.missed)) if name in EARLY_KINDS else 0.0
        worst[0] = max(worst[0], first)
        assert missed <= NEWTON_ULPS, (f"{name}: groups {np.flatnonzero(found.missed > NEWTON_ULPS)}, where 0.4.0's "
                                       f"end stays within one ulp of the pose or its rounding, end {missed:.3g} units "
                                       f"of the hull's size apart")
        assert first <= 1.0, (f"{name}: groups {np.flatnonzero(np.max(early.missed, axis=0) > 1.0)} apart within its "
                              f"first steps, {first:.3g} times the bound of the two programs' rounding")

    draws()
    left, groups = (sum(counts[i] for counts in unjudged.values()) for i in (0, 1))
    measured = {"cpu": cpu_model(), "unjudged": left, "groups": groups, "share": left / groups if groups else 0.0,
                "by kind": unjudged, "ceiling": UNJUDGED_CEILING,
                "early comparison's largest share of its bound": worst[0]}
    record_property("unjudged", json.dumps(measured))
    print(f"\nthe groups the Newton steps' property leaves unjudged, {dtype}: {measured}")
    assert groups > 0 and not over_the_ceiling(left, groups), (
        f"{left} of {groups} groups left unjudged, past the ceiling of {UNJUDGED_CEILING:.0%}: 0.4.0's own ends move "
        f"under a change of one ulp on more groups than the rule was measured to leave out")


#: Draws for the ceiling's control: each kind at one pose (the spin, tilt, about, towards and share of the property
#: above), and the two draws that ended elsewhere.
WIDENED_POSES = [("sphere", (1.0, 1.0, 2.0, 1.03, 0.5)), ("capsule", (4.0, 1.2, 0.7, 3.5, 0.2)),
                 ("cylinder", (2.5, 0.8, 5.0, 2.2, 0.8)), ("ellipsoid", (0.3, 1.4, 3.3, 5.1, 0.4)),
                 ("screw", (1.0, 1.0, 2.0, 1.03, 0.0)), ("screw", (5.796875, 1.0, 0.3984375, 5.78125, 0.0))]


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_rule_widened_so_that_most_groups_read_unresolved_is_over_the_ceiling(dtype):
    """0.4.0's width of 128 units set to zero: a group is unresolved wherever 0.4.0's own end moves at all under a
    change of one ulp, which is most of them. The ceiling must report it, on every run."""
    left = groups = 0
    for name, draw in WIDENED_POSES:
        found = searched_against_0_4_0(name, dtype, placed(name, *draw[:4], draw[4] * MARGIN), width=0.0)
        left, groups = left + int(np.sum(~found.resolved)), groups + len(found.resolved)
    print(f"\nthe rule widened to zero, {dtype}: {left} of {groups} groups unresolved")
    assert over_the_ceiling(left, groups)


#: The two draws on which the looped and unrolled searches ended about 1.7 mm apart on CI's runners in the pull request's
#: first round (stress run 14, and the default 3.14 lane on an AMD EPYC 7763), FL-9 in float32, the spin, tilt, about,
#: towards and share of the property above. At both one group's kept start stands about 0.9 mm (nine margins) off the
#: shrunk wall, where the search may end anywhere on the body (``_contact.shallower``), and 0.4.0's alone ends on two to
#: four points one ulp of the pose away, or does not converge (measured on a laptop's CPU and on an AMD EPYC 9V74): a
#: property of the search both versions share, and no point 0.4.0 resolves.
ENDED_ELSEWHERE = {"stress run 14": (1.0, 1.0, 2.0, 1.03, 0.0),
                   "default 3.14": (5.796875, 1.0, 0.3984375, 5.78125, 0.0)}


def ended_elsewhere_pose(draw: str) -> np.ndarray:
    spin, tilt, about, towards, share = ENDED_ELSEWHERE[draw]
    return placed("screw", spin, tilt, about, towards, share * MARGIN)


@pytest.mark.parametrize("draw", ENDED_ELSEWHERE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_looped_newton_steps_at_the_draws_that_ended_elsewhere_are_the_unrolled_ones_where_0_4_0_resolves_them(
        dtype, draw, record_property):
    found = searched_against_0_4_0("screw", dtype, ended_elsewhere_pose(draw))
    measured = {"cpu": cpu_model(), "missed": found.missed.tolist(), "alone": found.alone.tolist(),
                "spread": found.spread.tolist(), "value": found.value.tolist()}
    record_property("measured", json.dumps(measured))
    print(f"\n{draw} {dtype}: {measured}")
    assert float(np.max(found.missed)) <= NEWTON_ULPS


@pytest.mark.parametrize("draw", ENDED_ELSEWHERE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_looped_search_from_the_next_groups_start_ends_where_0_4_0_does_not(dtype, draw):
    """The looped search started from each group's neighbour's kept start, at the draws above: on the groups 0.4.0
    resolves it ends far from 0.4.0's, so leaving out the groups 0.4.0 does not resolve leaves the comparison able to
    fail at a draw that holds one."""
    looped = searches("screw", dtype)[-1]

    def from_the_next(q, at, kept):
        return looped(q, at, jnp.roll(kept, 1, axis=0))

    found = searched_against_0_4_0("screw", dtype, ended_elsewhere_pose(draw), planted=from_the_next)
    assert float(np.max(found.missed)) > PLANT_FACTOR * NEWTON_ULPS


#: The draw on which the looped search's group 1 ended in a stationary point of the support 1 mm shallower than 0.4.0's,
#: FL-9 in float32 (the release push run's ``openmaterica-main`` lane, shard 12, and a laptop's CPU on the default
#: lanes' OpenMaterica alike): the spin, tilt, about, towards and share of the property above. The group's start stands
#: 0.74 mm off the wall, its first step overshoots to 50 hull sizes and the second amplifies that step's rounding about
#: a thousandfold; the loop's body rounds the first step 17 units in its last place away from the unrolled program's
#: (one looped step and one unrolled step from the same point are bit for bit the same), and the end moves. Six other groups
#: hold the deepest point in both versions, so the net wrench does not change. In float64 both end at the deepest.
SHALLOW_BASIN = (0.0, 1.0, 0.08979514983112065, 5.320289449654194, 0.0)
SHALLOW_GROUP = 1


def shallow_basin_pose() -> np.ndarray:
    spin, tilt, about, towards, share = SHALLOW_BASIN
    return placed("screw", spin, tilt, about, towards, share * MARGIN)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_looped_newton_steps_at_the_shallow_basin_draw_are_the_unrolled_ones_where_0_4_0_resolves_them(
        dtype, record_property):
    found = searched_against_0_4_0("screw", dtype, shallow_basin_pose())
    measured = {"cpu": cpu_model(), "missed": found.missed.tolist(), "alone": found.alone.tolist(),
                "spread": found.spread.tolist(), "value": found.value.tolist()}
    record_property("measured", json.dumps(measured))
    print(f"\nthe shallow basin draw {dtype}: {measured}")
    assert float(np.max(found.missed)) <= NEWTON_ULPS


def test_control_the_poses_alone_read_the_shallow_basin_group_as_resolved_and_the_rattled_searches_do_not():
    """0.4.0's own end of group 1 at ``SHALLOW_BASIN`` stays within one unit of the pose's under a change of one ulp
    of any entry of the pose, and moves by millions under its rounding rattled: the rule that judged by the poses alone
    held the group to 0.4.0's end, and the looped search's rounding put it elsewhere."""
    alone = searched_against_0_4_0("screw", "float32", shallow_basin_pose(), rattles=0)
    rattled = searched_against_0_4_0("screw", "float32", shallow_basin_pose())
    assert alone.resolved[SHALLOW_GROUP] and alone.spread[SHALLOW_GROUP] < 0.1 * NEWTON_ULPS
    assert not rattled.resolved[SHALLOW_GROUP] and rattled.spread[SHALLOW_GROUP] > PLANT_FACTOR * NEWTON_ULPS


#: A stop's residual, the ``early`` plant of ``planted_search``: a share of the hull's size.
EARLY_STOP = 1e-5


def planted_search(name: str, dtype: str, plant: str, count: int = contact.CONTACT_NEWTON):
    """The looped search of ``count`` steps with a planted defect, jitted, from the node's constants, the pose and the
    kept starts: ``short``, five steps short of its count (seven of the twelve); ``capped``, no more than seven steps
    (seven of the twelve, and the clean count of seven or fewer); ``stale``, each step taken from the iterate two steps
    back; ``early``, a group stopped once its residual is within ``EARLY_STOP`` of the hull's size."""
    kind, held = SUPPORTED[name]
    node = hull_node(kind, held, SPACING, mass=MASS, inertia=INERTIA)

    def run(q, pose, seeds_):
        found = node.contact_pairs(q, node.hull_points(q, pose.dtype), pose)[0]._replace(loops=True)
        hull, wall = found.body, found.other
        values = jax.lax.stop_gradient((hull.pose, contact._numbers(hull.parameters),
                                        contact._numbers(wall.parameters), wall.margin))
        parameters = {**hull.parameters, **values[1]}
        surface = contact.Wall(wall.shape, {**wall.parameters, **values[2]}, values[3])

        def step(x):
            return contact._step(found, x, values[0], parameters, surface)

        if plant in ("short", "capped"):
            x = jax.lax.fori_loop(0, max(count - 5, 0) if plant == "short" else min(count, 7), lambda _, x: step(x),
                                  seeds_)
        elif plant == "stale":
            x = jax.lax.fori_loop(0, count, lambda _, held_: (held_[1], step(held_[0])), (seeds_, seeds_))[1]
        else:
            def stopped(_, x):
                near = jnp.linalg.norm(x - contact._supported(found, x, values[0], parameters, surface)[0],
                                       axis=-1) < EARLY_STOP * node._size
                return jnp.where(near[:, None], x, step(x))

            x = jax.lax.fori_loop(0, count, stopped, seeds_)
        reached = contact._supported(found, x, values[0], parameters, surface)[0]
        return x, reached, contact.search_tolerance(x.dtype, node._size) - jnp.linalg.norm(x - reached, axis=-1)

    return jax.jit(run)


#: The pinned controls: for each plant of ``planted_search`` and each dtype, draws (the spin, tilt, about, towards and
#: share of the property above) at which the plant is caught, on every run. Chosen from run 43871832051's draws (a group
#: 0.4.0 resolves to within 16 units, the clean looped end within 16, the plant's past 64 times the bound) and judged on
#: nine runner arms of four CPU models a dtype (run 43877350461): on the two cylinder draws every plant is infinitely
#: apart (its end unconverged where 0.4.0's resolved end converged, or the reverse) in float64, and the short loop and
#: the stale carry in float32, every group resolved, the clean search at 0 units. A stop at 1e-5 of the size is under
#: the bound at both in float32 (7 and 24 units) and at all but 3 of the 133,600 cylinder draws of run 43871832051 past
#: four times it: its float32 controls are FL-9's, where it stood at least 1,430 and 1,470 units apart on every arm (the
#: clean search within 5 and 7).
PINNED_CYLINDER = (("cylinder", (3.85659060848782, 0.4300469233088282, 5.195302023152226, 0.4300469233088282,
                                 -0.2660959494337816)),
                   ("cylinder", (5.14871469598194, 0.44253834367545297, 1.1295627824972891, 5.81562520775212,
                                 -0.2248737720168823)))
PINNED_SCREW = (("screw", (0.0, 0.25, 1.5, 4.0, 0.7285262466692002)),
                ("screw", (3.5967220181521684, 0.9208618346550588, 0.9208618346550588, 3.5967220181521684,
                           0.9208618346550588)))
PLANT_DRAWS = {("short", "float32"): PINNED_CYLINDER, ("stale", "float32"): PINNED_CYLINDER,
               ("early", "float32"): PINNED_SCREW, ("short", "float64"): PINNED_CYLINDER,
               ("stale", "float64"): PINNED_CYLINDER, ("early", "float64"): PINNED_CYLINDER}


@pytest.mark.parametrize("plant, dtype", PLANT_DRAWS)
def test_control_each_plant_is_caught_at_its_pinned_draws(plant, dtype):
    for kind, (spin, tilt, about, towards, share) in PLANT_DRAWS[plant, dtype]:
        pose = placed(kind, spin, tilt, about, towards, share * MARGIN)
        found = searched_against_0_4_0(kind, dtype, pose, planted=planted_search(kind, dtype, plant))
        assert float(np.max(found.missed)) > NEWTON_ULPS, f"{plant} {dtype} at {kind} {spin, tilt, about, towards, share}"


#: The plants the early comparison sees within ``EARLY_STEPS`` (``capped`` is the clean loop at seven steps or fewer).
EARLY_PLANTS = ("short", "stale", "early")
#: The early comparison's pinned draws, two cylinders (the spin, tilt, about, towards and share of the property
#: above): on eleven runner arms a dtype of six CPU models (run 43927793890) each plant failed it at both, the stop at
#: 1e-5 of the size by 10.1 to 13.3 times the bound in float32 and 5e9 times in float64, the short loop and the stale
#: carry by 1e5 times and more, and the clean search stood at 0 (bit for bit). ``PINNED_CYLINDER``'s first draw is not
#: one: the float32 stop stood at 0.79 of the bound there on every arm.
EARLY_PINNED = (("cylinder", (2.052718941131401, 0.9110605435756676, 2.6036556336742747, 1.955715350275813,
                              -0.3524855429988902)),
                ("cylinder", (1.804626857088745, 0.7071424954843506, 0.47111529515606554, 2.8203637689262715,
                              0.3753564597465622)))


@pytest.mark.parametrize("plant", EARLY_PLANTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_control_each_plant_fails_the_early_comparison_at_its_pinned_draws_where_the_clean_search_passes(dtype, plant):
    for kind, (spin, tilt, about, towards, share) in EARLY_PINNED:
        pose = placed(kind, spin, tilt, about, towards, share * MARGIN)
        clean = float(np.max(early_against_0_4_0(kind, dtype, pose).missed))
        caught = float(np.max(early_against_0_4_0(kind, dtype, pose, plant=plant).missed))
        assert clean <= 1.0 < caught, (f"{plant} {dtype} at {kind} {spin, tilt, about, towards, share}: clean {clean}, "
                                       f"planted {caught} of the bound")


#: Two draws of FL-9 on which stress runs of the pull request read a looped end not stationary against a floor of the
#: step's rounding: a group converging linearly (its residual 1,267 units of the dtype's resolution after twelve steps,
#: converged by the flag's tolerance of 2,896) in both versions alike.
SLOW_CONVERGING_DRAWS = ((0.0, 1.0, 4.882796315068688, 0.0, 0.0), (0.0, 1.0, 2.3107361820010146, 3.014619487560795, 0.0))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind, draw", (*PINNED_CYLINDER, *PINNED_SCREW, *(("screw", d) for d in SLOW_CONVERGING_DRAWS)))
def test_the_clean_looped_search_passes_at_the_draws_the_controls_plant_into(kind, draw, dtype):
    spin, tilt, about, towards, share = draw
    found = searched_against_0_4_0(kind, dtype, placed(kind, spin, tilt, about, towards, share * MARGIN))
    assert float(np.max(found.missed)) <= NEWTON_ULPS


#: FL-9's draws on which a judgement of a group 0.4.0 does not resolve flagged the clean looped search on CI's runners,
#: float32, and the group: the looped end converged one step short of another stationary point (23 units of the size's
#: resolution, an AMD EPYC 7763, run 43862787891), and unconverged two steps short of another where all of 0.4.0's arms
#: converged (210,100 units, an Intel Xeon 8573C, run 43871832051). On nine runner arms of four CPU models (run
#: 43877350461) 0.4.0 resolved neither group in float32, and both in float64.
FLAGGED_CLEAN = {"one step short": ((4.87705747991847, 0.35266508542049807, 1.6715090783522921, 2.7785117170337896,
                                     0.971075033193801), 2),
                 "unconverged": ((3.1642171208849503, 0.4550519353221549, 1.5530031407758926, 0.5,
                                  0.0854266099205776), 3)}


@pytest.mark.parametrize("case", FLAGGED_CLEAN)
def test_the_groups_a_judgement_of_unresolved_groups_flagged_clean_are_unresolved_and_left_unjudged(case):
    (spin, tilt, about, towards, share), group = FLAGGED_CLEAN[case]
    found = searched_against_0_4_0("screw", "float32", placed("screw", spin, tilt, about, towards, share * MARGIN))
    assert not found.resolved[group] and float(np.max(found.missed)) <= NEWTON_ULPS


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
    # The inertia is the tensor's nine numbers: its largest entry is a diagonal one, the scale of a moment.
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
    """The body's mass and moments one ulp up: the ``mass`` leaf, and each nonzero entry of the ``inertia`` leaf, the
    tensor's nine numbers row-major (zeros kept). A symmetric pair holds equal numbers, which move to equal numbers, so
    the tensor stays symmetric (checked)."""
    leaves = graph.params["nodes"][body]
    leaves.update({name: one_ulp_up(leaves[name]) for name in ("mass", "inertia")})
    tensor = np.asarray(leaves["inertia"]).reshape(3, 3)
    assert tensor.shape == (3, 3) and np.array_equal(tensor, tensor.T), "the nudge left the inertia tensor unsymmetric"


def mass_scaled(delta: float):
    """A change for ``rerun``: the body's mass leaf times ``1 + delta``, in its dtype."""
    def change(graph: Any, body: str) -> None:
        leaves = graph.params["nodes"][body]
        leaves["mass"] = np.asarray(leaves["mass"]) * np.asarray(1.0 + delta, dtype=np.asarray(leaves["mass"]).dtype)
    return change


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
    assert all(isinstance(n, RigidInertialTensorNode) for n in nodes.values()), "the body is not the tensor node"
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
        # A plant near the bound: the looped arm's mass scaled, sized from a probe's response (the misses are linear in
        # a small mass error) to land at about twice the bound. It tests the power near the bound that the carry lost,
        # at 500 bounds, cannot. It runs first: the plant below clears jax's caches, and these reruns would compile.
        def over_the_bound(delta: float) -> float:
            with jax.enable_x64(DTYPES[dtype]):
                arm = rerun(built["0.5.0"], steps, shape, body, mass_scaled(delta))
            return max(value / bound[label] for label, value in apart(node, reference, arm, eps_dtype).items())

        probe = over_the_bound(MASS_PROBE)
        assert probe > 0.0, "the mass probe moved nothing: the near-bound plant cannot be sized"
        measured["near-bound plant, mass scaled by"] = NEAR_BOUND * MASS_PROBE / probe
        measured["near-bound plant, of the bound"] = over_the_bound(NEAR_BOUND * MASS_PROBE / probe)

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
        assert measured["near-bound plant, of the bound"] > 1.0, "a mass error of about twice the bound passed"


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


def whole_step(loops: bool):
    """FL-9 1e-7 m from the shrunk wall, turned 12 degrees across the tube, moving: its node's whole step as a function
    of the applied wrench, returning the pose, the twist, the held points and their forces as one vector."""
    kind, settings = SUPPORTED["screw"]
    quaternion = turned(math.radians(12.0))
    pose = (*placed_at(1e-7, quaternion), *quaternion)
    node = hull_node(kind, settings, SPACING, mass=MASS, inertia=INERTIA, pose=pose, loops=loops)
    state = {**node.initial_state(), "pose": jnp.asarray(pose), "twist": jnp.asarray([1e-3, 0.0, -1e-3, 0.0, 0.0, 0.0])}

    def step(wrench):
        out = node.update(state, {"applied_wrench": wrench, "resistance": jnp.zeros((6, 6)),
                                  "background_load": jnp.zeros(6)}, 5e-4)
        return {"state": jnp.concatenate([out["pose"], out["twist"], out["contact_points"].ravel()]),
                "forces": out["contact_forces"]}

    return step, jnp.asarray([0.0, 0.0, -1e-6, 0.0, 0.0, 2e-9])


def test_reverse_mode_through_the_whole_looped_step_is_forward_mode_to_rounding():
    """One reverse-mode derivative through the whole 0.5.0 step, not only its sweeps: ``jacrev`` against ``jacfwd`` of
    the pose, the twist, the held points and the forces in the applied wrench, FL-9 at the wall, float64. Both finite,
    some forces active, each part within its tolerance of its largest entry: the state's and points' within
    ``SLOPE_ULPS``, the forces', which the sweeps give, within ``SWEEPS_ULPS``. (Measured at the review, on a laptop's
    CPU: 1 unit and 34 units.)"""
    step, wrench = whole_step(loops=True)
    forward, reverse = jax.jacfwd(step)(wrench), jax.jacrev(step)(wrench)
    for part, bound in (("state", SLOPE_ULPS), ("forces", SWEEPS_ULPS)):
        found, reference = np.asarray(reverse[part]), np.asarray(forward[part])
        assert np.all(np.isfinite(found)) and np.all(np.isfinite(reference)), f"the {part}' derivative is not finite"
        scale = float(np.max(np.abs(reference)))
        assert scale > 0.0, f"the {part}' derivative is zero: the check cannot discriminate"
        miss = ulps(found, reference, scale, np.float64)
        print(f"\nreverse against forward mode through the whole step, {part}: {miss:.3g} units of its largest entry")
        assert miss <= bound
