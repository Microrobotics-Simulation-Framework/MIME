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
  wall's reach, from each group's kept start: the steps' point and the support there within ``NEWTON_ULPS`` of the
  hull's size on the rows whose search converged, and the converged flag the same outside a band of that width.
- **The whole step**, the contact active: script A's FL-9 held from its first step (``hull_cases.hull_contact``, 20
  steps) and the tube demo (``examples/sphere_settling_tube.py``, 40), each built fresh on each version and run, in
  float32 and float64: the pose's position and orientation, the twist's two parts and the held points each within
  ``STEP_ULPS`` of its own scale over the run. The forces are judged by their net wrench on the body (each row's force
  times its row ``[n; r x n]``, summed), never row by row: the rows of one body share their forces in no unique way,
  and rounding redistributes them. It is judged on the scale it came from, the momentum rate that stops the body's
  motion in a step (``m |v| / dt``, ``I |omega| / dt``), within ``WRENCH_ULPS``; the per-row forces and the net
  wrench in units of its own largest magnitude are printed beside it, not judged.
- **The derivatives**, float64: forward mode of the couple's net wrench in its torque (the long rod pressed against both
  walls, ``test_hull_contact.py``'s statics) and of the held points in the pose (a rod and FL-9 1 um from the wall),
  within ``SLOPE_ULPS`` of each derivative's largest entry: they pass through the one flowing step at the stopped
  steps' point, which moves by a few units, and through the sweeps' forward derivative, which rounds as they do; and
  reverse mode through the looped sweeps, which a fixed bound lets through, within ``SWEEPS_ULPS``.
- **Controls**, each failing on every run, on fixed hand-built coupled cases: the looped sweeps one short and with the
  rows' order reversed each miss the unrolled ones by far more than ``SWEEPS_ULPS``. A third,
  ``tests/saved/test_switch_harness.py``'s, removes the collector's fallback to the unrolled loops and finds its row
  failing.

What this bounds: 0.5.0 against 0.4.0 at these operating points (a CPU; the tube demo and FL-9 in script A's tube),
not either against the physics, which ``test_hull_contact.py`` and its neighbours hold 0.4.0 to."""

import importlib.util
import math
from functools import cache, partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st

import mime.nodes.body._contact as contact
from hull_cases import FL9_SETTINGS, MARGIN, TUBE_RADIUS, hull_contact, hull_node, kind_points, rotation
from mime.build import Attachment
from mime.interactions import KEPT
from mime.scene import Capsule, Cylinder, Ellipsoid, Screw, Sphere
from mime.testing import first_steps
from mime.units import dimensionless, mm

UnrolledContact, _ = KEPT["contact", "0.4.0"]
ROOT = Path(__file__).resolve().parents[2]
#: The tolerances, in units in the last place of each result's scale: committed with their predictions before the cases
#: first ran (largest misses expected 16, 8, 12, 12 and 8), from the sizes measured on the pull request's first head:
#: the held points 2.8 units, the state 10, the net wrench 10 of its momentum rate.
SWEEPS_ULPS, NEWTON_ULPS, STEP_ULPS, WRENCH_ULPS, SLOPE_ULPS = 128, 128, 64, 64, 64
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
    assert sweeps_miss(coupling, start, count, dtype, looped=short) > 16 * SWEEPS_ULPS


@pytest.mark.parametrize("count", (10, 20))
@pytest.mark.parametrize("dtype", DTYPES)
def test_control_the_looped_sweeps_with_their_rows_reversed_miss_the_unrolled_ones(dtype, count):
    coupling, start = TIED

    def reversed_rows(held, speeds):
        return swept(count, True)(held[::-1, ::-1], speeds[::-1])[::-1]

    assert sweeps_miss(coupling, start, count, dtype, looped=reversed_rows) > 16 * SWEEPS_ULPS


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


@pytest.mark.parametrize("dtype", DTYPES)
@settings(deadline=None)
@given(name=st.sampled_from(sorted(SUPPORTED)), spin=st.floats(0.0, 2 * math.pi), tilt=st.floats(0.0, 1.5),
       about=st.floats(0.0, 2 * math.pi), towards=st.floats(0.0, 2 * math.pi), share=st.floats(-0.5, 1.0))
def test_the_looped_newton_steps_are_the_unrolled_ones_to_rounding(dtype, name, spin, tilt, about, towards, share):
    pose = placed(name, spin, tilt, about, towards, share * MARGIN)
    with jax.enable_x64(DTYPES[dtype]):
        p, size, seeds, unrolled, looped = searches(name, dtype)
        q, at = traced(p, dtype), jnp.asarray(pose, dtype=dtype)
        kept = seeds(q, at)
        (x_u, s_u, w_u), (x_l, s_l, w_l) = ([np.asarray(v) for v in arm(q, at, kept)] for arm in (unrolled, looped))
    eps = float(np.finfo(dtype).eps)
    band = NEWTON_ULPS * eps * size
    flipped = ((w_u >= 0) != (w_l >= 0)) & (np.abs(w_u) > band)
    judged = (w_u >= 0) & (w_l >= 0)
    assume(judged.any())
    missed = max(ulps(x_l[judged], x_u[judged], size, np.dtype(dtype)),
                 ulps(s_l[judged], s_u[judged], size, np.dtype(dtype)))
    event("bit for bit" if missed == 0.0 and not flipped.any() else "rounding apart")
    assert not flipped.any(), f"{name}: the converged flag differs off its band at rows {np.flatnonzero(flipped)}"
    assert missed <= NEWTON_ULPS, f"{name}: {missed:.3g} units of the hull's size apart"


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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_the_whole_step_on_the_looped_contact_is_the_unrolled_ones_to_rounding(case, dtype):
    make, steps = CASES[case]
    with jax.enable_x64(DTYPES[dtype]):
        built = {"0.4.0": on_the_kept_contact(make()).build(), "0.5.0": make().build()}
        histories = {version: first_steps(arm, steps) for version, arm in built.items()}
    (body,) = [name for name, fields in histories["0.4.0"].items() if "contact_forces" in fields]
    nodes = {version: arm.graph.get_node(body) for version, arm in built.items()}
    assert (nodes["0.4.0"].loops, nodes["0.5.0"].loops) == (False, True), "the arms are one program"
    reference, found = (histories[version][body] for version in ("0.4.0", "0.5.0"))
    eps_dtype = np.dtype(dtype)
    pushed = int(np.sum(np.any(np.asarray(reference["contact_forces"]) > 0, axis=-1)))
    assert pushed > 0, "no step took a contact force: the case cannot discriminate"

    parts = {"position": ("pose", slice(0, 3)), "orientation": ("pose", slice(3, 7)),
             "velocity": ("twist", slice(0, 3)), "angular velocity": ("twist", slice(3, 6)),
             "held points": ("contact_points", slice(None))}
    missed = {}
    for label, (field, part) in parts.items():
        want, got = np.asarray(reference[field])[..., part], np.asarray(found[field])[..., part]
        missed[label] = ulps(got, want, float(np.max(np.abs(want))), eps_dtype)

    node = nodes["0.4.0"]
    constants = node.constants()
    initial = np.asarray(node.initial_state()["pose"], dtype=np.float64)
    wrenches = [net_wrench(node, {k: np.asarray(v) for k, v in h.items()}, initial) for h in (reference, found)]
    twist = np.asarray(reference["twist"], dtype=np.float64)
    dt = float(node.delta_t)
    rate = {"force": float(constants["mass"]) * float(np.max(np.abs(twist[:, :3]))) / dt,
            "torque": float(np.max(constants["inertia"])) * float(np.max(np.abs(twist[:, 3:]))) / dt}
    wrench_missed, own = {}, {}
    for label, part in (("force", slice(0, 3)), ("torque", slice(3, 6))):
        want, got = wrenches[0][:, part], wrenches[1][:, part]
        largest = float(np.max(np.abs(want)))
        wrench_missed[label] = ulps(got, want, max(largest, rate[label]), eps_dtype)
        own[label] = ulps(got, want, largest, eps_dtype)
    rows = ulps(found["contact_forces"], reference["contact_forces"],
                float(np.max(np.abs(np.asarray(reference["contact_forces"])))), eps_dtype)
    same = sorted(k for k in reference if np.array_equal(np.asarray(reference[k]), np.asarray(found[k])))
    print(f"\n{case} {dtype}: {pushed} of {steps} steps pushed; bit for bit: {same}; state and points, units of each "
          f"part's scale: {', '.join(f'{k} {v:.3g}' for k, v in missed.items())}; net wrench, units of its "
          f"momentum rate: force {wrench_missed['force']:.3g}, torque {wrench_missed['torque']:.3g}; not judged: of "
          f"its own magnitude force {own['force']:.3g}, torque {own['torque']:.3g}, per-row forces {rows:.3g}")
    assert {k: v for k, v in missed.items() if v > STEP_ULPS} == {}
    assert {k: v for k, v in wrench_missed.items() if v > WRENCH_ULPS} == {}


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
