"""The near-contact law's sphere form (``mime.nodes.hydro.near_contact``, design rows #324, #341, #354), against exact
answers and independent references, with its controls: W2a PR 1b-i's gates, predictions committed first
(``tests/oracles/predictions_w2a_1b.json``).

- **The table** (``sphere_plane_resistance``, MIME's, from the bispherical solutions): the pinned file; its build's
  checks (the closed-form series, reciprocity, the no-slip condition off the nodes, convergence); its spline against the
  midpoints the build computed (control: linear interpolation); against the graded method of fundamental solutions at
  every gap it resolves (an independent method); its continuation below its range C1 at the edge (control: a kink).
- **The seam:** on a plane the hybrid is the table to rounding; the ambient's zero on the opened wall; the plan's pick of
  the form by the body's kind, named in the build report.
- **Gate 1, a sphere above a plane at the family's spacing, drawn orientations:** every diagonal entry within 7 % of the
  exact answer at every gap 0.3 to 0.001 radii, each coupling within 7 % of its own size; the normal against Brenner's
  series, the others against the graded MFS. With the increment on its own scale equal to the plane's exact increment
  through the node (control: the increment x1.1).
- **Gate 2:** a force-free sphere spun on the plane translates at the exact rate per rotation within 10 % at every gap
  (control: the increment's coupling counted twice); in a tube (sphere-to-tube ratios 0.22, 0.28, 0.34) the rolling
  rates along the axis and round it, judged on the plane rate's scale, and the normal, against the tube MFS where it
  resolves (0.1 to 0.0075 radii) (controls: the plane's form kept in the tube; the coefficients from the gap's
  curvature alone).
- **Gate 8:** forward and reverse mode equal Richardson's differences across the hand-over gap, the table's lower edge
  and the roughness cutoff (controls: the opening's derivative stopped; a hard max; the continuation's kink).
- **Dtypes and two builds:** float32 within 1e-3 of float64; the two-builds check quiet in the sphere's radius.

Float64 (``tests/oracles/conftest.py``) but the float32 gate. The controls honour ``W2A_PLANT``
(``lubrication_cases.planted``) and gate 1 draws by ``DETECTION_SEED``, for ``.github/probe/detection_rate.py``."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import _cache_ceiling
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lubrication_cases import (
    FINE,
    RADIUS,
    above_plane,
    compared,
    in_tube,
    planted,
    plane_pose,
    scaled,
    stepper,
    translation_per_rotation,
    tube_pose,
)
from plane_oracles import brenner

from mime.data import asset, asset_bytes
from mime.nodes.hydro._film import smooth_floor
from mime.nodes.hydro.mrs._handover import HANDOVER_SPACINGS
from mime.nodes.hydro.near_contact import AMBIENT_ZERO, LIFT_WIDTH, NearContact, ambient_point, increment, opening
from mime.nodes.hydro.near_contact import _sphere
from mime.scene import Sphere
from mime.units import m

HEADROOM_MIB = 1024
LIMIT_MIB: list[float] = []


@pytest.fixture(autouse=True)
def programs_given_back():
    """As ``test_lubrication.py``'s: give JAX's compiled programs back once a worker grows past its headroom."""
    yield
    now = _cache_ceiling.resident_mib()
    if now is None:
        return
    if not LIMIT_MIB or now > LIMIT_MIB[0]:
        _cache_ceiling.give_back_programs()
        LIMIT_MIB[:] = [(_cache_ceiling.resident_mib() or now) + HEADROOM_MIB]


REFERENCES = Path(__file__).parents[1] / "differential" / "references"
#: The files' pins, written out here rather than read from MIME.
TABLE_SHA256 = "346bc025d0b56460a20ba98bdfc9023457cba32ca975d44e363585f80a904f5f"
PLANE_MFS_SHA256 = "36487b7add26702abe832ccc882e568a70c07d9299af989efb622bbe950ae426"
TUBE_MFS_SHA256 = "aa1ead278b453d082d144dc8658461d9fb503b3b6ee14b1347a4034872920bc5"
TUBE_MFS_ABOVE_SHA256 = "c0bea240dd5b3f2b47171c15cc460c8e4f44a1bd1edde2669455067360ffa7e9"
#: The gaps (radii) the gates judge above a plane: every gap of the graded MFS.
GAPS = (0.3, 0.2, 0.15, 0.1, 0.075, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.002, 0.001)
#: Gate 1's and gate 2's committed lines (predictions_w2a_1b.json).
ENTRY_LINE, RATE_LINE = 0.07, 0.10


def pinned(path: Path, sha256: str) -> dict:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha256, f"{path.name} is not the file pinned"
    return json.loads(data)


def table_file() -> dict:
    return json.loads(asset_bytes(asset(_sphere.TABLE)))


def plane_mfs() -> dict:
    return pinned(REFERENCES / "sphere_plane_graded_mfs.json", PLANE_MFS_SHA256)["rows"]


def tube_mfs() -> dict:
    return pinned(REFERENCES / "sphere_tube_mfs.json", TUBE_MFS_SHA256)["rows"]


def rate(resistance: np.ndarray) -> float:
    """A unit sphere spun about y at a unit rate, force-free: its translation along x (``U / (a Omega)``)."""
    return float(-np.linalg.solve(resistance[:3, :3], resistance[:3, 4])[0])


# -- the table --------------------------------------------------------------------------------------------------------

def test_the_table_is_the_packaged_file_pinned():
    assert asset(_sphere.TABLE).sha256 == TABLE_SHA256
    assert hashlib.sha256(asset_bytes(asset(_sphere.TABLE))).hexdigest() == TABLE_SHA256


def test_the_tables_build_checks_hold():
    """The solver against Brenner's and Jeffery's series, the coupling's reciprocity, the no-slip condition at nodes not
    collocated, and the change with the series' length and the quadrature, as the build measured them: its derivation
    record's checks, each within its tolerance (the file holds the numbers the law reads, the checks the record and the
    build's log)."""
    from mime.assets import shipped_derivations

    record = shipped_derivations()[_sphere.TABLE]
    for check in record.checks:
        print(f"\n{check.name}: {check.measured:.2g} ({check.tolerance})")
    assert record.sha256 == TABLE_SHA256 and len(record.checks) == 4 and record.failed() == ()


def midpoint_errors(read) -> np.ndarray:
    held = table_file()["midpoints"]
    d = np.asarray(held["d"])
    want = np.stack([np.asarray(held[name]) for name in _sphere.ENTRIES])
    return d, want, np.stack([read(x) for x in d], 1)


def test_the_spline_reads_the_midpoints_the_build_computed():
    """Within a radius of the wall each entry within 1e-6 of itself at every midpoint (measured: 2.7e-7, the coupling);
    everywhere within 1e-7 of the free sphere's translational drag (6 pi), the coupling, which vanishes as the gap
    grows, read on that scale."""
    d, want, got = midpoint_errors(_sphere.entries_at)
    near = d <= 1.0
    relative = np.abs(got[:, near] / want[:, near] - 1.0).max(axis=1)
    on_scale = (np.abs(got - want) / (6.0 * math.pi)).max(axis=1)
    print(f"\nwithin a radius, relative {relative}; everywhere, over 6 pi {on_scale}")
    assert relative.max() <= 1e-6 and on_scale.max() <= 1e-7


def test_control_the_knots_linear_interpolation_misses_the_midpoints():
    knots = table_file()
    k = np.asarray(knots["d"])
    held = np.stack([np.asarray(knots[name]) for name in _sphere.ENTRIES]) - _sphere.interpolated_singular(k)

    def linear(x: float) -> np.ndarray:
        rest = np.asarray([np.interp(math.log(x), np.log(k), row) for row in held])
        return rest + _sphere.interpolated_singular(np.asarray(x))

    d, want, got = midpoint_errors(linear)
    near = d <= 1.0
    relative = np.abs(got[:, near] / want[:, near] - 1.0).max(axis=1)
    print(f"\nlinear interpolation, within a radius: {relative}")
    assert relative.max() > 1e-6


def test_the_table_agrees_with_the_graded_mfs_where_it_resolves():
    """Each of xx, ryry and the coupling (on its own size), and the rate, within three times the MFS's own resolution
    (its level B against C), floored at 2e-4, from 0.3 to 0.005 radii; at 0.002 and 0.001, where the MFS is not
    resolved (its normal 1.9 % and 6.5 % off Brenner's series), within 1 % (xx, ryry) and 1.5 % (the coupling, the
    rate). The normal is Brenner's series in both."""
    rows = plane_mfs()
    for gap in GAPS:
        fine, coarse = (np.asarray(rows[f"{level} {gap!r}"]["resistance"]) for level in "CB")
        exact = _sphere.plane_matrix(jnp.asarray(_sphere.entries_at(gap)))
        exact = np.asarray(exact)
        found, resolution = {}, {}
        for name, (i, j) in {"xx": (0, 0), "ryry": (4, 4), "x_ry": (0, 4)}.items():
            found[name] = abs(exact[i, j] / fine[i, j] - 1.0)
            resolution[name] = abs(coarse[i, j] / fine[i, j] - 1.0)
        found["rate"] = abs(rate(exact) / rate(fine) - 1.0)
        resolution["rate"] = abs(rate(coarse) / rate(fine) - 1.0)
        print(f"\nh/a {gap}: " + " ".join(f"{k} {v:.2e} (MFS {resolution[k]:.1e})" for k, v in found.items()))
        if gap >= 0.005:
            assert all(found[k] <= max(3.0 * resolution[k], 2e-4) for k in found), (gap, found, resolution)
        else:
            assert found["xx"] <= 0.01 and found["ryry"] <= 0.01 and found["x_ry"] <= 0.015 and found["rate"] <= 0.015


def test_the_continuation_below_the_tables_range_is_c1_at_its_edge():
    """The remainder's value and slope just below and just above the edge agree to the step's own size; with the
    continuation matched in value only (the plant), the slope jumps."""
    def slopes(read) -> tuple[np.ndarray, np.ndarray]:
        edge, step = _sphere.table().edge, 1e-4
        x = np.asarray([edge * (1 - 2 * step), edge * (1 - step), edge, edge * (1 + step), edge * (1 + 2 * step)])
        held = np.stack([read(v) for v in x], 1) - _sphere.interpolated_singular(x)
        below = (held[:, 1] - held[:, 0]) / (x[1] - x[0])
        above = (held[:, 4] - held[:, 3]) / (x[4] - x[3])
        return below, above

    below, above = slopes(_sphere.entries_at)
    scale = np.maximum(np.abs(below), np.abs(above)) + 1.0
    jump = np.abs(below - above) / scale
    print(f"\nthe remainder's slope either side of the edge: {below} | {above}; jump {jump}")
    assert jump.max() <= 1e-2
    with planted("edge_kink"):
        below, above = slopes(_sphere.entries_at)
    kinked = np.abs(below - above) / (np.maximum(np.abs(below), np.abs(above)) + 1.0)
    print(f"the continuation in value only: jump {kinked}")
    assert kinked.max() > 0.1


#: The knots a rebuild compares with the file, its widest gaps (the cheapest solves: 37 to 41 terms) and one at a
#: radius (55): each built in a fresh process, as the script builds the file, its kernels pinned (``PINNED``).
REBUILT_KNOTS = (96, 95, 72)


def test_the_script_rebuilds_the_files_knots_byte_for_byte():
    """The file holds the numbers and nothing of the run that made them (no time, no versions), and its numbers are
    made in pinned kernels, so a rebuild matches it: each knot's five entries, its gap and its solve's record equal
    the shipped file's, float for float."""
    held = table_file()
    assert set(held) == {"description", "entries", "range", "knots", "d", *_sphere.ENTRIES, "midpoints", "solves"}
    assert all(set(solve) == {"terms", "nodes"} for solve in held["solves"])
    done = subprocess.run([sys.executable, str(Path(__file__).parents[2] / "scripts" / "sphere_plane_table.py"),
                           "knots", *map(str, REBUILT_KNOTS)], capture_output=True, text=True, check=False, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    rebuilt = json.loads(done.stdout.strip().splitlines()[-1])
    for i in REBUILT_KNOTS:
        row = rebuilt[str(i)]
        shipped = {"d": held["d"][i], **{name: held[name][i] for name in _sphere.ENTRIES}, "solve": held["solves"][i]}
        moved = {k: (row[k], shipped[k]) for k in shipped if row[k] != shipped[k]}
        print(f"\nknot {i} (d {row['d']:.6g}): " + ("equal" if not moved else f"moved {moved}"))
        assert not moved, (i, moved)


# -- the seam ---------------------------------------------------------------------------------------------------------

def turned_frame(angle: float) -> np.ndarray:
    """A right-handed frame whose third row is the plane's normal turned by ``angle`` about x."""
    c, s = math.cos(angle), math.sin(angle)
    normal = np.array([0.0, -s, c])
    first = np.array([1.0, 0.0, 0.0])
    return np.stack([first, np.cross(normal, first), normal])


def plane_contact(gap: float, radius: float, angle: float, centre_offset: np.ndarray) -> tuple[NearContact, np.ndarray]:
    frame = turned_frame(angle)
    point = np.array([0.3, -0.2, 0.7]) * radius
    k = 1.0 / radius
    held = NearContact(*(jnp.asarray(v) for v in (point, frame, gap, k, k, k, k)))
    return held, point + radius * frame[2] + centre_offset


def test_on_a_plane_the_sphere_form_is_the_tables_increment_to_rounding():
    """The seam's sphere form at a plane contact (the gap's and the body's curvatures all one over the radius), any
    frame, about a centre off the sphere's: the plane table's exact increment between the two gaps, carried there."""
    from mime.nodes.hydro._film import about

    radius, viscosity = 2e-3, 1.3e-3
    for gap, delta, angle in ((1e-5, 3e-4, 0.4), (4e-4, 1e-4, 2.0), (2.5e-6, 6e-4, -1.1)):
        offset = np.array([1e-4, -3e-4, 2e-4])
        contact, centre = plane_contact(gap, radius, angle, offset)
        found = np.asarray(increment("sphere", contact, jnp.asarray(delta), jnp.asarray(1e-12), jnp.asarray(0.0),
                                     jnp.asarray(viscosity), jnp.asarray(centre), jnp.asarray(radius)).resistance)
        unit = _sphere.entries_at(gap / radius) - _sphere.entries_at((gap + delta) / radius)
        arms = np.array([1.0, 1.0, 1.0, radius, radius, radius])
        local = np.asarray(_sphere.plane_matrix(jnp.asarray(unit))) * viscosity * radius * np.outer(arms, arms)
        want = np.asarray(about(jnp.asarray(local), jnp.asarray(turned_frame(angle)), jnp.asarray(-offset)))
        scale = np.sqrt(np.abs(np.outer(np.diag(want), np.diag(want))))
        worst = float(np.max(np.abs(found - want) / scale))
        print(f"\nh {gap}, delta {delta}: the sphere form against the table's increment {worst:.2e}")
        assert worst <= 1e-10, (gap, worst)


def test_the_ambient_is_read_with_its_zero_on_the_opened_wall():
    """A shear flow ``u = (gamma z, 0, 0)`` vanishing on the true wall z = 0, read where the seam reads it at the opened
    geometry (the wall moved to z = -delta), vanishes on the opened wall and gives the body's points the shear at their
    lifted height; the true wall's convention (the point unmoved) would leave it -gamma delta there."""
    assert AMBIENT_ZERO == "opened wall"
    contact, _ = plane_contact(1e-5, 1e-3, 0.0, np.zeros(3))
    delta, gamma = 2e-4, 3.0

    def shear(point: np.ndarray) -> float:
        return gamma * float(point[2])

    on_opened_wall = np.array([0.1e-3, 0.0, -delta])
    read = np.asarray(ambient_point(jnp.asarray(on_opened_wall), contact, jnp.asarray(delta)))
    assert shear(read) == pytest.approx(0.0, abs=1e-15)
    assert abs(shear(on_opened_wall)) == pytest.approx(gamma * delta)


def test_the_plan_picks_the_form_by_the_bodys_kind_and_the_report_names_it():
    from mime.nodes.hydro.mrs._handover import UNIFORM_SPHERE_TEXT
    from mime.scene import Ellipsoid

    # a coarse blob: the form the plan picks and the report's line do not depend on it
    sphere = stepper(above_plane(None, blob=0.25).build())[0]
    ellipsoid = stepper(above_plane(0.15, blob=0.25, kind=Ellipsoid(semi_axes=m((1.1 * RADIUS, RADIUS, 0.95 * RADIUS))))
                        .build())[0]
    assert sphere.near_contact_form == "sphere" and ellipsoid.near_contact_form == "film"
    lines = sphere.built_lines({"pose": plane_pose(0.01 * RADIUS)}, None)
    named = [line for line in lines if "near-contact law's increment, by its form 'sphere'" in line]
    print("\n" + "\n".join(named))
    assert len(named) == 1 and "sphere_plane_resistance" in named[0]
    # the error it reports is the measured one at the spacing it runs at, not a number from another spacing
    assert f"a uniform force spacing of 0.251 radii: {UNIFORM_SPHERE_TEXT}" in named[0], named[0]


def graded_sphere(near_contact: bool | None) -> Any:
    """The sphere above the plane at a force spacing of 0.2 radii, its force points graded towards the contact for a
    gap of 0.075 radii (``ForceSampling(contact=...)``), the hand-over as ``near_contact`` (unwritten where ``None``)."""
    from mime.interactions import Drag, ForceSampling

    return above_plane(None, name="graded_sphere", drag=Drag.MRS(
        spacing=m(0.2 * RADIUS), wall_kind="plane", **({} if near_contact is None else {"near_contact": near_contact}),
        sampling=ForceSampling(contact=(0.0, 0.0, -1.0), contact_gap=m(0.075 * RADIUS))))


def test_a_graded_sphere_is_refused_the_spheres_form_with_its_measured_error_and_its_routes():
    from mime.build import BuildRefused

    for near_contact in (None, True):
        with pytest.raises(BuildRefused) as refused:
            graded_sphere(near_contact).plan()
        found = [r for r in refused.value.refusals if r.code == "drag.near-contact-graded"]
        assert len(found) == 1, refused.value.codes()
        print(f"\n{found[0].message}\n{found[0].action}")
        assert "+9.5 %" in found[0].message and "near_contact=False" in found[0].action
    graded_sphere(False).plan()  # the graded MRS alone near the wall: not this refusal's


def retired(rung: str, written: object) -> Any:
    """The refusal ``lubrication=written`` meets on drag 0.5.0's ``rung`` (a sphere above the plane)."""
    from mime.build import BuildRefused
    from mime.data import asset
    from mime.interactions import Drag

    drag = Drag.MRS(spacing=m(0.2 * RADIUS), wall_kind="plane", lubrication=written) if rung == "mrs" else \
        Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=written)
    with pytest.raises(BuildRefused) as refused:
        above_plane(None, name="retired", drag=drag).plan()
    found = [r for r in refused.value.refusals if r.code == "lubrication.retired"]
    assert len(found) == 1, refused.value.codes()
    print(f"\n{rung}, lubrication={written!r}: {found[0].action}")
    return found[0]


@pytest.mark.parametrize("written", [False, True])
def test_the_retired_blend_on_the_mrs_rung_names_the_hand_overs_own_switch_at_the_value_written(written):
    """``Drag.MRS(lubrication=False)``, the hand-over's name before the near-contact law: its way out keeps the user's
    intent, ``near_contact=False`` (never "leave lubrication out", which turns the hand-over on)."""
    action = retired("mrs", written).action
    assert f"near_contact={written!r}" in action and "leave lubrication out" not in action, action


def test_the_retired_blend_on_the_surrogate_names_drag_0_4_0_which_keeps_it():
    from mime.interactions import Lubrication
    from mime.units import mm

    action = retired("surrogate", Lubrication(epsilon=mm(0.15))).action
    assert "KEPT['drag', '0.4.0']" in action and "near_contact" not in action, action


# -- gates 1 and 2 above a plane, at the family's spacing ------------------------------------------------------------

def drawn_orientations(count: int) -> list[tuple[float, ...]]:
    import os

    rng = np.random.default_rng(int(os.environ.get("DETECTION_SEED", "1")))
    return [tuple(float(c) for c in q / np.linalg.norm(q)) for q in rng.normal(size=(count, 4))]


def states(gaps, *, plant: str | None = None, orientations=((1.0, 0.0, 0.0, 0.0),), roughness: float = FINE):
    """The node's state at each (orientation, gap), a sphere above a plane at the family's spacing."""
    with planted(plant):
        node, state, leaves, step = stepper(above_plane(None, roughness=roughness).build())
        return node, leaves, {(q, gap): {k: np.asarray(v) for k, v in step(plane_pose(gap * RADIUS, q), leaves).items()}
                              for q in orientations for gap in gaps}


def entry_errors(resistance: np.ndarray, gap: float, rows: dict) -> dict[str, float]:
    """Each diagonal entry against the graded MFS (the normal against Brenner's series), each coupling on its own
    size."""
    found, held = scaled(resistance), np.asarray(rows[f"C {gap!r}"]["resistance"])
    out = {name: float(found[i, i] / held[i, i] - 1.0) for i, name in enumerate(("xx", "yy", "zz", "rxrx", "ryry",
                                                                                 "rzrz"))}
    out["zz"] = float(found[2, 2] / (6 * math.pi * brenner(gap)) - 1.0)
    out["x_ry"] = float(found[0, 4] / held[0, 4] - 1.0)
    out["y_rx"] = float(found[1, 3] / held[1, 3] - 1.0)
    return out


def test_gate1_every_entry_within_its_line_at_every_gap_and_drawn_orientation():
    rows = plane_mfs()
    node, leaves, found = states(GAPS, orientations=drawn_orientations(5))
    handover = HANDOVER_SPACINGS * float(leaves["handover_spacing"]) / RADIUS
    assert handover == pytest.approx(HANDOVER, rel=1e-12)  # the family's spacing gate 8 straddles
    print(f"\nthe family's spacing {float(leaves['handover_spacing']) / RADIUS:.4g} radii, h_m {handover:.4g}")
    worst: dict[str, float] = {}
    for (q, gap), out in found.items():
        assert np.all(np.isfinite(out["resistance"])) and float(out["least_eigenvalue"]) > 0.0, (q, gap)
        errors = entry_errors(out["resistance"], gap, rows)
        for name, value in errors.items():
            worst[name] = max(worst.get(name, 0.0), abs(value))
        assert max(abs(v) for v in errors.values()) <= ENTRY_LINE, (q, gap, errors)
        assert float(out["handover_window"]) == 0.0 and float(out["film_unsolved"]) == 0.0
    print("worst over the gaps and draws: " + " ".join(f"{k} {v:.4f}" for k, v in worst.items()))


def plane_increment(gap: float, handover: float) -> np.ndarray:
    """The plane's exact increment between ``gap`` and its opened gap (radii), unit sphere and viscosity."""
    opened = float(smooth_floor(jnp.asarray(gap), jnp.asarray(handover), jnp.asarray(LIFT_WIDTH * handover)))
    return np.asarray(_sphere.plane_matrix(jnp.asarray(_sphere.entries_at(gap) - _sphere.entries_at(opened))))


def increment_errors(plant: str | None = None) -> dict[float, float]:
    node, leaves, found = states(GAPS, plant=plant)
    handover = HANDOVER_SPACINGS * float(leaves["handover_spacing"]) / RADIUS
    out = {}
    for gap in GAPS:
        got, want = scaled(found[((1.0, 0.0, 0.0, 0.0), gap)]["contact_increment"]), plane_increment(gap, handover)
        if not np.any(want):
            continue
        scale = np.sqrt(np.abs(np.outer(np.diag(want), np.diag(want))))
        out[gap] = float(np.max(np.abs(got - want)[scale > 0] / scale[scale > 0]))
    return out


#: The node's increment against the table's: the node reads the sphere's radius from its support's curvature,
#: regularised by RADIUS_FLOOR (1e-6) of the body's size, so its radius is 1e-6 of itself long (measured: 2.6e-6 to
#: 3.0e-6 of each entry's own scale).
INCREMENT_LINE = 1e-5


def test_gate1_the_increment_through_the_node_is_the_planes_exact_increment():
    """The node's ``contact_increment`` against the table's increment between the gap and the opened gap, entry by
    entry on its own diagonal scale, at every gap below the opening's band: the node's frame, arm and scaling."""
    found = increment_errors()
    print("\n" + " ".join(f"{g}: {e:.1e}" for g, e in found.items()))
    assert found and max(found.values()) <= INCREMENT_LINE, found


def test_gate1_control_the_increment_x11_is_caught_at_every_gap_below_the_band():
    found = increment_errors("increment_x1.1")
    print("\n" + " ".join(f"{g}: {e:.3f}" for g, e in found.items()))
    assert found and min(found.values()) >= 0.09, found


def rate_errors(plant: str | None = None) -> dict[float, float]:
    rows = plane_mfs()
    _, _, found = states(GAPS, plant=plant)
    return {gap: translation_per_rotation(found[((1.0, 0.0, 0.0, 0.0), gap)]["resistance"])
            / rate(np.asarray(rows[f"C {gap!r}"]["resistance"])) - 1.0 for gap in GAPS}


def test_gate2_a_sphere_spun_on_the_plane_translates_at_the_exact_rate_at_every_gap():
    found = rate_errors()
    print("\n" + " ".join(f"{g}: {e:+.4f}" for g, e in found.items()))
    assert max(abs(e) for e in found.values()) <= RATE_LINE, found


def test_gate2_control_the_coupling_counted_twice_is_caught_below_the_handover():
    _, leaves, _ = states((0.3,))
    handover = HANDOVER_SPACINGS * float(leaves["handover_spacing"]) / RADIUS
    found = rate_errors("coupling_twice")
    below = {gap: e for gap, e in found.items() if gap <= handover * (1.0 - LIFT_WIDTH)}
    print("\n" + " ".join(f"{g}: {e:+.4f}" for g, e in found.items()))
    assert below and min(abs(e) for e in below.values()) > RATE_LINE, below


# -- gate 2 in a tube -------------------------------------------------------------------------------------------------

#: The ratios and the gaps the tube MFS resolves (radii): its level B against C under 0.7 % on the plane rate's scale.
RATIOS = (0.22, 0.28, 0.34)
TUBE_GAPS = (0.1, 0.075, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075)
#: The lines on the plane rate's scale (round the circumference and along the axis) and the normal's, committed; along
#: the axis at 0.34 the prediction (10 %) was missed at the family's spacing, its measured range pinned: a stated limit
#: of the hybrid, the maintainer's decision (design row #360), never a line moved to pass it.
ACROSS_LINE, ALONG_LINE, NORMAL_LINE = 0.04, 0.10, 0.06
ALONG_MISS = {0.34: (-0.13, -0.11)}
#: Every entry of the 6 x 6 in the tube's frame (the contact along -x, the axis z), judged against the tube MFS where
#: the MFS resolves it (its level B against C within RESOLVED of the entry): each diagonal on its own size within
#: ENTRY_LINE (gate 1's), and each coupling on the plane's coupling at that gap (the scale the rates are judged on)
#: within COUPLING_LINE, the rolling along the axis's at 0.34 its measured range pinned (``COUPLING_MISS``). Not on the
#: coupling's own size, as the fix round predicted: in the tube both couplings pass through zero among these gaps
#: (round the circumference at 0.22 near 0.015 radii, along the axis at 0.28 between 0.075 and 0.06), where their own
#: size cannot discriminate (measured: +118 % and -168 % of it there, +3.0 % and +8.7 % of the plane's coupling).
TUBE_DIAGONAL = {"normal": (0, 0), "yy": (1, 1), "zz": (2, 2), "rxrx": (3, 3), "ryry": (4, 4), "rzrz": (5, 5)}
TUBE_COUPLINGS = {"y_rz": (1, 5), "z_ry": (2, 4)}
RESOLVED = 0.01
COUPLING_LINE = 0.10
COUPLING_MISS: dict[tuple[str, float], tuple[float, float]] = {("z_ry", 0.34): (0.14, 0.18)}
#: The ratios above the gated range the PR's review measured the rolling along the axis at (the tube MFS rows of
#: ``references/sphere_tube_mfs_above.json``), and the misses there, pinned: the flag ``handover_tube_ratio`` says so.
ABOVE = (0.40, 0.46)
ABOVE_MISS = {0.40: (-0.20, -0.16), 0.46: (-0.27, -0.23)}


def tube_mfs_above() -> dict:
    return pinned(REFERENCES / "sphere_tube_mfs_above.json", TUBE_MFS_ABOVE_SHA256)["rows"]


def rolls(r: np.ndarray) -> tuple[float, float]:
    """A unit sphere spun about y and about z, force-free: its speed along the axis and round the circumference."""
    v = -np.linalg.solve(r[:3, :3], r[:3, 3:])
    return float(v[2, 1]), float(v[1, 2])


def tube_errors(ratio: float, plant: str | None = None, rows: dict | None = None) -> dict[float, dict[str, float]]:
    """At each gap, the rates' errors over the plane rate at that gap, the normal's own, each entry's against the tube
    MFS (a diagonal on its own size, a coupling on the plane's coupling at that gap; ``None`` where the MFS does not
    resolve it), and the node's flags."""
    rows = tube_mfs() if rows is None else rows
    tube = RADIUS / ratio
    with planted(plant):
        node, state, leaves, step = stepper(in_tube(None, tube=tube).build())
        assert node.near_contact_form == "sphere"
        out = {}
        for gap in TUBE_GAPS:
            found = step(tube_pose(gap * RADIUS, tube=tube), leaves)
            got = scaled(np.asarray(found["resistance"]))
            want = np.asarray(rows[f"{ratio!r} C {gap!r}"]["resistance"])
            coarse = np.asarray(rows[f"{ratio!r} B {gap!r}"]["resistance"])
            plane_rate = -float(_sphere.entries_at(gap)[3] / _sphere.entries_at(gap)[1])
            (along, across), (along_want, across_want) = rolls(got), rolls(want)
            out[gap] = {"along": (along - along_want) / plane_rate, "across": (across - across_want) / plane_rate,
                        "normal": float(got[0, 0] / want[0, 0] - 1.0),
                        "tube_ratio": float(found["handover_tube_ratio"])}
            plane_coupling = abs(float(_sphere.entries_at(gap)[3]))
            for name, (i, j) in (TUBE_DIAGONAL | TUBE_COUPLINGS).items():
                resolved = abs(coarse[i, j] / want[i, j] - 1.0) <= RESOLVED
                scale = want[i, j] if i == j else plane_coupling
                out[gap][f"entry {name}"] = float((got[i, j] - want[i, j]) / scale) if resolved else None
    return out


def entry_worst(found: dict[float, dict[str, float]], name: str) -> tuple[float, int]:
    """An entry's worst error over the gaps the MFS resolves it at, and how many those are."""
    held = [found[g][f"entry {name}"] for g in TUBE_GAPS if found[g][f"entry {name}"] is not None]
    return (max(held, key=abs) if held else 0.0), len(held)


@pytest.mark.parametrize("ratio", RATIOS)
def test_gate2_in_a_tube_the_hybrid_rolls_at_the_tube_mfss_rates(ratio):
    found = tube_errors(ratio)
    worst = {k: max((found[g][k] for g in TUBE_GAPS), key=abs) for k in ("along", "across", "normal")}
    print(f"\na/R {ratio}: worst " + " ".join(f"{k} {v:+.4f}" for k, v in worst.items()))
    assert abs(worst["across"]) <= ACROSS_LINE and abs(worst["normal"]) <= NORMAL_LINE, worst
    pinned_range = ALONG_MISS.get(ratio)
    if pinned_range is None:
        assert abs(worst["along"]) <= ALONG_LINE, worst
    else:
        assert pinned_range[0] <= worst["along"] <= pinned_range[1], worst
    # every entry where the MFS resolves it: the rotations the rates do not read among them
    for name in (*TUBE_DIAGONAL, *TUBE_COUPLINGS):
        error, judged = entry_worst(found, name)
        print(f"  {name}: worst {error:+.4f} over the {judged} gaps the MFS resolves it at")
        assert judged >= 3, (name, judged)  # never a gate that judged nothing
        miss = COUPLING_MISS.get((name, ratio))
        if miss is not None:
            assert miss[0] <= error <= miss[1], (name, error)
        else:
            assert abs(error) <= (ENTRY_LINE if name in TUBE_DIAGONAL else COUPLING_LINE), (name, error)
    assert all(found[g]["tube_ratio"] == 0.0 for g in TUBE_GAPS)  # within the gated range: never flagged


@pytest.mark.memory_heavy
@pytest.mark.parametrize("plant", ["plane_in_tube", "relative_coefficients"])
def test_gate2_in_a_tube_controls_the_planes_form_and_the_gaps_curvature_alone(plant):
    for ratio in RATIOS:
        found = tube_errors(ratio, plant)
        across = max((found[g]["across"] for g in TUBE_GAPS), key=abs)
        print(f"\n{plant}, a/R {ratio}: across {across:+.4f}")
        assert abs(across) > ACROSS_LINE, (plant, ratio, across)


@pytest.mark.memory_heavy
def test_gate2_in_a_tube_control_the_rotations_swapped_is_caught_on_the_rotation_entries():
    """The two rotations' denominators swapped (equal on a plane, so only a tube sees it): the rates and the normal do
    not read the rotation entries, which the gate's entries do."""
    for ratio in RATIOS:
        found = tube_errors(ratio, "rot_swap")
        worst = max((abs(entry_worst(found, name)[0]) for name in ("ryry", "rzrz")))
        print(f"\nrot_swap, a/R {ratio}: the rotations' worst {worst:+.4f}")
        assert worst > 3 * ENTRY_LINE, (ratio, worst)


@pytest.mark.parametrize("ratio", ABOVE)
def test_above_the_gated_ratio_the_step_is_flagged_and_the_along_axis_miss_is_the_measured_one(ratio):
    """Past TESTED_TUBE_RATIO every step within the opening's band raises handover_tube_ratio, and the build report says
    so; the rolling along the axis there misses the tube MFS by the measured amount (the flag's text), round the
    circumference and the normal holding their lines."""
    from mime.nodes.hydro.mrs._handover import TESTED_TUBE_RATIO

    assert ratio > TESTED_TUBE_RATIO
    found = tube_errors(ratio, rows=tube_mfs_above())
    worst = {k: max((found[g][k] for g in TUBE_GAPS), key=abs) for k in ("along", "across", "normal")}
    print(f"\na/R {ratio}: worst " + " ".join(f"{k} {v:+.4f}" for k, v in worst.items()))
    assert all(found[g]["tube_ratio"] == 1.0 for g in TUBE_GAPS)
    low, high = ABOVE_MISS[ratio]
    assert low <= worst["along"] <= high, worst
    assert abs(worst["across"]) <= ACROSS_LINE and abs(worst["normal"]) <= NORMAL_LINE, worst
    tube = RADIUS / ratio
    node = stepper(in_tube(None, tube=tube).build())[0]
    lines = node.built_lines({"pose": tube_pose(0.01 * RADIUS, tube=tube)}, None)
    assert any("raises handover_tube_ratio" in line for line in lines), lines


# -- gate 8 -----------------------------------------------------------------------------------------------------------

GRADIENT_LINE = 1e-5
#: The family's spacing for this sphere is 0.251 radii, its hand-over gap 0.18825 (the opening's kink without its
#: smoothing) and the opening's band 0.094 to 0.282; the table's edge is 1e-3 radii; a roughness of 2 um on each
#: surface puts the cutoff at 2.83 um, its band 1.41 to 4.24 um.
HANDOVER = HANDOVER_SPACINGS * 0.251
HANDOVER_GAPS = (0.3, 0.25, HANDOVER, 0.15, 0.1)
EDGE_GAPS = (1.2e-3, 1e-3, 0.85e-3)
CUTOFF_GAPS = (0.0035, 0.0028, 0.002)


@pytest.mark.parametrize("where", ["handover", "edge", "cutoff"])
def test_gate8_forward_and_reverse_equal_the_difference(where):
    gaps, roughness, leaves = {"handover": (HANDOVER_GAPS, FINE, ("wall_offset",)),
                               "edge": (EDGE_GAPS, FINE, ("wall_offset",)),
                               "cutoff": (CUTOFF_GAPS, 2e-6, ("wall_offset", "roughness_body"))}[where]
    with planted():
        found = compared(above_plane(None, roughness=roughness).build(), lambda g: plane_pose(g * RADIUS), gaps, leaves)
    for gap, name, reverse, forward, difference, scale in found:
        print(f"\n{where} h/a {gap} {name}: reverse {reverse:.10g} forward {forward:.10g} difference {difference:.10g}")
        assert abs(reverse - difference) <= GRADIENT_LINE * scale, (gap, name, reverse, difference)
        assert abs(forward - difference) <= GRADIENT_LINE * scale, (gap, name, forward, difference)


@pytest.mark.parametrize("plant", ["lift_stopped", "hard_max", "edge_kink"])
def test_gate8_controls(plant):
    gaps = {"lift_stopped": (0.1, 0.0028), "hard_max": (HANDOVER,), "edge_kink": (1e-3,)}[plant]
    with planted(plant):
        found = compared(above_plane(None).build(), lambda g: plane_pose(g * RADIUS), gaps, ("wall_offset",),
                         forward_too=False)
    worst = max(abs(reverse - difference) / scale for _, _, reverse, _, difference, scale in found)
    print(f"\n{plant}: the worst |AD - FD| / scale {worst:.3g}")
    assert worst > 10 * GRADIENT_LINE, plant


# -- gate 13: the form's derivative in the sphere's radius -----------------------------------------------------------

#: The relative step of the sphere's radius across the two builds, and the line of the derivative against their
#: difference (whose truncation is about the step squared): the film's gate 13's.
SHAPE_STEP, SHAPE_LINE = 1e-3, 1e-4
#: A force spacing written, so that the hand-over gap holds as the radius moves (the family's follows the radius).
SHAPE_SPACING = 0.251
#: The entries the increment holds, by the wall (the others are zero by symmetry, their ratio rounding's): above the
#: plane (normal z) the normal, along the wall, its rotation, their coupling and the spin; in the tube (the contact along
#: -x, the axis z) the normal, round the circumference and along the axis, the rotations about them and their couplings.
SHAPE_ENTRIES = {"plane": ((2, 2), (0, 0), (4, 4), (0, 4), (5, 5)),
                 "tube": ((0, 0), (1, 1), (2, 2), (4, 4), (5, 5), (1, 5), (2, 4))}
SHAPE_GAPS = (0.05, 0.005)
SHAPE_TUBE_RATIO = 0.28


def shape_cases(wall: str) -> dict[float, tuple]:
    """The sphere at a (1 + factor), the wall at its place, the force spacing written: three builds."""
    tube = RADIUS / SHAPE_TUBE_RATIO
    return {factor: stepper((above_plane(SHAPE_SPACING, kind=Sphere(radius=m(factor * RADIUS))) if wall == "plane" else
                             in_tube(SHAPE_SPACING, kind=Sphere(radius=m(factor * RADIUS)), tube=tube)).build())
            for factor in (1.0, 1.0 + SHAPE_STEP, 1.0 - SHAPE_STEP)}


def increment_at(case, wall: str, radius, gap: float):
    """The node's increment with the sphere's radius leaf at ``radius``, its surface ``gap`` (radii) off the wall."""
    node, state, leaves, _ = case
    held = {**leaves, "shape_radius": radius}
    pose = plane_pose(gap * RADIUS, lowest=radius) if wall == "plane" else \
        tube_pose(gap * RADIUS, reach=radius, tube=RADIUS / SHAPE_TUBE_RATIO)
    return node.update(state, {"pose": pose}, 1e-3, params=held)["contact_increment"]


def radius_ratios(wall: str, plant: str | None = None) -> dict[tuple[float, tuple[int, int]], float]:
    """d(contact increment)/d(shape_radius) by reverse mode through the node's own step, over the two builds' central
    difference, at each gap and entry."""
    with planted(plant):
        cases = shape_cases(wall)
        rows, columns = (jnp.asarray(index) for index in zip(*SHAPE_ENTRIES[wall], strict=True))
        found = {}
        for gap in SHAPE_GAPS:
            derivatives = np.asarray(jax.jacrev(lambda r: increment_at(cases[1.0], wall, r, gap)[rows, columns])(
                jnp.asarray(RADIUS)))
            up = np.asarray(increment_at(cases[1.0 + SHAPE_STEP], wall, (1.0 + SHAPE_STEP) * RADIUS, gap))
            down = np.asarray(increment_at(cases[1.0 - SHAPE_STEP], wall, (1.0 - SHAPE_STEP) * RADIUS, gap))
            for k, (i, j) in enumerate(SHAPE_ENTRIES[wall]):
                rebuilt = (float(up[i, j]) - float(down[i, j])) / (2.0 * SHAPE_STEP * RADIUS)
                found[gap, (i, j)] = float(derivatives[k]) / rebuilt
    return found


@pytest.mark.memory_heavy
@pytest.mark.parametrize("wall", ["plane", "tube"])
def test_gate13_the_spheres_increment_in_its_radius_through_the_node_equals_two_builds_difference(wall):
    """As the film's gate 13 (``test_lubrication.py``): the sphere's form reads its radius from the body's curvature at
    the contact, so its derivative in the radius goes through the plane remainder's scaling and the singular terms'
    curvatures, which a single build cannot check against another."""
    found = radius_ratios(wall)
    for (gap, entry), ratio in found.items():
        print(f"\n{wall} h/a {gap} entry {entry}: through the node over two builds {ratio:.8f}")
    assert all(abs(ratio - 1.0) <= SHAPE_LINE for ratio in found.values()), found


def test_gate13_control_the_radius_without_its_derivative_is_caught():
    found = radius_ratios("plane", "radius_stopped")
    worst = max(abs(ratio - 1.0) for ratio in found.values())
    print(f"\nthe radius stopped: the worst |ratio - 1| {worst:.3g}")
    assert worst > 100 * SHAPE_LINE, found


# -- dtypes and two builds -------------------------------------------------------------------------------------------

def test_float32_lies_within_its_line_of_float64():
    gaps = (0.3, 0.1, 0.05, 0.01, 0.002)
    wide = states(gaps)[2]
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        narrow = states(gaps)[2]
    finally:
        jax.config.update("jax_enable_x64", held)
    for gap in gaps:
        key = ((1.0, 0.0, 0.0, 0.0), gap)
        a, b = wide[key]["resistance"], np.asarray(narrow[key]["resistance"], dtype=np.float64)
        scale = np.sqrt(np.abs(np.outer(np.diag(a), np.diag(a))))
        found = float(np.max(np.abs(b - a) / scale))
        print(f"\nh/a {gap}: float32 against float64 {found:.3g} of the diagonal scale")
        assert found <= 1e-3, (gap, found)


def test_the_two_builds_check_is_quiet_in_the_spheres_radius():
    from mime.testing import two_builds_of

    found = two_builds_of(above_plane(None), "ball.parts.shell.radius", run=False)
    print(f"\n{found.verdict}")
    assert found.verdict == "quiet", found


def test_the_opening_is_the_seams_lift():
    """The MRS hands over by the seam's opening: zero from the hand-over gap plus its width up, the gap lifted to the
    hand-over gap below its band."""
    resolved = jnp.asarray(0.15)
    assert float(opening(jnp.asarray(0.3), resolved)) == 0.0
    assert float(opening(jnp.asarray(0.01), resolved)) == pytest.approx(0.14)
