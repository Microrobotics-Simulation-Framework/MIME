"""Lubrication's hand-over on the MRS rung (drag 0.5.0's ``lubrication``), body to wall, against exact answers and its
controls: W2a's gates 1-4 and 7-14, predictions committed first (``tests/oracles/predictions_w2a.json``, and for the
fix round's gates ``predictions_w2a_fix.json``).

- **Gate 1, a sphere above a plane, every gap, drawn orientations:** the lifted resistance's normal entry against
  Brenner's series at every gap (0.002 to 0.3 radii), and every entry against the method of fundamental solutions
  where it resolves (0.06 to 0.3 radii, to 2e-4: ``references/lubrication_mfs.json``), at force spacings of 0.15 and
  0.2 radii (0.2 is the widest the film's window allows a sphere). The committed 7 % is missed at one place, the normal
  at 0.06 radii at a spacing of 0.2 (-6.7 to -7.4 % over orientations): reported as missed, its range pinned. Controls:
  the lift left out (the coarse MRS alone) misses at every gap below the hand-over gap; the film's increment x1.1 is
  caught only where a tenth of the increment exceeds the line less the measured error, a range the test prints and
  holds.
- **The increment on its own scale:** in the lift's band against the method of fundamental solutions' increments
  (each entry's ratio pinned: MIME-ANO-154's form), and between two deep gaps against Brenner and Goldman, Cox and
  Brenner, within 7 %. Control: the increment x1.1, caught by both.
- **Gate 2, the coupling:** a sphere spun on the wall, force-free, translates at the exact rate per rotation: against
  the method of fundamental solutions at every gap it resolves (the lift's band among them), and against Goldman, Cox
  and Brenner's asymptote at 0.01 radii and below, where its own error is under about 2 %. Below the hand-over gap the
  committed 10 % is missed at every gap (MIME-ANO-154), each pinned. Control: the film's coupling counted twice.
- **Gate 3, the form:** the normal entry's local exponent in the gap is -1 and its limit ``R_zz h / (6 pi mu a^2)`` is 1
  as the gap closes (the node, roughness 1 nm); the film alone on a line contact goes as ``h^-3/2``. Controls: a sphere's
  law on the line contact (the exponent), the radius off by two (the limit).
- **Gate 4, definiteness:** drawn poses of a sphere and an ellipsoid above the plane and of a sphere in a tube are
  definite. Control: an indefinite increment stops the run (NaN).
- **Gate 7, the register's x2 control:** the term doubled moves the force-free speed under a normal load to what the
  node's own reported resistance and increment predict. Control: the term left single.
- **Gate 8, gradients:** forward and reverse against central differences across the hand-over gap and the roughness
  cutoff, in the plane's offset, the body's height and the tube's radius. Controls: a ``stop_gradient`` on the lift; a
  hard max (its derivative jumps at the hand-over gap).
- **Gate 9:** the two-builds check is quiet in each surface's roughness and the sphere's radius.
- **Gate 10 (dtypes):** float32 within 1e-3 of float64 on the diagonal scale.
- **Gate 11, never silent:** a wrong film solve is flagged (``film_unsolved``) at every step; a clean run never.
- **Gate 12, the cutoff:** the public reader (``roughness_cutoff``, standing for W2b's traction law) gives the cutoff
  the film uses; the film's normal force below the cutoff follows its floored gap; value and slope continuous through
  the cutoff. Control: a film whose cutoff is not the reader's.
- **Gate 13, a shape leaf:** the film's increment differentiated in the sphere's radius through the node equals the
  difference across two builds. Control: the film's curvature floored at one over a held size.
- **Gate 14, the tube's film:** the tube's normal entry times the gap tends to the leading order for the contact's
  relative curvatures. Control: the wall's curvature added to the body's.
- **The film's window, the build's rule:** a body whose least radius of curvature puts the hand-over gap past the
  window is the coarse MRS's (FL-9 at its family's spacing; the sphere at 0.3 radii), reported; written on, refused.

Float64 (``tests/oracles/conftest.py``) but for gate 10. Each gate's own test honours ``W2A_PLANT``
(``lubrication_cases.planted``), and gates 1 and 4 draw by ``DETECTION_SEED``, so ``.github/probe/detection_rate.py``
runs their controls over fresh processes."""

import hashlib
import json
import math
from pathlib import Path

import _cache_ceiling
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lubrication_cases import (
    FINE,
    RADIUS,
    TUBE,
    VISCOSITY,
    above_plane,
    in_tube,
    planted,
    plane_pose,
    scaled,
    screw_in_tube,
    stepper,
    taken_default,
    translation_per_rotation,
    tube_pose,
    turned,
)
from plane_oracles import brenner

from mime.nodes.hydro._film import film
from mime.interactions import roughness_cutoff
from mime.nodes.hydro.mrs._handover import HANDOVER_SPACINGS, LIFT_WIDTH, ROUGHNESS_WIDTH
from mime.scene import Ellipsoid, Sphere
from mime.units import m

#: How far above where a worker stands after giving its compiled programs back this file's tests let its resident memory
#: grow before the next gives them back again (tests/_cache_ceiling.py), as tests/oracles/test_hull_contact.py does:
#: each gate builds the drag's node at blob 0.05 and compiles its step, its gradients or its float32 twin, and JAX keeps
#: every program while its node lives, so a worker running the gates one after another grows by each.
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


REFERENCE = Path(__file__).parents[1] / "differential" / "references" / "lubrication_mfs.json"
REFERENCE_SHA256 = "80df714b5d1ca3fcdd80867b8f2193e39bbebd711a2d3731d5a4e196fa52ac9b"
#: The gaps (radii) judged against the method of fundamental solutions, and those against Brenner's series alone.
MFS_GAPS = (0.3, 0.2, 0.15, 0.1, 0.075, 0.06)
NORMAL_GAPS = (0.03, 0.01, 0.005, 0.002)
#: Each spacing judged (of the radius) and its line per entry (predictions_w2a.json, gate 1): 7 %. A sphere's
#: hand-over gap is within the film's window up to a spacing of 0.2 radii; at 0.3 the build's rule leaves it out.
LINES = {0.15: 0.07, 0.2: 0.07}
#: Where gate 1's committed line is missed (the film's form at the window's edge, MIME-ANO-154's): the normal entry at
#: a spacing of 0.2 radii and a gap of 0.06, its range over drawn orientations, pinned (measured -6.79 to -7.27 % over
#: 20 orientations by the review, -6.92 % unturned).
NORMAL_MISS = {(0.2, 0.06): (-0.076, -0.065)}
#: The orientations gate 1 draws a run (by DETECTION_SEED).
ORIENTATIONS = 5
#: Gate 2's line on the translation per rotation.
COUPLING_LINE = 0.1


def drawn_orientations(count: int) -> list[tuple[float, float, float, float]]:
    import os

    rng = np.random.default_rng(int(os.environ.get("DETECTION_SEED", "1")))
    return [tuple(float(c) for c in q / np.linalg.norm(q)) for q in rng.normal(size=(count, 4))]


def reference() -> dict:
    data = REFERENCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == REFERENCE_SHA256, "the reference file is not the one pinned"
    return json.loads(data)


def exact(gap: float) -> np.ndarray:
    return np.asarray(reference()["rows"][f"fine {gap}"]["resistance"])


def errors(resistance: np.ndarray, gap: float) -> dict[str, float]:
    """Each diagonal entry's error against the method of fundamental solutions, the couplings' on sqrt(R_ii R_jj),
    and the normal's against Brenner."""
    found, held = scaled(resistance), exact(gap)
    scale = np.sqrt(np.abs(np.outer(np.diag(held), np.diag(held))))
    out = {f"diag{i}": float(found[i, i] / held[i, i] - 1.0) for i in range(6)}
    out["coupling"] = float(max(abs(found[0, 4] - held[0, 4]) / scale[0, 4], abs(found[1, 3] - held[1, 3]) / scale[1, 3]))
    out["normal"] = float(found[2, 2] / (6 * math.pi * brenner(gap)) - 1.0)
    return out


def at_gaps(spacing: float, gaps, *, lubrication: bool = True, roughness: float = FINE, plant: str | None = None,
            orientations=((1.0, 0.0, 0.0, 0.0),)):
    """The node's state at each gap (radii), the sphere at each of ``orientations`` (the unturned one alone, keyed by
    gap; several, keyed by (orientation, gap)), its hand-over at ``spacing``."""
    with planted(plant):
        node, state, leaves, step = stepper(above_plane(spacing, lubrication=lubrication, roughness=roughness).build())
        found = {(q, gap): {key: np.asarray(value) for key, value in step(plane_pose(gap * RADIUS, q), leaves).items()}
                 for q in orientations for gap in gaps}
    return {gap: out for (_, gap), out in found.items()} if len(orientations) == 1 else found


def test_the_reference_is_the_pinned_file_and_resolves_to_its_line():
    held = reference()
    for gap in MFS_GAPS:
        coarse, fine = (np.asarray(held["rows"][f"{s} {gap}"]["resistance"]) for s in ("coarse", "fine"))
        scale = np.sqrt(np.abs(np.outer(np.diag(fine), np.diag(fine))))
        assert float(np.max(np.abs(coarse - fine) / scale)) < 5e-4, gap
        assert abs(held["rows"][f"fine {gap}"]["normal_over_brenner"] - 1.0) < 2e-4, gap


# -- gate 1 ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spacing", sorted(LINES))
def test_gate1_every_entry_within_its_line_of_the_exact_answer_at_every_gap_and_orientation(spacing):
    """Every entry within the line at every gap over drawn orientations, but where the line is missed
    (:data:`NORMAL_MISS`): there the miss stays in its measured range."""
    line = LINES[spacing]
    draws = drawn_orientations(ORIENTATIONS)
    found = at_gaps(spacing, MFS_GAPS + NORMAL_GAPS, orientations=draws)
    for (q, gap), out in found.items():
        assert np.all(np.isfinite(out["resistance"])), gap
        miss = errors(out["resistance"], gap) if gap in MFS_GAPS else {
            "normal": float(scaled(out["resistance"])[2, 2] / (6 * math.pi * brenner(gap)) - 1.0)}
        print(f"\ns/a {spacing}, h/a {gap}, q {np.round(q, 3)}: " + " ".join(f"{k} {v:+.4f}" for k, v in miss.items()))
        pinned = NORMAL_MISS.get((spacing, gap))
        if pinned is not None:
            assert pinned[0] <= miss.pop("normal") <= pinned[1], (gap, q)
            miss.pop("diag2", None)
        assert max(abs(v) for v in miss.values()) <= line, (gap, q, miss)
        assert float(out["handover_window"]) == 0.0 and float(out["handover_far_side"]) == 0.0, gap
        assert float(out["least_eigenvalue"]) > 0.0 and float(out["film_unsolved"]) == 0.0, gap


def test_gate1_control_the_lift_left_out_misses_at_every_gap_below_the_handover():
    spacing = 0.2
    handover = HANDOVER_SPACINGS * spacing
    # where the lift holds the contact at the hand-over gap: at h_m itself the coarse MRS alone is its own hand-over's
    # value, within the line, so the control cannot discriminate there
    gaps = tuple(g for g in MFS_GAPS + NORMAL_GAPS if g <= handover * (1.0 - LIFT_WIDTH) + 1e-12)
    for gap, out in at_gaps(spacing, gaps, lubrication=False).items():
        normal = float(scaled(out["resistance"])[2, 2] / (6 * math.pi * brenner(gap)) - 1.0) \
            if np.all(np.isfinite(out["resistance"])) else math.inf
        print(f"\nthe coarse MRS alone, h/a {gap}: normal {normal:+.4f}")
        assert abs(normal) > LINES[spacing], (gap, normal)


def test_gate1_control_the_increment_x11_is_caught_where_its_tenth_exceeds_the_line():
    """The discriminating range of the x1.1 control: where 10 % of the film's share exceeds the line less the lift's
    measured error. Below the smallest gap printed as caught, the gate sees a 10 % error in the film's coefficients;
    above it, it is silent on one (the increment's own judgement, below, is not)."""
    spacing = 0.2
    gaps = MFS_GAPS + NORMAL_GAPS
    clean, bent = at_gaps(spacing, gaps), at_gaps(spacing, gaps, plant="increment_x1.1")
    caught = []
    for gap in gaps:
        a, b = scaled(clean[gap]["resistance"]), scaled(bent[gap]["resistance"])
        normal = float(b[2, 2] / (6 * math.pi * brenner(gap)) - 1.0)
        if abs(normal) > LINES[spacing]:
            caught.append(gap)
        print(f"\nx1.1, h/a {gap}: normal {normal:+.4f} (clean {a[2, 2] / (6 * math.pi * brenner(gap)) - 1.0:+.4f})")
    print(f"\ncaught at h/a {caught}")
    assert {0.005, 0.002} <= set(caught), caught


# -- the increment on its own scale -----------------------------------------------------------------------------------

def gcb(gap: float) -> dict[str, float]:
    """Goldman, Cox and Brenner (1967), Chem. Eng. Sci. 22, 637, with O'Neill and Stewartson (1967), JFM 27, 705, for
    the force: a sphere translating and rotating along a plane wall at a gap ``gap`` radii, unit radius and viscosity,
    the force and torque on the fluid about the centre (the reference file's convention). Their lubrication
    asymptotes, accurate where the gap is small: against the method of fundamental solutions the tangential entry
    reads 2.0 % low at 0.06 radii and the coupling 35 % low (its constant is a small difference there)."""
    log = math.log(1.0 / gap)
    return {"xx": 6 * math.pi * ((8 / 15) * log + 0.9588), "xry": -6 * math.pi * ((2 / 15) * log - 0.2526),
            "ryy": 8 * math.pi * ((2 / 5) * log + 0.3817), "zz": 6 * math.pi * brenner(gap)}


#: The film's increment over the method of fundamental solutions', entry by entry, in the lift's band (a spacing of
#: 0.2 radii, the lift at the hand-over gap 0.15 exactly): MIME-ANO-154's form (its patch at the hand-over gap), pinned;
#: and the line between two deep gaps, where the patch's part cancels.
INCREMENT_BAND = {0.075: {"xx": 0.685, "zz": 0.816, "ryry": 0.730, "x-ry": 1.829},
                  0.06: {"xx": 0.706, "zz": 0.843, "ryry": 0.751, "x-ry": 1.765}}
INCREMENT_PIN = 0.01
DEEP_PAIRS = ((0.001, 0.005), (0.002, 0.01))
DEEP_LINE = 0.07
ENTRIES = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "rxrx": (3, 3), "ryry": (4, 4), "x-ry": (0, 4), "y-rx": (1, 3)}
#: Which pinned ratio each entry is judged against (the sphere's two along-wall directions alike).
SAME = {"yy": "xx", "rxrx": "ryry", "y-rx": "x-ry"}


def increment_ratios(plant: str | None = None) -> tuple[dict[float, dict[str, float]], dict[tuple, dict[str, float]]]:
    found = at_gaps(0.2, tuple(INCREMENT_BAND) + tuple(sorted({g for pair in DEEP_PAIRS for g in pair})), plant=plant)
    held = reference()["rows"]
    exact = {g: np.asarray(held[f"fine {g}"]["resistance"]) for g in (0.15, *INCREMENT_BAND)}
    band = {}
    for gap in INCREMENT_BAND:
        film, wanted = scaled(found[gap]["film_increment"]), exact[gap] - exact[0.15]
        band[gap] = {name: float(film[i, j] / wanted[i, j]) for name, (i, j) in ENTRIES.items()}
    deep = {}
    for near, far in DEEP_PAIRS:
        moved = scaled(found[near]["resistance"]) - scaled(found[far]["resistance"])
        a, b = gcb(near), gcb(far)
        wanted = {"xx": a["xx"] - b["xx"], "yy": a["xx"] - b["xx"], "zz": a["zz"] - b["zz"], "rxrx": a["ryy"] - b["ryy"],
                  "ryry": a["ryy"] - b["ryy"], "x-ry": a["xry"] - b["xry"], "y-rx": -(a["xry"] - b["xry"])}
        deep[(near, far)] = {name: float(moved[i, j] / wanted[name]) for name, (i, j) in ENTRIES.items()}
    return band, deep


def test_the_increment_judged_on_its_own_scale_in_the_band_and_between_deep_gaps():
    band, deep = increment_ratios()
    for gap, ratios in band.items():
        print(f"\nband h/a {gap}: " + " ".join(f"{k} {v:.3f}" for k, v in ratios.items()))
        for name, ratio in ratios.items():
            assert abs(ratio - INCREMENT_BAND[gap][SAME.get(name, name)]) <= INCREMENT_PIN, (gap, name, ratio)
    for pair, ratios in deep.items():
        print(f"\ndeep {pair}: " + " ".join(f"{k} {v:.4f}" for k, v in ratios.items()))
        assert max(abs(r - 1.0) for r in ratios.values()) <= DEEP_LINE, (pair, ratios)


def test_control_the_increment_x11_is_caught_in_the_band_and_between_deep_gaps():
    band, deep = increment_ratios(plant="increment_x1.1")
    assert all(abs(ratio - INCREMENT_BAND[gap][SAME.get(name, name)]) > INCREMENT_PIN
               for gap, ratios in band.items() for name, ratio in ratios.items()), band
    assert all(max(abs(r - 1.0) for r in ratios.values()) > DEEP_LINE for ratios in deep.values()), deep


# -- gate 2 ----------------------------------------------------------------------------------------------------------

#: The gaps gate 2 judges against Goldman, Cox and Brenner's asymptote: where its own translation per rotation is within
#: about 2 % of the exact (it reads 32 % fast at 0.06 radii against the method of fundamental solutions, and its error
#: falls about as the gap [inference: the coupling's next order, about 0.7 times the gap of its 6 pi a^2, fitted to
#: the method of fundamental solutions at 0.06 to 0.1 radii]).
ASYMPTOTE_GAPS = (0.01, 0.005, 0.002, 0.001)
#: The translation per rotation's error where the committed 10 % is missed (MIME-ANO-154: every gap below the hand-over
#: gap, the lift's band and the asymptote's gaps), measured through the node, pinned so that a change either way is
#: seen; every other gap judged is held to the line.
COUPLING_MISS = {0.2: {0.15: 0.078, 0.1: 0.277, 0.075: 0.381, 0.06: 0.428,
                       0.01: 0.413, 0.005: 0.343, 0.002: 0.282, 0.001: 0.251},
                 0.15: {0.1: 0.096, 0.075: 0.202, 0.06: 0.265, 0.01: 0.329, 0.005: 0.275, 0.002: 0.227, 0.001: 0.203}}
#: How far a pinned miss may move before the test fails: the same build, so only a change of the model moves it.
COUPLING_PIN = 0.01


def gcb_rate(gap: float) -> float:
    """A force-free sphere spun about y at a unit rate: its translation along x per rotation from the asymptotes."""
    held = gcb(gap)
    return -held["xry"] / held["xx"]


def coupling_errors(spacing: float, plant: str | None = None) -> dict[float, float]:
    found = at_gaps(spacing, MFS_GAPS + ASYMPTOTE_GAPS, plant=plant)
    out = {}
    for gap in MFS_GAPS + ASYMPTOTE_GAPS:
        if gap in MFS_GAPS:
            held = exact(gap)
            want = float(-np.linalg.solve(held[:3, :3], held[:3, 4])[0])
        else:
            want = gcb_rate(gap)
        out[gap] = translation_per_rotation(found[gap]["resistance"]) / want - 1.0
        print(f"\ns/a {spacing}, h/a {gap}: U / (a Omega) {out[gap]:+.4f} against the "
              f"{'method of fundamental solutions' if gap in MFS_GAPS else 'asymptote'}")
    return out


@pytest.mark.parametrize("spacing", sorted(COUPLING_MISS))
def test_gate2_a_sphere_spun_on_the_wall_translates_at_the_exact_rate_per_rotation_or_its_pinned_miss(spacing):
    """Within the line at every gap judged where it holds; where it is missed (MIME-ANO-154), pinned."""
    for gap, miss in coupling_errors(spacing).items():
        pinned = COUPLING_MISS[spacing].get(gap)
        if pinned is None:
            assert abs(miss) <= COUPLING_LINE, (gap, miss)
        else:
            assert abs(miss - pinned) <= COUPLING_PIN, (gap, miss)


def test_gate2_control_the_coupling_counted_twice_moves_it_past_its_pin():
    clean, twice = coupling_errors(0.2), coupling_errors(0.2, plant="coupling_twice")
    for gap, pinned in COUPLING_MISS[0.2].items():
        assert abs(twice[gap] - pinned) > 10 * COUPLING_PIN, (gap, twice[gap], clean[gap])


# -- gate 3 ----------------------------------------------------------------------------------------------------------

FORM_GAPS = (1e-3, 1.5e-3, 2.2e-3, 3.3e-3, 5e-3)


def test_gate3_the_sphere_normal_goes_as_one_over_the_gap_and_its_limit_is_one():
    found = at_gaps(0.2, FORM_GAPS)
    normal = np.asarray([found[g]["resistance"][2, 2] for g in FORM_GAPS])
    exponents = np.diff(np.log(normal)) / np.diff(np.log(FORM_GAPS))
    limit = float(normal[0] * FORM_GAPS[0] * RADIUS / (6 * math.pi * VISCOSITY * RADIUS ** 2))
    print(f"\nexponents {exponents}, limit {limit:.5f}")
    assert np.all(np.abs(exponents + 1.0) <= 0.05), exponents
    assert abs(limit - 1.0) <= 0.03, limit


def film_normal(gaps, k1, k2) -> np.ndarray:
    return np.asarray([float(film(jnp.asarray(g), jnp.asarray(k1), jnp.asarray(k2), jnp.asarray(1.0), jnp.asarray(1.0),
                                  jnp.asarray(1e-6)).resistance[2, 2]) for g in gaps])


LINE_GAPS = (1e-4, 2e-4, 4e-4, 8e-4)


def test_gate3_the_film_on_a_line_contact_goes_as_the_gap_to_minus_three_halves():
    exponents = np.diff(np.log(film_normal(LINE_GAPS, 1.0, 0.0))) / np.diff(np.log(LINE_GAPS))
    print(f"\nline contact exponents {exponents}")
    assert np.all(np.abs(exponents + 1.5) <= 0.05), exponents


def test_gate3_controls_a_spheres_law_on_the_line_and_a_radius_off_by_two():
    sphere_law = np.diff(np.log(film_normal(LINE_GAPS, 1.0, 1.0))) / np.diff(np.log(LINE_GAPS))
    assert np.all(np.abs(sphere_law + 1.5) > 0.3), sphere_law
    halved = float(film_normal((1e-4,), 0.5, 0.5)[0] * 1e-4 / (6 * math.pi))
    print(f"\nthe sphere's law on the line: {sphere_law}; the limit with the radius doubled {halved:.4f}")
    assert abs(halved - 1.0) > 1.0, halved


# -- gate 4 ----------------------------------------------------------------------------------------------------------

def draws(count: int, seed: int) -> list[tuple[float, tuple[float, float, float, float]]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        q = rng.normal(size=4)
        out.append((float(np.exp(rng.uniform(math.log(1e-3), math.log(1.0)))), tuple(float(c) for c in q / np.linalg.norm(q))))
    return out


def lowest(node, leaves, orientation) -> float:
    from mime.core import quaternion

    points = np.asarray(node.body({k: leaves[k] for k in node.body_leaf_names()}).points)
    turn = np.asarray(quaternion.rotation_matrix(jnp.asarray(orientation)))
    return float(-np.min((points @ turn.T)[:, 2]))


@pytest.mark.memory_heavy
@pytest.mark.parametrize("case", ["sphere above the plane", "ellipsoid above the plane", "sphere in the tube"])
def test_gate4_the_lift_form_is_definite_at_drawn_poses(case):
    import os

    seed = int(os.environ.get("DETECTION_SEED", "4"))
    with planted():
        if case == "sphere in the tube":
            built = in_tube(0.2).build()
        elif case.startswith("ellipsoid"):
            # its least radius of curvature, at the ends of its longest axis, is 0.82 of the sphere's: the window holds
            # its hand-over gap at a spacing of 0.15 radii
            built = above_plane(0.15, kind=Ellipsoid(semi_axes=m((1.1 * RADIUS, RADIUS, 0.95 * RADIUS)))).build()
        else:
            built = above_plane(0.2).build()
        node, state, leaves, step = stepper(built)
        indefinite = 0
        for gap, orientation in draws(25, seed):
            if case == "sphere in the tube":
                pose = tube_pose(gap * RADIUS, orientation)
            else:
                pose = plane_pose(gap * RADIUS, orientation, lowest(node, leaves, orientation))
            out = step(pose, leaves)
            if not (float(out["least_eigenvalue"]) > 0.0 and np.all(np.isfinite(np.asarray(out["resistance"])))):
                indefinite += 1
    print(f"\n{case}: {indefinite} of 25 indefinite")
    assert indefinite == 0


def test_gate4_control_an_indefinite_increment_stops_the_run():
    found = at_gaps(0.2, (0.01,), plant="indefinite")[0.01]
    assert not np.all(np.isfinite(found["resistance"])) and float(found["indefinite"]) == 1.0


# -- gate 7 ----------------------------------------------------------------------------------------------------------

def test_gate7_the_term_doubled_moves_the_speed_by_what_the_six_by_six_predicts():
    """Under a unit normal load, force-free otherwise, the speed with the film's increment doubled against the
    prediction from the node's own reported resistance and increment, ``(R + dTF)^-1 F``."""
    gaps = (0.05, 0.02, 0.005)
    single, doubled = at_gaps(0.2, gaps), at_gaps(0.2, gaps, plant="increment_x2")
    load = np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    for gap in gaps:
        predicted = np.linalg.solve(single[gap]["resistance"] + single[gap]["film_increment"], load)
        got = np.linalg.solve(doubled[gap]["resistance"], load)
        moved = np.linalg.solve(single[gap]["resistance"], load)
        print(f"\nh/a {gap}: speed x1 {moved[2]:.6g}, x2 {got[2]:.6g}, predicted {predicted[2]:.6g}")
        assert abs(got[2] / predicted[2] - 1.0) <= 1e-9, gap
        assert abs(moved[2] / predicted[2] - 1.0) > 1e-3, gap  # the control: the term left single misses


# -- gate 8 ----------------------------------------------------------------------------------------------------------

def compared(built, pose_of, gaps, leaf_names, step=1e-6):
    """Reverse, forward and Richardson's central difference of a weighted sum of the resistance in each leaf, at each
    gap; each leaf stepped by ``step`` of its size. The loss takes the pose and the weights as arguments, so that each
    derivative compiles once for every gap."""
    node, state, leaves, _ = stepper(built)

    def loss(values, pose, along, weights):
        held = {**leaves, **{k: v for k, v in zip(leaf_names, values[:-1], strict=True)}}
        moved = pose + values[-1] * along
        return jnp.sum(weights * node.update(state, {"pose": moved}, 1e-3, params=held)["resistance"])

    value, reverse_of, forward_of = jax.jit(loss), jax.jit(jax.grad(loss)), jax.jit(jax.jacfwd(loss))
    found = []
    for gap in gaps:
        pose = pose_of(gap)
        scale = np.asarray(node.update(state, {"pose": pose}, 1e-3, params=leaves)["resistance"])
        weights = jnp.asarray((1.0 + 0.1 * np.arange(36).reshape(6, 6) / 36.0)
                              / np.sqrt(np.abs(np.outer(np.diag(scale), np.diag(scale)))))
        along = jnp.zeros(7).at[2 if float(pose[2]) != 0.0 else 0].set(1.0)
        at = jnp.asarray([*(leaves[k] for k in leaf_names), 0.0])
        reverse = np.asarray(reverse_of(at, pose, along, weights))
        forward = np.asarray(forward_of(at, pose, along, weights))
        here = []
        for i, name in enumerate((*leaf_names, "position")):
            size = RADIUS if name in ("position", "wall_offset", "wall_radius") else abs(float(at[i]))
            h = step * size
            e = jnp.zeros_like(at).at[i].set(h)
            coarse = (float(value(at + e, pose, along, weights)) - float(value(at - e, pose, along, weights))) / (2 * h)
            fine = (float(value(at + e / 2, pose, along, weights)) - float(value(at - e / 2, pose, along, weights))) / h
            # each derivative on the loss's scale per unit relative move of its leaf, so that a leaf the resistance
            # does not read at this gap (the roughness above the cutoff) is judged against the others
            here.append((gap, name, reverse[i] * size, forward[i] * size, (4 * fine - coarse) / 3 * size))
        scale = max(max(abs(d), abs(r)) for _, _, r, _, d in here)
        found += [(*row, scale) for row in here]
    return found


#: Gaps straddling the hand-over gap (h_m = 0.15 radii at s/a 0.2, the lift's band 0.075 to 0.225) and the cutoff (a
#: roughness of 2 um on each surface: 2.83 um, its band 1.41 to 4.24 um), in radii.
GRADIENT_GAPS = (0.3, 0.2, 0.15, 0.1, 0.0035, 0.0028, 0.002)
GRADIENT_LINE = 1e-5


@pytest.mark.memory_heavy
@pytest.mark.parametrize("wall", ["plane", "tube"])
def test_gate8_forward_and_reverse_equal_the_difference_across_the_handover_and_the_cutoff(wall):
    with planted():
        if wall == "plane":
            built = above_plane(0.2, roughness=2e-6).build()
            found = compared(built, lambda g: plane_pose(g * RADIUS), GRADIENT_GAPS, ("wall_offset", "roughness_body"))
        else:
            built = in_tube(0.2, roughness=2e-6).build()
            found = compared(built, lambda g: tube_pose(g * RADIUS), GRADIENT_GAPS[:4], ("wall_radius",))
    for gap, name, reverse, forward, difference, scale in found:
        print(f"\n{wall} h/a {gap} {name}: reverse {reverse:.10g} forward {forward:.10g} difference {difference:.10g}")
        assert abs(reverse - difference) <= GRADIENT_LINE * scale, (gap, name, reverse, difference)
        assert abs(forward - difference) <= GRADIENT_LINE * scale, (gap, name, forward, difference)


@pytest.mark.memory_heavy
def test_gate8_controls_a_stopped_lift_and_a_hard_max():
    gaps = (0.1, 0.0028)
    for plant in ("lift_stopped", "hard_max"):
        with planted(plant):
            built = above_plane(0.2, roughness=2e-6).build()
            found = compared(built, lambda g: plane_pose(g * RADIUS), gaps if plant == "lift_stopped" else (0.15,),
                             ("wall_offset",))
        worst = max(abs(reverse - difference) / scale for _, _, reverse, _, difference, scale in found)
        print(f"\n{plant}: the worst |AD - FD| / scale {worst:.3g}")
        assert worst > 100 * GRADIENT_LINE, plant


# -- gate 9 ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["ball.roughness", "floor.roughness", "ball.parts.shell.radius"])
def test_gate9_the_two_builds_check_is_quiet_on_the_lubricated_step(path):
    from mime.testing import two_builds_of

    with planted():
        found = two_builds_of(above_plane(0.2), path, run=False)
    print(f"\n{path}: {found.verdict}")
    assert found.verdict == "quiet", (path, found)


def test_gate9_control_the_bodys_size_baked_into_the_film_is_a_bake():
    """The first build's own defect planted back: the film's size read from the search's build-time number, a
    constant of the program, which the two-builds check reports in the sphere's radius."""
    from mime.testing import two_builds_of

    with planted("baked_size"):
        found = two_builds_of(above_plane(0.2), "ball.parts.shell.radius", run=False)
    print(f"\nbaked size: {found.verdict}")
    assert found.verdict != "quiet", found


# -- gate 10 (dtypes) -----------------------------------------------------------------------------------------------

def test_gate10_float32_lies_within_its_line_of_float64():
    gaps = (0.3, 0.1, 0.05, 0.01, 0.002)
    wide = at_gaps(0.2, gaps)
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        narrow = at_gaps(0.2, gaps)
    finally:
        jax.config.update("jax_enable_x64", held)
    for gap in gaps:
        a, b = wide[gap]["resistance"], np.asarray(narrow[gap]["resistance"], dtype=np.float64)
        scale = np.sqrt(np.abs(np.outer(np.diag(a), np.diag(a))))
        found = float(np.max(np.abs(b - a) / scale))
        print(f"\nh/a {gap}: float32 against float64 {found:.3g} of the diagonal scale")
        assert found <= 1e-3 and float(narrow[gap]["film_unsolved"]) == 0.0, (gap, found)


# -- gate 11 ---------------------------------------------------------------------------------------------------------

def test_gate11_a_wrong_film_solve_is_flagged_at_every_step_and_a_clean_one_never():
    gaps = (0.1, 0.05, 0.01, 0.002)
    for plant, expected in ((None, 0.0), ("wrong_solve", 1.0)):
        for gap, out in at_gaps(0.2, gaps, plant=plant).items():
            print(f"\n{plant} h/a {gap}: residual {float(out['film_residual']):.2g}, flag {float(out['film_unsolved'])}")
            assert float(out["film_unsolved"]) == expected, (plant, gap)


# -- gate 12 ---------------------------------------------------------------------------------------------------------

def test_gate12_the_reader_gives_the_films_cutoff_and_the_film_stops_growing_below_it():
    roughness = 2e-6
    cutoff = roughness_cutoff(roughness, roughness)
    gaps = (0.002, 0.001, 0.0005)  # radii: 2, 1 and 0.5 um, below the floor's band (1.41 to 4.24 um)
    found = at_gaps(0.2, gaps, roughness=roughness)
    for gap, out in found.items():
        assert float(out["film_cutoff"]) == pytest.approx(cutoff, rel=1e-12), gap
    floor = cutoff * (1.0 - ROUGHNESS_WIDTH)  # below the band's lower end the film's gap is the cutoff itself
    normals = [float(found[g]["resistance"][2, 2]) for g in gaps if g * RADIUS < floor]
    print(f"\ncutoff {cutoff:.4g} m; normal entries below the floor's band {normals}")
    assert len(normals) >= 2 and max(normals) / min(normals) - 1.0 <= 1e-9, normals


def test_gate12_the_film_is_continuous_in_value_and_slope_through_the_cutoff():
    roughness = 2e-6
    cutoff = roughness_cutoff(roughness, roughness)
    gaps = np.linspace(0.3 * cutoff, 1.7 * cutoff, 29) / RADIUS
    found = at_gaps(0.2, tuple(float(g) for g in gaps), roughness=roughness)
    values = np.asarray([float(found[float(g)]["resistance"][2, 2]) for g in gaps])
    slope = np.diff(values) / np.diff(gaps)
    jumps = np.abs(np.diff(slope)) / np.max(np.abs(slope))
    print(f"\nthe slope's largest step between neighbours {float(np.max(jumps)):.3g} of its largest")
    assert np.all(np.isfinite(values)) and float(np.max(jumps)) <= 0.2


def test_gate12_the_floored_normal_moves_with_the_walls_roughness_as_its_leading_order_predicts():
    """Below both cutoffs' floors (a gap of 1 um; the body's roughness 2 um, the wall's 2 um and then 4 um, cutoffs
    2.83 and 4.47 um) the film's normal increment is TF(cutoff) - TF(h_m); its leading order, 6 pi mu a^2 / h,
    predicts the ratio of the two, (1/c1 - 1/h_m) / (1/c2 - 1/h_m), to the size of its next order (a fifth of the
    logarithm over 1/h, a few parts in a thousand). Its control is the cutoff from the body's roughness alone (the
    wall's move then moves nothing)."""
    gap, body = 1e-3, 2e-6  # radii; m
    handover = HANDOVER_SPACINGS * 0.2 * RADIUS
    walls = (2e-6, 4e-6)
    found = {}
    with planted():
        for wall in walls:
            node, state, leaves, step = stepper(above_plane(0.2, roughness=body, wall_roughness=wall).build())
            found[wall] = float(step(plane_pose(gap * RADIUS), leaves)["film_increment"][2, 2])
    cutoffs = {wall: roughness_cutoff(body, wall) for wall in walls}
    measured = found[walls[0]] / found[walls[1]]
    predicted = (1 / cutoffs[walls[0]] - 1 / handover) / (1 / cutoffs[walls[1]] - 1 / handover)
    print(f"\nthe floored normal's ratio {measured:.5f}, predicted {predicted:.5f}")
    assert abs(measured / predicted - 1.0) <= 0.02, (measured, predicted)


def test_gate12_control_a_film_whose_cutoff_is_not_the_readers_is_caught():
    roughness = 2e-6
    out = at_gaps(0.2, (0.002,), roughness=roughness, plant="cutoff_body_only")[0.002]
    assert float(out["film_cutoff"]) != pytest.approx(roughness_cutoff(roughness, roughness), rel=1e-6)


def test_the_tube_is_opened_by_dilation_and_its_contact_found_off_the_axis():
    built = in_tube(0.2).build()
    node, state, leaves, step = stepper(built)
    assert node.lubricated
    out = step(tube_pose(0.01 * RADIUS), leaves)
    assert float(out["contact_gap"]) == pytest.approx(0.01 * RADIUS, rel=1e-6)
    assert float(out["least_eigenvalue"]) > 0.0 and float(out["film_annulus"]) == 0.0
    assert TUBE > RADIUS


def test_bodies_sharing_a_medium_near_a_wall_are_refused_the_handover_until_they_write_it_off():
    """The hand-over is built for one body against its wall; two bodies in one medium are refused it
    (``drag.lubrication-near-field``), the refusal naming ``lubrication=False`` as the way out (the near field's own
    cases build with it: ``tests/near_field/near_field_cases.py``)."""
    from mime.build import BuildRefused
    from mime.dynamics import RigidInertial
    from mime.interactions import Drag
    from mime.scene import Sphere
    from mime.things import Body, Part, State
    from mime.units import kg, kg_m2, m_per_s, rad_per_s

    def pair(**drag):
        experiment = above_plane(0.3, name="lubricated_pair")
        experiment.add_body(Body("other", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(RADIUS)))},
                                 initial=State(position=m((4.0 * RADIUS, 0.0, 3.0 * RADIUS)),
                                               orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
                                               angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        experiment.declare("other.mass", kg(1e-5), reason="a millimetre body's order")
        experiment.declare("other.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
        experiment.attach(Drag.MRS(spacing=m(0.3 * RADIUS), wall_kind="plane", **drag), body="other", medium="water")
        return experiment

    with pytest.raises(BuildRefused) as refused:
        pair().build()
    assert "drag.lubrication-near-field" in refused.value.codes()


# -- gate 13 ---------------------------------------------------------------------------------------------------------

#: The relative step of the sphere's radius across the two builds, and the line of the derivative against their
#: difference (whose truncation is about the step squared).
SHAPE_STEP = 1e-3
SHAPE_LINE = 1e-4
#: The film's entries judged: the normal, the tangential, the rotation and the coupling about the centre.
FILM_ENTRIES = ((2, 2), (0, 0), (4, 4), (0, 4))


def film_at(case, radius: float, gap: float):
    node, state, leaves, _ = case
    held = {**leaves, "shape_radius": radius}
    return node.update(state, {"pose": plane_pose(gap, lowest=radius)}, 1e-3, params=held)["film_increment"]


@pytest.mark.memory_heavy
def test_gate13_the_films_increment_in_the_spheres_radius_through_the_node_equals_two_builds_difference():
    """d(film increment)/d(shape_radius) by reverse mode through the node's own step, the body held a gap of 0.01 radii
    off the plane as its radius moves, against the central difference of two builds at a (1 +- SHAPE_STEP)."""
    gap = 0.01 * RADIUS
    with planted():
        # a spacing of 0.19 radii holds the hand-over gap within the window at a (1 - SHAPE_STEP) as well
        cases = {factor: stepper(above_plane(0.19, kind=Sphere(radius=m(factor * RADIUS))).build())
                 for factor in (1.0, 1.0 + SHAPE_STEP, 1.0 - SHAPE_STEP)}
        base = cases[1.0]
        rows, columns = (jnp.asarray(index) for index in zip(*FILM_ENTRIES, strict=True))
        derivatives = np.asarray(jax.jacrev(lambda r: film_at(base, r, gap)[rows, columns])(jnp.asarray(RADIUS)))
        up = np.asarray(film_at(cases[1.0 + SHAPE_STEP], (1.0 + SHAPE_STEP) * RADIUS, gap))
        down = np.asarray(film_at(cases[1.0 - SHAPE_STEP], (1.0 - SHAPE_STEP) * RADIUS, gap))
        for k, (i, j) in enumerate(FILM_ENTRIES):
            derivative = float(derivatives[k])
            rebuilt = (float(up[i, j]) - float(down[i, j])) / (2.0 * SHAPE_STEP * RADIUS)
            print(f"\nentry ({i}, {j}): through the node {derivative:.8g}, two builds {rebuilt:.8g}, "
                  f"ratio {derivative / rebuilt:.6f}")
            assert abs(derivative / rebuilt - 1.0) <= SHAPE_LINE, ((i, j), derivative, rebuilt)


# -- gate 14 ---------------------------------------------------------------------------------------------------------

TUBE_GAPS = (1e-3, 1.5e-3, 2.2e-3)


def leading_normal(gap: float, k1: float, k2: float) -> float:
    """The leading order of the film's normal resistance at a gap ``gap`` with relative curvatures ``k1, k2``:
    Reynolds' squeeze equation on the gap ``h + (k1 x^2 + k2 y^2) / 2`` is solved exactly by ``p = 6 mu V / ((k1 +
    k2) h^2)`` (substituting gives ``div(h^3 grad p) = -12 mu V``, the squeeze's source), whose force is ``12 pi mu V
    / (gap (k1 + k2) sqrt(k1 k2))`` [derived; at k1 = k2 = 1/a it is the sphere's 6 pi mu a^2 V / h, Brenner's
    (1961) and Cox and Brenner's (1967) leading term]."""
    return 12.0 * math.pi * VISCOSITY / (gap * (k1 + k2) * math.sqrt(k1 * k2))


def test_gate14_the_tubes_normal_tends_to_the_leading_order_for_the_contacts_relative_curvatures():
    """A sphere of radius a in a tube of radius R (a/R 0.5): the contact's relative curvatures are 1/a along the axis
    and 1/a - 1/R round it. The node's normal entry (along the contact's normal, x) over the leading order tends to 1
    and goes as the inverse gap."""
    k1, k2 = 1.0 / RADIUS, 1.0 / RADIUS - 1.0 / TUBE
    with planted():
        node, state, leaves, step = stepper(in_tube(0.2).build())
        normal = np.asarray([float(step(tube_pose(g * RADIUS), leaves)["resistance"][0, 0]) for g in TUBE_GAPS])
    ratios = normal / np.asarray([leading_normal(g * RADIUS, k1, k2) for g in TUBE_GAPS])
    exponents = np.diff(np.log(normal)) / np.diff(np.log(TUBE_GAPS))
    print(f"\nthe tube's normal over the leading order {ratios}, exponents {exponents}")
    assert abs(float(ratios[0]) - 1.0) <= 0.03, ratios
    assert np.all(np.abs(exponents + 1.0) <= 0.05), exponents


def test_the_far_side_is_flagged_where_the_body_fills_more_than_its_share_of_the_lumen():
    """``handover_far_side`` with the contact opened in a tube whose diameter the sphere fills past FAR_SIDE_FILL
    (a/R 0.55), and not at 0.5 (the gates' tube), nor with the contact beyond the lift."""
    from mime.nodes.hydro.mrs._handover import FAR_SIDE_FILL

    node, state, leaves, step = stepper(in_tube(0.2).build())
    for ratio, gap, raised in ((0.5, 0.02, 0.0), (0.55, 0.02, 1.0), (0.55, 0.3, 0.0)):
        radius = RADIUS / ratio
        held = {**leaves, "wall_radius": jnp.asarray(radius)}
        out = step(jnp.asarray((-(radius - RADIUS - gap * RADIUS), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)), held)
        print(f"\na/R {ratio}, h/a {gap}: handover_far_side {float(out['handover_far_side'])}")
        assert float(out["handover_far_side"]) == raised, (ratio, gap)
    assert FAR_SIDE_FILL == 0.5


# -- the film's window, the build's rule -----------------------------------------------------------------------------

def test_a_body_past_the_films_window_is_the_coarse_mrss_by_the_builds_rule_and_reported():
    """FL-9 in its tube at its family's spacing: its least radius of curvature (its caps' rims, an edge; its crest's
    0.197 mm) puts the hand-over gap far past the window, so the plan takes lubrication False, the node is the coarse
    MRS, and its build report says why with the gap against the radius. The sphere at a spacing of 0.3 radii (hand-over
    gap 0.225 radii) is past it too; at 0.2 (0.15) it is within, and takes the hand-over."""
    screw = screw_in_tube()
    assert taken_default(screw, "lubrication") is False
    node = stepper(screw.build())[0]
    assert not node.lubricated
    lines = [line for line in node.built_lines(None, None) if "hand-over is not taken" in line]
    print("\n" + "\n".join(lines))
    assert len(lines) == 1 and "past the film's window" in lines[0]
    assert taken_default(above_plane(0.3, lubrication=None), "lubrication") is False
    assert taken_default(above_plane(0.2, lubrication=None), "lubrication") is True


def test_lubrication_written_on_past_the_window_is_refused_naming_the_spacing_that_brings_it_in():
    from mime.build import BuildRefused

    for experiment in (screw_in_tube(lubrication=True), above_plane(0.3)):
        with pytest.raises(BuildRefused) as refused:
            experiment.plan()
        assert refused.value.codes() == ["drag.lubrication-window"], refused.value.codes()
        assert "spacing=mm(" in refused.value.refusals[0].action


def test_control_the_rule_removed_gives_fl9s_default_to_the_film(monkeypatch):
    import mime.interactions.hydrodynamic as hydrodynamic

    monkeypatch.setattr(hydrodynamic, "WINDOW", math.inf)
    assert taken_default(screw_in_tube(), "lubrication") is True


def test_a_shared_flowing_medium_collects_both_refusals_at_once():
    """Two spheres within the window in one flowing tube: each drag refused for the flow (``drag.medium-flows``) and
    the medium refused the hand-over for bodies sharing it (``drag.lubrication-near-field``), in one build's refusals,
    so writing the one route does not reveal the other."""
    from mime.build import BuildRefused
    from mime.dynamics import RigidInertial, Womersley
    from mime.interactions import Drag
    from mime.things import Body, Part, State
    from mime.units import kg, kg_m2, m_per_s, rad_per_s

    experiment = in_tube(0.2, lubrication=None, flow=Womersley(), name="lubricated_pair_flowing")
    experiment.add_body(Body("other", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(RADIUS)))},
                             initial=State(position=m((0.0, 0.0, 6.0 * RADIUS)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("other.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("other.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    experiment.attach(Drag.MRS(spacing=m(0.2 * RADIUS), wall_kind="tube"), body="other", medium="water")
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert sorted(refused.value.codes()) == ["drag.lubrication-near-field", "drag.medium-flows",
                                             "drag.medium-flows"], refused.value.codes()


def test_the_report_says_where_the_roughness_floor_reaches_the_lifts_band():
    """A roughness of 40 um on each surface (cutoff 57 um, its floor up to 85 um) reaches the lift's band (from 75 um
    at a spacing of 0.2 mm): one line says so; at the gates' 1 nm, none does."""
    for roughness, said in ((40e-6, True), (FINE, False)):
        node, state, leaves, _ = stepper(above_plane(0.2, roughness=roughness).build())
        lines = node.built_lines({"pose": plane_pose(0.01 * RADIUS)}, None)
        assert any("roughness floor reaches the lift's band" in line for line in lines) == said, roughness
