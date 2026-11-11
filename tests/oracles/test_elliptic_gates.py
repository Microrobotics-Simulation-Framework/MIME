"""The elliptic point form and several contacts through the drag node (W2a PR 1b-ii-a; predictions committed first,
``tests/oracles/predictions_w2a_1b2.json``):

- **Gate 1, spheroids above a plane:** a prolate spheroid of aspect 2, lying and tilted 30 degrees, at a force spacing of
  0.2 of its minor semi-axis ``b`` (hand-over gap 0.15 b), against the graded MFS (``references/
  spheroid_plane_graded_mfs.json``, level L3; L2 its resolution) at gaps 0.1 to 0.02 b and drawn turns about the wall's
  normal: the force-free rolling rate (the translation vector per rotation, worst of the two wall-parallel spin axes)
  and each entry on its own scale, pinned where measured (MIME-ANO-154's stated limitation for non-spheres, not an
  accuracy bound). The plan's window rule is set aside here, as MIME-ANO-154's second probe did: a spheroid's tips put
  it past the window at any spacing, while its contact here is the equator's or near it. Control: the summed increment
  doubled.
- **Gate 3, the form through the node:** an ellipsoid's normal entry goes as the inverse gap, and its limit is the
  elliptic contact's ``12 pi mu / (s (k1 + k2))`` for the curvatures at its lowest point. Control: the sphere's law on
  that elongated contact (the larger curvature taken for both).
- **Gate 5, two contacts on one surface:** an ellipsoid lying across a tube, its two ends below the hand-over gap on
  opposite walls: the lift (the tube dilated once, two increments summed) is definite and within its line of a finer
  MRS of the same family (``references/ellipsoid_across_tube_mrs.json``, ``tests/differential/two_contact_mrs.py``),
  which it bounds against, saying nothing about hardware. Controls, each failing with both contacts closed: "one at a
  time" (each contact opened alone by the tube dilated and shifted, the others at their actual gaps), S4's subtraction
  (route (b): the coarse MRS at the actual gaps plus each contact's one-contact correction) and the per-body lift (the
  body moved off the nearer wall, closing the other gap).
- **Gate 8 on two contacts:** forward and reverse against Richardson's differences in the tube's radius and centre and
  the body's position, both contacts below the hand-over gap and across it.

Float64 (``tests/oracles/conftest.py``). The controls honour ``W2A_PLANT`` (``lubrication_cases.planted``) and gate 1
draws by ``DETECTION_SEED``, so ``.github/probe/detection_rate.py`` runs them over fresh processes."""

import hashlib
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lubrication_cases import RADIUS, VISCOSITY, above_plane, across, across_tube, compared, planted, stepper

from mime.core import as_run_array
from mime.nodes.hydro.mrs._handover import HANDOVER_SPACINGS, handed_over
from mime.nodes.hydro.near_contact import increment, opening
from mime.scene import Ellipsoid
from mime.units import mm

REFERENCES = Path(__file__).parents[1] / "differential" / "references"
SPHEROIDS = REFERENCES / "spheroid_plane_graded_mfs.json"
SPHEROIDS_SHA256 = "062c4f33fe1540ba1c0b045eb6737c29105586de50c272e53ec94d250251bc30"
#: Gate 1's gaps (of the minor semi-axis), each below the hand-over gap 0.15, and its turns about the normal a run.
SPHEROID_GAPS = (0.1, 0.075, 0.06, 0.04, 0.03, 0.02)
TURNS = 3
#: The measured worst misses gate 1 pins (rate; entry on its own scale), by tilt, and how far each may move.
#: Measured (W2a PR 1b-ii-a, laptop heavy slot, float64, 3 turns a tilt): lying, the rate 22.46 % (probe B's 22.5 %)
#: and an entry 33.3 % (probe B's T_y 28.5 %); at 30 degrees 16.67 % (17.0 %) and 17.9 % (17.9 %).
SPHEROID_PINNED: dict[float, tuple[float, float]] = {0.0: (0.2246, 0.3331), 30.0: (0.1667, 0.1793)}
SPHEROID_PIN = 0.02
#: The coupling entries judged on their own size: those over this share of their block's largest (probe B's rule).
SIGNIFICANT = 0.05


@pytest.fixture(autouse=True)
def window_set_aside(monkeypatch):
    """Gate 1's spheroid and gate 5's ellipsoid are judged at their contacts, inside the window there; the plan's rule,
    judged at the body's least radius (a spheroid's tips), is set aside, as the step's own ``handover_window`` reports."""
    import mime.interactions.hydrodynamic as hydrodynamic

    monkeypatch.setattr(hydrodynamic, "WINDOW", math.inf)


def spheroid_rows() -> dict:
    data = SPHEROIDS.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SPHEROIDS_SHA256, "the spheroid reference is not the one pinned"
    return json.loads(data)["rows"]


def tilted(degrees: float) -> tuple[float, float, float, float]:
    """The quaternion of the reference's tilt: about y, the body's x axis raised towards +z."""
    half = -math.radians(degrees) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def turn_about_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def composed(q: tuple[float, ...], angle: float) -> tuple[float, float, float, float]:
    """The turn ``angle`` about z after the body's orientation ``q``."""
    w, x, y, z = q
    c, s = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)


def lowest_of(axes: np.ndarray, rotation: np.ndarray) -> float:
    """How far below its centre an ellipsoid of semi-axes ``axes`` turned by ``rotation`` reaches: its support along -z."""
    down = rotation.T @ np.array([0.0, 0.0, -1.0])
    return float(np.sqrt(np.sum((axes * down) ** 2)))


def rolling(resistance: np.ndarray) -> np.ndarray:
    """The force-free translation vector per unit rotation about x and about y (rows)."""
    return np.stack([-np.linalg.solve(resistance[:3, :3], resistance[:3, 3 + k]) for k in (0, 1)])


def judged(found: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """The worst rolling rate's miss (vector, against the reference's size, the two spin axes) and the worst entry on
    its own scale (each diagonal against itself; each significant coupling against its own size)."""
    want, got = rolling(reference), rolling(found)
    rate = float(max(np.linalg.norm(got[k] - want[k]) / np.linalg.norm(want[k]) for k in (0, 1)))
    entries = [abs(found[i, i] / reference[i, i] - 1.0) for i in range(6)]
    block = np.max(np.abs(reference[:3, 3:]))
    entries += [abs(found[i, j] / reference[i, j] - 1.0) for i in range(3) for j in range(3, 6)
                if abs(reference[i, j]) > SIGNIFICANT * block]
    return rate, float(max(entries))


def spheroid_misses(tilt: float, plant: str | None = None) -> dict[tuple[float, float], tuple[float, float]]:
    """Gate 1's misses at each gap and drawn turn about the normal: the spheroid's node at s 0.2 b, its pose turned."""
    rows = spheroid_rows()
    axes = np.array([2.0, 1.0, 1.0]) * RADIUS
    rng = np.random.default_rng(int(os.environ.get("DETECTION_SEED", "1")))
    angles = rng.uniform(0.0, 2.0 * math.pi, size=TURNS)
    out = {}
    with planted(plant):
        node, state, leaves, step = stepper(above_plane(0.2, kind=Ellipsoid(semi_axes=mm((2.0, 1.0, 1.0))),
                                                        name="lubricated_spheroid").build())
        for angle in angles:
            q = composed(tilted(tilt), float(angle))
            from mime.core import quaternion

            rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray(q)))
            spin = turn_about_z(float(angle))
            both = np.block([[spin, np.zeros((3, 3))], [np.zeros((3, 3)), spin]])
            for gap in SPHEROID_GAPS:
                pose = jnp.asarray((0.0, 0.0, lowest_of(axes, rotation) + gap * RADIUS, *q))
                found = np.asarray(step(pose, leaves)["resistance"]) / VISCOSITY
                reference = np.asarray(rows[f"2 {tilt:g} L3 {gap:g}"]["resistance"])
                scale = np.array([1.0, 1.0, 1.0, RADIUS, RADIUS, RADIUS])
                reference = both @ reference @ both.T * RADIUS * np.outer(scale, scale)
                out[(float(angle), gap)] = judged(found, reference)
    return out


def test_the_spheroid_reference_is_pinned_and_resolved_between_its_levels():
    rows = spheroid_rows()
    for tilt in (0, 30):
        for gap in SPHEROID_GAPS:
            fine, coarse = (np.asarray(rows[f"2 {tilt} {level} {gap:g}"]["resistance"]) for level in ("L3", "L2"))
            rate, entry = judged(coarse, fine)
            print(f"\ntilt {tilt}, h/b {gap}: L2 against L3, rate {rate:.4f}, entry {entry:.4f}")
            assert rate <= 0.02 and entry <= 0.02, (tilt, gap, rate, entry)


@pytest.mark.memory_heavy
@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_gate1_a_spheroids_rolling_and_entries_against_the_mfs_at_every_gap_below_the_handover(tilt):
    found = spheroid_misses(tilt)
    for (angle, gap), (rate, entry) in sorted(found.items()):
        print(f"\ntilt {tilt}, turn {angle:.3f}, h/b {gap}: rate {rate:+.4f}, entry {entry:+.4f}")
    worst = (max(r for r, _ in found.values()), max(e for _, e in found.values()))
    print(f"\ntilt {tilt}: worst rate {worst[0]:.4f}, worst entry {worst[1]:.4f}")
    pinned = SPHEROID_PINNED.get(tilt)
    if pinned is None:  # the committed bands, until the measured worst is pinned
        assert 0.0 <= worst[0] <= 0.25 and 0.0 <= worst[1] <= 0.35, worst
    else:
        assert abs(worst[0] - pinned[0]) <= SPHEROID_PIN and abs(worst[1] - pinned[1]) <= SPHEROID_PIN, worst


# -- gate 3 ----------------------------------------------------------------------------------------------------------

FORM_GAPS = (1e-3, 1.5e-3, 2.2e-3, 3.3e-3, 5e-3)


def test_gate3_an_ellipsoids_normal_goes_as_one_over_the_gap_to_its_elliptic_limit():
    """An ellipsoid of semi-axes (1.5, 1, 1) mm, its lowest point the end of its z axis: the gap's curvatures there are
    1/1.5^2 and 1 per mm (a contact 1.5 times longer than wide); the normal entry times the gap tends to ``12 pi mu /
    (s (k1 + k2))``, the elliptic contact's, and goes as the gap to the minus one."""
    with planted() as plant:
        if plant == "sphere_law":
            from mime.nodes.hydro.near_contact import _elliptic

            held = _elliptic.singular
            _elliptic.singular = lambda k1, k2, b1, b2, b12, *rest: held(k1, k1, b1, b2, b12, *rest)
        try:
            node, state, leaves, step = stepper(above_plane(0.13, kind=Ellipsoid(semi_axes=mm((1.5, 1.0, 1.0))),
                                                            name="lubricated_ellipsoid").build())
            normal = np.asarray([float(step(jnp.asarray((0.0, 0.0, RADIUS + g * RADIUS, 1.0, 0.0, 0.0, 0.0)),
                                            leaves)["resistance"][2, 2]) for g in FORM_GAPS])
        finally:
            if plant == "sphere_law":
                _elliptic.singular = held
    k1, k2 = 1.0 / RADIUS, (1.0 / 1.5 ** 2) / RADIUS
    limit = float(normal[0] * FORM_GAPS[0] * RADIUS / (12.0 * math.pi * VISCOSITY / ((k1 + k2) * math.sqrt(k1 * k2))))
    exponents = np.diff(np.log(normal)) / np.diff(np.log(FORM_GAPS))
    print(f"\nexponents {exponents}, limit {limit:.5f}")
    assert np.all(np.abs(exponents + 1.0) <= 0.05), exponents
    assert abs(limit - 1.0) <= 0.03, limit


# -- gate 5 ----------------------------------------------------------------------------------------------------------

ACROSS = REFERENCES / "ellipsoid_across_tube_mrs.json"
#: Gate 5's coarse force spacing (m): the hand-over gap 0.066 mm, so that both ends' gaps (summing to 0.1 mm) lie under
#: it; the reference's spacing 0.05 mm at blob 0.1 (resolving gaps from 0.0375 mm), within a runner's memory.
COARSE = 0.088e-3
#: Gate 5's line per entry against the finer MRS, and the line past which a control counts as failed.
TWO_CONTACT_LINE = 0.10
CONTROL_LINE = 0.20


def across_rows() -> dict:
    return json.loads(ACROSS.read_text())


def two_contacts(near_contact: bool):
    return stepper(across_tube(COARSE, near_contact).build())


def entry_misses(found: np.ndarray, reference: np.ndarray) -> float:
    """The worst entry on its own diagonal scale (the couplings on ``sqrt(R_ii R_jj)``)."""
    scale = np.sqrt(np.abs(np.outer(np.diag(reference), np.diag(reference))))
    return float(np.max(np.abs(found - reference) / scale))


def definite(resistance: np.ndarray) -> bool:
    held = 0.5 * (resistance + resistance.T)
    d = np.sqrt(np.abs(np.diag(held)))
    return bool(np.all(np.isfinite(held)) and np.min(np.linalg.eigvalsh(held / np.outer(d, d))) > 0.0)


def composed_controls(pair: str, lifted, coarse) -> dict[str, np.ndarray | None]:
    """The lift and the three controls at ``pair``: each a 6 x 6 in SI, or ``None`` where it cannot be formed."""
    node, state, leaves, step = lifted
    _, cstate, cleaves, cstep = coarse
    h1, h2 = (float(v) * 1e-3 for v in pair.split(","))
    pose = across(h1, h2)
    lift = np.asarray(step(pose, leaves)["resistance"])
    true = np.asarray(cstep(pose, cleaves)["resistance"])
    # the contacts and their own openings, as the node finds them
    wall = node._wall()
    held = node.params_pytree()
    constants = {name: held.get(f"wall_{name}", node.constants()[f"wall_{name}"]) for name in wall.parameters}
    points = node._sampling.samples(node._kind, node._kind_parameters(held)).points
    found = handed_over(node._search, node._kind, node._kind_parameters(held), points, as_run_array(pose), wall,
                        constants, held, held["viscosity"], node.near_contact_form)
    handover = HANDOVER_SPACINGS * float(held["handover_spacing"])
    reach = float(np.max(np.linalg.norm(np.asarray(points), axis=-1)))
    contacts = found.contacts
    shares = np.asarray(contacts.merged)
    distinct = []
    for k in range(shares.shape[0]):
        point = np.asarray(contacts.contacts.point[k])
        if float(contacts.contacts.gap[k]) < 2.0 * handover and \
                all(np.linalg.norm(point - np.asarray(contacts.contacts.point[j])) > 0.05 * reach for j in distinct):
            distinct.append(k)
    assert len(distinct) == 2, distinct

    def own(k: int, delta: float) -> np.ndarray:
        contact = jax.tree_util.tree_map(lambda v: v[k], contacts.contacts)
        return np.asarray(increment(node.near_contact_form, contact, jnp.asarray(delta), found.cutoff,
                                    held["viscosity"], pose[:3], jnp.asarray(reach)).resistance)

    def tube_at(radius_add: float, shift: np.ndarray) -> dict:
        return {**cleaves, "wall_radius": cleaves["wall_radius"] + radius_add,
                "wall_centre": jnp.asarray(cleaves["wall_centre"]) + jnp.asarray(shift)}

    def lifted_at(radius_add: float, shift: np.ndarray) -> np.ndarray:
        moved = {**leaves, "wall_radius": leaves["wall_radius"] + radius_add,
                 "wall_centre": jnp.asarray(leaves["wall_centre"]) + jnp.asarray(shift)}
        return np.asarray(step(pose, moved)["resistance"])

    one_at_a_time, subtraction = true.copy(), true.copy()
    for k in distinct:
        gap = float(contacts.contacts.gap[k])
        delta = float(opening(jnp.asarray(gap), jnp.asarray(handover)))
        outward = -np.asarray(contacts.contacts.frame[k][2])  # the wall's move away from the medium at this contact
        # this contact's wall alone moved out by delta: the tube dilated by delta / 2, its centre shifted delta / 2
        alone = np.asarray(cstep(pose, tube_at(delta / 2.0, outward * delta / 2.0))["resistance"])
        one_at_a_time += alone - true + own(k, delta)
        # the other wall moved far (0.3 mm) away: the one-contact lift less the one-contact coarse MRS
        far = 0.3e-3
        alone_far = np.asarray(cstep(pose, tube_at(far / 2.0, -outward * far / 2.0))["resistance"])
        subtraction += lifted_at(far / 2.0, -outward * far / 2.0) - alone_far
    gaps = sorted(float(contacts.contacts.gap[k]) for k in distinct)
    move = max(0.0, handover - gaps[0])
    per_body = None
    if gaps[1] - move > 0.0:  # the body moved off the nearer wall towards the other (else it cannot be formed)
        nearer = min(distinct, key=lambda k: float(contacts.contacts.gap[k]))
        along = np.asarray(contacts.contacts.frame[nearer][2])
        moved = pose.at[:3].add(jnp.asarray(along * move))
        per_body = np.asarray(cstep(moved, cleaves)["resistance"])
        for k in distinct:  # the other contact's increment taken back over its closed span
            per_body += own(k, move if k == nearer else -move)
    return {"lift": lift, "one_at_a_time": one_at_a_time, "subtraction": subtraction, "per_body_lift": per_body}


@pytest.mark.memory_heavy
def test_gate5_two_contacts_closed_the_lift_holds_and_each_control_fails():
    rows = across_rows()
    plant = os.environ.get("W2A_PLANT") or None
    lifted, coarse = two_contacts(True), two_contacts(False)
    for pair, row in rows["rows"].items():
        reference = np.asarray(row["resistance"])
        forms = composed_controls(pair, lifted, coarse)
        judged_form = forms[plant] if plant in forms else forms["lift"]
        misses = {name: (None if value is None else entry_misses(value, reference), None if value is None
                         else definite(value)) for name, value in forms.items()}
        print(f"\ngaps {pair} mm: " + "; ".join(f"{k} {v[0] if v[0] is None else round(v[0], 4)} "
                                                f"{'definite' if v[1] else 'not definite'}" for k, v in misses.items()))
        assert judged_form is not None, (pair, plant, "cannot be formed")
        assert definite(judged_form), (pair, plant)
        assert entry_misses(judged_form, reference) <= TWO_CONTACT_LINE, (pair, plant, misses)


# -- gate 8 on two contacts ------------------------------------------------------------------------------------------

@pytest.mark.memory_heavy
def test_gate8_two_contacts_forward_and_reverse_equal_the_difference():
    with planted():
        built = across_tube(COARSE, True).build()
        found = compared(built, lambda g: across(g * RADIUS, 0.1e-3 - g * RADIUS), (0.07, 0.06, 0.04),
                         ("wall_radius",))
    for gap, name, reverse, forward, difference, scale in found:
        print(f"\nh1/a {gap} {name}: reverse {reverse:.10g} forward {forward:.10g} difference {difference:.10g}")
        assert abs(reverse - difference) <= 1e-5 * scale and abs(forward - difference) <= 1e-5 * scale, (gap, name)
