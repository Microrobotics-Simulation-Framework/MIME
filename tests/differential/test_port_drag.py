"""The drag's node against the 0.2 port's (the weakest oracle). The port's ``MLPResistanceNode`` and
``LubricationCorrectionNode`` ran once, in the port's own environment, on fixed random poses with script A's
parameters (``generate_port_drag_references.py``; ``references/port_drag*`` holds the outputs and their provenance:
MIME-port041 at 6485516, MADDENING 0.4.1, float64, CPU); MIME 0.3's ``SurrogateDragNode`` runs here on the same poses,
with the tube along +z through the origin, as the port assumed it. The port's ``mime`` is never imported beside this
one.

**Declared before the first run.** Both evaluate the same formulas, in other orders (a frame built from vectors where
the port took the cosine and sine of an ``arctan2``; the surrogate and the lubrication in one node), so in float64:

- every resistance agrees within 1e-12 of the largest magnitude, over all the poses, of the 3 x 3 block it sits in;
- the drag, ``-R [v; omega]``, within 1e-12 of its field's largest magnitude, and the clamp's flag exactly.

The only differences allowed are the declared fixes. The axis: the port's tube is along +z, which the comparison
takes, so there should be none from it. On the axis (an offset below 4096 units in the last place of the tube's
radius): the port took the world's x axis for the offset's direction (``arctan2(0, 0) = 0``) and for the lubrication's
normal, where 0.3 takes the body's own x axis across the tube. For a body whose orientation is the identity the two
are one; for a drawn orientation, 0.3's resistance is the port's turned about the axis by the angle from the world's x
axis to the body's, and nothing else.

**Found on the first run, not declared before it**, and traced to the port: on the axis the port's frame is not the
world's x axis but ``arctan2`` of the position's signed zeros, ``0`` for ``(+0, +0)`` and ``-pi`` for ``(-0, -0)``
(the generator's ``0 * cos(angle)`` gives either), while its lubrication's normal stays ``+x``: on the axis the port's
surrogate and lubrication frames can disagree by half a turn. With that frame taken for the port's surrogate and
``+x`` for its lubrication, 0.3's resistance on the axis is the port's turned about the axis to 0.3's frame, within
the same tolerance. A facet of the on-axis defect 0.3 fixes, not a separate anomaly.

What the two share: the formulas, the weights (the same file, by sha256), A's parameters and SI units. The controls:
the wavenumber off by one part in a million, and the tube's radius off by one part in a million, are each caught.
"""

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mime.data import asset
from mime.nodes.hydro import SurrogateDragNode

REFERENCES = Path(__file__).parent / "references"
PORT = np.load(REFERENCES / "port_drag.npz")
PROVENANCE = json.loads((REFERENCES / "port_drag_provenance.json").read_text())
KINDS = list(PROVENANCE["kinds"])
RTOL = 1e-12
NU, LENGTH, RADIUS, TUBE, MU, EPSILON = 2.33, 7.47e-3, 1.56e-3, 3.175e-3, 1e-3, 1.5e-4


def node(*, epsilon: float | None, nu: float = NU, tube: float = TUBE,
         reference_point: str = "end") -> SurrogateDragNode:
    """0.3's node as the port computes: the resistance about the end of the body's axis (the test-only switch), unless
    ``reference_point="centre"``, drag 0.2.0's."""
    return SurrogateDragNode("drag", 5e-4, weights="dejongh_mlp_v2", weights_sha256=asset("dejongh_mlp_v2").sha256,
                             viscosity=MU, helix_wavenumber=nu, length=LENGTH, radius=RADIUS,
                             vessel_centre=(0.0, 0.0, 0.0), vessel_axis=(0.0, 0.0, 1.0), vessel_radius=tube,
                             lubrication_epsilon=epsilon, reference_point=reference_point)


def outputs(n: SurrogateDragNode) -> dict[str, np.ndarray]:
    found = [n.update(n.initial_state(), {"pose": jnp.concatenate([jnp.asarray(p), jnp.asarray(q)])}, 5e-4)
             for p, q in zip(PORT["position"], PORT["orientation"], strict=True)]
    return {key: np.array([np.asarray(f[key]) for f in found]) for key in ("resistance", "clamp_fired")}


def drag_of(resistance: np.ndarray) -> np.ndarray:
    motion = np.concatenate([PORT["velocity"], PORT["omega"]], axis=1)
    return -np.einsum("nij,nj->ni", resistance, motion)


def moved_by_block(found: np.ndarray, port: np.ndarray) -> float:
    """The largest difference in each 3 x 3 block, relative to that block's largest magnitude over all the poses."""
    worst = 0.0
    for rows in (slice(0, 3), slice(3, 6)):
        for columns in (slice(0, 3), slice(3, 6)):
            scale = float(np.max(np.abs(port[:, rows, columns])))
            worst = max(worst, float(np.max(np.abs(found[:, rows, columns] - port[:, rows, columns]))) / scale)
    return worst


def moved(found: np.ndarray, port: np.ndarray) -> float:
    return float(np.max(np.abs(found - port))) / float(np.max(np.abs(port)))


def kind(*names: str) -> np.ndarray:
    return np.isin(PORT["kind"], [KINDS.index(name) for name in names])


AGREE = kind("inside", "between", "past_clamp", "near_axis")
ON_AXIS = kind("on_axis_identity", "on_axis_turned")


def test_the_references_are_the_ports_on_the_same_weights_with_as_parameters():
    assert PROVENANCE["weights"]["sha256"] == asset("dejongh_mlp_v2").sha256
    assert PROVENANCE["parameters"]["vessel_axis"] == 2 and PROVENANCE["x64"] is True
    assert int(AGREE.sum()) == 56 and int(ON_AXIS.sum()) == 4
    assert 0 < int(PORT["clamp_fired"].sum()) < len(PORT["clamp_fired"]), "some poses are clamped and some not"


@pytest.mark.parametrize("lubricated", [False, True], ids=["the surrogate", "with lubrication"])
def test_off_the_axis_the_resistance_and_drag_agree_with_the_ports(lubricated):
    found = outputs(node(epsilon=EPSILON if lubricated else None))
    port = PORT["lub_resistance"] if lubricated else PORT["resistance"]
    worst = moved_by_block(found["resistance"][AGREE], port[AGREE])
    print(f"\nworst blockwise difference {worst:.2e}")
    assert worst <= RTOL
    force, torque = (PORT["lub_force"], PORT["lub_torque"]) if lubricated else (PORT["drag_force"],
                                                                               PORT["drag_torque"])
    drag = drag_of(found["resistance"])
    assert moved(drag[AGREE, :3], force[AGREE]) <= RTOL and moved(drag[AGREE, 3:], torque[AGREE]) <= RTOL
    assert np.array_equal(found["clamp_fired"], PORT["clamp_fired"])


def about_the_axis(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    turn = np.zeros((6, 6))
    turn[:3, :3] = turn[3:, 3:] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return turn


def frame_angle(orientation: np.ndarray) -> float:
    """The angle about the axis (z) of 0.3's frame on the axis: the body's x axis across the tube, or its y axis
    where x lies within 60 degrees of the axis."""
    w, x, y, z = orientation
    across = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    if across @ across <= 0.25:
        across = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z)])
    return math.atan2(across[1], across[0])


def test_on_the_axis_the_resistance_is_the_ports_turned_to_the_bodys_frame():
    """The declared difference, with the one found on the first run: 0.3's frame on the axis is the body's own axis
    across the tube; the port's surrogate took ``arctan2`` of the position's signed zeros and its lubrication ``+x``."""
    found = outputs(node(epsilon=EPSILON))["resistance"]
    surrogate = outputs(node(epsilon=None))["resistance"]
    for index in np.flatnonzero(ON_AXIS):
        position, angle = PORT["position"][index], frame_angle(PORT["orientation"][index])
        port_angle = math.atan2(position[1], position[0])  # the signed zeros, as the port's jnp.arctan2 read them
        turn, from_port = about_the_axis(angle), about_the_axis(angle - port_angle)
        expected_surrogate = from_port @ PORT["resistance"][index] @ from_port.T
        lubrication = PORT["lub_resistance"][index] - PORT["resistance"][index]
        expected = expected_surrogate + turn @ lubrication @ turn.T
        assert moved_by_block(surrogate[index][None], expected_surrogate[None]) <= RTOL
        assert moved_by_block(found[index][None], expected[None]) <= RTOL
    assert {math.atan2(p[1], p[0]) for p in PORT["position"][ON_AXIS]} == {0.0, -math.pi}, "both signs of zero drawn"
    # Where both frames are the world's x axis (the identity at +0) the two agree outright; elsewhere they differ.
    for index in np.flatnonzero(ON_AXIS):
        same = PORT["kind"][index] == KINDS.index("on_axis_identity") and not np.signbit(PORT["position"][index][0])
        moved_from_port = moved_by_block(found[index][None], PORT["lub_resistance"][index][None])
        assert (moved_from_port <= RTOL) if same else (moved_from_port > 1e-4)


@pytest.mark.parametrize("mutation", ["the wavenumber off by one part in a million",
                                      "the tube's radius off by one part in a million"])
def test_a_planted_mutation_is_caught(mutation):
    planted = node(epsilon=EPSILON, nu=NU * (1 + 1e-6)) if "wavenumber" in mutation else node(
        epsilon=EPSILON, tube=TUBE * (1 + 1e-6))
    assert moved_by_block(outputs(planted)["resistance"][AGREE], PORT["lub_resistance"][AGREE]) > RTOL


def test_about_the_centre_the_resistance_is_the_ports_with_exactly_the_declared_shift():
    """drag 0.2.0 (MIME-ANO-112): off the axis, the port's surrogate resistance turned end for end where the body's x
    axis points against the tube's (``G``, the half turn about the offset's direction ``n``) and moved from the end to
    the centre, ``T^T (G R G^T) T`` with ``T = [[I, [d]x], [0, I]]`` and ``d = (L/2) s z``; the lubrication
    unchanged."""
    found = outputs(node(epsilon=EPSILON, reference_point="centre"))["resistance"]
    worst = 0.0
    for index in np.flatnonzero(AGREE):
        w, x, y, z = PORT["orientation"][index]
        side = 1.0 if 2 * (x * z - w * y) >= 0.0 else -1.0  # the body x axis's z component
        position = PORT["position"][index]
        n = np.array([position[0], position[1], 0.0]) / math.hypot(position[0], position[1])
        half_turn = 2 * np.outer(n, n) - np.eye(3)
        g = np.zeros((6, 6))
        g[:3, :3] = g[3:, 3:] = half_turn if side < 0 else np.eye(3)
        d = side * LENGTH / 2
        cross = np.array([[0.0, -d, 0.0], [d, 0.0, 0.0], [0.0, 0.0, 0.0]])
        move = np.eye(6)
        move[:3, 3:] = cross
        surrogate = PORT["resistance"][index]
        lubrication = PORT["lub_resistance"][index] - surrogate
        expected = move.T @ (g @ surrogate @ g.T) @ move + lubrication
        worst = max(worst, moved_by_block(found[index][None], expected[None]))
    print(f"\nworst blockwise difference from the declared shift {worst:.2e}")
    assert worst <= RTOL
    assert moved_by_block(found[AGREE], PORT["lub_resistance"][AGREE]) > 1e-2
