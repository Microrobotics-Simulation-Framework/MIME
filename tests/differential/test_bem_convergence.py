"""How the de Jongh helix's boundary-element resistance converges, by two discretisations of its lateral surface at
the same counts: the 0.2 line's rule (flat triangles between vertices of the surface, a point at each centroid, its
area the weight) and the kind's own samples through ``Drag.BEM``'s node. The measurement is recorded
(``references/bem_convergence.json``, written by ``measure_bem_convergence.py``, not collected) at five levels, the
0.2 rule's vertex grid 12 x 10 to 36 x 30, 216 to 2,088 samples, and the kind's lateral samples at the same counts
(twice the grid's turn by one less than its length); FL-9, water, the blob half the mean spacing, float64.

**The 0.2 rule here is independent of the kind's code**: its vertices come from de Jongh's Eq. 1 written again in
this file, and at the 0.2 mesh's own grid they are the frozen 0.2 vertices, and the rule on them gives the frozen 0.2
resistance (``references/port_bem.npz``): this rule is the 0.2 code's. The kind's path is the rung's: the kind's
samples on its lateral patch, placed by the part and by the body's pose (the screw moved and turned, its resistance
turned back into its frame), and solved by the node.

**What the record shows.** On the diagonal the two approach the converged value from opposite sides at every level,
the 0.2 rule's rising and the kind's falling, so each level brackets it, and the finest bracket holds it under that
monotone approach (an inference from five levels, not a proof). A fit ``R + c1 / sqrt(N) + c2 / N`` to each
sequence (five points, three parameters: an estimate, not a bound): in translation both fitted limits lie in the
finest bracket and agree within 1%, and the kind's finest is the nearer to either; in rotation the fits disagree by
more, and the 0.2 rule's limit about the axis lies outside the bracket, so which is nearer is not established there.
The axial coupling is not bracketed: both sequences grow in magnitude, the 0.2 rule's ahead at every level, so its
is the nearer there, and both are still moving at the finest level.

**The bound on the kind's path**, which the golden reproduction (``test_port_bem.py``, fed the 0.2 samples) does not
give: at the CI levels the node reproduces the record, lies above the 0.2 rule and falls with the count, and the
three-level fits of its translation and the 0.2 rule's agree within 1%: a sampling or placement fault of a few per
cent in the kind's path, which would read as a model difference, fails here. The control: the screw's samples read
2% too large fail that agreement. The five levels are reproduced in full by the slow test."""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from bem_cases import bem, drag_node, free_space
from script_a import fl9

from mime.build._maddening import node_state
from mime.core import quaternion
from mime.nodes.hydro.bem import BEMDragNode, blob_squared, resistance
from mime.registry import Samples
from mime.units import mm

HERE = Path(__file__).parent
RECORD = HERE / "references" / "bem_convergence.json"
#: The record as written, pinned: the CI levels are reproduced on every push and the rest by the slow test.
RECORD_SHA256 = "21c23ce309f9a0c6f2da177f478acc0a132765bfc22c178f28d46f1f4f708c04"
#: The vertex grids (round the turn, along the axis) of the five levels; the CI test takes the first three.
LEVELS = ((12, 10), (18, 15), (24, 20), (30, 25), (36, 30))
CI_LEVELS = 3
#: FL-9 in SI, as ``script_a.fl9()`` declares it, and the water's viscosity.
RADIUS, LENGTH, WAVENUMBER, MODULATION, STARTS, VISCOSITY = 1.56e-3, 7.47e-3, 2.33, 0.33, 2, 1e-3
BLOB = 0.5
#: Within which the record is reproduced (of each entry's scale), and the fitted limits of translation agree.
REPRODUCED, LIMITS_AGREE = 1e-9, 1e-2


def screw_vertices(n_theta: int, n_zeta: int, radius: float = RADIUS, length: float = LENGTH) -> np.ndarray:
    """De Jongh's Eq. 1 at a grid of ``n_theta`` angles round the turn and ``n_zeta`` stations along the axis, in the
    0.2 line's frame (the axis along z, from the end): ``r = zeta z + rho(theta) (cos a x + sin a y)``,
    ``a = nu zeta / R + theta``, ``rho = R (1 + eps sin(N theta))``."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, length, n_zeta)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")
    rho = radius * (1.0 + MODULATION * np.sin(STARTS * th))
    angle = WAVENUMBER * ze / radius + th
    return np.stack([rho * np.cos(angle), rho * np.sin(angle), ze], axis=-1)


def triangle_rule(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 0.2 rule: each quad of the grid split into two triangles (the turn closed), a point at each centroid and
    its area as its weight."""
    n_theta, n_zeta = vertices.shape[:2]
    triangles = np.asarray([tri for i in range(n_theta) for j in range(n_zeta - 1) for tri in (
        (vertices[i, j], vertices[(i + 1) % n_theta, j], vertices[(i + 1) % n_theta, j + 1]),
        (vertices[i, j], vertices[(i + 1) % n_theta, j + 1], vertices[i, j + 1]))])
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    return triangles.mean(axis=1), area


def the_02_rule(n_theta: int, n_zeta: int) -> np.ndarray:
    """The resistance of FL-9's lateral surface by the 0.2 rule, in the kind's frame (the axis along x, the origin at
    its centroid), about that origin."""
    centroids, area = triangle_rule(screw_vertices(n_theta, n_zeta))
    points = np.stack([centroids[:, 2] - LENGTH / 2, centroids[:, 0], centroids[:, 1]], axis=-1)
    return np.asarray(resistance(jnp.asarray(points), blob_squared(jnp.asarray(area), BLOB), VISCOSITY, jnp.zeros(3)))


def lateral_rung():
    """``Drag.BEM`` on FL-9's lateral patch in free space, built; the screw starts moved and turned."""
    return free_space("convergence", {"screw": fl9()}, bem(mm(1.0), patches=("lateral",))).build()


def through_the_node(built, n_theta: int, n_zeta: int, cls: type[BEMDragNode] = BEMDragNode) -> np.ndarray:
    """The kind's lateral samples at the 0.2 rule's count (``2 n_theta`` round the turn by ``n_zeta - 1`` along),
    through the rung's node at the screw's starting pose, its resistance turned back into the screw's frame."""
    node = drag_node(built, "screw")
    held = dict(node.constants())
    held.pop("counts")
    fresh = cls(node.name, node.delta_t, counts=[(2 * n_theta, n_zeta - 1)], **held)
    pose = np.asarray(node_state(built.graph, "screw")["pose"], dtype=np.float64)
    world = np.asarray(fresh.update(fresh.initial_state(), {"pose": jnp.asarray(pose)}, 1e-3)["resistance"])
    frame = np.kron(np.eye(2), np.asarray(quaternion.rotation_matrix(jnp.asarray(pose[3:]))))
    return frame.T @ world @ frame


def measured(levels=LEVELS, cls: type[BEMDragNode] = BEMDragNode, rule: bool = True) -> list[dict]:
    built = lateral_rung()
    found = []
    for n_theta, n_zeta in levels:
        level = {"grid": [n_theta, n_zeta], "samples": 2 * n_theta * (n_zeta - 1),
                 "kind": through_the_node(built, n_theta, n_zeta, cls).tolist()}
        if rule:
            level["rule_02"] = the_02_rule(n_theta, n_zeta).tolist()
        found.append(level)
    return found


def recorded() -> list[dict]:
    return json.loads(RECORD.read_text(encoding="utf-8"))["levels"]


def entries(levels: list[dict], name: str) -> np.ndarray:
    """Each level's diagonal (translation along and across the axis, then rotation) and its axial coupling."""
    return np.asarray([[*np.diag(np.asarray(level[name])), np.asarray(level[name])[0, 3]] for level in levels])


def fitted_limits(levels: list[dict], name: str) -> np.ndarray:
    """``R`` of the least-squares fit ``R + c1 / sqrt(N) + c2 / N`` to each entry over the levels."""
    count = np.asarray([level["samples"] for level in levels], dtype=float)
    design = np.stack([np.ones_like(count), count ** -0.5, 1.0 / count], axis=1)
    return np.linalg.lstsq(design, entries(levels, name), rcond=None)[0][0]


def test_the_02_rule_here_is_the_02_codes_on_its_own_mesh():
    """At the frozen helix's grid and scale (the radius 1, the viscosity 1), these vertices are the 0.2 code's and the
    rule on them gives its resistance, about the 0.2 frame's origin."""
    held = np.load(HERE / "references" / "port_bem.npz")
    length = LENGTH / RADIUS
    vertices = screw_vertices(12, 10, radius=1.0, length=length)
    assert np.abs(vertices - held["helix_coarse/vertices"]).max() <= 64 * np.finfo(float).eps * length
    centroids, area = triangle_rule(vertices)
    found = np.asarray(resistance(jnp.asarray(centroids), blob_squared(jnp.asarray(area), BLOB), 1.0, jnp.zeros(3)))
    expected = held["helix_coarse/R_library"]
    assert np.abs(found - expected).max() <= 1e-10 * np.abs(expected).max()


def test_the_recorded_convergence_brackets_the_diagonal_and_the_kind_is_the_nearer_in_translation():
    assert hashlib.sha256(RECORD.read_bytes()).hexdigest() == RECORD_SHA256
    levels = recorded()
    assert [level["grid"] for level in levels] == [list(grid) for grid in LEVELS]
    rule, kind = entries(levels, "rule_02"), entries(levels, "kind")
    limits = {"rule_02": fitted_limits(levels, "rule_02"), "kind": fitted_limits(levels, "kind")}
    print("\nlevel, samples: the 0.2 rule's and the kind's diagonal and axial coupling, against the kind's at the "
          "finest")
    for level, row, own in zip(levels, rule, kind, strict=True):
        print(f"  {level['grid']}, {level['samples']}: " + " ".join(f"{d:+.2%}" for d in row / kind[-1] - 1)
              + " | " + " ".join(f"{d:+.2%}" for d in own / kind[-1] - 1))
    for name, limit in limits.items():
        print(f"  {name}'s finest against its fitted limit: " + " ".join(
            f"{d:+.2%}" for d in entries(levels, name)[-1] / limit - 1))
    print("  the kind's fitted limits against the 0.2 rule's: " + " ".join(
        f"{d:+.2%}" for d in limits["kind"] / limits["rule_02"] - 1))
    diagonal = slice(0, 6)
    assert np.all(rule[:, diagonal] < kind[:, diagonal])
    assert np.all(np.diff(rule[:, diagonal], axis=0) > 0) and np.all(np.diff(kind[:, diagonal], axis=0) < 0)
    translation = slice(0, 3)
    for limit in limits.values():
        assert np.all((rule[-1, translation] < limit[translation]) & (limit[translation] < kind[-1, translation]))
        assert np.all(np.abs(kind[-1, translation] / limit[translation] - 1)
                      < np.abs(rule[-1, translation] / limit[translation] - 1))
    assert np.all(np.abs(limits["kind"][translation] / limits["rule_02"][translation] - 1) < LIMITS_AGREE)
    coupling = 6
    assert np.all(np.diff(np.abs(rule[:, coupling])) > 0) and np.all(np.diff(np.abs(kind[:, coupling])) > 0)
    assert np.all(np.abs(rule[:, coupling]) > np.abs(kind[:, coupling]))


def bound_misses(cls: type[BEMDragNode] = BEMDragNode) -> tuple[float, np.ndarray]:
    """The kind's path at the CI levels: how far it is from the record (of each entry's scale), and how far its
    three-level fitted limits of translation are from the 0.2 rule's; asserting it lies in each level's bracket."""
    levels = recorded()[:CI_LEVELS]
    found = measured(LEVELS[:CI_LEVELS], cls, rule=False)
    reproduced = 0.0
    for level, again in zip(levels, found, strict=True):
        expected, held = np.asarray(level["kind"]), np.asarray(again["kind"])
        scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        reproduced = max(reproduced, float(np.abs((held - expected) / scale).max()))
    kind = entries(found, "kind")
    assert np.all(entries(levels, "rule_02")[:, :6] < kind[:, :6]) and np.all(np.diff(kind[:, :6], axis=0) < 0)
    return reproduced, fitted_limits(found, "kind")[:3] / fitted_limits(levels, "rule_02")[:3] - 1


def test_the_kinds_path_through_the_node_is_bounded_by_the_02_rules_convergence():
    reproduced, limits = bound_misses()
    print(f"\nthe node against the record: {reproduced:.1e}; its three-level limits of translation against the 0.2 "
          f"rule's: " + " ".join(f"{d:+.2%}" for d in limits))
    assert reproduced <= REPRODUCED
    assert np.all(np.abs(limits) < LIMITS_AGREE)


class _WiderScrewNode(BEMDragNode):
    """The screw's samples read 2% too large: a sampling fault of the size the model differences are."""

    def samples(self, p):
        held = super().samples(p)
        return Samples(1.02 * held.points, held.normals, 1.02 ** 2 * held.weights)


def test_control_a_screw_sampled_two_percent_too_large_fails_the_bound():
    reproduced, limits = bound_misses(_WiderScrewNode)
    print(f"\nsampled 2% too large: the node against the record {reproduced:.1e}; its limits of translation against "
          f"the 0.2 rule's: " + " ".join(f"{d:+.2%}" for d in limits))
    assert np.all(np.abs(limits) > LIMITS_AGREE)


@pytest.mark.slow
def test_every_recorded_level_is_reproduced():
    for level, again in zip(recorded(), measured(), strict=True):
        for name in ("rule_02", "kind"):
            expected, held = np.asarray(level[name]), np.asarray(again[name])
            scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
            assert np.abs((held - expected) / scale).max() <= REPRODUCED, (level["grid"], name)
