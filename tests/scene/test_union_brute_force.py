"""Gate 1 of the seam-aware union (``mime.scene.ExactUnion``): exact against a brute force, tilted and offset; where
two walls' normals are opposed (the hairpin aligned with the lattice); a corner of the filleted loop found from just off
its plane; float32 against float64 at the world's origin and 0.1 to 0.3 m from it. The draws, plants and lines are
``test_exact_union.py``'s (the gates' module docstring); this file holds gate 1 apart from it so that CI's shards, which
deal unlisted files by their path, carry the union's gates on three shards rather than one."""

import numpy as np
import pytest
from test_exact_union import (
    BRUTE_SPACING,
    CONTRACT_LINE,
    DTYPE_LINE,
    composed,
    drawn,
    drawn_y,
    planted,
)
from union_oracles import D, REACH, brute_distance, filleted_distance, loop_tree, plain_minimum, rotation, seed_list


# -- gate 1 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("kind", ["spike", "asymmetric", "loop", "hairpin"])
@pytest.mark.parametrize("seed", seed_list((11,)))
def test_the_union_is_exact_against_a_brute_force(kind, seed, dtype, monkeypatch):
    planted(monkeypatch)
    tree, points, h, middle, reach = drawn(kind, seed, 8)
    plain = plain_minimum(tree, points)
    band = np.abs(plain) < 3.0 * h
    points, plain = points[band], plain[band]
    exact = brute_distance(tree, points, BRUTE_SPACING * h, middle, reach + 0.5 * D)
    judged = (np.abs(plain) < 2.0 * h) | (np.abs(exact) < 2.0 * h)
    distance, found = composed(tree, points, dtype)
    error = np.abs(distance - exact)[judged] / h
    fluid = exact[judged] < 0.0
    print(f"\n{kind} seed {seed} {dtype}: {int(judged.sum())} points, {int(np.sum(found.failing))} failing the check; "
          f"largest error {error[fluid].max():.4f} cells in the fluid, {error[~fluid].max():.4f} in the solid; "
          f"the plain minimum {np.abs(plain - exact)[judged][fluid].max() / h:.3f}")
    assert int(np.sum(np.asarray(found.failing)[judged])) > 0, "no point failed the check: the draw cannot discriminate"
    assert int(found.unrefined) == 0 and int(found.unconverged) == 0
    assert error.max() <= CONTRACT_LINE, f"largest error {error.max():.4f} cells over the band's line {CONTRACT_LINE}"


#: A point of the split-and-merge loop's plane, in its own frame (D), whose nearest point of the filleted lumen's
#: boundary is a corner of its second junction (two, one each side of the plane): the first round's defect, which gate
#: 3's draw 31 crossed 1.6e-5 D from the plane (and its draws 3016 and 3023 near it, under the fix round's first
#: seam starts).
MERGE_CORNER = np.asarray([-0.178, 0.0, 2.207])


@pytest.mark.parametrize("seed", seed_list((15,)))
def test_a_corner_off_the_loops_plane_is_found_from_it(seed, monkeypatch):
    """Points 1e-6 D either side of the filleted loop's plane about ``MERGE_CORNER`` (a patch of 0.04 D), the loop
    turned at random and moved by a sub-cell offset: against the filleted brute force (``filleted_distance``).
    Gauss-Newton from the point and from its walls' nearest points stays near the plane, where neither corner is (it
    misses from 1e-9 to 3e-5 D off the plane at the patch's centre); a corner also starts from four points a size away
    across its first wall's normal, off the plane. **Control** (``MIME_PLANT_UNION=corner-starts``): those starts left
    out."""
    planted(monkeypatch)
    h = D / 8.0
    rng = np.random.default_rng(seed + 7919)
    turn, offset = rotation(seed), rng.uniform(0.0, h, 3)
    tree = loop_tree(0.1).turned(turn, offset)
    across = np.linspace(-0.02, 0.02, 5) * D
    local = MERGE_CORNER * D + np.stack(np.meshgrid(across, [-1e-6 * D, 1e-6 * D], across, indexing="ij"),
                                        -1).reshape(-1, 3)
    points = local @ turn.T + offset
    exact = filleted_distance(tree, points, BRUTE_SPACING * h, points.mean(axis=0), 0.8 * D)
    distance, found = composed(tree, points, "float64")
    error = np.abs(distance - exact) / h
    print(f"\nseed {seed}: {int(np.sum(found.failing))} of {len(points)} failing, largest error {np.nanmax(error):.4f} "
          f"cells, depth {-np.nanmax(exact) / D:.3f} to {-np.nanmin(exact) / D:.3f} D")
    assert np.all(np.isfinite(exact)) and np.all(np.asarray(found.failing))
    assert int(found.unrefined) == 0 and int(found.unconverged) == 0
    assert np.max(error) <= CONTRACT_LINE


@pytest.mark.parametrize("seed", seed_list((13,)))
def test_the_union_is_exact_at_sixteen_cells_a_diameter(seed):
    tree, points, h = drawn_y("asymmetric", seed, 16)
    plain = plain_minimum(tree, points)
    band = np.abs(plain) < 3.0 * h
    points, plain = points[band], plain[band]
    exact = brute_distance(tree, points, BRUTE_SPACING * h, np.asarray(tree.points[1]), (REACH + 0.25) * D)
    judged = (np.abs(plain) < 2.0 * h) | (np.abs(exact) < 2.0 * h)
    distance, found = composed(tree, points, "float64")
    error = np.abs(distance - exact)[judged] / h
    assert int(found.unrefined) == 0 and int(found.unconverged) == 0
    assert error.max() <= CONTRACT_LINE


@pytest.mark.parametrize("place", ["origin", "far"])
@pytest.mark.parametrize("seed", seed_list((14,)))
def test_float32_agrees_with_float64(seed, place, monkeypatch):
    """float32 against float64 on the same draw, at the world's origin and 0.1 to 0.3 m from it along each axis (a
    scanner's coordinates). **Control** (``MIME_PLANT_UNION=world-allowance``): the allowances scaled with the
    distance from the world's origin, as before the union had a frame of its own."""
    planted(monkeypatch)
    tree, points, h, _, _ = drawn("asymmetric", seed, 8, far=place == "far")
    points = points[np.abs(plain_minimum(tree, points)) < 2.0 * h]
    wide = composed(tree, points, "float64")[0]
    narrow, found = composed(tree, points, "float32")
    apart = np.max(np.abs(narrow - wide)) / h
    print(f"\n{place} seed {seed}: float32 against float64 {apart:.2e} cells, {int(np.sum(found.failing))} failing")
    assert int(found.unrefined) == 0 and int(found.unconverged) == 0
    assert apart <= DTYPE_LINE
