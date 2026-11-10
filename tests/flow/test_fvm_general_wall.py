"""The FVM on a tube tree's wall (F2a's PR 2b): the band's contract with the union, and FA3, FA5 to FA8, each through
the node's own step on a vessel periodic along z, turned about z at random (the loop's daughters then off every grid
plane) and moved by a sub-cell offset (``wall_cases``):

- **the contract**: the node's wall reads the union's exact distance on the faces it holds and the plain minimum only
  where every field is flat, its counts zero; control, the plain minimum everywhere (``MIME_PLANT_WALL=minimum``);
- **FA3**, a biconical taper: the wall's shift inferred from the flux against lubrication's, ``ln(Q / Q_lub) /
  (h d ln Q / d delta)``, within ``SHIFT_LINE`` of the planar calibration; control, every radius a quarter cell out
  (``MIME_PLANT_WALL=wall``);
- **FA5**, a 75 % stenosis: the same shift against the Richardson reference's flux (32 and 64 cells a diameter,
  stored), with the same control;
- **FA6**, the split-and-merge loop's split at 30 degrees (daughters 0.4 and 0.3 D, filleted): within ``SPLIT_LINE``
  of the Richardson reference; control, the first daughter a quarter cell out (``MIME_PLANT_WALL=daughter``);
- **FA7**, the symmetric loop (both 0.4 D): its split within ``SYMMETRY_LINE`` of a half, the same control;
- **FA8**, the derivative of the first daughter's flux in its radius: reverse mode through the steady step against a
  central difference in one discretisation (the same cells, faces held and plan), within ``DERIVATIVE_LINE``; control,
  the radii held out of the derivative (``MIME_PLANT_WALL=stopped``).

The predictions are ``predictions_general_wall.json``'s, committed before these ran."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import in_dtype, run, step_of
from test_fvm_tube import planar_calibration
from wall_cases import (
    Tree,
    box_flux,
    draw,
    loop,
    lubrication_flux,
    lubrication_sensitivity,
    mid_plane_split,
    plant,
    references,
    seeds,
    stenosis,
    taper,
    tree_node,
)

from mime.nodes.flow.fvm import MACGrid

#: The wall's shift against the planar calibration, in cells (S1's line, as the tube's).
SHIFT_LINE = 0.15
#: FA6's line on the loop's split against its reference.
SPLIT_LINE = 0.01
#: FA7's line on the symmetric split's distance from a half.
SYMMETRY_LINE = 0.005
#: FA8's line, the reverse derivative against the central difference, relative.
DERIVATIVE_LINE = 1e-4
#: The contract's line, the node's distance against the union's on the held faces, in cells.
CONTRACT_LINE = 1e-9
#: The first daughter's segments in ``wall_cases.loop``.
FIRST_DAUGHTER = (2, 3, 4)
QUARTER = 0.25


def _planted(tree: Tree, dh: int, segments: tuple[int, ...], name: str) -> Tree | None:
    return tree.grown(segments, QUARTER / dh) if plant(name) else None


# -- the contract -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", seeds(601))
def test_the_nodes_wall_is_the_unions_distance_where_the_fields_read_it(seed):
    """The filleted loop at 8 cells a diameter: on each face set, the node's distance equals the union's (all its
    candidates at every held face) to rounding, its counts are zero, and every face it does not hold lies where the
    fields are flat (deeper than the band's fluid half, or past the fillet's radius in the solid)."""
    angle, offset = draw(seed)
    with in_dtype("float64"):
        placed = tree_node(loop(), 8, angle=angle, offset=offset)
        node = placed.node
        grid = MACGrid(placed.layout.cells, placed.layout.spacing)
        held = node._step_constants(None)  # noqa: SLF001 - the step's own constants
        if plant("minimum"):
            solid = node.solids(held, grid)[0]
            phis = tuple(solid.union.minimum(solid.parameters, grid.face_points(k, "float64").reshape(-1, 3))
                         .reshape(grid.cells) for k in range(3))
            counts = jnp.zeros(3)
        else:
            fields = node.fields(held, grid)
            phis, counts = fields.signed_distance, fields.counts
        tree = node.tree()
        near, _, reach = node.held_faces(grid)
        h = placed.layout.spacing
        worst = 0.0
        for k in range(3):
            points = np.asarray(grid.face_points(k, "float64")).reshape(-1, 3)
            exact = np.asarray(tree.lumen().signed_distance(tree.parameters(), points[near[k]]).distance)
            phi = np.asarray(phis[k]).reshape(-1)
            worst = max(worst, float(np.max(np.abs(phi[near[k]] - exact))) / h)
            outside = np.ones(len(phi), dtype=bool)
            outside[near[k]] = False
            assert not np.any(outside & (phi > -reach[0]) & (phi <= reach[1]))
    print(f"\nseed {seed}: the node against the union {worst:.2e} cells, counts {np.asarray(counts)}")
    assert np.all(np.asarray(counts) == 0.0)
    assert worst <= CONTRACT_LINE


# -- FA3 and FA5 ----------------------------------------------------------------------------------------------------
def _inferred_shift(tree: Tree, flux: float, reference: float, h: float) -> float:
    return math.log(flux / reference) / (h * lubrication_sensitivity(tree)[1])


@pytest.mark.parametrize("seed", seeds(701, 702))
def test_fa3_the_taper_holds_the_planar_calibration_against_lubrication(seed):
    tree, dh = taper(), 16
    offset = draw(seed)[1]
    with in_dtype("float64"):
        placed = tree_node(tree, dh, offset=offset, tolerance=1e-10,
                           solver=_planted(tree, dh, tuple(range(len(tree.segments))), "wall"))
        state = run(placed.node, 1)
    shift = _inferred_shift(tree, box_flux(state), lubrication_flux(tree), placed.layout.spacing)
    planar = planar_calibration(16, "float64")
    print(f"\nseed {seed}: taper's shift {shift:+.4f} cells against lubrication, planar {planar:+.4f}; "
          f"counts {[float(state[k]) for k in ('wall_unrefined', 'wall_unconverged', 'wall_unplanned')]}")
    assert float(state["wall_inexact"]) == 0.0
    assert abs(shift - planar) < SHIFT_LINE


@pytest.mark.parametrize("seed", seeds(801, 802))
def test_fa5_the_stenosis_holds_the_planar_calibration_against_its_reference(seed):
    reference = references().get("stenosis")
    if reference is None:
        pytest.fail("no stored reference for the stenosis (generate_wall_references.py)")
    tree, dh = stenosis(), 16
    with in_dtype("float64"):
        placed = tree_node(tree, dh, offset=draw(seed)[1], tolerance=1e-10,
                           solver=_planted(tree, dh, tuple(range(len(tree.segments))), "wall"))
        state = run(placed.node, 1)
    shift = _inferred_shift(tree, box_flux(state), reference["flux"], placed.layout.spacing)
    planar = planar_calibration(16, "float64")
    print(f"\nseed {seed}: stenosis's shift {shift:+.4f} cells against its reference, planar {planar:+.4f}")
    assert float(state["wall_inexact"]) == 0.0
    assert abs(shift - planar) < SHIFT_LINE


# -- FA6 and FA7 ----------------------------------------------------------------------------------------------------
def _split(tree: Tree, dh: int, seed: int) -> tuple[float, dict]:
    angle, offset = draw(seed)
    with in_dtype("float64"):
        placed = tree_node(tree, dh, angle=angle, offset=offset, tolerance=1e-10,
                           solver=_planted(tree, dh, FIRST_DAUGHTER, "daughter"))
        state = run(placed.node, 1)
    first, second = mid_plane_split(placed, state)
    return first / (first + second), state


@pytest.mark.parametrize("seed", seeds(901, 902))
def test_fa6_the_loops_split_at_thirty_degrees_holds_its_reference(seed):
    reference = references().get("loop")
    if reference is None:
        pytest.fail("no stored reference for the loop's split (generate_wall_references.py)")
    split, state = _split(loop(), 16, seed)
    print(f"\nseed {seed}: split {split:.5f} against the reference {reference['split']:.5f}")
    assert float(state["wall_inexact"]) == 0.0
    assert abs(split - reference["split"]) <= SPLIT_LINE


@pytest.mark.parametrize("seed", seeds(1001, 1002))
def test_fa7_the_symmetric_split_is_a_half(seed):
    split, state = _split(loop((0.4, 0.4)), 16, seed)
    print(f"\nseed {seed}: symmetric split {split:.5f}")
    assert float(state["wall_inexact"]) == 0.0
    assert abs(split - 0.5) <= SYMMETRY_LINE


# -- FA8 ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", seeds(1101))
def test_fa8_the_first_daughters_flux_in_its_radius_is_the_central_difference(seed):
    """At 8 cells a diameter, float64: ``d Q_1 / d R_1`` along the first daughter's radii (both ends of its three
    segments), reverse mode through one steady step against ``(Q(R + e) - Q(R - e)) / 2e`` at ``e`` 5e-5 of the radius,
    both sides on the node as built (its cells, held faces and plan), their counts zero."""
    angle, offset = draw(seed)
    with in_dtype("float64"):
        placed = tree_node(loop(), 8, angle=angle, offset=offset, tolerance=1e-12)
        node = placed.node
        held = node._step_constants(None)  # noqa: SLF001 - the step's own constants
        direction = np.zeros_like(np.asarray(held["tree_radii"]))
        direction[list(FIRST_DAUGHTER)] = 1.0
        layout = placed.layout
        h = layout.spacing
        grid = MACGrid(layout.cells, h)
        k = int(round((placed.tree.middle - layout.origin[2]) / h))
        points = np.asarray(grid.face_points(2, "float64"))[:, :, k] + np.asarray(layout.origin)
        side = points[..., 0] * math.cos(angle) + points[..., 1] * math.sin(angle) > 0.0
        stopped = plant("stopped") is not None

        def flux(radii):
            given = {**held, "tree_radii": jax.lax.stop_gradient(radii) if stopped else radii}
            ended = step_of(node, given)(node.initial_state())
            return jnp.sum(jnp.where(side, ended["velocity_z"][:, :, k], 0.0)) * h * h, ended

        (value, ended), gradient = jax.jit(jax.value_and_grad(flux, has_aux=True))(jnp.asarray(held["tree_radii"]))
        reverse = float(jnp.sum(gradient * direction))
        step = 5e-5 * 0.4
        up = float(jax.jit(flux)(jnp.asarray(held["tree_radii"]) + step * direction)[0])
        down = float(jax.jit(flux)(jnp.asarray(held["tree_radii"]) - step * direction)[0])
    central = (up - down) / (2.0 * step)
    print(f"\nseed {seed}: dQ1/dR1 reverse {reverse:.8e}, central {central:.8e}, relative "
          f"{abs(reverse - central) / abs(central):.2e}; flux {float(value):.6e}")
    assert float(ended["wall_inexact"]) == 0.0
    assert abs(reverse - central) <= DERIVATIVE_LINE * abs(central)
