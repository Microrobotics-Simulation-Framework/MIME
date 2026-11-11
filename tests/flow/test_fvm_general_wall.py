"""The FVM on a tube tree's wall (F2a's PR 2b): the band's contract with the union, and FA3, FA5 to FA8, each through
the node's own step on a vessel periodic along z, turned about z at random (the loop's daughters then off every grid
plane) and moved by a sub-cell offset (``wall_cases``):

- **the contract**, judged without the node's own selection: on every face where a field is not flat, the node's
  fields are those of the union's exact distance (all its candidates at every face), its counts zero, the wall its
  chunk makes once the step's own; controls, the plain minimum everywhere (``MIME_PLANT_WALL=minimum``) and the
  union's candidates chosen only above half the band's fluid half (``MIME_PLANT_WALL=floor``);
- **FA3**, a biconical taper: the wall's shift inferred from the flux against lubrication's, ``ln(Q / Q_lub) /
  (h d ln Q / d delta)``, within ``SHIFT_LINE`` of the planar calibration; control, every radius a quarter cell out
  (``MIME_PLANT_WALL=wall``);
- **FA5**, a 75 % stenosis: the same shift against the Richardson reference's flux (32 and 64 cells a diameter,
  stored), with the same control;
- **FA6**, the split-and-merge loop's split at 30 degrees (daughters 0.4 and 0.3 D, filleted): within ``SPLIT_LINE``
  of the Richardson reference; control, the first daughter a quarter cell out (``MIME_PLANT_WALL=daughter``);
- **FA7**, the symmetric loop (both 0.4 D): its split within ``SYMMETRY_LINE`` of a half, the same control;
- **float32**: the loop's split at 8 cells a diameter in float32 within ``FLOAT32_LINE`` of float64's, no flag; the same
  control in the float32 run alone;
- **FA8**, the derivative of the first daughter's flux in its radius: reverse mode through the steady step against a
  central difference in one discretisation (the same cells and plan size), within ``DERIVATIVE_LINE``; control,
  the radii held out of the derivative (``MIME_PLANT_WALL=stopped``).

The predictions are ``predictions_general_wall.json``'s, committed before these ran."""

from __future__ import annotations

import dataclasses
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

from mime.core._node import RUN_CONSTANTS_GIVEN
from mime.nodes.flow.fvm import MACGrid
from mime.nodes.flow.fvm._band import INTERIOR_RAMP_END, band_named, interior_depth, interior_ramp, ramp, wall_fields

#: The wall's shift against the planar calibration, in cells (S1's line, as the tube's).
SHIFT_LINE = 0.15
#: FA6's line on the loop's split against its reference.
SPLIT_LINE = 0.01
#: FA7's line on the symmetric split's distance from a half.
SYMMETRY_LINE = 0.005
#: The float32 arm's line on the loop's split against float64's: float32's eps (1.2e-7) times a solve's growth of about a
#: hundred (the review measured 3.9e-7).
FLOAT32_LINE = 1e-5
#: FA8's line, the reverse derivative against the central difference, relative.
DERIVATIVE_LINE = 1e-4
#: The contract's lines: the node's distance against the exact union's where a field is not flat, in cells; its
#: indicator and resistivity against those of the exact distance, each in its own scale (the indicator's one, the
#: resistivity's peak). Both far above the rounding of one distance read two ways (1e-15 measured) and far below a
#: wall's shift a flow could see.
CONTRACT_LINE = 1e-9
FIELD_LINE = 1e-6
#: The first daughter's segments in ``wall_cases.loop``.
FIRST_DAUGHTER = (2, 3, 4)
QUARTER = 0.25


def _planted(tree: Tree, dh: int, segments: tuple[int, ...], name: str) -> Tree | None:
    return tree.grown(segments, QUARTER / dh) if plant(name) else None


# -- the contract -------------------------------------------------------------------------------------------------
#: Where the contract judges the node's fields against the exact union's, in cells of the plain minimum: a superset of
#: every face where a field is not flat by either reading (in the fluid the minimum is never deeper than the union, and
#: every field is flat from the band's fluid half, a cell, down; in the solid the minimum is the union's from the
#: fillet's radius on, and the interior's ramp ends three cells in).
JUDGED = (-1.5, 3.5)
#: The faces the exact union is evaluated at together (every candidate at each), the last chunk padded.
CHUNK = 4096


def _exact(union, parameters, points: np.ndarray, judged: np.ndarray) -> np.ndarray:
    """The union's exact distance (its candidates at every point) at ``points[judged]``, in chunks of one shape."""
    found = np.empty(len(judged))
    for start in range(0, len(judged), CHUNK):
        index = judged[start:start + CHUNK]
        padded = np.concatenate([index, np.full(CHUNK - len(index), index[0])])
        found[start:start + len(index)] = np.asarray(
            union.signed_distance(parameters, jnp.asarray(points[padded])).distance)[:len(index)]
    return found


def measure_contract(seed: int, dh: int = 8) -> dict[str, float]:
    """The contract's reading at a draw, judged without the node's own selection: on every face :data:`JUDGED` holds,
    the node's fields (its indicator and resistivity, and its distance where either reading's fields are not flat)
    against the fields made from the union's exact distance; the node's counts; and the run-constant's wall against the
    step's own (the chunk's made once, read as given). The node is read as built, or planted: the plain minimum
    everywhere (``MIME_PLANT_WALL=minimum``), or the union's candidates chosen only above half the band's fluid half
    (``MIME_PLANT_WALL=floor``, the cost lever a smaller plan would pull)."""
    angle, offset = draw(seed)
    with in_dtype("float64"):
        placed = tree_node(loop(), dh, angle=angle, offset=offset)
        node = placed.node
        grid = MACGrid(placed.layout.cells, placed.layout.spacing)
        held = dict(node._step_constants(None))  # noqa: SLF001 - the step's own constants
        fields = node.fields(held, grid)
        if plant("floor"):
            c = node.constants()
            solid = dataclasses.replace(node.tree_solid(held, grid), floor=0.5 * node.wall_floor(grid.spacing))
            fields = wall_fields(grid, [solid], band_named(c["band"]), interior_depth(c["interior"]), "float64")
        given = {**held, **node.params_pytree(), RUN_CONSTANTS_GIVEN: ()}
        given.update(node.run_constants(given))
        constant = max(float(jnp.max(jnp.abs(a - b))) for a, b in
                       zip(fields.signed_distance, node.fields(given, grid).signed_distance, strict=True))
        tree = node.tree()
        union, parameters = tree.lumen(), tree.parameters()
        c = node.constants()
        band, interior = band_named(c["band"]), interior_depth(c["interior"])
        h = placed.layout.spacing
        peak = (1.0 + (band.depth / interior) ** 2) / (band.depth * h) ** 2
        read = {"phi": 0.0, "chi": 0.0, "resistivity": 0.0, "faces": 0.0, "constant": constant / h,
                "counts": float(np.sum(np.asarray(fields.counts)))}
        for k in range(3):
            points = np.asarray(grid.face_points(k, "float64")).reshape(-1, 3)
            minimum = np.asarray(union.minimum(parameters, jnp.asarray(points)))
            judged = np.nonzero((minimum > JUDGED[0] * h) & (minimum < JUDGED[1] * h))[0]
            exact = _exact(union, parameters, points, judged)
            phi = np.asarray(fields.signed_distance[k]).reshape(-1)[judged]
            if plant("minimum"):
                phi = minimum[judged]
            chi = np.asarray(ramp(jnp.asarray(phi), band.width * h))
            resistivity = np.asarray(ramp(jnp.asarray(phi), band.width * h)
                                     + interior_ramp(jnp.asarray(phi), band, h, INTERIOR_RAMP_END)
                                     * (band.depth / interior) ** 2) / (band.depth * h) ** 2
            if not plant("minimum"):
                chi = np.asarray(fields.chi[k]).reshape(-1)[judged]
                resistivity = np.asarray(fields.inverse_permeability[k]).reshape(-1)[judged]
            chi_exact = np.asarray(ramp(jnp.asarray(exact), band.width * h))
            resistivity_exact = np.asarray(ramp(jnp.asarray(exact), band.width * h)
                                           + interior_ramp(jnp.asarray(exact), band, h, INTERIOR_RAMP_END)
                                           * (band.depth / interior) ** 2) / (band.depth * h) ** 2
            non_flat = (np.minimum(phi, exact) < INTERIOR_RAMP_END * h) & \
                (np.maximum(phi, exact) > -0.5 * band.width * h)
            read["faces"] += len(judged)
            read["phi"] = max(read["phi"], float(np.max(np.abs(phi - exact)[non_flat], initial=0.0)) / h)
            read["chi"] = max(read["chi"], float(np.max(np.abs(chi - chi_exact), initial=0.0)))
            read["resistivity"] = max(read["resistivity"],
                                      float(np.max(np.abs(resistivity - resistivity_exact), initial=0.0)) / peak)
    return read


@pytest.mark.parametrize("seed", seeds(601))
def test_the_nodes_fields_are_the_exact_unions_wherever_a_field_is_not_flat(seed):
    """The filleted loop at 8 cells a diameter, float64: on every face where a field is not flat by either reading,
    the node's indicator and resistivity are those of the union's exact distance (all its candidates at every face)
    within ``FIELD_LINE`` of their own scale, and its distance within ``CONTRACT_LINE``; its counts are zero; and the
    wall a chunk makes once (its run-constant) is the step's own, bit for bit."""
    read = measure_contract(seed)
    print(f"\nseed {seed}: on {read['faces']:.0f} faces, the node against the exact union: distance "
          f"{read['phi']:.2e} cells, indicator {read['chi']:.2e}, resistivity {read['resistivity']:.2e} of its peak; "
          f"counts {read['counts']:.0f}; the run-constant against the step's own {read['constant']:.1e} cells")
    assert read["counts"] == 0.0 and read["constant"] == 0.0
    assert read["phi"] <= CONTRACT_LINE and read["chi"] <= FIELD_LINE and read["resistivity"] <= FIELD_LINE


# -- FA3 and FA5 ----------------------------------------------------------------------------------------------------
def _inferred_shift(tree: Tree, flux: float, reference: float, h: float) -> float:
    return math.log(flux / reference) / (h * lubrication_sensitivity(tree)[1])


def measure_shift(geometry: str, seed: int) -> dict[str, float]:
    """FA3's and FA5's reading at a draw: the wall's inferred shift against its oracle (lubrication's flux for the
    taper, the reference's for the stenosis) less the planar calibration, in cells, and the wall's flag."""
    tree, dh = (taper() if geometry == "taper" else stenosis()), 16
    with in_dtype("float64"):
        placed = tree_node(tree, dh, offset=draw(seed)[1], tolerance=1e-10,
                           solver=_planted(tree, dh, tuple(range(len(tree.segments))), "wall"))
        state = run(placed.node, 1)
    oracle = lubrication_flux(tree) if geometry == "taper" else references()["stenosis"]["flux"]
    shift = _inferred_shift(tree, box_flux(state), oracle, placed.layout.spacing)
    return {"shift": shift - planar_calibration(16, "float64"), "inexact": float(state["wall_inexact"])}


@pytest.mark.parametrize("seed", seeds(701, 702))
def test_fa3_the_taper_holds_the_planar_calibration_against_lubrication(seed):
    read = measure_shift("taper", seed)
    print(f"\nseed {seed}: the taper's shift against lubrication, less the planar calibration, {read['shift']:+.4f} "
          f"cells")
    assert read["inexact"] == 0.0
    assert abs(read["shift"]) < SHIFT_LINE


@pytest.mark.parametrize("seed", seeds(801, 802))
def test_fa5_the_stenosis_holds_the_planar_calibration_against_its_reference(seed):
    assert "stenosis" in references(), "no stored reference for the stenosis (generate_wall_references.py)"
    read = measure_shift("stenosis", seed)
    print(f"\nseed {seed}: the stenosis's shift against its reference, less the planar calibration, "
          f"{read['shift']:+.4f} cells")
    assert read["inexact"] == 0.0
    assert abs(read["shift"]) < SHIFT_LINE


# -- FA6 and FA7 ----------------------------------------------------------------------------------------------------
def _split(tree: Tree, dh: int, seed: int, dtype: str = "float64", plantable: bool = True) -> tuple[float, dict]:
    """The first daughter's share of the mid-plane's flux at a draw, in ``dtype`` (float32 at its stop's default, at
    the floor its rounding reaches); planted where ``plantable`` and a plant is set."""
    angle, offset = draw(seed)
    with in_dtype(dtype):
        placed = tree_node(tree, dh, angle=angle, offset=offset, tolerance=1e-10 if dtype == "float64" else None,
                           solver=_planted(tree, dh, FIRST_DAUGHTER, "daughter") if plantable else None)
        state = run(placed.node, 1)
        first, second = mid_plane_split(placed, state)
    return first / (first + second), state


@pytest.mark.parametrize("seed", seeds(901, 902))
def test_fa6_the_loops_split_at_thirty_degrees_holds_its_reference(seed):
    reference = references().get("loop")
    assert reference is not None, "no stored reference for the loop's split (generate_wall_references.py)"
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


@pytest.mark.parametrize("seed", seeds(1401))
def test_the_loops_split_in_float32_is_float64s(seed):
    """The filleted loop at 8 cells a diameter in float32 (the stop at its dtype's default) against float64: the split
    within ``FLOAT32_LINE``, no flag in either, the wall's counts zero. The plant (the first daughter a quarter cell out)
    is the float32 run's alone."""
    split64, state64 = _split(loop(), 8, seed, plantable=False)
    split32, state32 = _split(loop(), 8, seed, "float32")
    print(f"\nseed {seed}: split {split32:.7f} in float32 against {split64:.7f} in float64 "
          f"({abs(split32 - split64):.1e})")
    for state in (state64, state32):
        assert float(state["wall_inexact"]) == 0.0 and float(state["solve_flagged"]) == 0.0
    assert abs(split32 - split64) <= FLOAT32_LINE


# -- FA8 ------------------------------------------------------------------------------------------------------------
def measure_derivative(seed: int) -> dict[str, float]:
    """FA8's reading at a draw: the reverse derivative, the central difference, their relative difference, and the
    wall's flag."""
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
    return {"reverse": reverse, "central": central, "relative": abs(reverse - central) / abs(central),
            "flux": float(value), "inexact": float(ended["wall_inexact"])}


@pytest.mark.parametrize("seed", seeds(1101))
def test_fa8_the_first_daughters_flux_in_its_radius_is_the_central_difference(seed):
    """At 8 cells a diameter, float64: ``d Q_1 / d R_1`` along the first daughter's radii (both ends of its three
    segments), reverse mode through one steady step against ``(Q(R + e) - Q(R - e)) / 2e`` at ``e`` 5e-5 of the radius,
    both sides on the node as built (its cells and its plan's size), their counts zero."""
    read = measure_derivative(seed)
    print(f"\nseed {seed}: dQ1/dR1 reverse {read['reverse']:.8e}, central {read['central']:.8e}, relative "
          f"{read['relative']:.2e}; flux {read['flux']:.6e}")
    assert read["inexact"] == 0.0
    assert read["relative"] <= DERIVATIVE_LINE
