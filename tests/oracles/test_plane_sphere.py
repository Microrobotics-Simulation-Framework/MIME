"""A sphere near a plane wall through MIME's own drag (``Drag.MRS`` in the plane wall kind), against exact answers:
Brenner's (1961) series for motion along the normal (``plane_oracles.brenner``, summed in the test) and, for the
translation along the wall, the rotation about an axis along it and their coupling, the method of fundamental
solutions with Blake's singular image (``plane_oracles.mfs``, sharing no code with MIME), whose values at the gaps judged
are committed (``tests/differential/references/plane_mfs.json``, written by ``generate_plane_mfs.py``, pinned by its
sha256). The sphere at a force spacing of 0.3 radii with a blob of 0.1, its quadrature by the blob's rule; the image at
the force points (the default) a radius and half a radius from the wall, and integrated over the cells near the wall
(``wall_refinement=2``) a force spacing from it; each gap with the sphere unturned (a pole of its sampling grid facing
the wall) and turned by 1.2 rad, the same body sampled otherwise against the wall.

**Where the line holds, and where it does not** (the phase's map, the node's own path, float64): at a force spacing or
more from the wall every entry lies within 3.7% of its oracle, at spacings of 0.3 and 0.2 radii, in both orientations.
At half a spacing the normal entry, the squeeze, is the first to go: integrated over the cells, 2.6% and 4.6% off at a
spacing of 0.3 radii, 5.0% and 14.8% at 0.2 (the sphere's grid pole facing the wall the worse), and at the force points
it overshoots and then collapses below about half a spacing; that is the drag's near-wall validity line (the drag
raises ``near_wall`` within a spacing), not this image's, and lubrication's hand-over is what closes it.

- **The entries** (float64): each within ``ENTRY_LINE`` of its oracle, the coupling judged on ``sqrt(R_tt R_rr)``, the
  scale it couples (far from the wall it is near zero, and its relative error noise). The line holds the coarse MRS's
  own error near the wall (``measured`` below) with room, and lies far under the planted errors' shifts.
- **Definiteness** (both dtypes): the resistance positive definite (its least eigenvalue, each row and column scaled by
  its diagonal, above zero) at every gap at or above the drag's clear line, ``CONFINED_GAP`` force spacings at the
  force points and ``INTEGRATED_GAP`` integrated (dev1's lines, measured in the tube; the phase's map of the sphere near
  the plane found them on the safe side).
- **Both dtypes:** the float32 resistance within ``FLOAT32_LINE`` of the float64 one, entry by entry on its diagonal
  scale.

**The controls**, failing on every run (nothing is drawn): the image's Stokeslet, its doublet and its dipole each with
its sign flipped move the entries far past the line at every gap; **the rotlet's flip moves them by less than the MRS's
own error, so this gate cannot see the rotlet** (its control is the image's no-slip test,
``tests/oracles/test_plane_image.py``), nor a blob dropped from the image."""

import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plane_cases import RADIUS, VISCOSITY, drag_node, scaled, sphere_above_plane
from plane_oracles import brenner

from mime.build._maddening import node_state, parameter_leaves
from mime.nodes.hydro.mrs import _plane
from mime.nodes.hydro.mrs._node import CONFINED_GAP, INTEGRATED_GAP

REFERENCE = Path(__file__).parents[1] / "differential" / "references" / "plane_mfs.json"
REFERENCE_SHA256 = "9f457cbec8526288b75fb213c1d838d70f90725136b0cd1d03bf6e4bd60968d9"
SPACING = 0.3
#: The gaps judged, in the sphere's radius, by the image's form: at the force points, and integrated over the cells.
GAPS = {None: (1.0, 0.5), 2: (0.3,)}
#: The orientations: unturned (a pole of the sphere's sampling grid faces the wall) and turned by 1.2 rad about x.
TURNS = ((1.0, 0.0, 0.0, 0.0), (math.cos(0.6), math.sin(0.6), 0.0, 0.0))
#: Each entry's line against its oracle: the coarse MRS's own error at these gaps measured at most 2.8% (the normal,
#: integrated, the sphere turned; the rotation 1.8% to 2.3%, along the wall 0.8% to 1.2%, the coupling under 0.2% of its
#: scale); the flipped terms' least shift, 57% (the doublet's, on the normal a radius out), eleven times over it.
ENTRY_LINE = 0.05
#: Float32 against float64 on each entry's diagonal scale: measured at most 5.0e-6 at these gaps and half a spacing
#: (integrated, 0.15 radii), twenty times under it.
FLOAT32_LINE = 1e-4
TERMS = ("stokeslet", "doublet", "dipole")


def reference() -> dict:
    data = REFERENCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == REFERENCE_SHA256, "the reference file is not the one pinned"
    return json.loads(data)


def oracle(gap: float) -> dict[str, float]:
    """The exact entries at ``gap``: Brenner's on the normal, the fine MFS's on the rest, in the literature's units."""
    held = np.asarray(reference()["rows"][f"fine {gap}"]["resistance"])
    return {"normal": brenner(gap), "tangential": float(held[0, 0] / (6 * math.pi)),
            "rotational": float(held[3, 3] / (8 * math.pi)), "coupling": float(held[0, 4] / (6 * math.pi)),
            "coupling_scale": float(math.sqrt(held[0, 0] * held[4, 4]) / (6 * math.pi))}


def resistances(refinement, gaps, x64: bool = True, orientation=TURNS[0]) -> dict[float, dict]:
    """The node's resistance and least scaled eigenvalue at each gap, the sphere at ``orientation``, built once."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64)
    try:
        built = sphere_above_plane(SPACING, 2.0, wall_refinement=refinement).build()
        node = drag_node(built)
        state = node_state(built.graph, "drag:ball")
        leaves = parameter_leaves(built.graph)["drag:ball"]
        step = jax.jit(lambda pose: node.update(state, {"pose": pose}, 1e-3, params=leaves))
        found = {}
        for gap in gaps:
            pose = jnp.asarray((0.0, 0.0, RADIUS * (1.0 + gap), *orientation), dtype=state["resistance"].dtype)
            out = step(pose)
            found[gap] = {"resistance": np.asarray(out["resistance"], dtype=np.float64),
                          "least": float(out["least_eigenvalue"])}
        return found
    finally:
        jax.config.update("jax_enable_x64", held)


def misses(resistance: np.ndarray, gap: float) -> dict[str, float]:
    """Each entry's error against its oracle, the coupling's on the scale it couples."""
    found, exact = scaled(resistance), oracle(gap)
    out = {name: found[name] / exact[name] - 1.0 for name in ("normal", "tangential", "rotational")}
    out["coupling"] = (found["coupling"] - exact["coupling"]) / exact["coupling_scale"]
    return out


def test_the_reference_is_the_pinned_file_and_its_mfs_agrees_with_brenner_and_itself():
    held = reference()
    for gap in held["gaps"]:
        for setting in ("coarse", "fine"):
            row = held["rows"][f"{setting} {gap}"]
            assert abs(row["normal_over_brenner"] - 1.0) <= 1e-4, (setting, gap, row["normal_over_brenner"])
            assert row["asymmetry"] <= 1e-5, (setting, gap)
        coarse, fine = (np.asarray(held["rows"][f"{s} {gap}"]["resistance"]) for s in ("coarse", "fine"))
        scale = np.sqrt(np.abs(np.outer(np.diag(fine), np.diag(fine))))
        assert float(np.max(np.abs(coarse - fine) / scale)) <= ENTRY_LINE / 50.0, gap


@pytest.mark.parametrize("orientation", TURNS, ids=["unturned", "turned"])
@pytest.mark.parametrize("refinement", sorted(GAPS, key=str), ids=["at the force points", "integrated"])
def test_each_entry_lies_within_the_line_of_its_exact_answer(refinement, orientation):
    for gap, held in resistances(refinement, GAPS[refinement], orientation=orientation).items():
        found = misses(held["resistance"], gap)
        print(f"\nwall_refinement {refinement}, gap {gap} a ({gap / SPACING:.3g} s): {found}")
        assert all(abs(value) <= ENTRY_LINE for value in found.values()), (gap, found)


def test_the_build_reports_the_planes_regularised_image_and_the_tubes_singular_one():
    built = sphere_above_plane(SPACING, 1.0).build()
    lines = [line for held in built.report.checked.values() for line in held]
    assert any("the wall's image (plane): regularised, the drag's blob, eps = " in line for line in lines), lines
    from mrs_cases import mrs_tube

    tube = [line for held in mrs_tube().build().report.checked.values() for line in held]
    assert any("the wall's image (tube): singular" in line for line in tube), tube


def test_a_body_starting_through_the_plane_is_refused_and_one_above_it_is_not():
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as refused:
        sphere_above_plane(SPACING, -0.1).build()
    assert "reaches through the wall at its starting pose" in str(refused.value)
    sphere_above_plane(SPACING, 0.5).build()


def test_the_planes_offset_is_one_trainable_leaf_and_moving_it_is_a_build_at_the_moved_offset():
    """The drag's ``wall_offset`` is a trainable leaf the compiled step reads (no rebuild: the image reads it at each
    call), and the resistance with it moved is a fresh build's at that offset, bit for bit; the normal is held as built
    (a leaf the step reads, not trainable)."""
    built = sphere_above_plane(SPACING, 1.0).build()
    node = drag_node(built)
    assert node.param_specs()["wall_offset"].trainable and not node.param_specs()["wall_normal"].trainable
    state = node_state(built.graph, "drag:ball")
    pose = jnp.asarray((0.0, 0.0, RADIUS * 2.0, 1.0, 0.0, 0.0, 0.0))
    leaves = {**parameter_leaves(built.graph)["drag:ball"], "wall_offset": jnp.asarray(1e-4)}
    moved = np.asarray(node.update(state, {"pose": pose}, 1e-3, params=leaves)["resistance"])
    fresh = sphere_above_plane(SPACING, 1.0 - 1e-4 / RADIUS, offset=1e-4).build()
    again = np.asarray(drag_node(fresh).update(node_state(fresh.graph, "drag:ball"), {"pose": pose}, 1e-3,
                                               params=parameter_leaves(fresh.graph)["drag:ball"])["resistance"])
    assert np.array_equal(moved, again)


#: The gaps the definiteness is judged at, in force spacings, by the image's form: from the drag's clear line out.
CLEAR = {None: (CONFINED_GAP, 1.0, 2.0), 2: (INTEGRATED_GAP, 0.5, 1.0)}


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
@pytest.mark.parametrize("refinement", sorted(CLEAR, key=str), ids=["at the force points", "integrated"])
def test_the_resistance_is_definite_at_and_beyond_the_drags_clear_line(refinement, x64):
    gaps = tuple(s * SPACING for s in CLEAR[refinement])
    for gap, held in resistances(refinement, gaps, x64=x64).items():
        assert held["least"] > 0.0, (gap / SPACING, held["least"])


@pytest.mark.parametrize("refinement", sorted(GAPS, key=str), ids=["at the force points", "integrated"])
def test_float32_lies_within_its_line_of_float64(refinement):
    wide, narrow = resistances(refinement, GAPS[refinement]), resistances(refinement, GAPS[refinement], x64=False)
    for gap in GAPS[refinement]:
        a, b = wide[gap]["resistance"], narrow[gap]["resistance"]
        scale = np.sqrt(np.abs(np.outer(np.diag(a), np.diag(a))))
        found = float(np.max(np.abs(b - a) / scale))
        print(f"\nwall_refinement {refinement}, gap {gap} a: float32 against float64 {found:.3g} of the diagonal scale")
        assert found <= FLOAT32_LINE, (gap, found)


def flipped(monkeypatch, term):
    """The image built with ``term`` flipped in sign (each case builds its own node, which traces the image again)."""
    held = _plane.image_terms

    def terms(targets, sources, parameters, eps_squared):
        found = held(targets, sources, parameters, eps_squared)
        return found._replace(**{term: -getattr(found, term)})

    monkeypatch.setattr(_plane, "image_terms", terms)


@pytest.mark.parametrize("term", TERMS)
@pytest.mark.parametrize("refinement", sorted(GAPS, key=str), ids=["at the force points", "integrated"])
def test_control_a_term_flipped_misses_past_the_line_at_every_gap(monkeypatch, refinement, term):
    """Caught where an entry misses by more than the line, or where the planted resistance is not a number (the node
    stops a run on an indefinite one)."""
    flipped(monkeypatch, term)
    for gap, held in resistances(refinement, GAPS[refinement]).items():
        if not np.all(np.isfinite(held["resistance"])):
            continue
        found = misses(held["resistance"], gap)
        assert max(abs(value) for value in found.values()) > 3 * ENTRY_LINE, (term, gap, found)


@pytest.mark.parametrize("refinement", sorted(CLEAR, key=str), ids=["at the force points", "integrated"])
def test_control_the_dipole_flipped_is_indefinite_at_the_drags_clear_line(monkeypatch, refinement):
    flipped(monkeypatch, "dipole")
    gap = CLEAR[refinement][0] * SPACING
    assert resistances(refinement, (gap,))[gap]["least"] <= 0.0
