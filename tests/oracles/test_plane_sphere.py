"""A sphere near a plane wall through MIME's own drag (``Drag.MRS`` in the plane wall kind), against exact answers:
Brenner's (1961) series for motion along the normal (``plane_oracles.brenner``, summed in the test) and, for the
translation along the wall, the rotation about an axis along it and their coupling, the method of fundamental
solutions with Blake's singular image (``plane_oracles.mfs``, sharing no code with MIME), whose values at the gaps judged
are committed (``tests/differential/references/plane_mfs.json``, written by ``generate_plane_mfs.py``, pinned by its
sha256). The sphere at a force spacing of 0.3 radii with a blob of 0.1, its quadrature by the blob's rule; the image at
the force points (the default) a radius and half a radius from the wall, and integrated over the cells near the wall
(``wall_refinement=2``) a force spacing from it; each gap with the sphere unturned (a pole of its sampling grid facing
the wall) and turned by 1.2 rad, the same body sampled otherwise against the wall.

**Where the line holds, and where it does not** (the map made when the wall kind was built, the node's own path, float64): at a force spacing or
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
  force points and ``INTEGRATED_GAP`` integrated (dev1's lines, measured in the tube; the map of the sphere near
  the plane found them on the safe side, not contradicted). **The point image's line is judged at a blob of 0.5, where
  the resistance goes indefinite at 0.3 spacings and a control reads it so in the same build; at the blob 0.1 of the
  other gates it is definite down to 0.1 spacings but for one pose, so no line there is tested. The integrated line is
  a regression check only:** no integrated pose was mapped indefinite at any setting, so nothing here discriminates it.
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
from plane_cases import BLOB, RADIUS, VISCOSITY, drag_node, scaled, sphere_above_plane
from plane_oracles import brenner

from mime.build._maddening import node_state, parameter_leaves
from mime.nodes.hydro.mrs import _plane
from mime.nodes.hydro.mrs._node import CONFINED_GAP, INTEGRATED_GAP
from mime.units import m

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


def resistances(refinement, gaps, x64: bool = True, orientation=TURNS[0], blob: float = BLOB) -> dict[float, dict]:
    """The node's resistance and least scaled eigenvalue at each gap, the sphere at ``orientation`` and the drag's
    ``blob``, built once."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64)
    try:
        built = sphere_above_plane(SPACING, 2.0, blob=blob, wall_refinement=refinement).build()
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


def test_the_report_prices_the_plane_with_its_own_figures_and_not_the_tubes():
    """The plane has no table: its cost line says so and quotes the plane's step and memory, and its near-wall line the
    plane's own errors; the tube's line (reads of the wall's table, a step of 45 s at 4,816 points, a fit to six runs
    in a tube) is the tube's, and is still the tube's."""
    from mime.nodes.hydro.mrs import _node
    from mrs_cases import mrs_tube

    plane = [line for held in sphere_above_plane(SPACING, 1.0).build().report.checked.values() for line in held]
    cost = next(line for line in plane if line.startswith("what a step costs"))
    assert "closed form, with no table to read" in cost and _plane.PLANE_COST in cost, cost
    assert "reads of the wall's table" not in cost and _node.STEP_MEASURED not in cost and "a fit of" not in cost, cost
    assert _plane.PLANE_NEAR_WALL in plane[0] and _node.NEAR_WALL_MAPPED not in plane[0], plane[0]
    tube = [line for held in mrs_tube().build().report.checked.values() for line in held]
    cost = next(line for line in tube if line.startswith("what a step costs"))
    assert "reads of the wall's table" in cost and _node.STEP_MEASURED in cost, cost
    assert _node.NEAR_WALL_MAPPED in tube[0], tube[0]


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


def test_the_offset_is_reached_as_the_node_leaf_and_no_override_or_study_axis_names_it():
    """The shipped text says so: the offset is the drag node's leaf ``wall_offset`` (the test above), and a structure's
    setting is not yet an override or a study axis. The refusals, by code and route: ``floor.offset`` as an override is
    ``value.unknown`` (a structure has no value), ``drag:ball.wall_offset`` and ``drag:ball.offset`` as an override are
    ``study.unknown-path``, and as a study axis likewise; ``floor.offset`` as a study axis plans and its run refuses
    every point (``study.refused-points``, each ``value.unknown``). Control: a node setting the same
    route does reach (``drag:ball.blob``) is accepted, so the refusal is the offset's own."""
    from mime import Study
    from mime.build import ArgumentRefused, BuildRefused
    from mime.study import grid
    from mime.units import s

    experiment = sphere_above_plane(SPACING, 2.0)
    built = experiment.build()
    with pytest.raises(BuildRefused) as structure:
        built.run(s(0.001), override={"floor.offset": m(1e-4)})
    assert structure.value.codes() == ["value.unknown"]
    for path in ("drag:ball.wall_offset", "drag:ball.offset"):
        with pytest.raises(ArgumentRefused) as override:
            built.run(s(0.001), override={path: m(1e-4)})
        assert override.value.codes() == ["study.unknown-path"], path
        with pytest.raises(ArgumentRefused) as axis:
            Study(experiment).sweep(grid({path: [m(1e-4), m(2e-4)]}), duration=s(0.001)).plan()
        assert axis.value.codes() == ["study.unknown-path"], path
    with pytest.raises(BuildRefused) as study_structure:
        Study(experiment).sweep(grid({"floor.offset": [m(1e-4), m(2e-4)]}), duration=s(0.001)).run()
    assert study_structure.value.codes() == ["study.refused-points"]
    assert "[value.unknown] a structure has no value 'offset'" in str(study_structure.value)
    Study(experiment).sweep(grid({"drag:ball.blob": [0.1, 0.2]}), duration=s(0.001)).plan()


#: The gaps the definiteness is judged at, in force spacings, by the image's form: from the drag's clear line out.
CLEAR = {None: (CONFINED_GAP, 1.0, 2.0), 2: (INTEGRATED_GAP, 0.5, 1.0)}


#: The blob at which the image at the force points goes indefinite near the plane (a map of the sphere at 0.3 radii,
#: float64: indefinite at 0.33 spacings and every mapped gap below it but 0.1, definite from 0.36 up, so the drag's clear
#: line, 0.55, lies in a band where the resistance is definite and 0.22 spacings above where it is not), and the gap
#: in spacings, with that blob, at which the control reads it indefinite (least -3.3 in float64).
INDEFINITE_BLOB, INDEFINITE_GAP = 0.5, 0.3


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_the_image_at_the_force_points_is_definite_at_and_beyond_the_clear_line_where_it_goes_indefinite_below(x64):
    """The clear line is judged where it can fail: at a blob of 0.5 the resistance is indefinite at 0.3 spacings (the
    control, read in the same build, so a test that cannot see an indefinite resistance fails here) and definite from the
    drag's clear line, ``CONFINED_GAP``, out. At the other blob of this file, 0.1, the resistance is definite down to
    0.1 spacings but for one pose (0.36 spacings, the point image), so a line at or above 0.4 passes: not evidence
    for the line."""
    gaps = tuple(s * SPACING for s in (*CLEAR[None], INDEFINITE_GAP))
    found = resistances(None, gaps, x64=x64, blob=INDEFINITE_BLOB)
    control = INDEFINITE_GAP * SPACING
    assert found[control]["least"] <= 0.0, ("the control: indefinite", found[control]["least"])
    for gap, held in found.items():
        if gap != control:
            assert held["least"] > 0.0, (gap / SPACING, held["least"])


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_the_integrated_image_is_definite_at_and_beyond_the_integrated_line_a_regression_check_only(x64):
    """A regression check on three poses, not evidence for ``INTEGRATED_GAP``: no integrated pose of the sphere
    near the plane was mapped indefinite at either force spacing or blob (0.3 and 0.2 radii, blobs 0.1 and 0.5, down to
    0.1 spacings), so no setting here discriminates the line, which rests on the tube's measurement (dev1)."""
    gaps = tuple(s * SPACING for s in CLEAR[2])
    for gap, held in resistances(2, gaps, x64=x64).items():
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
