"""The rft rung's validity lines are its record's (phase R's gate 3; #175: measured, never quoted).

- **The lines are the record's by the spec's rule**: ``MEASURED`` equals what ``scripts/slender_validity.py``'s
  ``families`` derives from ``docs/data/slender_validity/record.json`` (each family's largest miss, rounded up to two
  significant figures, over the slenderness its points span; a refused point claims none), and the record names the
  spec it was run under by its hash. **Control:** the spec edited after the record (its hash moved) is refused here.
- **A body outside them is flagged, never silent**: one of a measured family at the default coefficients and spacing
  raises no ``unmeasured``, and the build report says its lines hold; one of another family, or the same body with a
  coefficient set or a coarser spacing it was not measured at, raises it at every step and the report says its error
  is not measured; with the tube's table, a step with the filament closer to the wall than the least measured gap
  raises it."""

import hashlib
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest
from rft_cases import R6_TUBE, SMALL_HELIX, SMALL_SETTINGS, helix_body, rft

from mime.build._maddening import node_state, parameter_leaves
from mime.nodes.hydro._rft_drag import MEASURED
from mime.scene import Helix, Tube
from mime.units import mm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data" / "slender_validity"
sys.path.insert(0, str(ROOT / "scripts"))
import slender_validity  # noqa: E402


def _record():
    return json.loads((DATA / "spec.json").read_text()), json.loads((DATA / "record.json").read_text())


def test_the_lines_are_the_records_by_the_rule():
    spec, record = _record()
    derived = slender_validity.families(spec, record)
    held = [{"kind": m.kind, "shape": dict(m.shape), "wall": m.walls[0], "gap": m.gap,
             "slenderness": list(m.slenderness), "speed": m.speed, "resistance": m.resistance} for m in MEASURED]
    assert held == [{k: v for k, v in f.items() if k != "points"} for f in derived]


def spec_problem(spec_bytes: bytes, record: dict) -> str | None:
    if hashlib.sha256(spec_bytes).hexdigest() != record["spec_sha256"]:
        return "the record was run under another spec than the one beside it"
    return None


def test_the_record_names_its_spec_and_a_spec_edited_after_it_is_refused():
    _, record = _record()
    spec_bytes = (DATA / "spec.json").read_bytes()
    assert spec_problem(spec_bytes, record) is None
    assert spec_problem(spec_bytes.replace(b"largest miss", b"mean miss"), record) is not None


def _step(experiment):
    built = experiment.build()
    node = built.graph.get_node("drag:helix")
    out = node.update(node_state(built.graph, "drag:helix"), {"pose": node_state(built.graph, "helix")["pose"]},
                      1e-3, params=parameter_leaves(built.graph)["drag:helix"])
    return node, out, built


R6 = Helix(radius=mm(0.15), pitch=mm(0.6), turns=2.5, filament_radius=mm(0.03))
R6_SETTINGS = {"radius": 0.15e-3, "pitch": 0.6e-3, "turns": 2.5, "filament_radius": 0.03e-3}
OTHER = Helix(radius=mm(0.15), pitch=mm(0.75), turns=2.5, filament_radius=mm(0.03))
OTHER_SETTINGS = {"radius": 0.15e-3, "pitch": 0.75e-3, "turns": 2.5, "filament_radius": 0.03e-3}


@pytest.mark.parametrize("kind, settings, law, measured", [
    (R6, R6_SETTINGS, rft(wall_kind="free_space"), True),
    (OTHER, OTHER_SETTINGS, rft(wall_kind="free_space"), False),
    (R6, R6_SETTINGS, rft(wall_kind="free_space", spacing=mm(0.06)), False),
], ids=["measured", "another_family", "coarser_spacing"])
def test_a_body_outside_the_measured_points_is_flagged(kind, settings, law, measured):
    node, out, _ = _step(helix_body("rft_lines", kind, settings, law))
    assert float(out["unmeasured"]) == (0.0 if measured else 1.0)
    (line,) = node.built_lines({"pose": out["stopped_pose"]}, None)
    assert ("hold for this body's family" in line) == measured
    assert ("not measured here" in line) == (not measured)


def _tube(radius_mm):
    return Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(radius_mm), wall=mm(0.1), length=mm(20.0))


@pytest.mark.parametrize("radius_mm, offset, gap, flagged", [
    (0.5, 0.0, 44.0, False), (0.5, 0.2e-3, 24.0, True), (0.8, 0.0, 74.0, True), (1.5, 0.0, 144.0, True)],
    ids=["at_the_measured_gap", "closer", "a_wider_tube", "a_much_wider_tube"])
def test_with_the_tube_a_body_at_another_gap_than_the_one_measured_is_flagged(radius_mm, offset, gap, flagged):
    """The tube family was measured at one gap, 44 filament radii: a narrower gap and a wider one are both flagged each
    step, and the build report says the start is not measured there. (The family's ``measured`` is the shape's, the
    slenderness', the coefficients' and the spacing's; the gap is judged at the pose each step.)"""
    node, out, built = _step(helix_body("rft_gap", SMALL_HELIX, SMALL_SETTINGS, rft(wall_kind="tube"),
                                        tube=_tube(radius_mm), offset=(0.0, offset, 0.0)))
    assert node.constants()["measured"]
    assert float(out["unmeasured"]) == (1.0 if flagged else 0.0)
    pose = node_state(built.graph, "helix")["pose"]
    held = {**node.params_pytree(), **{k: jnp.asarray(v) if isinstance(v, float) else v
                                       for k, v in node.constants().items()}}
    assert float(node.wall_gap(held, pose)) == pytest.approx(gap, rel=1e-3)
    (line,) = node.built_lines({"pose": pose}, None)
    assert ("were measured at a gap of 44 filament radii" in line) == flagged
    assert ("at the gap measured" in line) == (not flagged)


def test_the_tube_family_is_its_measured_gap_only_and_free_space_stays_its_own_family():
    tube = [m for m in MEASURED if "tube" in m.walls]
    assert [(m.gap, m.walls) for m in tube] == [(44.0, ("tube",))]
    assert all(m.gap == 0.0 for m in MEASURED if "free_space" in m.walls)
