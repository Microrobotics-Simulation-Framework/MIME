"""An inertia tensor whose principal axes are not the body frame's (``parts.inertia-axes``).

``research`` refuses it, and the refusal prints what a user needs to act on it (dev1 phase 7l-c, design #191 item 7): the
tensor the parts give in the body's frame with its units, its principal moments and axes, and the declaration that
states the diagonal with a reason, written out to paste. ``exploratory`` waives it (design #225): the diagonal is used
with the products dropped, and the build stamps it pending with the products' size, through the profile machinery's
waiver, not a case of its own.

**Controls.** The pasted declaration, executed, builds in research: the refusal's fix is a fix. A plain body (a sphere,
whose tensor is diagonal) is neither refused nor stamped, and a tensor whose product is planted large is refused with
that product named, so the message reads the tensor and not a constant. The waiver is held by removing it: the same
experiment, built in exploratory with the rule for the class taken out of the tables, is refused as research refuses
it, so the waiver is the profile's rule and nothing else; and the stamp's sizes are compared with the tensor the
refusal prints, so they are read and not constants."""

import re

import numpy as np
import pytest
from profile_cases import parts_screw

from mime import Body, Experiment, State
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.scene import Box, Scene, Screw, Sphere, Z
from mime.things import Part
from mime.units import dimensionless, kg_m2, kg_per_m3, m_per_s, m_per_s2, mm, ms, rad_per_s

AT_REST = dict(orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
               angular_velocity=rad_per_s((0.0, 0.0, 0.0)))


def screw_experiment(profile: str = "exploratory") -> Experiment:
    return parts_screw(profile)


def refusal_of(exp: Experiment | None = None):
    exp = exp if exp is not None else screw_experiment("research")
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    (found,) = [r for r in refused.value.refusals if r.code == "parts.inertia-axes"]
    return found


def tensor_in(message: str) -> np.ndarray:
    rows = re.findall(r"^\s+\[([^\]]*)\]\s*$", message, re.M)
    return np.array([[float(c) for c in row.split()] for row in rows])


def test_the_refusal_prints_the_tensor_in_the_bodys_frame_its_principal_moments_and_the_declaration_to_paste():
    found = refusal_of()
    assert "in the body's frame (kg m^2)" in found.message and "its principal moments, with their axes" in found.message
    tensor = tensor_in(found.message)
    assert tensor.shape == (3, 3) and np.allclose(tensor, tensor.T)
    assert tensor[1, 2] != 0.0 and tensor[0, 1] == 0.0
    moments = np.linalg.eigvalsh(tensor)
    printed = [float(n) for n in re.findall(r"^\s+([0-9.e+-]+) kg m\^2 about", found.message, re.M)]
    assert printed == pytest.approx(sorted(moments), rel=1e-5)
    assert 'exp.declare("umr.inertia", kg_m2((' in found.action and 'reason="' in found.action
    assert "its axes lie at most" in found.message and "differ from the diagonal's by at most" in found.message


def test_the_pasted_declaration_builds_and_declares_the_diagonal_the_message_prints():
    found = refusal_of()
    line = re.search(r'(exp\.declare\("umr\.inertia", kg_m2\(\([^)]*\)\), reason="[^"]*"\))', found.action)
    assert line is not None
    exp = screw_experiment("research")
    exec(line.group(1), {"exp": exp, "kg_m2": kg_m2})  # noqa: S102 - the refusal's own line, run to show it works
    built = exp.build()
    declared = [v for v in built.report.values if v.path == "umr.inertia"]
    assert declared and tuple(declared[0].value) == pytest.approx(tuple(np.diag(tensor_in(found.message))), rel=1e-5)
    assert "dropped" in declared[0].reason


def test_a_body_with_a_diagonal_tensor_is_not_refused():
    exp = Experiment("ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(position=mm((0.0, 0.0, 0.0)), **AT_REST),
                      parts={"steel": Part(Sphere(radius=mm(0.5)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    assert exp.build().plans.dt == pytest.approx(1e-3)


def test_the_message_reads_the_tensor_it_is_given_a_planted_product_is_named_and_shown():
    from mime.things import PartMass, derive

    cube = PartMass("a", "box", 1e-9, (0.0, 0.0, 0.0), ((1.0e-6, 0.0, 0.0), (0.0, 1.0e-6, 3.0e-7), (0.0, 3.0e-7, 2.0e-6)))
    (inertia,) = [derived for name, derived in derive("planted", [cube], {"a": 1000.0}, {}).items() if name == "inertia"]
    ((code, message, action),) = (inertia.simplified.problem,)
    assert code == "parts.inertia-axes" and "3e-13 kg m^2" in message and "+3.000000e-13" in message
    numbers = re.search(r"kg_m2\(\(([^)]*)\)\)", action)
    assert numbers is not None
    assert [float(n) for n in numbers.group(1).split(",")] == pytest.approx([1e-12, 1e-12, 2e-12], rel=1e-9)


def planted(product: float) -> tuple[str, str]:
    """The refusal of a body whose tensor has the diagonal (1, 1, 2) e-12 and ``product`` between its y and z axes (the
    part's mass is 1e-6 kg, so the per-mass tensor is a million times these)."""
    from mime.things import PartMass, derive

    part = PartMass("a", "box", 1e-9, (0.0, 0.0, 0.0), ((1.0e-6, 0.0, 0.0), (0.0, 1.0e-6, product * 1e6), (0.0, product * 1e6, 2.0e-6)))
    ((_, message, action),) = (derive("planted", [part], {"a": 1000.0}, {})["inertia"].simplified.problem,)
    return message, action


def test_the_action_is_ordered_by_the_products_size_declare_the_diagonal_first_where_small_place_the_parts_where_not():
    small_message, small = planted(1.0e-14)  # 0.5% of the largest moment, below the stated 1%
    large_message, large = planted(4.0e-13)  # 20%
    assert small.startswith("declare the moments about the body's axes") and ", or place the parts symmetrically" in small
    assert large.startswith("place the parts symmetrically") and "the products are 20% of the largest moment" in large
    assert "above the 1% past which dropping them is not offered first" in large
    assert "or, to use the diagonal all the same, declare the moments" in large
    assert "of the largest from the tensor's principal moments" in large and "degrees from the body's" in large
    assert "above the 1%" not in small and "the products are" not in small


def test_the_order_turns_at_the_stated_fraction_and_not_by_chance_either_side_of_it():
    """The control: a product 0.9 and 1.1 times the fraction of the largest moment (2e-12) falls either side."""
    from mime.things._parts import LARGE_PRODUCT

    below, above = planted(0.9 * LARGE_PRODUCT * 2.0e-12)[1], planted(1.1 * LARGE_PRODUCT * 2.0e-12)[1]
    assert below.startswith("declare") and above.startswith("place")


def test_the_diagonal_in_a_large_product_refusals_declaration_is_the_tensors_own():
    _, action = planted(4.0e-13)
    line = re.search(r'(exp\.declare\("planted\.inertia", kg_m2\(\([^)]*\)\), reason="[^"]*"\))', action)
    assert line is not None
    numbers = re.search(r"kg_m2\(\(([^)]*)\)\)", line.group(1))
    assert numbers is not None
    assert [float(n) for n in numbers.group(1).split(",")] == pytest.approx([1e-12, 1e-12, 2e-12], rel=1e-9)


# -- exploratory: the waiver (design #225) ----------------------------------------------------------------------------


def stamped(built) -> tuple[object, object]:
    outcome = built.report.profile
    (item,) = [each for each in outcome.pending if each.target == "umr.inertia"]
    return outcome, item


def test_exploratory_builds_the_screw_whose_parts_give_a_product_and_stamps_the_inertia_pending():
    built = screw_experiment().build()
    outcome, item = stamped(built)
    assert item.codes == ("inertia_products_dropped",) and not item.listed
    assert "umr.inertia (products of inertia dropped, 0.11% of the largest moment)" in outcome.stamp().text()
    assert any(w.target == "umr.inertia" and w.reasons == ("inertia_products_dropped",) and w.generated
               for w in outcome.waivers), "the profile's rule generated a waiver, as for every class"
    (line,) = [v for v in built.report.values if v.path == "umr.inertia"]
    tensor = tensor_in(refusal_of().message)
    assert tuple(line.value) == pytest.approx(tuple(np.diag(tensor)), rel=1e-5), "the diagonal is what is used"
    assert any("products of inertia dropped" in d or "the products of inertia dropped" in d for d in line.details)


def test_the_stamp_names_the_products_size_in_numbers_read_from_the_tensor_not_constants():
    tensor = tensor_in(refusal_of().message)
    diagonal, products = np.diag(tensor), tensor - np.diag(np.diag(tensor))
    moments, axes = np.linalg.eigh(tensor)
    share = 100.0 * np.abs(products).max() / np.abs(diagonal).max()
    turn = np.degrees(max(np.arccos(min(1.0, float(np.abs(axes[:, k]).max()))) for k in range(3)))
    shift = 100.0 * np.abs(moments - np.sort(diagonal)).max() / np.abs(diagonal).max()
    _, item = stamped(screw_experiment().build())
    numbers = [float(n) for n in re.findall(r"([0-9.]+)(?:%| degrees)", item.measured)]
    assert numbers == pytest.approx([share, turn, shift], rel=5e-2), "two significant figures, read from the tensor"
    assert f"{share:.2g}%" in item.kind


def test_the_item_pastes_as_the_declaration_of_the_diagonal_with_its_reason_pending():
    _, item = stamped(screw_experiment().build())
    assert item.python.startswith('exp.declare("umr.inertia", kg_m2((') and "reason=<reason>" in item.python
    exp = screw_experiment()
    exec(item.python.replace("<reason>", '"the parts\' tensor, its products dropped"'),  # noqa: S102 - the item's own line
         {"exp": exp, "kg_m2": kg_m2})
    outcome = exp.build().report.profile
    assert not [each for each in outcome.pending if each.target == "umr.inertia"], "declared, it is no longer waived"


def test_a_body_with_a_diagonal_tensor_is_not_stamped_for_it():
    exp = Experiment("ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(position=mm((0.0, 0.0, 0.0)), **AT_REST),
                      parts={"steel": Part(Sphere(radius=mm(0.5)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    assert not [each for each in exp.build().report.profile.items if "inertia" in each.target]


def test_the_control_with_the_waiver_removed_exploratory_refuses_as_research_does():
    """The waiver is the profile's rule for the class and nothing else: without the row, the experiment that built is
    refused with the same code and the same declaration to paste."""
    from mime.validity import FINDING_CLASSES, PROFILE_RULES, Profile, ProfileRules, using_tables

    rules = ProfileRules({key: rule for key, rule in PROFILE_RULES.rules.items()
                          if key != (Profile.EXPLORATORY, "simplification")})
    with using_tables(FINDING_CLASSES, rules):
        found = refusal_of(screw_experiment())
    assert found.message == refusal_of().message and found.action == refusal_of().action
    assert screw_experiment().build() is not None, "and with the row in place it builds"


def test_the_waiver_is_in_the_saved_form_and_a_loaded_experiment_rebuilds_it(tmp_path):
    """The lock records the waiver (target and code, never words), so a load reproduces the input hash with it, and the
    loaded experiment stamps the same item."""
    import json

    from mime import Experiment
    from mime.saved import LOCK_FILE

    screw_experiment().save(tmp_path / "saved", steps=2)
    lock = json.loads((tmp_path / "saved" / LOCK_FILE).read_text())
    assert {"target": "umr.inertia", "codes": ["inertia_products_dropped"]} in lock["resolved"]["profile"]["pending"]
    assert any(w["target"] == "umr.inertia" for w in lock["resolved"]["profile"]["waivers"])
    loaded = Experiment.load(tmp_path / "saved")
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()
    assert [i.target for i in loaded.build().report.profile.pending] == ["umr.inertia"]
