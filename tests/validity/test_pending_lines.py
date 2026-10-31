"""``built.report.pending()`` and the lines it prints (design #190 item 5): each pending item in plain words, with a
line of Python to paste and a fragment of the saved document to merge, through one formatter. **The pasted lines
build**: the settling ball's script, written with each item's line pasted where the item says, builds in exploratory
with the same items pending, now the experiment's own; its saved document with each fragment merged in builds the
same way; and both, each ``pending()`` replaced by a reason and the profile raised, build in research. The control: the
lines with one item left out leave that item to the profile, and in research refuse it."""

import json
from pathlib import Path

import pytest
import yaml
from profile_cases import BARE_PENDING, settling_ball

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE
from mime.validity import PENDING_PYTHON, PENDING_YAML, also_line, document_fragment, python_line

#: The settling ball as a new user writes it, with a slot where each kind of item's line is pasted.
SCRIPT = '''
from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.materials import Acceptance, Record, packaged_record
from mime.observations import SwimSpeed
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import degC, kg_per_m3, m, m_per_s, mm, ms, rad_per_s, s
from mime.validity import pending

exp = Experiment("settling_ball", scene=Scene.new(up=Z, units="m"), dt=ms(1.0), profile=PROFILE,
                 conditions={"temperature": degC(20.0, plus_minus=0.5)})
water = exp.add_medium(Medium("water", material=Record(
    "water", packaged_record("openmaterica_example_water"), require={"test_method": ("reference_formulation",)},
    RECORD)))
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Sphere(radius=mm(0.5)))}))
DECLARE
exp.attach(Drag.BEM(DRAG), body=ball, medium=water)
exp.attach(Gravity(), body=ball, medium=water)
exp.observe(SwimSpeed(ball, along=-Z, after=s(0.004)))
AFTER
'''
SLOTS = {"Record(...)": "RECORD", "exp.declare(...)": "DECLARE", "Drag(...)": "DRAG"}


def pasted(items, reason: str = PENDING_PYTHON, profile: str = "exploratory") -> Experiment:
    """The script with each item's line pasted into the slot its ``where`` names, run."""
    slots: dict[str, list[str]] = {slot: [] for slot in [*SLOTS.values(), "AFTER"]}
    for item in items:
        slot = next((slot for name, slot in SLOTS.items() if item.where.endswith(name)), "AFTER")
        slots[slot].append(python_line(item, reason))
        if item.also:
            slots["AFTER"].append(also_line(item, reason))
    text = SCRIPT.replace("PROFILE", repr(profile))
    for slot, lines in slots.items():
        joiner = "\n" if slot in ("DECLARE", "AFTER") else ", "
        held = joiner.join(lines)
        text = text.replace(slot, held if held or slot != "DECLARE" else "pass")
    namespace: dict[str, object] = {}
    exec(compile(text, "<pasted script>", "exec"), namespace)  # noqa: S102 - the test pastes the lines MIME printed
    return namespace["exp"]


def test_the_report_prints_each_item_with_its_lines():
    report = settling_ball().build().report
    text = report.pending()
    assert text.startswith("profile exploratory: 4 items pending")
    for target in BARE_PENDING:
        assert target in text
    assert 'accept={"fluid.dynamic_viscosity": Acceptance(covers=("condition_unstated:shear_rate",), ' \
           'reason=pending())}' in text
    assert "reason: {pending: true}" in text
    assert "built.report.pending()" in report.text()


def test_the_pasted_python_lines_build_and_sealed_build_in_research():
    items = settling_ball().plan().profile.pending
    plans = pasted(items).plan()
    assert sorted(item.target for item in plans.profile.pending) == sorted(BARE_PENDING)
    assert not [w for w in plans.profile.waivers if w.generated], "every item is now the experiment's own"
    sealed = pasted(items, reason='"a reason stated in review"', profile="research").plan()
    assert sealed.profile.pending == ()


def test_the_control_an_item_left_out_is_the_profile_s_again_and_refused_in_research():
    items = settling_ball().plan().profile.pending
    kept = [item for item in items if item.target != "drag:ball.spacing"]
    plans = pasted(kept).plan()
    assert "drag:ball.spacing" in [w.target for w in plans.profile.waivers if w.generated]
    with pytest.raises(BuildRefused) as refused:
        pasted(kept, reason='"a reason"', profile="research").plan()
    assert refused.value.codes() == ["setting.numerical-default"]


def merged(document: object, fragment: object) -> object:
    """``fragment`` merged into ``document``: mappings by key, lists of mappings by their ``id`` or ``segment``."""
    if isinstance(document, dict) and isinstance(fragment, dict):
        return {**document, **{key: merged(document[key], value) if key in document else value
                               for key, value in fragment.items()}}
    if isinstance(document, list) and isinstance(fragment, list):
        out = list(document)
        for item in fragment:
            key = next((k for k in ("id", "segment") if k in item), None)
            index = next((i for i, held in enumerate(out) if key and isinstance(held, dict)
                          and held.get(key) == item[key]), None)
            if index is None:
                out.append(item)
            else:
                out[index] = merged(out[index], item)
        return out
    return fragment


def sealed_text(document: object, reason: str) -> object:
    if document == dict(PENDING_YAML):
        return reason
    if isinstance(document, dict):
        return {key: sealed_text(value, reason) for key, value in document.items()}
    if isinstance(document, list):
        return [sealed_text(value, reason) for value in document]
    return document


def test_the_document_fragments_merged_in_build_and_sealed_build_in_research(tmp_path: Path):
    built = settling_ball().build()
    built.save(tmp_path / "saved")
    document = yaml.safe_load((tmp_path / "saved" / DOCUMENT_FILE).read_text())
    for item in built.report.profile.pending:
        document = merged(document, json.loads(json.dumps(document_fragment(item))))
    folder = tmp_path / "saved"
    (folder / DOCUMENT_FILE).write_text(yaml.safe_dump(document, sort_keys=False))
    plans = Experiment.from_document(folder / DOCUMENT_FILE).plan()
    assert sorted(item.target for item in plans.profile.pending) == sorted(BARE_PENDING)
    assert not [w for w in plans.profile.waivers if w.generated]
    document = {**sealed_text(document, "a reason stated in review"), "profile": "research"}
    (folder / DOCUMENT_FILE).write_text(yaml.safe_dump(document, sort_keys=False))
    assert Experiment.from_document(folder / DOCUMENT_FILE).plan().profile.pending == ()
