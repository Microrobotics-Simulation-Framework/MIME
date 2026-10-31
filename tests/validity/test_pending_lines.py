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
from mime.validity import PENDING_PYTHON, PENDING_YAML, also_line, document_fragment, pending_text, python_line

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


#: The walk's T1 and T4 as a new user writes them with no paperwork, each importing only what the walk's script
#: imported, with a slot for the imports the printed lines need and one where each kind of line is pasted.
T1_BARE = '''
from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.observations import SwimSpeed
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m_per_s, m_per_s2, mPa_s, rad_per_s, s, um, ms, mm_per_s
IMPORTS
exp = Experiment("t1", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05), profile=PROFILE)
water = exp.add_medium(Medium("water"))
exp.declare("water.density", kg_per_m3(998.0))
exp.declare("water.viscosity", mPa_s(1.0))
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0))
exp.attach(Drag.BEM(DRAG), body=ball, medium=water)
exp.attach(Gravity(), body=ball, medium=water)
exp.observe(SwimSpeed(ball, along=-Z, after=s(0.04)))
AFTER
'''
T4_BARE = '''
from mime import Apparatus, Body, Experiment, Medium, PermanentMagnet, State, Structure
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Drag, Lubrication, Magnetic
from mime.materials import Nominal, Record, packaged_record
from mime.observations import SwimSpeed
from mime.scene import Box, Scene, Screw, Tube, X, Z
from mime.things import Part
from mime.units import (A_m2, Hz, T, degC, dimensionless, kg_m2, kg_per_m3, m_per_s, m_per_s2, mm, ms, mT, rad_per_s,
                        s)
IMPORTS
FL9 = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33), starts=2)
exp = Experiment("t4", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(0.5), profile=PROFILE,
                 conditions={"temperature": degC(20.0, plus_minus=0.5)})
tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6), length=mm(100.0))))
water = exp.add_medium(Medium("water", fills=tube.lumen, material=Record(
    "water", packaged_record("openmaterica_example_water"), require={"test_method": ("reference_formulation",)},
    RECORD_water)))
n45 = Record("n45", packaged_record("openmaterica_example_ndfeb_n45"), nominal={"magnetic.remanence": Nominal(
    T(1.345), reason="the middle of the grade's limits: nothing measured places these magnets within them")}, RECORD_n45)
umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={
    "shell": Part(FL9),
    "magnet_a": Part(Box(size=mm((1.0, 1.0, 1.0))), material=n45, at=mm((-0.5, 0.0, 0.0)), inside="shell"),
    "magnet_b": Part(Box(size=mm((1.0, 1.0, 1.0))), material=n45, at=mm((0.5, 0.0, 0.0)), inside="shell"),
}, magnets={"magnet_a": PermanentMagnet(direction=(0.0, 1.0, 0.0)), "magnet_b": PermanentMagnet(direction=(0.0, 1.0, 0.0))}))
exp.declare("umr.parts.shell.density", kg_per_m3(1410.0))
exp.declare("umr.inertia", kg_m2((1.7158e-10, 4.2689e-10, 4.2689e-10)))
field = exp.add_apparatus(Apparatus("field", fields={
    "magnetic": UniformRotating(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
    commands={"frequency": Hz(5.0), "amplitude": mT(1.2)}))
exp.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15))), body=umr, medium=water)
exp.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
swim = exp.observe(SwimSpeed(umr, along=tube.axis, after=s(2.0)))
AFTER
'''


def printed_imports(items) -> list[str]:
    """The imports the formatter prints above the lines (``mime pending`` prints the same), read from its text."""
    if not items:
        return []
    prefix = "In Python, the lines need, at the top of the script: "
    (line,) = [held for held in pending_text("exploratory", items).splitlines() if held.startswith(prefix)]
    return line.removeprefix(prefix).removesuffix(".").split("; ")


def pasted_bare(script: str, items, reason: str = PENDING_PYTHON, profile: str = "exploratory") -> Experiment:
    """``script`` with the imports the formatter prints for ``items`` (read from its text) and each item's line pasted
    where it says: in place of its ``exp.declare``, into its material's ``Record(...)`` or the drag's ``Drag(...)``,
    or after the experiment is made."""
    text = script.replace("PROFILE", repr(profile)).replace("IMPORTS", "\n".join(printed_imports(items)))
    after: list[str] = []
    records: dict[str, list[str]] = {}
    drag: list[str] = []
    for item in items:
        line = python_line(item, reason)
        if item.where == "in place of its exp.declare(...)":
            (held,) = [held for held in text.splitlines() if held.startswith(f'exp.declare("{item.target}",')]
            text = text.replace(held, line)
        elif item.where.endswith("Record(...)"):
            records.setdefault(item.where.split("'")[1], []).append(line)
        elif item.where.endswith("Drag(...)"):
            drag.append(line)
        else:
            after.append(line)
        if item.also:
            after.append(also_line(item, reason))
    for name in ("water", "n45"):
        text = text.replace(f"RECORD_{name}", ", ".join(records.get(name, [])))
    text = text.replace("DRAG", ", ".join(drag)).replace("AFTER", "\n".join(after))
    namespace: dict[str, object] = {}
    exec(compile(text, "<pasted walk task>", "exec"), namespace)  # noqa: S102 - the test pastes the lines MIME printed
    return namespace["exp"]


@pytest.mark.parametrize(("script", "count"), [(T1_BARE, 5), (T4_BARE, 4)], ids=["T1", "T4"])
def test_the_walk_s_tasks_written_bare_reach_research_by_the_printed_lines_alone(script, count):
    """W1 and K3: each walk task, written with no paperwork, leaves one item per thing pending (T4's N45 read waives
    four reasons in one item, one acceptance); its printed lines, with the imports printed above them and nothing else
    added to a script that imports only what the walk's did, build in exploratory with the same items now the
    experiment's own, and sealed, in research with nothing pending."""
    bare = pasted_bare(script, [])
    items = bare.plan().profile.pending
    assert len(items) == count == len({item.target for item in items}), [item.target for item in items]
    if script is T4_BARE:
        (n45,) = [item for item in items if item.target == "material 'n45' magnetic.remanence"]
        assert len(n45.codes) == 4 and n45.python.count("Acceptance(") == 1
    own = pasted_bare(script, items).plan().profile
    assert sorted(item.target for item in own.pending) == sorted(item.target for item in items)
    assert not [w for w in own.waivers if w.generated]
    sealed = pasted_bare(script, items, reason='"a reason stated in review"', profile="research").plan()
    assert sealed.profile.pending == ()


def test_the_control_the_lines_without_their_imports_do_not_run():
    items = pasted_bare(T1_BARE, []).plan().profile.pending
    with pytest.raises(NameError, match="Pa_s"):
        pasted_bare(T1_BARE.replace("IMPORTS", "from mime.validity import pending"), items)


def test_a_saved_folder_is_sealed_by_its_document_and_built_from_it_in_research(tmp_path: Path):
    """K2: the sealing route for a saved folder: its document's fragments merged in, sealed, the profile raised;
    ``Experiment.load`` refuses the edited folder (its lock is the earlier build's), ``mime pending`` on the document
    exits 0, and ``Experiment.from_document`` builds it in research."""
    from mime.cli import main as mime_main

    built = settling_ball().build()
    built.save(tmp_path / "saved", steps=1)
    document = yaml.safe_load((tmp_path / "saved" / DOCUMENT_FILE).read_text())
    for item in built.report.profile.pending:
        document = merged(document, json.loads(json.dumps(document_fragment(item))))
    document = {**sealed_text(document, "a reason stated in review"), "profile": "research"}
    (tmp_path / "saved" / DOCUMENT_FILE).write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BuildRefused) as refused:
        Experiment.load(tmp_path / "saved")
    assert "load.input-hash" in refused.value.codes()
    assert "Experiment.from_document" in refused.value.refusals[0].action
    assert mime_main(["pending", str(tmp_path / "saved" / DOCUMENT_FILE)]) == 0
    sealed = Experiment.from_document(tmp_path / "saved" / DOCUMENT_FILE).build()
    assert sealed.report.profile.profile.value == "research" and sealed.report.profile.pending == ()
    assert "Experiment.from_document" in built.report.pending()
