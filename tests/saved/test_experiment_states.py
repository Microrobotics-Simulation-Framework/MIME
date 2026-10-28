"""The ``Experiment`` API as a state machine: Hypothesis writes random scripts of ``add_*``, ``declare``, ``attach``,
``observe``, ``build``, ``save`` and ``load`` calls over MIME's toy entries and script A's real ones, and three
invariants hold after every call:

- a build succeeds or refuses with reasons (``BuildRefused`` carrying at least one refusal, each with its code, its
  message and its action line), and never raises anything else; so does a save, which builds first;
- ``load(save(x))`` keeps the derived hash, the physics' and the observations';
- a refused build or save leaves the experiment as it was.

Every call is one a script can write, with values at the scales a body has: names from a small pool, so that calls
bind to things declared before or after them, to nothing, or twice; values at A's scale, scaled by a factor from 0.5
to 2; toy laws bound to real bodies and real laws to toy ones. Such scripts are often refused, which is what the
first and third invariants need.

The controls, planted here: a save whose document drops a setting breaks the second invariant, and a build that
changes the experiment before it refuses breaks the third. Each runs the same machine, at the same example counts,
and must be caught on every run.
"""

import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Phase, settings
from hypothesis import strategies as st
from hypothesis.errors import Flaky
from hypothesis.stateful import RuleBasedStateMachine, initialize, precondition, rule, run_state_machine_as_test
from script_a import EPSILON, EXTENT, INERTIA, LENGTH, MASS, MOMENT, NU, RADIUS, VISCOSITY, VOLUME, WATER, fl9, script_a

import mime.saved._folder as folder_module
from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused, Refusal
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.experiment import _experiment
from mime.fields import UniformRotatingField
from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Scene, Tube, X, Z
from mime.testing.toys import (
    PROBE_START,
    ToyBody,
    ToyCouple,
    ToyDrag,
    ToyField,
    ToyHold,
    ToyLift,
    ToyRecorder,
    ToyRelaxation,
    ToySpring,
    ToyTwist,
)
from mime.things import Part
from mime.units import Hz, m, m_per_s, m_per_s2, mm, ms, mT, rad_per_s, s

#: Hypothesis's defaults (100 programs of up to 50 calls), kept: each build constructs a graph, and each save also
#: runs its first steps, loads the folder back and builds again.
MACHINE = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

BODIES, MEDIA = ("umr", "probe"), ("water", "bath")
STEPS = (ms(0.5), ms(2.0), s(0.02))
VALUES = {"umr.mass": MASS, "umr.inertia": INERTIA, "umr.buoyant_volume": VOLUME, "umr.magnetic_moment": MOMENT,
          "umr.extent": EXTENT, "umr.centre_of_buoyancy": m((0.0, 0.0, 0.0)), "umr.helix_wavenumber": NU,
          "umr.length": LENGTH, "umr.radius": RADIUS, "water.density": WATER, "water.viscosity": VISCOSITY,
          "probe.mass": MASS}
LAWS = {
    "spring": lambda body, medium: (ToySpring(), {"body": body}),
    "toy drag, linear": lambda body, medium: (ToyDrag.Linear(), {"body": body}),
    "toy drag, quadratic": lambda body, medium: (ToyDrag.Quadratic(), {"body": body}),
    "lift": lambda body, medium: (ToyLift(), {"body": body, "medium": medium}),
    "couple": lambda body, medium: (ToyCouple(), {"body": body, "medium": medium}),
    "hold": lambda body, medium: (ToyHold(), {"body": body, "field": "source.push"}),
    "twist": lambda body, medium: (ToyTwist(), {"body": body}),
    "torque": lambda body, medium: (Magnetic.Torque(), {"body": body, "field": "field.magnetic"}),
    "gravity": lambda body, medium: (Gravity(), {"body": body, "medium": medium}),
    "contact": lambda body, medium: (Contact.Projection(margin=mm(0.1)), {"body": body,
                                                                          "surface": "tube.inner_surface"}),
    "surrogate drag": lambda body, medium: (Drag.Surrogate(weights=asset("dejongh_mlp_v2"),
                                                           lubrication=Lubrication(epsilon=EPSILON)),
                                            {"body": body, "medium": medium}),
}
UNCHANGED = "a refused build changed the experiment"
#: What the calls came to, over every program of a run, so a run shows that it reached what the invariants are about.
OUTCOMES: Counter[str] = Counter()
KEPT = "load(save(x)) did not keep the derived hash"


def state_of(experiment: Experiment) -> tuple[object, ...]:
    """Everything an experiment holds, by identity and by value: its settings, each declaration, and what it was
    loaded from."""
    held = vars(experiment)
    return (repr(held["_settings"]), [(id(d), repr(d)) for d in held["_declarations"]], id(held["_saved"]),
            repr(held["load_report"]))


def assert_reasons(refused: BuildRefused) -> None:
    assert refused.refusals, "a refusal without a reason"
    for reason in refused.refusals:
        assert isinstance(reason, Refusal) and reason.code and reason.message and reason.action, reason


class ExperimentStates(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.folder = Path(tempfile.mkdtemp(prefix="mime-states-"))
        self.saves = 0
        self.changed = True
        self.unsaved = True
        self.saved: Path | None = None

    def teardown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def _change(self) -> None:
        self.changed = self.unsaved = True

    @initialize(start=st.sampled_from(["nothing", "the coupled toy", "script A"]), dt=st.sampled_from(STEPS))
    def begin(self, start: str, dt: object) -> None:
        if start == "script A":
            self.experiment = script_a("states", dt=dt)
            return
        self.experiment = Experiment("states", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt)
        if start == "the coupled toy":
            self.experiment.add_medium(Medium("bath", flow=ToyRelaxation()))
            self.experiment.add_body(Body("probe", dynamics=ToyBody(), initial=PROBE_START))
            self.experiment.attach(ToySpring(), body="probe")
            self.experiment.attach(ToyDrag.Quadratic(), body="probe")

    @rule(inner_radius=st.floats(2.5, 4.0))
    def add_tube(self, inner_radius: float) -> None:
        self.experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(inner_radius),
                                                                      wall=mm(1.6), length=mm(100.0))))
        self._change()

    @rule(name=st.sampled_from(MEDIA))
    def add_medium(self, name: str) -> None:
        medium = Medium("water", fills="tube.lumen") if name == "water" else Medium("bath", flow=ToyRelaxation())
        self.experiment.add_medium(medium)
        self._change()

    @rule(name=st.sampled_from(BODIES), along=st.floats(-20.0, 20.0))
    def add_body(self, name: str, along: float) -> None:
        if name == "umr":
            body = Body("umr", dynamics=RigidInertial(), initial=State(
                position=mm((along, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
                angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(fl9())})
        else:
            body = Body("probe", dynamics=ToyBody(), initial=PROBE_START)
        self.experiment.add_body(body)
        self._change()

    @rule(name=st.sampled_from(["field", "source"]), frequency=st.floats(1.0, 20.0), amplitude=st.floats(0.5, 5.0))
    def add_apparatus(self, name: str, frequency: float, amplitude: float) -> None:
        if name == "field":
            apparatus = Apparatus("field", fields={"magnetic": UniformRotatingField(
                rotation_axis="tube.axis", initial_direction=(0.0, 1.0, 0.0))},
                commands={"frequency": Hz(frequency), "amplitude": mT(amplitude)})
        else:
            apparatus = Apparatus("source", fields={"push": ToyField()})
        self.experiment.add_apparatus(apparatus)
        self._change()

    @rule(path=st.sampled_from(sorted(VALUES)), factor=st.floats(0.5, 2.0))
    def declare(self, path: str, factor: float) -> None:
        self.experiment.declare(path, VALUES[path] * factor, reason="drawn at A's scale")
        self._change()

    @rule(law=st.sampled_from(sorted(LAWS)), body=st.sampled_from(BODIES), medium=st.sampled_from(MEDIA))
    def attach(self, law: str, body: str, medium: str) -> None:
        entry, bindings = LAWS[law](body, medium)
        self.experiment.attach(entry, **bindings)
        self._change()

    @rule(kind=st.sampled_from(["swim speed", "recorder"]), body=st.sampled_from(BODIES))
    def observe(self, kind: str, body: str) -> None:
        observation = SwimSpeed(body, along="tube.axis") if kind == "swim speed" else ToyRecorder(body=body)
        self.experiment.observe(observation)
        self._change()

    @precondition(lambda self: self.changed)
    @rule()
    def build(self) -> None:
        self.changed = False
        before = state_of(self.experiment)
        try:
            self.experiment.build()
        except BuildRefused as refused:
            assert_reasons(refused)
            OUTCOMES["build refused"] += 1
        else:
            OUTCOMES["build built"] += 1
        assert state_of(self.experiment) == before, UNCHANGED

    @precondition(lambda self: self.unsaved)
    @rule()
    def save(self) -> None:
        self.unsaved = False
        before = state_of(self.experiment)
        folder = self.folder / f"saved_{self.saves}"
        self.saves += 1
        try:
            lock = self.experiment.save(folder)
        except BuildRefused as refused:
            assert_reasons(refused)
            assert state_of(self.experiment) == before, UNCHANGED
            OUTCOMES["save refused"] += 1
            return
        OUTCOMES["save round-tripped"] += 1
        assert state_of(self.experiment) == before, "a save changed the experiment"
        try:
            hashes = Experiment.load(folder).build().record.hashes()
        except BuildRefused as refused:
            raise AssertionError(f"{KEPT}: the load was refused: {refused}") from refused
        saved = lock["hashes"]
        assert (hashes["derived"], hashes["observations"]) == (saved["derived"], saved["observations"]), KEPT
        self.saved = folder

    @precondition(lambda self: self.saved is not None)
    @rule()
    def load(self) -> None:
        assert self.saved is not None
        self.experiment = Experiment.load(self.saved)
        self.changed = self.unsaved = True
        OUTCOMES["experiment replaced by its load"] += 1


def test_the_experiment_api_keeps_its_invariants_over_random_scripts():
    OUTCOMES.clear()
    run_state_machine_as_test(ExperimentStates, settings=MACHINE)
    print(f"\n{dict(OUTCOMES)}")
    assert OUTCOMES["build built"] and OUTCOMES["build refused"] and OUTCOMES["save round-tripped"]


#: The controls search as the property does, with Hypothesis's default example counts; they only skip shrinking, which
#: reports a counterexample and does not find one.
CONTROL = settings(MACHINE, phases=(Phase.explicit, Phase.reuse, Phase.generate), database=None)


def caught(expected: str) -> str:
    with pytest.raises((AssertionError, Flaky)) as failure:
        run_state_machine_as_test(ExperimentStates, settings=CONTROL)
    message = str(failure.value)
    assert expected in message, message
    return message


def test_a_save_that_drops_a_setting_breaks_the_hash_invariant(monkeypatch):
    written = folder_module.dump_document

    def dropping(document):
        held = dict(document)
        for i, entry in enumerate(held.get("interactions", [])):
            if entry.get("settings"):
                kept = dict(entry["settings"])
                kept.pop(next(iter(kept)))
                held["interactions"] = [*held["interactions"][:i], {**entry, "settings": kept},
                                        *held["interactions"][i + 1:]]
                break
        return written(held)

    monkeypatch.setattr(folder_module, "dump_document", dropping)
    print(caught(KEPT))


def test_a_build_that_changes_the_experiment_before_refusing_breaks_the_unchanged_invariant(monkeypatch):
    planned = _experiment.Experiment.plan

    def changing(self):
        held = vars(self)["_declarations"]
        if held:
            held.append(held[0])
        return planned(self)

    monkeypatch.setattr(_experiment.Experiment, "plan", changing)
    print(caught(UNCHANGED))
