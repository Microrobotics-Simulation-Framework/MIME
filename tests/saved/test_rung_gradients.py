"""Each rung's own gradient status (``rung_gradients``): what a user differentiates through is the rung they attached.
The lock's resolved record holds the attached rung's status, so a saved experiment whose drag is the MRS
rung records ``exact`` and one whose drag is the surrogate records ``piecewise_jump``, as before; an entry that
declares no rung of its own records its own status, as before. The registry refuses a declaration whose rungs are not
the entry's, whose entry status is not the least of its rungs', or whose switches are not the entry's; the docs check
holds each rung's status to its own switches.

The controls: a lock that writes the entry's status for every rung records ``piecewise_jump`` for the mrs rung; each
malformed declaration is refused naming what is wrong; and a rung declared ``exact`` beside switches fails the docs
check."""

import dataclasses
import json
from dataclasses import dataclass

import pytest
from mrs_cases import BALL, mrs, free_space

import mime.saved._record as saved_record
from mime.core import switch
from mime.data import asset
from mime.docs import read_list
from mime.docs._check import LISTS, Report, switch_problems
from mime.gradients import GradientStatus
from mime.interactions import Drag
from mime.registry import EntryKind, GradientRecord, Registered, RungGradient, register_interaction
from mime.saved import LOCK_FILE
from mime.testing import HYDRODYNAMIC
from mime.testing.toys import ToyDrag
from mime.units import mm
from mime.vocabulary import register_namespace

register_namespace("rungs", "0.0.1")


def recorded_status(experiment, folder, owner: str) -> str:
    """The status the saved lock's resolved record holds for ``owner``, read back from the file."""
    experiment.save(folder, steps=1)
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))["resolved"]["entries"][owner][
        "differentiability"]


def mrs_only():
    return free_space("mrs_only", {"ball": BALL}, mrs(mm(1.5)))


def test_a_mrs_only_experiments_lock_records_the_mrs_rungs_status(tmp_path):
    assert recorded_status(mrs_only(), tmp_path, "drag:ball") == "exact"


def test_a_surrogate_experiments_lock_records_the_surrogate_rungs_status_and_a_rungless_entry_its_own(tmp_path):
    HYDRODYNAMIC.rig(Drag.Surrogate(weights=asset("dejongh_mlp_v2"))).save(tmp_path, steps=1)
    entries = json.loads((tmp_path / LOCK_FILE).read_text(encoding="utf-8"))["resolved"]["entries"]
    assert entries["drag:probe"]["differentiability"] == "piecewise_jump"
    assert entries["probe"]["differentiability"] == "exact"


def test_control_a_lock_that_writes_the_entrys_status_for_every_rung_records_the_mrs_rung_wrongly(tmp_path,
                                                                                                   monkeypatch):
    monkeypatch.setattr(saved_record, "rung_gradient", lambda cls, rung: RungGradient(
        differentiability=cls.differentiability, gradient=cls.gradient))
    assert recorded_status(mrs_only(), tmp_path, "drag:ball") == "piecewise_jump"


CLAMP = switch.Declaration(owner="rungs.clamp", indicator="the offset less the clamp", route="a smooth saturation",
                           orders={"load": 0})


def declared(**class_vars) -> type:
    """``ToyDrag`` (rungs ``quadratic`` and ``linear``) under a name of its own, with ``class_vars`` in place; its
    module MIME's, as the switch check reads a switch's package from it."""
    return dataclass(frozen=True, kw_only=True)(type("Rungs", (ToyDrag,), {"name": "rungs.toy",
                                                                           "__module__": "mime.planted", **class_vars}))


@pytest.mark.parametrize(("class_vars", "said"), [
    ({"rung_gradients": {"quadratic": RungGradient(differentiability=GradientStatus.EXACT)}},
     "rung_gradients names the rungs ['quadratic'], and it builds ['linear', 'quadratic']"),
    ({"rung_gradients": {"quadratic": RungGradient(differentiability=GradientStatus.EXACT),
                         "linear": RungGradient(differentiability=GradientStatus.PIECEWISE_KINK,
                                                gradient=GradientRecord(switches=(CLAMP,)))},
      "gradient": GradientRecord(switches=(CLAMP,))},
     "it is declared exact, and the least of its rungs' statuses is piecewise_kink"),
    ({"differentiability": GradientStatus.PIECEWISE_KINK,
      "rung_gradients": {"quadratic": RungGradient(differentiability=GradientStatus.EXACT),
                         "linear": RungGradient(differentiability=GradientStatus.PIECEWISE_KINK,
                                                gradient=GradientRecord(switches=(CLAMP,)))}},
     "its rungs declare the switches ['rungs.clamp'], and its gradient record []"),
])
def test_a_rung_declaration_that_is_not_the_entrys_is_refused_at_registration(class_vars, said):
    with pytest.raises(TypeError, match="cannot register") as refused:
        register_interaction(declared(**class_vars))
    assert said in str(refused.value)


def test_control_a_rung_declared_exact_beside_switches_fails_the_docs_check(tmp_path):
    good = {"differentiability": GradientStatus.PIECEWISE_KINK, "gradient": GradientRecord(switches=(CLAMP,)),
            "rung_gradients": {"quadratic": RungGradient(differentiability=GradientStatus.EXACT),
                               "linear": RungGradient(differentiability=GradientStatus.PIECEWISE_KINK,
                                                      gradient=GradientRecord(switches=(CLAMP,)))}}
    wrong = dict(good, rung_gradients={**good["rung_gradients"], "linear": dataclasses.replace(
        good["rung_gradients"]["linear"], differentiability=GradientStatus.EXACT)})
    listed = tmp_path / "raw_switches.txt"
    listed.write_text("", encoding="utf-8")
    found = {}
    for label, class_vars in (("good", good), ("wrong", wrong)):
        cls = declared(**class_vars)
        report = Report()
        switch_problems([Registered(EntryKind.INTERACTION, cls.name, cls.version, cls)],
                        read_list(listed, LISTS["raw_switches"]), report)
        found[label] = [problem for problem in report.problems if "rungs.toy" in problem]
    assert found["good"] == []
    assert found["wrong"] == ["interaction rungs.toy 0.2.0 rung linear: it is declared exact, and its switches' lowest "
                              "order is 0 (rungs.clamp): it is piecewise_kink"]
