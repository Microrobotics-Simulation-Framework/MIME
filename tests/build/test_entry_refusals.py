"""A refusal of the user's or the environment's class that MIME's own API raises inside an entry's ``plan`` or ``build``
is that refusal, with its own code, class and words, and the CLI gives it exit 2; it is not the entry's defect.

What stays the entry's defect (``entry.defect``, internal, exit 3) is everything else the entry raises: a plain
exception, a MIME error of the internal class (MIME's own defect, met while a lab's entry called it), and a MIME error
holding both a user's refusal and an internal one. An internal MIME error inside an entry shows its own code, words and
docs link as plain lines under the headline, never as an escaped ``repr``.

The seams: an entry's ``build``, its ``plan``, a default's rule (``numerical_default(..., by=...)``), and the stage's
drawing of a geometry kind. The licence gate's refusal (``asset.licence-not-accepted``, from ``mime.assets.locate``) is
the user's refusal as a lab's entry meets it: a gated file in the cache, no acceptance on record, nobody to ask. The
controls, which stay defects, fail a pass-through that passes everything, one that passes when any refusal is the user's,
and (the licence tests) one that passes nothing; the rates of each mutant are in the PR.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mime import assets
from mime.assets import AssetRef
from mime.build import BuildRefused, InternalError, Refusal, RefusalClass
from mime.build._stage import stage_spec
from mime import Experiment, Scene, Structure, cli
from mime.gradients import GradientStatus
from mime.registry import (
    Delivery,
    Interaction,
    Plan,
    Read,
    RoleKind,
    numerical_default,
    register_interaction,
)
from mime.terms import CustomLicence
from mime.scene import Tube, Z
from mime.testing.toys import LOAD, POSITION, ToySpringNode, coupled_toy
from mime.units import mm, ms, standard_gravity
from mime.vocabulary import register_namespace

register_namespace("entryrefusal", "0.0.1")

LICENCE = CustomLicence("LicenseRef-Maker-NC", "Maker Licence 1.1", "https://example.org/licence", "not_permitted",
                        ("non-commercial use only",))
DATA = b"a closed scan" * 100
GATED = AssetRef("scan.bin", hashlib.sha256(DATA).hexdigest(), len(DATA), LICENCE.id, ("https://example.org/scan.bin",),
                 licences=(LICENCE,))


@dataclass(frozen=True, kw_only=True)
class _Raising(Interaction):
    """A lab's interaction, otherwise sound, that does ``act`` at the seam it is named for."""

    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    act: ClassVar[Any] = staticmethod(lambda: None)

    def build(self, ctx, b):
        node = b.add_response_node(ToySpringNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class RaisesInBuild(_Raising):
    name = "entryrefusal.raises_in_build"

    def build(self, ctx, b):
        type(self).act()
        super().build(ctx, b)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class RaisesInPlan(_Raising):
    name = "entryrefusal.raises_in_plan"

    def plan(self, ctx):
        type(self).act()
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class RaisesInRule(_Raising):
    name = "entryrefusal.raises_in_rule"
    setting_roles = {"gain": numerical_default("a rule of the test's", "a basis of the test's", by="default_gain")}
    gain: float | None = None

    def default_gain(self, ctx):
        type(self).act()
        return 1.0


ENTRIES = [RaisesInBuild, RaisesInPlan, RaisesInRule]
SEAMS = ["its build", "its plan", "its default's rule"]


@pytest.fixture
def nobody_to_ask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A gated file placed in the cache and recorded as verified, with no acceptance on record and nobody to ask."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MIME_ASSETS_DIR", str(cache))
    monkeypatch.setenv("MIME_OFFLINE", "1")
    monkeypatch.setenv("MIME_LICENCE_ACCEPTANCES", str(tmp_path / "licences.json"))
    (cache / GATED.sha256).write_bytes(DATA)
    with assets.asking(None):
        assert assets.check(GATED) == "ok"
        yield


def locate_the_gated_file() -> None:
    assets.locate(GATED)


INTERNAL = Refusal("lag.unplanned", RefusalClass.INTERNAL, "a MIME check failed", "report it to MIME")
USER = Refusal("value.not-positive", RefusalClass.USER, "a radius of zero", "give it above zero")


def internal_error() -> None:
    raise InternalError([INTERNAL])


def mixed_refusals() -> None:
    raise BuildRefused([USER, INTERNAL])


def plain_exception() -> None:
    raise ValueError("a lab's own slip")


def user_refusal() -> None:
    raise BuildRefused([USER])


def toy_with(entry: type[_Raising]) -> Experiment:
    experiment = coupled_toy()
    experiment.attach(entry(), body="probe")
    return experiment


def refused_when(entry: type[_Raising], act, monkeypatch: pytest.MonkeyPatch) -> BuildRefused:
    monkeypatch.setattr(entry, "act", staticmethod(act))
    with pytest.raises(BuildRefused) as refused:
        toy_with(entry).build()
    return refused.value


def stage_refused_when(act, monkeypatch: pytest.MonkeyPatch) -> BuildRefused:
    """The stage's drawing of a geometry kind raising what ``act`` raises."""
    experiment = Experiment("stage", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(1))
    experiment.add_structure(Structure("held", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(1.0), wall=mm(0.5),
                                                              length=mm(10.0))))
    plans = experiment.plan()
    monkeypatch.setattr(Tube, "drawing", lambda self, settings, segments: act())
    with pytest.raises(BuildRefused) as refused:
        stage_spec(plans)
    return refused.value


CLI_SEAMS = ["build", "plan"]


def run_from_the_command_line(seam: str, act, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> tuple[int, str]:
    """A saved experiment that uses the lab fixture's ``labx.nudge`` (a saved folder can only name an installed
    package's entry), rebuilt by ``mime run`` with its ``build`` or ``plan`` raising what ``act`` raises."""
    import mime_lab_fixture as lab

    experiment = coupled_toy()
    experiment.attach(lab.Nudge(), body="probe")
    experiment.build().save(tmp_path / "saved", steps=2)
    monkeypatch.setattr(lab.Nudge, seam, lambda self, *arguments: act())
    capsys.readouterr()
    code = cli.main(["run", str(tmp_path / "saved"), "--duration", "0.002", "--out", str(tmp_path / "out")])
    return code, capsys.readouterr().err


@pytest.mark.parametrize("entry", ENTRIES, ids=SEAMS)
def test_the_licence_refusal_inside_an_entry_is_the_refusal_not_the_entrys_defect(entry, nobody_to_ask, monkeypatch):
    error = refused_when(entry, locate_the_gated_file, monkeypatch)
    (held,) = error.refusals
    assert held.code == "asset.licence-not-accepted" and held.category is RefusalClass.USER, error
    assert LICENCE.id in held.action and "scan.bin" in held.message, held
    assert entry.name in held.message and "entry.defect" not in str(error), "the entry is named as where, not as the culprit"


def test_the_licence_refusal_inside_the_stages_drawing_is_the_refusal_too(nobody_to_ask, monkeypatch):
    (held,) = stage_refused_when(locate_the_gated_file, monkeypatch).refusals
    assert held.code == "asset.licence-not-accepted" and held.category is RefusalClass.USER, held
    assert "(while drawing the stage of " in held.message


@pytest.mark.parametrize("seam", CLI_SEAMS)
def test_the_licence_refusal_inside_an_entry_is_exit_two_from_mime_run(seam, nobody_to_ask, tmp_path, monkeypatch,
                                                                       capsys):
    code, shown = run_from_the_command_line(seam, locate_the_gated_file, tmp_path, monkeypatch, capsys)
    assert code == cli.REFUSED, shown
    assert "[user] asset.licence-not-accepted" in shown and "entry.defect" not in shown, shown
    assert "(while " in shown


@pytest.mark.parametrize("act", [plain_exception, internal_error, mixed_refusals],
                         ids=["a plain exception", "an internal MIME error", "a user's refusal and an internal one"])
@pytest.mark.parametrize("entry", ENTRIES, ids=SEAMS)
def test_control_what_is_not_wholly_the_users_refusal_inside_an_entry_is_still_its_defect(entry, act, monkeypatch):
    (held,) = refused_when(entry, act, monkeypatch).refusals
    assert held.code == "entry.defect" and held.category is RefusalClass.INTERNAL, held


@pytest.mark.parametrize("act", [plain_exception, internal_error, mixed_refusals],
                         ids=["a plain exception", "an internal MIME error", "a user's refusal and an internal one"])
def test_control_the_stages_drawing_raising_what_is_not_wholly_the_users_is_still_the_kinds_defect(act, monkeypatch):
    assert [r.code for r in stage_refused_when(act, monkeypatch).refusals] == ["entry.defect"]


@pytest.mark.parametrize("act", [plain_exception, internal_error, mixed_refusals],
                         ids=["a plain exception", "an internal MIME error", "a user's refusal and an internal one"])
@pytest.mark.parametrize("seam", CLI_SEAMS)
def test_control_the_same_from_mime_run_is_exit_three(seam, act, tmp_path, monkeypatch, capsys):
    code, shown = run_from_the_command_line(seam, act, tmp_path, monkeypatch, capsys)
    assert code == cli.FAILED, shown
    assert "[internal] entry.defect" in shown, shown


@pytest.mark.parametrize("act", [internal_error, mixed_refusals], ids=["an internal MIME error", "a mixed one"])
@pytest.mark.parametrize("seam", [*SEAMS, "the stage's drawing"])
def test_an_internal_mime_error_inside_an_entry_shows_its_own_code_words_and_link_as_plain_lines(seam, act, monkeypatch):
    if seam == "the stage's drawing":
        error = stage_refused_when(act, monkeypatch)
    else:
        error = refused_when(ENTRIES[SEAMS.index(seam)], act, monkeypatch)
    (held,) = error.refusals
    assert "[internal] lag.unplanned: a MIME check failed\n" in held.message, held.message
    assert "see: " in held.message and "lag.unplanned" in held.message.split("see: ")[-1], held.message
    assert "\\n" not in held.message and "BuildRefused(" not in held.message and "InternalError(" not in held.message
    assert "raised" in held.message, "the headline says it was raised"
