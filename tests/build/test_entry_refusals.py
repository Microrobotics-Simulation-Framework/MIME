"""A refusal of the user's or the environment's class that MIME's own API raises inside an entry's ``plan`` or ``build``
is that refusal, with its own code, class and words, and the CLI gives it exit 2; it is not the entry's defect.

What stays the entry's defect (``entry.defect``, internal, exit 3) is everything else the entry raises: a plain
exception, and a MIME error of the internal class (MIME's own defect, met while a lab's entry called it).

The first test is the licence gate's refusal (``asset.licence-not-accepted``, from ``mime.assets.locate``) as a lab's
entry meets it: a gated file in the cache, no acceptance on record, nobody to ask. The two tests after it are its
controls, the refusals that stay defects; between them they fail on a pass-through that passes everything and on one that
passes nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from mime import assets
from mime.assets import AssetRef
from mime.build import ArgumentRefused, BuildRefused, InternalError, Refusal, RefusalClass
from mime.cli import FAILED, REFUSED
from mime.cli import refused as exit_code
from mime.gradients import GradientStatus
from mime.registry import Delivery, Interaction, Plan, Read, RoleKind, register_interaction
from mime.terms import CustomLicence
from mime.testing.toys import LOAD, POSITION, ToySpringNode, coupled_toy
from mime.vocabulary import register_namespace

register_namespace("entryrefusal", "0.0.1")

LICENCE = CustomLicence("LicenseRef-Maker-NC", "Maker Licence 1.1", "https://example.org/licence", "not_permitted",
                        ("non-commercial use only",))
DATA = b"a closed scan" * 100
GATED = AssetRef("scan.bin", hashlib.sha256(DATA).hexdigest(), len(DATA), LICENCE.id, ("https://example.org/scan.bin",),
                 licences=(LICENCE,))


@dataclass(frozen=True, kw_only=True)
class _Raising(Interaction):
    """A lab's interaction, otherwise sound, that does ``act`` in the phase it is named for."""

    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    act: ClassVar[object] = None


@register_interaction
@dataclass(frozen=True, kw_only=True)
class RaisesInBuild(_Raising):
    name = "entryrefusal.raises_in_build"

    def build(self, ctx, b):
        type(self).act()
        node = b.add_response_node(ToySpringNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class RaisesInPlan(_Raising):
    name = "entryrefusal.raises_in_plan"

    def plan(self, ctx):
        type(self).act()
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values))


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


def refused_when(entry: type[_Raising], act, monkeypatch: pytest.MonkeyPatch) -> BuildRefused:
    monkeypatch.setattr(entry, "act", staticmethod(act))
    experiment = coupled_toy()
    experiment.attach(entry(), body="probe")
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    return refused.value


def locate_the_gated_file() -> None:
    assets.locate(GATED)


def internal_error() -> None:
    raise InternalError([Refusal("lag.unplanned", RefusalClass.INTERNAL, "a MIME check failed", "report it to MIME")])


def plain_exception() -> None:
    raise ValueError("a lab's own slip")


@pytest.mark.parametrize("entry", [RaisesInBuild, RaisesInPlan], ids=["its build", "its plan"])
def test_the_licence_refusal_inside_an_entry_is_the_refusal_not_the_entrys_defect(entry, nobody_to_ask, monkeypatch,
                                                                                capsys):
    error = refused_when(entry, locate_the_gated_file, monkeypatch)
    (held,) = error.refusals
    assert held.code == "asset.licence-not-accepted" and held.category is RefusalClass.USER, error
    assert LICENCE.id in held.action and "scan.bin" in held.message, held
    assert entry.name in held.message and "entry.defect" not in str(error), "the entry is named as where, not as the culprit"
    assert exit_code(error, "build") == REFUSED
    assert "[internal]" not in capsys.readouterr().err


@pytest.mark.parametrize("entry", [RaisesInBuild, RaisesInPlan], ids=["its build", "its plan"])
def test_control_a_plain_exception_inside_an_entry_is_still_its_defect(entry, monkeypatch, capsys):
    error = refused_when(entry, plain_exception, monkeypatch)
    (held,) = error.refusals
    assert held.code == "entry.defect" and held.category is RefusalClass.INTERNAL and "a lab's own slip" in held.message
    assert exit_code(error, "build") == FAILED
    assert "[internal] entry.defect" in capsys.readouterr().err


@pytest.mark.parametrize("entry", [RaisesInBuild, RaisesInPlan], ids=["its build", "its plan"])
def test_control_an_internal_mime_error_inside_an_entry_is_still_its_defect(entry, monkeypatch):
    error = refused_when(entry, internal_error, monkeypatch)
    (held,) = error.refusals
    assert held.code == "entry.defect" and held.category is RefusalClass.INTERNAL
    assert "a MIME check failed" in held.message
    assert exit_code(error, "build") == FAILED


def test_a_user_refusal_of_a_kind_other_than_the_builds_passes_too(monkeypatch):
    """The pass-through is for every MIME error of the user's class, not the build's alone: an argument refused."""
    def refuses_an_argument() -> None:
        raise ArgumentRefused([Refusal("value.not-positive", RefusalClass.USER, "a radius of zero", "give it above zero")])

    (held,) = refused_when(RaisesInBuild, refuses_an_argument, monkeypatch).refusals
    assert held.code == "value.not-positive" and held.category is RefusalClass.USER
