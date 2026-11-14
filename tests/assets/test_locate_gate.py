"""``locate`` is a gate: the first use of a gated file is asked where a build reads it, as well as where it is fetched.

Bytes a person copies into ``MIME_ASSETS_DIR`` (which a cluster fills by hand) are recorded as fully checked by the
resolver (``resolve`` adopts a placed file before it asks about the licence) and by ``mime assets check``; ``locate``,
which every build calls, once returned them with no acceptance on record. These tests hold:

- both routes to a recorded verification, a refused ``resolve`` and ``assets.check``, then ``locate``: refused
  (``asset.licence-not-accepted``) where nobody can be asked;
- a build with nobody to ask passes where the acceptance is on record (``licences.json``, or named for the batch with
  ``approved([], accept_licences=...)``), for that licence and no other;
- a person at a terminal is asked once, and a yes is remembered; a no is declined and nothing is returned;
- an open licence asks nothing.

**Controls**, measured against ``locate`` itself and each failing on every run (the rates are in the PR): the gate removed
(the two routes' refusals, the batch's, and the terminal's question all fail), and the gate made to refuse everything
(the tests of a build with the acceptance on record fail).
"""
import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from assets_rig import Rig

from mime import assets
from mime.assets import AssetRef
from mime.assets._acceptances import remember
from mime.build import BuildRefused
from mime.terms import CustomLicence

LICENCE = CustomLicence("LicenseRef-Maker-NC", "Maker Licence 1.1", "https://example.org/licence", "not_permitted",
                        ("non-commercial use only",))
OTHER = CustomLicence("LicenseRef-Other", "Other Licence 2.0", "https://example.org/other", "not_permitted",
                      ("for teaching only",))
DATA = b"a closed scan" * 100


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    held = Rig(tmp_path, monkeypatch)
    monkeypatch.setenv("MIME_OFFLINE", "1")
    monkeypatch.setenv("MIME_LICENCE_ACCEPTANCES", str(tmp_path / "licences.json"))
    held.cache.mkdir()
    yield held
    held.server.close()


def gated(licence: CustomLicence = LICENCE, name: str = "scan.bin", data: bytes = DATA) -> AssetRef:
    return AssetRef(name, hashlib.sha256(data).hexdigest(), len(data), licence.id, (f"https://example.org/{name}",),
                    licences=(licence,))


def placed(rig: Rig, ref: AssetRef, data: bytes = DATA) -> AssetRef:
    """``ref``'s bytes copied into the cache by hand, and the full verification recorded for them as a refused resolve
    leaves it (``assets.check`` writes the same record)."""
    (rig.cache / ref.sha256).write_bytes(data)
    with assets.asking(None):
        assert assets.check(ref) == "ok"
    return ref


class Yes:
    """An asker that answers the licence question as told and counts the questions."""

    def __init__(self, answer: bool = True) -> None:
        self.answer, self.asked = answer, 0

    def licences(self, licences: Sequence[object], notice: str | None) -> bool:
        self.asked += 1
        return self.answer

    def confirm(self, question: str) -> bool:
        return True

    def line(self, text: str) -> None:
        pass


@pytest.mark.parametrize("route", ["a refused resolve", "mime assets check"])
def test_locate_refuses_a_gated_record_no_gated_call_put_here(rig: Rig, route: str):
    ref = gated()
    (rig.cache / ref.sha256).write_bytes(DATA)
    with assets.asking(None):
        if route == "a refused resolve":
            with pytest.raises(BuildRefused) as refused:
                assets.resolve(ref)
            assert refused.value.codes() == ["asset.licence-not-accepted"]
        else:
            assert assets.check(ref) == "ok"
        with pytest.raises(BuildRefused) as located:
            assets.locate(ref)
    assert located.value.codes() == ["asset.licence-not-accepted"]
    assert LICENCE.id in located.value.refusals[0].action


def test_a_build_with_nobody_to_ask_passes_where_the_acceptance_is_on_record_for_that_licence_only(rig: Rig):
    ref, other = placed(rig, gated()), placed(rig, gated(OTHER, "other.bin", b"another" * 50), b"another" * 50)
    remember(list(assets.licences_of(ref)))
    with assets.asking(None):
        assert assets.locate(ref).read_bytes() == DATA
        with pytest.raises(BuildRefused) as refused:
            assets.locate(other)
    assert refused.value.codes() == ["asset.licence-not-accepted"] and OTHER.id in refused.value.refusals[0].action


def test_a_licence_named_for_the_batch_is_accepted_for_the_builds_inside_it_and_nothing_after(rig: Rig):
    ref = placed(rig, gated())
    with assets.asking(None):
        with assets.approved([], accept_licences={LICENCE.id}):
            assert assets.locate(ref).is_file()
        with pytest.raises(BuildRefused):
            assets.locate(ref)
    assert assets.read_acceptances() == []


def test_a_person_at_a_terminal_is_asked_once_and_the_yes_is_remembered(rig: Rig):
    ref, asker = placed(rig, gated()), Yes()
    with assets.asking(asker):  # type: ignore[arg-type]
        assert assets.locate(ref).is_file()
        assert assets.locate(ref).is_file()
    assert asker.asked == 1
    assert [each.licence.id for each in assets.read_acceptances()] == [LICENCE.id]


def test_a_no_at_the_terminal_returns_nothing(rig: Rig):
    ref, asker = placed(rig, gated()), Yes(False)
    with assets.asking(asker), pytest.raises(BuildRefused) as refused:  # type: ignore[arg-type]
        assets.locate(ref)
    assert refused.value.codes() == ["asset.licence-declined"] and assets.read_acceptances() == []


def test_an_open_licence_asks_nothing_and_a_missing_file_is_still_missing_first(rig: Rig):
    ref = AssetRef("open.bin", hashlib.sha256(DATA).hexdigest(), len(DATA), "CC0-1.0", ("https://example.org/open.bin",))
    placed(rig, ref)
    asker = Yes()
    with assets.asking(asker):  # type: ignore[arg-type]
        assert assets.locate(ref).is_file()
        with pytest.raises(BuildRefused) as absent:
            assets.locate(gated(LICENCE, "absent.bin", b"nothing here"))
    assert asker.asked == 0 and absent.value.codes() == ["asset.missing"]
