"""The download gates (``mime.assets._gates``): the licence a download needs accepted, and the disk space it needs.

The licence gate lives in the library, so a script and a notebook meet it as the command line does. These tests hold:

- an open licence asks nothing, and a licence with conditions beyond credit is not downloaded until it is accepted: by
  a person at a terminal (asked once, remembered for the licence as it was accepted), or by name (``accept_licences``),
  never by ``yes`` or ``force``, and an acceptance of one licence accepts no other;
- without a terminal the gate refuses, naming the licence and ``--accept-licence <id>``, before a request is made;
- the disk gate, with ``shutil.disk_usage`` patched to little, middling and ample space: asked above 1 GB or a tenth of
  the free space, stopped under 2 GB or 5 % of the drive left; ``yes`` answers the question and ``force`` overrides the
  stop, each only its own;
- a download that fails part-way leaves nothing behind but the asset's lock file.

Each control fails on every run: the gate removed, ``yes`` or ``force`` accepting a licence, an acceptance stored under
the wrong id, the stop's thresholds ten times off. ``MUTANTS`` lists them; ``test_every_mutant_is_caught_on_every_run``
plants each, three times, and fails if the scenario that should notice passes. The tests download nothing from the network: a server on this machine, as the assets tests
use."""

from __future__ import annotations

import collections
import errno
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from assets_rig import Rig

from mime import assets
from mime.assets import AssetRef, Licence, _acceptances, _gates
from mime.build import BuildRefused
from mime.terms import CustomLicence

CLOSED = "LicenseRef-Maker-NonCommercial"
OTHER = "LicenseRef-Other-Maker"
LINES = ("non-commercial use only", "no redistribution in any form")
DATA = b"a mesh" * 300
GB = 10**9

Usage = collections.namedtuple("Usage", "total used free")


def custom(identifier: str = CLOSED, *, title: str = "Maker Licence 1.1", lines: Sequence[str] = LINES) -> CustomLicence:
    return CustomLicence(identifier, title, "https://example.org/licence", "not_permitted", tuple(lines))


def closed(rig: Rig, name: str = "mesh.stl", payload: bytes = DATA, licence: CustomLicence | None = None,
           ) -> AssetRef:
    """A record under a custom licence with conditions, served by the rig's server."""
    held = licence or custom()
    rig.server.files[f"/{name}"] = payload
    import hashlib

    return AssetRef(name, hashlib.sha256(payload).hexdigest(), len(payload), held.id, (rig.server.url(f"/{name}"),),
                    licences=(held,), role="presentation")


def open_record(rig: Rig, name: str = "open.bin", payload: bytes = b"open" * 100) -> AssetRef:
    import hashlib

    rig.server.files[f"/{name}"] = payload
    return AssetRef(name, hashlib.sha256(payload).hexdigest(), len(payload), "CC0-1.0",
                    (rig.server.url(f"/{name}"),))


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    held = Rig(tmp_path, monkeypatch)
    monkeypatch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "none.toml"))
    (tmp_path / "none.toml").write_text("", encoding="utf-8")
    yield held
    held.server.close()


class Scripted:
    """An asker that answers as told and keeps what it was asked and told."""

    def __init__(self, *, licence: bool = True, confirm: bool = True) -> None:
        self.licence_answer, self.confirm_answer = licence, confirm
        self.asked: list[tuple[str, ...]] = []
        self.questions: list[str] = []
        self.lines: list[str] = []
        self.notices: list[str | None] = []

    def licences(self, licences: Sequence[Licence], notice: str | None) -> bool:
        self.asked.append(tuple(each.id for each in licences))
        self.notices.append(notice)
        return self.licence_answer

    def confirm(self, question: str) -> bool:
        self.questions.append(question)
        return self.confirm_answer

    def line(self, text: str) -> None:
        self.lines.append(text)


def codes(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def allowed(action: Callable[[], object]) -> object:
    """What ``action`` returns; a refusal is an assertion that failed (the scenarios that judge a mutant say so)."""
    try:
        return action()
    except BuildRefused as refused:
        raise AssertionError(f"refused although it should go ahead: {refused.codes()}") from None


def refusal_text(action: Callable[[], object]) -> str:
    try:
        action()
    except BuildRefused as refused:
        return str(refused)
    raise AssertionError("not refused")


@contextmanager
def space(free: int, total: int = 500 * GB) -> Iterator[None]:
    """The drive holding the cache has ``free`` of ``total`` bytes."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("shutil.disk_usage", lambda path: Usage(total, total - free, free))
        yield


# --- which licences are asked about ------------------------------------------------------------------------------------


@pytest.mark.parametrize("licence", ["CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "MIT OR CC-BY-NC-4.0"])
def test_a_licence_with_no_condition_beyond_credit_asks_nothing(licence: str):
    ref = AssetRef("a.bin", "ab" * 32, 10, licence, ("https://example.org/a",))
    assert assets.licences_of(ref) == ()


def test_a_creative_commons_licence_with_elements_states_each_in_words():
    ref = AssetRef("a.bin", "ab" * 32, 10, "CC-BY-NC-SA-4.0", ("https://example.org/a",))
    (found,) = assets.licences_of(ref)
    assert found.id == "CC-BY-NC-SA-4.0" and found.url == "https://spdx.org/licenses/CC-BY-NC-SA-4.0.html"
    assert found.conditions == ("non-commercial use only", "anything made from it must be shared under the same licence")


def test_a_custom_licence_states_its_own_condition_lines_and_one_without_says_so():
    stated = AssetRef("a.bin", "ab" * 32, 10, CLOSED, ("https://example.org/a",), licences=(custom(),))
    (found,) = assets.licences_of(stated)
    assert (found.id, found.title, found.conditions) == (CLOSED, "Maker Licence 1.1", LINES)
    bare = AssetRef("a.bin", "ab" * 32, 10, CLOSED, ("https://example.org/a",), licences=(custom(lines=()),))
    (silent,) = assets.licences_of(bare)
    assert silent.conditions[0].startswith("it may not be redistributed") and "read the licence" in silent.conditions[1]
    unnamed = AssetRef("a.bin", "ab" * 32, 10, CLOSED, ("https://example.org/a",))
    assert "does not describe this licence" in assets.licences_of(unnamed)[0].conditions[0]


# --- the licence gate ----------------------------------------------------------------------------------------------------


def test_an_open_licence_is_downloaded_with_no_question(rig: Rig):
    ref = open_record(rig)
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(ref)
    assert asker.asked == [] and assets.is_cached(ref)


def test_without_a_terminal_the_gate_refuses_naming_the_licence_and_the_flag_before_a_request(rig: Rig):
    ref = closed(rig)
    with assets.asking(None):
        text = refusal_text(lambda: assets.fetch(ref))
        assert codes(lambda: assets.fetch(ref)) == ["asset.licence-not-accepted"]
    assert f"--accept-licence {CLOSED}" in text and "Maker Licence 1.1" in text and LINES[0] in text
    assert rig.requests == 0 and not assets.is_cached(ref)


def test_the_flag_accepts_the_licence_it_names_for_the_call_and_nothing_is_remembered(rig: Rig):
    ref = closed(rig)
    with assets.asking(None):
        assets.fetch(ref, accept_licences={CLOSED})
        assert assets.is_cached(ref) and _acceptances.read_acceptances() == []
        other = closed(rig, "second.stl", b"second" * 100)
        assert codes(lambda: assets.fetch(other)) == ["asset.licence-not-accepted"], "the flag was for one call"


def test_an_acceptance_of_one_licence_accepts_no_other(rig: Rig):
    first, second = closed(rig), closed(rig, "second.stl", b"second" * 100, custom(OTHER, title="Other Licence 2"))
    with assets.asking(None):
        assert codes(lambda: assets.fetch(second, accept_licences={CLOSED})) == ["asset.licence-not-accepted"]
        assert OTHER in refusal_text(lambda: assets.fetch(second, accept_licences={CLOSED}))
        assets.fetch(first, accept_licences={CLOSED})
        assets.fetch(second, accept_licences={OTHER})
    assert rig.requests == 2 and assets.is_cached(first) and assets.is_cached(second)


def test_a_person_at_a_terminal_is_asked_once_and_the_yes_is_remembered(rig: Rig):
    ref, again = closed(rig), closed(rig, "second.stl", b"second" * 100)
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(ref)
        assert asker.asked == [(CLOSED,)] and assets.is_cached(ref)
        assets.fetch(again)
    assert asker.asked == [(CLOSED,)], "the second download under the licence asks nothing"
    (kept,) = _acceptances.read_acceptances()
    assert kept.licence == Licence(CLOSED, "Maker Licence 1.1", "https://example.org/licence", LINES) and kept.date


def test_a_no_downloads_nothing_and_refuses_saying_so(rig: Rig):
    ref = closed(rig)
    asker = Scripted(licence=False)
    with assets.asking(asker):
        assert codes(lambda: assets.fetch(ref)) == ["asset.licence-declined"]
    assert rig.requests == 0 and not assets.is_cached(ref) and _acceptances.read_acceptances() == []


def test_a_licence_whose_version_changed_is_asked_about_again(rig: Rig):
    """An acceptance is of the licence's id and its version, which is its title."""
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(closed(rig))
        assets.fetch(closed(rig, "second.stl", b"second" * 100, custom(title="Maker Licence 1.2")))
    assert asker.asked == [(CLOSED,), (CLOSED,)]


@pytest.mark.parametrize("same", [{"lines": ("non-commercial use only",)}, {"lines": ()},
                                  {"url": "https://example.org/moved"}])
def test_the_same_licence_in_another_form_is_one_acceptance(rig: Rig, same: dict[str, object]):
    """Keyed by the id and the version alone: the lines of conditions and the link are not what was accepted, so the
    manifest's form of a licence and the saved lock's (whose form has no key for lines) are asked about once."""
    asker = Scripted()
    other = custom(lines=same.get("lines", LINES))  # type: ignore[arg-type]
    if "url" in same:
        other = CustomLicence(CLOSED, "Maker Licence 1.1", str(same["url"]), "not_permitted", LINES)
    with assets.asking(asker):
        assets.fetch(closed(rig))
        assets.fetch(closed(rig, "second.stl", b"second" * 100, other))
    assert asker.asked == [(CLOSED,)]


def test_alternating_the_manifests_form_and_the_locks_form_of_a_licence_asks_once(rig: Rig):
    """The review's S3: the manifest states four lines and the lock none, and four fetches asked four times."""
    manifest_form, lock_form = custom(), custom(lines=())
    asker = Scripted()
    with assets.asking(asker):
        for index, held in enumerate((manifest_form, lock_form, manifest_form, lock_form)):
            assets.fetch(closed(rig, f"m{index}.stl", f"mesh {index}".encode() * 40, held))
    assert asker.asked == [(CLOSED,)]
    assert [each.licence.id for each in _acceptances.read_acceptances()] == [CLOSED]


def test_a_batch_asks_once_for_all_its_files_and_shows_the_notice_with_the_licence(rig: Rig):
    refs = [closed(rig, f"m{i}.stl", f"mesh {i}".encode() * 50) for i in range(3)]
    asker = Scripted()
    with assets.asking(asker), assets.approved(refs, notice="The notice, in the licence's own words.") as approval:
        for ref in refs:
            assets.fetch(ref)
    assert asker.asked == [(CLOSED,)] and asker.notices == ["The notice, in the licence's own words."]
    assert approval.notice_shown and all(assets.is_cached(ref) for ref in refs)


def test_a_file_already_in_the_cache_whose_licence_is_accepted_asks_nothing_and_says_nothing(rig: Rig):
    ref = closed(rig)
    with assets.asking(Scripted()):
        assets.fetch(ref)
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(ref)
        assets.approve([ref])
    assert asker.asked == [] and asker.lines == [] and rig.requests == 1


def test_a_cached_file_with_no_recorded_acceptance_asks_once_at_its_first_use(rig: Rig):
    """A cache filled before the gates existed, or by ``--accept-licence`` (which is not remembered): the first use of a
    file whose licence has conditions asks, downloads nothing, and remembers the yes."""
    ref = closed(rig)
    assets.fetch(ref, accept_licences={CLOSED})
    assert _acceptances.read_acceptances() == [] and rig.requests == 1
    declined = Scripted(licence=False)
    with assets.asking(declined):
        assert codes(lambda: assets.resolve(ref)) == ["asset.licence-declined"]
        assert codes(lambda: assets.fetch(ref)) == ["asset.licence-declined"]
    with assets.asking(None):
        assert codes(lambda: assets.resolve(ref)) == ["asset.licence-not-accepted"]
        assert f"--accept-licence {CLOSED}" in refusal_text(lambda: assets.resolve(ref))
        assert assets.resolve(ref, accept_licences={CLOSED}) == assets.file_path(ref), "named, for this call"
    asker = Scripted()
    with assets.asking(asker):
        assert assets.resolve(ref) == assets.file_path(ref)
        assets.resolve(ref)
        assets.fetch(ref)
    assert asker.asked == [(CLOSED,)] and rig.requests == 1
    assert [each.licence.id for each in _acceptances.read_acceptances()] == [CLOSED]


def test_a_batch_over_cached_files_asks_once_at_its_entry_for_all_of_them(rig: Rig):
    refs = [closed(rig, f"m{i}.stl", f"mesh {i}".encode() * 50) for i in range(3)]
    for ref in refs:
        assets.fetch(ref, accept_licences={CLOSED})
    declined = Scripted(licence=False)
    with assets.asking(declined):
        with pytest.raises(BuildRefused):
            with assets.approved(refs):
                raise AssertionError("the block ran although the licence was declined")
    assert declined.asked == [(CLOSED,)], "asked when the batch is entered, before any file is read"
    asker = Scripted()
    with assets.asking(asker), assets.approved(refs, notice="The notice.") as approval:
        assert asker.asked == [(CLOSED,)] and approval.licences[0].id == CLOSED and approval.notice_shown
        for ref in refs:
            assets.fetch(ref)
    assert asker.asked == [(CLOSED,)] and approval.plan is None and rig.requests == 3
    assert asker.lines == [], "nothing is downloaded, so there is no line about the drive"


def test_a_licence_nobody_stated_is_gated_with_a_panel_that_says_the_terms_are_not_stated(rig: Rig):
    rig.server.files["/n.bin"] = b"unstated" * 50
    import hashlib

    ref = AssetRef("n.bin", hashlib.sha256(b"unstated" * 50).hexdigest(), 400, "NOASSERTION", (rig.server.url("/n.bin"),))
    (found,) = assets.licences_of(ref)
    assert found.id == "NOASSERTION" and "not stated" in found.title + " ".join(found.conditions)
    assert "you are responsible for checking them" in " ".join(found.conditions)
    with assets.asking(None):
        text = refusal_text(lambda: assets.fetch(ref))
        assert codes(lambda: assets.fetch(ref, yes=True, force=True)) == ["asset.licence-not-accepted"]
        assert "--accept-licence NOASSERTION" in text and rig.requests == 0
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(ref)
    assert asker.asked == [("NOASSERTION",)] and [e.licence.id for e in _acceptances.read_acceptances()] == [
        "NOASSERTION"], "accepted by the panel's yes, and remembered as the others are"
    named = AssetRef("n3.bin", hashlib.sha256(b"third" * 50).hexdigest(), 250, "NOASSERTION", (rig.server.url("/n3.bin"),))
    rig.server.files["/n3.bin"] = b"third" * 50
    _acceptances.revoke("NOASSERTION")
    with assets.asking(None):
        assets.fetch(named, accept_licences={"NOASSERTION"})
    assert assets.is_cached(named)


def test_the_acceptances_are_listed_and_revoked_by_id(rig: Rig):
    with assets.asking(Scripted()):
        assets.fetch(closed(rig))
    assert CLOSED in assets.acceptances_text() and "accepted 20" in assets.acceptances_text()
    assert assets.revoke(OTHER) == 0 and len(assets.read_acceptances()) == 1
    assert assets.revoke(CLOSED) == 1 and assets.read_acceptances() == []
    assert assets.acceptances_text().startswith("no licence accepted on this machine")
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(closed(rig, "second.stl", b"second" * 100))
    assert asker.asked == [(CLOSED,)], "after a revoke the next download asks again"


def test_the_acceptances_file_is_in_the_config_directory_unless_a_variable_names_another(monkeypatch: pytest.MonkeyPatch,
                                                                                         tmp_path: Path):
    monkeypatch.delenv("MIME_LICENCE_ACCEPTANCES")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for variable in ("APPDATA", "LOCALAPPDATA", "WIN_PD_OVERRIDE_APPDATA", "WIN_PD_OVERRIDE_LOCAL_APPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "config"))  # Windows: platformdirs reads its overrides first
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _acceptances.acceptances_path().name == "licences.json" and "mime" in _acceptances.acceptances_path().parts
    assert tmp_path in _acceptances.acceptances_path().parents


@pytest.mark.parametrize("text", ["not json", "[]", '{"format": "x", "accepted": []}',
                                  '{"format": "mime.licence_acceptances/1", "accepted": [{"id": "x"}]}'])
def test_a_file_that_cannot_be_read_is_refused_never_read_as_no_acceptance_and_written_over(rig: Rig, text: str):
    path = _acceptances.acceptances_path()
    path.write_text(text, encoding="utf-8")
    assert codes(assets.read_acceptances) == ["asset.licences-file"]
    with assets.asking(Scripted()):
        assert codes(lambda: assets.fetch(closed(rig))) == ["asset.licences-file"]
    assert path.read_text(encoding="utf-8") == text and rig.requests == 0


# --- a notebook asks as a terminal does ------------------------------------------------------------------------------------


def test_in_a_notebook_the_library_asks_through_input_and_a_kernel_that_takes_none_is_no_terminal(
        rig: Rig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    import builtins
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # a kernel's standard input is no terminal
    assert not assets.can_ask()
    monkeypatch.setattr(builtins, "get_ipython", lambda: object(), raising=False)
    assert assets.can_ask(), "a notebook (or IPython) can answer: input() asks in the page"
    prompts: list[str] = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": prompts.append(prompt) or "y")
    ref = closed(rig)
    assets.fetch(ref)
    out = capsys.readouterr().out
    assert prompts == ["Do you accept these conditions? [y/N] "] and "Licence conditions: your acceptance is needed" in out
    assert LINES[0] in out and "Maker Licence 1.1" in out and assets.is_cached(ref)

    class StdinNotImplementedError(RuntimeError):
        """ipykernel's, raised by input() when the frontend allows no stdin (nbconvert, a scheduled run)."""

    def no_stdin(prompt: str = "") -> str:
        raise StdinNotImplementedError("raw_input was called, but this frontend does not support input requests.")

    monkeypatch.setattr(builtins, "input", no_stdin)
    assert codes(lambda: assets.fetch(closed(rig, "second.stl", b"second" * 100, custom(OTHER, title="Other")))) == [
        "asset.licence-not-accepted"]


def test_a_terminal_on_standard_input_alone_is_nobody_to_ask(monkeypatch: pytest.MonkeyPatch):
    """With both output streams redirected the question would go into a file unseen, so nobody is asked (and Windows'
    null device, a terminal to ``isatty``, with a pipe for output is such a case)."""
    import io

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Terminal(""))
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())
    assert not assets.can_ask()
    monkeypatch.setattr("sys.stderr", Terminal())
    assert assets.can_ask(), "the control: the same input with a terminal to show the question on"


def test_standard_input_that_ends_before_an_answer_is_no_terminal_and_not_a_no(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    """Windows reads its null device as a terminal; a program run with standard input from it is asked and gets the end of
    the input at once. That is nobody to ask: the refusal names the flag, as it does without a terminal."""
    import builtins
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr(builtins, "get_ipython", lambda: object(), raising=False)  # something that claims it can ask

    def ends(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", ends)
    ref = closed(rig)
    assert codes(lambda: assets.fetch(ref)) == ["asset.licence-not-accepted"]
    assert rig.requests == 0


# --- no generic switch accepts a licence --------------------------------------------------------------------------------


@pytest.mark.parametrize("switch", [{"yes": True}, {"force": True}, {"yes": True, "force": True}])
def test_neither_yes_nor_force_accepts_a_licence(rig: Rig, switch: dict[str, bool]):
    ref = closed(rig)
    with assets.asking(None):
        assert codes(lambda: assets.fetch(ref, **switch)) == ["asset.licence-not-accepted"]
        assert codes(lambda: assets.approve([ref], **switch)) == ["asset.licence-not-accepted"]
    assert rig.requests == 0 and not assets.is_cached(ref)


# --- the disk gate ---------------------------------------------------------------------------------------------------------


def sized(rig: Rig, size: int) -> AssetRef:
    """A record that declares ``size`` bytes (its server is never asked: the gate decides first)."""
    return AssetRef("big.bin", "cd" * 32, size, "CC0-1.0", (rig.server.url("/big.bin"),))


#: (free, total, size, asks, stops): the arithmetic of the two rules, by hand. Asked when the download is over 1 GB or over
#: a tenth of the free space; stopped when it leaves under 2 GB or under 5 % of the drive.
THRESHOLDS = [
    (500 * GB, 500 * GB, 100 * 10**6, False, False),         # ample space, a small download
    (500 * GB, 500 * GB, 2 * GB, True, False),                # over 1 GB
    (30 * GB, 500 * GB, 3_100_000_000, True, False),          # over a tenth of 30 GB; leaves 26.9 GB, over 5 % (25 GB)
    (30 * GB, 500 * GB, 900 * 10**6, False, False),           # under 1 GB and under 3 GB (a tenth); leaves 29.1 GB
    (5 * GB, 500 * GB, 600 * 10**6, True, True),              # under 1 GB, over a tenth of 5 GB; leaves 4.4 GB, under 5 %
    (10 * GB, 100 * GB, 5 * GB, True, False),                 # leaves 5 GB: exactly 5 % of the drive, not under it
    (10 * GB, 100 * GB, 5_100_000_000, True, True),           # leaves 4.9 GB: under 5 % of the drive
    (4 * GB, 20 * GB, 2 * GB, True, False),                   # leaves exactly 2 GB, which is over 5 % of 20 GB (1 GB)
    (4 * GB, 20 * GB, 2_000_000_001, True, True),             # leaves a byte under 2 GB
    (3 * GB, 20 * GB, 100 * 10**6, False, False),             # leaves 2.9 GB of a small drive
    (2_300_000_000, 500 * GB, 400 * 10**6, True, True),       # a nearly full drive
]


@pytest.mark.parametrize("free,total,size,asks,stops", THRESHOLDS)
def test_the_gate_asks_above_a_gigabyte_or_a_tenth_of_the_free_space_and_stops_under_two_gigabytes_or_five_percent(
        free: int, total: int, size: int, asks: bool, stops: bool):
    plan = _gates.Plan(1, size, 0, Path("."), free, total)
    assert (plan.asks, plan.stops) == (asks, stops)


def test_the_stop_refuses_whatever_the_terminal_and_only_force_overrides_it(rig: Rig):
    ref = sized(rig, 400 * 10**6)
    asker = Scripted()
    with space(free=2_300_000_000), assets.asking(asker):
        assert codes(lambda: assets.approve([ref])) == ["asset.disk-low"]
        assert codes(lambda: assets.approve([ref], yes=True)) == ["asset.disk-low"], "yes answers the question only"
        text = refusal_text(lambda: assets.approve([ref]))
        assert "400 MB" in text and "2.3 GB" in text and "--force" in text
        assert asker.asked == [] and asker.questions == [], "a stop asks nothing"
        assert assets.approve([ref], force=True).plan is not None
    assert asker.questions == ["Download 400 MB now?"], "force overrides the stop and nothing else: the question stands"


def test_the_question_is_asked_when_a_terminal_can_answer_and_refused_without_one(rig: Rig):
    ref = sized(rig, 2 * GB)
    with space(free=300 * GB):
        asker = Scripted(confirm=True)
        with assets.asking(asker):
            assert assets.approve([ref]).plan is not None
        assert asker.questions == ["Download 2.0 GB now?"]
        no = Scripted(confirm=False)
        with assets.asking(no):
            assert codes(lambda: assets.approve([ref])) == ["asset.download-declined"]
        with assets.asking(None):
            assert codes(lambda: assets.approve([ref])) == ["asset.download-large"]
            text = refusal_text(lambda: assets.approve([ref]))
            assert "--yes" in text and "2.0 GB" in text
            assert assets.approve([ref], yes=True).plan is not None
        never = Scripted()
        with assets.asking(never):
            assets.approve([ref], yes=True)
        assert never.questions == [], "yes answers the question without asking it"


def test_a_confirm_callable_answers_the_question_in_place_of_the_askers(rig: Rig):
    ref = sized(rig, 2 * GB)
    seen: list[str] = []
    with space(free=300 * GB), assets.asking(None):
        assets.approve([ref], confirm=lambda question: seen.append(question) or True)
    assert seen == ["Download 2.0 GB now?"]


def test_the_line_says_what_the_download_takes_and_the_drive_has_in_one_line(rig: Rig):
    refs = [sized(rig, 1_200_000_000), AssetRef("two.bin", "ef" * 32, 12, "CC0-1.0", (rig.server.url("/two.bin"),))]
    asker = Scripted()
    with space(free=18_400_000_000, total=250 * GB), assets.asking(asker):
        assets.approve(refs, yes=True, force=True)
    assert asker.lines == [f"Download 1.2 GB (2 files) into {rig.cache}: 18.4 GB free of 250 GB."]


def test_a_bake_is_counted_with_the_files_it_reads(rig: Rig):
    refs = [sized(rig, 100 * 10**6)]
    assert assets.bake_bytes(refs) == 30 * 10**6 and assets.BAKE_OUTPUT_FACTOR == 0.30
    asker = Scripted()
    with space(free=100 * GB), assets.asking(asker):
        plan = assets.approve(refs, extra_bytes=assets.bake_bytes(refs)).plan
    assert plan is not None and plan.needed == 130 * 10**6
    assert "bake about 30.0 MB more" in asker.lines[0]


def test_the_drive_is_the_caches_own_and_follows_the_variable(rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: list[Path] = []

    def usage(path: Path) -> Usage:
        seen.append(Path(path))
        return Usage(500 * GB, 0, 500 * GB)

    monkeypatch.setattr("shutil.disk_usage", usage)
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "elsewhere" / "deeper"))
    assets.approve([sized(rig, 10)])
    assert seen and seen[-1] == tmp_path, "the nearest folder that exists, here the temporary one, is read"
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "elsewhere"))
    (tmp_path / "elsewhere").mkdir()
    assets.approve([sized(rig, 10)])
    assert seen[-1] == tmp_path / "elsewhere"


def test_the_thresholds_are_named_in_one_place():
    assert (assets.CONFIRM_ABOVE_BYTES, assets.CONFIRM_ABOVE_FRACTION) == (1_000_000_000, 0.10)
    assert (assets.STOP_BELOW_BYTES, assets.STOP_BELOW_FRACTION) == (2_000_000_000, 0.05)


def test_a_bundle_of_a_stated_size_is_held_to_the_same_gate(rig: Rig):
    rig.server.files["/bundle"] = b"b" * 400
    with space(free=2_300_000_000), assets.asking(None):
        assert codes(lambda: assets.retrieve(rig.server.url("/bundle"), name="bundle", size=400 * 10**6)) == [
            "asset.disk-low"]
    assert rig.requests == 0
    with space(free=300 * GB), assets.asking(None):
        assert codes(lambda: assets.retrieve(rig.server.url("/bundle"), name="bundle", size=2 * GB,
                                             max_bytes=3 * GB)) == ["asset.download-large"]
    assert rig.requests == 0


# --- what a download that fails leaves behind --------------------------------------------------------------------------


def leftovers(cache: Path) -> list[str]:
    return sorted(path.name for path in cache.rglob("*") if path.is_file() and not path.name.endswith(".lock"))


def test_a_download_whose_write_fails_part_way_leaves_no_file_behind(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    """The drive fills while the bytes arrive: the download's temporary folder goes with it, nothing reaches the cache's
    own path, and the asset is not recorded as downloaded."""
    ref = open_record(rig, payload=b"x" * 5000)
    from mime.assets import _fetch

    original = _fetch._Capped.write
    calls = {"n": 0}

    def failing(self: object, chunk: bytes) -> int:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(self, chunk)  # type: ignore[arg-type]

    monkeypatch.setattr(_fetch._Capped, "write", failing)
    monkeypatch.setattr(_fetch, "BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(_fetch, "_BLOCK", 1000)
    with assets.asking(None):
        with pytest.raises(BuildRefused):
            assets.fetch(ref)
    assert calls["n"] > 1, "the write failed after bytes had been written"
    assert leftovers(rig.cache) == [] and not assets.is_cached(ref)
    assert [path.name for path in rig.cache.iterdir() if ".incoming-" in path.name] == []
    monkeypatch.setattr(_fetch._Capped, "write", original)
    assets.fetch(ref)
    assert assets.is_cached(ref), "the next fetch is not troubled by what the failed one did"


def test_a_bake_whose_write_fails_leaves_no_half_written_layer_or_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mime.viz import _meshes, _visuals

    cache = tmp_path / "cache"
    monkeypatch.setenv("MIME_ASSETS_DIR", str(cache))
    mesh = _meshes.Mesh(_points(), _triangles())
    target = _visuals.visuals_dir() / "layer.usdc"

    def full(source: object, destination: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_meshes.os, "replace", full)
        with pytest.raises(OSError):
            _meshes.write_mesh(target, mesh, "mesh")
    assert list(target.parent.iterdir()) == [], "the layer written beside it is removed with the failure"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_visuals.os, "replace", full)
        with pytest.raises(OSError):
            _visuals._written(target.with_suffix(".json"), "{}")
    assert list(target.parent.iterdir()) == []


def _points():
    import numpy as np

    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def _triangles():
    import numpy as np

    return np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)


def test_a_bake_is_gated_as_a_fetch_is_and_the_notice_is_shown_once(rig: Rig):
    import struct

    from mime.viz import bake

    body = b"".join(struct.pack("<12fH", 0.0, 0.0, 1.0, *(0.01 * ((i + k) % 5) for k in range(9)), 0) for i in range(40))
    data = b"made".ljust(80, b"\0") + struct.pack("<I", 40) + body
    held = manifest_data()
    ref = closed(rig, "made.stl", data)
    held["assets"] = [dict(ref.to_record(conditions=True), role="presentation")]
    held["licences"] = {CLOSED: {"title": "Maker Licence 1.1", "url": "https://example.org/licence",
                                 "redistribution": "not_permitted", "conditions": list(LINES)}}
    manifest = assets.read_manifest_data(held, "made")
    said: list[str] = []
    with assets.asking(None):
        assert codes(lambda: bake(manifest, say=said.append)) == ["asset.licence-not-accepted"]
        assert rig.requests == 0 and said == []
        for switch in ({"yes": True}, {"force": True}):
            assert codes(lambda switch=switch: bake(manifest, say=said.append, **switch)) == [
                "asset.licence-not-accepted"]
        (baked,) = bake(manifest, say=said.append, accept_licences={CLOSED})
    assert baked.layer.is_file() and rig.requests == 1 and "terms" in said[0], "no terminal: the notice is said as before"
    asker = Scripted()
    shown: list[str] = []
    rig.cache.joinpath(ref.sha256).unlink()
    for each in rig.cache.glob("*.verified"):
        each.unlink()
    with assets.asking(asker):
        bake(manifest, say=shown.append)
    assert asker.asked == [(CLOSED,)] and asker.notices == ["terms"]
    assert "terms" not in shown[0], "a panel that has just shown the notice does not say it again"


# --- the manifest format and the conditions ---------------------------------------------------------------------------------


def manifest_data(format_: str = "mime.asset_manifest/2", *, conditions: bool = True) -> dict[str, object]:
    licence: dict[str, object] = {"title": "Maker Licence 1.1", "url": "https://example.org/licence",
                                  "redistribution": "not_permitted"}
    if conditions:
        licence["conditions"] = list(LINES)
    return {"format": format_, "name": "made", "title": "a made mesh", "credit": "a maker", "notice": "terms",
            "licences": {CLOSED: licence}, "archives": [],
            "assets": [{"name": "made.stl", "sha256": "ab" * 32, "size": 10, "licence": CLOSED,
                        "locations": ["https://example.org/made.stl"]}],
            "files": {"meshes/made.stl": {"asset": "made.stl"}}}


def test_a_second_form_manifest_states_conditions_which_reach_each_record_and_round_trip():
    held = assets.read_manifest_data(manifest_data(), "made")
    (found,) = assets.licences_of(held.assets[0])
    assert found.conditions == LINES
    written = held.to_json()
    assert written["format"] == "mime.asset_manifest/3" and written["assets"][0]["licences"][CLOSED]["conditions"] == list(LINES)
    assert assets.read_manifest_data(written, "again").to_json() == written


def test_a_first_form_manifest_is_still_read_with_no_conditions_and_one_that_states_them_is_refused():
    held = assets.read_manifest_data(manifest_data("mime.asset_manifest/1", conditions=False), "made")
    assert held.assets[0].licences[0].conditions == () and assets.licences_of(held.assets[0])[0].conditions[0].startswith(
        "it may not be redistributed")
    assert codes(lambda: assets.read_manifest_data(manifest_data("mime.asset_manifest/1"), "made")) == ["asset.manifest"]
    data = manifest_data("mime.asset_manifest/1", conditions=False)
    data["assets"][0]["licences"] = {CLOSED: {"title": "t", "url": "https://example.org/l",  # type: ignore[index]
                                              "redistribution": "not_permitted", "conditions": ["x"]}}
    assert codes(lambda: assets.read_manifest_data(data, "made")) == ["asset.manifest"]


@pytest.mark.parametrize("bad", ["a line", [1], [""], ["x" * 200], ["a\nb"]])
def test_conditions_that_are_not_short_lines_of_text_are_refused(bad: object):
    data = manifest_data()
    data["licences"][CLOSED]["conditions"] = bad  # type: ignore[index]
    assert codes(lambda: assets.read_manifest_data(data, "made")) == ["asset.manifest"]


def test_the_saved_lock_holds_the_manifest_as_the_first_form_without_the_condition_lines():
    held = assets.read_manifest_data(manifest_data(), "made")
    in_lock = held.to_json(lock=True)
    assert in_lock["format"] == "mime.asset_manifest/1" and "conditions" not in json.dumps(in_lock)
    assert held.assets[0].to_record() == held.assets[0].to_record(conditions=False)
    assert "conditions" in json.dumps(held.assets[0].to_record(conditions=True))


# --- the conditions are looked up by the licence's id -----------------------------------------------------------------------


def lock_form(rig: Rig, name: str = "locked.stl") -> AssetRef:
    """The record as a saved folder's lock holds it: its custom licence has the title and link, and no lines."""
    return closed(rig, name, b"locked" * 80, custom(lines=()))


def cached_index(rig: Rig, *, lines: Sequence[str] = LINES, identifier: str = CLOSED) -> Path:
    """A baked set's index in the cache, as ``bake`` writes it: the manifest's licences, with their lines."""
    folder = rig.cache / "visuals"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "made.json"
    path.write_text(json.dumps({"format": "mime.visual_index/1", "set": "made", "files": {},
                                "licences": {identifier: {"title": "Maker Licence 1.1",
                                                          "url": "https://example.org/licence",
                                                          "conditions": list(lines)}}}), encoding="utf-8")
    return path


def test_a_licence_read_from_a_lock_finds_its_lines_in_a_cached_manifest_by_its_id(rig: Rig):
    ref = lock_form(rig)
    assert "does not describe" in assets.licences_of(ref)[0].conditions[-1]
    cached_index(rig)
    (found,) = assets.licences_of(ref)
    assert found.conditions == LINES and found.title == "Maker Licence 1.1"
    cached_index(rig, identifier="LicenseRef-Someone-Else")
    assert "does not describe" in assets.licences_of(ref)[0].conditions[-1], "the control: another id's lines are not borrowed"
    # A licence the record names without defining it (its title and link are nowhere in it) is found whole.
    cached_index(rig)
    bare = AssetRef("b.stl", "ab" * 32, 10, CLOSED, ("https://example.org/b",))
    (named,) = assets.licences_of(bare)
    assert (named.title, named.url, named.conditions) == ("Maker Licence 1.1", "https://example.org/licence", LINES)


def test_a_licence_read_from_a_lock_finds_its_lines_in_an_installed_manifest_by_its_id(rig: Rig,
                                                                                         monkeypatch: pytest.MonkeyPatch):
    installed = assets.read_manifest_data(manifest_data(), "made")
    monkeypatch.setattr("mime.assets._sources.shared_manifests", lambda: (("package", installed),))
    (found,) = assets.licences_of(lock_form(rig))
    assert found.conditions == LINES
    monkeypatch.setattr("mime.assets._sources.shared_manifests", lambda: ())
    assert "does not describe" in assets.licences_of(lock_form(rig))[0].conditions[-1], "the control: with none installed"


def test_where_no_manifest_gives_lines_the_panel_shows_the_notice_title_and_link_as_it_did(rig: Rig):
    ref = lock_form(rig)
    asker = Scripted()
    with assets.asking(asker), assets.approved([ref], notice="The notice."):
        pass
    assert asker.asked == [(CLOSED,)] and asker.notices == ["The notice."]
    (found,) = assets.licences_of(ref)
    assert (found.title, found.url) == ("Maker Licence 1.1", "https://example.org/licence")
    assert found.conditions[0].startswith("it may not be redistributed") and "read the licence" in found.conditions[1]


# --- what a stranger's text can do at the terminal ------------------------------------------------------------------------

ESCAPES = {"clear the screen": "\x1b[2J\x1b[1;1H", "set the window title": "\x1b]0;pwned\x07", "erase a line": "\x1b[2K\r",
           "conceal": "\x1b[8m", "colour": "\x1b[31m", "a forged line": "\nI accept everything"}


def hostile(field: str, planted: str) -> Licence:
    base = {"id": CLOSED, "title": "Maker Licence 1.1", "url": "https://example.org/licence",
            "conditions": ("non-commercial use only",)}
    if field == "conditions":
        base["conditions"] = (f"non-commercial{planted}use only",)
    else:
        base[field] = f"{base[field]}{planted}"
    return Licence(**base)  # type: ignore[arg-type]


#: Each field a record's text sits in, with each thing planted in it; the notice is prose and keeps its line breaks (as every
#: refusal's does), so a forged line is not planted there.
PLANTS = [(field, what) for field in ("title", "url", "conditions", "id", "notice") for what in ESCAPES
          if (field, what) != ("notice", "a forged line")]


@pytest.mark.parametrize("field,what", PLANTS)
def test_the_plain_licence_prompt_writes_a_manifests_text_as_plain_characters(
        field: str, what: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    """The review's B1: the plain prompt printed a stranger's title, link and notice raw, and a clear-screen between the
    conditions and the question could erase what the person was about to accept."""
    planted = ESCAPES[what]
    held = hostile("title" if field == "notice" else field, planted if field != "notice" else "")
    notice = f"The notice{planted}ends." if field == "notice" else None

    class Terminal:
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            written.append(text)
            return len(text)

        def flush(self) -> None:
            pass

    written: list[str] = []
    monkeypatch.setattr("sys.stdout", Terminal())
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert not _gates.PlainAsker().licences([held], notice)
    out = "".join(written)
    if what == "a forged line":
        assert "\nI accept everything" not in out and "\\nI accept everything" in out
    else:
        assert "\x1b" not in out and "\x07" not in out and "\r" not in out, out
    assert "\\x1b" in out or what == "a forged line"
    assert "Do you accept these conditions? [y/N]" in out
    assert capsys.readouterr().out == ""


def test_a_hostile_licence_is_escaped_in_lines_the_plan_and_the_listing(rig: Rig):
    lines = _gates.licence_lines([hostile("title", ESCAPES["clear the screen"])], "n\x1b[2Jotice")
    assert not any("\x1b" in line for line in lines)
    _acceptances.remember([hostile("title", ESCAPES["set the window title"]), hostile("url", ESCAPES["colour"])])
    assert "\x1b" not in assets.acceptances_text() and "\x07" not in assets.acceptances_text()
    assert "\x1b" not in _gates.Plan(1, 10, 0, Path("a\x1b[2Jb"), 10**12, 10**12).line()
    held = assets.read_manifest_data({**manifest_data(), "title": "a\x1b[2Jmade", "notice": "n\x1b]0;x\x07"}, "made")
    assert "\x1b" not in held.text() and "\x07" not in held.text()


# --- a corrupt cached file is repaired as a download ---------------------------------------------------------------------


def flipped(ref: AssetRef) -> None:
    """One byte of the cached file changed, its size kept (bit rot): the cheap check no longer vouches for it."""
    path = assets.file_path(ref)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("repair", ["fetch", "resolve"])
def test_a_bit_flipped_cached_file_is_repaired_only_as_a_download_that_is_gated(rig: Rig, repair: str):
    """The review's S1: a same-size corrupt file was re-downloaded with no licence question and no disk check."""
    ref = closed(rig)
    with assets.asking(Scripted()):
        assets.fetch(ref)
    again = getattr(assets, repair)
    flipped(ref)
    _acceptances.revoke(CLOSED)
    with assets.asking(None):
        assert codes(lambda: again(ref)) == ["asset.licence-not-accepted"]
    assert rig.requests == 1, "no request was made to repair it"
    _acceptances.remember(assets.licences_of(ref))
    with space(free=2_100_000_000), assets.asking(None):
        assert codes(lambda: again(ref)) == ["asset.disk-low"]
    assert rig.requests == 1
    with assets.asking(None):
        again(ref)
    assert rig.requests == 2 and assets.check(ref) == "ok", "the repair itself still happens when it is allowed"


# --- the pinned behaviours of the gate (the review's S2) -----------------------------------------------------------------


def test_a_batch_of_an_open_and_a_closed_asset_is_gated_for_the_closed_one(rig: Rig):
    batch = [open_record(rig), closed(rig)]
    with assets.asking(None):
        assert codes(lambda: assets.approve(batch)) == ["asset.licence-not-accepted"]
        assert codes(lambda: assets.approve(list(reversed(batch)))) == ["asset.licence-not-accepted"]
        with pytest.raises(BuildRefused):
            with assets.approved(batch):
                pass
    assert rig.requests == 0
    asker = Scripted()
    with assets.asking(asker), assets.approved(batch):
        for ref in batch:
            assets.fetch(ref)
    assert asker.asked == [(CLOSED,)] and all(assets.is_cached(ref) for ref in batch)


@pytest.mark.parametrize("named", ["Maker Licence 1.1", "Maker", "maker licence 1.1", "https://example.org/licence",
                                   "LicenseRef", "LicenseRef-Maker-NonCommercia", "*"])
def test_accepting_by_name_takes_the_id_and_nothing_else_the_title_included(rig: Rig, named: str):
    ref = closed(rig)
    with assets.asking(None):
        assert codes(lambda: assets.fetch(ref, accept_licences={named})) == ["asset.licence-not-accepted"]
        assert codes(lambda: assets.approve([ref], accept_licences=[named])) == ["asset.licence-not-accepted"]
    assert rig.requests == 0
    assets.fetch(ref, accept_licences={CLOSED})
    assert rig.requests == 1, "the control: the id accepts"


@pytest.mark.parametrize("answer", ["", " ", "n", "no", "N", "maybe", "yes please", "ye", "\n"])
def test_the_plain_asker_takes_only_y_or_yes_and_enter_is_no(answer: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.stdout", Tty())
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    assert _gates.PlainAsker().licences([custom_licence()], None) is False
    assert _gates.PlainAsker().confirm("Download 2.0 GB now?") is False
    for yes in ("y", "Y", " yes ", "YES"):
        monkeypatch.setattr("builtins.input", lambda *_, yes=yes: yes)
        assert _gates.PlainAsker().licences([custom_licence()], None) is True


class Tty:
    def __init__(self) -> None:
        self.written: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        pass


def custom_licence() -> Licence:
    return Licence(CLOSED, "Maker Licence 1.1", "https://example.org/licence", LINES)


def test_a_plain_no_downloads_nothing_and_writes_no_acceptance(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    ref = closed(rig)
    monkeypatch.setattr("sys.stdout", Tty())
    monkeypatch.setattr("builtins.input", lambda *_: "")
    with assets.asking(_gates.PlainAsker()):
        assert codes(lambda: assets.fetch(ref)) == ["asset.licence-declined"]
    assert rig.requests == 0 and _acceptances.read_acceptances() == []


@pytest.mark.parametrize("stdin_is_a_terminal,output_is_a_terminal,expected",
                         [(True, True, True), (False, True, False), (True, False, False), (False, False, False)])
def test_a_person_can_answer_only_where_standard_input_is_a_terminal_and_the_question_has_a_screen(
        stdin_is_a_terminal: bool, output_is_a_terminal: bool, expected: bool, monkeypatch: pytest.MonkeyPatch):
    import io

    class Stream(io.StringIO):
        def __init__(self, terminal: bool) -> None:
            super().__init__("")
            self.terminal = terminal

        def isatty(self) -> bool:
            return self.terminal

    monkeypatch.setattr("sys.stdin", Stream(stdin_is_a_terminal))
    monkeypatch.setattr("sys.stdout", Stream(output_is_a_terminal))
    monkeypatch.setattr("sys.stderr", Stream(False))
    assert assets.can_ask() is expected


# --- a bundle of no stated size is held to its cap by the stop ------------------------------------------------------------


def test_a_bundle_with_no_stated_size_is_stopped_by_its_cap_and_never_asked_about(rig: Rig):
    """The review's S4: ``mime pull <url>`` states no size, and was the one download the stop did not see."""
    rig.server.files["/bundle"] = b"b" * 400
    with space(free=1_000_000, total=GB), assets.asking(None):
        assert codes(lambda: assets.retrieve(rig.server.url("/bundle"), name="bundle")) == ["asset.disk-low"]
        text = refusal_text(lambda: assets.retrieve(rig.server.url("/bundle"), name="bundle"))
        assert "up to 500 MB" in text and "--force" in text
        assert codes(lambda: assets.retrieve(rig.server.url("/bundle"), name="bundle", yes=True)) == ["asset.disk-low"]
    assert rig.requests == 0
    with space(free=1_000_000, total=GB), assets.asking(None):
        assets.retrieve(rig.server.url("/bundle"), name="bundle", force=True)
    assert rig.requests == 1
    rig.server.files["/second"] = b"s" * 400
    asker = Scripted()
    with space(free=400 * GB, total=500 * GB), assets.asking(asker):
        assets.retrieve(rig.server.url("/second"), name="bundle")
    assert asker.questions == [] and asker.lines and asker.lines[0].startswith("Download up to 500 MB")
    with space(free=3_000_000_000, total=20 * GB), assets.asking(None):
        # 500 MB is over a tenth of 3 GB and would be asked about were it a stated size; it leaves 2.5 GB, so no stop
        assets.retrieve(rig.server.url("/second"), name="bundle")


# --- the controls: each fails on every run ----------------------------------------------------------------------------


def _gate_removed(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(_gates, "licences_of", lambda ref: ())


def _plant_switch(patch: pytest.MonkeyPatch, which: str) -> None:
    """``yes`` (or ``force``) makes the gate treat every licence the download needs as accepted."""
    real = _gates.approve

    def planted(refs, *, accept_licences=(), yes=False, force=False, **rest):  # type: ignore[no-untyped-def]
        if yes if which == "yes" else force:
            accept_licences = {each.id for ref in list(refs) for each in _gates.licences_of(ref)}
        return real(refs, accept_licences=accept_licences, yes=yes, force=force, **rest)

    patch.setattr(_gates, "approve", planted)
    patch.setattr("mime.assets._fetch.approve", planted)


def _stored_for_the_wrong_id(patch: pytest.MonkeyPatch) -> None:
    real = _acceptances.remember

    def wrong(licences: Sequence[Licence]) -> None:
        real([Licence(OTHER, each.title, each.url, each.conditions) for each in licences])

    patch.setattr(_gates, "remember", wrong)


def _acceptance_ignores_the_licence(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(_gates, "accepted", lambda licence: bool(_acceptances.read_acceptances()))


def _set(name: str, value: float) -> Callable[[pytest.MonkeyPatch], None]:
    return lambda patch: patch.setattr(_gates, name, value)


def scenario_gate(rig: Rig) -> None:
    ref = closed(rig)
    with assets.asking(None):
        assert codes(lambda: assets.fetch(ref)) == ["asset.licence-not-accepted"]
    assert rig.requests == 0


def scenario_switches(rig: Rig) -> None:
    ref = closed(rig)
    with assets.asking(None):
        for switch in ({"yes": True}, {"force": True}):
            assert codes(lambda switch=switch: assets.fetch(ref, **switch)) == ["asset.licence-not-accepted"]
    assert rig.requests == 0


def scenario_the_other_licence(rig: Rig) -> None:
    first, second = closed(rig), closed(rig, "second.stl", b"second" * 100, custom(OTHER, title="Other Licence 2"))
    with assets.asking(Scripted()):
        assets.fetch(first)
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(second)
    assert asker.asked == [(OTHER,)], "the yes to one licence is not a yes to another"


def scenario_the_remembered_id(rig: Rig) -> None:
    with assets.asking(Scripted()):
        assets.fetch(closed(rig))
    assert [each.licence.id for each in _acceptances.read_acceptances()] == [CLOSED]


def scenario_the_stop(rig: Rig) -> None:
    """Two cases that must stop (one by each rule) and two that must not (one by each rule)."""
    with assets.asking(None):
        with space(free=4 * GB, total=20 * GB):  # leaves 1.9 GB: under 2 GB, over 5 % of 20 GB
            assert codes(lambda: assets.approve([sized(rig, 2_100_000_000)], yes=True)) == ["asset.disk-low"]
        with space(free=30 * GB, total=500 * GB):  # leaves 24 GB: over 2 GB, under 5 % of 500 GB
            assert codes(lambda: assets.approve([sized(rig, 6 * GB)], yes=True)) == ["asset.disk-low"]
        with space(free=15 * GB, total=20 * GB):  # leaves 14.6 GB
            assert allowed(lambda: assets.approve([sized(rig, 400 * 10**6)]))
        with space(free=30 * GB, total=500 * GB):  # leaves 29.6 GB, over 5 % of 500 GB
            assert allowed(lambda: assets.approve([sized(rig, 400 * 10**6)]))


SCENARIOS: dict[str, Callable[[Rig], None]] = {
    "the gate": scenario_gate, "the switches": scenario_switches, "the other licence": scenario_the_other_licence,
    "the remembered id": scenario_the_remembered_id, "the stop": scenario_the_stop}

#: (what is planted, how, the scenario that must then fail)
MUTANTS: list[tuple[str, Callable[[pytest.MonkeyPatch], None], str]] = [
    ("the licence gate removed", _gate_removed, "the gate"),
    ("--yes accepting a licence", lambda patch: _plant_switch(patch, "yes"), "the switches"),
    ("--force accepting a licence", lambda patch: _plant_switch(patch, "force"), "the switches"),
    ("an acceptance of one licence accepting another", _acceptance_ignores_the_licence, "the other licence"),
    ("an acceptance stored for the wrong id", _stored_for_the_wrong_id, "the remembered id"),
    ("the stop's bytes ten times too high", _set("STOP_BELOW_BYTES", 20 * GB), "the stop"),
    ("the stop's bytes ten times too low", _set("STOP_BELOW_BYTES", 200 * 10**6), "the stop"),
    ("the stop's fraction ten times too high", _set("STOP_BELOW_FRACTION", 0.5), "the stop"),
    ("the stop's fraction ten times too low", _set("STOP_BELOW_FRACTION", 0.005), "the stop"),
]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_the_scenarios_the_mutants_are_judged_by_pass_unplanted(scenario: str, rig: Rig):
    SCENARIOS[scenario](rig)


@pytest.mark.parametrize("run", range(3))
@pytest.mark.parametrize("name,plant,scenario", MUTANTS, ids=[each[0] for each in MUTANTS])
def test_every_mutant_is_caught_on_every_run(name: str, plant: Callable[[pytest.MonkeyPatch], None], scenario: str,
                                             run: int, tmp_path: Path):
    """Plant the fault and run the scenario that should notice it: the scenario must fail, on every run (an
    ``AssertionError``: a refusal of another kind would be the scenario failing for another reason)."""
    with pytest.MonkeyPatch.context() as patch:
        held = Rig(tmp_path, patch)
        patch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "none.toml"))
        (tmp_path / "none.toml").write_text("", encoding="utf-8")
        try:
            plant(patch)
            with pytest.raises(AssertionError):
                SCENARIOS[scenario](held)
        finally:
            held.server.close()
