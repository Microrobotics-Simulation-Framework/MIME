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


@pytest.mark.parametrize("licence", ["CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "NOASSERTION", "MIT OR CC-BY-NC-4.0"])
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


@pytest.mark.parametrize("change", [{"title": "Maker Licence 1.2"}, {"lines": ("non-commercial use only",)}])
def test_a_licence_that_changed_is_asked_about_again(rig: Rig, change: dict[str, object]):
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(closed(rig))
        assets.fetch(closed(rig, "second.stl", b"second" * 100, custom(**change)))  # type: ignore[arg-type]
    assert asker.asked == [(CLOSED,), (CLOSED,)]


def test_a_changed_link_is_asked_about_again(rig: Rig):
    asker = Scripted()
    moved = CustomLicence(CLOSED, "Maker Licence 1.1", "https://example.org/moved", "not_permitted", LINES)
    with assets.asking(asker):
        assets.fetch(closed(rig))
        assets.fetch(closed(rig, "second.stl", b"second" * 100, moved))
    assert asker.asked == [(CLOSED,), (CLOSED,)]


def test_a_batch_asks_once_for_all_its_files_and_shows_the_notice_with_the_licence(rig: Rig):
    refs = [closed(rig, f"m{i}.stl", f"mesh {i}".encode() * 50) for i in range(3)]
    asker = Scripted()
    with assets.asking(asker), assets.approved(refs, notice="The notice, in the licence's own words.") as approval:
        for ref in refs:
            assets.fetch(ref)
    assert asker.asked == [(CLOSED,)] and asker.notices == ["The notice, in the licence's own words."]
    assert approval.notice_shown and all(assets.is_cached(ref) for ref in refs)


def test_a_file_already_in_the_cache_is_not_a_download_and_asks_nothing(rig: Rig):
    ref = closed(rig)
    assets.fetch(ref, accept_licences={CLOSED})
    asker = Scripted()
    with assets.asking(asker):
        assets.fetch(ref)
        assets.approve([ref])
    assert asker.asked == [] and asker.lines == []


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
    monkeypatch.setenv("APPDATA", str(tmp_path / "config"))
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
    monkeypatch.undo()
    monkeypatch.setenv("MIME_ASSETS_DIR", str(rig.cache))
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

    monkeypatch.setattr(_meshes.os, "replace", full)
    with pytest.raises(OSError):
        _meshes.write_mesh(target, mesh, "mesh")
    assert list(target.parent.iterdir()) == [], "the layer written beside it is removed with the failure"
    monkeypatch.undo()
    monkeypatch.setenv("MIME_ASSETS_DIR", str(cache))
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
    assert written["format"] == "mime.asset_manifest/2" and written["assets"][0]["licences"][CLOSED]["conditions"] == list(LINES)
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
