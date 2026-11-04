"""Where records come from, which one wins, and where this machine gets the bytes (design #229).

- **The sources, in order**: an installed package's records arrive through its ``mime.assets`` entry point (the lab
  fixture's ``labx_marker.usda``, its bytes inside the package), beside MIME's own (the wall tables).
- **The pin wins**: a name pinned to one hash resolves to that hash while a shared source holds another, and the line
  that says so is gathered for the build report. Two shared sources that hold one name as different files, with no pin,
  are refused (``asset.conflict``), and so are two that hold the same bytes under different terms, whichever source
  comes first; the same bytes under the same terms in two sources are one record with both sources' locations.
- **The locations file**: its locations are tried after the cache and before the record's own (a lab's mirror first),
  by sha256, by name and as mirrors for every asset; a malformed one is refused, never read as no mirror; and a saved
  lock holds the record's own locations only, never a machine's.
- **``mime assets sources``** names each location, reachable or not and why, and offline says it did not look.

The controls, each failing on every run: a resolver that ignores the pin returns the shared source's hash; a merge that
checks the hashes alone lets the first source's terms win; a fetch that puts the locations file's locations after the
record's own asks the record's server first."""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from mime import assets
from mime.assets import _fetch, _sources
from mime.build import BuildRefused
from mime.cli import main
from mime.terms import CustomLicence
from mime.testing import LoopbackStore

PAYLOAD = b"a lab's table, in miniature. " * 500
RUNS = 20


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MIME_OFFLINE", raising=False)
    monkeypatch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "locations.toml"))
    (tmp_path / "locations.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    return tmp_path / "cache"


def ref_of(data: bytes, *locations: str, name: str = "lab_table") -> assets.AssetRef:
    return assets.AssetRef(name, hashlib.sha256(data).hexdigest(), len(data), "CC0-1.0", locations)


@contextmanager
def a_second_source(*refs: assets.AssetRef, name: str = "labz") -> Iterator[None]:
    """Another installed source holding ``refs``, beside the real ones."""
    real = _sources.shared()
    records = {key: list(value) for key, value in real.records.items()}
    for ref in refs:
        records.setdefault(ref.name, []).append(_sources.Sourced(ref, name))
    planted = _sources.Shared({key: tuple(value) for key, value in records.items()}, real.manifests)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_sources, "shared", lambda: planted)
        yield


def test_an_installed_packages_records_arrive_through_its_entry_point_beside_mimes_own():
    sources = {each.source for held in _sources.shared().records.values() for each in held}
    assert {"mime", "labx"} <= sources
    assert assets.reference("labx_marker.usda").locations == ("package:labx/marker.usda",)
    assert "unit_tube_R0.997" in assets.known()


def test_the_pin_wins_and_the_build_report_is_told(cache):
    old, new = ref_of(PAYLOAD, "https://example.org/old"), ref_of(PAYLOAD + b"v2", "https://example.org/new")
    with a_second_source(new):
        assert assets.reference("lab_table").sha256 == new.sha256
        with assets.pinned([old]) as notes:
            assert assets.reference("lab_table").sha256 == old.sha256
        assert len(notes) == 1 and "keeps its pin" in notes[0] and new.sha256 in notes[0] and "labz" in notes[0]


def test_two_shared_sources_that_disagree_with_no_pin_are_refused_and_a_pin_settles_them(cache):
    one, two = ref_of(PAYLOAD, "https://example.org/one"), ref_of(PAYLOAD + b"!", "https://example.org/two")
    with a_second_source(one, name="laba"), a_second_source(two, name="labb"):
        with pytest.raises(BuildRefused) as refused:
            assets.reference("lab_table")
        assert refused.value.codes() == ["asset.conflict"] and "laba" in str(refused.value)
        assert "lab_table" not in assets.known()
        with assets.pinned([two]):
            assert assets.reference("lab_table").sha256 == two.sha256


def test_the_same_bytes_in_two_sources_are_one_record_with_both_locations(cache):
    one, two = ref_of(PAYLOAD, "https://example.org/one"), ref_of(PAYLOAD, "https://mirror.example.org/one")
    with a_second_source(one, name="laba"), a_second_source(two, name="labb"):
        assert assets.reference("lab_table").locations == ("https://example.org/one", "https://mirror.example.org/one")


CLOSED_LICENCE = CustomLicence("LicenseRef-Closed", "a maker's closed licence", "https://example.org/licence",
                               "not_permitted")


def closed_of(data: bytes, location: str) -> assets.AssetRef:
    return assets.AssetRef("lab_table", hashlib.sha256(data).hexdigest(), len(data), CLOSED_LICENCE.id, (location,),
                           licences=(CLOSED_LICENCE,))


@pytest.mark.parametrize("closed_first", [False, True])
def test_the_same_bytes_under_different_terms_in_two_sources_are_refused_whichever_comes_first(cache, closed_first):
    """One source says the table is CC0, another that it may not be redistributed: a disagreement on what may be done
    with the file, refused as two hashes are, so that neither source's order decides the tier."""
    open_, closed = ref_of(PAYLOAD, "https://a.example.org/t"), closed_of(PAYLOAD, "https://z.example.org/t")
    first, second = (closed, open_) if closed_first else (open_, closed)
    with a_second_source(first, name="laba"), a_second_source(second, name="labb"):
        assert "lab_table" not in assets.known() and "lab_table" in assets.conflicts()
        with pytest.raises(BuildRefused) as refused:
            assets.reference("lab_table")
        text = str(refused.value)
        assert refused.value.codes() == ["asset.conflict"] and "the same file under different terms" in text
        assert "LicenseRef-Closed" in text and "CC0-1.0" in text
        with assets.pinned([closed]) as notes:
            held = assets.reference("lab_table")
        assert held.tier == "referenced" and held.licence == CLOSED_LICENCE.id, "the pin settles it"
        assert any("other terms" in note for note in notes)


@contextmanager
def terms_unchecked() -> Iterator[None]:
    """The plant: a merge that checks the hashes alone, as the review found it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_sources, "agree", lambda held: len({each.ref.sha256 for each in held}) == 1)
        yield


def test_control_a_merge_that_checks_hashes_alone_lets_the_first_sources_terms_win_every_run(cache):
    caught = 0
    for _ in range(RUNS):
        with terms_unchecked():
            try:
                test_the_same_bytes_under_different_terms_in_two_sources_are_refused_whichever_comes_first(cache, False)
            except AssertionError:
                caught += 1
    assert caught == RUNS


@contextmanager
def pins_ignored() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_sources, "active_pins", lambda: {})
        yield


def pin_is_kept() -> None:
    old, new = ref_of(PAYLOAD, "https://example.org/old"), ref_of(PAYLOAD + b"v2", "https://example.org/new")
    with a_second_source(new), assets.pinned([old]):
        assert assets.reference("lab_table").sha256 == old.sha256


def test_control_a_resolver_that_ignores_the_pin_is_caught_every_run():
    pin_is_kept()
    caught = 0
    for _ in range(RUNS):
        with pins_ignored():
            try:
                pin_is_kept()
            except AssertionError:
                caught += 1
    assert caught == RUNS


# --- the locations file ---------------------------------------------------------------------------------------------


def test_the_locations_file_is_tried_before_the_records_own_by_sha256_name_and_mirror(cache, tmp_path):
    with LoopbackStore() as lab, LoopbackStore() as upstream:
        lab.files["/mirror/t.npz"] = PAYLOAD
        upstream.files["/t.npz"] = PAYLOAD
        ref = ref_of(PAYLOAD, upstream.url("/t.npz"))
        (tmp_path / "locations.toml").write_text(f'[assets]\n"{ref.sha256}" = ["{lab.url("/mirror/t.npz")}"]\n',
                                                 encoding="utf-8")
        assert _fetch.tried_locations(ref) == (lab.url("/mirror/t.npz"), upstream.url("/t.npz"))
        assets.fetch(ref)
        assert len(lab.seen) == 1 and upstream.seen == []
    mirror = tmp_path / "share"
    mirror.mkdir()
    (mirror / "lab_table").write_bytes(PAYLOAD)
    (tmp_path / "locations.toml").write_text(f'mirrors = ["{mirror.as_uri()}/"]\n', encoding="utf-8")
    other = replace(ref_of(PAYLOAD, "https://example.org/never"), name="lab_table")
    assert _fetch.tried_locations(other)[0] == mirror.as_uri() + "/"
    assets.fetch(other, report=None)  # the cached copy, or the mirror: never example.org


@pytest.mark.parametrize("text", ['mirrors = "file:///x/"', 'mirrors = ["ftp://example.org/"]', "colour = 1",
                                  "[assets]\nx = 3", "not toml ="])
def test_a_malformed_locations_file_is_refused_never_read_as_no_mirror(cache, tmp_path, text):
    (tmp_path / "locations.toml").write_text(text, encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        _fetch.tried_locations(ref_of(PAYLOAD, "https://example.org/x"))
    assert refused.value.codes() == ["asset.locations-file"]


@contextmanager
def file_tried_last() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "tried_locations", lambda ref: tuple(dict.fromkeys(
            (*ref.locations, *_fetch.read_locations().extra(ref.name, ref.sha256)))))
        yield


def lab_first(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
        patch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "locations.toml"))
        with LoopbackStore() as lab, LoopbackStore() as upstream:
            lab.files["/t"] = upstream.files["/t"] = PAYLOAD
            ref = ref_of(PAYLOAD, upstream.url("/t"))
            (tmp_path / "locations.toml").write_text(f'mirrors = []\n[assets]\nlab_table = ["{lab.url("/t")}"]\n',
                                                     encoding="utf-8")
            assets.fetch(ref)
            assert upstream.seen == [], "the record's own location was asked before the lab's"


def test_control_a_fetch_that_tries_the_locations_file_last_is_caught_every_run(tmp_path):
    lab_first(tmp_path / "unplanted")
    caught = 0
    for run in range(RUNS):
        with file_tried_last():
            try:
                lab_first(tmp_path / str(run))
            except AssertionError:
                caught += 1
    assert caught == RUNS


# --- mime assets sources --------------------------------------------------------------------------------------------


def test_sources_names_each_location_reachable_or_not_and_why_and_offline_does_not_look(cache, tmp_path, capsys,
                                                                                        monkeypatch):
    with LoopbackStore() as server:
        server.files["/t.npz"] = PAYLOAD
        folder = tmp_path / "manifested"
        folder.mkdir()
        import json

        record = dict(ref_of(PAYLOAD, server.url("/t.npz"), server.url("/gone.npz")).to_record())
        (folder / "visuals.json").write_text(json.dumps({
            "format": "mime.asset_manifest/1", "name": "lab", "title": "a lab's files", "credit": "a lab",
            "notice": "none", "archives": [], "assets": [record], "files": {}}), encoding="utf-8")
        assert main(["assets", "sources", str(folder)]) == 0
        out = capsys.readouterr().out
        assert f"{server.url('/t.npz')}: reachable: answered" in out
        assert f"{server.url('/gone.npz')}: not reachable: no file there" in out
        assert "tier open" in out and server.seen[0].headers.get("Range") == "bytes=0-0"
        asked = len(server.seen)
        monkeypatch.setenv("MIME_OFFLINE", "1")
        assert main(["assets", "sources", str(folder)]) == 0
        assert "not looked at: MIME_OFFLINE is set" in capsys.readouterr().out and len(server.seen) == asked


def test_sources_says_a_referenced_assets_bytes_never_travel(cache, capsys):
    from chain_cases import AR4_VISUALS

    monkey = pytest.MonkeyPatch()
    monkey.setenv("MIME_OFFLINE", "1")
    try:
        assert main(["assets", "sources", str(AR4_VISUALS)]) == 0
    finally:
        monkey.undo()
    out = capsys.readouterr().out
    assert out.count("tier referenced (not permitted, referenced)") == 18
    assert out.count("its bytes never travel inside a bundle, a push or a package") == 18
