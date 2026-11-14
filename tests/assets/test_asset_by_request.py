"""A record with no location (saved form 14): its publisher makes the file available by request alone, and names where
to ask (``request``) or not. MIME never fetches it; a copy a person places in the cache is taken only if its sha256 is
the record's, and a build that does not find it is refused as ``asset.by-request``, which says where to ask and where
to put the file, never as a file to download.

**The controls.** A copy with one byte changed, at a position and to a value drawn from ``MIME_SEED``, is refused
(``MIME_PLANT_PLACED=byte``, which the probe's ``detection_rate.py`` counts: a planted run fails here); the right bytes
are taken (the unplanted run). ``locate`` with the by-request branch removed (``MIME_PLANT_PLACED=missing``) says
``asset.missing``, which the refusal test catches. A record a locations file names a mirror for is not by request."""

import hashlib
import os
import random
from pathlib import Path

import pytest
from assets_rig import Rig, run_with_rig

from mime import assets
from mime.assets import AssetRef, _cache, _fetch, _sources
from mime.assets._reference import REQUEST_FORMS, request_problem
from mime.assets._sources import Sourced
from mime.build import BuildRefused
from mime.terms import CustomLicence

REQUEST = "mailto:imaging@hospital.example.org"


def record(payload: bytes, *locations: str, request: str | None = REQUEST, name: str = "vessel_scan") -> AssetRef:
    return AssetRef(name, hashlib.sha256(payload).hexdigest(), len(payload), "CC-BY-4.0", locations,
                    role="results", request=request)


def drawn(seed: int) -> bytes:
    """A file of a size and bytes drawn from ``seed``: a few bytes to a hundred kilobytes."""
    chosen = random.Random(seed)
    return chosen.randbytes(chosen.randint(1, 100_000))


def no_locations_file(rig: Rig, root: Path) -> None:
    (root / "locations.toml").write_text("", encoding="utf-8")
    os.environ["MIME_ASSET_LOCATIONS"] = str(root / "locations.toml")


def codes_of(action) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


# --- the record --------------------------------------------------------------------------------------------------------


def test_a_record_with_no_location_is_written_with_its_key_kept_and_read_back():
    held = record(b"a scan")
    written = held.to_record()
    assert written["locations"] == [] and written["request"] == REQUEST and held.by_request
    assert AssetRef.from_record(written) == held
    assert "request" not in record(b"a scan", request=None).to_record()


@pytest.mark.parametrize("request_", [
    "https://doi.org/10.5281/zenodo.123/availability", "doi:10.5281/zenodo.123", "mailto:imaging@hospital.example.org",
    "Dr A. Lab, imaging@hospital.example.org", "imaging@hospital.example.org"])
def test_a_request_is_a_contact_or_a_link(request_: str):
    assert request_problem(request_) is None
    assert record(b"a scan", request=request_).request == request_


@pytest.mark.parametrize("request_", [
    "", " imaging@hospital.example.org", "imaging@hospital.example.org\nand more", "http://example.org/availability",
    "doi:11.1/x", "doi:10.5281", "ask the lab", "https://example.org/?token=secret", "ftp://imaging@hospital.example.org/",
    "imaging@hospital.example.org " + "x" * 200, "imaging@hospital\tsomewhere.org"])
def test_anything_else_is_refused_as_the_record_shape(request_: str):
    assert request_problem(request_) == REQUEST_FORMS
    with pytest.raises(BuildRefused) as refused:
        record(b"a scan", request=request_)
    assert refused.value.codes() == ["asset.reference"] and REQUEST_FORMS in str(refused.value)


def test_a_route_is_no_term_two_sources_naming_different_routes_agree():
    first, second = record(b"a scan", request=None), record(b"a scan", request="doi:10.5281/zenodo.123")
    held = [Sourced(first, "mime"), Sourced(second, "labx")]
    assert _sources.agree(held) and _sources.merged(held) == second
    assert not _sources.agree([Sourced(first, "mime"), Sourced(record(b"another scan"), "labx")]), "the control"


# --- where a build looks, and what it says -----------------------------------------------------------------------------


def a_build_is_refused_by_request(rig: Rig, root: Path) -> None:
    no_locations_file(rig, root)
    ref = record(drawn(1))
    refused = codes_of(lambda: assets.locate(ref, folder=root / "fl9"))
    assert refused == ["asset.by-request"], refused
    with pytest.raises(BuildRefused) as said:
        assets.locate(ref, folder=root / "fl9")
    (refusal,) = said.value.refusals
    assert refusal.message == (f"MIME cannot fetch 'vessel_scan' ({assets.size_text(ref.size)}): its publisher gives "
                               f"no download location (licence CC-BY-4.0). Request it from: {REQUEST}")
    assert str(rig.cache / "vessel_scan") in refusal.action and "mime assets fetch" in refusal.action
    assert "sha256 is the record's" in refusal.action
    assert codes_of(lambda: assets.fetch(ref)) == ["asset.by-request"]
    assert not _fetch.needs_download(ref) and rig.requests == 0
    no_route = record(drawn(1), request=None)
    with pytest.raises(BuildRefused) as said:
        assets.locate(no_route)
    assert said.value.refusals[0].message.endswith("Request it from: no route given")


def test_a_build_that_does_not_find_the_file_is_refused_by_request(tmp_path: Path):
    plant = os.environ.get("MIME_PLANT_PLACED", "")
    with pytest.MonkeyPatch.context() as patch:
        if plant == "missing":
            patch.setattr(_cache, "no_location", lambda ref: False)
        run_with_rig(lambda rig: a_build_is_refused_by_request(rig, tmp_path), tmp_path)


def test_a_mirror_in_the_locations_file_is_a_location(tmp_path: Path):
    """The control of the refusal's condition: a record with no location of its own, for which this machine's locations
    file names a mirror, is a file to fetch, refused as missing (with the command) until fetched."""

    def scenario(rig: Rig) -> None:
        payload = drawn(2)
        ref = record(payload)
        (tmp_path / "locations.toml").write_text(
            f'[assets]\n"vessel_scan" = ["{rig.server.url("/scan")}"]\n', encoding="utf-8")
        os.environ["MIME_ASSET_LOCATIONS"] = str(tmp_path / "locations.toml")
        rig.server.files["/scan"] = payload
        assert codes_of(lambda: assets.locate(ref)) == ["asset.missing"]
        assert assets.fetch(ref).read_bytes() == payload and rig.requests >= 1

    run_with_rig(scenario, tmp_path)


def a_placed_copy_is_taken(rig: Rig, root: Path, seed: int, plant: str) -> None:
    no_locations_file(rig, root)
    payload = drawn(seed)
    ref = record(payload)
    copy = bytearray(payload)
    if plant == "byte":
        chosen = random.Random(seed + 1)
        at = chosen.randrange(len(copy))
        copy[at] ^= chosen.randint(1, 255)
    rig.cache.mkdir(parents=True, exist_ok=True)
    (rig.cache / ref.name).write_bytes(bytes(copy))
    assert assets.fetch(ref).read_bytes() == payload
    assert assets.locate(ref).read_bytes() == payload and rig.requests == 0


def test_a_copy_placed_in_the_cache_is_taken_only_by_its_hash(tmp_path: Path):
    """Unplanted, the right bytes under the asset's name are taken. ``MIME_PLANT_PLACED=byte`` changes one byte of the
    copy, which must be refused: a planted run fails here."""
    seed = int(os.environ.get("MIME_SEED", "0"))
    run_with_rig(lambda rig: a_placed_copy_is_taken(rig, tmp_path, seed,
                                                    os.environ.get("MIME_PLANT_PLACED", "")), tmp_path)


@pytest.mark.parametrize("seed", range(4))
def test_a_copy_with_one_byte_changed_is_refused_and_left_where_it_is(tmp_path: Path, seed: int):
    def scenario(rig: Rig) -> None:
        with pytest.raises(BuildRefused) as refused:
            a_placed_copy_is_taken(rig, tmp_path, seed, "byte")
        assert refused.value.codes() == ["asset.placed-file"]
        assert (rig.cache / "vessel_scan").is_file() and not (rig.cache / record(drawn(seed)).sha256).exists()

    run_with_rig(scenario, tmp_path)


def test_sources_says_by_request_and_where_to_ask(tmp_path: Path):
    def scenario(rig: Rig) -> None:
        no_locations_file(rig, tmp_path)
        (view,) = assets.sources_view([record(b"a scan")])
        assert view.locations == () and view.reachable == "by request"
        text = assets.sources_text([record(b"a scan")])
        assert f"no download location (by request): request it from {REQUEST}" in text
        assert "locations, in the order" not in text
        assert "request it from no route given" in assets.sources_text([record(b"a scan", request=None)])

    run_with_rig(scenario, tmp_path)


# --- beside the licence gate in locate (P0) -----------------------------------------------------------------------------

GATED = CustomLicence("LicenseRef-Imaging-Lab-NC", "The imaging lab's data licence 1.0",
                      "https://example.org/imaging-lab/LICENSE.txt", "not_permitted", ("non-commercial research only",))


def the_order(rig: Rig, root: Path) -> list[tuple[str, int]]:
    """What a gated record with no location meets, step by step, as (the outcome, the licence questions asked so far):
    not placed, placed under its name and not taken, taken with nobody to ask, then read with somebody to ask."""
    from test_locate_gate import Yes

    no_locations_file(rig, root)
    os.environ["MIME_LICENCE_ACCEPTANCES"] = str(root / "licences.json")
    payload = drawn(3)
    ref = AssetRef("vessel_scan", hashlib.sha256(payload).hexdigest(), len(payload), GATED.id, (), role="results",
                   licences=(GATED,), request=REQUEST)
    asker, seen = Yes(), []

    def outcome(action) -> None:
        try:
            got = action()
            seen.append(("read" if got.read_bytes() == payload else "other bytes", asker.asked))
        except BuildRefused as refused:
            seen.append((refused.codes()[0], asker.asked))

    with assets.asking(asker):
        outcome(lambda: assets.locate(ref))
        rig.cache.mkdir(parents=True, exist_ok=True)
        (rig.cache / ref.name).write_bytes(payload)
        outcome(lambda: assets.locate(ref))
    with assets.asking(None):
        outcome(lambda: assets.fetch(ref))
        outcome(lambda: assets.locate(ref))
    with assets.asking(asker):
        outcome(lambda: assets.locate(ref))
        outcome(lambda: assets.locate(ref))
    return seen


#: The order: by request while the bytes are not under their sha256, with no licence asked (nothing is read); the take
#: of a placed copy asks (refused with nobody to ask, the copy kept as checked); then the gate, as for any file, never
#: the by-request words; a yes asked once and remembered.
ORDER = [("asset.by-request", 0), ("asset.by-request", 0), ("asset.licence-not-accepted", 0),
         ("asset.licence-not-accepted", 0), ("read", 1), ("read", 1)]


def test_a_gated_file_had_by_request_is_by_request_until_its_bytes_are_here_then_its_licence_is_asked(tmp_path: Path):
    seen: list[tuple[str, int]] = []
    run_with_rig(lambda rig: seen.extend(the_order(rig, tmp_path)), tmp_path)
    assert seen == ORDER


def test_control_the_gate_asked_first_is_seen_by_the_order_test(tmp_path: Path):
    """The order test discriminates: with the licence asked before the cheap check, a file not here is asked about."""
    from mime.assets._gates import require_accepted

    found = _cache._found

    def gate_first(ref, folder, checksum):
        require_accepted([ref])
        return found(ref, folder, checksum)

    seen: list[tuple[str, int]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_cache, "_found", gate_first)
        run_with_rig(lambda rig: seen.extend(the_order(rig, tmp_path)), tmp_path)
    assert seen != ORDER and seen[0] == ("asset.by-request", 1)


# --- the batch (`mime assets fetch <folder>`) meets by request before the licence, as locate does ---------------------


def batch_outcomes(rig: Rig, root: Path) -> list[tuple[str, int]]:
    """What `mime assets fetch` does for a gated record with no location, as (the outcome, the licence questions asked so
    far): the batch (`approved`) then the fetch, first with nothing in place and nobody to ask, then with somebody to ask
    and nothing in place, then with a copy placed (somebody to ask), then with a copy placed and nobody to ask."""
    from test_locate_gate import Yes

    no_locations_file(rig, root)
    os.environ["MIME_LICENCE_ACCEPTANCES"] = str(root / "licences.json")
    payload = drawn(4)
    ref = AssetRef("vessel_scan", hashlib.sha256(payload).hexdigest(), len(payload), GATED.id, (), role="results",
                   licences=(GATED,), request=REQUEST)
    asker, seen = Yes(), []

    def batch() -> None:
        try:
            with assets.approved([ref]):
                assets.fetch(ref)
            seen.append(("fetched", asker.asked))
        except BuildRefused as refused:
            seen.append((refused.codes()[0], asker.asked))

    with assets.asking(None):
        batch()
    with assets.asking(asker):
        batch()
    rig.cache.mkdir(parents=True, exist_ok=True)
    (rig.cache / ref.name).write_bytes(payload)
    with assets.asking(None):
        batch()
    with assets.asking(asker):
        batch()
    return seen


#: By request with nothing in place and no licence asked, with or without somebody to ask; once a copy is placed the gate
#: asks as for any file (refused with nobody to ask: P0's gate stays whole), and with a yes the copy is taken.
BATCH_ORDER = [("asset.by-request", 0), ("asset.by-request", 0), ("asset.licence-not-accepted", 0), ("fetched", 1)]


def test_the_batch_says_by_request_before_it_asks_about_the_licence(tmp_path: Path):
    seen: list[tuple[str, int]] = []
    run_with_rig(lambda rig: seen.extend(batch_outcomes(rig, tmp_path)), tmp_path)
    assert seen == BATCH_ORDER


def test_control_the_batch_counting_the_unplaced_licence_is_seen(tmp_path: Path):
    """The batch test discriminates: with the by-request record counted for acceptance (the earlier behaviour), the
    licence is refused or asked first."""
    seen: list[tuple[str, int]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "awaits_a_copy", lambda ref: False)
        run_with_rig(lambda rig: seen.extend(batch_outcomes(rig, tmp_path)), tmp_path)
    assert seen != BATCH_ORDER and seen[0] == ("asset.licence-not-accepted", 0)


# --- the action never names a command that would be refused as an unknown asset -----------------------------------------


def commands_in(action: str) -> list[str]:
    """The arguments of every `mime assets fetch` the text names."""
    import re
    import shlex

    return [shlex.split(each)[0] for each in re.findall(r"mime assets fetch ([^`:)]+)", action)]


def would_be_unknown(argument: str) -> bool:
    if argument.startswith("<"):  # the placeholder for a folder, which the reader fills in
        return False
    if Path(argument).is_dir() or argument.endswith("/"):
        return False
    try:
        assets.reference(argument)
    except BuildRefused as refused:
        return refused.codes() == ["asset.unknown"]
    return False


def actions_of(ref: AssetRef, folder: Path | None) -> list[str]:
    try:
        assets.locate(ref, folder=folder)
    except BuildRefused as refused:
        return [each.action for each in refused.refusals]
    raise AssertionError("not refused")


def test_by_request_and_missing_never_name_a_command_that_is_an_unknown_asset(tmp_path: Path):
    from test_sources import a_second_source

    def scenario(rig: Rig) -> None:
        no_locations_file(rig, tmp_path)
        folder = tmp_path / "saved"
        folder.mkdir()
        by_request, located = record(drawn(5)), record(drawn(6), name="hub_scan")
        (rig.cache).mkdir(parents=True, exist_ok=True)
        # a located record whose file is not in the cache's records: refused as missing, not by request
        stub_ref = AssetRef("stub_scan", "0" * 64, 4, "CC-BY-4.0", ("https://example.org/stub",))
        (rig.cache / stub_ref.sha256).write_bytes(b"xxxx")
        unlisted = [actions_of(by_request, None), actions_of(stub_ref, None)]
        assert [codes_of(lambda: assets.locate(each)) for each in (by_request, stub_ref)] == [
            ["asset.by-request"], ["asset.missing"]]
        for action in (each for actions in unlisted for each in actions):
            assert not any(would_be_unknown(argument) for argument in commands_in(action)), action
            assert "mime assets fetch vessel_scan" not in action and "mime assets fetch stub_scan" not in action
        # with a folder the command is the folder's, and it works
        (action,) = actions_of(by_request, folder)
        assert commands_in(action) == [f"{folder}/"] or commands_in(action) == [str(folder) + "/"], action
        # a record the sources hold by name keeps the command by name, which does resolve
        with a_second_source(located):
            (action,) = actions_of(located, None)
            assert commands_in(action) == ["hub_scan"] and not would_be_unknown("hub_scan"), action

    run_with_rig(scenario, tmp_path)


def test_control_naming_the_asset_by_name_where_nothing_holds_it_is_seen(tmp_path: Path):
    """The test above discriminates: with the by-name command named whatever the sources hold (the earlier behaviour) it
    names one that is refused as an unknown asset."""
    def scenario(rig: Rig) -> None:
        no_locations_file(rig, tmp_path)
        ref = record(drawn(5))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_cache, "named_by_the_sources", lambda ref, folder=None: True)
            (action,) = actions_of(ref, None)
        assert any(would_be_unknown(argument) for argument in commands_in(action)), action

    run_with_rig(scenario, tmp_path)
